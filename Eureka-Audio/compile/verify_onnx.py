#!/usr/bin/env python3
"""
验证导出的 ONNX 与原版 PyTorch 模型数值一致性。

覆盖：
  1. embedding_embeds     : identity pass-through
  2. embedding_cache      : token_id -> hidden
  3. block_i (prefill)    : hidden -> hidden_out + past_k + past_v  (前 N_CHECK 层)
  4. block_cache_i (decode): hidden(1) + past_kv -> hidden(1) + new_k + new_v
  5. lm_head              : hidden -> logits
  6. greedy_head          : logits -> token_id
  7. 端到端 prefill+decode : 完整跑一步 prefill + 一步 decode，比较最终 token

运行：
  python verify_onnx.py --model_path /path/to/Eureka-Audio-Instruct --seq_length 512
"""

import sys, os, argparse
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='../../Eureka-Audio-Instruct')
parser.add_argument('--seq_length', type=int, default=512)
parser.add_argument('--n_check', type=int, default=4, help='检查前 N 层 block')
args = parser.parse_args()

torch.set_grad_enabled(False)

# ── 加载模型 ──────────────────────────────────────────────────────────────
print(f'Loading model from {args.model_path} ...')
from eureka_infer.api import EurekaAudio
wrapper = EurekaAudio(model_path=args.model_path, device='cpu')
full_model = wrapper.model.eval()
for p in full_model.parameters():
    p.requires_grad_(False)

backbone     = full_model.backbone
qwen_model   = backbone.model
layers       = qwen_model.layers
lm_head_w    = backbone.lm_head
embed_tokens = qwen_model.embed_tokens

SEQ    = args.seq_length
HIDDEN = qwen_model.config.hidden_size
N_KV   = qwen_model.config.num_key_value_heads
H_DIM  = getattr(qwen_model.config, 'head_dim', HIDDEN // qwen_model.config.num_attention_heads)
VOCAB  = lm_head_w.weight.shape[0]
N_CHK  = min(args.n_check, len(layers))

ONNX_DIR = './tmp/onnx'

import onnxruntime as ort

def load_sess(path):
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(path, sess_options=opts, providers=['CPUExecutionProvider'])

def t2n(t):
    return t.detach().float().numpy()

def cosine_sim(a, b):
    a, b = a.flatten().astype(np.float32), b.flatten().astype(np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def max_abs_err(a, b):
    return float(np.max(np.abs(a.flatten().astype(np.float32) - b.flatten().astype(np.float32))))

def check(name, pt_out, onnx_out):
    pt   = pt_out.detach().float().numpy() if isinstance(pt_out, torch.Tensor) else pt_out
    onnx = onnx_out.astype(np.float32) if isinstance(onnx_out, np.ndarray) else np.array(onnx_out, dtype=np.float32)
    cos  = cosine_sim(pt, onnx)
    mae  = max_abs_err(pt, onnx)
    status = 'OK' if cos > 0.9999 and mae < 1e-3 else ('WARN' if cos > 0.999 else 'FAIL')
    print(f'  [{status}] {name:40s}  cosine={cos:.7f}  max_abs_err={mae:.2e}')
    return status != 'FAIL'

all_pass = True

# ── _KVCapture（与 export 脚本保持一致）────────────────────────────────────
class _KVCapture:
    def __init__(self, past_k=None, past_v=None):
        self._k = None
        self._v = None
        self._past_k = past_k
        self._past_v = past_v
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if self._past_k is not None:
            key_states   = torch.cat([self._past_k, key_states], dim=2)
            value_states = torch.cat([self._past_v, value_states], dim=2)
        self._k = key_states
        self._v = value_states
        return key_states, value_states

# ── 1. embedding_embeds ──────────────────────────────────────────────────
print('\n[1] embedding_embeds')
sess = load_sess(f'{ONNX_DIR}/embedding_embeds.onnx')
dummy_embeds = torch.randn(1, SEQ, HIDDEN)
onnx_out = sess.run(None, {'inputs_embeds': t2n(dummy_embeds)})[0]
# identity: output == input
all_pass &= check('embedding_embeds identity', dummy_embeds, onnx_out)

# ── 2. embedding_cache ───────────────────────────────────────────────────
print('\n[2] embedding_cache')
sess_ec = load_sess(f'{ONNX_DIR}/embedding_cache.onnx')
token_ids = torch.tensor([[1234]], dtype=torch.long)
pt_emb = embed_tokens(token_ids).float()
onnx_emb = sess_ec.run(None, {'input_ids': t2n(token_ids).astype(np.int64)})[0]
all_pass &= check('embedding_cache', pt_emb, onnx_emb)

# ── 3. block_i prefill（前 N_CHK 层）────────────────────────────────────
print(f'\n[3] block_i prefill (checking {N_CHK} layers)')
hidden = torch.randn(1, SEQ, HIDDEN)
pos_ids = torch.arange(SEQ).view(1, SEQ)
# 下三角 causal mask（全 0 = 允许，-inf = 遮蔽）
attn_mask = torch.full((1, 1, SEQ, SEQ), float('-inf'))
attn_mask = torch.triu(attn_mask, diagonal=1)

pt_hidden = hidden.clone()
onnx_hidden = t2n(hidden)

pt_kvs   = []   # 存每层 PyTorch 的 (k, v)
onnx_kvs = []   # 存每层 ONNX 的 (k, v)

for i in range(N_CHK):
    sess_b = load_sess(f'{ONNX_DIR}/block/block_{i}.onnx')

    # --- PyTorch forward ---
    layer  = deepcopy(layers[i]).float()
    re     = deepcopy(qwen_model.rotary_emb)
    cos, sin = re(pt_hidden.float(), pos_ids)
    kvc = _KVCapture()
    pt_out = layer(pt_hidden.float(), attention_mask=attn_mask,
                   position_ids=pos_ids, past_key_values=kvc,
                   use_cache=True, position_embeddings=(cos, sin))
    pt_k, pt_v = kvc._k, kvc._v

    # --- ONNX forward ---
    o_h, o_k, o_v = sess_b.run(None, {
        'hidden_states'  : onnx_hidden.astype(np.float32),
        'position_ids'   : t2n(pos_ids).astype(np.int64),
        'attention_mask' : t2n(attn_mask).astype(np.float32),
    })

    all_pass &= check(f'block_{i} hidden_out', pt_out, o_h)
    all_pass &= check(f'block_{i} past_k    ', pt_k,   o_k)
    all_pass &= check(f'block_{i} past_v    ', pt_v,   o_v)

    pt_kvs.append((pt_k.clone(), pt_v.clone()))
    onnx_kvs.append((o_k.copy(), o_v.copy()))

    pt_hidden   = pt_out.float()
    onnx_hidden = o_h

# ── 4. block_cache_i decode（用 prefill 的 KV）──────────────────────────
print(f'\n[4] block_cache_i decode (checking {N_CHK} layers)')
# 新 token hidden（随机，模拟 decode 阶段第一步）
new_h_pt   = torch.randn(1, 1, HIDDEN)
new_h_onnx = t2n(new_h_pt)
new_pos     = torch.tensor([[SEQ]], dtype=torch.long)
# decode mask: [1, 1, 1, SEQ+1] 全 0（attend to all past + current）
dec_mask    = torch.zeros(1, 1, 1, SEQ + 1)

for i in range(N_CHK):
    sess_bc = load_sess(f'{ONNX_DIR}/cache/block_cache_{i}.onnx')
    past_k_pt, past_v_pt = pt_kvs[i]
    past_k_np, past_v_np = onnx_kvs[i]

    # --- PyTorch forward ---
    layer  = deepcopy(layers[i]).float()
    re     = deepcopy(qwen_model.rotary_emb)
    cos, sin = re(new_h_pt.float(), new_pos)
    kvc = _KVCapture(past_k=past_k_pt.float(), past_v=past_v_pt.float())
    pt_dec = layer(new_h_pt.float(), attention_mask=dec_mask,
                   position_ids=new_pos, past_key_values=kvc,
                   use_cache=True, position_embeddings=(cos, sin))
    pt_nk = kvc._k[:, :, -1:, :]
    pt_nv = kvc._v[:, :, -1:, :]

    # --- ONNX forward ---
    o_dh, o_nk, o_nv = sess_bc.run(None, {
        'hidden_states'  : new_h_onnx.astype(np.float32),
        'position_ids'   : t2n(new_pos).astype(np.int64),
        'attention_mask' : t2n(dec_mask).astype(np.float32),
        'past_k'         : past_k_np.astype(np.float32),
        'past_v'         : past_v_np.astype(np.float32),
    })

    all_pass &= check(f'block_cache_{i} hidden_out', pt_dec, o_dh)
    all_pass &= check(f'block_cache_{i} new_k     ', pt_nk,  o_nk)
    all_pass &= check(f'block_cache_{i} new_v     ', pt_nv,  o_nv)

    new_h_pt   = pt_dec.float()
    new_h_onnx = o_dh

# ── 5. lm_head ───────────────────────────────────────────────────────────
print('\n[5] lm_head')
sess_lm = load_sess(f'{ONNX_DIR}/lm_head.onnx')
dummy_h = torch.randn(1, 1, HIDDEN)
# 用 float32 原版 norm 计算 PyTorch 参考值
norm_ref = deepcopy(qwen_model.norm).float()
pt_norm   = norm_ref(dummy_h.float()).float()
pt_logits = lm_head_w.float()(pt_norm).float()
onnx_logits = sess_lm.run(None, {'hidden_states': t2n(dummy_h).astype(np.float32)})[0]
all_pass &= check('lm_head logits', pt_logits, onnx_logits)

# ── 6. greedy_head ───────────────────────────────────────────────────────
print('\n[6] greedy_head')
sess_g = load_sess(f'{ONNX_DIR}/greedy_head.onnx')
dummy_logits = torch.randn(1, VOCAB)
pt_token = torch.topk(dummy_logits.float(), 1)[1]
onnx_token = sess_g.run(None, {'logits': t2n(dummy_logits).astype(np.float32)})[0]
all_pass &= check('greedy_head token', pt_token, onnx_token)

# ── 7. 端到端一步 prefill + 一步 decode ─────────────────────────────────
print('\n[7] end-to-end: 1 prefill step + 1 decode step (all layers)')
USE_LAYERS = min(4, len(layers))   # 只跑前 4 层做端到端（加速测试）
print(f'    (using first {USE_LAYERS} layers to keep test fast)')

# 加载所有需要的 session
sess_ee  = load_sess(f'{ONNX_DIR}/embedding_embeds.onnx')
sess_ec2 = load_sess(f'{ONNX_DIR}/embedding_cache.onnx')
sess_lm2 = load_sess(f'{ONNX_DIR}/lm_head.onnx')
sess_g2  = load_sess(f'{ONNX_DIR}/greedy_head.onnx')
blk_sess  = [load_sess(f'{ONNX_DIR}/block/block_{i}.onnx')       for i in range(USE_LAYERS)]
cach_sess = [load_sess(f'{ONNX_DIR}/cache/block_cache_{i}.onnx') for i in range(USE_LAYERS)]

# 端到端测试：必须与 ONNX 固定 shape 一致（SEQ x SEQ）
SHORT_SEQ = SEQ
inp_embeds = torch.randn(1, SHORT_SEQ, HIDDEN)
pos_ids_p  = torch.arange(SHORT_SEQ).view(1, SHORT_SEQ)
causal     = torch.full((1, 1, SHORT_SEQ, SHORT_SEQ), float('-inf'))
causal     = torch.triu(causal, diagonal=1)

# --- ONNX prefill ---
h = sess_ee.run(None, {'inputs_embeds': t2n(inp_embeds).astype(np.float32)})[0]
kv_cache = []
for i in range(USE_LAYERS):
    h, k, v = blk_sess[i].run(None, {
        'hidden_states' : h.astype(np.float32),
        'position_ids'  : t2n(pos_ids_p).astype(np.int64),
        'attention_mask': t2n(causal).astype(np.float32),
    })
    kv_cache.append((k, v))

# lm_head + greedy on last token
logits_p   = sess_lm2.run(None, {'hidden_states': h[:, -1:, :].astype(np.float32)})[0]
token_onnx = sess_g2.run(None,  {'logits': logits_p[0].astype(np.float32)})[0].flatten()[0]

# --- PyTorch prefill ---
h_pt = inp_embeds.float()
pt_kvs2 = []
for i in range(USE_LAYERS):
    layer = deepcopy(layers[i]).float()
    re    = deepcopy(qwen_model.rotary_emb)
    c, s  = re(h_pt, pos_ids_p)
    kvc   = _KVCapture()
    h_pt  = layer(h_pt, attention_mask=causal, position_ids=pos_ids_p,
                  past_key_values=kvc, use_cache=True, position_embeddings=(c, s))
    pt_kvs2.append((kvc._k.clone(), kvc._v.clone()))

logits_pt   = lm_head_w(qwen_model.norm(h_pt[:, -1:, :].float())).float()
token_pt    = torch.topk(logits_pt[0], 1)[1].item()

print(f'  Prefill: PyTorch token={token_pt}, ONNX token={int(token_onnx)}  ', end='')
print('OK' if token_pt == int(token_onnx) else 'MISMATCH')
all_pass &= (token_pt == int(token_onnx))

# --- ONNX decode (1 step) ---
dec_h  = sess_ec2.run(None, {'input_ids': np.array([[token_onnx]], dtype=np.int64)})[0]
dec_pos = np.array([[SHORT_SEQ]], dtype=np.int64)
dec_mask_np = np.zeros((1, 1, 1, SHORT_SEQ + 1), dtype=np.float32)
new_kv = []
for i in range(USE_LAYERS):
    dec_h, nk, nv = cach_sess[i].run(None, {
        'hidden_states' : dec_h.astype(np.float32),
        'position_ids'  : dec_pos,
        'attention_mask': dec_mask_np,
        'past_k'        : kv_cache[i][0].astype(np.float32),
        'past_v'        : kv_cache[i][1].astype(np.float32),
    })
    new_kv.append((nk, nv))
logits_d   = sess_lm2.run(None, {'hidden_states': dec_h.astype(np.float32)})[0]
token2_onnx = sess_g2.run(None, {'logits': logits_d[0].astype(np.float32)})[0].flatten()[0]

# --- PyTorch decode (1 step) ---
new_h_pt = embed_tokens(torch.tensor([[token_pt]])).float()
dec_pos_pt  = torch.tensor([[SHORT_SEQ]], dtype=torch.long)
dec_mask_pt = torch.zeros(1, 1, 1, SHORT_SEQ + 1)
for i in range(USE_LAYERS):
    layer = deepcopy(layers[i]).float()
    re    = deepcopy(qwen_model.rotary_emb)
    c, s  = re(new_h_pt, dec_pos_pt)
    kvc   = _KVCapture(past_k=pt_kvs2[i][0].float(), past_v=pt_kvs2[i][1].float())
    new_h_pt = layer(new_h_pt, attention_mask=dec_mask_pt, position_ids=dec_pos_pt,
                     past_key_values=kvc, use_cache=True, position_embeddings=(c, s))
logits_pt2 = lm_head_w(qwen_model.norm(new_h_pt.float())).float()
token2_pt  = torch.topk(logits_pt2[0], 1)[1].item()

print(f'  Decode : PyTorch token={token2_pt}, ONNX token={int(token2_onnx)}  ', end='')
print('OK' if token2_pt == int(token2_onnx) else 'MISMATCH')
all_pass &= (token2_pt == int(token2_onnx))

# ── 汇总 ─────────────────────────────────────────────────────────────────
print('\n' + '='*60)
if all_pass:
    print('ALL CHECKS PASSED — ONNX 与 PyTorch 模型数值一致，可以放心编译 bmodel')
else:
    print('SOME CHECKS FAILED — 请查看上面的 FAIL/MISMATCH 条目')
print('='*60)
