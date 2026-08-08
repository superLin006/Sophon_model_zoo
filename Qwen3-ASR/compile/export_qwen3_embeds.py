#!/usr/bin/env python3
"""
导出 Qwen3-ASR-0.6B 的 LLM（language_model）为 inputs_embeds 版多网络 ONNX，
再编译为 bmodel（流程复刻 Eureka-Audio/compile/export_qwen3_embeds.py，适配 transformers 5.14）。

与原始 token_id 版本的区别：
  - prefill 网络输入变为 inputs_embeds [1, SEQ, 1024]（而非 input_ids）
    → 可接受拼好的 prefix + audio embeds + suffix 连续向量
  - decode 阶段仍用 embedding_cache（单 token id 查表）→ 无需修改

网络划分：
  embedding_embeds   : inputs_embeds [1,SEQ,1024] → hidden [1,SEQ,1024]   (identity)
  block_i (i=0..27)  : hidden → hidden + past_k + past_v   (prefill, KV 输出 [1,SEQ,N_KV,HEAD])
  block_cache_i      : hidden(1) + kv_cache → hidden(1) + new_k + new_v (decode)
  lm_head            : hidden[last] → logits [1, vocab]
  greedy_head        : logits → token_id
  embedding_cache    : token_id → hidden [1,1,1024]  (decode 查表)

适配 transformers 5.14 的注意点（已实测确认）：
  - Qwen3RMSNorm 不继承 nn.RMSNorm → 按类名替换为 ONNX 兼容实现
    （注意 self_attn 内的 q_norm/k_norm 也是 Qwen3RMSNorm，需一并替换）
  - 需强制每层 _attn_implementation='eager'（5.14 默认可能 sdpa，ONNX 不友好）
  - Qwen3Attention.forward 已用 position_embeddings，DecoderLayer 接受 position_ids 但透传忽略 → OK
  - eager_attention_forward 的 mask 格式为 float 4D（直接加法）→ 传 0/-1e9 矩阵

用法（qwen3-asr conda env）:
  python export_qwen3_embeds.py --model_path ../models --seq_length 2048
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
from copy import deepcopy

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='../models')
parser.add_argument('--seq_length', type=int, default=2048)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--out_dir', type=str, default='./tmp/onnx')
args = parser.parse_args()

from transformers import AutoModelForMultimodalLM

# ── 加载模型 ──────────────────────────────────────────────────────────────────
print(f'Loading model from {args.model_path} ...')
full_model = AutoModelForMultimodalLM.from_pretrained(
    args.model_path, dtype=torch.bfloat16, torch_dtype=torch.bfloat16).eval()
for p in full_model.parameters():
    p.requires_grad_(False)

# Qwen3 backbone 内部结构（Qwen3ASR: model.language_model 是标准 Qwen3Model）
qwen_model  = full_model.model.language_model   # Qwen3Model
layers      = qwen_model.layers
lm_head     = full_model.lm_head
embed_tokens = qwen_model.embed_tokens

device = torch.device(args.device)
dtype  = torch.float32   # export 用 float32，bmodel 量化时指定 W4BF16

# ── 参数 ──────────────────────────────────────────────────────────────────────
SEQ_LENGTH          = args.seq_length
NUM_LAYERS          = len(layers)
HIDDEN_SIZE         = qwen_model.config.hidden_size
NUM_ATTENTION_HEADS = qwen_model.config.num_attention_heads
NUM_KV_HEADS        = qwen_model.config.num_key_value_heads
HEAD_DIM            = getattr(qwen_model.config, 'head_dim', HIDDEN_SIZE // NUM_ATTENTION_HEADS)
VOCAB_SIZE          = lm_head.weight.shape[0]
RMS_EPS             = qwen_model.config.rms_norm_eps

print(f'Layers={NUM_LAYERS}  Hidden={HIDDEN_SIZE}  Heads={NUM_ATTENTION_HEADS}  '
      f'KV={NUM_KV_HEADS}  HeadDim={HEAD_DIM}  Vocab={VOCAB_SIZE}  Seq={SEQ_LENGTH}')

folder = args.out_dir
os.makedirs(folder + '/block', exist_ok=True)
os.makedirs(folder + '/cache', exist_ok=True)


# ── 工具：替换 Qwen3RMSNorm 为 ONNX 兼容实现（含 q_norm/k_norm）────────────────
def replace_rmsnorm(module: nn.Module):
    for name, child in list(module.named_children()):
        # 5.14 的 Qwen3RMSNorm 不继承 nn.RMSNorm，按类型名匹配
        if type(child).__name__ == 'Qwen3RMSNorm':
            w, eps = child.weight.data.clone().float(), child.variance_epsilon
            class _RMSNormOnnx(nn.Module):
                def __init__(self, w, eps):
                    super().__init__()
                    self.weight = nn.Parameter(w)
                    self.eps = eps
                def forward(self, x):
                    rms = (x.float().pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()
                    return (x.float() * rms) * self.weight
            setattr(module, name, _RMSNormOnnx(w, eps))
        else:
            replace_rmsnorm(child)


def force_eager(layer: nn.Module):
    """强制走 eager attention（ONNX 可导出的手写 attention 路径）"""
    for name, child in layer.named_children():
        if type(child).__name__ == 'Qwen3Attention':
            child.config._attn_implementation = 'eager'
        else:
            force_eager(child)


# ── 网络模块定义 ───────────────────────────────────────────────────────────────

class EmbeddingEmbeds(nn.Module):
    """prefill 用：直接接受 inputs_embeds，返回原样。
    注意：不能写 .float()——在 bf16 模型下会导出成 Cast op，
    TPU-MLIR 解析该 Cast 时报 'operand hidden_states not found'。"""
    def forward(self, inputs_embeds):
        return inputs_embeds


class EmbeddingCache(nn.Module):
    """decode 用：单 token id 查表"""
    def __init__(self):
        super().__init__()
        self.embed = embed_tokens

    def forward(self, input_ids):
        return self.embed(input_ids).float()


class _KVCapture:
    """鸭子类型 Cache，只实现 Qwen3Attention 需要的 update() 接口"""
    def __init__(self, past_k=None, past_v=None):
        self._k = None
        self._v = None
        self._past_k = past_k
        self._past_v = past_v

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if self._past_k is not None:
            key_states = torch.cat([self._past_k, key_states], dim=2)
            value_states = torch.cat([self._past_v, value_states], dim=2)
        self._k = key_states
        self._v = value_states
        return key_states, value_states


class Block(nn.Module):
    """单层 Transformer block，prefill 阶段"""
    def __init__(self, layer_id):
        super().__init__()
        self.layer = deepcopy(layers[layer_id]).float()
        replace_rmsnorm(self.layer)
        force_eager(self.layer)
        self.rotary_emb = deepcopy(qwen_model.rotary_emb)

    def forward(self, hidden_states, position_ids, attention_mask):
        cos, sin = self.rotary_emb(hidden_states.float(), position_ids)
        cache = _KVCapture()
        out = self.layer(
            hidden_states.float(),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            position_embeddings=(cos, sin),
        )
        # KV 内部 layout [1,N_KV,SEQ,HEAD] → 对外 [1,SEQ,N_KV,HEAD]（单 token 连续，便于 decode 偏移）
        # KV 输出 F16（KV 量化：传输/存储减半，精度影响小）
        past_k = cache._k.transpose(1, 2).contiguous().half()  # [1,SEQ,N_KV,HEAD] f16
        past_v = cache._v.transpose(1, 2).contiguous().half()
        return out.float(), past_k, past_v


class BlockCache(nn.Module):
    """单层 Transformer block，decode 阶段（显式传入 past_k/v）"""
    def __init__(self, layer_id):
        super().__init__()
        self.layer = deepcopy(layers[layer_id]).float()
        replace_rmsnorm(self.layer)
        force_eager(self.layer)
        self.rotary_emb = deepcopy(qwen_model.rotary_emb)

    def forward(self, hidden_states, position_ids, attention_mask, past_k, past_v):
        cos, sin = self.rotary_emb(hidden_states.float(), position_ids)
        # 对外 past_k/v layout [1,SEQ,N_KV,HEAD]（F16 输入）→ 内部 attention 需要 [1,N_KV,SEQ,HEAD] f32
        pk = past_k.float().transpose(1, 2).contiguous()
        pv = past_v.float().transpose(1, 2).contiguous()
        cache = _KVCapture(past_k=pk, past_v=pv)
        out = self.layer(
            hidden_states.float(),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            position_embeddings=(cos, sin),
        )
        # 取最后 token 的新 KV [1,N_KV,1,HEAD] → 对外 [1,1,N_KV,HEAD]（F16 输出）
        new_k = cache._k[:, :, -1:, :].transpose(1, 2).contiguous().half()  # [1,1,N_KV,HEAD] f16
        new_v = cache._v[:, :, -1:, :].transpose(1, 2).contiguous().half()
        return out.float(), new_k, new_v


class LmHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm    = deepcopy(qwen_model.norm).float()
        replace_rmsnorm(self.norm)
        self.lm_head = lm_head

    def forward(self, hidden_states):
        hidden_states = self.norm(hidden_states.float())
        return self.lm_head(hidden_states).float()


class GreedyHead(nn.Module):
    def forward(self, logits):
        _, token = torch.topk(logits.float(), 1)
        return token


# ── 转换函数 ──────────────────────────────────────────────────────────────────

def convert_embedding_embeds():
    m = EmbeddingEmbeds()
    dummy = torch.randn(1, SEQ_LENGTH, HIDDEN_SIZE)
    torch.onnx.export(
        m, (dummy,),
        f'{folder}/embedding_embeds.onnx',
        input_names=['inputs_embeds'],
        output_names=['hidden_states'],
        do_constant_folding=True,
        opset_version=18,
        dynamo=False,
    )
    print('  embedding_embeds.onnx')


def convert_embedding_cache():
    m = EmbeddingCache()
    dummy = torch.tensor([[0]], dtype=torch.long)
    torch.onnx.export(
        m, (dummy,),
        f'{folder}/embedding_cache.onnx',
        input_names=['input_ids'],
        output_names=['hidden_states'],
        do_constant_folding=True,
        opset_version=18,
        dynamo=False,
    )
    print('  embedding_cache.onnx')


def convert_block(i):
    m = Block(i).float()
    hidden     = torch.randn(1, SEQ_LENGTH, HIDDEN_SIZE)
    pos_ids    = torch.arange(SEQ_LENGTH).view(1, SEQ_LENGTH)
    attn_mask  = torch.zeros(1, 1, SEQ_LENGTH, SEQ_LENGTH)
    torch.onnx.export(
        m, (hidden, pos_ids, attn_mask),
        f'{folder}/block/block_{i}.onnx',
        input_names=['hidden_states', 'position_ids', 'attention_mask'],
        output_names=['hidden_states_out', 'past_k', 'past_v'],
        do_constant_folding=True,
        opset_version=17,
        dynamo=False,
    )


def convert_block_cache(i):
    m = BlockCache(i).float()
    hidden    = torch.randn(1, 1, HIDDEN_SIZE)
    pos_ids   = torch.tensor([[0]], dtype=torch.long)
    attn_mask = torch.zeros(1, 1, 1, SEQ_LENGTH + 1)
    past_k    = torch.randn(1, SEQ_LENGTH, NUM_KV_HEADS, HEAD_DIM)
    past_v    = torch.randn(1, SEQ_LENGTH, NUM_KV_HEADS, HEAD_DIM)
    torch.onnx.export(
        m, (hidden, pos_ids, attn_mask, past_k, past_v),
        f'{folder}/cache/block_cache_{i}.onnx',
        input_names=['hidden_states', 'position_ids', 'attention_mask', 'past_k', 'past_v'],
        output_names=['hidden_states_out', 'past_k_out', 'past_v_out'],
        do_constant_folding=True,
        opset_version=17,
        dynamo=False,
    )


def convert_lm_head():
    m = LmHead().float()
    dummy = torch.randn(1, 1, HIDDEN_SIZE)
    torch.onnx.export(
        m, (dummy,),
        f'{folder}/lm_head.onnx',
        input_names=['hidden_states'],
        output_names=['logits'],
        do_constant_folding=True,
        opset_version=17,
        dynamo=False,
    )
    print('  lm_head.onnx')


def convert_greedy_head():
    m = GreedyHead()
    dummy = torch.randn(1, VOCAB_SIZE)
    torch.onnx.export(
        m, (dummy,),
        f'{folder}/greedy_head.onnx',
        input_names=['logits'],
        output_names=['token'],
        do_constant_folding=True,
        opset_version=18,
        dynamo=False,
    )
    print('  greedy_head.onnx')


# ── 执行导出 ──────────────────────────────────────────────────────────────────
print('\nConverting embedding_embeds ...')
convert_embedding_embeds()

print('Converting embedding_cache ...')
convert_embedding_cache()

print('Converting lm_head ...')
convert_lm_head()
convert_greedy_head()

print(f'Converting {NUM_LAYERS} blocks + {NUM_LAYERS} block_cache ...')
from tqdm import tqdm
for i in tqdm(range(NUM_LAYERS)):
    convert_block(i)
    convert_block_cache(i)

print('\nAll ONNX exported!')
