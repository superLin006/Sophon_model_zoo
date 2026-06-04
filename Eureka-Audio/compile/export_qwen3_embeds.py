#!/usr/bin/env python3
"""
导出 Qwen3-1.7B 的 inputs_embeds 版本为多个 ONNX，再编译为 bmodel。

与原始 token_id 版本的区别：
  - embedding 网络输入变为 inputs_embeds [1, SEQ, 2048]（而非 input_ids）
  - 前缀阶段（prefill）：接受混合了音频 embedding 的 inputs_embeds
  - decode 阶段：仍用 embedding_cache（单 token id 查表）→ 无需修改

网络划分（与 qwen3/python_demo/chat.cpp 保持一致）：
  embedding_embeds   : inputs_embeds [1,SEQ,2048] → hidden [1,SEQ,2048]   (仅做 identity，供后续 block 使用)
  block_i (i=0..27)  : hidden → hidden + past_k + past_v
  block_cache_i      : hidden(1) + kv_cache → hidden(1) + new_k + new_v
  lm_head            : hidden[last] → logits [1, vocab]
  greedy_head        : logits → token_id
  embedding_cache    : token_id → hidden [1,1,2048]  （decode 阶段复用）

用法：
  python export_qwen3_embeds.py --model_path ../../Eureka-Audio-Instruct --seq_length 512
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from copy import deepcopy

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='../../Eureka-Audio-Instruct')
parser.add_argument('--seq_length', type=int, default=512)
parser.add_argument('--device', type=str, default='cpu')
args = parser.parse_args()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ── 加载模型 ──────────────────────────────────────────────────────────────────
print(f'Loading model from {args.model_path} ...')
from eureka_infer.api import EurekaAudio
wrapper = EurekaAudio(model_path=args.model_path, device=args.device)
full_model = wrapper.model.eval()
for p in full_model.parameters():
    p.requires_grad_(False)

# Qwen3 backbone 内部结构
backbone   = full_model.backbone          # Qwen3ForCausalLM
qwen_model = backbone.model               # Qwen3Model
layers     = qwen_model.layers
lm_head    = backbone.lm_head
embed_tokens = qwen_model.embed_tokens

device = torch.device(args.device)
dtype  = torch.float32   # export 用 float32，bmodel 量化时指定 BF16

# ── 参数 ──────────────────────────────────────────────────────────────────────
SEQ_LENGTH         = args.seq_length
NUM_LAYERS         = len(layers)
HIDDEN_SIZE        = qwen_model.config.hidden_size
NUM_ATTENTION_HEADS = qwen_model.config.num_attention_heads
NUM_KV_HEADS       = qwen_model.config.num_key_value_heads
HEAD_DIM           = getattr(qwen_model.config, 'head_dim', HIDDEN_SIZE // NUM_ATTENTION_HEADS)
VOCAB_SIZE         = lm_head.weight.shape[0]

print(f'Layers={NUM_LAYERS}  Hidden={HIDDEN_SIZE}  Heads={NUM_ATTENTION_HEADS}  KV={NUM_KV_HEADS}  HeadDim={HEAD_DIM}  Vocab={VOCAB_SIZE}  Seq={SEQ_LENGTH}')

folder = './tmp/onnx'
os.makedirs(folder, exist_ok=True)
os.makedirs(folder + '/block', exist_ok=True)
os.makedirs(folder + '/cache', exist_ok=True)


# ── 工具：替换 nn.RMSNorm 为 ONNX 兼容实现 ───────────────────────────────────
def replace_rmsnorm(module: nn.Module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.RMSNorm):
            class _RMSNormOnnx(nn.Module):
                def __init__(self, w):
                    super().__init__()
                    self.weight = nn.Parameter(w.data.clone().float())
                    self.eps = 1e-6
                def forward(self, x):
                    rms = (x.float().pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()
                    return (x.float() * rms) * self.weight
            setattr(module, name, _RMSNormOnnx(child.weight))
        else:
            replace_rmsnorm(child)


# ── 网络模块定义 ───────────────────────────────────────────────────────────────

class EmbeddingEmbeds(nn.Module):
    """prefill 用：直接接受 inputs_embeds，返回 float 供后续 block 使用"""
    def forward(self, inputs_embeds):
        return inputs_embeds.float()


class EmbeddingCache(nn.Module):
    """decode 用：单 token id 查表"""
    def __init__(self):
        super().__init__()
        self.embed = embed_tokens

    def forward(self, input_ids):
        return self.embed(input_ids).float()


class _KVCapture:
    """鸭子类型 Cache，只实现 Qwen3Attention 需要的 update() 接口。
    避免继承 transformers.Cache 以防 DynamicCache 的 TracerWarning 污染多轮 ONNX export。"""
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
        past_k = cache._k.transpose(1, 2).contiguous().float()  # [1,SEQ,N_KV,HEAD]
        past_v = cache._v.transpose(1, 2).contiguous().float()
        return out.float(), past_k, past_v


class BlockCache(nn.Module):
    """单层 Transformer block，decode 阶段（显式传入 past_k/v）"""
    def __init__(self, layer_id):
        super().__init__()
        self.layer = deepcopy(layers[layer_id]).float()
        replace_rmsnorm(self.layer)
        self.rotary_emb = deepcopy(qwen_model.rotary_emb)

    def forward(self, hidden_states, position_ids, attention_mask, past_k, past_v):
        cos, sin = self.rotary_emb(hidden_states.float(), position_ids)
        # 对外 past_k/v layout [1,SEQ,N_KV,HEAD] → 内部 attention 需要 [1,N_KV,SEQ,HEAD]
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
        # 取最后 token 的新 KV [1,N_KV,1,HEAD] → 对外 [1,1,N_KV,HEAD]
        new_k = cache._k[:, :, -1:, :].transpose(1, 2).contiguous().float()  # [1,1,N_KV,HEAD]
        new_v = cache._v[:, :, -1:, :].transpose(1, 2).contiguous().float()
        return out.float(), new_k, new_v


class LmHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm    = deepcopy(qwen_model.norm).float()
        replace_rmsnorm(self.norm)
        self.lm_head = backbone.lm_head

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
    # KV layout 改为 [1,SEQ,N_KV,HEAD]（单 token 连续，decode 偏移高效）
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

# ── 打印 TPU-MLIR 编译命令 ────────────────────────────────────────────────────
print('\n========================================================')
print('下一步：用 TPU-MLIR 编译 bmodel（在 Docker 内运行）')
print('========================================================')
print(f'''
# ----- 编译 embedding_embeds -----
model_transform.py \\
  --model_name embedding_embeds \\
  --model_def {folder}/embedding_embeds.onnx \\
  --input_shapes [[1,{SEQ_LENGTH},{HIDDEN_SIZE}]] \\
  --mlir ./tmp/embedding_embeds.mlir

model_deploy.py \\
  --mlir ./tmp/embedding_embeds.mlir \\
  --quantize BF16 --chip bm1684x \\
  --model ./tmp/embedding_embeds.bmodel

# ----- 编译 embedding_cache -----
model_transform.py \\
  --model_name embedding_cache \\
  --model_def {folder}/embedding_cache.onnx \\
  --input_shapes [[1,1]] \\
  --mlir ./tmp/embedding_cache.mlir

model_deploy.py \\
  --mlir ./tmp/embedding_cache.mlir \\
  --quantize BF16 --chip bm1684x \\
  --model ./tmp/embedding_cache.bmodel

# ----- 编译 block_i（28 层）-----
for i in $(seq 0 {NUM_LAYERS-1}); do
  model_transform.py \\
    --model_name block_$i \\
    --model_def {folder}/block/block_$i.onnx \\
    --input_shapes [[1,{SEQ_LENGTH},{HIDDEN_SIZE}],[1,{SEQ_LENGTH}],[1,1,{SEQ_LENGTH},{SEQ_LENGTH}]] \\
    --mlir ./tmp/block_$i.mlir
  model_deploy.py \\
    --mlir ./tmp/block_$i.mlir \\
    --quantize W4BF16 --chip bm1684x \\
    --model ./tmp/block_$i.bmodel
done

# ----- 编译 block_cache_i -----
for i in $(seq 0 {NUM_LAYERS-1}); do
  model_transform.py \\
    --model_name block_cache_$i \\
    --model_def {folder}/cache/block_cache_$i.onnx \\
    --input_shapes [[1,1,{HIDDEN_SIZE}],[1,1],[1,1,1,{SEQ_LENGTH+1}],[1,{NUM_KV_HEADS},{SEQ_LENGTH},{HEAD_DIM}],[1,{NUM_KV_HEADS},{SEQ_LENGTH},{HEAD_DIM}]] \\
    --mlir ./tmp/block_cache_$i.mlir
  model_deploy.py \\
    --mlir ./tmp/block_cache_$i.mlir \\
    --quantize W4BF16 --chip bm1684x \\
    --model ./tmp/block_cache_$i.bmodel
done

# ----- 编译 lm_head + greedy_head -----
model_transform.py --model_name lm_head \\
  --model_def {folder}/lm_head.onnx \\
  --input_shapes [[1,1,{HIDDEN_SIZE}]] \\
  --mlir ./tmp/lm_head.mlir
model_deploy.py --mlir ./tmp/lm_head.mlir \\
  --quantize BF16 --chip bm1684x \\
  --model ./tmp/lm_head.bmodel

model_transform.py --model_name greedy_head \\
  --model_def {folder}/greedy_head.onnx \\
  --input_shapes [[1,{VOCAB_SIZE}]] \\
  --mlir ./tmp/greedy_head.mlir
model_deploy.py --mlir ./tmp/greedy_head.mlir \\
  --quantize BF16 --chip bm1684x \\
  --model ./tmp/greedy_head.bmodel

# ----- 合并所有 bmodel -----
model_tool --combine \\
  ./tmp/embedding_embeds.bmodel \\
  ./tmp/embedding_cache.bmodel \\
  $(for i in $(seq 0 {NUM_LAYERS-1}); do echo ./tmp/block_$i.bmodel; done) \\
  $(for i in $(seq 0 {NUM_LAYERS-1}); do echo ./tmp/block_cache_$i.bmodel; done) \\
  ./tmp/lm_head.bmodel \\
  ./tmp/greedy_head.bmodel \\
  -o ../models/BM1684X/qwen3_1.7b_embeds_w4bf16_seq{SEQ_LENGTH}_bm1684x.bmodel
''')
