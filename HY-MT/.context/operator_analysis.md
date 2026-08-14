# HY-MT1.5-1.8B operator/converter analysis

## Compatibility result

The model is a dense decoder-only transformer. Linear, RMSNorm, SwiGLU, GQA,
causal attention and KV cache are already represented by TPU-MLIR's generic
`LlmConverter`.

TPU-MLIR v1.28.1 does not dispatch `model_type=hunyuan_v1_dense`, so a small
converter extension is required.

## Semantically important differences from Qwen3

1. Weight names are `query_layernorm` / `key_layernorm`, not `q_norm` / `k_norm`.
2. Hunyuan applies Q/K projection -> RoPE -> learned QK RMSNorm. Qwen3 and the
   generic converter apply QK RMSNorm before RoPE. The operations cannot be
   reordered because the RMSNorm weights are not pairwise equal.
3. Hunyuan's dynamic RoPE uses `base = theta * alpha ** (dim / (dim - 2))`,
   with alpha=1000 for this checkpoint. Generic Llama RoPE produces materially
   different cos/sin tables and must not be used.
4. The checkpoint stores both `lm_head.weight` and `embed_tokens.weight`, while
   declaring `tie_word_embeddings=true`; the two tensors are byte-for-byte
   equal. The final build may safely use the converter's tied-weight path.

## Initial implementation

`compile/patch_tpumlir_hymt.py` adds a Hunyuan model-info mapping, correct RoPE,
post-RoPE QK normalization and dispatcher support. The first probe generates
only embedding/lm_head and block-0 MLIR at seq_len 64 before any full compile.

## Block-0 MLIR numerical probe

The generated graph was executed by `model_runner.py` and compared against the
21-token native PyTorch dump (the MLIR input was padded to seq_len 64):

| output | cosine | mean abs | max abs |
|---|---:|---:|---:|
| block output | 0.99999951 | 0.0001249 | 0.06739 |
| K cache | 0.99999293 | 0.0034658 | 0.03055 |
| V cache | 0.99999641 | 0.0000399 | 0.00077 |

The MLIR graph order was inspected directly and is `projection -> RoPE ->
query/key RMSNorm -> attention`, matching Transformers 4.56.1.
