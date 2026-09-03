# HY-MT1.5-1.8B PyTorch baseline

## Environment

- Python 3.10.20
- PyTorch 2.6.0+cu124
- Transformers 4.56.1
- NVIDIA RTX 3060 12GB
- dtype: BF16
- attention: eager
- generation: deterministic greedy
- weights: `$MODEL_PATH`（环境变量指定，不写入仓库）

## Architecture

- 32 decoder layers, hidden 2048, intermediate 6144
- 16 query heads, 4 KV heads, head_dim 128
- QK-Norm after RoPE
- dynamic RoPE with theta=10000 and alpha=1000
- vocabulary 120818
- first-layer KV shape for an N-token prompt: `[1, 4, N, 128]`

## Saved references

`outputs/baseline/*_reference.npz` contains input/output IDs, embedding output,
layers 0/15/31, final hidden state, last-token logits and layer-0 KV cache.
`outputs/baseline/baseline.json` contains prompts, decoded outputs, timing and
tensor summaries.

Basic English/Chinese/Japanese translation succeeds. The formatting case emits
an extra nested `<sn>` tag in native PyTorch; this is retained as native behavior
for later bmodel regression rather than treated as a conversion error.

