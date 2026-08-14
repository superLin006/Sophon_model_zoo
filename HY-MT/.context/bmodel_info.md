# HY-MT1.5-1.8B BM1684X bmodel

## Artifact

- TPU-MLIR: v1.28.1-20260429
- chip: BM1684X
- quantization: W8BF16, BF16 KV cache
- static sequence length: 512
- size: 2,125,438,976 bytes (1.98 GiB)
- path: `models/BM1684X/w8bf16_seq512/models_w8bf16_seq512_bm1684x_1dev_static_20260812_173234.bmodel`
- networks: embedding, embedding_cache, lm_head, 32 block, 32 block_cache

## Converter changes

TPU-MLIR 1.28.1 was extended by `compile/patch_tpumlir_hymt.py` with:

- `hunyuan_v1_dense` dispatcher support
- Hunyuan QK-Norm weight names
- Hunyuan DynamicNTKAlphaRotary implementation
- post-RoPE QK-Norm ordering for prefill and cache graphs

## Numerical validation

Block-0 F32 MLIR versus native PyTorch:

- output cosine: 0.99999951
- K cache cosine: 0.99999293
- V cache cosine: 0.99999641

## Board validation

Board: BM1684X SoC, aarch64. Runtime: pure C++ BMRuntime + tokenizers-cpp.

| case | board output matches PyTorch | prefill | decode |
|---|:---:|---:|---:|
| English -> Chinese | yes | 204.84 ms | 23.45 tok/s |
| Chinese -> English | yes | 205.31 ms | 23.22 tok/s |
| Japanese -> Chinese | yes | 204.66 ms | 23.43 tok/s |
| terminology intervention | yes | 205.85 ms | 23.20 tok/s |
| formatted translation | yes | 205.12 ms | 23.04 tok/s |

The formatted case contains an extra nested `<sn>` in both native PyTorch and
the bmodel, so it is model behavior rather than conversion error.

