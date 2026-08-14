# Qwen3-TTS bmodel 转换报告（M2）

转换日期: 2026-08-13
工具: TPU-MLIR v1.28.1（sophon/tpuc_dev:v3.4-tpumlir-1.28.1）
芯片: BM1684X
精度: F16（首版，后续再评估 W4BF16）

## bmodel 产物

| bmodel | 大小 | 网络数 | 说明 |
|---|---|---|---|
| `qwen3_tts_talker_F16.bmodel` | 2.3GB | 59 | embedding_code/text + 28 block + 28 cache + codec_head |
| `qwen3_tts_cp_F16.bmodel` | 422MB | 39 | 5 cp_block + 5 cp_cache + 15 lm_head + 14 embedding |
| `codec_decoder.bmodel` | 235MB | 1 | codec 12Hz 解码（固定 T=325） |

## 网络接口（供 C++ 调用）

### talker block（prefill）
- net: `talker_block_i`（i=0..27）
- 输入: `input_states [1,256,1024] f32`, `position_ids [3,1,256] int32`, `attention_mask [1,1,256,256] f32`
- 输出: `hidden_states [1,256,1024] f32`, `past_k [1,8,256,128] f32`, `past_v [1,8,256,128] f32`

### talker block_cache（decode）
- net: `talker_block_cache_i`
- 输入: `input_states [1,1,1024]`, `position_ids [3,1,1] int32`, `attention_mask [1,1,1,257]`, `history_k [1,8,256,128]`, `history_v [1,8,256,128]`
- 输出: `hidden_states [1,1,1024]`, `past_k [1,8,1,128]`, `past_v [1,8,1,128]`（仅新 token）

### 其他 talker 网络
- `embedding_code`: input_ids [1,256] int32 → [1,256,1024] f32
- `embedding_text`: input_ids [1,256] int32 → [1,256,1024] f32（bf16 权重）
- `codec_head`: hidden [1,1,1024] → logits [1,1,3072]

### code_predictor
- `cp_block_i`（prefill）: [1,2,1024] + [1,2] pos + [1,1,2,2] mask → [1,2,1024] + KV
- `cp_block_cache_i`（decode）: [1,1,1024] + [1,1] pos + [1,1,1,16] mask + [1,8,15,128] hist×2 → [1,1,1024] + 新 KV
- `cp_lm_head_g`（g=0..14）: [1,1,1024] → [1,1,2048]
- `cp_embedding_e`（e=0..13）: [1,1] int32 → [1,1,1024]

### codec decoder
- net: `codec_decoder`
- 输入: `codes [1,16,325] int32` → 输出: `wav [1,1,624000] f32`
- 短序列按 0 padding，输出截取前 `T*1920` 采样

## 关键说明

- 所有 int64 ONNX 输入已被 TPU-MLIR 降为 int32（C++ 用 int32 上传）
- attention_mask 为显式 4D 因果 mask（已绕开 transformers vmap mask）
- talker position_ids 为 3D（mRoPE），C++ 需传 [3,1,*] 全同值
- code_predictor prefill 固定 2 token，decode 历史最大 16（2+14）

## 精度验证状态

- codec decoder ONNX vs PyTorch: max_diff 4e-6 ✅
- talker 单层 wrapper vs PyTorch: max_diff 0 ✅
- embedding_code / codec_head ONNX vs PyTorch: ~0 / 9.5e-7 ✅
- bmodel vs PyTorch: 待板卡验证（M4）

## 编译脚本

- ONNX 导出: `python/export_talker.py` + `python/export_spike.py`
- bmodel 编译: `python/gen_bmodel.sh`（容器内运行）
