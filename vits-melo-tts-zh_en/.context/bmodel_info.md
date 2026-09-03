# vits-melo-tts-zh_en BM1684X bmodel 转换报告

## 转换结果

### 生成的 bmodel 文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `models/BM1684X/vits-melo-tts-zh_en_decoder_T256_F32.bmodel` | 62 MB | HiFi-GAN 解码器，F32 精度 |
| `models/BM1684X/vits-melo-tts-zh_en_decoder_T256_F16.bmodel` | 34 MB | HiFi-GAN 解码器，F16 精度 |

### bmodel 接口

**bmodel 输入（2个）：**

| 名称 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `/Mul_10_output_0` | [1, 192, 256] | float32 | Flow 解码器输出（z_hat），padding 到 T_mel_fixed |
| `/Unsqueeze_6_output_0` | [1, 256, 1] | float32 | Speaker embedding |

**bmodel 输出（1个）：**

| 名称 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `y` | [1, 1, 131072] | float32 | 音频波形（T_mel=256 * 512 = 131072 samples ≈ 2.97s @ 44100Hz） |

---

## 为什么不是完整模型的 bmodel

原始 `model.onnx` (163MB) **无法**直接编译为 bmodel，原因如下：

### 原因 1：`RandomNormalLike` (×2) — TPU-MLIR 未实现
- 两个节点：`/sdp/RandomNormalLike`，`/RandomNormalLike`
- 错误：`UNREACHABLE executed at RandnLike.cpp:20`
- 方案 A 尝试：将 RandomNormalLike 替换为 graph input → 仍遇到问题 2

### 原因 2：`NonZero` (×21) — 数据相关 shape
- 全部位于 SDP（Stochastic Duration Predictor）模块中
- NonZero 输出 shape 取决于运行时数值，不是输入 shape
- 导致 TPU-MLIR 的 `--shape-infer` pass 崩溃：`commonShapeValInfer assertion failed`
- 方案尝试：`--op_custom_shape` 仍然崩溃

### 原因 3：`T_mel` 动态维度
- T_mel = sum(ceil(exp(log_w))) 由 duration predictor 在运行时决定
- 对于 L_padded=128，T_mel 范围约 100-400（取决于文本内容）
- TPU-MLIR 的静态编译无法处理此动态 shape

**注意：`NonZero` 在 TPU-MLIR 的 OnnxConverter 中已有映射（`top.NonZeroOp`），但 `--shape-infer` 的 `commonShapeValInfer` 对数据相关的输出 shape 会崩溃。这是 TPU-MLIR v1.28.1 的已知限制。**

---

## 解决方案：模型拆分

```
Part 1（CPU/ONNX）: model.onnx 中动态部分
  Input:  x[1,128], x_lengths[1], tones[1,128], sid[1],
          noise_scale[1], length_scale[1], noise_scale_w[1]
  Key ops: 文本编码器 + SDP 时长预测 + 单调对齐 + Flow 反向
  Output: z_hat[1,192,T_mel]   (动态 T_mel)
          g_emb[1,256,1]       (固定)
          T_mel                (整数，由内容决定)

Part 2（BM1684X bmodel）: decoder_T256.onnx
  Input:  z_hat_padded[1,192,256]  (Pad/trim to T_mel_fixed=256)
          g_emb[1,256,1]
  Output: audio[1,1,131072]        (固定长度)
  有效帧: audio[:,:,:T_mel*512]    (截取 T_mel_actual*512 个采样)
```

---

## C++ 推理流程

```cpp
// Step 1: 运行 ONNX Part 1 (CPU)
// 使用 onnxruntime 运行 model.onnx，获取中间输出：
//   - /flow/flows.0/Concat_output_0  [1,192,T_mel]  → z_hat
//   - /Unsqueeze_6_output_0          [1,256,1]      → g_emb
// 同时记录 T_mel_actual

// Step 2: Pad z_hat 到固定长度
// float z_hat_padded[1][192][T_MEL_FIXED] = {0};
// memcpy(z_hat_padded, z_hat, 1*192*T_mel_actual * sizeof(float));

// Step 3: 运行 bmodel decoder
// bm_tensor_t inputs[2] = {z_hat_padded, g_emb};
// 运行 bmodel，得到 audio_out[1,1,131072]

// Step 4: 截取有效音频
// int valid_samples = T_mel_actual * 512;
// float* audio_valid = audio_out;  // 只使用前 valid_samples 个
// 输出到 WAV 文件（44100Hz）
```

---

## 验证结果

### test_zh（L_actual=61，T_mel_actual=232）

| 指标 | 值 |
|------|-----|
| audio_full range | [-0.130, +0.144]（正常，有效声音信号） |
| decoder mean_diff vs baseline | 0.000013 |
| decoder max_diff vs baseline | 0.004094 |
| 结论 | **PASS** |

### test_en_zh（L_actual=53，T_mel_actual=223）

| 指标 | 值 |
|------|-----|
| audio_full range | [-0.267, +0.239]（正常） |
| decoder mean_diff vs baseline | 0.000017 |
| decoder max_diff vs baseline | 0.004376 |
| 结论 | **PASS** |

### Bug 说明
test_onnx.py 初版传入了 sid=0（应为 sid=1，来自模型 metadata speaker_id=1）。
sid=0 时模型输出接近全零（静音），已修复。C++ 端必须传 sid=1。

---

## 调试文件（`python/test/outputs/debug/`）

| 文件 | Shape | 说明 |
|------|-------|------|
| `test_zh_input_x_padded.npy` | [1, 128] int64 | padding 后的 token IDs |
| `test_zh_input_tones_padded.npy` | [1, 128] int64 | padding 后的 tone IDs |
| `test_zh_input_x_lengths.npy` | [1] int64 | 实际长度（=61） |
| `test_zh_z_hat.npy` | [1, 192, 184] float32 | flow 输出（动态 T_mel） |
| `test_zh_z_hat_padded.npy` | [1, 192, 256] float32 | 填充后（bmodel 输入） |
| `test_zh_g_emb.npy` | [1, 256, 1] float32 | Speaker embedding |
| `test_zh_audio_full.npy` | [1, 1, 94208] float32 | baseline 完整音频 |
| `test_zh_audio_decoder.npy` | [1, 1, 131072] float32 | decoder 输出（含 padding 帧） |

---

## 相关文件

| 文件 | 说明 |
|------|------|
| `python/fix_onnx.py` | 修复 ONNX 脚本（含 decoder 子模型提取） |
| `python/gen_bmodel.sh` | bmodel 转换脚本（decoder 子模型） |
| `python/test/test_onnx.py` | ONNX 验证脚本 |
| `models/onnx/vits-melo-tts-zh_en/model_fixed.onnx` | 方案 A ONNX（移除 RandomNormalLike，但 T_mel 仍动态） |
| `models/onnx/vits-melo-tts-zh_en/decoder_T256.onnx` | decoder 子模型（可直接编译 bmodel） |

---

## 方案 C 实际报错（完整 model.onnx 编译失败）

```
UNREACHABLE executed at /__w/tpu-mlir/tpu-mlir/lib/Dialect/Top/Interfaces/RandnLike.cpp:20!
```

## 方案 A 实际报错（model_fixed.onnx 编译失败）

```
tpuc-opt: commonShapeValInfer: Assertion `mlir::succeeded(ret)' failed.
```
（由 SDP 内的 NonZero 节点触发，其输出 shape 为数据相关）

---

*生成时间: 2026-05-14*
*TPU-MLIR 版本: v1.28.1-20260429*
*Chip: BM1684X*
