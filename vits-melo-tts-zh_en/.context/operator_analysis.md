# 算子兼容性分析

**模型**: vits-melo-tts-zh_en（VITS-MeloTTS 中英混合 TTS）
**ONNX 路径**: models/onnx/vits-melo-tts-zh_en/model.onnx
**分析日期**: 2026-05-14
**工具**: TPU-MLIR v1.28.1 (sophgo/tpuc_dev:latest), ONNX opset 18

---

## 模型使用的算子（共 50 种，10680 节点）

| 算子 | 数量 | 兼容性 | 备注 |
|------|-----:|:------:|------|
| Add | 285 | ✅ | |
| And | 3 | ✅ | |
| Cast | 314 | ✅ | |
| Ceil | 1 | ✅ | |
| Clip | 1 | ✅ | |
| Concat | 543 | ✅ | |
| Constant | 4497 | ✅ | 大量常量（VITS 展开流典型特征） |
| ConstantOfShape | 246 | ✅ | |
| Conv | 249 | ✅ | opset 18 可能缺 kernel_shape，需 _fix_conv_kernel_shape() |
| ConvTranspose | 5 | ✅ | |
| CumSum | 7 | ✅ | |
| Div | 59 | ✅ | |
| Equal | 64 | ✅ | |
| Erf | 24 | ✅ | |
| Exp | 7 | ✅ | |
| Expand | 93 | ✅ | |
| Gather | 383 | ✅ | 用于 Embedding 查表，int64 索引 |
| GatherElements | 21 | ✅ | |
| GatherND | 15 | ✅ | |
| GreaterOrEqual | 6 | ✅ | |
| LayerNormalization | 62 | ✅ | |
| LeakyRelu | 86 | ✅ | |
| Less | 3 | ✅ | |
| LessOrEqual | 3 | ✅ | |
| MatMul | 79 | ✅ | |
| Mul | 371 | ✅ | |
| Neg | 10 | ✅ | |
| NonZero | 21 | ✅ | SDP 内部动态索引，固定 shape 后可正常编译 |
| Not | 3 | ✅ | |
| Pad | 154 | ✅ | |
| Pow | 21 | ✅ | |
| Range | 41 | ✅ | |
| ReduceMax | 1 | ✅ | |
| ReduceSum | 4 | ✅ | |
| Relu | 20 | ✅ | |
| Reshape | 525 | ✅ | |
| ScatterND | 31 | ✅ | |
| Shape | 504 | ✅ | 固定 shape 后大量 Shape 节点可被 constant-fold |
| Slice | 310 | ✅ | |
| Softmax | 24 | ✅ | |
| Softplus | 3 | ✅ | |
| Split | 9 | ✅ | |
| Sqrt | 3 | ✅ | |
| Squeeze | 1 | ✅ | |
| Sub | 216 | ✅ | |
| Tanh | 1 | ✅ | |
| Transpose | 388 | ✅ | |
| Unsqueeze | 881 | ✅ | |
| Where | 80 | ✅ | |
| **RandomNormalLike** | **2** | **⚠️** | **不在 TPU-MLIR 支持列表，需处理** |

---

## 需要处理的问题

### 1. RandomNormalLike（唯一不支持算子）

**作用分析**：
- 节点 1 `/RandomNormalLike`：VITS 先验分布采样噪声 z，shape 随 T（动态音频长度）变化。后接 `Mul(/Mul_8)` * sigma，再 `Mul(/Mul_9)` * `noise_scale`，最终 `Add` 到均值张量。公式为 `z = mu + noise_scale * sigma * eps`，eps ~ N(0,1)。
- 节点 2 `/sdp/RandomNormalLike`：SDP（Stochastic Duration Predictor）内部噪声，shape 随序列长度 L 动态变化。后接 `Cast` + `Mul` * `noise_scale_w`。

**可选方案**：

| 方案 | 描述 | 代价 | 推荐 |
|------|------|------|------|
| A（推荐）| 将两个 RandomNormalLike 替换为新增图输入（`z_flow`, `z_sdp`），CPU 侧调用前 numpy 生成随机数传入 | 需修改 ONNX + C++ 增加 2 个输入 | ✅ 保留随机性 |
| B | 替换为全零常量（deterministic mode，等价 noise_scale=0 时效果）| 输出不含噪声，语音略显机械 | ⚠️ 音质略降 |
| C | 固定 shape 后直接 `model_transform.py` 尝试，观察是否 fallback | 若 tpu-mlir 内部有扩展支持则无需修改 | 先试 |

**推荐策略**：先执行方案 C（直接尝试转换），若 `model_transform.py` 报错再执行方案 A。

**方案 A 实现要点**：
```python
# 修改 ONNX 图：用 graph input 替代 RandomNormalLike
# 1. 添加 graph input: z_flow [1, 2, L_flow] float32，z_sdp [1, 2, L_sdp] float32
#    注意 L_flow 和 L_sdp 在固定 x shape 后均为确定值
# 2. 删除 RandomNormalLike 节点
# 3. 将其 output 重定向到新 input
# C++ 侧：推理前 std::mt19937 生成随机 float32 数组传入
```

### 2. 动态 Shape

**问题**：
- `x`/`tones`：L 维度动态（序列长度，含 blank 后 = 2*N_phonemes + 1）
- `y`：T 维度动态（音频长度 ≈ L * duration * length_scale * 256）

**解决方案**：转换时用 `--input_shapes` 固定，运行时 padding/截断，无需修改 ONNX。

**推荐 shape 策略**：

| 版本 | L（含 blank） | 对应原始音素数 | 覆盖场景 |
|------|-------------|--------------|---------|
| 短句（推荐） | 128 | 63 个音素 ≈ 30~40 个汉字，或 60 个英文词 | 覆盖 95% 日常短句 |
| 长句 | 256 | 127 个音素 ≈ 60~80 个汉字 | 覆盖长文本 |

**推荐方案**：编译 **2 个 bmodel**（L=128 和 L=256），C++ 侧按实际序列长度选择。

Baseline 测试用例：test_zh L=61，test_en_zh L=53，均在 L=128 范围内。

**具体转换参数**：
```bash
# L=128 版本
model_transform.py \
    --model_name vits_melo_tts_L128 \
    --model_def model_fixed.onnx \
    --input_shapes [[1,128],[1],[1,128],[1],[1],[1],[1]] \
    --mlir vits_melo_tts_L128.mlir

model_deploy.py \
    --mlir vits_melo_tts_L128.mlir \
    --quantize F32 \
    --chip bm1684x \
    --model vits_melo_tts_L128_F32.bmodel

# L=256 版本（同理，改 input_shapes 为 [[1,256],[1],[1,256],[1],[1],[1],[1]]）
```

注意：若方案 A 生效，input_shapes 需增加 z_flow/z_sdp 两项（形状在固定 L 后可静态推断）。

### 3. Conv kernel_shape 补全

**问题**：opset 18 导出的 Conv 节点可能缺少 `kernel_shape` 属性（知识库已知问题 #1）。

**解决方案**：在导出/修改 ONNX 后调用 `_fix_conv_kernel_shape()` 函数补全。

### 4. 复杂图结构（大节点数）

**问题**：10680 个节点，4497 个 Constant，图结构复杂。

**解决方案**：转换时加 `--disable_layer_group` 参数（知识库已知问题 #3）。

---

## 移植策略

**整体评估**：本模型移植**复杂度中等**，49/50 算子（98%）直接支持，唯一问题是 `RandomNormalLike`。

**推荐执行顺序**：

```
阶段 0: 验证（不修改 ONNX）
  └─ 直接尝试 model_transform.py 转换
  └─ 若 RandomNormalLike 报错 → 进入阶段 1

阶段 1: 修改 ONNX（fix_onnx.py）
  ├─ 替换 RandomNormalLike → 新 graph input（z_flow, z_sdp）
  ├─ 补全 Conv kernel_shape
  └─ onnxsim 简化后验证精度（与 baseline 对比）

阶段 2: 转换 bmodel（gen_bmodel.sh）
  ├─ L=128 + F32（基准验证）
  ├─ L=128 + F16（生产版本）
  └─ L=256 + F16（长句版本，可选）

阶段 3: C++ 推理
  ├─ 继承 baseline.md 的前处理逻辑（lexicon/tone 查表、add_blank）
  ├─ padding/截断到固定 L
  ├─ 若方案 A：CPU 侧 std::normal_distribution 生成 z_flow/z_sdp
  └─ 输出 clip + int16 写 WAV
```

**是否需要创建 export_onnx.py**：
- 如果方案 C 成功（直接转换）：**不需要任何 ONNX 修改**，只需 `gen_bmodel.sh`。
- 如果需要处理 RandomNormalLike（方案 A）：需创建 `python/fix_onnx.py`（非 export_onnx.py，因 ONNX 已存在，只需修图）。

---

## 风险评估

| 风险项 | 等级 | 说明 |
|--------|:----:|------|
| RandomNormalLike 不支持 | 中 | 仅 2 个节点，替换方案清晰；方案 A 修改量小 |
| 大图编译超时/OOM | 低-中 | 10680 节点较多，可能需要 `--disable_layer_group` |
| F16 精度损失 | 低 | TTS 对精度敏感，建议先 F32 验证再 F16 |
| NonZero 动态性 | 低 | 固定 shape 后 NonZero 输入确定，tpu-mlir 应能处理 |
| 输出 T 维度 padding | 低 | bmodel 固定 L 后 T 也固定（由 duration predictor 决定），C++ 需裁剪尾部 silence |

**综合风险等级**: **低-中**

主要风险集中在 `RandomNormalLike` 和大图编译稳定性；算子层面几乎全覆盖，无需大幅重构。

---

## 关键数据

- 模型节点数：10680（其中 Constant 4497）
- 独特算子数：50 种
- 支持算子：49 种（98%）
- 不支持算子：1 种（RandomNormalLike × 2 节点）
- ONNX opset：18
- 无控制流（If/Loop/Scan）
- 无自定义算子
