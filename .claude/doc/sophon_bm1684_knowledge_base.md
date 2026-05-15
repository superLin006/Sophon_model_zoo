# Sophon BM1684X 移植知识库

**目标芯片**: BM1684X (SDK-23.09 LTS SP4)
**转换工具**: TPU-MLIR v1.28.1 (Docker: `sophgo/tpuc_dev:latest`)
**更新日期**: 2026-05-12

---

## 转换链路

```
PyTorch (.pt)
    ↓  export_onnx.py  [conda: sophon-export]
ONNX (.onnx)
    ↓  model_transform.py  [Docker: sophgo/tpuc_dev:latest]
.mlir
    ↓  model_deploy.py --chip bm1684x --quantize F32|F16
.bmodel
    ↓  交叉编译 C++ 推理程序  [Docker: sophon-cross-build]
aarch64 可执行文件  →  scp 到 BM1684X 板卡
```

---

## BM1684X 支持的精度

| 精度 | 支持 | 说明 |
|------|------|------|
| FP32 | ✅ | 精度最高，文件较大 |
| FP16 | ✅ | 推荐生产使用，速度约 4-8x，精度损失极小 |
| INT8 | ✅ | 需要校准数据集（run_calibration.py），本项目暂未使用 |

---

## 标准转换命令

### FP32 / FP16

```bash
# Step 1: ONNX → MLIR
model_transform.py \
    --model_name mymodel \
    --model_def mymodel_sim.onnx \
    --input_shapes [[1,3,224,224]] \
    --mlir mymodel.mlir

# Step 2: MLIR → bmodel (F32)
model_deploy.py \
    --mlir mymodel.mlir \
    --quantize F32 \
    --chip bm1684x \
    --model mymodel_F32.bmodel

# Step 2: MLIR → bmodel (F16)
model_deploy.py \
    --mlir mymodel.mlir \
    --quantize F16 \
    --chip bm1684x \
    --model mymodel_F16.bmodel
```

### INT8（需要校准数据）

```bash
run_calibration.py mymodel.mlir \
    --dataset ./calibration_data \
    --input_num 100 \
    -o mymodel_cali_table

model_deploy.py \
    --mlir mymodel.mlir \
    --quantize INT8 \
    --calibration_table mymodel_cali_table \
    --chip bm1684x \
    --model mymodel_INT8.bmodel
```

---

## 常见问题 & 已验证的 Workaround

### 1. Conv 节点缺少 `kernel_shape` 属性

**触发**：新版 `torch.onnx.export`（opset 17）导出时 Conv 节点不写 `kernel_shape`，`model_transform.py` 报错。

**修复**：导出后用以下函数补全：
```python
def _fix_conv_kernel_shape(model):
    init_map = {init.name: init for init in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        if any(attr.name == "kernel_shape" for attr in node.attribute):
            continue
        weight_name = node.input[1]
        if weight_name not in init_map:
            continue
        kernel_shape = list(init_map[weight_name].dims[2:])
        node.attribute.append(onnx.helper.make_attribute("kernel_shape", kernel_shape))
    return model
```

### 2. 含 KV Cache 的模型：dummy tensor 不能用 `* n` 乘法创建

**触发**：`[tensor] * n_layer` 创建的是同一个对象的 n 个引用，ONNX tracer 视为同一张量，constant folding 将 layer 1-N 的 KV 输入消除，bmodel 输入数量比预期少。

**修复**：必须用列表推导式，每层独立创建：
```python
# 错误
dummy_k = [torch.zeros(1, 448, 512)] * 6

# 正确
dummy_k = [torch.zeros(1, 448, 512) for _ in range(6)]
```

### 3. 复杂 Decoder 图需加 `--disable_layer_group`

**触发**：输入输出数量多（如 Whisper Decoder 28 输入 25 输出）时，`model_deploy.py` 的 layer group 优化不稳定，报错或生成异常 bmodel。

**修复**：
```bash
model_deploy.py --mlir ... --disable_layer_group --chip bm1684x ...
```

### 4. tpu_mlir pip install 在无外网容器内超时

**触发**：容器内 `pip install tpu_mlir*.whl` 尝试拉取网络依赖，无外网时挂起。

**修复**：先 `--no-deps` 安装，失败再正常安装（依赖已预装在容器内）：
```bash
WHL=$(ls /toolkits/tpu_mlir*.whl | head -1)
pip install "$WHL" -q --no-deps 2>/dev/null || pip install "$WHL" -q
```

### 5. 交叉编译产物 glibc 版本过高

**触发**：WSL 原生 gcc（15.x）编译的 aarch64 二进制在 BM1684X 板卡（Ubuntu 20.04, glibc 2.31）上报 `GLIBC_2.34 not found`。

**修复**：使用 Ubuntu 20.04 Docker 镜像（gcc 9.4）做交叉编译，产物最高依赖 glibc 2.29：
```dockerfile
FROM ubuntu:20.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    cmake g++-aarch64-linux-gnu gcc-aarch64-linux-gnu make
```

### 6. 确认板卡型号：BM1684 vs BM1684X

在板卡上执行：
```bash
bm-smi
# 输出包含 "1684X-SOC" 即为 BM1684X
```

`gen_bmodel.sh` 中必须使用 `--chip bm1684x`（非 `bm1684`），否则运行时报：
```
runtime arch[BM1684X] is not the same with bmodel arch[BM1684]
```

---

## 项目目录规范

```
<model_name>/
├── README.md               # 用法、性能、踩坑记录
├── python/
│   ├── export_onnx.py      # PyTorch → ONNX（开发机执行）
│   └── gen_bmodel.sh       # ONNX → bmodel（TPU-MLIR Docker 内执行）
├── cpp/
│   ├── CMakeLists.txt
│   ├── build.sh            # 交叉编译（sophon-cross-build Docker）
│   └── src/
├── models/
│   ├── BM1684X/            # bmodel + 资产文件（不入库，.gitkeep 占位）
│   └── onnx/               # 中间 ONNX（不入库，.gitkeep 占位）
└── test_data/              # 测试音频/图片（不入库，.gitkeep 占位）
```

---

## 移植检查清单

### ONNX 导出
- [ ] 模型可在 PyTorch 正常推理，输出 shape 符合预期
- [ ] dummy tensor 用列表推导式创建（KV Cache 场景必须）
- [ ] `onnx.checker.check_model()` 通过
- [ ] `_fix_conv_kernel_shape()` 处理 Conv 节点
- [ ] onnxsim 简化后验证精度一致

### bmodel 转换
- [ ] `--chip bm1684x`（注意不是 bm1684）
- [ ] `--quantize F32` 或 `F16`
- [ ] 复杂图（多输入输出）加 `--disable_layer_group`
- [ ] `bmrt_test --bmodel xxx.bmodel` 确认输入数量正确

### 交叉编译
- [ ] 使用 `sophon-cross-build` 镜像（Ubuntu 20.04 + gcc 9.4）
- [ ] 链接 `.so` 时设置 rpath：`-Wl,-rpath,/opt/sophon/libsophon-0.5.1/lib`
- [ ] 静态库依赖全部列出（如 kaldi-native-fbank 需同时链接 libkissfft-float.a）

### 板卡部署
- [ ] `bm-smi` 确认芯片型号
- [ ] 模型加载正常，输入输出数量与导出时一致
- [ ] 统计 RTF 时只计特征提取 + TPU 推理（不含模型加载）

---

---

## VITS-MeloTTS BM1684X 移植经验

### 推理链路说明

原始 MeloTTS 包含以下模块，整体是一个含动态 shape 和不支持算子的单图：

```
text tokens/tones
    ↓
[enc_p]  文本编码器（Transformer）
    ↓ h[1,192,L], dp_w[1,1,L], x_mask[1,1,L]
[DP]     Deterministic Duration Predictor
    ↓ dp_w（对数时长）
[MAS]    Monotonic Alignment Search（CPU，~8ms）
    ↓ z_p[1,192,T_mel]，y_mask[1,1,T_mel]
[Flow]   Normalizing Flow（逆变换，8 层）
    ↓
[Decoder] HiFi-GAN 声码器（上采样 512x）
    ↓
audio[1,1,T_audio]  @ 44100 Hz
```

**最终三段式拆分方案（全TPU，仅 MAS 在 CPU）：**

```
Part A（TPU）: enc_p + DP
    输入：x[1,128 int32], x_lengths[1 int32], tones[1,128 int32]
    输出：dp_w[1,1,128 f32], h[1,192,128 f32], x_mask[1,1,128 f32]
    文件：vits_part_a_F32.bmodel / vits_part_a_F16.bmodel

Part B（CPU，~8ms）: MAS
    dur[i] = ceil(exp(dp_w[i]) * x_mask[i])
    T_mel  = sum(dur)
    attn[T_mel, L] = 对角块矩阵
    z_p[Z_DIM, T_mel] = h[:, :L] @ attn.T   ← 注意 h 的步长是 L_MAX=128

Part C（TPU）: Flow（逆）+ Decoder
    输入：z_p[1,192,256 f32]（pad到T_MEL_FIXED=256），y_mask[1,1,256 f32]
    输出：audio[1,1,131072 f32]（截取前 T_mel*UPSAMPLE 个有效采样）
    文件：vits_part_c_F32.bmodel / vits_part_c_F16.bmodel
```

**性能（BM1684X，F32）：**
- Part A: ~6ms，Part B: ~8ms，Part C: ~305ms
- 总计 ~320ms，RTF ≈ 0.12（生成 2.7s 音频只需 320ms）
- 相比原始 CPU onnxruntime 方案（~6.8s），加速约 **20×**

---

### VITS-MeloTTS 移植遇到的问题与解决方案

#### 问题 1：SDP（随机时长预测器）含不支持算子 NonZero × 21

**触发**：`model_transform.py` 编译整图时报错，因为 SDP 内有大量 NonZero 算子（TPU 不支持）。

**解决**：在 `make_tpu_model.py` 中将 SDP 的输入替换为全零常量，只保留 DP（确定性时长预测器），彻底去掉 SDP 分支：
```python
# 将 /Add_1（SDP 输出）替换为零常量
zero = np.zeros(..., dtype=np.float32)
node = onnx.helper.make_node("Constant", [], ["/Add_1_output_0"], value=...)
```

#### 问题 2：Flow 模块含 RandomNormalLike（TPU 不支持）

**触发**：`model_transform.py` 报 `UNREACHABLE at Range.cpp` 或算子不支持。

**解决**：noise_scale=0 时该分支的贡献为零，直接将 `/Add_2_output_0` 替换为 `/Transpose_3_output_0`（即绕过随机噪声分支），并将 noise_scale 常量化为 0：
```python
# 绕过噪声分支，直接用 z_p 作为 Flow 输入
```

#### 问题 3：Range 算子导致整图无法编译到 TPU

**触发**：MAS 内部用 `torch.arange` 生成索引，shape 依赖运行时的 T_mel，TPU 无法静态编译。

**解决**：不要把 MAS 放到 TPU。将模型拆成三段：enc_p+DP（Part A）、MAS（CPU）、Flow+Decoder（Part C）。Part C 用固定 T_mel=256 编译 bmodel。

#### 问题 4：Part C bmodel 编译时 T_mel 必须静态化

**触发**：`part_c_flow_decoder.onnx` 的 T_mel 维度是动态 `unk__3`，`model_deploy.py` 需要静态 shape。

**解决**：在 `make_split_models.py` 中提取子图后用 onnxsim 固化 shape（`--overwrite-input-shape`），将 T_mel 强制设为 256，再编译 bmodel。推理时 z_p pad 到 256 列，输出截取前 T_mel * UPSAMPLE 个有效采样。

#### 问题 5：BMRuntime 无 BM_INT64 类型

**触发**：交叉编译报 `'BM_INT64' was not declared in this scope`。

**原因**：BM1684X SDK（bmdef.h）只有 BM_INT8/INT16/INT32/INT4，没有 INT64。TPU-MLIR 编译 ONNX 的 int64 输入时自动降为 int32。

**解决**：在 C++ 中将 int64 输入 cast 到 int32 再上传 device，dtype 设 BM_INT32：
```cpp
std::vector<int32_t> buf(n);
for (int i = 0; i < n; ++i) buf[i] = (int32_t)data[i];
t.dtype = BM_INT32;
```

#### 问题 6：matmul_ht 步长错误导致音频全部为噪声

**触发**：Part A bmodel 输出 h 存储为 `[Z_DIM=192, L_MAX=128]` 的连续缓冲区（步长 128），但 `matmul_ht` 传入 `L=seq_len`（如 61），导致跨行地址偏移错误，每个 channel 读到错误数据，z_p 值域严重偏小，输出全噪声。

**定位**：加调试打印后发现 `z_p[0..4]` 全是同一个值（第 0 个 phoneme 的 h 值），证明 matmul 按错误步长寻址。

**解决**：给 `matmul_ht` 增加 `h_stride` 参数，传入 `L_MAX` 而非 `seq_len`：
```cpp
// 错误
matmul_ht(h.data(), attn_vec.data(), Z_DIM, seq_len, T_mel, z_p.data());

// 正确
matmul_ht(h.data(), L_MAX,           // ← h 的行步长是 L_MAX，不是 seq_len
          attn_vec.data(), Z_DIM, seq_len, T_mel, z_p.data());
```

**教训**：bmodel 输出缓冲区大小由编译时的固定 shape 决定（此处 L_MAX=128），而非运行时的实际序列长度。在 CPU 端操作 bmodel 输出的多维数组时，**步长必须使用 bmodel 的固定维度，而非实际输入长度**。

---

## 相关资源

- SDK: SDK-23.09 LTS SP4
- TPU-MLIR: https://github.com/sophgo/tpu-mlir
- Sophon Demo 参考: https://github.com/sophgo/sophon-demo
