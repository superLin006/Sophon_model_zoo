# Whisper Base — BM1684X 移植

Whisper-base 语音识别模型完整移植到 Sophon BM1684X，支持 FP32 / FP16，中英文均可识别。

## 目录结构

```
whisper/
├── python/
│   ├── export_onnx.py      # Step 1: PyTorch → ONNX（在开发机执行）
│   └── gen_bmodel.sh       # Step 2: ONNX → bmodel（在 TPU-MLIR Docker 内执行）
├── cpp/
│   ├── CMakeLists.txt
│   ├── build.sh            # 交叉编译脚本（依赖 sophon-cross-build Docker 镜像）
│   └── src/
│       ├── main.cpp
│       ├── whisper_inference.h / .cpp
│       └── utils/
│           └── audio_utils.h / .cpp   # WAV 读取 + Mel 特征计算（依赖 fftw3）
├── models/
│   ├── BM1684X/            # bmodel 文件（不入库，本地自行生成）
│   │   ├── whisper_base_encoder_F32.bmodel
│   │   ├── whisper_base_encoder_F16.bmodel
│   │   ├── whisper_base_decoder_F32.bmodel
│   │   ├── whisper_base_decoder_F16.bmodel
│   │   ├── mel_80_filters.txt          # 80 维 Mel 滤波器系数
│   │   ├── positional_embedding.npy    # Decoder positional embedding [448, 512]
│   │   └── vocab.txt                   # token 词表
│   └── onnx/               # 中间 ONNX 文件（不入库）
└── test_data/              # 测试音频（不入库）
```

## 快速开始

### 0. 环境准备

**开发机**（WSL / Linux x86）需要：
- Python 3.8+、PyTorch、openai-whisper、onnx、onnxsim
- Docker（bmodel 转换 + 交叉编译）

**第三方库**（gitignore，需手动准备到仓库内）：
- `0_Toolkits/soc-sdk-sp4/`：Sophon SOC SDK（bmruntime 头文件 + so 库）
- `1_third_party/fftw/`：fftw3f aarch64 静态库（用于 Mel 特征计算）

### Step 1：导出 ONNX

```bash
cd whisper/python
python export_onnx.py
```

产物（位于 `models/onnx/`）：
```
whisper_base_encoder_sim.onnx     # Encoder
whisper_base_decoder_sim.onnx     # Decoder（单步，含完整 KV Cache IO）
```

> 同时会在 `models/BM1684X/` 生成 `mel_80_filters.txt`、`positional_embedding.npy`、`vocab.txt`，
> 这三个文件是 C++ 推理必须的资产文件，需一并部署到板卡。

### Step 2：转换 bmodel

```bash
# 从仓库根目录执行，-v 把 0_Toolkits 挂载为 /toolkits 供脚本找 whl
docker run --rm \
  -v $(pwd)/whisper:/workspace \
  -v $(pwd)/0_Toolkits:/toolkits \
  sophgo/tpuc_dev:latest \
  bash /workspace/python/gen_bmodel.sh F16

# 支持 F32（默认）或 F16
```

产物：
```
models/BM1684X/whisper_base_encoder_F16.bmodel
models/BM1684X/whisper_base_decoder_F16.bmodel
```

### Step 3：交叉编译 C++ 推理程序

```bash
# 先构建交叉编译镜像（只需一次）
docker build -t sophon-cross-build docker/

# 从仓库根目录执行
bash whisper/cpp/build.sh
# 产物：whisper/cpp/build/whisper_bm1684
```

### Step 4：部署到 BM1684X 板卡

```bash
# 上传二进制
scp whisper/cpp/build/whisper_bm1684 root@<board_ip>:/path/to/whisper/

# 首次上传模型文件
scp whisper/models/BM1684X/whisper_base_encoder_F16.bmodel \
    whisper/models/BM1684X/whisper_base_decoder_F16.bmodel \
    whisper/models/BM1684X/mel_80_filters.txt \
    whisper/models/BM1684X/positional_embedding.npy \
    whisper/models/BM1684X/vocab.txt \
    root@<board_ip>:/path/to/whisper/models/

# 在板卡上运行
./whisper_bm1684 models/ test.wav zh F16
# 参数: <model_dir> <audio.wav> [zh|en] [F32|F16]
# 默认值: language=zh, precision=F32
```

## 性能（BM1684X 实测，~5.8s 音频）

| 精度 | 总耗时（含模型加载） |
|------|-----------------|
| F32 | ~1.86s |
| F16 | ~1.01s |

> Whisper 使用**自回归解码**，每个 token 跑一次 Decoder 前向，耗时随输出文本长度线性增长，
> 不适合用固定 RTF 衡量。推荐生产使用 F16。

## 模型结构说明

Whisper 推理分两阶段：

```
音频 WAV
  → Mel 特征 [1, 80, 3000]
  → Encoder → audio_features [1, 1500, 512]

Decoder（自回归循环，每步一个 token）：
  输入：token_id、audio_features、pos_emb、attn_mask、past_self_k/v × 6、cross_k/v × 6
  输出：logits、new_self_k/v × 6、new_cross_k/v × 6
```

- **Cross KV**：首步由 `audio_features` 计算得到，之后每步复用，不重复计算
- **Self KV**：每步追加到 KV Cache，Cache 长度从 0 增长到最多 448

## 踩过的坑

### 1. ONNX 导出：`[tensor] * n_layer` 导致 KV 输入被 constant folding 消除

**现象**：导出的 ONNX/bmodel 只有 8 个输入，而不是预期的 28 个（1 + 1 + 1 + 1 + 6×4）。`bmrt_test` 验证时 KV Cache 输入全部消失。

**原因**：Python 的 `[tensor] * 6` 创建的是同一个对象的 6 个引用，ONNX tracer 识别为同一张量，constant folding 将 layer 0-4 的 KV 全部折叠掉。

**修复**：必须用列表推导式，每层创建独立张量：
```python
# 错误写法
dummy_past_self_k = [torch.zeros(1, PADDING_SIZE, n_state)] * n_layer

# 正确写法
dummy_past_self_k = [torch.zeros(1, PADDING_SIZE, n_state) for _ in range(n_layer)]
```

### 2. ONNX 导出：Conv 节点缺少 `kernel_shape` 属性

**现象**：`onnx.checker.check_model()` 报错或 tpu-mlir `model_transform.py` 报 `KeyError: 'kernel_shape'`。

**原因**：新版 `torch.onnx.export`（opset 17）在某些情况下不写入 Conv 的 `kernel_shape` 属性。

**修复**：导出后从 weight initializer 的 dims 推导补全：
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

### 3. bmodel 转换：Decoder 必须加 `--disable_layer_group`

**现象**：不加时 `model_deploy.py` 对 Decoder 报错或生成异常 bmodel。

**原因**：Decoder 图结构复杂（28 输入、25 输出），layer group 优化在此类图上不稳定。

**修复**：在 `gen_bmodel.sh` 中对 Decoder 单独加参数：
```bash
model_deploy.py \
    --mlir "whisper_base_decoder.mlir" \
    --quantize F16 \
    --chip bm1684x \
    --disable_layer_group \          # ← 必须加
    --model "..."
```

### 4. tpu_mlir pip install 超时

**现象**：Docker 容器内 `pip install tpu_mlir*.whl` 尝试从网络拉取依赖，在无外网环境下超时失败。

**修复**：先尝试 `--no-deps` 跳过依赖，失败再正常安装（依赖已在容器内预装）：
```bash
pip install "$WHL" -q --no-deps 2>/dev/null || pip install "$WHL" -q
```

### 5. 交叉编译：glibc 版本不兼容

**现象**：WSL 原生 gcc（15.x）编译的二进制在 BM1684X 板卡（Ubuntu 20.04, glibc 2.31）上报 `GLIBC_2.34 not found`。

**原因**：高版本 gcc 默认链接新版 glibc 符号。

**修复**：用 Ubuntu 20.04 Docker 镜像（gcc 9.4）做交叉编译，产物最高依赖 glibc 2.29，兼容板卡环境。参见 `docker/Dockerfile.cross-build`。

### 6. 板卡芯片识别：BM1684 vs BM1684X

**现象**：运行时报 `runtime arch[BM1684X] is not the same with bmodel arch[BM1684]`。

**原因**：板卡实际是 BM1684X，但早期 `gen_bmodel.sh` 中 `chip="bm1684"` 写错。

**修复**：`gen_bmodel.sh` 中统一使用 `chip="bm1684x"`，用 `bm-smi` 确认板卡型号：
```bash
bm-smi   # 输出中看 "1684X-SOC" 即为 BM1684X
```
