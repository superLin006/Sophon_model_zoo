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
│   ├── BM1684X/            # bmodel 文件（不入库，本地自行生成或下载）
│   │   ├── whisper_base_encoder_F32.bmodel
│   │   ├── whisper_base_encoder_F16.bmodel
│   │   ├── whisper_base_decoder_F32.bmodel
│   │   ├── whisper_base_decoder_F16.bmodel
│   │   ├── mel_80_filters.txt
│   │   ├── positional_embedding.npy
│   │   └── vocab.txt
│   └── onnx/               # 中间 ONNX 文件（不入库）
└── test_data/              # 测试音频（不入库）
```

## 快速开始

### 环境准备

1. **开发机**（WSL / Linux x86）
   - Python 3.8+，PyTorch，openai-whisper，onnx，onnxsim
   - Docker（用于 bmodel 转换和交叉编译）

2. **第三方库**（gitignore，需手动准备）
   - `0_Toolkits/soc-sdk-sp4/`：Sophon SOC SDK（包含 bmruntime 头文件和 so）
   - `1_third_party/fftw/`：fftw3 aarch64 静态库（用于 Mel 特征计算）

### Step 1：导出 ONNX

```bash
cd whisper/python
python export_onnx.py
# 产物：models/onnx/whisper_base_encoder.onnx
#        models/onnx/whisper_base_decoder.onnx
```

### Step 2：转换 bmodel

```bash
# 在 TPU-MLIR Docker 内执行（从仓库根目录）
docker run --rm \
  -v $(pwd):/workspace \
  -v $(pwd)/0_Toolkits:/toolkits \
  sophgo/tpuc_dev:latest \
  bash /workspace/whisper/python/gen_bmodel.sh F16

# 支持 F32（默认）或 F16
```

### Step 3：交叉编译 C++ 推理程序

```bash
# 先构建交叉编译镜像（只需一次）
docker build -t sophon-cross-build docker/

# 交叉编译
bash whisper/cpp/build.sh
# 产物：whisper/cpp/build/whisper_bm1684
```

### Step 4：部署到 BM1684X 板卡

```bash
# 上传二进制
scp whisper/cpp/build/whisper_bm1684 root@<board_ip>:/your/path/

# 上传模型文件（首次）
scp -r whisper/models/BM1684X/ root@<board_ip>:/your/path/models/

# 在板卡上运行
./whisper_bm1684 models/ test.wav zh F16
# 参数: <model_dir> <audio.wav> [zh|en] [F32|F16]
```

## 性能（BM1684X 实测，~5.8s 音频）

| 精度 | 总耗时（含加载） | 说明 |
|------|----------------|------|
| F32 | ~1.86s | |
| F16 | ~1.01s | 推荐生产环境使用 |

> Whisper 使用**自回归解码**（每个 token 一次前向），RTF 受输出文本长度影响，不适合用固定 RTF 衡量。

## 关键技术说明

- **KV Cache**：Decoder 分 `decoder_main`（首步）和 `decoder_loop`（循环步），past_k/v 手动管理，与 MTK 移植逻辑对齐
- **ONNX 导出 bug**：`[tensor] * n_layer` 创建共享引用，constant folding 会消除 KV 输入；必须用 `[torch.zeros(...) for _ in range(n)]` 列表推导式
- **Cross KV Cache**：Encoder 输出的 cross_k/cross_v 只在首步计算，后续复用
- **glibc 兼容**：使用 Ubuntu 20.04 Docker（gcc 9.4）交叉编译，产物兼容服务器 glibc 2.31
