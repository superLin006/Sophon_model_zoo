# Sophon Model Zoo

Sophon BM1684X 平台深度学习模型移植工作区，基于 SDK-23.09 LTS SP4。

## 支持的模型

| 模型 | 类型 | 精度 | 指标 | 状态 |
|------|------|------|------|------|
| [Whisper Base](whisper/) | 语音识别（自回归） | FP16 | — | ✅ 完成 |
| [SenseVoice Small](sensevoice/) | 语音识别 + 情感/事件（CTC） | FP16 | RTF 0.0095 | ✅ 完成 |
| [ChatTTS](chatTTS/) | 文本转语音（自回归 + DVAE + Vocos） | GPT INT4 + FP16 | RTF 0.53（非流式）/ 0.59（流式），TTFA ~980ms | ✅ 完成 |
| [VITS-MeloTTS](vits-melo-tts-zh_en/) | 文本转语音（中英双语） | FP32 | RTF ~0.12 | ✅ 完成 |

## 项目结构

```
Sophon_model_zoo/
├── 0_Toolkits/               # Sophon SOC SDK（不入库，.gitkeep 占位）
├── 1_third_party/            # 第三方库头文件与静态库（不入库，.gitkeep 占位）
│   ├── fftw/                 # fftw3 aarch64（whisper / chatTTS iSTFT 使用）
│   ├── kaldi_native_fbank/   # kaldi-native-fbank aarch64（sensevoice 使用）
│   ├── sophon/               # Sophon SDK 头文件 + libbmlib/libbmrt（chatTTS 使用）
│   └── nlohmann/             # nlohmann/json.hpp 头文件（chatTTS 使用）
├── 2_utils/                  # 公共 C 工具库（图像/音频处理）
├── 3_docker/
│   ├── Dockerfile.cross-build  # Ubuntu 20.04 aarch64 交叉编译镜像
│   ├── run_docker.sh           # 启动 TPU-MLIR 转换容器
│   └── README.md
├── environment.yml           # Conda 环境（模型导出用）
├── whisper/                  # Whisper Base 移植
├── sensevoice/               # SenseVoice Small 移植
├── chatTTS/                  # ChatTTS 移植（纯 bmruntime C++，支持流式）
└── vits-melo-tts-zh_en/      # VITS-MeloTTS 中英双语移植
```

## 转换流程

```
PyTorch (.pt)
    ↓  export_onnx.py  [Conda: sophon-export]
ONNX (.onnx)
    ↓  gen_bmodel.sh  [Docker: sophgo/tpuc_dev:latest]
BModel (.bmodel)
    ↓  build.sh  [Docker: sophon-cross-build]
aarch64 可执行文件  →  scp 到 BM1684X 板卡运行
```

## 快速开始

### 1. 准备 Conda 环境

```bash
conda env create -f environment.yml
conda activate sophon-export
```

### 2. 构建交叉编译镜像（只需一次）

```bash
docker build -t sophon-cross-build 3_docker/
```

### 3. 移植某个模型

参考各模型目录下的 `README.md`：
- [whisper/README.md](whisper/README.md)
- [sensevoice/README.md](sensevoice/README.md)
- [chatTTS/README.md](chatTTS/README.md)
- [vits-melo-tts-zh_en/README.md](vits-melo-tts-zh_en/README.md)

## 技术栈

- **芯片**: Sophon BM1684X
- **SDK**: SDK-23.09 LTS SP4
- **转换工具**: TPU-MLIR v1.28.1（`sophgo/tpuc_dev:latest`）
- **交叉编译**: Ubuntu 20.04 + gcc 9.4（兼容板卡 glibc 2.31）
- **支持精度**: FP32 / FP16 / INT8 / INT4
