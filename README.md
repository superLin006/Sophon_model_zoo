# Sophon Model Zoo

Sophon BM1684X 平台深度学习模型移植工作区，基于 SDK-23.09 LTS SP4。

## 支持的模型

| 模型 | 类型 | 精度 | 指标 | 状态 |
|------|------|------|------|------|
| [Whisper](whisper/) | 语音识别（自回归，base / large-v3-turbo） | FP16 / W4F16 | base 端到端 ~1.0s；turbo ~1.9s（W4F16，中英无损） | ✅ 完成 |
| [SenseVoice Small](sensevoice/) | 语音识别 + 情感/事件（CTC） | FP16 | RTF 0.0095 | ✅ 完成 |
| [Moonshine](moonshine/) | 语音识别（轻量流式 ASR，streaming-small） | FP16 / FP32 | RTF 0.045（F16，6.6s 音频实测） | ✅ 完成 |
| [Zipformer](zipformer/) | 流式中英双语语音识别（Transducer） | FP16 | RTF 0.024–0.071，8/8 token-perfect vs Python Sail | ✅ 完成 |
| [ChatTTS](chatTTS/) | 文本转语音（自回归 + DVAE + Vocos） | GPT INT4 + FP16 | RTF 0.53（非流式）/ 0.59（流式），TTFA ~980ms | ✅ 完成 |
| [VITS-MeloTTS](vits-melo-tts-zh_en/) | 文本转语音（中英双语） | FP32 | RTF ~0.12 | ✅ 完成 |
| [Eureka-Audio](Eureka-Audio/) | 音频指令分类（whisper encoder + Qwen3-1.7B） | W4BF16 | 准确率 ~90%，端到端 ~2.3s/条（Python·sail / C++ 均跑通） | ✅ 完成 |
| [QwenLLM 系列](QwenLLM/) | LLM 意图识别（Qwen3-0.6B，v95 系列） | W4BF16 / W8BF16 | v95 recall + 意图分类组合，FTL ~0.3s | ✅ 完成 |
| [Qwen3-ASR](Qwen3-ASR/) | 语音识别 + 语种识别（30 语种 + 22 中文方言，LLM 类） | F16 + W4BF16 | 离线 RTF 0.15（0.85s）；流式 1s 块实时（平均 RTF 0.64） | ✅ 完成 |

## 项目结构

```
Sophon_model_zoo/
├── 0_Toolkits/               # Sophon SOC SDK（不入库，.gitkeep 占位）
├── 1_third_party/            # 第三方库头文件与静态库（不入库，.gitkeep 占位）
├── 2_utils/                  # 公共 C 工具库（图像/音频处理）
├── 3_docker/
│   ├── Dockerfile.cross-build  # Ubuntu 20.04 aarch64 交叉编译镜像
│   ├── run_docker.sh           # 启动 TPU-MLIR 转换容器
│   └── README.md
├── environment.yml           # Conda 环境（模型导出用）
├── whisper/                  # Whisper 移植（base + large-v3-turbo，FP16/W4F16）
├── sensevoice/               # SenseVoice Small 移植
├── moonshine/                # Moonshine streaming-small 移植（轻量流式 ASR，FP16/FP32）
├── zipformer/                # Zipformer 流式中英双语 ASR（103/96 streaming，C++ BMRuntime CLI）
├── chatTTS/                  # ChatTTS 移植（纯 bmruntime C++，支持流式）
├── vits-melo-tts-zh_en/      # VITS-MeloTTS 中英双语移植
├── Eureka-Audio/             # 音频指令分类（whisper encoder + Qwen3-1.7B，Python·sail + C++）
├── Qwen3-ASR/                # Qwen3-ASR-0.6B 语音识别 + 语种识别（C++ bmrt + sail，支持流式）
└── QwenLLM/                  # Qwen 系列 LLM 意图识别（v95 系列）
    ├── LLM-TPU/              # sophgo 官方 demo（不入库，本地克隆）
    ├── scripts/              # 编译/部署/下载脚本
    ├── cpp/                  # C++ BMRuntime 推理
    └── demo/                 # Python 演示
```

## 转换流程

```
PyTorch / Safetensors
    ↓  llm_convert.py  [Docker: sophgo/tpuc_dev:latest, TPU-MLIR v1.28.1]
BModel (.bmodel)
    ↓  deploy_to_board.sh
aarch64 板卡运行
```

非 LLM 模型额外需要 ONNX 中间步骤：

```
PyTorch (.pt)  →  export_onnx.py  →  ONNX  →  gen_bmodel.sh  →  BModel
```

## 快速开始

### 1. 准备 Conda 环境

```bash
conda env create -f environment.yml
conda activate sophon-export
```

### 2. 启动 TPU-MLIR 容器

```bash
./3_docker/run_docker.sh
```

### 3. 移植某个模型

参考各模型目录下的 `README.md`：
- [whisper/README.md](whisper/README.md)
- [sensevoice/README.md](sensevoice/README.md)
- [zipformer/README.md](zipformer/README.md)（流式中英双语 Transducer）
- [chatTTS/README.md](chatTTS/README.md)
- [vits-melo-tts-zh_en/README.md](vits-melo-tts-zh_en/README.md)
- [Eureka-Audio/](Eureka-Audio/)
- [QwenLLM/](QwenLLM/)

## 技术要点

- **芯片**: Sophon BM1684X，SDK-23.09 LTS SP4
- **转换工具**: TPU-MLIR v1.28.1（`sophgo/tpuc_dev:latest`）
- **LLM 转换**: `llm_convert.py`，AWQ 量化只支持 `--quantize w4f16`
- **容器 transformers 版本**: 固定 4.51.1（`pip install transformers==4.51.1`），5.x 与 PyTorch 2.1.0+cpu 不兼容
- **交叉编译**: Ubuntu 20.04 + gcc 9.4（兼容板卡 glibc 2.31）
- **大模型编译**: bmodel > 500MB 的网络（如 whisper turbo encoder 1.3G）`model_deploy.py` 必须加 `--disable_layer_group`，否则 v1.28.1 的 layer_group 优化会让推理时板卡 kernel panic 重启
- **大 ONNX 导出**: >2GB 模型导出需 ≥12GB 内存（WSL 默认仅 8GB 会 OOM），且要跳过 onnxsim、用 external data 另存以避开 protobuf 2GB 上限
