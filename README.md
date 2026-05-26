# Sophon Model Zoo

Sophon BM1684X 平台深度学习模型移植工作区，基于 SDK-23.09 LTS SP4。

## 支持的模型

| 模型 | 类型 | 精度 | 指标 | 状态 |
|------|------|------|------|------|
| [Whisper Base](whisper/) | 语音识别（自回归） | FP16 | — | ✅ 完成 |
| [SenseVoice Small](sensevoice/) | 语音识别 + 情感/事件（CTC） | FP16 | RTF 0.0095 | ✅ 完成 |
| [ChatTTS](chatTTS/) | 文本转语音（自回归 + DVAE + Vocos） | GPT INT4 + FP16 | RTF 0.53（非流式）/ 0.59（流式），TTFA ~980ms | ✅ 完成 |
| [VITS-MeloTTS](vits-melo-tts-zh_en/) | 文本转语音（中英双语） | FP32 | RTF ~0.12 | ✅ 完成 |
| [Qwen2.5-3B](QwenLLM/) | LLM 意图识别 | W4BF16 (GPTQ) | FTL 1.49s，TPS 20.9，10/10 | ✅ 完成 |
| [Qwen3-1.7B](QwenLLM/) | LLM 意图识别（no_think） | W4BF16 (PTQ) | FTL 0.88s，TPS 29.1，E2E 1.20s，9/10 ⭐ | ✅ 完成 |
| [Qwen3.5-0.8B](QwenLLM/) | LLM 意图识别（GatedDeltaNet hybrid） | BF16 dynamic | 待测 | 🔄 已编译 |
| [Qwen3-0.6B](QwenLLM/) | LLM 意图识别（no_think） | W4BF16 (PTQ) | FTL 0.51s，TPS 52.6，E2E 0.92s，8/10 | ⚠️ 不稳定 |
| [Qwen3-4B](QwenLLM/) | LLM 意图识别（no_think） | W4F16 (AWQ) | FTL 2.05s，TPS 14.8，E2E 2.95s，10/10 | ✅ 完成 |

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
├── whisper/                  # Whisper Base 移植
├── sensevoice/               # SenseVoice Small 移植
├── chatTTS/                  # ChatTTS 移植（纯 bmruntime C++，支持流式）
├── vits-melo-tts-zh_en/      # VITS-MeloTTS 中英双语移植
└── QwenLLM/                  # Qwen 系列 LLM 意图识别
    ├── qwen2.5/              # Qwen2.5-3B-INT4（推荐，seq=2048）
    ├── qwen3/                # Qwen3-4B-W4F16（no_think 模式，seq=2048）
    ├── LLM-TPU/              # sophgo 官方 demo（Qwen2/Qwen2_5/Qwen3）
    ├── benchmark_intent.py   # 意图识别 benchmark 脚本
    └── BENCHMARK_RESULTS.md  # 详细性能对比
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
- [chatTTS/README.md](chatTTS/README.md)
- [vits-melo-tts-zh_en/README.md](vits-melo-tts-zh_en/README.md)
- [QwenLLM/BENCHMARK_RESULTS.md](QwenLLM/BENCHMARK_RESULTS.md)

## 技术要点

- **芯片**: Sophon BM1684X，SDK-23.09 LTS SP4
- **转换工具**: TPU-MLIR v1.28.1（`sophgo/tpuc_dev:latest`）
- **LLM 转换**: `llm_convert.py`，AWQ 量化只支持 `--quantize w4f16`
- **容器 transformers 版本**: 固定 4.51.1（`pip install transformers==4.51.1`），5.x 与 PyTorch 2.1.0+cpu 不兼容
- **交叉编译**: Ubuntu 20.04 + gcc 9.4（兼容板卡 glibc 2.31）
