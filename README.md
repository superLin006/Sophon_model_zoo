# Sophon Model Zoo

Sophon BM1684X 平台深度学习模型移植工作区，基于 SDK-23.09 LTS SP4，TPU-MLIR v1.28.1。

**性能与验收数据的唯一正式来源是 [PERF_SUMMARY.md](PERF_SUMMARY.md)**（含精度、产物大小、数据集、统计口径、测试日期）。下表只作索引，指标列与 PERF_SUMMARY 不一致时以后者为准。

## 支持的模型

| 模型 | 类型 | 交付精度 | 指标摘要 | 状态 |
|------|------|------|------|------|
| [Whisper](whisper/README.md) | 语音识别（自回归，base / large-v3-turbo） | turbo **W4F16** | turbo W4F16 RTF 0.346（52 条均值）/ 0.343（单条 5.6s），中英逐字无损、省 65% 内存；**base 有产物但无板卡实测数据** | ✅ turbo 已验证；base 未上板 |
| [SenseVoice Small](sensevoice/README.md) | 语音识别 + 情感/事件（CTC） | **F16** | RTF 0.0095，~54ms（5.6s 音频），约 105× 实时；F16 与 F32 结果逐字一致 | ✅ 完成 |
| [Moonshine](moonshine/README.md) | 语音识别（轻量流式 ASR，streaming-small） | **F16** | RTF 0.045，296ms（6.6s 音频，27 decode 步）；10s 固定输入，超长按需切段 | ✅ 完成 |
| [Zipformer](zipformer/README.md) | 流式中英双语语音识别（Transducer） | **F16** | warm RTF 0.0410、首轮 0.0636–0.2264；8/8 token 序列与 Python Sail 完全一致 | ✅ 完成 |
| [ChatTTS](chatTTS/README.md) | 文本转语音（自回归 GPT + DVAE + Vocos） | GPT **INT4** + decoder/vocos **BF16** | RTF 0.533（非流式）/ 0.59（流式），TTFA ~980ms，70/70 RTF<1 | ✅ 完成 |
| [VITS-MeloTTS](vits-melo-tts-zh_en/README.md) | 文本转语音（中英双语，三段式） | **F16** | RTF ~0.12（Part A 6ms + MAS 8ms + Part C 305ms）；仅 2 条 smoke，未做批量回归 | ✅ 跑通（无批量回归） |
| [Eureka-Audio](Eureka-Audio/README.md) | 音频指令分类（whisper encoder + Qwen3-1.7B，端到端） | whisper **F16** + qwen3 **W4BF16** | 端到端 ~2.3s/条；准确率 **5-6/9**（ChatTTS 长指令集，与原版 PyTorch GPU 基线持平） | ⚠️ **bmodel 产物在，运行时资产与源权重已缺失，当前不可部署、不可复现** |
| [QwenLLM](QwenLLM/README.md) | LLM 意图识别（Qwen3-0.6B dispatch，v95 系列） | **W8BF16** / seq2048 | v95e-soup 为交付候选：prefill ~485ms、decode ~45 tok/s（2026-09-03 板卡） | ✅ 完成 |
| [Qwen3-ASR](Qwen3-ASR/README.md) | 语音识别 + 语种识别（30 语种 + 22 中文方言，LLM 类） | **W4F16 g64** 单文件 646MB（W8BF16 942MB 为精度基线） | 13/13 有效多语种音频；RTF 中位数 0.128（W8BF16 0.161）；decode 64–65 tok/s；流式 Final 与离线逐字一致 | ✅ 完成 |
| [Qwen3-TTS](Qwen3-TTS/README.md) | 文本转语音（12Hz 流式 codec，多音色/多语言，0.6B） | talker **W4F16 g64**（W8BF16 为保守回退）+ CP F32/cache F16 + codec F16 | 54 条 batch **54/54**，加权 RTF 1.818（W8BF16 1.900）；`en_03`、`zh_03` 两条已知音质异常 | ✅ 完成 |
| [HY-MT](HY-MT/README.md) | 机器翻译（HY-MT1.5-1.8B，中英/日中/中日等多语对） | **W8BF16** / seq512 | 61/61 用例通过，prefill ~205ms、decode 22.2–24.0 tok/s；W4BF16 g64 为速度档（34.41 tok/s，质量偏移明显） | ✅ 完成 |

各模型的量化档位对照、历史档位与已知限制见 [.claude/doc/sophon_bm1684_knowledge_base.md](.claude/doc/sophon_bm1684_knowledge_base.md) §2 与 [PERF_SUMMARY.md](PERF_SUMMARY.md) 的说明段。

## 项目结构

```
Sophon_model_zoo/
├── 0_Toolkits/               # Sophon SOC SDK + TPU-MLIR wheel（不入库，.gitkeep 占位）
├── 1_third_party/            # 第三方库头文件与静态库（大部分不入库；tokenizers-cpp 例外，见下）
├── 2_utils/                  # 公共 C 工具库（图像/音频处理）
├── 3_docker/
│   ├── Dockerfile.cross-build  # Ubuntu 20.04 + GCC 9.4 aarch64 交叉编译镜像
│   ├── Dockerfile.tpumlir      # TPU-MLIR 1.28.1 转换镜像
│   ├── build_tpumlir.sh        # 从 wheel 构建转换镜像
│   ├── run_docker.sh           # 启动/进入转换容器（默认名 sophon-tpumlir-v128）
│   └── README.md
├── 4_tools/check_repo_structure.py  # 仓库结构与交付约定检查（提交前运行）
├── environment.yml           # Conda 通用兜底模板（非精确，正式复现用各模型 requirements.txt）
├── PERF_SUMMARY.md           # 性能与验收数据唯一正式来源
├── AGENTS.md                 # 工具链、约定与踩坑经验（AI 协作入口）
├── TUTORIAL.md               # 使用指南（目录约定、两条转换路线、构建与部署流程）
├── whisper/                  # Whisper 移植（base + large-v3-turbo，F32/F16/W4F16）
├── sensevoice/               # SenseVoice Small 移植
├── moonshine/                # Moonshine streaming-small 移植（轻量流式 ASR，F32/F16）
├── zipformer/                # Zipformer 流式中英双语 ASR（103 帧窗口 / 96 帧步进，C++ BMRuntime CLI）
├── chatTTS/                  # ChatTTS 移植（纯 bmruntime C++，支持流式；Python/SAIL 支持参考音频克隆）
├── vits-melo-tts-zh_en/      # VITS-MeloTTS 中英双语移植（三段式）
├── Eureka-Audio/             # 音频指令分类（whisper encoder + Qwen3-1.7B，Python·sail + C++）
├── Qwen3-ASR/                # Qwen3-ASR-0.6B 语音识别 + 语种识别（C++ bmrt + sail，支持流式）
├── Qwen3-TTS/                # Qwen3-TTS-12Hz-0.6B 语音合成（纯 bmruntime C++，批量模式）
├── HY-MT/                    # HY-MT1.5-1.8B 机器翻译（W8BF16 交付 / W4BF16 速度档）
└── QwenLLM/                  # Qwen3-0.6B dispatch 意图识别（v95 系列）
    ├── LLM-TPU/              # sophgo 官方 demo（第三方源码，不入库，本地克隆）
    ├── compile/              # v95e → bmodel 编译脚本
    ├── scripts/              # 部署入口
    ├── cpp/                  # C++ BMRuntime 推理
    └── demo/                 # Python/sail benchmark 的 pybind demo 与配置
```

**入库约定**：bmodel、ONNX、模型权重与编译中间物全部不入 Git；小体积验收音频（35 个 wav / 13MB）与运行时必需的小资产（如 `moonshine/models/tokens.txt`）入库。`1_third_party/` 中 `tokenizers-cpp/` 的 2 个 `.a`（36MB）与头文件是**已入库的例外**——它们是 QwenLLM / Qwen3-ASR 交叉编译的硬依赖且无稳定上游预编译地址；其余第三方库靠 `.gitkeep` 占位、需本机自备。详见 [AGENTS.md](AGENTS.md) 的「入库实况」。

## 转换流程

```
PyTorch / Safetensors
    ↓  llm_convert.py  [Docker: sophon/tpuc_dev:v3.4-tpumlir-1.28.1]
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

每个模型目录**根目录的 `requirements.txt` 是唯一建环境入口**（新机器依此重建，不依赖现有环境）：

```bash
conda create -n <model_env> python=3.10 -y          # 具体版本以各模型 requirements 说明为准
conda run -n <model_env> python -m pip install --upgrade pip
conda run -n <model_env> python -m pip install -r <模型目录>/requirements.txt
# 导出 ONNX 一律用 conda run，不要裸 pip / conda activate
conda run -n <model_env> --no-capture-output python <模型目录>/python/export_onnx.py
```

> 根目录 `environment.yml` 仅是宽松的通用兜底模板，不能代替各模型 requirements.txt。
> Qwen3-ASR 用 Python 3.11；QwenLLM 的 Safetensors 转换在 TPU-MLIR 容器内完成，不需要本地 Python 转换环境；HY-MT 的原生 baseline 环境名见 `HY-MT/README.md`。

### 2. 启动 TPU-MLIR 容器

```bash
./3_docker/build_tpumlir.sh    # 首次：从 0_Toolkits/ 的 wheel 构建派生镜像
./3_docker/run_docker.sh       # 启动/进入容器，默认名 sophon-tpumlir-v128
```

若已有同名容器但镜像不符，脚本会拒绝复用并报错，处置方式见 [3_docker/README.md](3_docker/README.md)。

### 3. 交叉编译镜像

```bash
docker build -t sophon-cross-build -f 3_docker/Dockerfile.cross-build 3_docker
```

上下文用 `3_docker` 而非 `.`：该 Dockerfile 无 COPY，用 `.` 会把整个工作树（含数十 GB 模型产物）交给 docker daemon。

### 4. 移植 / 验证某个模型

参考各模型目录下的 `README.md`：

- [whisper/README.md](whisper/README.md)（base + turbo，F32/F16/W4F16）
- [sensevoice/README.md](sensevoice/README.md)
- [moonshine/README.md](moonshine/README.md)
- [zipformer/README.md](zipformer/README.md)（流式中英双语 Transducer）
- [chatTTS/README.md](chatTTS/README.md)（含 [cpp/README.md](chatTTS/cpp/README.md)、[python/README.md](chatTTS/python/README.md)）
- [vits-melo-tts-zh_en/README.md](vits-melo-tts-zh_en/README.md)
- [Eureka-Audio/README.md](Eureka-Audio/README.md)
- [Qwen3-ASR/README.md](Qwen3-ASR/README.md)（W8/W4BF16/W4F16 单文件与流式验证）
- [Qwen3-TTS/README.md](Qwen3-TTS/README.md)（W4F16/W8 A/B、3 bmodel 合并与批量合成）
- [HY-MT/README.md](HY-MT/README.md)（61 用例回归与 W8/W4BF16 对比）
- [QwenLLM/README.md](QwenLLM/README.md)

## 仓库结构检查

提交前运行只读检查，验证模型目录、产物位置、文档链接、脚本语法、本机路径与跨文档一致性：

```bash
python3 4_tools/check_repo_structure.py            # 只检查 git 跟踪的内容
python3 4_tools/check_repo_structure.py --disk     # 额外扫描磁盘上的产物目录（含被 gitignore 的中间物）
```

## 技术要点

- **芯片**：Sophon BM1684X SoC，SDK-23.09 LTS SP4，glibc **2.31**
- **转换工具**：TPU-MLIR v1.28.1（镜像 `sophon/tpuc_dev:v3.4-tpumlir-1.28.1`，由 `3_docker/build_tpumlir.sh` 从仓库内 wheel 构建）
- **LLM 转换**：`llm_convert.py` 支持 `w8bf16`、`w4bf16`、`w4f16` 等档位；W4 方案必须按模型逐一实测，不能仅凭转换成功判断质量
- **LLM 转换容器环境**：torch 2.4.1+cpu、transformers 4.57.6+、`huggingface-hub<1.0`；旧镜像中的 torch 2.1 / transformers 5.x 可能在 dtype 初始化阶段失败
- **交叉编译**：Ubuntu 20.04 + GCC 9.4，产物只引用到 `GLIBC_2.29`，在板卡 glibc 2.31 范围内
- **大模型编译**：单网络 bmodel > 500MB（如 whisper turbo encoder 1.3G）`model_deploy.py` 必须加 `--disable_layer_group`，否则 v1.28.1 的 layer_group 优化会让推理时板卡 kernel panic 重启；逐层拆分的小网络不用加；个别模型加了反而 SHA 校验失败——逐模型验证
- **大 ONNX 导出**：>2GB 模型导出要跳过 onnxsim、用 external data 另存以避开 protobuf 2GB 上限；内存建议 `.wslconfig` 配 `memory=14GB`（实机 `free -g` 约 13GB 可用，turbo encoder 全量导出已在此配置下跑通），不足时 OOM killer 报 Exit 137
- **生成式模型量化**：Qwen3-TTS talker 的 W4F16 已在 54 条板卡 batch 中验证总体稳定，但 `en_03`、`zh_03` 有已知音质异常；CP 组件保持 F32、cache 保持 F16；W8BF16 仍作为保守回退。Qwen3-ASR / HY-MT 的 W4F16 结论仅适用于各自已测模型和数据集，不能跨模型泛化
- **转换中间产物落盘**：`model_deploy.py` 的 npz/mlir/prototxt 等中间产物写入**当前工作目录**。所有编译脚本必须在隔离目录执行并在结束时清理（zipformer 的 `gen_bmodel.sh` 已内置 `mktemp` + `trap` 自动清理；Qwen3-TTS 的 `gen_final.sh` 固定写 `compile/tmp`）。否则一次编译会把 ~7.5G 中间物散落到仓库根，甚至泄漏到仓库外
- **产物目录会反复被污染**：`models/BM1684X/` 只应保留最终采用的 bmodel，但每轮编译都会生成 `*.bmodel.json`、`*.net_0.profile`、逐网络子目录等杂物，且这些文件被 gitignore、默认检查看不见。**每次跑完编译或端到端验证都要用 `--disk` 模式复查**
