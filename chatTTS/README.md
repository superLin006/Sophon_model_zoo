# ChatTTS — BM1684X 移植

## 目录

- [1. 简介](#1-简介)
- [2. 特性与验证范围](#2-特性与验证范围)
- [3. 准备模型](#3-准备模型)
- [4. 例程测试](#4-例程测试)
- [5. 部署到 BM1684X 板卡](#5-部署到-bm1684x-板卡)
- [6. 性能（BM1684X 实测）](#6-性能bm1684x-实测)

## 1. 简介

ChatTTS 是一款专门为对话场景（例如 LLM 助手）设计的文本转语音模型。本例程参考了 [ChatTTS-ONNX](https://github.com/ZillaRU/ChatTTS-ONNX)，对 [ChatTTS 官方仓库](https://github.com/2noise/ChatTTS) 中的算法进行移植，在 **Sophon BM1684X** 上完成推理与验收。

三条实现路径，能力不同：

| 路径 | 位置 | 推理后端 | 参考音频克隆 |
|---|---|---|---|
| Python / SAIL | `python/ChatTTS/` | `sophon.sail` + PyTorch(CPU) DVAE | ✅ 支持直接喂 wav |
| C++（纯 bmruntime） | `cpp/` | libbmrt，无第三方推理框架 | ⚠️ 仅支持加载**预计算**的音频 prompt |
| sherpa-onnx SDK | 仓库外交付路径 | sherpa-onnx | ❌ 仅固定音色向量 `spk_emb` |

## 2. 特性与验证范围

- **目标平台：BM1684X（SoC）**。本仓库的全部实测数据都在 BM1684X 上取得；上游例程宣称的 BM1688 / x86 PCIe 支持不在本仓验证范围内（见 §6 附录）
- **已验证的量化组合：GPT INT4 + decoder BF16 + vocos BF16**。上游提到的 INT8 编译在本仓**未做验证**，不要当作可用能力引用
- 支持流式输出（首字延迟约 1s）
- **参考音频音色克隆**（2026-09 板卡验证）：8 组内置音色，ECAPA cosine 均值 **0.637**（min 0.513 / max 0.741），显著高于「同一 `spk_emb` 但不克隆」的 0.409 与默认随机音色的 0.218。详见 [python/README.md §2.3](python/README.md) 与 [`PERF_SUMMARY.md` 第二节](../PERF_SUMMARY.md)

  > ⚠️ 克隆条件是**成对**的：参考音频的 DVAE 音频码必须配该音频的**逐字转写**。只给音频码时模型会把参考语音当成"已说完"而提前 EOS（实测转写留空只出 0~0.6s 无声/半句，补上转写后正常出 4.2s）。

## 3. 准备模型

### 3.1 使用提供的模型

`scripts/` 下提供了下载脚本 `download.sh`（按芯片分目录）。运行前保证存储空间大于 3GB：

```bash
chmod -R +x scripts/
./scripts/download.sh
```

执行后目录结构如下：

```text
chatTTS/
├── README.md                        # 本例程指南
├── cpp/                             # C++ 推理（纯 bmruntime，支持流式）
├── docs/
│   ├── ChatTTS_Export_Guide.md      # ONNX 导出和 bmodel 编译指南
│   └── flowchart.png                # 流程图
├── models/                          # 模型产物（不入库，download.sh 下载）
│   ├── asset/                       # 不需编译成 bmodel 的权重（DVAE / tokenizer）
│   └── BM1684X/
│       ├── chattts-llama_int4_1dev_1024_bm1684x.bmodel   # gpt int4, seq=1024
│       ├── decoder_1-768-1024_bm1684x.bmodel             # decoder bf16 [1,768,1024]
│       └── vocos_1-100-2048_bm1684x.bmodel               # vocos bf16 [1,100,2048]
├── python/                          # Python 推理（sophon.sail）
│   ├── ChatTTS/                     # 封装好的 ChatTTS 模块
│   ├── README.md                    # 运行指南（含 §2.3 参考音频克隆）
│   ├── test.py                      # 非流式调用示例
│   ├── test_stream.py               # 流式调用示例
│   └── test_clone.py                # 参考音频克隆示例
├── scripts/                         # 下载 + bmodel 编译脚本
│   ├── download.sh
│   ├── gen_gpt_bmodel.sh
│   ├── gen_decoder_bmodel.sh
│   └── gen_vocos_bmodel.sh
├── tools/                           # ONNX 导出（模型结构 + exporter）
│   ├── config.py / dvae.py / gpt.py / modeling_llama.py
│   └── exporter.py
└── test_data/                       # 测试音频（不入库，.gitkeep 占位）
```

三个 bmodel 合计 240MB（gpt 154M + decoder 55M + vocos 31M）。

### 3.2 自行编译模型

见 [docs/ChatTTS_Export_Guide.md](docs/ChatTTS_Export_Guide.md)。

> ⚠️ **ONNX 无法从本仓库单独复现**：重导出需要 ChatTTS 的 GPT / Decoder / Vocos checkpoint（HF `2Noise/ChatTTS`）。本机当前没有这些权重（`python/.pretrained_models/*.ckpt` 是 143 字节占位文件），因此**只有 bmodel 产物在、上游权重不在**时无法重跑导出。三个 bmodel 本身可长期使用，不受影响。

## 4. 例程测试

- **Python / SAIL 例程**：[python/README.md](python/README.md) —— 非流式、流式、参考音频克隆、70 样本 benchmark
- **C++ 例程**：[cpp/README.md](cpp/README.md) —— 命令行参数、流式、批量合成、`chattts_bench` 性能测试、speaker embedding 提取
- **已知问题与优化历程**：[cpp/ISSUES.md](cpp/ISSUES.md)

板卡 Python 侧 sail 的导入方式是 `import sophon.sail as sail`（板上由 `sophon-arm-pcie` 提供），裸 `import sail` 会失败。

## 5. 部署到 BM1684X 板卡

先构建 C++ 程序，再用部署入口上传三个 bmodel 和 `models/asset/` 运行时资产。默认使用 SSH key，板卡信息通过环境变量提供：

```bash
bash chatTTS/cpp/build.sh
BOARD_IP=<board_ip> bash chatTTS/deploy_to_board.sh --test
```

可选环境变量：`BOARD_USER`、`BOARD_PORT`、`BOARD_DIR`（默认 `/data/chattts`）、`MODEL_DIR`、`BINARY`、`BOARD_PASS`。上传后脚本逐文件执行 md5 校验（scp 大文件会静默损坏，见 `AGENTS.md` 关键约束）。

`--test` 使用默认中文文本执行一次最小 smoke test：

```bash
./chattts --model-dir models --text '你好，这是 ChatTTS 的板卡 smoke test。' --output chattts_smoke.wav
```

也可以通过 `SPK_EMB=/path/to/spk_emb.bin` 上传指定 speaker embedding。**仓库不提供现成的 `spk_emb.bin`**，需从 `python/slct_voice_240605.json` 提取——步骤见 [cpp/README.md 的「提取 Speaker Embedding」](cpp/README.md)。不传 `--spk-emb` 时使用随机音色。

板端手动运行：

```bash
./chattts --model-dir models --spk-emb spk_emb.bin \
  --text "你好，很高兴认识你。" --output output.wav
# 加载预计算音频 prompt（C++ 侧的克隆方式）：
./chattts --model-dir models --voice-prompt voice_prompt.bin \
  --ref-text "参考音频的逐字转写" --text "目标文本" --output clone.wav
```

## 6. 性能（BM1684X 实测）

权威数据见 [`PERF_SUMMARY.md` 第二节](../PERF_SUMMARY.md)。以下为 70 样本 benchmark（`chattts_bench`，与 `python/benchmark.py` 同一套样本）的分组结果，配置为 GPT INT4 + decoder/vocos BF16：

| 分组 | 样本数 | C++ 平均 RTF | Python 参考 RTF | RTF < 1 |
|------|:------:|:--------:|:--------:|:-------:|
| 中文短句 | 25 | 0.606 | ~0.75 | 25/25 |
| 中文长文 | 10 | 0.497 | ~0.50 | 10/10 |
| 英文短句 | 25 | 0.623 | ~0.75 | 25/25 |
| 英文长文 | 10 | 0.497 | ~0.50 | 10/10 |
| **整体** | **70** | **0.533** | **~0.65** | **70/70** |

- **70/70 全部 RTF < 1（实时）**，整体平均 RTF 0.533
- 流式模式 RTF 0.59，首字延迟 TTFA ~980ms
- RTF 只计推理，不含模型加载

优化历程（KV cache 设备常驻 + decode 缓冲预分配）与逐条数据见 [cpp/ISSUES.md](cpp/ISSUES.md)。

### 附录：其它平台的历史数据（非本仓验证范围）

上游例程曾在 **SE9-16（BM1688，SDK V1.7）** 上测得 `test.py` + gpt(int4)/decoder(bf16)/vocos(bf16) 的 RTF 为 **2.5**，TPU 利用率 15%~30%、CPU 利用率 100%~150%。

> ⚠️ **该 2.5 不是 BM1684X 的数据，不得引用为本平台结果。** 本仓早期文档曾把它与 BM1684X 的 0.533 混在一起，`PERF_SUMMARY.md` 也专门为此加过警告。保留在此仅为说明数据来源，避免后人再次误引。
