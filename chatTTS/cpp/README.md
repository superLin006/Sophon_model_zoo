# ChatTTS BM1684X C++ 推理

ChatTTS 在 SOPHON BM1684X 上的纯 C++ 推理实现，完全基于 bmruntime C API，不依赖 sophon-sail 或 Python 运行时。

## 目录结构

```
cpp/
├── CMakeLists.txt          # 构建脚本，产出 chattts 和 chattts_bench
├── cmake/
│   └── aarch64.cmake       # 交叉编译工具链
├── include/                # 头文件
├── src/                    # 源码
│   ├── main.cpp            # 推理入口（单句）
│   ├── benchmark.cpp       # 70样本 benchmark
│   ├── chattts.cpp         # 顶层 TTS 接口
│   ├── gpt_engine.cpp      # GPT 推理引擎（含 KV cache 管理）
│   ├── decoder_engine.cpp  # DVAE Decoder
│   ├── vocos_engine.cpp    # Vocos 声码器
│   ├── istft.cpp           # iSTFT（CPU，fftw3）
│   ├── tokenizer.cpp       # BERT tokenizer + ChatTTS 扩展词表
│   └── normalizer.cpp      # 中文文本归一化
├── assets/
│   ├── homophones_map.json # 同音字归一化表
│   └── tokenizer/          # vocab.txt 等
└── ISSUES.md               # 移植过程中遇到的问题记录
```

## 环境依赖

| 组件 | 版本 |
|------|------|
| SOPHON SDK | 23.09 LTS SP4 |
| 交叉编译工具链 | aarch64-linux-gnu-g++ 9.4（Ubuntu 20.04）|
| FFTW3 | aarch64 静态库（`1_third_party/fftw/`）|
| CMake | ≥ 3.16 |

## 编译

在 WSL/Linux 上用统一脚本交叉编译，不需要上板：

```bash
# 在仓库根目录执行
bash chatTTS/cpp/build.sh
```

产出：
- `chatTTS/cpp/build/chattts` — 单句推理
- `chatTTS/cpp/build/chattts_bench` — 70 样本 benchmark

## 板卡上的模型文件

运行前需将以下文件准备好（路径可通过参数指定）：

```
/data/chattts/models/
├── chattts-llama_int4_1dev_1024_bm1684x.bmodel   # GPT 模型（INT4 量化）
├── decoder_1-768-1024_bm1684x.bmodel             # DVAE Decoder
├── vocos_1-100-2048_bm1684x.bmodel               # Vocos 声码器
└── asset/
    ├── homophones_map.json
    └── tokenizer/
        └── vocab.txt
```

 speaker embedding 二进制文件（float32，768维），可从 `slct_voice_240605.json` 中提取。

参考音频克隆采用 Python 端 DVAE 编码、C++ 端 GPT prompt 注入：先用 Python 生成
`CTTSPRM1` prompt 文件，再由 C++ 加载。这样无需在板卡 C++ 侧引入 torchaudio 或
FSQ 依赖。

## 使用方法

### 单句推理

```bash
./chattts \
  --model-dir /data/chattts/models \
  --spk-emb   /path/to/spk_emb.bin \
  --text      "你好，很高兴认识你。" \
  --output    output.wav
```

### 参考音频音色克隆

先在安装了 Python/SAIL 的环境中导出 prompt（会同时执行一次合成）：

```bash
cd python
python3 test_clone.py \
  --ref ../test_data/zh_short_1.wav \
  --ref-text "参考音频的逐字转写" \
  --prompt-out /tmp/voice_prompt.bin \
  --output /tmp/clone.wav
```

再在 C++ 程序中加载同一个 prompt，并必须传入完全一致的参考音频转写：

```bash
./chattts \
  --model-dir ../models \
  --voice-prompt /tmp/voice_prompt.bin \
  --ref-text "参考音频的逐字转写" \
  --text "这是使用参考音色生成的新句子。" \
  --output clone_cpp.wav
```

`--voice-prompt` 与 `--spk-emb` 不能同时使用。prompt 文件是二进制小文件，包含
`[frames, 4]` 的 ChatTTS DVAE code；参考文本不会写入文件，调用时必须另传。

完整参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-dir` | `../models` | bmodel 和 asset 所在目录 |
| `--text` | 内置默认句 | 要合成的文本 |
| `--output` | `output.wav` | 输出 WAV 文件路径 |
| `--spk-emb` | （无，使用随机音色）| speaker embedding 二进制路径 |
| `--voice-prompt` | （无） | Python `--prompt-out` 导出的 DVAE prompt 文件 |
| `--ref-text` | （无） | 参考音频逐字转写；与 `--voice-prompt` 配套 |
| `--speed` | `5` | 语速，1~9 |
| `--temp` | `0.0001` | 采样温度 |
| `--max-tokens` | `2048` | 最大生成 token 数 |
| `--tpu-id` | `0` | TPU 设备编号 |

### Benchmark（70样本）

```bash
./chattts_bench \
  --model-dir /data/chattts/models \
  --spk-emb   /path/to/spk_emb.bin \
  --warmup    3
```

与 `python/benchmark.py` 使用完全相同的 70 个测试样本（25 中文短句 + 25 英文短句 + 10 中文长文 + 10 英文长文），输出相同格式的统计报告。

## 性能

在 BM1684X 上的实测结果（GPT 模型 INT4 量化）：

| 分组 | 样本数 | 平均 RTF | RTF < 1 |
|------|:------:|:--------:|:-------:|
| 中文短句 | 25 | 0.606 | 25/25 |
| 中文长文 | 10 | 0.497 | 10/10 |
| 英文短句 | 25 | 0.623 | 25/25 |
| 英文长文 | 10 | 0.497 | 10/10 |
| **整体** | **70** | **0.533** | **70/70** |

> RTF < 1 表示实时，RTF 越低越好。Python 参考版本整体 RTF ~0.65。

## 提取 Speaker Embedding

从 `slct_voice_240605.json` 中提取某个音色并存为二进制：

```python
import json, struct
with open("slct_voice_240605.json") as f:
    data = json.load(f)
tensor = data["6"]["tensor"]   # 选择 key "6" 的音色
with open("spk_emb.bin", "wb") as f:
    f.write(struct.pack(f"{len(tensor)}f", *tensor))
```
