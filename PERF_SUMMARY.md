# Sophon BM1684X 性能与验收汇总

> **平台**：Sophon BM1684X，SoC 模式（6GB DDR / 9070MB TPU DevMem），SophonSDK v23.09 LTS-SP4，TPU-MLIR v1.28.1
> **最后更新**：2026-09-03（各行的测试日期见「测试日期」列，本文件不代表单一时间点的快照）
> **本文件是仓库唯一的正式性能数据源。** 各模型 README 只保留逐条原始结果与已知限制，凡与本文件冲突以本文件为准；新增数据必须同时补齐下表要求的六项标注。

## 数据登记要求

新增或更新任何一条性能数据，必须同时记录：

1. 芯片型号与运行模式（BM1684X SoC / PCIe）；
2. SDK、TPU-MLIR 版本与模型精度档位；
3. 测试日期与测试数据集（用例数、音频总时长或语料构成）；
4. 模型加载是否计入；
5. RTF / FTL / Prefill / Decode 的统计口径（单条 / 多轮均值 / 中位数 / 区间；warm 还是首轮）；
6. 回归通过数量与已知异常样本。

**指标定义**

- **RTF**（Real-Time Factor）= 推理耗时 / 音频时长，越小越好，< 1 即实时。用于 ASR / TTS。
- **FTL**（First-Token Latency）= 从输入到吐出第一个 token 的耗时。**Prefill** = 处理输入 prompt 的吞吐（tok/s）。**Decode TPS** = 自回归逐字生成吞吐（tok/s）。用于 LLM。
- **RTF 统计口径**：只计特征提取 + TPU 推理，**不含模型加载**（实际部署时模型预加载常驻）。程序输出格式 `[Timing] audio=<x>ms feat=<x>ms infer=<x>ms total=<x>ms RTF=<x>`。
- **DevMem 占用**：TPU 设备内存峰值（总 9070MB），用 `bm-smi --noloop` 采样 `Memory-Usage` 列。标「实测」者为专项上板采样，其余引自各模型 README。
- **体积单位**：本文件统一按 **1MB = 1048576 字节（即 MiB）** 取整；标注 `GiB` 的按其字面值。仓库历史上存在两种口径混用（例如 Qwen3-TTS README 的 talker「1.14G」是十进制 GB = 1.06 GiB），引用他处数字时以本文件为准。

---

## 一、ASR（语音识别）

| 模型 | 精度 | 产物大小 | DevMem | 关键指标 | 数据集 / 用例 | 口径 | 测试日期 |
|---|---|---|---|---|---|---|---|
| **SenseVoice Small** ⭐ | F16 | 451MB | **444MB**（实测） | **RTF 0.0095**，~54ms | 单条 5.6s 中文 | 单条，特征 34ms + 推理 20ms | 2026-06-22 |
| SenseVoice Small | F32 | 893MB | — | RTF 0.034，~189ms | 同上 | 单条，特征 34ms + 推理 155ms；与 F16 结果逐字一致 | 2026-06-22 |
| **Whisper large-v3-turbo** ⭐ | W4F16 | 594MB（enc 369M + dec 222M） | **715MB**（实测） | **RTF 0.346** | 52 条中英混合校准集，639s，有效 44/52 | 多轮均值；enc 1474ms + dec 897ms；中文 0.264 / 英文 0.403；**转录与 F16 逐字一致（无损）** | 见 `whisper/README.md` |
| Whisper large-v3-turbo | F16 | 1.7GB（enc 1.3G + dec 460M） | enc 1.66G / dec 669M | RTF 0.353 | 同上，有效 50/52 | 多轮均值；enc 1372ms + dec 1228ms；中文 0.281 / 英文 0.419 | 见 `whisper/README.md` |
| Whisper large-v3-turbo | W4F16 | 594MB | 715MB（实测） | RTF 0.343，~2.4s | 单条 5.6s | 单条专项采样（与上表 52 条均值口径不同，不可直接比较） | 2026-06-22 |
| Whisper base | F16 | 201MB（enc 46M + dec 155M） | 未实测 | **板上无实测数据** | — | 见下方说明 | — |
| **Moonshine streaming-small** | F16 | 280MB（enc 109M + dec 171M） | — | **RTF 0.045**，296ms | `0.wav` 6.6s，27 decode 步 | 单条，feat 14.3ms + infer 282ms；`8k.wav` 4.8s → 206ms / RTF 0.043 | 见 `moonshine/README.md` |
| Moonshine streaming-small | F32 | 548MB（enc 220M + dec 352M） | — | RTF 0.099，653ms | 同上 | 单条，feat 14.4ms + infer 639ms；`8k.wav` → 457ms / RTF 0.095 | 见 `moonshine/README.md` |
| **Zipformer**（流式双语 Transducer） | F16 | 67MB（enc 44M + joiner 12M + dec 11M） | — | **warm RTF 0.0410**；首轮 RTF 0.0636–0.2264 | 8 条（6 条官方 WAV + 2 条中英补充），3.9–17.6s；warm 数据为 17.6s 样本各 5 次 | warm 与首轮分列；chunk P50/P95 = 35.98/37.71 ms；**8/8 token 序列与 Python Sail 完全一致** | 2026-09-03 |
| Zipformer | F32 | — | — | warm RTF 0.0513 | 同上 | chunk P50/P95 = 45.07/46.55 ms；F16 token 与 F32 ORT 完全一致 | 2026-09-03 |
| **Qwen3-ASR-0.6B** | W8BF16 | 942MB 单文件 | NPU heap 净占 ~1.1GB | **RTF 中位数 0.161**，decode 64–65 tok/s | 13 条有效多语种音频（中/英/日/德/法/西），13/13 | 受控多轮中位数；不取单次最低值 | 2026-08-18 |
| Qwen3-ASR-0.6B | W4F16 g64 | 646MB 单文件 | 同上 | **RTF 中位数 0.128**，decode 64–65 tok/s | 同上，13/13 | 同上；60 网络合并为单 bmodel | 2026-08-18 |
| Qwen3-ASR-0.6B | W4BF16 g64 | 646MB 单文件 | 同上 | RTF 未做受控多轮测量 | 13 条逐条结果见 `Qwen3-ASR/README.md` | 逐条端到端 0.27–2.28s（RTF 0.040–0.54，短音频偏高） | 2026-08-10 |

**说明与取舍**

- **Whisper base 无板卡数据**：`PERF_SUMMARY` 早期版本给出的「~1.01s（F16）/ ~1.86s（F32）」在仓库内**无任何可溯源的板卡测量记录**；仓库中唯一的 base 耗时证据是 `whisper/python/test/outputs/baseline/summary.json` 的 `elapsed_ms`（test_en 932.4ms、test_zh 763.5ms），那是**开发机 PyTorch CPU baseline，不是板卡 bmodel 数据**，且两者均值 848ms 也对不上 1.01s。base 的 bmodel 产物（F32/F16/W4F16 共 6 个）在仓库中存在，板卡端到端未做。需要 base 的板上指标时应实测后补录，不要沿用旧数字。
- **SenseVoice vs Whisper turbo**：SenseVoice RTF 0.0095（~105× 实时）、内存小，但只做识别 + 语种/情感/事件标签，无翻译，适合实时识别主力。Whisper turbo 多语言转录精度高，约 2.8×~4× 实时；W4F16 与 F16 端到端持平且逐字无损、省 65% 内存，**turbo 一律推荐 W4F16**。
- **Moonshine** 介于两者之间：轻量英文流式 ASR，10s 固定输入，F16 约 22× 实时、模型仅 280MB，适合内存受限场景的英文转写。超过 10s 需外部切段（`test_data/1.wav` 16.7s 超上限，未验证）。
- **Zipformer** 的 RTF 必须区分 warm 与首轮：首轮含三网首次运行与 BMRuntime 调度，显著高于 warm。历史上根 README 与知识库出现过的「RTF 0.024–0.071」**无测量出处**（2026-07-31 进入根 README 时即为裸结论，后被知识库转抄），已按上表实测值替换。
- **Qwen3-ASR** 三个档位的 RTF 口径不同，不可横向直接比较：W8BF16 / W4F16 是受控多轮中位数，W4BF16 只有逐条单次结果。旧版此处标注的「W4BF16 RTF 0.10」**无出处**，已删除。`long_60s.wav` / `long_60s_sil.wav`（57.6s / 58.6s）超出 encoder 30s 硬上限，**属预期失败，不计入回归**。

---

## 二、TTS（语音合成）

| 模型 | 精度 | 产物大小 | RTF（非流式） | RTF（流式） | TTFA | 数据集 / 用例 | 测试日期 |
|---|---|---|---|---|---|---|---|
| **ChatTTS** | GPT **INT4** + decoder/vocos **BF16** | 240MB（gpt 154M + dec 55M + vocos 31M） | **0.533** | 0.59 | ~980ms | 70 条（25 中短 + 25 英短 + 10 中长 + 10 英长），**70/70 RTF<1** | 见 `chatTTS/cpp/README.md` |
| **VITS-MeloTTS 中英双语** | **F16** | 99MB（part_a 19M + part_c1 48M + part_c2 33M） | **~0.12** | — | — | 2 条 smoke（中文 61 token / 中英混 53 token） | 见 `vits-melo-tts-zh_en/python/README.md:97` |
| **Qwen3-TTS-12Hz-0.6B** | talker **W8BF16** + CP F32（cache F16） + codec F16 | 2.56GB 三文件（talker 1.5G + cp 696M + codec 235M） | **加权 RTF 1.900** | — | — | 54 条 batch，**54/54 成功**，模型常驻，seed 42 | 2026-08-17 |
| Qwen3-TTS-12Hz-0.6B | talker **W4F16 g64** + CP/codec 同上 | 2.12GB 三文件（talker 1.14G） | **加权 RTF 1.818** | — | — | 同上，**54/54 成功**；talker 体积 -27.8%、三文件 -17.2% | 2026-08-17 |

**说明**

- **ChatTTS 精度是 BF16 不是 FP16**：decoder 与 vocos 均为 BF16（bmodel 编译脚本与 `chatTTS/README.md` 性能表一致）。根 README 旧版写作「GPT INT4 + FP16」，已更正。
- **ChatTTS 的 RTF 2.5 是 BM1688（SE9-16，SDK V1.7）旧数据，与本平台无关**，不得引用为 BM1684X 结果。
- **ChatTTS 参考音频音色克隆**（2026-09-02/03 板卡验证，详见 `chatTTS/python/README.md` §2.3）：

  | 指标 | 数值 | 口径 |
  |---|---|---|
  | ECAPA cosine：参考音频 ↔ 克隆输出 | **均值 0.637**（min 0.513 / max 0.741） | 8 组不同内置音色，`eval_speaker_sim.py` 在 x86 机器上打分 |
  | 对照：参考音频 ↔ 同一 `spk_emb` 但不克隆 | 0.409 | 同上 |
  | 对照：参考音频 ↔ 默认随机音色 | 0.218 | 同上 |
  | RTF（Python/SAIL 全链路，含 DVAE 编码参考音频） | 1.96 | 单条 `zh_short_1.wav`，参考编码 134 codes(~2.9s) |
  | RTF（C++ `--voice-prompt --ref-text`） | 0.57 | 同一 prompt，输出时长与 Python 侧一致 |
  | RTF（只算 infer/音频比） | 克隆 0.87–1.18 / 不克隆同音色 0.88–0.95 | 属抖动范围，克隆无结构性回退 |

  结论：**音频码 prompt 的音色传递能力明显强于 768 维 `spk_emb` 通道**（0.637 > 0.409 > 0.218）。三个 RTF 口径互不可比——1.96 含 DVAE 编码参考音频的一次性开销（约 0.6s / 3~4s 参考），0.57 是 C++ 路径，0.87–1.18 只计推理。
  
  **克隆条件是成对的**：参考音频的 DVAE 音频码（`spk_smp`）必须配该音频的逐字转写（`txt_smp`）。只给音频码时模型把参考语音当成"已说完"而提前 EOS——实测同一段参考，转写留空只出 0~0.6s 无声/半句音频，补上转写后正常出 4.2s。生产化时相似度阈值必须连同"参考音频需 ASR 转写"这一前置条件一起交付。
  
  **能力边界**：Python/SAIL 支持直接读取参考音频并编码 prompt；C++ 侧支持加载预计算音频 prompt（`--voice-prompt` + `--ref-text`）；sherpa-onnx 交付路径仍只支持固定音色向量 `spk_emb`。
- **VITS-MeloTTS 的 0.12 拆解**：Part A（TPU 6ms）→ MAS 对齐（CPU 8ms）→ Part C（TPU 305ms），合计 319ms。仓库内只有 F16 三件产物，无 F32 bmodel；根 README 旧版标注「FP32」已更正。该模型只跑过 2 条 smoke，**未做批量回归**，0.12 属单链路实测而非统计值。
- **Qwen3-TTS 的「54 条」= `test_outputs/batch_verify.txt`(14 行) + `test_outputs/batch_verify_54.txt`(40 行) 的并集**，两文件零重叠、合计去重正好 54 行。旧文档从未说明这一点，导致「54 条」看起来无法复现。
- **Qwen3-TTS 的 RTF 有多个历史值，必须带 SEQLEN 标注**：1.900 / 1.818 对应 **SEQLEN=192**（当前交付配置）；知识库与部署规范中出现过的 **2.51 / 2.85** 对应早期 **SEQLEN=128**，不是同一配置的测量，不可并列比较。
- **Qwen3-TTS 已知异常**：W4F16 的 `en_03`、`zh_03` 两条有明确音质问题（人工试听确认），保留为已知限制。W4F16 是当前体积/速度/质量的最佳候选，**W8BF16 作为保守回退**。
- **SEQLEN=192 意味着 prefill 上限约 15.4s 语音**；codec 128 帧上限对应最长 10.24s 输出 wav，超出属设计截断，不算失败。

---

## 三、LLM 意图识别（Qwen3-0.6B dispatch，no_think 模式）

> 场景：语音助手意图识别（ASR 文本 → action + params JSON）

| 版本 | 量化 | 产物大小 | 加载 | Prefill | Decode TPS | 用例 | 测试日期 |
|---|---|---|---|---|---|---|---|
| **v95e-soup**（正式交付候选） | **W8BF16** / seq2048 | 772MB | — | **~485ms** | **~45 tok/s** | 10 条中文意图用例 | 2026-09-03 |

**说明**

- 仓库当前**只有 W8BF16 产物**（`QwenLLM/models/` 下 4 个变体：`v95_normal_2e`、`v95c_soup25`、`v95d_soup`、`v95e_soup`，全部 w8bf16），**无任何 w4 档位产物**。根 README 旧版标注「W4BF16 / W8BF16」已更正。
- 本文件的早期版本曾列出 **Qwen3-1.7B（W4BF16，9/10，推荐）** 与 **Qwen3-4B（W4F16 AWQ，10/10）** 两行并给出推荐结论。**这两个模型的权重目录已从仓库移除，产物不存在**，相关行与推荐结论一并删除。历史结论保留在 `.claude/doc/sophon_bm1684_knowledge_base.md` 的量化证据矩阵中，标注为已失效档位。
- `v95-normal-2e`、`v95-r4`、`v95-recall`、`v95c-locked7k`、`v95c-soup25`、`v95d-soup` 保留用于质量对比，**部署脚本不会自动扫描它们**，也不作为交付目标。
- 生产运行必须 `--no_think`。benchmark 结果 JSON 写在板卡当前工作目录，**不提交仓库**。
- 旧版此处记载的 8/10–10/10 准确率对应**早期原始 Qwen3 模型**，不是 v95 系列；v95e 的准确率验收记录见 `QwenLLM/README.md`。

---

## 四、机器翻译

HY-MT 的每个数字都能追溯到已入库的原始板上日志（`HY-MT/outputs/`）。**引用时必须带上用例集与统计口径**——同一档位在不同用例集、不同统计量下会得到不同数值，这不是矛盾而是口径差异。

| 档位 | 产物大小 | 用例集 | Prefill | Decode TPS | 质量 / 回归 | 原始日志 |
|---|---|---|---|---|---|---|
| **W8BF16** ⭐（交付档） | 1.98GiB 单文件 | **61 项全量** | 中位 205.95ms，区间 204.6–210.4ms | **中位 23.37，区间 22.20–23.95** | 61 个用例全部执行完成；已知 3 处质量瑕疵（见下） | `outputs/board_extended/w8_full_61cases.log`（2026-08-14） |
| W8BF16 | 同上 | 61 项（另一轮） | 中位 206.04ms | 中位 **23.31**，均值 23.03，区间 19.09–23.84 | 同上 | `outputs/board_w4f16_20260817/hymt_w8_46.log`（2026-08-17） |
| W8BF16 | 同上 | **16 项短中长** | 均值 **205.83ms** | 均值 **23.06**，中位 22.99，区间 22.85–23.48 | 9/16，平均字符相似度 **0.964** | `outputs/board_extended/extended_w8bf16.log`（2026-08-12） |
| W4BF16 g64（速度档） | 1.26GiB | 61 项 | 中位 203.45ms | 中位 **34.35**，均值 34.37，区间 33.73–34.95 | 与 W4F16 输出 48/61 完全一致 | `outputs/board_w4f16_20260817/hymt_w4bf16_46.log` |
| W4BF16 g64 | 同上 | 16 项 | 均值 **203.45ms** | 均值 **34.41**，中位 34.38 | **3/16**，平均字符相似度 **0.815** | `outputs/board_extended/extended_w4g64.log` |
| W4BF16 g64 | 同上 | 3 用例 × 3 次重复 | 中位 203.59ms | 中位 34.25，区间 33.19–34.90 | 短/中/长各 3 次：均值 34.88 / 34.03 / 33.64，σ 0.04 / 0.74 / 0.55；输出确定一致 | `outputs/board_extended/perf_stability_w4g64.log` |
| W4F16 g64（实验档，**bmodel 产物未保留**） | 1.26GiB | 61 项 | 中位 **205.98ms**（均值 242.95，被单条异常拉高） | **中位 33.21**，均值 31.18，区间 **1.35**–34.83 | 61 个用例全部执行完成 | `outputs/board_w4f16_20260817/hymt_w4f16_46.log` |

**说明**

- **W8BF16 是唯一交付档**。W4 两档存在共同的质量问题（术语、格式标签 XML 嵌套、长文本偏移），16 项字符相似度 0.815 对 W8 的 0.964，仅作速度对照。
- **W8 的 22.2–24.0 tok/s 是 61 项的 min–max 区间**，23.06 是 16 项均值，23.31 与 23.37 分别是两轮 61 项的中位数。四者都对，不可互相替换。
- **W4BF16 的 34.41 与 34.35 不是笔误**：34.41 是 16 项**均值**（`extended_w4g64.log`），34.35 是 61 项**中位数**（`hymt_w4bf16_46.log`）。
- **W4F16 有一条严重离群**：`terminology_short` 单条 decode 仅 **1.35 tok/s**，是自身中位数 33.21 的约 1/25，把该档 decode 均值拉到 31.18、prefill 均值拉到 242.95ms（中位数仍为 205.98ms）。W8 最慢为 19.09、W4BF16 最慢为 33.73，均无同类异常。**这条异常此前未在任何文档中披露**，评估 W4F16 作为速度档时必须计入。
- **W4F16 比 W8 快约 42.5%**（33.21 / 23.31 = 1.425，均取 61 项中位数）。
- **W4F16 档位已无可部署 bmodel**：其编译输出落在 `HY-MT/compile/tmp/w4f16_seq512`，该目录已清空，本机亦不再保留 4GB 源 safetensors（`models/BM1684X/` 下只有 `w8bf16_seq512/` 与 `w4bf16_g64_seq512/`）。**板上验证日志已入库**，因此测量结论可查证，但产物复现需先取回源权重再执行 `HY-MT/compile/` 脚本。根 README 旧版把 W4F16 的 61/61 与 33.21 tok/s 作为 HY-MT 头条指标，已更正为 W8BF16。
- **「61/61」的准确含义是 61 个用例全部执行完成，不是 61 条通过质量判定**。回归脚本只在进程失败时打印 `[FAIL]`，日志本身不含逐条质量判定；16 项的 9/16、3/16 才是质量口径（字符相似度阈值判定）。
- **回归脚本汇总行曾少报用例数**：`test_data/board_regression_full.sh` 的收尾行硬编码 `ALL DONE (${REPEATS}x46 cases)`，而脚本实际定义 61 个用例，导致上述四份日志的汇总行全部写成 46。已改为动态计数并附带失败数（`ALL DONE (1x61 cases = 61 runs, 0 failed)`）。**已入库的历史日志仍是旧的 46 字样，读取时以用例块计数为准。**
- **已知 3 处质量瑕疵（W8）**：术语用例输出 "edge computing" 未用参考词 "edge inference"；格式用例 XML 标签嵌套错位；names_products 措辞非最优。
- KV cache 为 BF16（llm_convert 标准行为）。改成 FP16 仍是 16bit，不降低 cache 带宽或容量。

---

## 五、音频指令分类（端到端，音频直接 → 指令）

| 模型 | 精度 | 产物大小 | 端到端 | 准确率 | 状态 |
|---|---|---|---|---|---|
| **Eureka-Audio** | whisper enc **F16**（文件名保留 `_bf16` 后缀）+ Qwen3-1.7B **W4BF16** | ~4.1GB（enc 1.4G + qwen3 2.7G） | ~2.3s/条（whisper 0.64s + prefill 0.67s + decode 16.4 tok/s，约 16 token/句）；模型加载 ~8s（预热后） | **5-6/9**（ChatTTS 合成的口语化长指令集，与原版 PyTorch GPU 基线持平） | ⚠️ **bmodel 产物在，但运行时资产与源权重缺失，当前不可部署、不可复现** |

**说明**

- **准确率口径**：`~90%` 是**早期短命令词集**的结果，该测试集已不在仓库；当前仓库内的验收集是 `test_audios/intent/long_01~09.wav`（9 条 ChatTTS 合成长指令），实测 **5-6/9**，与原版 PyTorch(GPU) 持平——难点在合成音偏噪、口语化语义弱，**不是移植回归**。引用准确率必须带数据集。
- **不可复现的具体缺口**：运行时需要 `prefix_embeds.bin`、`suffix_embeds.bin`、`mel_filters.npz`、`tokenizer.json` 四个资产，以及 `Eureka-Audio-Instruct` 源权重目录；这些在本机均已不存在，`deploy_to_board.sh` 会在 `require_file` 处直接中止。`compile/gen_prefix_embeds.py` 可从源权重重新生成这四个资产，但前提是先取回权重。
- **已放弃的路线**：`llm_convert.py` 层融合重编 qwen3（w4bf16/w8bf16）可把 decode 提到 22-30 tok/s，但 audio-embed 注入场景语义有损，同音频准确率从 5-6/9 降到 **3/9**，已放弃，保留 `model_deploy` 通用编译版。
- **文件名陷阱**：`whisper_encoder_b1_bf16.bmodel` 实际是 **F16** 量化（BF16 对 attention 精度损失过大，cosine 仅 0.51；F16 为 0.99）。文件名保留 `_bf16` 后缀未改，因为改名会牵动 deploy 脚本与 C++ 默认参数。**不要按文件名判断精度。**
- 与「ASR + 文本 LLM」两段式不同，本模型是 whisper encoder 直连 Qwen3-1.7B 的端到端音频意图模型。

---

## 六、数据口径与注意点

1. **专项上板采样项**（2026-06-22）：SenseVoice F16 与 Whisper turbo W4F16 的 DevMem 峰值与单条 RTF，用 `bm-smi --noloop` 500ms 间隔采样 `Memory-Usage` 列（总 9070MB）。其余数值引自各模型 README 的实测记录，日期见各行。
2. **「—」/「未实测」**：该项无可靠记录，不得凭经验填写。Whisper base 板上未部署故无内存与 RTF；VITS 未做批量回归故无通过率。
3. **内存口径**：LLM 用 DevMem（TPU 设备内存，总 9070MB）。Whisper F16 标注的是 encoder/decoder 各自 device 峰值，turbo W4F16 的 715MB 为整进程合计峰值。Qwen3-ASR 的 ~1.1GB 是 NPU heap 净占用（646MB bmodel + 运行时 KV/激活）；`bm_get_stat` 报的 9070MB 才是 bmrt 实际可用上限，ion debug 接口报的 npu heap 3.86G 是单分区，会误导。
4. **热降频**：BM1684X 连续推理后降频明显，模型越大影响越大。部署需考虑散热或推理间隔。这是早期 Qwen3-4B 端到端从 2.95s 升至 6.49s、decode 从 14.8 降至 5.8 tok/s 的直接原因。
5. **原始日志的归档状态**：
   - HY-MT：`outputs/board_extended/`（61 用例、16 项扩展、稳定性）与 `outputs/board_w4f16_20260817/`（W8 / W4BF16 / W4F16 三档同轮对照）——**本文件第四节每个数字都可从这些日志复算**
   - Qwen3-ASR：`test_outputs/`（3 份全量）与 `test_outputs/bench_20260817/`（W8 / W4F16 各 4 轮）
   - Qwen3-TTS：`test_outputs/final_20260814/README.md` 是 **14 条逐条 RTF 表**（W8BF16 SEQLEN=192，均值 ~1.9，区间 1.785–2.380），已入库

   HY-MT 与 Qwen3-ASR 的日志此前被各自 `.gitignore` 以 `*.log` 或整目录规则排除，造成"文档引用了日志、归档里没有日志"的死引用；现已反向选回（合成 wav 仍排除，体积大）。

   **两处遗留证据缺口**：
   - **Qwen3-TTS 的 54 条加权 RTF（W8BF16 1.900 / W4F16 1.818）在全仓没有逐条原始数据**，只以汇总数字形式存在于 `Qwen3-TTS/README.md` 与知识库。`test_outputs/ab_20260817/` 下的 4 份 log **不是 RTF 记录**，而是用 Qwen3-ASR 转写 Qwen3-TTS 合成音频的结果（内容为 `[ASR] ... language Chinese<asr_text>`，文件名为 `asr_w4_batch` / `asr_w8_batch` / `asr_w8_extra_{w4,w8}audio`），属**合成音频可懂度的间接证据**，交叉了 TTS 档位 × ASR 档位；同目录的合成 wav 因体积被排除。因此 54/54 与加权 RTF 目前**不可独立复算**，只能重跑 batch 复现。
   - **Qwen3-ASR 的 benchmark 日志每行只有 `[ASR] N tokens, total Xs (...)`，不含音频文件名**，无法单凭日志把每条结果对应到具体用例，RTF 中位数同样不可从日志独立复算。

   两者均已登记为待修项（见文末「后续修 bug 候选」）。
6. **验收用例的可复现性**：Zipformer 的 8/8 验收依赖 6 条官方 WAV，它们位于 gitignore 的 `zipformer/assets/`；仓库内 `test_data/golden.json` 只固定了其中 2 条（test_en / test_zh）的 token 与文本 golden。因此 8/8 中只有 2/8 可从归档直接复现，其余需按 `zipformer/README.md` 的资产获取步骤重建。
