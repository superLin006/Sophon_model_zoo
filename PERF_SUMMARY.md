# Sophon BM1684X 已跑通模型性能汇总

> **平台**：Sophon BM1684X，SoC 模式（6GB DDR / 9070MB TPU DevMem），SophonSDK v23.09 LTS-SP4
> **板卡**：172.16.25.169（root/1）
> **更新日期**：2026-06-22
>
> **指标说明**
> - **RTF**（Real-Time Factor）= 推理耗时 / 音频时长，越小越好，< 1 即实时。**针对 ASR / TTS**。
> - **首字延迟 / token 生成速度（Prefill / Decode）**：**针对 LLM**。
>   - FTL（First-Token Latency，首字延迟）：从输入到吐出第一个 token 的耗时。
>   - Prefill：处理输入 prompt 的吞吐（tok/s）。
>   - Decode TPS：自回归逐字生成的吞吐（tok/s）。
> - **DevMem 占用**：TPU 设备内存峰值（总 9070MB），用 `bm-smi` 实测；标「实测」者为本次上板采样，其余引用各模型 README 实测值。

---

## 一、ASR（语音识别）

| 模型 | 精度 | 模型大小 | DevMem 占用 | RTF | 端到端耗时 | 备注 |
|------|------|---------|-------------|-----|-----------|------|
| **SenseVoice Small** ⭐ | F16 | 451MB | **444MB**（实测） | **0.0094**（实测） | **~53ms**（5.6s音频） | 非自回归 CTC，实时率 ~105×；带语种/情感/事件标签 |
| SenseVoice Small | F32 | 893MB | — | 0.034 | ~189ms | 与 F16 结果完全一致 |
| **Whisper large-v3-turbo** ⭐ | W4F16 | 594MB（enc 369M + dec 222M） | **715MB**（实测） | **0.343**（实测） | ~2.4s（5.6s音频） | 省 65% 内存、逐字无损、**推荐** |
| Whisper large-v3-turbo | F16 | 1.7GB（enc 1.3G + dec 460M） | enc 峰值 1.66G / dec 669M | 0.353 | ~2.6s | 中文 RTF 0.281 / 英文 0.419 |
| Whisper base | F16 | ~201MB（enc 46M + dec 155M） | 未实测（板上未部署） | — | ~1.01s（5.8s音频） | 自回归；F32 ~1.86s |
| **Moonshine streaming-small** | F16 | 280MB（enc 109M + dec 171M） | — | **0.045**（实测） | **~296ms**（6.6s音频） | 轻量英文流式 ASR，10s 固定输入；F32 RTF 0.099 / 653ms |
| **Qwen3-ASR-0.6B** | F16 + W4BF16 | 1.5GB | 1838MB | **0.15**（实测） | **0.85s**（5.6s音频） | LLM 类 ASR（30语种+22方言+语种识别），≤8s 音频，2026-08-08 实测 |

**SenseVoice 与 Whisper 取舍**
- SenseVoice：RTF 0.0094 速度碾压、内存小，但仅做识别 + 语种/情感/事件，无翻译。适合实时识别主力。
- Whisper turbo：多语言转录精度高，RTF 约 2.8×~4× 实时；W4F16 与 F16 端到端持平且**逐字无损**，turbo 一律推荐 W4F16。
- Moonshine streaming-small：轻量英文流式 ASR（10s 固定输入），F16 RTF 0.045（~22×实时），介于 SenseVoice 与 Whisper 之间，模型小（280MB）、适合内存受限场景的英文转写。

---

## 二、TTS（语音合成）

| 模型 | 精度 | 模型大小 | RTF（非流式） | RTF（流式） | 首字延迟 TTFA | 备注 |
|------|------|---------|--------------|------------|--------------|------|
| **ChatTTS** | GPT-INT4 + Decoder/Vocos-BF16 | ~242MB（gpt 154M + dec 56M + vocos 32M） | **0.53** | **0.59** | **~980ms** | 纯 bmruntime C++，支持流式，70/70 样本验证 |

> ChatTTS 对话音色自然，支持流式输出；首字延迟约 1s，RTF < 1 满足实时。
> （VITS-MeloTTS 已跑通但本次不纳入统计。）

---

## 三、LLM 意图识别（Qwen 系列，no_think 模式）

> 场景：语音助手意图识别（ASR 文本 → action + params JSON）

| 模型 | 量化 | 模型大小 | DevMem 占用 | 加载耗时 | 首字延迟 FTL | Prefill | Decode TPS | 端到端 | 准确率 |
|------|------|---------|-------------|---------|-------------|---------|-----------|--------|--------|
| **Qwen3-1.7B** ⭐ | W4BF16 | 1.4GB | 1731MB | 2.36s | **0.878s** | **82.6 tok/s** | **29.1 tok/s** | **1.20s** | 9/10 |
| Qwen3-0.6B | W4BF16 | 562MB | 871MB | 1.19s | 0.511s | 142.0 tok/s | 52.6 tok/s | 0.92s | 8/10 |
| Qwen3-4B | W4F16(AWQ) | 2.6GB | 3132MB | 57.5s(冷) | 2.05s | — | 14.8 tok/s | 2.95s | 10/10 |

**结论：推荐 Qwen3-1.7B**
- 内存适中（1.7GB）、加载快（2.36s）、端到端 1.2s、准确率 9/10（唯一失败 case 可用系统提示词修复 → 10/10）。
- Qwen3-0.6B 最快但准确率不足，且输出包裹 markdown 代码块，JSON 解析需额外剥离。
- Qwen3-4B 准确率满分，但占用大（2.6G/3.1G），**连续推理热降频严重**（E2E 从 2.95s 升至 6.49s，Decode 从 14.8 降至 5.8 tok/s）。

---

## 四、音频指令分类（端到端，音频直接 → 指令）

| 模型 | 精度 | 模型大小 | 端到端耗时 | 准确率 | 备注 |
|------|------|---------|-----------|--------|------|
| **Eureka-Audio** | W4BF16 | ~4.1GB（whisper enc 1.4G + Qwen3-1.7B 2.7G） | ~2.3s/条 | ~90% | 音频直接→指令分类；Python·sail 与 C++ 均跑通 |

> 与「ASR + 文本 LLM」两段式不同，Eureka-Audio 是 whisper encoder 接 Qwen3-1.7B 的端到端音频意图模型。

---

## 数据口径与注意点

1. **本次实测项**（2026-06-22，本机上板采样）：SenseVoice F16 与 Whisper turbo W4F16 的 DevMem 峰值与 RTF，用 `bm-smi --noloop` 500ms 间隔采样 `Memory-Usage` 列（总 9070MB）。其余数值引用各模型 README 实测记录。
2. **「—」/「未实测」**：该项无可靠记录。Whisper base 板上未部署故未测内存；SenseVoice F32、Whisper F16 内存引用 README。
3. **内存口径**：LLM 用 DevMem（TPU 设备内存，总 9070MB）；Whisper F16 标注的是 encoder/decoder 各自 device 峰值，turbo W4F16 的 715MB 为整进程合计峰值。
4. **ChatTTS RTF**：本表 0.53/0.59 为 **BM1684X 实测**；chatTTS 子目录 README 中的 RTF 2.5 是 BM1688(SE9-16) 旧数据，勿混淆。
5. **热降频**：BM1684X 连续推理后降频明显，模型越大影响越大（Qwen3-4B 最严重），部署需考虑散热或推理间隔。
