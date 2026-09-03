# Baseline 测试结果

## 模型信息
- 模型名称: vits-melo-tts-zh_en
- 模型路径: models/onnx/vits-melo-tts-zh_en/model.onnx
- 模型类型: VITS-MeloTTS TTS（端到端 text-to-speech，中英混合单说话人）
- 模型来源: sherpa-onnx 预转换 ONNX（无需从 PyTorch 导出）
- 输入数量: 7
  - x: ['N', 'L'] int64（音素 token ids，N=batch, L=序列长度）
  - x_lengths: ['N'] int64（序列长度，值等于 L）
  - tones: ['N', 'L'] int64（声调 ids）
  - sid: [1] int64（固定值 1，从模型 metadata 读取）
  - noise_scale: [1] float32（默认 0.667）
  - length_scale: [1] float32（默认 1.0，1/speed）
  - noise_scale_w: [1] float32（默认 0.8）
- 输出: y，shape ['N', 1, 'T'] float32，采样率 44100Hz
- 模型大小: 163MB（model.onnx）

## 模型 Metadata（onnxruntime 读取）
```
model_type: melo-vits
version: 2
language: Chinese + English
sample_rate: 44100
add_blank: 1
n_speakers: 1
speaker_id: 1
bert_dim: 1024
ja_bert_dim: 768
lang_id: 3
tone_start: 0
comment: melo
```

## lexicon.txt 格式
每行格式：`word phone1 phone2 ... phoneN tone1 tone2 ... toneN`
说明：前 N 列为音素，后 N 列为声调（total_cols - 1 必须为偶数）。

示例：
```
今 j in 1 1              # 2 phones, 2 tones
公 g ong 1 1
天 t ian 1 1
world w er l d 7 9 7 7   # 英文，4 phones, 4 tones（声调≥7 为英文区）
test t eh s t 7 9 7 7
```
- 总词条数: 195830
- 文件大小: 6.6MB
- 英文词条声调取值范围: 7~10（language_tone_start=0，英文 tone offset 为 7）
- 中文词条声调取值范围: 1~5（普通话 5 声调）

## tokens.txt 格式
每行格式：`token id`，共 112 个 token（id 0~111）

关键 token：
- 0: _ (blank/space)
- 103: !
- 104: ?
- 106: ,
- 107: .
- 110: SP
- 111: UNK

## 测试数据
- test_zh: "今天天气真好，我们去公园散步吧。"
- test_en_zh: "你好world，这是一个test句子。"

## Baseline 结果（onnxruntime 推理，add_blank=1 已修复）
| 测试用例 | Token 数（含 blank） | 输出 Shape | 样本数 | 时长 | WAV 文件 |
|---------|---------|-----------|-------|------|---------|
| test_zh | 61 | [1, 1, 118784] | 118784 | 2.694s | outputs/baseline/test_zh.wav |
| test_en_zh | 53 | [1, 1, 114176] | 114176 | 2.589s | outputs/baseline/test_en_zh.wav |

### add_blank 说明
模型 metadata add_blank=1，前处理必须在音素序列中插入 blank(0)：
`[0, ph0, 0, ph1, 0, ..., phN, 0]`，长度 = 2*N+1。
原始音素数：test_zh=30，test_en_zh=26。

## 环境信息
- Conda 环境: sophon-vitsMeloTTS
- Python 版本: 3.10.20
- onnxruntime 版本: 1.23.2
- jieba 版本: 0.42.1
- numpy 版本: 2.2.6

## 文本前处理实现说明
- jieba 分词 → 每个词查 lexicon.txt（词→音素序列+声调序列）
- 字符级回退：词 OOV 时逐字符查 lexicon
- 中文标点规范化：'。→.' '，→,' '？→?' '！→!'
- 英文按整词查 lexicon，若 OOV 则逐字母查
- 参考实现：sherpa-onnx/csrc/melo-tts-lexicon.cc

## 关键 Sophon BM1684X 移植注意事项

### 动态 Shape 处理
- TPU 不支持动态 shape：x 和 tones 的 L 维度（序列长度）必须编译时固定
- 建议静态 shape：N=1，L 固定为 128 或 256
- 超长输入需截断，短输入需 padding（用 token_id=0 即 blank）
- 需要为不同长度分别编译 bmodel，或选取代表性最大长度

### 输入准备（CPU 侧）
- x 和 tones：需 pad/截断到固定长度 L
- x_lengths：仍然传原始长度（非 padding 后长度）
- sid/noise_scale/length_scale/noise_scale_w：标量，直接准备

### 模型结构特点
- 无 BERT 特征依赖（sherpa-onnx 版本已将 bert/ja_bert 固定为零向量，导出时已 hardcode）
- 单说话人（n_speakers=1），sid 固定为 1
- add_blank=1：前处理中 blank token 已 hardcode 进 lexicon 逻辑
- 输出为原始 float32 音频波形，shape [1, 1, T]，需 clip+转 int16 写 WAV

### token 范围（Sophon 量化参考）
- token id 范围：0~111（共 112 个）
- tone 范围：0~10（0 为标点/blank，1~5 中文，7~10 英文）
- 音频输出范围：大致 [-1, 1] float32

### 推荐编译参数（待验证）
```bash
# F32 版本（基准）
model_deploy --model model.onnx --quantize F32 --chip bm1684x \
  --input_shapes [1,128],[1],[1,128],[1],[1],[1],[1] --output model_F32.bmodel

# F16 版本
model_deploy --model model.onnx --quantize F16 --chip bm1684x \
  --input_shapes [1,128],[1],[1,128],[1],[1],[1],[1] --output model_F16.bmodel
```
注意：x_lengths shape 为 [1]（非 [1,1]），sid/noise_scale 等均为 [1]。
