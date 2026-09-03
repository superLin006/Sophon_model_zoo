# Baseline 测试结果

> 本文件中的 baseline 数值是在原始 `model.onnx` 可用时记录的历史参考；当前仓库不归档该单体文件，交付回归以三段 ONNX 和 BM1684X C++ 端到端结果为准。

## 模型信息
- 模型名称: vits-melo-tts-zh_en
- 模型路径: 原始 `model.onnx` 未随仓库归档；当前交付输入为 `models/onnx/vits-melo-tts-zh_en/` 下三段 ONNX
- 模型类型: VITS-MeloTTS TTS（中英混合单说话人）
- 模型来源: sherpa-onnx 预转换 ONNX（无需从 PyTorch 导出）
- 输入数量: Part A 为 3 个
  - x: [1, 128] int64（音素 token ids，padding 到固定长度）
  - x_lengths: [1] int64（实际序列长度）
  - tones: [1, 128] int64（声调 ids，padding 到固定长度）
- Part A 输出: dp_w、h、attn_mask、x_mask
- Part C1 输入: z_p [1,192,512]、y_mask [1,1,512]
- Part C2 输出: y [1,1,262144]，采样率 44100Hz
- 模型大小: 当前交付为 99MB（三段 F16 bmodel）

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
- Conda 环境: sophon-vits-melo-tts-zh-en
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

### 静态 Shape 处理
- Part A 固定为 `N=1, L=128`，短输入由 C++ 用 token/tone 0 padding，并单独传入实际 `x_lengths`
- Part C1/C2 固定为 `T_mel=512`，C++ 将 CPU MAS 结果 padding 到 512；超过上限时在进入 TPU 前截断
- `seq_len` 支持范围为 1~128；超长文本应在前端分句后分别合成，不能直接传入单次推理

### 输入准备（CPU 侧）
- x 和 tones：需 pad 到 `[1,128]`
- x_lengths：传实际序列长度，范围 1~128
- Part C 的 `y_mask`：有效 mel 帧为 1，padding 帧为 0

### 模型结构特点
- 无 BERT 特征依赖（sherpa-onnx 版本已将 bert/ja_bert 固定为零向量，导出时已 hardcode）
- 单说话人（n_speakers=1），sid 固定为 1
- add_blank=1：前处理中 blank token 已 hardcode 进 lexicon 逻辑
- 输出为原始 float32 音频波形，shape [1, 1, T]，需 clip+转 int16 写 WAV

### token 范围（Sophon 量化参考）
- token id 范围：0~111（共 112 个）
- tone 范围：0~10（0 为标点/blank，1~5 中文，7~10 英文）
- 音频输出范围：大致 [-1, 1] float32

### 当前编译入口
请使用仓库内脚本生成三段 F16/F32 bmodel；不要直接对原始 `model.onnx` 执行 `model_deploy`。原始 ONNX 含动态时长和 TPU-MLIR 不支持的算子，必须先按 `python/README.md` 执行三段拆分：

```bash
conda run -n sophon-vits-melo-tts-zh-en python vits-melo-tts-zh_en/python/make_tpu_model.py
conda run -n sophon-vits-melo-tts-zh-en python vits-melo-tts-zh_en/python/make_split_models.py
docker exec sophon-tpumlir-v128 bash /workspace/vits-melo-tts-zh_en/python/gen_bmodel.sh F16
```

C++ 端的 `sid` 已固化在 Part A/C 的模型路径中，不再作为 bmodel 输入。
