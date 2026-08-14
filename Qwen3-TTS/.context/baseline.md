# Qwen3-TTS PyTorch Baseline（M0）

## 模型信息

- 模型: Qwen3-TTS-12Hz-0.6B-CustomVoice（自回归 LLM 型 TTS）
- 模型路径: `/home/xh/itc_project/RK_model_zoo/models/Qwen3-TTS-12Hz-0.6B-CustomVoice`
- 主权重: `model.safetensors`（1.8GB，含 talker + code_predictor + 文本/码本嵌入）
- codec: `speech_tokenizer/model.safetensors`（682MB，12Hz 编解码器）
- 类型: custom_voice（9 种预定义音色，0.6B 不支持 instruct）

## 推理三段结构

```
1) 文本 → Qwen3 tokenizer → input_ids [1, L]
2) Talker(0.6B, 28层, GQA16/8, QK-Norm, mRoPE→1D, vocab3072)
   prefill: inputs_embeds [1, L_pre, 1024]（文本+说话人+语言+控制码复合嵌入）
   decode:  每 code 步 1 次 talker + 嵌套 code_predictor 15 步
   输出:    audio_codes [T, 16]（T 个 code 步 × 16 层量化码）
3) speech_tokenizer 12Hz codec: codes [1,T,16] → wav [T×1920] @24kHz
```

## 关键配置

| 模块 | 关键参数 |
|---|---|
| talker | hidden 1024, 28层, 16头/8KV头, head_dim 128, intermediate 3072, vocab 3072, text_vocab 151936, text_hidden 2048, num_code_groups 16 |
| code_predictor | hidden 1024, 5层, vocab 2048, 15套 embedding/lm_head（共享5层） |
| codec | 输入/输出 24kHz, decode_upsample_rate 1920, encoder_valid_num_quantizers 16 |

## 环境

- Conda: `qwen3-tts-sophon`（python 3.10.20 / torch 2.6.0+cu124 / transformers 4.57.3 / qwen-tts 0.1.1）
- 设备: NVIDIA RTX 3060 12.9GB, dtype bf16, attn_implementation=eager（无 flash-attn）
- 模型加载耗时: 7.3s

## 测试用例

- 文本: `今天天气真好，我们一起去公园散步吧。`
- 语言: Chinese, 说话人: Vivian
- input_ids: [1, 19]（含 `<|im_start|>assistant\n...` 包装）

## Baseline 结果

| 模式 | codes | hidden | wav | 采样率 | 生成耗时 | RMS |
|---|---|---|---|---|---|---|
| greedy（do_sample=False） | [56,16] int32 | [56,1024] f32 | 107520 samples (4.48s) | 24000 | 14.50s | 0.096 |
| sample（do_sample=True, seed=0） | - | - | 96000 samples (4.00s) | 24000 | 12.44s | 0.101 |

## 产出文件

```
python/test/outputs/
├── baseline/
│   ├── result.json        # 元数据（shape/时序/配置）
│   ├── greedy_codes.npy   # [56,16] int32（确定性子图基准，供 ONNX/bmodel diff）
│   ├── greedy_hidden.npy  # [56,1024] float32
│   ├── greedy.wav         # 24kHz 确定性子图基准
│   └── sample.wav         # 24kHz 自然采样参考
└── debug/
    └── input_ids.npy      # [1,19] int32
```

## 移植关键要点（后续步骤）

1. **talker prefill 输入是 inputs_embeds（复合嵌入）**，非 input_ids：
   `text_embedding(151936→2048) → text_projection(2048→1024)` + `codec/speaker/control 嵌入(3072→1024)` 组合。
2. **decode 每步**：talker 采样第 0 层码 → code_predictor 自回归 15 步补全 16 层码 → 16 个嵌入求和 + tts_pad_embed → 下一步 talker。
3. **codec decode**：codes [1,T,16] → RVQ 查表 → 8层 transformer → conv 上采样 ×1920 → wav。官方用 chunked_decode(chunk 300 + 左上下文 25)。
4. **mRoPE**：纯文本场景三个 position 维度相同，退化为 1D RoPE，需 M1 验证。
5. **量化风险**：TTS 音质对精度敏感，F16 为基准，W4 需主观 + 波形 A/B。

## 参考

- 主参考: `chatTTS/`（三段式自回归 TTS 拆分）
- 次参考: `Qwen3-ASR/`（Qwen3-0.6B backbone W4BF16 g64 + C++ bmrt 引擎）
- 完整规划: `.context/plan.md`
