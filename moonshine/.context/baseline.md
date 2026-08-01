# Moonshine streaming-small Baseline 测试结果

## 模型信息
- 模型名称: moonshine-streaming-small(ASR, encoder-decoder)
- 模型路径: models/moonshine-streaming-small/(model.safetensors + config.json + tokenizer.json + preprocessor_config.json)
- 加载类: `MoonshineStreamingForConditionalGeneration` + `AutoProcessor`(transformers 5.14.1)
- 参数量: 140.1M
- 结构:
  - Encoder: hidden 620, 10 层, 8 heads, head_dim 64, vocab 无关;streaming 变体
  - Decoder: hidden 512, 10 层, 8 heads, head_dim 64, vocab 32768;proj_out [512->32768] 无 bias,与 embed_tokens 权重 tied
  - RoPE: partial_rotary_factor 0.5, theta 10000

## Encoder 预处理管线(模型内部,非外部特征提取)
streaming 变体的 encoder 自带完整预处理,导出时全部打进 ONNX:
1. reshape: [1, N] -> [1, T, frame_len=80](frame_len = 16000 * 5ms / 1000 = 80)
2. Frame CMVN(逐帧归一化, eps=1e-6)
3. Asinh 压缩: `asinh(exp(log_k) * x)`, k=0.75
4. Linear(80 -> 620, bias=False) + silu -> [1, T, 620]
5. CausalConv1d(k=5, s=2) + silu + CausalConv1d(k=5, s=2) -> [1, T_enc, 620]
6. 10 层 attention(每层带 sliding window mask)+ final_norm

**shape 公式(实测验证)**:
- T = N // 80(帧数)
- T_enc = ((T-1)//2+1 -1)//2+1 = (T-1)//4 + 1
- 10s 固定 shape:N=160000 -> T=2000 -> **T_enc=500**(与任务预期一致)
- 每层 sliding_windows = [[16,4]x2, [16,0]x6, [16,4]x2](attention 额外 AND 上窗口约束)

## Decoder 关键细节
- forward 签名(已确认): `decoder(input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, encoder_hidden_states, encoder_attention_mask)`
- **cross-attention 前处理(重要)**: 对 encoder_hidden_states 加学习的 pos_emb(位置 arange(T_enc)),然后 `proj` Linear(620 -> 512, bias=False) 后才做 cross-attention
- 输出: `proj_out(decoder.last_hidden_state)` -> logits [1, T, 32768]
- 解码: sos=1, eos=2, pad=0; greedy argmax; max_new_tokens = 音频秒数*6 + 10

## 测试数据
来源: sherpa-onnx-moonshine-base-en-int8/test_wavs/(原版 base 的测试集,streaming-small 输出以实测为准,已复制到 test_data/)
- 0.wav: 6.62s, 16kHz, 106000 samples
- 1.wav: 16.71s, 16kHz, 267440 samples(超过 10s 固定 shape,后续固定 shape 导出需注意/截断)
- 8k.wav: 4.83s, 8kHz, 38600 samples(torchaudio 重采样到 16kHz -> 77200 samples)

## Baseline 结果(F32, CPU, PyTorch 2.11.0)
| 测试用例 | 时长 | input shape | T_enc | decoder steps | 耗时(gen/手动循环) |
|---------|------|------------|-------|--------------|-------------------|
| 0.wav | 6.62s | [1, 106000] | 332 | 28 | 0.53s / 0.37s |
| 1.wav | 16.71s | [1, 267440] | 836 | 62 | 1.33s / 0.94s |
| 8k.wav | 4.83s | [1, 77200] | 242 | 19 | 0.35s / 0.26s |

文本输出(model.generate 与手动 decoder 循环完全一致,3/3 match):
- 0.wav: "After early nightfall, the yellow lamps would light up here and there, the squalid quarter of the brothels."
- 1.wav: "God, as a direct consequence of the sin which man thus punished, had given her a lovely child, whose place was on that same dishonored bosom, to connect her parent forever with the race and descent of mortals, and to be finally a blessed soul in heaven."
- 8k.wav: "Yet these thoughts affected Hester Prynne less with hope than apprehension."

与参考转写(trans.txt)内容一致,仅大小写/标点差异。

## 关键中间输出(python/test/outputs/debug/,给 C++ 对比用)
每个 wav 保存:
- `{name}_input_values.npy`: [1, N] f32,processor 输出(原始音频,无外部特征)
- `{name}_attention_mask.npy`: [1, N] int64,全 1
- `{name}_encoder_output.npy`: [1, T_enc, 620] f32,encoder last_hidden_state(ground truth)
- `{name}_encoder_attention_mask.npy`: [1, T_enc] int32,encoder 内部下采样后的 mask
- `{name}_decoder_logits.npy`: [steps, 32768] f32,每步 argmax 前的 logits(ground truth)
- `{name}_decoder_tokens.npy`: [steps] int64,逐步 token(sos 开头,eos 结尾)

baseline 记录: python/test/outputs/baseline/{0,1,8k}.json + .txt + summary.json

## 固定 shape 导出参数(10s)
- 输入: input_values [1, 160000] f32, attention_mask [1, 160000] int32
- encoder 输出: last_hidden_state [1, 500, 620] f32 + encoder_attention_mask [1, 500]
- decoder 输入: input_ids [1, 1] int64 + encoder_hidden_states [1, 500, 620] + KV cache
- 注意: 1.wav(16.71s, T_enc=836)超过固定 shape,基准测试保留原始长度;固定 shape 验证建议用 0.wav / 8k.wav 或截断

## 环境信息
- Conda 环境: sophon-whisper
- Python 3.10.20, PyTorch 2.11.0+cpu, transformers 5.14.1, numpy 2.2.6, soundfile 0.13.1, torchaudio 2.11.0+cpu, onnx 1.21, onnxsim 0.6.3
