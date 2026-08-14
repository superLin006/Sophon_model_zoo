# Qwen3-TTS 移植到 Sophon BM1684X 规划

## 目标

将 `Qwen3-TTS-12Hz-0.6B-CustomVoice`（自回归 LLM 型 TTS）移植到 Sophon BM1684X（172.16.25.248，root/1），C++ 部署并完成 RTF 测试。

## 模型架构

```
文本 → Qwen3 文本 tokenizer → embedding 组合(text_projection + speaker/language/control)
  → Talker(28层 decoder · GQA 16/8 · QK-Norm · mRoPE→1D · vocab 3072 · 0.6B)
  → codec_head → 采样第0层码
  → code_predictor(5层 · 15套 embedding/lm_head，嵌套自回归 15 步)
  → audio_codes[1,T,16]
  → speech_tokenizer 12Hz codec(RVQ查表 + 8层transformer + conv上采样×1920)
  → 24kHz 波形
```

## 参考项目（按优先级）

| 优先级 | 项目 | 借鉴 |
|---|---|---|
| 主 | ChatTTS | 三段式自回归 TTS 拆分、per-layer GPT 导出、嵌套 code、codec 解码、C++ 引擎 7 坑 |
| 次 | Qwen3-ASR | Qwen3-0.6B backbone W4BF16 g64 精度经验、model_tool --combine、C++ bmrt 引擎、交叉编译 |
| 工具链 | QwenLLM/LLM-TPU | llm_convert.py 与 Qwen3 官方支持 |
| 规范 | vits-melo | 部署/输出规范（架构不适用） |

## bmodel 拆分

- bmodel ① Talker：block_0..27 + block_cache_0..27 + codec_head + embedding 子模型
- bmodel ② code_predictor：5层共享 block + 15 embedding + 15 lm_head
- bmodel ③ codec decoder：RVQ 查表 + transformer + conv 上采样（chunked_decode 固定 chunk）

CPU 侧：文本 tokenizer + embedding 组合 + 采样逻辑。

## 量化

先 F16 跑通精度，再评估 W4BF16（group64，参考 Qwen3-ASR），TTS 音质需主观 + 波形 A/B 验证。

## 阶段

- M0：PyTorch baseline（三段输入输出 shape + 参考输出落盘）
- M1：算子分析 + ONNX 导出 spike（mRoPE→1D RoPE 验证、三段拆分）
- M2：bmodel 转换（F16 对齐）
- M3：C++ 实现 + 交叉编译
- M4：板卡部署 + RTF 测试

## 关键风险

1. mRoPE：纯文本场景退化为 1D RoPE，需 M1 spike 验证
2. 嵌套自回归性能：每 code 步 = 1 talker + 15 code_predictor
3. 量化音质：TTS 对精度敏感
4. 动态 shape：固定 max + padding + 截取
5. codec 算子：因果卷积 / SnakeBeta / RVQ Gather 需单独确认

## 环境

- Conda: `qwen3-tts-sophon`（py3.10 / torch 2.6 / transformers 4.57.3 / qwen-tts 0.1.1）
- TPU-MLIR: `sophgo/tpuc_dev:latest`（Docker）+ `0_Toolkits/tpu_mlir-1.28.1-py3-none-any.whl`
- 交叉编译: `sophon-cross-build`（Docker）+ `0_Toolkits/soc-sdk-sp4`
- 模型: `/home/xh/itc_project/RK_model_zoo/models/Qwen3-TTS-12Hz-0.6B-CustomVoice`
