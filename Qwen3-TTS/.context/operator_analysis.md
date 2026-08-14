# Qwen3-TTS 算子兼容性分析（M1）

分析日期: 2026-08-13
工具: torch 2.6 / onnx 1.22 / TPU-MLIR v1.28.1
目标芯片: BM1684X

## 一、结论

三段子图使用的算子**全部在 TPU-MLIR 支持列表内**，无 NonZero / RandomNormalLike / If / Loop 等阻断性算子。主要工程风险是**图规模**（codec decoder 2932 节点）和 **transformers 4.57 的 vmap causal-mask 无法被 torch.onnx trace**（导出层问题，非算子问题）。

## 二、三段子图拆分与算子

### 1. talker（自回归主模型，28层）

- 结构: Qwen3-0.6B 同族 backbone（GQA 16/8, QK-Norm, SwiGLU, RMSNorm, causal attention）
- 位置编码: **mRoPE**（mrope_section [24,20,20], interleaved）→ 纯文本场景三个 position 维度相同，等价 1D RoPE
- 算子: MatMul/Linear, RMSNorm, SiLU, Sin/Cos/Neg/Concat/Split（RoPE 逐元素）, Softmax, RepeatKV, Gather（嵌入查表）
- **全部支持**

### 2. code_predictor（嵌套自回归，5层）

- 结构: 与 talker 同型 decoder 层（Qwen3TTSDecoderLayer），但 1D RoPE + Identity QK-Norm
- 15 套独立 embedding/lm_head（每量化层级一套），共享 5 层 transformer
- 算子: 同 talker，全部支持

### 3. codec decoder（12Hz codec，前馈）

已实际导出 ONNX（`models/onnx/codec_decoder_T325.onnx`，456MB f32，2932 节点，输入 [1,16,325] → wav）：

```
Add 158  And 1  Cast 148  Clip 17  Concat 138  Constant 1036  ConstantOfShape 30
Conv 31  ConvTranspose 6  Cos 1  Div 53  Equal 1  Erf 2  Exp 58  Expand 1
Gather 118  Less 1  LessOrEqual 1  MatMul 79  Mul 204  Neg 16  Pad 29  Pow 48
Range 2  Reciprocal 29  ReduceMean 21  Reshape 94  Shape 103  Sigmoid 8  Sin 30
Slice 75  Softmax 8  Split 2  Sqrt 19  Squeeze 16  Sub 63  Transpose 94
Unsqueeze 189  Where 2
```

共 39 种算子，**全部在支持列表**。Gather(118)=RVQ 码本查表；Conv/ConvTranspose=因果卷积上采样；Sin=SnakeBeta/RoPE；Range=位置 id（固定 T 时可折叠）。

## 三、导出层发现的 2 个问题（已解决）

### 问题 1: transformers 4.57 vmap causal-mask 无法 trace

`create_causal_mask` / `create_sliding_window_causal_mask` 内部用 `_vmap_for_bhqkv`（functorch vmap），torch.onnx（TorchScript exporter）直接报 `_Map_base::at`。

**解决**: 导出时 monkey-patch 这两个函数，用等价的可 trace 张量 mask 替换（见 `python/export_spike.py`）：
- full_attention: 上三角 -inf 因果 mask
- sliding_attention: 因果 + 窗口（codec decoder 8 层全 sliding，window=72）

### 问题 2: 默认走 SDPA

`attn_implementation` 未生效时 decoder 用 `sdpa_attention_forward`，dict mask 报错。

**解决**: 导出前显式 `config._attn_implementation = "eager"`。

## 四、M2 导出/转换方案

### talker

沿用 ChatTTS per-layer 思路：
- `embedding` 子模型: 文本嵌入（151936→2048→1024 投影）+ 码本嵌入（3072→1024）
- `block_0..27`（prefill，KV cache 输出）+ `block_cache_0..27`（decode，KV 输入/更新）
- `codec_head`（1024→3072）
- 用 `model_tool --combine` 合并单 bmodel

prefill 输入 `inputs_embeds [1, L_pre, 1024]` 由 CPU/embedding 子模型组合而成（文本+说话人+语言+控制码）。

### code_predictor

- 5 层共享 block（prefill `[1,2,1024]` + decode 单 token）
- 15 个 embedding 子模型 + 15 个 lm_head 子模型（每量化层级一套）

### codec decoder

- 单前馈 bmodel，固定 T（首版 T=325 = 300 chunk + 25 左上下文）
- 转换加 `--disable_layer_group`（图大）

### CPU 侧

- Qwen3 文本 tokenizer（tokenizers-cpp）
- embedding 组合逻辑 + top-k/top-p/temperature/repetition_penalty 采样
- 码流长度截取

## 五、风险

| 风险 | 等级 | 说明 |
|---|---|---|
| codec decoder 图规模（2932 节点） | 中 | 加 `--disable_layer_group`；必要时再拆 transformer 与 conv 两段 |
| 嵌套自回归性能 | 高 | 每 code 步 = 1 talker + 15 code_predictor，需常驻 KV + 预分配 |
| W4 量化音质 | 中 | 先 F16 对齐，再 W4 A/B |
| prefill 复合嵌入 CPU 开销 | 低 | 文本嵌入表 622MB，CPU 查表 + 投影一次性 |

## 参考

- 主参考: `chatTTS/`（per-layer GPT 导出 + codec 拆分）
- 次参考: `Qwen3-ASR/`（Qwen3-0.6B W4BF16 g64 + C++ 引擎）
