# Moonshine streaming-small BM1684X 算子兼容性分析

**生成日期**: 2026-08-01
**目标芯片**: BM1684X (TPU-MLIR v1.28.1, Docker `sophgo/tpuc_dev:latest`)
**目标精度**: F32 + F16
**固定输入**: 10s 音频 → samples=160000 → T=2000 帧 → T_enc=500
**验证方式**: 全部结论基于实际 ONNX 导出 + onnxruntime 逐步推理 + 与 transformers 5.14.1 官方模型逐层数值对比(验证脚本位于 `/tmp/moonshine_analysis/`,导出产物在 `/tmp/moonshine_analysis/onnx/`)

---

## 0. 结论摘要(决策一览)

| 决策点 | 结论 | 理由(实测) |
|--------|------|------------|
| Encoder 方案 | **方案 B**(C++ 做 CMVN+Asinh, bmodel 输入 x_frames [1,2000,80]) | CMVN 在 F16 下特征误差实测 ~3.4%(max 2.5e-2);方案 B 的 bmodel 图无 ReduceMean/Log/Sqrt/Pow,且与参考已验证划分一致。方案 A 仅 F32 目标可选(误差 3.4e-5,可忽略) |
| Decoder 单步 | 23 输入 / 21 输出, embed/pos_emb/proj/RoPE/mask 全部进模型 | 算子全支持(Gather/Cos/Sin/Less/Or/Where 均可), 与 whisper Sophon 风格一致 |
| max_dec_len | **128** | 10s 理论解码步数 ≈72(T_enc·384/16000·6), 实测 16.71s 音频 62 步;128 留余量 |
| KV cache 格式 | 逐层 past_k_i/past_v_i [1,128,512] f32, **存旋转后 K**(与 transformers DynamicCache 一致) | 实测与 HF 逐层数值 0.0 差异 |
| RoPE cos/sin | **模型内计算**(Cos/Sin 算子支持), 回退方案: C++ 传 cos/sin [1,1,32] | 模型内接口简单;F16 trig 精度有 ~2e-2 模拟误差, 若转换后精度不达标切回退方案(接口一行切换) |
| encoder mask | 不需要(固定 10s 全有效), sliding window mask 预计算为常量 | 10 层 mask 与 HF `create_bidirectional_mask` 逐元素 0 差异 |
| 解码验证 | wrapper 41 步 token 序列与 HF 完全一致, logits max_abs=9.5e-6, ORT 逐步 argmax 100% | — |

---

## 1. Encoder 导出方案

### 1.1 两种方案(均已实际导出验证)

**方案 A — 整个 encoder 进 bmodel**(输入原始音频)
- 输入 `input_values [1,160000] f32` → 输出 `encoder_out [1,500,620] f32`
- 模型内: reshape 分帧 → CMVN(ReduceMean/Sub/Pow/Sqrt/Div, 全支持)→ asinh 分解为 `log(y+sqrt(y²+1))`(Log/Sqrt 支持, **Asinh 不在 TPU-MLIR 算子列表**)
- 实测算子集: Add, Conv, Div, Erf, LayerNormalization, Log, MatMul, Mul, Pow, ReduceMean, Reshape, Sigmoid, Softmax, Sqrt, Sub, Transpose — **全部支持**
- 数值: vs HF max_abs=3.4e-5(输出幅度 p99=1.35, 相对 ~1e-5); asinh 分解 f32 误差 7.2e-7
- **F16 风险(实测模拟)**: CMVN 在 f16 中间精度下 x_frames 误差 max_abs=2.5e-2(相对 ~3.4%)→ 方案 A 在 F16 目标有真实精度损失

**方案 B — C++ 做 CMVN+Asinh**(推荐)
- C++ 输入原始音频 → 输出 x_frames [1,2000,80] f32(纯数学, 40 行, 无 STFT/无 fftw3)
- bmodel 输入 `x_frames [1,2000,80] f32` → 输出 `encoder_out [1,500,620] f32`
- 实测算子集: Add, Conv, Div, Erf, LayerNormalization, MatMul, Mul, Reshape, Sigmoid, Softmax, Transpose — 全部支持, 比方案 A 少 6 个算子
- 数值: vs HF **bit 级一致(0.0 差异)**(同一份 embedder 前端输出)

### 1.2 决策: **方案 B**

1. F16 目标下 CMVN 实测误差 ~3.4%(语音波形幅度小, 均值相减存在灾难性消去);C++ 用 f32/f64 完全消除
2. bmodel 图更小更稳(无 ReduceMean/Log/Sqrt/Pow)
3. 与参考已验证的 CPU/NPU 划分一致, C++ 预处理逻辑可复用
4. C++ `std::asinh` 直接可用, 零分解误差

> 方案 A 保留为 F32 目标备选: 精度实测可接受(argmax 不变, logits 差 7.6e-6)。若后续要做"零预处理"的纯 F32 版, 直接采用 1.4 节代码。

### 1.3 方案 B 代码级实现(python/export_onnx.py 直接采用)

```python
# ============ Encoder 方案 B: 输入 x_frames [1,2000,80] ============
def make_sliding_mask(t_enc, left, right, dtype=torch.float32):
    """transformers 语义 sliding window mask(与 HF create_bidirectional_mask 全1mask 逐元素一致, 已实测 0 差异):
    keep: (dist>=0 & dist<left) | (dist<0 & -dist<right), dist = q - k
    [16,4]: dist∈[-3,15]; [16,0]: dist∈[0,15]   ← 注意与参考实现差 1 个位置(见 6.5)"""
    q = torch.arange(t_enc)[:, None]; k = torch.arange(t_enc)[None, :]
    dist = q - k
    keep = ((dist >= 0) & (dist < left)) | ((dist < 0) & (-dist < right))
    return torch.where(keep, torch.zeros((), dtype=dtype),
                       torch.tensor(torch.finfo(dtype).min, dtype=dtype)).unsqueeze(0).unsqueeze(0)

class EncoderWrapper(nn.Module):
    """方案 B: x_frames [1,2000,80] -> encoder_out [1,500,620]"""
    def __init__(self, encoder, t_enc=500):
        super().__init__()
        self.enc = encoder
        for i, (l, r) in enumerate(encoder.config.sliding_windows):
            self.register_buffer(f"mask_{i}", make_sliding_mask(t_enc, l, r), persistent=False)

    def forward(self, x_frames):
        enc = self.enc
        x = F.silu(enc.embedder.linear(x_frames))            # [1,2000,620]
        x = x.transpose(1, 2)                                # [1,620,2000]
        # 注意: HF MoonshineStreamingCausalConv1d.forward 内部自带 left_pad(4),
        # 不要再外层 F.pad(否则双重 padding -> 输出长度 503 而非 500)
        x, _ = enc.embedder.conv1(x); x = F.silu(x)          # [1,1240,1000]
        x, _ = enc.embedder.conv2(x)                         # [1,620,500]
        x = x.transpose(1, 2)                                # [1,500,620]
        for i, layer in enumerate(enc.layers):
            x = layer(x, attention_mask=getattr(self, f"mask_{i}"))
        return enc.final_norm(x)                             # [1,500,620]
```

**C++ 预处理**(方案 B 唯一新增代码, 纯数学):

```cpp
// input: 原始音频 [1,160000] f32; frame_len=80; eps=1e-6; log_k=模型参数(见下)
// 1) 分帧: x[i][f] = audio[i*80+f]
// 2) CMVN: mean=Σx/80; centered=x-mean; rms=sqrt(Σcentered²/80+1e-6); x=centered/rms
// 3) Asinh: y = exp(log_k)*x; out = std::asinh(y)   // cmath 直接可用, 零分解误差
// 输出 x_frames [1,2000,80] f32 = 640KB
```

**log_k 参数导出**: `log_k` 是 `encoder.embedder.comp.log_k`(标量 Parameter, 官方 k_init=0.75 训练得到)。方案 B 中 C++ 需要它——从模型权重直接读取导出为文本:
```python
# export 脚本中:
log_k = float(model.model.encoder.embedder.comp.log_k.detach())
np.save(f"{asset_dir}/log_k.npy", np.float32(log_k))   # 或写一行文本
```
> 注意: 方案 A 的 wrapper 里用 `torch.exp(self.log_k)` 作为常量(导出为 initializer), 无需手工迁移。

### 1.4 方案 A 代码级实现(备选, 仅 F32 目标)

```python
class EncoderWrapperFull(nn.Module):
    """方案 A: input_values [1,160000] -> encoder_out [1,500,620]
    asinh 分解为 log(y+sqrt(y^2+1)); y = exp(log_k)*x (log_k 常量折叠)"""
    def __init__(self, encoder, t_enc=500, t=2000):
        super().__init__()
        self.enc = encoder; self.t = t
        with torch.no_grad():
            self.k = torch.exp(encoder.embedder.comp.log_k.detach())   # 标量常量
        for i, (l, r) in enumerate(encoder.config.sliding_windows):
            self.register_buffer(f"mask_{i}", make_sliding_mask(t_enc, l, r), persistent=False)

    def forward(self, input_values):
        enc = self.enc
        x = input_values.reshape(1, self.t, 80)              # 显式静态 reshape(避免 -1 动态)
        mean = x.mean(dim=-1, keepdim=True)                  # ReduceMean ✓
        centered = x - mean
        rms = (centered.pow(2).mean(dim=-1, keepdim=True) + 1e-6).sqrt()
        x = centered / rms
        y = self.k * x
        x = torch.log(y + torch.sqrt(y * y + 1.0))           # Asinh 不支持, 分解为 Log/Sqrt
        x = F.silu(enc.embedder.linear(x))
        x = x.transpose(1, 2)
        x, _ = enc.embedder.conv1(x); x = F.silu(x)
        x, _ = enc.embedder.conv2(x)
        x = x.transpose(1, 2)
        for i, layer in enumerate(enc.layers):
            x = layer(x, attention_mask=getattr(self, f"mask_{i}"))
        return enc.final_norm(x)
```

---

## 2. Decoder 单步导出方案

### 2.1 设计决策(对照 whisper 的 Sophon 风格)

| 逻辑 | 位置 | 依据 |
|------|------|------|
| token embedding lookup | **模型内**(Gather) | Gather ✓ 支持; whisper 同款 |
| encoder pos_emb(arange(500) 查表)+ proj 620→512 | **模型内**(Gather 折叠为常量 + MatMul) | 与 HF 语义一致(每步重算, HF 也每步重算) |
| RoPE cos/sin 计算(位置=cache_len) | **模型内**(MatMul/Mul + Cos/Sin + Concat) | Cos/Sin ✓ 支持; 回退: C++ 传 cos/sin [1,1,32](见 5.3) |
| causal mask 生成 | **模型内**(Less + Or + Where, 从 cache_len 生成) | 免去 C++ 每步造 mask; whisper 是 C++ 传 mask, 两种都可行 |
| encoder(cross) mask | **不需要** | 固定 10s 全有效; 如未来变长再补 [1,1,1,500] 输入 |
| cross-attn KV | **每步重算**(无缓存) | 与 HF/参考实现一致; 500×512 每层每步 ~26 MFLOP, 可忽略 |
| KV cache | 逐层输入 past_k/v_i [1,128,512], 输出 new_k/v_i [1,1,512] | whisper 风格(逐层命名), C++ 指针直通; 存**旋转后 K** |

### 2.2 输入输出完整定义(最终接口)

**输入(23 个)**:

| name | shape | dtype | 说明 |
|------|-------|-------|------|
| token | [1,1] | int64(ONNX)/int32(bmodel) | 当前 token id, 模型内做 embedding |
| encoder_out | [1,500,620] | f32 | encoder 输出(原始, 模型内加 pos_emb + proj) |
| cache_len | [1] | int64(ONNX)/int32(bmodel) | 当前已填充 KV 长度 = 当前 token 位置 |
| past_k_0..9 | [1,128,512] | f32 | 各层 past K(**旋转后**), 未填充槽为 0 |
| past_v_0..9 | [1,128,512] | f32 | 各层 past V |

**输出(21 个)**:

| name | shape | dtype | 说明 |
|------|-------|-------|------|
| logits | [1,1,32768] | f32 | 下一步 token 分布 |
| new_k_0..9 | [1,1,512] | f32 | 各层当前 token 的 K(旋转后), C++ 写入 cache 位置 cache_len |
| new_v_0..9 | [1,1,512] | f32 | 各层当前 token 的 V |

> int64 → int32: bmruntime 无 INT64, TPU-MLIR 编译时自动降 int32(知识库规则), C++ 上传时 cast 到 int32 即可(与 whisper 相同)。

### 2.3 代码级实现

```python
class DecoderWrapper(nn.Module):
    """单步 decoder: token[1,1]int64 + encoder_out[1,500,620]f32 + cache_len[1]int64
       + past_k/v_i[1,128,512]f32 -> logits[1,1,32768] + new_k/v_i[1,1,512]f32"""
    def __init__(self, model, max_dec_len=128, t_enc=500):
        super().__init__()
        dec = model.model.decoder
        self.embed_tokens = dec.embed_tokens      # Embedding(32768,512)
        self.pos_emb = dec.pos_emb                # Embedding(4096,620)
        self.proj = dec.proj                      # Linear(620,512,bias=False)
        self.layers = dec.layers
        self.norm = dec.norm                      # LayerNorm(512)
        self.proj_out = model.proj_out            # Linear(512,32768,bias=False, tied)
        self.max_dec_len = max_dec_len
        self.scaling = 64 ** -0.5                 # head_dim=64
        with torch.no_grad():
            self.inv_freq = dec.rotary_emb.inv_freq.detach().float()   # [16] 常量
        self.register_buffer("pos_idx", torch.arange(t_enc, dtype=torch.long), persistent=False)  # [500]
        self.register_buffer("mask_arange",
                             torch.arange(max_dec_len + 1, dtype=torch.long).view(1, 1, 1, -1),
                             persistent=False)                                                        # [1,1,1,129]

    def apply_rope(self, q, k, cos, sin):
        """HF apply_rotary_pos_emb 等价(interleaved), 显式 stack/flatten 替代 repeat_interleave
        (onnx RepeatInterleave 无对应算子; 与参考实现的 rotate_half 相同手法)"""
        cos_i = torch.stack([cos[..., :16], cos[..., :16]], dim=-1).flatten(-2)   # [1,1,32] c0,c0,c1,c1,...
        sin_i = torch.stack([sin[..., :16], sin[..., :16]], dim=-1).flatten(-2)
        q_rot, q_pass = q[..., :32], q[..., 32:]
        k_rot, k_pass = k[..., :32], k[..., 32:]
        q_half = torch.stack([-q_rot[..., 1::2], q_rot[..., 0::2]], dim=-1).flatten(-2)
        k_half = torch.stack([-k_rot[..., 1::2], k_rot[..., 0::2]], dim=-1).flatten(-2)
        q_embed = q_rot * cos_i + q_half * sin_i
        k_embed = k_rot * cos_i + k_half * sin_i
        return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)

    def forward(self, token, encoder_out, cache_len, *kv):
        n = 10
        past_k, past_v = kv[0:n], kv[n:2*n]

        # 1) encoder 预处理: pos_emb(arange(T_enc)) + proj 620->512 (每步重算, 与 HF 一致)
        pe = self.pos_emb(self.pos_idx).unsqueeze(0)            # [1,500,620] (pos_idx 常量 -> 折叠)
        enc = self.proj(encoder_out + pe)                       # [1,500,512]

        # 2) token embedding (Gather)
        x = self.embed_tokens(token)                            # [1,1,512]

        # 3) RoPE: 位置=cache_len, 模型内计算 cos/sin (f32)
        pos = cache_len.to(torch.float32).unsqueeze(-1)         # [1,1]
        freqs = pos * self.inv_freq.unsqueeze(0)                # [1,16]
        emb = torch.cat([freqs, freqs], dim=-1)                 # [1,32]
        cos = emb.cos().unsqueeze(1)                            # [1,1,32]
        sin = emb.sin().unsqueeze(1)

        # 4) causal mask: 过去位置 0..cache_len-1 + 尾部当前 token 槽(max_dec_len) 有效
        #    ★当前 token 的 k/v 拼接在缓存末尾(位置 128), 不是缓存开头!
        valid = ((self.mask_arange < cache_len.view(1, 1, 1, 1)) |
                 (self.mask_arange == self.max_dec_len))        # [1,1,1,129]
        mask = torch.where(valid, torch.zeros((), dtype=encoder_out.dtype),
                           torch.tensor(-1e9, dtype=encoder_out.dtype))

        new_keys, new_values = [], []
        for i, layer in enumerate(self.layers):
            # ---- self-attn ----
            h = layer.input_layernorm(x)
            q = layer.self_attn.q_proj(h).view(1, 1, 8, 64).transpose(1, 2)
            k = layer.self_attn.k_proj(h).view(1, 1, 8, 64).transpose(1, 2)
            v = layer.self_attn.v_proj(h).view(1, 1, 8, 64).transpose(1, 2)
            q, k = self.apply_rope(q, k, cos, sin)
            k_new = k.transpose(1, 2).reshape(1, 1, -1)         # [1,1,512] 旋转后 K(与 HF cache 格式一致)
            v_new = v.transpose(1, 2).reshape(1, 1, -1)
            k_full = torch.cat([past_k[i], k_new], dim=1)       # [1,129,512]
            v_full = torch.cat([past_v[i], v_new], dim=1)
            kf = k_full.view(1, self.max_dec_len + 1, 8, 64).transpose(1, 2)
            vf = v_full.view(1, self.max_dec_len + 1, 8, 64).transpose(1, 2)
            w = F.softmax((q * self.scaling) @ kf.transpose(-2, -1) + mask,
                          dim=-1, dtype=torch.float32).to(q.dtype)
            o = (w @ vf).transpose(1, 2).reshape(1, 1, -1)
            x = x + layer.self_attn.o_proj(o)

            # ---- cross-attn (无 mask: 固定 10s 全有效) ----
            h = layer.post_attention_layernorm(x)
            qc = layer.encoder_attn.q_proj(h).view(1, 1, 8, 64).transpose(1, 2)
            kc = layer.encoder_attn.k_proj(enc).view(1, 500, 8, 64).transpose(1, 2)
            vc = layer.encoder_attn.v_proj(enc).view(1, 500, 8, 64).transpose(1, 2)
            wc = F.softmax((qc * self.scaling) @ kc.transpose(-2, -1),
                           dim=-1, dtype=torch.float32).to(qc.dtype)
            oc = (wc @ vc).transpose(1, 2).reshape(1, 1, -1)
            x = x + layer.encoder_attn.o_proj(oc)

            # ---- MLP (silu GLU; 显式 Slice 替代 chunk, 避免 Split 算子) ----
            y = layer.mlp.fc1(layer.final_layernorm(x))         # [1,1,4096]
            hidden, gate = y[..., :2048], y[..., 2048:]
            x = x + layer.mlp.fc2(F.silu(gate) * hidden)

            new_keys.append(k_new); new_values.append(v_new)

        logits = self.proj_out(self.norm(x))                    # [1,1,32768]
        return (logits, *new_keys, *new_values)                 # 21 个输出
```

**导出要点**:
- `model.config._attn_implementation = "eager"`(强制 eager 路径, 避免 sdpa 自定义算子)
- opset_version=17(torch 2.11 可能升 18 并写 Split 的 num_outputs — 见 6.3)
- KV dummy 用列表推导式: `[torch.zeros(1, 128, 512) for _ in range(10)]`(知识库规则, 不能用 `* 10`)
- 实测导出的 decoder onnx: 847 节点, opset 17, onnxsim ok, checker ok

### 2.4 KV cache 语义(重要)

- transformers `DynamicCache` 存的是 **旋转后 K**(`apply_rotary_pos_emb` 之后 `cache.update`)→ 本方案 k_new 输出旋转后 K, C++ 原样写入
- 每层缓存 [1,128,512] 由 C++ 持有: 初始全 0; 第 t 步把 new_k_i/new_v_i 写入位置 t
- mask 有效位 = 位置 0..t-1(过去)+ 位置 128(当前 token 拼接槽); **其余 -1e9**
- C++ 每步上传 20 个 cache tensor(128×512×4B = 262KB × 20 = 5.2MB/步); 性能优化: 用 `bmrt_tensor_with_device` + 常驻设备缓存(知识库 ChatTTS 经验), 把 past_k/v 输入路由到常驻缓存, new_k/v 输出经 4KB scratch 回写, 每步设备侧流量降到 ~30KB(ONNX 接口不变)

### 2.5 max_dec_len = 128

- 理论: T_enc·384/16000·6 = 500·384/16000·6 ≈ 72 步
- 实测: baseline 中 16.71s 音频 62 步; 10s 预计 ~70 步
- 128 留 ~80% 余量; KV 成本 128×512×4B×2×10 = 5.2MB, 无压力

---

## 3. 算子分类清单(实测导出扫描)

### Encoder 方案 B(推荐版, 331 节点)

| 算子 | 出现次数 | 分类 | 说明 |
|------|---------|------|------|
| MatMul | 81 | ✅ 完全支持 | Linear/attention |
| Add | 60 | ✅ 完全支持 | residual/mask 加法 |
| Mul | 53 | ✅ 完全支持 | scaling/gelu |
| Transpose | 42 | ✅ 完全支持 | head 重排 |
| Reshape | 40 | ✅ 完全支持 | view/head 拆分 |
| LayerNormalization | 21 | ✅ 完全支持 | MoonshineStreamingLayerNorm(unit_offset 折叠) |
| Erf | 10 | ✅ 完全支持 | encoder MLP gelu 分解(TPU-MLIR 有 Erf) |
| Softmax | 10 | ✅ 完全支持 | attention softmax(f32) |
| Div | 10 | ✅ 完全支持 | silu 分解 |
| Sigmoid | 2 | ✅ 完全支持 | silu |
| Conv | 2 | ⚠️ 需补属性 | torch 2.11 不写 kernel_shape, 需 fix_conv_kernel_shape(见 4.1) |

> onnxsim 后额外出现 Split:10(onnxsim 折叠 Reshape 产生, 标准 split 属性形式, TPU-MLIR 支持)。

### Encoder 方案 A(全模型版, 345 节点)

B 的全部算子 + 以下 6 个(全支持):

| 算子 | 出现次数 | 分类 | 说明 |
|------|---------|------|------|
| ReduceMean | 2 | ✅ 完全支持 | CMVN 均值/RMS |
| Sqrt | 2 | ✅ 完全支持 | RMS |
| Log | 1 | ✅ 完全支持 | asinh 分解 |
| Pow | 1 | ✅ 完全支持 | centered² |
| Sub | 1 | ✅ 完全支持 | 均值相减 |
| Div | +1 | ✅ 完全支持 | CMVN 归一化 |

> **Asinh 本身不在 TPU-MLIR 列表**(也无 Asin/Sinh/Cosh)→ 方案 A 必须分解为 Log/Sqrt(已实测 f32 误差 7.2e-7)。

### Decoder 单步(847 节点, 全部支持)

| 算子 | 出现次数 | 分类 | 说明 |
|------|---------|------|------|
| MatMul | 142 | ✅ | 三层 attention + MLP + proj_out |
| Reshape | 143 | ✅ | head 拆分/拼接 |
| Transpose | 100 | ✅ | head 布局 |
| Slice | 102 | ✅ | rotate_half 奇偶切分(step=2) + GLU 切半 |
| Add | 81 | ✅ | residual/mask |
| Mul | 81 | ✅ | RoPE 旋转/scaling |
| Unsqueeze | 46 | ✅ | 广播准备 |
| LayerNormalization | 31 | ✅ | 4×10 层 + 输出 norm |
| Neg | 20 | ✅ | rotate_half 取负 |
| Softmax | 20 | ✅ | self/cross attention |
| Sigmoid | 10 | ✅ | silu |
| Concat | 63 | ✅ | KV 拼接/rope emb |
| Cos / Sin | 1/1 | ✅ | RoPE(模型内计算) |
| Gather | 2(简化后 1) | ✅ | embed_tokens; pos_emb 常量折叠 |
| Cast | 1 | ✅ | cache_len f32 |
| Less / Or / Where | 1/1/1 | ✅ | causal mask 生成 |
| Split | 0 | — | 已用 Slice 替代 chunk(见 4.2) |

**无任何不支持算子**。所有算子均落在 TPU-MLIR 支持列表内。

---

## 4. 问题算子与修改方案(代码级)

### 4.1 Conv 缺 kernel_shape 属性(encoder, 必改)

torch 2.11 导出 Conv 不写 kernel_shape → `model_transform.py` 报错。补全(whisper 已验证的修复):

```python
def fix_conv_kernel_shape(model):
    init_map = {init.name: init for init in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        if any(attr.name == "kernel_shape" for attr in node.attribute):
            continue
        weight_name = node.input[1]
        if weight_name not in init_map:
            continue
        kernel_shape = list(init_map[weight_name].dims[2:])
        node.attribute.append(onnx.helper.make_attribute("kernel_shape", kernel_shape))
    return model
```

### 4.2 Split(num_outputs) 与 onnxsim 0.6.3 冲突(decoder, 已规避)

- torch 2.11 的 `chunk(2, dim=-1)` 导出为 Split 并带 `num_outputs` 属性(opset 18 格式), onnxsim 0.6.3 的 C++ parser 崩溃(`Unrecognized attribute: num_outputs`)
- **规避方案(已采用)**: wrapper 里用显式 Slice 替代 chunk:
  ```python
  hidden, gate = y[..., :2048], y[..., 2048:]   # 等价于 chunk(2, dim=-1), 导出为 Slice
  ```
- 若其他模型仍遇到: 用 fix_onnx.py 把 num_outputs 换算成 split 属性并转回 opset 17(`/tmp/moonshine_analysis/fix_onnx.py`)

### 4.3 Asinh 不支持(方案 A 才有, 已分解)

`asinh(y) = log(y + sqrt(y²+1))`, 分解为 Mul/Add/Sqrt/Log(全支持)。f32 误差 7.2e-7(实测)。

### 4.4 opset 统一

- decoder 导出 opset 17 ✓; encoder 经 onnxsim 后可能变 18 — 转 bmodel 前统一为 17(`model.opset_import[0].version = 17`, 所用算子全部 opset≤13, 无兼容问题)

---

## 5. 精度影响评估(实测数据)

| 环节 | 实测误差 | 评估 |
|------|---------|------|
| asinh 分解(f32) | 7.2e-7 | 可忽略 |
| 方案 A encoder 输出 vs HF | max_abs 3.4e-5(幅度 p99=1.35) | F32 下可接受; logits 差 7.6e-6, argmax 不变 |
| **CMVN f16 模拟** | **x_frames max_abs 2.5e-2(相对 ~3.4%)** | **方案 B 的核心动机; F16 目标下方案 A 有风险** |
| RoPE cos/sin f16 模拟 | max_abs ~2e-2(位置 126, 角度 127 rad) | 中等风险, 见下 |
| decoder wrapper vs HF(41 步) | logits max_abs 9.5e-6, token 序列一致, argmax 100% | 图结构正确性已验证 |
| ORT 逐步 vs wrapper | max_abs 2.3e-5, argmax 100% | ONNX 图正确性已验证 |

**RoPE F16 风险与回退**: BM1684X 的 Sin/Cos 走 LUT, 实际精度以 bmodel 转换后实测为准。若 F16 下 cos/sin 误差导致转写劣化, 回退方案(一行切换): C++ 预计算 f32 的 cos/sin 表(`inv_freq` 表 [128,32], 参考实现 `precompute_rope_table` 同款逻辑), 每步按 cache_len 切片传 `cos_input/sin_input [1,1,32] f32` 两个输入, wrapper 内 `cos = cos_input.unsqueeze(1)` 替代第 3 步计算。**接口预留此两种形态, 转换后按实测精度二选一**。

---

## 6. 关键坑(实测发现, 移植必读)

1. **HF decoder 原地修改 encoder_hidden_states**: `decoder.forward` 内 `encoder_hidden_states += pos_emb` 会修改调用者的 tensor(baseline test_pytorch.py 的手动循环因此从第 2 步起是**双重 pos_emb**)。验证脚本必须每步传 `enc_out.clone()`。bmodel 是纯函数无此问题, 但**精度对比的 ground truth 必须以"每步传干净副本"的 HF 循环为准**(本分析已验证的 41 步轨迹即按此修正)。
2. **causal mask 的尾部槽语义**: 当前 token 的 k/v 拼接在缓存末尾(位置 max_dec_len), mask 必须标 `位置0..cache_len-1 + 位置max_dec_len` 有效——若按"位置 0..cache_len 有效"写, attention 会去关注全零的缓存槽, 输出完全错误(曾导致 argmax 0% 一致, 修复后 100%)。参考实现的 dummy mask(仅末尾有效)同样错误(只 attend 当前 token, 丢失历史上下文)。
3. **HF CausalConv1d 自带 left_pad**: `MoonshineStreamingCausalConv1d.forward` 内部 `F.pad(x, (left_pad, 0))`, wrapper 外层不能再 pad(双重 padding 输出 503 帧而非 500)。参考实现在外层 pad + 普通 Conv1d 是另一套实现, 勿混用。
4. **sliding window mask 边界与参考实现差 1 位**: transformers 语义 keep dist∈[-right+1, left-1]([16,4]→[-3,15]); 参考实现的 `create_sliding_window_mask` 是 [-right, left-1](含 -4)。必须用 transformers 语义(本方案 `make_sliding_mask` 与 HF 逐元素 0 差异)。
5. **onnxsim 0.6.3 不支持 Split 新格式**(见 4.2); **torch 2.11 导出必须显式 `config._attn_implementation="eager"`**(sdpa 会导出 F.scaled_dot_product_attention 自定义路径)。
6. **int64 边界**: ONNX 输入用 int64 导出, TPU-MLIR 编译时自动降 int32; C++ 上传 cast 到 int32(知识库规则)。
7. **KV dummy 列表推导式**: `[torch.zeros(...) for _ in range(10)]`, 不能用 `* 10`。

---

## 7. 移植策略与风险等级

**移植策略**(1-3 句): 按本方案实现 `python/export_onnx.py`(EncoderWrapper 方案 B + DecoderWrapper + fix_conv_kernel_shape + onnxsim), 产出 `moonshine_encoder_sim.onnx` / `moonshine_decoder_sim.onnx`; Docker 内 `model_transform.py --input_shapes [[1,2000,80]] / [[1,1],[1,500,620],[1],20×[1,128,512]]` + `model_deploy.py --chip bm1684x --quantize F16|F32 --disable_layer_group`(23 输入 21 输出必加); C++ 侧仅新增 CMVN+Asinh 预处理(~40 行)与逐步 KV 缓存管理(参照 whisper C++ 端口)。

**风险等级: 低**。全部算子受支持、图结构与 HF 数值验证通过、方案 B 消除 F16 预处理精度风险。剩余风险点: (a) F16 下 Sin/Cos LUT 精度(回退方案就绪); (b) LayerNorm/Softmax 在 TPU-MLIR F16 下的内部精度(所有 bmodel F16 共性问题, 转换后以 WER/文本对比确认); (c) 23 输入 21 输出需 `--disable_layer_group`, 已列入 checklist。

**性能预估**: encoder [1,2000,80]→[1,500,620] 单次 ~1.5 GFLOP(F16 下 <50ms); decoder 每步 ~0.6 GFLOP(含每步重算的 cross-attn KV 与 encoder proj), ~70 步总 ~45 GFLOP; 预计整体 RTF 远优于实时(whisper small 同量级架构在 BM1684X 上已跑通)。

---

## 8. 导出注意事项 Checklist

- [ ] `model.config._attn_implementation = "eager"`(encoder + decoder 导出前都设置)
- [ ] KV dummy tensor 用列表推导式(不能用 `[t]*10`)
- [ ] decoder MLP 用显式 Slice 替代 chunk(避免 Split num_outputs)
- [ ] encoder Conv 补 kernel_shape(`fix_conv_kernel_shape`)
- [ ] opset 统一为 17(onnxsim 后检查 `model.opset_import[0].version`)
- [ ] 10 层 sliding window mask 用 transformers 语义预计算(`make_sliding_mask`), 不用参考实现版
- [ ] causal mask 有效位 = `(arange < cache_len) | (arange == max_dec_len)`
- [ ] 方案 B: 导出 `log_k` 资产给 C++(`encoder.embedder.comp.log_k`)
- [ ] onnxsim 后 `onnx.checker.check_model` 通过
- [ ] bmodel 转换: `--chip bm1684x --disable_layer_group`(decoder 多输入输出); `bmrt_test` 确认输入 23 / 输出 21
- [ ] 精度对比: encoder 输出 vs HF(baseline debug npy, 注意 HF 循环的 enc_out 需每步 clone); decoder 逐步 logits vs 修正后的 HF 循环
- [ ] 1.wav(16.71s, T_enc=836)超过固定 shape, 固定 shape 验证用 0.wav/8k.wav 或截断

---

## 9. 与参考方案差异对照

| 项 | 参考实现(NPU 5.3) | 本方案(BM1684X) | 原因 |
|----|----------------|------------------|------|
| CMVN+Asinh | CPU | CPU(方案 B 同款) | 两边都缺算子/怕精度: 参考实现缺 Log, BM1684X 是 F16 精度风险 |
| encoder sliding mask | 预计算 buffer | 预计算 buffer | 相同; 但边界语义差 1 位(见 6.4), 必须用 transformers 版 |
| embed_tokens | CPU 查表 | **模型内 Gather** | BM1684X 支持 Gather |
| encoder pos_emb+proj | CPU | **模型内**(常量折叠 Gather + MatMul) | BM1684X 支持 |
| RoPE cos/sin | CPU 预计算查表 | **模型内 Cos/Sin 计算**(回退 CPU 传表) | BM1684X 支持 Cos/Sin |
| causal mask | CPU 生成 | **模型内 Less/Or/Where 生成** | BM1684X 支持 |
| KV 布局 | 堆叠 [10,1,64,512] 单输入 | **逐层 past_k_i [1,128,512] ×10** | whisper 风格, C++ 指针直通; 与常驻设备缓存优化兼容 |
| max_dec_len | 64 | **128** | 10s 目标, 留余量 |
| KV 内容 | 旋转后 K | 旋转后 K | 与 transformers 一致 |

---

## 10. 验证产物与复现

- 验证脚本: `/tmp/moonshine_analysis/verify_encoder.py`(mask 语义 + 数值 + 导出 + 扫描 + ORT)、`verify_decoder.py`(41 步自回归 + 导出 + 扫描 + ORT 逐步)、`precision.py`(asinh/CMVN/RoPE 精度)、`fix_onnx.py`(Split 修复)
- 导出产物: `/tmp/moonshine_analysis/onnx/moonshine_encoder_A_sim.onnx`(207MB)、`moonshine_encoder_B_sim.onnx`(207MB)、`moonshine_decoder.onnx`(含 _sim)
- 数值基准: 0.wav 循环填充到 160000 samples、全 1 attention_mask; HF encoder 输出 [1,500,620] 为 encoder ground truth; 修正后 HF decoder 循环 41 步为 decoder ground truth
