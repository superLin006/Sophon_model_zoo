# ChatTTS BM1684X C++ 移植 — 问题记录

## 背景

将 ChatTTS 移植到 SOPHON BM1684X（aarch64）上进行纯 C++ 推理，
完全基于 bmruntime C API，不依赖 sophon-sail 或 Python 运行时。

---

## 问题 1：tokenizer 词表不完整，导致特殊 token 识别失败

**现象**
- `[spk_emb]` token 无法被识别，`spk_idx = -1`，speaker embedding 未注入
- 首步 decode 就输出 `codes=[625,625,625,625]`，立即 EOS，无法生成任何音频

**根因**
ChatTTS 的 tokenizer 词表共 21178 个 token：
- 前 21128 个来自标准 BERT vocab.txt
- 后 50 个是 ChatTTS 专用特殊 token（`[spk_emb]`、`[Stts]`、`[Ptts]`、`[speed_0]`~`[speed_9]` 等）

代码只读了 `vocab.txt`（21128 条），导致所有特殊 token 全部映射为 `[UNK]`。

**修复**
在 `BertTokenizer` 构造函数中，读完 vocab.txt 后手动追加 50 个特殊 token：

```cpp
static const char* kChatTTSTokens[] = {
    "[Sasr]","[Pasr]","[Easr]","[Stts]","[Ptts]","[Etts]",
    "[Sbreak]","[Pbreak]","[Ebreak]","[uv_break]","[v_break]",
    "[lbreak]","[llbreak]","[undefine]","[laugh]","[spk_emb]","[empty_spk]",
    "[music]","[pure]","[break_0]",... "[speed_9]",
};
for (const char* tok : kChatTTSTokens) {
    int id = (int)id2tok_.size();
    tok2id_[tok] = id;
    id2tok_.push_back(tok);
}
```

追加后：`[spk_emb]=21143`，`[Stts]=21131`，`[Ptts]=21132`，`[speed_5]=21173`。

---

## 问题 2：prefill 后 hidden state 含 NaN，导致所有 logits 为 NaN

**现象**
- prefill 完成后 lm_head 输出 `logits = nan nan nan nan`
- decode 无法采样，直接崩溃或输出垃圾

**根因**
bmodel 的 prefill block 是静态形状：输入 `[1, SEQLEN=1024, 768]`，
但实际文本只有 N < 1024 个 token，其余位置 padding 为 0。

padding 位置在 attention mask 中被设为 `-inf`，导致：
- softmax(全-inf) → NaN（所有位置都被 mask 掉，分母为 0）
- 该 token 位置的 hidden state 变成 NaN
- 下一个 block 的 LayerNorm 遇到 NaN → 整行扩散

**修复**
每个 prefill block 执行完后，把 hidden state 下载到 host，
将 `[tok_len .. SEQLEN-1]` 位置清零，再上传回 device：

```cpp
if (i < NUM_LAYERS - 1 && pad_elems > 0) {
    std::vector<uint16_t> h(SEQLEN * HIDDEN_SIZE);
    bm_memcpy_d2s(hdl, h.data(), em_out[0].device_mem);
    std::fill(h.data() + tok_len * HIDDEN_SIZE, h.data() + SEQLEN * HIDDEN_SIZE, uint16_t(0));
    bm_memcpy_s2d(hdl, em_out[0].device_mem, h.data());
}
```

---

## 问题 3：decode position offset 错误，导致 KV cache 位置混乱

**现象**
- 音频长度异常，内容重复或语调混乱
- decode 初期正常，越往后越出错

**根因**
decode step `n` 对应的序列位置应为 `text_tok_len + n`（decode token 紧接在 prefill 文本之后），
但代码直接用 `n` 作为 position，导致 decode token 的 position embedding 与 prefill 重叠，
同时 KV cache 写入的槽位也从 0 开始，覆盖了 prefill 写入的 KV。

**修复**
在 prefill 结束时记录 `text_tok_len`，decode 时用偏移后的位置：

```cpp
// prefill 结束时
text_tok_len = tok_len;

// decode step n
int seq_pos = text_tok_len + step;
```

---

## 问题 4：`bm_memcpy_d2s` 修改 `device_mem.size` 导致堆损坏崩溃

**现象**
- 程序在 decode 阶段随机崩溃，错误信息：
  `malloc_consolidate(): invalid chunk size`

**根因**
为了只读取 KV cache 中某一个 token 的切片（`ATTEN_HEAD * ATTEN_DIM * 2 = 1536` 字节），
尝试通过修改 `bm_device_mem_t.size` 来缩小传输范围：

```cpp
bm_device_mem_t dm = dev_k[i];
dm.size = 1536;  // 错误！
bm_memcpy_d2s(hdl, buf.data(), dm);  // 崩溃
```

`bm_memcpy_d2s/s2d` 内部用 `device_mem.size` 做 DMA buffer 分配，
设为小值会破坏堆结构。

**修复**
始终用完整 tensor 大小做传输，不修改 `.size` 字段。
如需写入切片，只修改 `.device_addr`（偏移地址），`.size` 保持原始值（`s2d` 写方向实测可行）；
如需读取切片，先下载整个 tensor 再在 host 端取子集。

---

## 问题 5：speaker embedding 未 L2 归一化

**现象**
- 音频有内容，但音色偏差大，音调不稳定

**根因**
Python 版本在注入 speaker embedding 前会做 `F.normalize(spk, p=2, dim=0)`，
C++ 版本直接注入原始 float32 值，数值范围与模型训练时不一致。

**修复**
在 `set_speaker()` 中注入前先做 L2 归一化：

```cpp
double norm2 = 0.0;
for (float v : spk_emb_f32) norm2 += (double)v * v;
float scale = (norm2 > 1e-24) ? (float)(1.0 / std::sqrt(norm2)) : 1.0f;
for (size_t i = 0; i < spk_emb_f32.size(); ++i)
    impl_->spk_emb[i] = f32_to_f16(spk_emb_f32[i] * scale);
```

---

## 问题 6：RTF 偏高（初版 3.7x，优化前 1.36x）

**现象**
- 初版 RTF ~3.7，远超实时
- 即使 KV cache 移到 device 后，短文本 RTF 仍 ~1.36

**根因分析**

| 阶段 | 原因 |
|------|------|
| 初版高 RTF | 每个 decode step 把完整 past_k/v（1.5MB × 2 × 20层）上传 device |
| 优化后短文本偏高 | 每个 block 的 pid/mask 各自 alloc+upload（40次/step），hidden output 各自 malloc/free（20次/step） |

**修复**
1. KV cache 常驻 device，用 `bm_malloc_device_byte` + `bmrt_tensor_with_device` 直接路由输出
2. pre-allocate decode 专用缓冲区（pid、mask、hidden ping-pong、new_k/v scratch），在 init 时一次性分配，decode 时复用

**结果**

| 文本类型 | 优化前 RTF | 优化后 RTF |
|----------|-----------|-----------|
| 短文本（~14 tokens） | 1.36 | 0.75 |
| 长文本（~39 tokens） | 0.66 | 0.54 |
| 70样本整体（benchmark） | — | 0.533 |

---

## 最终性能（70样本 benchmark，对比 Python 参考）

| 分组 | Python RTF | C++ RTF |
|------|:----------:|:-------:|
| 中文短句（25条） | ~0.75 | 0.606 |
| 中文长文（10条） | ~0.50 | 0.497 |
| 英文短句（25条） | ~0.75 | 0.623 |
| 英文长文（10条） | ~0.50 | 0.497 |
| **整体** | **~0.65** | **0.533** |
| RTF < 1（实时） | — | **70/70 (100%)** |
