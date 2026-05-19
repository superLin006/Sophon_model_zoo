# Sophon BM1684X 移植知识库

**目标芯片**: BM1684X (SDK-23.09 LTS SP4)  
**转换工具**: TPU-MLIR v1.28.1 (Docker: `sophgo/tpuc_dev:latest`)  
**更新日期**: 2026-05-19

---

## 一、工具链与转换流程

### 转换链路

```
PyTorch (.pt)
    ↓  export_onnx.py
ONNX (.onnx)
    ↓  model_transform.py  [Docker: sophgo/tpuc_dev:latest]
.mlir
    ↓  model_deploy.py --chip bm1684x --quantize F16|INT4
.bmodel
    ↓  交叉编译 C++ 推理程序  [Docker: sophon-cross-build]
aarch64 可执行文件  →  scp 到 BM1684X 板卡
```

### 支持的精度

| 精度 | 说明 |
|------|------|
| FP16 | 推荐，速度快，精度损失极小 |
| INT8 | 需校准数据集（`run_calibration.py`） |
| INT4 | 大模型首选，显著压缩体积（ChatTTS GPT 使用此精度） |
| FP32 | 调试用，生产环境不推荐 |

### 标准转换命令

```bash
# Step 1: ONNX → MLIR
model_transform.py \
    --model_name mymodel \
    --model_def  mymodel.onnx \
    --input_shapes [[1,3,224,224]] \
    --mlir mymodel.mlir

# Step 2: MLIR → bmodel (F16)
model_deploy.py \
    --mlir mymodel.mlir \
    --quantize F16 \
    --chip bm1684x \
    --model mymodel_F16.bmodel
```

---

## 二、通用问题与解决方案

### 1. Conv 节点缺少 `kernel_shape` 属性

**触发**：新版 `torch.onnx.export`（opset 17）导出的 Conv 节点不写 `kernel_shape`，`model_transform.py` 报错。

**修复**：导出后补全属性：
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

### 2. KV Cache 模型：dummy tensor 不能用 `* n` 创建

**触发**：`[tensor] * n_layer` 是同一对象的 n 个引用，ONNX tracer 将其视为同一张量，constant folding 消除后续层的 KV 输入，bmodel 输入数量比预期少。

**修复**：必须用列表推导式，每层独立创建：
```python
# 错误
dummy_k = [torch.zeros(1, 448, 512)] * 6
# 正确
dummy_k = [torch.zeros(1, 448, 512) for _ in range(6)]
```

### 3. 复杂图（多输入输出）需加 `--disable_layer_group`

**触发**：输入输出数量多时（如 Whisper Decoder 28 输入 25 输出），layer group 优化不稳定，报错或生成异常 bmodel。

**修复**：
```bash
model_deploy.py --mlir ... --disable_layer_group --chip bm1684x ...
```

### 4. 交叉编译产物 glibc 版本过高

**触发**：WSL 原生 gcc（15.x）编译的 aarch64 二进制在板卡（Ubuntu 20.04, glibc 2.31）上报 `GLIBC_2.34 not found`。

**修复**：使用 Ubuntu 20.04 Docker 镜像（gcc 9.4）做交叉编译：
```dockerfile
FROM ubuntu:20.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    cmake g++-aarch64-linux-gnu gcc-aarch64-linux-gnu make
```

### 5. 确认板卡型号

```bash
bm-smi   # 输出含 "1684X-SOC" 则为 BM1684X
```

编译 bmodel 必须使用 `--chip bm1684x`（非 `bm1684`），否则运行时报架构不匹配。

### 6. bmruntime 没有 BM_INT64 类型

**原因**：BM1684X SDK 只有 BM_INT8/INT16/INT32/INT4，没有 INT64。TPU-MLIR 编译 ONNX 的 int64 输入时自动降为 int32。

**修复**：C++ 中将数据 cast 到 int32 再上传：
```cpp
std::vector<int32_t> buf(n);
for (int i = 0; i < n; ++i) buf[i] = (int32_t)data[i];
t.dtype = BM_INT32;
```

---

## 三、bmruntime C API 关键用法

### 基本推理流程

```cpp
bm_handle_t hdl;
bm_dev_request(&hdl, 0);

void* rt = bmrt_create(hdl);
bmrt_load_bmodel(rt, "model.bmodel");

const bm_net_info_t* net = bmrt_get_network_info(rt, "network_name");

// 分配输入 tensor 并上传数据
bm_tensor_t in_t;
bmrt_tensor(&in_t, rt, net->input_dtypes[0], net->stages[0].input_shapes[0]);
bm_memcpy_s2d(hdl, in_t.device_mem, host_data);

// 分配输出 tensor
bm_tensor_t out_t;
bmrt_tensor(&out_t, rt, net->output_dtypes[0], net->stages[0].output_shapes[0]);

// 推理
bmrt_launch_tensor_ex(rt, net->name, &in_t, 1, &out_t, 1, true, false);
bm_thread_sync(hdl);

// 下载结果
bm_memcpy_d2s(hdl, host_out, out_t.device_mem);

// 释放
bm_free_device(hdl, in_t.device_mem);
bm_free_device(hdl, out_t.device_mem);
bmrt_destroy(rt);
bm_dev_free(hdl);
```

### 使用预分配设备内存（避免 alloc/free 开销）

```cpp
// 预分配
bm_device_mem_t persistent_mem;
bm_malloc_device_byte(hdl, &persistent_mem, size_bytes);

// 将 tensor 路由到已有设备内存，不分配新内存
bm_tensor_t t;
bmrt_tensor_with_device(&t, persistent_mem, dtype, shape);

// 使用完后不要 free tensor 的 device_mem（它指向 persistent_mem）
// 只在生命周期结束时释放一次
bm_free_device(hdl, persistent_mem);
```

### 关键规则：不能修改 `device_mem.size`

`bm_memcpy_d2s/s2d` 内部用 `device_mem.size` 做 DMA buffer 分配。修改 size 后调用会破坏堆：

```cpp
// 错误：修改 size 导致 malloc_consolidate(): invalid chunk size 崩溃
bm_device_mem_t dm = full_tensor.device_mem;
dm.size = 1536;
bm_memcpy_d2s(hdl, buf, dm);  // 崩溃

// 正确：只修改 device_addr 做偏移（s2d 写方向实测可行，d2s 读方向会崩溃）
bm_device_mem_t slice = full_tensor.device_mem;
slice.u.device.device_addr += offset;
slice.size = slice_bytes;      // s2d 可用，d2s 不可用
bm_memcpy_s2d(hdl, slice, host_ptr);

// 如需读取子区域：下载完整 tensor，在 host 端取子集
bm_memcpy_d2s(hdl, full_host_buf, full_tensor.device_mem);
memcpy(target, full_host_buf + offset, slice_bytes);
```

---

## 四、ChatTTS BM1684X 移植经验

### 推理链路

```
文本
  ↓  BertTokenizer（21178词表）+ 文本归一化
token ids
  ↓  embedding_text [1,1024] int32 → [1,1024,768] f16
  ↓  block_0..19（prefill）         → KV cache + hidden
  ↓  lm_head_code                  → logits [626,4]
  ↓  采样 → vq_codes[4]
  ↓  embedding_code_cache（decode loop）
  ↓  block_cache_0..19             → 更新 KV cache
  ↓  lm_head_code                  → 下一步 logits
hiddens [T, 768] f16
  ↓  DecoderEngine（DVAE）         → mel [100, T*2]
  ↓  VocosEngine                  → mag/x/y
  ↓  iSTFT（CPU, fftw3）           → PCM 24kHz
```

### 移植过程中遇到的 6 个问题

#### 问题 1：tokenizer 词表截断，特殊 token 识别失败

**现象**：`[spk_emb]` 无法识别（`spk_idx=-1`），speaker embedding 未注入，首步即输出 EOS。

**根因**：vocab.txt 只有 21128 个标准 BERT token，ChatTTS 还需要 50 个专用特殊 token（ID 21128~21177）。

**修复**：在 `BertTokenizer` 构造函数末尾手动追加：
```cpp
static const char* kChatTTSTokens[] = {
    "[Sasr]","[Pasr]","[Easr]","[Stts]","[Ptts]","[Etts]",
    "[Sbreak]","[Pbreak]","[Ebreak]","[uv_break]","[v_break]",
    "[lbreak]","[llbreak]","[undefine]","[laugh]","[spk_emb]","[empty_spk]",
    "[music]","[pure]","[break_0]".."[break_7]",
    "[laugh_0]".."[laugh_2]","[oral_0]".."[oral_9]","[speed_0]".."[speed_9]",
};
for (const char* tok : kChatTTSTokens) {
    int id = (int)id2tok_.size();
    tok2id_[tok] = id;
    id2tok_.push_back(tok);
}
```

#### 问题 2：prefill padding 位置产生 NaN，扩散到所有后续层

**现象**：prefill 完成后 logits 全为 NaN，decode 无法采样。

**根因**：bmodel prefill block 静态形状 `[1, 1024, 768]`，实际文本只有 N < 1024 个 token。padding 位置在 attention mask 中全为 `-inf`，导致 softmax 分母为 0 → NaN。NaN 经 LayerNorm 扩散至整行。

**修复**：每个 prefill block 执行后，将 hidden state 的 padding 部分清零：
```cpp
if (i < NUM_LAYERS - 1 && tok_len < SEQLEN) {
    std::vector<uint16_t> h(SEQLEN * HIDDEN_SIZE);
    bm_memcpy_d2s(hdl, h.data(), em_out.device_mem);
    std::fill(h.data() + tok_len * HIDDEN_SIZE, h.data() + SEQLEN * HIDDEN_SIZE, uint16_t(0));
    bm_memcpy_s2d(hdl, em_out.device_mem, h.data());
}
```

#### 问题 3：decode position offset 错误，KV cache 槽位覆盖 prefill

**现象**：音频有内容，但越往后语调越混乱、出现重复。

**根因**：decode step `n` 应使用序列位置 `text_tok_len + n`，但代码直接用 `n`，导致 position embedding 与 prefill 重叠，KV cache 从槽 0 开始写，覆盖 prefill 写入的 KV。

**修复**：
```cpp
// prefill 结束时记录
text_tok_len = tok_len;

// decode 时使用偏移后的位置
int seq_pos = text_tok_len + step;
```

#### 问题 4：`bm_memcpy_d2s` 修改 `device_mem.size` 导致堆崩溃

**现象**：decode 阶段随机崩溃，`malloc_consolidate(): invalid chunk size`。

**根因**：为读取 KV slice（1536 字节）修改了 `dm.size`，`bm_memcpy_d2s` 内部用此值做 DMA buffer 分配，破坏堆结构。

**修复**：始终下载完整 tensor，在 host 端取子集。详见第三节规则。

#### 问题 5：speaker embedding 未 L2 归一化

**现象**：音频有内容，但音色偏差大，音调不稳定。

**根因**：Python 版本在注入前做 `F.normalize(spk, p=2, dim=0)`，C++ 版本缺失此步骤。

**修复**：
```cpp
double norm2 = 0.0;
for (float v : spk_emb_f32) norm2 += (double)v * v;
float scale = (norm2 > 1e-24) ? (float)(1.0 / std::sqrt(norm2)) : 1.0f;
for (size_t i = 0; i < spk_emb_f32.size(); ++i)
    spk_emb[i] = f32_to_f16(spk_emb_f32[i] * scale);
```

#### 问题 6：decode 循环开销过大，RTF 偏高

**根因分析**：

| 阶段 | 原始开销 |
|------|---------|
| 初版 | 每 decode step 上传完整 past_k/v（1.5MB × 2 × 20层）|
| 中间版 | 每个 block 独立 alloc pid/mask（40次/step），hidden output malloc/free（20次/step） |

**修复**：在 `init()` 时预分配所有 decode 专用设备缓冲区，推理时复用：
- `dec_pid_dm`：position id（4 字节），每 step 上传一次，20 个 block 共享
- `dec_mask_dm`：attention mask（2050 字节），每 step 上传一次，20 个 block 共享
- `dec_hid_dm`：hidden state ping-pong 缓冲区（1536 字节）
- `dec_nk_dm` / `dec_nv_dm`：new_k/v scratch（各 1536 字节）
- `dev_k[i]` / `dev_v[i]`：KV cache 常驻 device，prefill 直接用 `bmrt_tensor_with_device` 写入

#### 问题 7：连续流式推理时 BMRT 堆碎片化导致 OOM（约第 58 次崩溃）

**现象**：流式推理约在第 58~60 次请求时崩溃，报 `bm_alloc_gmem failed`，此后所有推理失败，需重启板卡。

**根因**：BMRT slab 分配器不合并已释放块（无 coalesce）。每次 prefill 时：
- `mask [1,1,1024,1024] f16 = 2MB` × 20 个 block 逐次 alloc/free
- `hidden [1,1024,768] f16 = 1.5MB` × 20 个 block 逐次 alloc/free
- Decoder/Vocos 输出 tensor 每次推理 alloc/free

经过约 60 次请求后，device memory 碎片化严重，无法找到足够的连续内存块。

**修复**：双管齐下

**1. 预分配持久设备缓冲区（关键）**：所有引擎在 `init()` 时一次性分配好固定尺寸的 in/out tensor buffer，推理时通过 `bmrt_tensor_with_device` 路由到预分配内存，永不释放。

GPT prefill 使用 ping-pong 缓冲区避免 block 链上的 alloc/free：
```cpp
// Impl 中新增（预分配，一次性）
bm_device_mem_t pf_em_out_dm;   // [1, SEQLEN, HIDDEN] f16，ping
bm_device_mem_t pf_hid_dm;      // [1, SEQLEN, HIDDEN] f16，pong
bm_device_mem_t pf_mask_dm;     // [1, 1, SEQLEN, SEQLEN] f16
bm_device_mem_t pf_lm_in_dm;    // [1, HIDDEN] f16
bm_device_mem_t pf_lm_out_dm;   // [1, NUM_AUDIO_TOKENS, NUM_VQ] f32

// prefill block 链，交替使用 ping/pong
bm_device_mem_t* in_hid  = &pf_em_out_dm;
bm_device_mem_t* out_hid = &pf_hid_dm;
for (int i = 0; i < NUM_LAYERS; ++i) {
    // ... 用 in_hid → out_hid
    std::swap(in_hid, out_hid);
}
```

**2. 共享 `bm_handle_t`**：ChatTTS 创建一个 handle，三个引擎（GPT/Decoder/Vocos）共用，各自创建独立 bmrt（一个 bmrt 不能加载多个独立 bmodel，否则 SIGSEGV）：
```cpp
// chattts.cpp
bm_dev_request(&impl_->shared_hdl, cfg.tpu_id);
impl_->gpt     = std::make_unique<GPTEngine>(path, impl_->shared_hdl, nullptr, gpt_cfg);
impl_->decoder = std::make_unique<DecoderEngine>(path, impl_->shared_hdl, nullptr);
impl_->vocos   = std::make_unique<VocosEngine>(path,   impl_->shared_hdl, nullptr);

// 析构顺序：先销毁引擎，再释放 handle
impl_->gpt.reset();
impl_->decoder.reset();
impl_->vocos.reset();
bm_dev_free(impl_->shared_hdl);
```

共享 handle 使所有引擎的 `bm_malloc_device_byte`（包括 BMRT 内部的 neuron workspace）走同一物理分配器池，某引擎释放的块可立即被其他引擎复用，从根本上消除跨引擎碎片化。

**关键细节**：必须使用 `bmrt_launch_tensor_ex(..., user_mem=true)` 而非 `bmrt_launch_tensor_ex(..., user_mem=false)` 或 `bmrt_launch_tensor`。非 `_ex` 版本和 `user_mem=false` 会忽略 `bmrt_tensor_with_device` 设置的设备内存，内部重新分配输出 tensor。

**验证**：70/70 样本全部通过（含 192 token 的长文本 prefill），原来第 59 条崩溃的样本正常完成。

### 最终性能

**非流式**（一次性推理，RTF 不含首包延迟）：

| 分组 | Python RTF | C++ RTF |
|------|:----------:|:-------:|
| 中文短句（25条） | ~0.75 | 0.592 |
| 中文长文（10条） | ~0.50 | 0.492 |
| 英文短句（25条） | ~0.75 | 0.610 |
| 英文长文（10条） | ~0.50 | 0.493 |
| **整体** | **~0.65** | **0.525** |
| RTF < 1（全部实时） | — | **70/70** |

**流式**（stream_batch=24，含 TTFA 统计）：

| 分组 | RTF | TTFA |
|------|:---:|:----:|
| 中文短句（25条） | 0.622 | 976ms |
| 中文长文（10条） | 0.570 | 978ms |
| 英文短句（25条） | 0.672 | 978ms |
| 英文长文（10条） | 0.573 | 984ms |
| **整体（70条）** | **0.626** | **978ms** |
| RTF < 1（全部实时） | **69/70** | — |

---

## 五、VITS-MeloTTS BM1684X 移植经验

### 推理链路（三段式拆分）

MeloTTS 原始单图含动态 shape 和不支持算子，拆分为三段：

```
Part A（TPU）: enc_p + DP
    输入：x[1,128,int32]、x_lengths[1,int32]、tones[1,128,int32]
    输出：dp_w[1,1,128,f32]、h[1,192,128,f32]、x_mask[1,1,128,f32]

Part B（CPU，~8ms）: MAS（Monotonic Alignment Search）
    输出：attn[T_mel, L]、z_p[1,192,T_mel]

Part C（TPU）: Flow（逆）+ Decoder（HiFi-GAN）
    输入：z_p[1,192,256,f32]（pad 到 T_MEL_FIXED=256）、y_mask[1,1,256,f32]
    输出：audio[1,1,131072,f32]（截取前 T_mel × UPSAMPLE 个有效采样）
```

**性能**：总计 ~320ms，RTF ≈ 0.12（生成 2.7s 音频），比 CPU onnxruntime 快约 20×。

### 移植遇到的问题

#### 问题 1：SDP 含 NonZero 算子（TPU 不支持）

将 SDP 输入替换为全零常量，只保留确定性 DP 分支：
```python
zero = np.zeros(..., dtype=np.float32)
node = onnx.helper.make_node("Constant", [], ["/Add_1_output_0"], value=...)
```

#### 问题 2：Flow 含 RandomNormalLike（TPU 不支持）

noise_scale=0 时噪声贡献为零，直接绕过随机噪声分支，将 noise_scale 常量化为 0。

#### 问题 3：MAS 含 Range 算子（动态 shape，无法静态编译）

将 MAS 保留在 CPU，模型拆成 Part A / CPU / Part C 三段。

#### 问题 4：bmodel 输出步长与实际序列长度不一致导致噪声

**根因**：Part A 输出 h 的存储步长是编译时固定的 `L_MAX=128`，而非运行时的 `seq_len`。在 CPU 端做 `h × attn` 矩阵乘法时若用 `seq_len` 作为步长，每行地址偏移错误，z_p 全部读到错误数据，输出全噪声。

**修复**：步长必须使用 bmodel 固定维度 `L_MAX`，而非实际输入长度：
```cpp
// 错误
matmul_ht(h.data(), seq_len, ...);
// 正确
matmul_ht(h.data(), L_MAX, ...);  // bmodel 输出的行步长是 L_MAX
```

**教训**：操作 bmodel 输出的多维数组时，步长必须用 bmodel 编译时的固定 shape，而非运行时输入长度。

---

## 六、移植检查清单

### ONNX 导出
- [ ] dummy tensor 用列表推导式创建（KV Cache 场景）
- [ ] `onnx.checker.check_model()` 通过
- [ ] Conv 节点 `kernel_shape` 属性已补全
- [ ] onnxsim 简化后验证输出一致

### bmodel 转换
- [ ] `--chip bm1684x`（不是 bm1684）
- [ ] 多输入输出模型加 `--disable_layer_group`
- [ ] `bmrt_test --bmodel xxx.bmodel` 确认输入输出数量正确

### 交叉编译
- [ ] 使用 `sophon-cross-build` 镜像（Ubuntu 20.04 + gcc 9.4）
- [ ] 设置 rpath：`/opt/sophon/libsophon-current/lib`

### 板卡部署
- [ ] `bm-smi` 确认芯片为 BM1684X
- [ ] 模型输入输出数量与导出时一致
- [ ] RTF 统计只计推理时间，不含模型加载
