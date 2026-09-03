# Moonshine streaming-small BM1684X bmodel 转换报告

**生成日期**: 2026-08-01
**芯片**: BM1684X, **TPU-MLIR**: v1.28.1-20260429 (Docker sophgo/tpuc_dev:latest, whl: 0_Toolkits/)
**模型**: moonshine-streaming-small (140.1M, encoder-decoder ASR, 10 层)
**固定输入**: 10s 音频 → C++ 做 CMVN+Asinh → x_frames [1,2000,80] → encoder_out [1,500,620]

---

## 1. 生成的 bmodel 文件(models/BM1684X/)

| 文件 | 大小 | 精度 |
|------|------|------|
| moonshine_encoder_F32.bmodel | 211 MB | FP32 |
| moonshine_decoder_F32.bmodel | 337 MB | FP32 |
| moonshine_encoder_F16.bmodel | 109 MB | FP16 |
| moonshine_decoder_F16.bmodel | 171 MB | FP16 |

转换脚本: `python/gen_bmodel.sh`(仓库根目录执行, 参数 F32/F16)
导出脚本: `python/export_onnx.py`;ONNX 验证: `python/test/test_onnx.py`
ONNX 产物: `models/onnx/moonshine_encoder_sim.onnx`(207MB) / `moonshine_decoder_sim.onnx`(347MB), 均 opset 17

---

## 2. bmodel 接口规格(板卡 bmrt_test 实测)

### Encoder(`moonshine_encoder`, 1 入 1 出, 无需 --disable_layer_group)

| 名称 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `x_frames` | [1, 2000, 80] | FLOAT32 | 方案 B: C++ 侧做分帧+CMVN+Asinh |
| `encoder_out_Mul` | [1, 500, 620] | FLOAT32 | 输出(TPU-MLIR 给名字加 `_Mul` 后缀, C++ 按 index 取) |

### Decoder(`moonshine_decoder`, 23 入 21 出, 转换必须 --disable_layer_group)

**输入**(顺序与 ONNX 一致; token/cache_len 由 int64 自动降为 **INT32**, 板卡实测 dtype=INT32):

| 名称 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `token` | [1, 1] | INT32 | 当前 token id(模型内 Gather 查 embed) |
| `encoder_out` | [1, 500, 620] | FLOAT32 | 模型内加 pos_emb + proj 620→512 后做 cross-attn |
| `cache_len` | [1] | INT32 | 当前已填充 KV 长度 = 当前 token 位置 |
| `past_k_0..9` | [1, 128, 512] | FLOAT32 | 各层 past K(**旋转后**, 与 transformers DynamicCache 一致) |
| `past_v_0..9` | [1, 128, 512] | FLOAT32 | 各层 past V |

**输出**(21 个, 实际名带后缀如 `logits_MatMul`、`new_k_0_Reshape`, C++ 按 index):

| 名称 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `logits` | [1, 1, 32768] | FLOAT32 | 下一步 token 分布 |
| `new_k_0..9` | [1, 1, 512] | FLOAT32 | 本步各层 K(旋转后), C++ 写入 cache 位置 cache_len |
| `new_v_0..9` | [1, 1, 512] | FLOAT32 | 本步各层 V |

**C++ 解码循环要点**:
- cache 初始全 0;第 t 步把 new_k_i/new_v_i 写入位置 t;mask 有效位 = 位置 `0..t-1` + 尾部当前槽 `128`, 其余 -1e9
- 解码: sos=1, eos=2, greedy argmax;max_dec_len=128(10s 实际 ~27-70 步)
- 每步上传 20 个 cache tensor(128×512×4B=262KB × 20 = 5.2MB/步);可参照 ChatTTS 经验用 `bmrt_tensor_with_device` 常驻设备缓存优化(ONNX 接口不变)
- 跨层 cross-attn KV 每步重算(无缓存, 500×512 每层每步 ~26 MFLOP)

---

## 3. C++ 特征提取公式(方案 B, 与 HF bit 级一致, 验证 max_abs=2.4e-07)

```text
1. 音频 16kHz mono float32, 尾部补零到 160000 samples(10s)
2. 分帧 reshape [2000, 80](frame_len=80 = 5ms); 不足 2000 帧尾部补零帧(全 0 → CMVN 后 ≈0)
3. CMVN: mean = x.mean(-1); centered = x - mean
         rms = sqrt(centered^2.mean(-1) + 1e-6); normed = centered / rms
4. Asinh: x_frames = asinh(exp(log_k) * normed)   // cmath std::asinh 直接可用
LOG_K = -0.4875200987   (k = exp(log_k) ≈ 0.6141; 0.75 只是 k_init, 勿用!)
```

资产文件: `models/log_k.npy` + `models/log_k.txt`(export_onnx.py 生成);eps=1e-6 硬编码。

---

## 4. 精度验证结果

### 4.1 ONNX 级(ORT vs HF 修正循环, 0.wav/8k.wav 尾部补零到 10s)

| 检查项 | 0.wav | 8k.wav |
|--------|-------|--------|
| x_frames numpy(C++公式) vs HF embedder | max_abs 2.4e-7 | 5.6e-7 |
| encoder ORT vs HF(同输入) | max_abs 3.0e-5 | 1.4e-5 |
| decoder ORT vs HF token | 27/27 一致 | 19/19 一致 |
| decoder ORT vs HF logits | max_abs 2.3e-5 | 2.7e-5 |

### 4.2 bmodel 级(板卡 BM1684X 实测, 0.wav, C++ bmruntime 全 27 步自回归)

| 精度 | encoder vs ORT max_abs | decoder logits max_abs | token 一致率 | argmax 不一致步数 |
|------|------------------------|------------------------|--------------|-------------------|
| F32 | 2.65e-5 | 3.24e-5 | 27/27 (100%) | 0 |
| F16 | 3.27e-2 (rel 5.7e-3) | 4.34e-2 | 27/27 (100%) | 0 |

**结论**: F32/F16 转写与 ONNX/HF 完全一致。F16 的 Sin/Cos(LUT) 精度担忧经实测不成立,
不需要 C++ 预计算 cos/sin 表回退方案。板卡 bmrt_test 冒烟通过(encoder 128ms/次)。

板卡验证工具(参考实现, 可作 C++ 移植起点): `/tmp/bmverify/`
- `bm_verify_moonshine.cpp`: bmrt 加载 2 个 bmodel + 逐步贪心(含 BM_INT32 上传、KV 槽位写入)
- `make_ref.py` / `compare.py`: 本地参考与对比

---

## 5. 算子处理记录(实测遇到并解决)

| 问题 | 现象 | 解决方案 |
|------|------|----------|
| **Or 算子无法转换** | TPU-MLIR 1.28.1 `convert_or_op` 用 getOperand 只认节点输出; 常量输入(onnxsim 提升的 initializer)报 `operand eq not found`(**Constant 节点也不行**, convert_constant_op 只 addWeight 不注册 operand) | **wrapper 层消除 Or**: causal mask 改嵌套 Where(`lt|eq` → `where(lt, 0, where(eq, 0, -1e9))`, 布尔语义逐位等价)。export_onnx.py 已加 Or 节点断言防回归 |
| Conv 缺 kernel_shape | torch 2.11 导出 Conv 不写 kernel_shape, model_transform 报错 | `fix_conv_kernel_shape` 从权重 dims 推断补上(encoder 2 个 Conv) |
| opset 升 18 | torch 2.11/onnxsim 结果可能是 18(onnx 1.21 转换器无 Pad 17 适配器) | 导出后统一 `opset_import[0].version = 17`(LayerNormalization 需 ≥17) |
| Split num_outputs 与 onnxsim 0.6.3 冲突 | torch chunk 导出 Split 带 num_outputs, onnxsim 崩溃 | MLP GLU 用显式 Slice `y[..., :2048] / y[..., 2048:]` |
| HF decoder 原地修改 encoder_hidden_states | `encoder_hidden_states += pos_emb` 原地累加(28 步后 enc 被加 28 次 pos_emb) | 验证/基准每步传 `enc_out.clone()`;bmodel 纯函数无此问题 |
| sliding window mask 边界 | 参考实现与 transformers 差 1 位 | 用 transformers 语义 `(dist>=0 & dist<left) | (dist<0 & -dist<right)`, 与 HF 逐元素 0 差异 |
| CausalConv1d 双重 padding | HF 模块内部自带 F.pad(4,0) | wrapper 外层不重复 pad(重复输出 503 帧) |
| int64 输入 | bmruntime 无 INT64 | TPU-MLIR 编译自动降 int32, C++ 上传 cast 到 int32(板卡实测 dtype=INT32) |
| KV dummy 共享引用 | `[t]*n` 共享同一 tensor | 列表推导式 `[torch.zeros(...) for _ in range(n)]` |
| onnx checker 警告 | 容器 onnx 1.14.1 checker 只支持 ir_version 9 | 良性警告, 不影响转换 |

---

## 6. C++ 对比文件(python/test/outputs/debug/)

| 文件 | Shape | 说明 |
|------|-------|------|
| `{0,1,8k}_input_values.npy` | [1,N] f32 | 原始音频(未填充) |
| `{0,1,8k}_encoder_output.npy` | [1,T_enc,620] f32 | HF encoder 输出(**未填充**音频的 ground truth) |
| `{0,1,8k}_decoder_logits.npy` | [steps,32768] f32 | HF 修正循环逐步 logits |
| `{0,1,8k}_decoder_tokens.npy` | [steps] int64 | HF 逐步 token(sos 开头, eos 结尾) |

> **重要 1**: baseline debug npy 已用**修正循环**重新生成(test_pytorch.py 每步
> `enc_out.clone()`)。旧 npy 的 encoder_output 是 `enc + n_steps*pos_emb` 的污染版本
> (全帧差异 ~数十), 若对比出现全帧巨大差异即污染, 需重新生成。
> **重要 2(边界效应)**: 短音频尾部补零到 10s 后, encoder 输出**最后 ~12 帧**
> (相对真实 T_enc)受填充影响(max ~0.29), 且会改变 0.wav 第 15 步 argmax
> (丢一个逗号 token 29892;8k.wav 无影响)。**C++ 与 debug npy 对比只比前 T_enc-12 帧**,
> 或与同条件(填充后)输出对比。1.wav(16.71s, T_enc=836)超过固定上限, 不用。

---

## 7. 转换命令备忘

```bash
# ONNX 导出 + 自检(本地, conda sophon-whisper)
python python/export_onnx.py
python python/test/test_onnx.py          # 0.wav + 8k.wav, 全部 PASS

# bmodel 转换(仓库根目录; F32 先, F16 后)
docker run --rm \
  -v $(pwd)/moonshine:/workspace \
  -v $(pwd)/0_Toolkits:/toolkits \
  sophgo/tpuc_dev:latest bash /workspace/python/gen_bmodel.sh F32
```

---

## 8. 环境信息

- 本地: conda `sophon-whisper`(torch 2.11.0+cpu, onnx 1.21.0, onnxsim 0.6.3, transformers 5.14.1)
- Docker: sophgo/tpuc_dev:latest(容器内 onnx 1.14.1, 经 pip 装 tpu_mlir whl)
- 板卡: BM1684X, libsophon 0.5.1(bmrt_tensor 收 bm_shape_t 结构体, 与新版 int* 签名不同), gcc 9.3
- 原始分析文档: `.context/operator_analysis.md`(wrapper 代码/坑/checklist)
