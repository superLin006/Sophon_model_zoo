# CP（code_predictor）调试记录（2026-08-13 下午）

## 最终结论：ONNX 5 层链 vs f32 PyTorch 完全一致（15/15 步 PASS）

```
seq [1995, 1642, 519, 22, 793, 1485, 422, 1902, 1728, 1446, 743, 1377, 914, 344, 1772, 125]
```

## 三个坑（均已定位/解决）

### 坑 1：mask 维度 —— 真实 generate 传 2D mask，PyTorch 参考必须用 2D
- 真实 generate 中 CP 收到 2D mask `[1, g+2]`（ones），内部 create_causal_mask 转 4D `[1,1,1,g+2]`
- 之前调试脚本用自建 4D mask 模拟，虽数学等价，但参考值以 `manual_cp_verify.py`（2D mask + 原始 forward）为准
- 验证：hook 实测真实 generate 的嵌套码流与 manual_cp_verify（f32）完全一致

### 坑 2：baseline 最后一位 1177 vs f32 的 125 —— dtype 差异，非 bug
- baseline（greedy_codes.npy）是 **bf16**（GPU）跑的 → 最后一步 argmax=1177
- f32（CPU）路径最后一步 argmax=125（`manual_cp_verify_bf16.py` 复现 1177 证实）
- **ONNX/bmodel 是 f32，正确对比基准是 125**，不是 baseline 码流

### 坑 3：cp_k{i}.npy 坏数据 —— ONNX/wrapper 链与真实分叉的根因
- 旧 `test/outputs/debug/cp_k{i}.npy`（14:13 生成）与真实 prefill KV **不一致**（k diff 0.5~1.2）
- 导致 wrapper 链/ONNX 链从错误起点出发，g=6 起 argmax 翻转（1579 vs 422）
- 修复：`onnx_cp_sim.py` 改为**内联跑 PyTorch prefill 拿每层 KV** 作为起点 → 15/15 全对
- 顺带验证：wrapper 链逻辑本身正确（用真实起点后与真实路径一致）

## 有效验证脚本（自包含）

| 脚本 | 作用 |
|---|---|
| `manual_cp_verify.py` | f32 真实路径参考（2D mask），期望最后一位 125 |
| `manual_cp_verify_bf16.py` | bf16 复现 baseline（最后一位 1177） |
| `onnx_cp_sim.py` | 完整 5 层 ONNX 链（内联真实 prefill KV），PASS |
| `cp_chain_debug.py` | wrapper vs 真实逐层对比（定位用） |
| `inspect_cp_full.py` | hook 真实 generate 每步入参/输出（验证用） |

## 下一步

- 重新 dump 正确的 5 层 prefill KV（供 C++/板卡端对比）→ `dump_cp_ref.py` 已改
- C++ 端 CP 实现对照 `onnx_cp_sim.py` 的 16 槽 KV + mask 构造（槽 0..g 可见 + 槽 15 = 当前）
- bmodel（qwen3_tts_cp_F16.bmodel）数值对比：talker 侧同思路（真实 KV 起点）

## 2026-08-13 根因确认：head-major 布局错位（已修复，完整链路跑通）

### 症状
- 独立 `bm_test_cp`（Python dump 输入）正确；集成 C++ `cp_block_cache_0` 输出错误（h[0..3] 差 1.31）
- CP codes16[2] = 1112（正确 519）
- `--cp_only`（跳过 talker）同样错误 → 排除 talker 污染
- prefill 后 dump cp_k_dev_[0]：槽 2..15 有 1792 非零（应全 0），且全部集中在 head0

### 根因（两处）
1. **KV cache 写回 memcpy 布局错位**：cache 布局 [8,16,128]（CP）/ [8,256,128]（talker）是 head-major ——
   head0 占前 8192 字节。直接 `memcpy(cache, kv, N)` 把 8 个 head 的数据全部塞进 head0 的 16 槽，
   head1..7 变 0。**必须逐 head 拷贝**：
   ```cpp
   for (int h = 0; h < KV_HEADS; h++)
       memcpy(cache + (h*16 + slot)*128, nk + h*128, 128*4);
   ```
   - CP prefill 写回（[8,2,128] → [8,16,128] 槽0..1）
   - CP decode 写回（[8,1,128] → [8,16,128] 槽pp）
   - talker decode 写回（[8,1,128] → [8,256,128] 槽pos）
2. **CP decode 输入 15 槽转换**：cache 网络输入 history 是 [1,8,15,128]（每 head 15 槽），
   读回的 [8,16,128] 不能直接 memcpy 前 61440B（head-major 下前 61440B = head0..6 + head7 前半）。
   必须逐 head 拷前 15 槽。

### 附带修复
- `cp_embedding_14` 缺失：模型有 15 个 embedding（0..14），talker decode 输入需要
  `Σ codec_embed(code0) + Σ_{g=1..15} embedding_{g-1}(code_g)`（官方源码 codec_hiddens.sum），
  而 export_talker.py 只导出了 14 个（n_emb = n_heads-1）→ C++ 访问 nullptr 段错误。
  已重新 combine 全部 40 个网络（5 prefill f32 + 5 cache f32 + 15 lm_head + 15 embedding）。
- 板卡实测 codes16 = [1995, 1642, 519, 22, 793, 1485, 422, 1902, 1728, 1446, 743, 1377, 914, 344, 1772, 125] 与 f32 参考逐 token 一致

### 验证结果
- 完整句子（今天天气真好…）77 帧 → 6.08s 音频（RTF 4.37）
- 前 6 帧 code0 与 baseline(bf16) 一致 [1995,215,212,1181,462,462]，step6 起分叉（talker F16 vs baseline bf16 精度累积，正常）

## 2026-08-13 性能优化：RTF 4.37 → 3.23（+26%），量化 CP 验证不可用

### 优化项（全部保留在最终版）
1. **KV cache host 镜像**：decode 输入直接取自 host 镜像（免 d2s 读回设备），
   输出 d2s 4KB 后逐 head 更新镜像 —— 设备上不再存常驻 KV buffer（删除 56MB 设备内存）
2. **CP cache 改 15 槽** [8,15,128]：与网络输入 shape 一致，免 16→15 转换
3. **embedding 缓存**：CP 生成的 code0+15 个 code 的 embedding 存入 last_cp_embs_，
   talker_decode_step 直接求和（省 16 次 launch/帧）
4. **talker/CP decode 批处理**：28 层 / 5 层连续 launch（hidden 设备内存直连），一次 sync
5. **TTS_PAD embedding 缓存**

### 量化实验结论（不可用，证据链已闭环）
- **F16 cache**：12 帧 codes16 全对 + CP 132→73ms，但长句生成退化（256 帧不 EOS、code0 卡 617 循环）
- **W8BF16 cache**：CP 132→58ms，最后 1 code 错（901 vs 125），长句同样退化
- **W8BF16 + 最后一层 f32**：误差仍存在（前 4 层已累积）
- **BF16 cache**（round-to-nearest-even 转换）：12 帧 codes16 从第 6 个错（f32 参考 ...422 1902 1728... 变 ...1579 1918 564...），长句退化（code0 卡 668 循环不 EOS）；RTF 2.83
- **BF16 交叉验证（排除环节 bug）**：
  - CPU bf16 模拟（torch bf16 权重 + f32 last_hidden）短句"你好"→ 同样从第 6 个错，与板卡逐位一致 → 板卡 bf16 网络计算精度 = CPU fp32 内部计算 + bf16 输入/权重，无环节 bug
  - 板卡 F16 talker 与 CPU f32 talker 的 last_hidden max diff 仅 0.06（同输入），talker F16 误差不是元凶
  - 长句 last_hidden + bf16 CP 模拟全对 → 量化误差靠近 argmax 边界，轨迹相关（短句敏感/长句不敏感）
- **结论：CP 必须全 F32**。量化误差经 15 步 × 5 层嵌套自回归逐帧累积（384 帧 × 15 步 × 5 层），一旦 argmax 翻转即进入错误轨迹 → 循环退化。CP 是 5 层小模型，无大模型冗余稀释；Qwen3-ASR/HY-MT 无嵌套自回归，故 w8bf16 可用而 CP 不行

### 根因升级（2026-08-13 深挖：编译器层内激活精度，非模型不可量化）
- **CPU 全模型 bf16 闭环（torch eager）长句"今天天气真好…"= 54 帧自然 EOS 完全正常** → bf16 权重+输入本身可量化
- **板卡 W8BF16/BF16 同输入长句 = 234+ 帧 code0 循环退化**；逐帧 last_hidden 差分：**frame 0 maxdiff 即 17.35**（非缓慢累积——CP codes16 从第 6 个错 → talker 输入不同 → 第一帧就分叉）
- **根因**：CPU torch bf16 模型层内是 fp32 计算（eager），仅层间激活 bf16；TPU 量化网络**每个算子中间激活存回量化 dtype**（层内多次 bf16/fp16 往返）→ CP 的平坦 argmax 一翻边界即进入错误轨迹
- W8BF16 + `--high_precision`（norm 等强制 fp32）仍从第 6 个错 → norm 非主误差源，matmul 层内激活才是 → **编译器层面无法修复**
- 证据链：CPU bf16 闭环正常（模型可量化）+ TPU 量化 12 帧错/长句崩（编译器精度限制）→ 非环节 bug、非模型不可量化
- **因此：CP 保持 F32；优化方向改为 talker（28 层大模型）W8BF16**（大模型 logits 尖，w8bf16 经验适用；CPU 闭环证明 talker bf16 权重输出误差 CP 可容错）

### 最终指标（allf32，76 帧完整句子）
- RTF 3.23（旧版 4.37），每帧 ~258ms：talker 88ms（F16）+ CP 132ms（f32）+ 其他 ~40ms
- wav 数值与优化前完全一致（零精度损失）

### 踩坑
- CP 批处理首版 pp=15 越界（15 槽镜像，g=14 时 pp=15 写越界 → free(): invalid size）→ 最后一轮不写
- talker pos ≥ SEQLEN 需防护（256 帧上限，超出即 force stop）
- model_transform --model_name 决定 bmodel 网络名（带后缀会导致运行时找不到网络）
- F16 网络的 input_dtypes 仍是 F32（TPU-MLIR F16 只量化权重，激活 f32）—— 与 bf16 网络不同

### 最终方案（2026-08-13 收尾）
- **talker W8BF16 + SEQLEN=128 + CP F32 = RTF 2.51**（4.37 → 3.23 → 2.51）
- SEQLEN 256→128：attention 减半 + KV DMA 减半（RTF 2.85→2.51）；128 shape 编译本身无问题（早先退化是 C++ merged 回退单层 bug 误判）
- **talker 量化同样受限**：F16/W8（256/128）在部分长句退化（CPU bf16 闭环正常 99 帧 vs TPU 120+ 帧翻车）；F16+high_precision 更差（负优化，数值路径改变引入新误差）——与 CP 同构：TPU 层内激活精度（7/10 位）不足，非 bug、非模型不可量化
- **C++ 关键修复**（本日）：prefill 批处理缺 resize → malloc 崩；bmrt_get_network_info 返回 nullptr 需判空；merged/seg 回退单层有数值 bug（恢复原 5 层批处理）；padding hidden/KV 每层清零（NaN）；embedding pad 自适应网络 shape
- 板卡最终：codec_decoder.bmodel + qwen3_tts_cp_allf32.bmodel + qwen3_tts_talker_w8bf16.bmodel（128 版）

## 2026-08-14：KV 设备常驻优化（对齐 v1 方案第 2/3 点）

- **改动**：talker/CP 的 KV cache 从 host 镜像改为设备常驻（init 分配 28×2×512KB + 5×2×61KB 设备 buffer；decode 输入 `bmrt_tensor_with_device` 零拷贝直连；新 KV `bm_memcpy_d2d_byte` 逐 head 写槽；prefill 输出 s2d 一次写入设备）。消除每帧 ~28MB KV s2d + host 镜像 memcpy
- **精度**：cp_only codes16 与 F32 参考逐 token 一致；短句/长句/疑问/欢迎语正常 EOS
- **RTF**：长句 2.513 → **2.358**（每帧省 ~12ms）
- **prefill 批处理未做**：28 层链式 launch 无法在层间插 padding NaN 清零（0×NaN=NaN 会污染有效 token），prefill 只跑一次收益有限，保持逐层
- **遗留**：talker W8 量化轨迹翻车（"这是一个比较长的句子…"）与优化前一致，非本次引入

## 2026-08-14：追实时优化（F16 CP 重验通过）

- **重大修正**：F16 CP cache 在修复后代码上**长句通过**（之前"F16 长句退化"是 merged 回退单层 bug 污染的结论）。cp_only codes16 逐 token 一致 + 4 句稳定性测试全过。CP 115→73ms
- **RTF 2.36 → 1.893**（KV 设备常驻 + F16 CP）
- 合并 lm_head/embedding（Gather 动态索引）：无效（Gather 开销 ≥ 15 次 launch），已回退
- SEQLEN=96：收益仅 0.03（1.893→1.863，96 非 2 次幂 TIU 效率低），已回退 128
- **物理极限**：talker W8 ~70ms + CP F16 ~73ms + codec ~7ms ≈ 150ms/帧 → RTF ~1.9 是 BM1684X 单核极限；RTF 1.0 不可达（talker+CP 双 W8 才 ~123ms/帧 → 1.5）
- 最终配置：talker W8 SEQLEN128 + CP F16 cache + CP 全 F32 组件（lm_head/embedding）+ KV 设备常驻 = **RTF 1.89**

## 2026-08-14：BF16 KV 验证（不可行，接受 v1 结论）
- 尝试 3 条路径编译 bf16 KV 输入网络（--input_types / 手动 mlir 编辑 + Cast / SSA 修复）全部失败：TPU-MLIR 1.28 model_transform 工具链对 bf16 输入有编译 bug（assert num_input==num_output / IndexError）
- v1 的 BF16 KV 实验用 llm_convert 工具链（原生支持 KV bf16），实测更慢（RTF 3.24 vs 1.24）——接受该结论，KV 保持 F32

## 2026-08-14：CP W8BF16 修复后重验（确认退化，非 bug 污染）
- W8 cache 在修复后代码重测：12 帧 codes16 从第 10 个 code 错（...564 1977 288... vs F32 参考 ...564 735 1657...）；长句 107+ 帧退化循环
- 结论确认：**W8 CP 退化是量化精度真实限制**（与 F16 的"旧结论被 bug 污染"不同）；F16 CP（12 帧全对 + 长句稳定）是 CP 量化下限
- 最终配置不变：CP F16 cache（73ms/帧），RTF 1.89

## 2026-08-14：16 槽改造 + 翻车根因定位
- CP 16 槽真实 shape（对齐 v1 s2/c16）：重导出 ONNX（history 16 槽/mask 17/rotary 位置动态 g+1）+ C++ 改造。cp_only 全对，但 10 条 v1 同文本复测 = 15 槽完全相同（5 成功 5 失败）
- **根因定位**：F32 CP 对照同样失败 → 翻车在 **talker W8 轨迹**（code0 循环不 EOS），非 CP 精度/槽数
- 失败集：Vivian"今天天气不错"、Hello.(Ryan)、Uncle_Fu 同句、数字长句、40token 长句
- F32 codec_head 尝试：combine 进 cp_allf32 导致 CP 网络 launch 卡死（combine 损坏），已回退；如要做需重新 combine talker 或独立 bmodel 参数
- 最终：16 槽 F16 保留（对齐 v1 语义），5/10 可用，剩余翻车待 talker 侧解决（候选：talker F16 重编 25 分钟）
