# Sophon BM1684X 移植知识库

**目标芯片**: BM1684X (SDK-23.09 LTS SP4)
**转换工具**: TPU-MLIR v1.28.1（Docker: `sophon/tpuc_dev:v3.4-tpumlir-1.28.1`，容器名统一 `sophon-tpumlir-v128`）
**交叉编译**: `sophon-cross-build` 容器（Ubuntu 20.04 + GCC 9.4 + glibc 2.31，产物符号 ≤ GLIBC_2.29）
**更新日期**: 2026-09-03（覆盖 11 个模型移植经验；本版逐项与磁盘产物、板上原始日志和容器实测对账；上一版 2026-08-17）
**算子级支持**: 见 `sophon_tpumlir_operators.md`（保持不变，本库引用）
**性能数字**: 本库只记趋势与结论，具体数值一律以 `PERF_SUMMARY.md` 为准

> 本文档按主题横向组织，每条经验标注**来源模型**（正是差异所在）与适用边界。
> 第六章列出历史结论的演进与矛盾裁决，避免误导。

---

## 一、工具链与转换流程

### 1.1 环境准备

| 坑 | 详情 | 来源 |
|---|---|---|
| **WSL2 内存上限** | 默认内存 = 物理 50%。大模型 ONNX 导出/onnxsim 峰值超限 → OOM killer，**Exit 137 是 OOM 不是 timeout**，会拖崩 VSCode/WSL 断联 | whisper / Eureka |
| 修复 | Windows `C:\Users\<user>\.wslconfig`：`[wsl2] memory=14GB swap=16GB`（实机 `free -g` 约 13GB 可用，whisper turbo encoder 全量导出已在此配置下跑通）；断联先 `uptime`（开机时间短 = 刚重启）+ `free -h` 确认 | — |
| 大 ONNX 导出 | 跳过 onnxsim；>2GB 用 external data 另存（protobuf 2GB 上限） | whisper / Eureka |
| **交叉编译 glibc** | **板卡 glibc 实测 2.31**（`ldd --version` = `Ubuntu GLIBC 2.31-0ubuntu9`）。WSL GCC15 产物需 `GLIBC_2.43`、服务器 Buildroot GCC10 需 `2.34`，都超出板卡 → 统一用 docker `sophon-cross-build`（Ubuntu 20.04 + GCC 9.4，产物符号 ≤ `GLIBC_2.29`）。判据是"产物需要的最高符号 ≤ 板卡 glibc"，不是"工具链 glibc 等于板卡 glibc" | whisper / sherpa / Eureka |
| 关 LTO | 交叉编译务必 `-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF`，否则板卡单核 OOM | sherpa |
| **容器名与镜像** | 仓库统一用 **`sophon-tpumlir-v128`**（镜像 `sophon/tpuc_dev:v3.4-tpumlir-1.28.1`，由 `3_docker/build_tpumlir.sh` 从仓库内 wheel 构建）。实测本机另有一个名为 `sophon-tpumlir` 的容器跑在 `sophgo/tpuc_dev:latest` 上，**两个容器的 tpu_mlir 都是 1.28.1、`model_transform.py`/`model_deploy.py`/`llm_convert.py` 都在**——因此"`sophgo/tpuc_dev:latest` 是干净镜像不含 TPU-MLIR"的说法是错的。但 `run_docker.sh` 有镜像守卫，同名容器镜像不符会 exit 2，所以不要混用两个名字 | 全部 |
| 容器挂载 | `run_docker.sh` **只创建 `/workspace`（= 仓库根）一个挂载**。脚本参数里写 `/models` 之类路径不存在，除非自己 `docker run` 追加 `-v` 并在文档写明 | HY-MT / QwenLLM |
| 板卡信息 | 由部署环境变量提供（`BOARD_IP`/`BOARD_USER`/`BOARD_PORT`/`BOARD_PASS`），不写入仓库；网络瞬断会自动恢复（`uptime` 正常即非重启） | 全部 |
| transformers 版本 | 按工具链阶段选：早期转换容器固定 4.51.1（5.x 与 PyTorch 2.1.0+cpu 不兼容）；llm_convert 时代容器内 torch 2.4.1 + transformers 4.57.6+（5.14 的 HF 标准权重可直接走官方 llm_convert） | Qwen3-ASR / HY-MT |

### 1.2 两条转换路线（先选路线再动手）

```
路线 A（通用算子模型）:  PyTorch → ONNX → model_transform.py → model_deploy.py → bmodel
路线 B（LLM 标准模型）:  Safetensors → llm_convert.py（层融合 + KV 独立）→ bmodel
```

- **路线 B（llm_convert）优先**：Qwen3-ASR 从自定义分块方案（1.83G）迁移到官方
  `llm_convert --qwen_asr`（896MB）：体积 -51%、内存 -49%、decode +83%——别重复造轮子。
  后续 seq512 版 encoder+LLM 合并单文件进一步到 **646MB**。QwenLLM / Qwen3-ASR / HY-MT
  全部走此路线。
- **路线 B 的边界**：audio-embed 等非标准输入注入会语义有损（Eureka 实测 5-6/9 → 3/9 已弃）。
  标准 text-token 输入无此问题。
- **路线 A 适用**：非 LLM 模型（whisper/sensevoice/chatTTS/vits/zipformer/moonshine）及
  TTS 类需要逐层拆分的生成式模型。

### 1.3 通用转换坑（算子层）

| # | 坑 | 根因 / 解法 | 来源 |
|---|---|---|---|
| T1 | Conv 缺 `kernel_shape` | 新版 torch.onnx.export（opset 17）不写该属性 → model_transform KeyError；从权重推断补全 | whisper / moonshine / sensevoice |
| T2 | KV dummy tensor 用 `[tensor]*n` | 同一引用被 constant folding 消除 → 必须列表推导式 `[tensor for _ in range(n)]` | whisper / Qwen3-TTS |
| T3 | `Or` 算子不支持常量输入 | onnxsim 提升 initializer 报 `operand eq not found` → causal mask 改嵌套 `Where` | moonshine |
| T4 | `RandomNormalLike` / `NonZero` 不可编译 | RandnLike UNREACHABLE + NonZero 数据相关 shape 崩溃 → **模型拆分**：动态部分留 CPU，可静态部分单独编译 | vits |
| T5 | 动态 shape 不可用 | TPU 静态编译；内容决定长度的输入（如 T_mel）→ 固定最大长度 + padding | vits / 通用 |
| T6 | int64 自动降 INT32 | bmodel 内 token/cache_len 为 INT32；C++ 上传需 cast | moonshine / zipformer |
| T7 | SplitMatMulPattern codegen 缺陷 | 相对位置 attention 的 Slice+MatMul 被错误赋 shape → 前推到 MatMul 输入，用 rank-3 MatMul 规避 | zipformer |
| T8 | TPU-MLIR 缺模型支持 | 需 patch 补 dispatcher/权重名（HY-MT 补 hunyuan_v1_dense、QK-Norm 权重名、DynamicNTKAlphaRotary）；补丁锚点校验 count==1 | HY-MT |
| T9 | **中间产物写 cwd** | `model_deploy.py` 的 npz/mlir/json 写入**当前工作目录**（不是 --mlir 指定目录）→ 编译脚本必须自带隔离与结束清理：`mktemp` + `trap`（zipformer `gen_bmodel.sh`）或固定写 `compile/tmp/`（Qwen3-TTS `gen_final.sh`）。仅"cd 到输出目录"不够——曾一次把 ~7.5G 散落到仓库根，并泄漏到仓库外（`../models/`） | Qwen3-TTS / zipformer |
| T10 | **>2GB 词表导出的 bf16 workaround** | Qwen3-TTS `embedding_text.onnx` 的 Sigmoid 是 **bf16**，这不是 bug：torch.onnx 有 2GiB 形状推断上限，而 fp32 词表达 1.25GB，只能降 bf16 导出。副作用是 **bf16 Sigmoid 在 ORT CPU 上无法运行**（数值自检要绕开这一层），但**不影响 TPU-MLIR 编译与板卡推理**（现存链路已经板卡验证）。FP32 导出实测必然失败，已复核，不要再试 | Qwen3-TTS |
| T11 | **`onnx.checker` 对 external data 误报** | whisper turbo `_sim.onnx` 用外部权重 `.onnx.data`（encoder 达 2.5G）时，`onnx.checker` 对 >2GB external data 会误报错误。**以 ORT 能否实际执行为准**，不要因 checker 报错就回退导出方案 | whisper |

### 1.4 `--disable_layer_group` 适用边界（2026-08 改写）

- **边界是"单网络 bmodel > 500MB"，不是"模型大"**。
  - 必须加：whisper turbo encoder 1.3G（不加 → 板卡 kernel panic 重启，连官方 bmrt_test 单独加载都挂；诊断关键：不是 OOM——device mem 1.66G < npu heap 3.86G，是 layer_group codegen bug）
  - 不用加：逐层拆分的小网络（Qwen3-TTS talker 每层 21MB，56 个网络没加也 OK）
  - 加了反而失败：个别模型（Eureka qwen3）加后 SHA 校验失败 → **逐模型验证，不能一刀切**

---

## 二、量化精度决策树

### 2.1 核心立场（团队共识，2026-08）

> **不同算法模型量化退化程度不一样，必须做具体实验**，不能套用经验。
> W8BF16 损失小但体积偏大，不是最佳；**体积/精度最佳目标是 W4BF16**。
> INT4 在本仓只对 ChatTTS 的 GPT 做过实验，因此本节的 INT4 结论仅对 ChatTTS 有效（见 §6「已知限制」）。

### 2.2 证据矩阵（11 模型 × 量化档位，全部实测）

| 模型 | 类型 | 量化档位 | 结果 |
|---|---|---|---|
| Whisper large-v3-turbo | 自回归 ASR | F16 / W4F16 | ✅ 无损；W4F16 省 65% 内存、RTF 与 F16 持平（数值见 PERF_SUMMARY 一） |
| Whisper encoder（独立） | encoder | BF16 | ❌ cosine 0.51（attention 累加精度不足） |
| Whisper encoder（独立） | encoder | F16 / W8F16 | ✅ F16 0.99 / ❌ W8F16 板上连续推理硬重启 |
| Whisper encoder（独立） | encoder | INT8 | ❌ 32 层误差累积致 audio_features 失真；decoder INT8 编译 tpuc-opt abort（compiler bug） |
| SenseVoice Small | CTC | FP16 / F32 | ✅ 两者结果逐字一致 |
| Moonshine | 流式 ASR | FP16 / FP32 | ✅ 两档 token 100% 一致，F16 约快 2.2× |
| Zipformer | Transducer | FP16 / FP32 | ✅ F16 token 与 F32 ORT 完全一致；RTF 需区分 warm 与首轮（数值见 PERF_SUMMARY 一） |
| ChatTTS | 生成式 TTS | **GPT INT4** + decoder/vocos **BF16** | ✅ 可用（INT4 在本仓唯一验证过的模型） |
| VITS-Melo | TTS | F16 | ✅ F16 通过（SDP 留 CPU）。仓库内**只有 F16 三件产物**，无 F32 bmodel |
| Eureka（whisper enc + Qwen3-1.7B） | 音频分类 | whisper **F16** + qwen3 **W4BF16** | ✅ 跑通，准确率 **5-6/9**（ChatTTS 长指令集，与原版 PyTorch GPU 持平）。⚠️ 运行时资产与源权重已不在仓库，当前不可复现 |
| QwenLLM 0.6B | LLM 意图 | w4bf16 | ❌ 28 层误差指数放大，top1 翻转、意图 8/10 格式不稳 |
| QwenLLM 1.7B | LLM 意图 | **w4bf16** | ✅ 曾是甜点（29 tok/s、9/10）。**权重与产物已从仓库移除，仅作量化机理证据保留** |
| QwenLLM 4B | LLM 意图 | W4F16(AWQ) | ✅ 曾测得冷加载 57.5s、10/10，但连续推理热降频严重。**权重与产物已从仓库移除，仅作证据保留** |
| QwenLLM 0.6B（v95 dispatch 系列） | LLM 意图 | **w8bf16** | ✅ 当前唯一交付档（v95e-soup / seq2048）。仓库内 4 个变体全部 w8bf16，**无任何 w4 产物** |
| Qwen3-ASR 0.6B | ASR（LLM 类） | w8bf16 / w4bf16 -g64 / **w4f16 -g64** | ✅ 13/13 有效多语种音频。三档 RTF 口径不同，数值见 PERF_SUMMARY 一 |
| HY-MT 1.8B | 翻译 | **w8bf16**（交付）/ w4bf16 g64（速度）/ w4f16 g64（实验） | ✅ W8 61 用例全部执行完成；W4 两档质量偏移明显（16 项字符相似度 0.815 对 W8 的 0.964）。**w4f16 的 bmodel 产物与源权重均未保留**，板上日志已入库 |
| Qwen3-TTS talker | 生成式 TTS | **W4F16 候选 / W8BF16 回退** | ✅ 54 条 batch 全部成功；W4F16 人工试听总体稳定，但 `en_03`/`zh_03` 有已知异常。**RTF 必须带 SEQLEN 标注**：1.900(W8)/1.818(W4F16) 是 SEQLEN=192；历史值 2.85→2.51 是 SEQLEN=128，不可并列比较 |
| Qwen3-TTS CP | 生成式组件 | F32 组件 + **F16 cache 下限** | ✅ 生成式路径最敏感 |

> 表中只记档位与定性结论；**所有具体数值以 `PERF_SUMMARY.md` 为准**（含数据集、口径、测试日期与原始日志出处）。
> 标注「已从仓库移除」的行保留是因为它们构成 §2.4 量化机理的关键对照证据（0.6B w4 失败 vs 1.7B w4 成功），删掉会让机理论证失去支撑；不代表这些档位仍可部署。

### 2.3 决策树（按模型特征）

```
模型是什么任务？
├─ 连续/嵌套自回归输出（TTS 生成式）→ 先试 W4F16（需同模型大样本 + 人工试听）
│    ├─ Qwen3-TTS W4F16：54 条 batch 全部成功，人工试听总体稳定，但保留 2 条已知异常
│    └─ 未经人工确认的其他生成式模型，不得从该结果外推；W8BF16 作为保守回退
├─ 离散 token 输出（ASR / 翻译 / 意图）→ 先试 w4bf16 -g 64 或 w4f16：
│    ├─ 每种激活 dtype 都要独立实测；W4F16 不是 W4BF16 的无风险替换
│    └─ 质量不稳定退回 W8BF16
└─ 非 LLM 编码器类（encoder/CTC/Transducer）→ F16 为主力（whisper/sensevoice/zipformer）
     ├─ 注意 BF16 对 attention 精度不够（whisper cosine 0.51）
     ├─ W4F16/W4BF16 需分别验证，不以编译成功代替质量验证
     └─ INT8 已试失败（whisper）；W4A8 硬件不支持（仅 BM1688）
```

### 2.4 量化下限的机理（为什么不能一刀切）

1. **TPU 量化网络层内激活以量化 dtype 存储**（bf16 尾数 7 位 / f16 10 位）→ 生成式模型
   （嵌套自回归、连续 hidden 输出）argmax 轨迹翻车；CPU torch bf16（eager fp32 层内）
   正常 → 铁证是**编译器层内激活精度限制**，`--high_precision` 无法修复（norm 非主误差源，
   matmul 层内才是）。
2. **w4 误差经层间指数放大**：0.6B 单层误差 ~0.05（group128）/ ~0.02（g64），经
   attention/FFN 每层放大 40-90 倍，28 层后 logits 完全错乱。残差 + RMSNorm 是放大机制。
3. **F16 > BF16（同 16bit 位宽）**：F16 尾数 10 位 vs BF16 7 位。whisper encoder：
   ONNX cosine 0.99 → BF16 bmodel 0.51 → F16 bmodel 0.99。对累加精度敏感的网络（attention）
   用 F16；权重已是 `.half()` 时 F16 最匹配。
4. **本地噪声模拟先行验证**（省编译时间）：embedding 输入加 ±0.03 噪声跑 28 层看 top1
   是否翻转——0.06 噪声 → layer1 max diff 5.4；0.005 噪声（w8 级）→ 28 层 top1 稳定。

### 2.5 量化档位速查

| 档位 | 名称 | 说明 |
|---|---|---|
| F16 | FP16 | 编码器类主力；whisper/sensevoice/zipformer/moonshine |
| BF16 | BFloat16 | 慎用：attention 累加精度不足（whisper 失败案例） |
| W8BF16 | 8bit 权重 | 损失小但体积偏大 → **仅兜底** |
| W4BF16 | 4bit 权重 + bf16 激活 | **体积/精度最佳目标**；PTQ 直接可用；必须 `-g 64`（group 128 失败） |
| W4F16 | 4bit 权重 + fp16 激活（AWQ/LLM） | 与 W4BF16 是不同激活路径；llm_convert 模型 config 必须声明 F16，必须逐模型验证（Qwen3-TTS/Qwen3-ASR/HY-MT 已分别实测） |
| INT4 | 整型 4bit | 仅 ChatTTS GPT 验证过；本仓未对其余模型做 INT4 实验，结论不可外推 |
| INT8 | 整型 8bit | whisper 已试失败（精度崩 + 编译 abort） |
| W4A8 | 4bit 权重 8bit 激活 | **BM1684X 硬件不支持**（仅 BM1688） |

---

## 三、bmruntime C++ 推理模式

### 3.1 launch / sync 策略（2026-08 裁决，两个项目结论统一）

`bmrt_launch_tensor_ex(..., true, false)` 最后两参数是 **user_mem / user_stmode**
（不是 is_sync），CPU 不被阻塞。规则分层：

1. **CPU 读结果前必须 `bm_thread_sync`**（无争议，所有项目一致）
2. **层间 launch/d2d 是否 sync 取决于产物类型**：
   - llm_convert 标准产物（层融合、KV 独立）：层间无 sync 稳定——同线程 launch/d2d
     进入驱动 FIFO 队列自然串行（HY-MT 61/61 实测、官方 llm_tpu demo 一致）
   - model_transform/deploy 通用产物 + 手写 d2d 链：逐层 sync 保守（Qwen3-ASR 读到
     0x7fff bf16 NaN / 上次进程残留，就是 sync 缺失）；Eureka 实测去掉 28 次 sync 无
     提速 → 同步点可省但保留更安全
   - QwenLLM 无显式 sync 能跑通是 d2d 链碰巧形成自然等待，**不可依赖**
3. **跨线程、有 host 参与（s2d 覆盖、host 判断）时每步 sync**

### 3.2 device memory 使用规则

| 规则 | 说明 | 来源 |
|---|---|---|
| 不改 `bm_device_mem_t.size` | d2s/s2d 内部用 size 做 DMA 分配 → 改 size 导致 heap 损坏（malloc invalid chunk）；部分读写 = 全量传输 + host 改 + 重新上传；切片用 `bm_mem_from_device(addr, size)`（比改 addr+size 干净） | ChatTTS / Qwen3-TTS |
| **KV cache 是 head-major** | `[H,S,D]` 布局：写回新 KV `[8,1,128]` 到槽位必须逐 head `memcpy(cache+(h*S+slot)*D, nk+h*D, D*4)`；直接 memcpy 全塞进 head0、head1..7 变 0，输出错乱且难察觉（读回与写回源一致）。排查靠"非零分布"（head0 全非零、head1-7 全零）。15 槽截取同理必须逐 head | Qwen3-TTS |
| 层间 d2d 写**下一层 in0** | 写中转 buffer 不生效（下一层读自己 in0 残留）；addr-mode1 下各层 pos/mask 共享同一 input_mem 只传一次 | Qwen3-ASR |
| net **输出** mem 可被 with_device 引用 | 零拷贝链路（上一层输出直连下一层输入）是推荐优化，释放顺序正确无 double-free | Qwen3-TTS / whisper / HY-MT |
| net **内置 input_mems** 谨慎 | Eureka 裸 bmrt 操作 input_mems + net 间 d2d 直传，真实数据下板卡硬重启（零数据不崩真实数据崩）→ 遇板卡 reboot 先隔离此项，改用 `bmrt_tensor` 独立分配 IO + host↔device 往返 | Eureka |
| `bmrt_get_network_info` 判空 | 缺网络返回 nullptr（缺 embedding_14 曾致段错误）→ init 统一判空并报出缺失网络名 | Qwen3-TTS |
| 输入长度防护 | 固定 shape 缓冲（SEQLEN 分配）必须在入口检查 token 长度，Release 下 assert 被编译掉 → 用显式 if + throw | HY-MT / Qwen3-TTS |

### 3.3 推理性能模式（按收益排序）

| 模式 | 做法 | 效果 | 来源 |
|---|---|---|---|
| **KV 设备常驻** | decode 输入零拷贝 with_device + 新 KV d2d 写槽，消除每帧 host↔device 搬运 | whisper decode 提速 3.1×（git 49b83c4）；Eureka 6.2→16 tok/s（2.6×）；Qwen3-TTS RTF 4.37→2.36 | whisper / Eureka / Qwen3-TTS |
| 批处理连续 launch + 一次 sync | 28 层连续 launch 后单次 bm_thread_sync | decode 提速 ~25% | Qwen3-TTS / HY-MT |
| 零拷贝 hidden 直连 | 上一层输出 device_mem 直连下一层输入（with_device + user_mem=true） | 省每帧 28MB DMA | Qwen3-TTS |
| SEQLEN 权衡 | 减半 attention/KV：ASR seq256 主力（512 仅 +35MB）；TTS SEQLEN=192 解决长句容量（128 时 prefill 占槽致 decode 上限低）；**非 2 次幂（96）TIU 效率低，收益仅 0.03 不值得** | Qwen3-TTS：SEQLEN 128→192 后 RTF 2.85→2.51（**128 配置**的历史值；192 配置的当前值见 PERF_SUMMARY） | Qwen3-ASR / Qwen3-TTS |
| embedding 缓存 | 免每帧 16 次小 launch | CP 帧内省 16 次 launch | Qwen3-TTS |
| 模型常驻批量 | 一次加载多句合成（--batch），消除重复加载 | batch 模式 RTF 稳定（SEQLEN=192 配置的数值见 PERF_SUMMARY 二） | Qwen3-TTS |
| 采样解码 + NaN guard | 采样模式修复 greedy 循环退化；NaN/全零权重回退 argmax（discrete_distribution 对 NaN 死循环） | 修复 CPU 100% 卡死 | Qwen3-TTS |

### 3.4 prefill 与数值正确性

- **prefill padding 产生 NaN**：seq_len < SEQLEN 时 padded hidden 行（全 masked attention）
  产生 NaN → 每层后把 `[tok_len..SEQLEN-1]` 清零；批处理 vector 必须 resize（越界堆损坏） | ChatTTS / Qwen3-TTS
- **多 bmodel 一 handle**：一个 bmrt 只能加载一个 bmodel；多模型（ChatTTS 三引擎）共享
  单 bm_handle | ChatTTS
- **C++ 运行时读维度**：所有维度（n_mels/n_state/n_layer/vocab/SEQLEN/language 偏移）从
  bmodel net_info 读出，一套代码多模型通用 | whisper / Qwen3-ASR / HY-MT
- 批处理/批量模式单线程顺序循环；引擎非线程安全（shared mutable 状态），并发需每线程一实例 | 通用

---

## 四、板卡部署与调试

### 4.1 部署

| 坑 | 解法 | 来源 |
|---|---|---|
| **scp 大文件静默损坏** | 板卡 crash/网络瞬断期间 scp 传的大文件**大小对内容错**，损坏文件导致诡异症状（decoder 加载失败 / pos_emb 全 0 → 输出空 / 数字串死循环）→ **每个上板文件 md5sum 比对本地**，别假设 scp 成功；排查先 md5 再怀疑模型/代码 | whisper / Eureka / Qwen3-TTS |
| eMMC 冷读像卡死 | 2.7G bmodel 冷读 ~60s 进 D(disk sleep) → 先 `cat *.bmodel >/dev/null` 预热 page cache | Eureka |
| 磁盘空间 | 根分区 overlay 易满（实测 `/` 5.8G 用 3.9G、`/data` 16G）→ 临时文件用 `TMPDIR=/data/tmp`；多模型串行验证时"部署→测→删"，不要并存 | Eureka |
| 网络瞬断 | 会自动恢复；判断是否真重启看 `uptime`（开机时间短 = 刚重启）。板卡地址由 `BOARD_IP` 注入，不写入仓库 | Eureka |
| 部署脚本 | 上传后统一 md5 校验（各模型 `deploy_to_board.sh` 已内置）；**不要手写裸 scp 部署**，那样会漏掉校验 | Qwen3-TTS |
| **root 属主产物删不掉** | docker 内生成的文件属 root → 用 `docker exec sophon-tpumlir-v128`（镜像内 root，仓库挂 `/workspace`）清理，**先 md5/内容核验再删，不用 sudo** | 全部 |
| scp 子系统不可用 | 个别板卡 SSH 不支持 scp 子系统 → `cat file \| ssh host 'cat > dest'`。**⚠️ 这是 RK3588 的经验，不是 BM1684X**；BM1684X 的 scp 正常可用，只是会静默损坏（见首行） | RK3588（非本平台，仅备查） |

### 4.2 调试方法论

1. **上板前先 Python/sail 验证**：Python 推理先证明 bmodel 没问题再写 C++（C++ 裸 bmrt
   真实数据会 reboot 的教训）；sail 的 SYSIO = bmrt_launch_data，同样禁止操作 input_mems
2. **NaN 排查顺序**：先验证前端数值（dump mel/特征，统计 min/max/nan 数）→ sail 对照隔离
   → 再怀疑 bmodel/驱动。数值计算（STFT 帧数、pad 边界）必须与 torch/numpy 参考精确对齐，
   C++ 越界读静默产 NaN（STFT `(len+2pad)/hop+1` 多算 3 帧 → 81% NaN，曾误判 bmodel 冲突）
3. **逐级 cosine 定位**：mel→encoder→embeds 逐级对比；ONNX vs bmodel 分开定位是导出 bug
   还是量化损失（block-0 输出 cosine 0.9999995 级为正常）
4. **内存核算**：`bm_get_stat` mem_total=9070MB 是 bmrt 实际可用（npu heap 3.86G 是单分区
   误导）；多 bmodel 共存需算账（4.1G 双模型 OK）
5. **板卡硬重启成本高**：每次 reboot 要等 + 换 eth1 IP；验证新量化先单句、再小心连续测；
   不稳定量化（W8F16）在 sail 下连续推理同样 reboot
6. **性能测量**：batch 常驻模式测 RTF（proc/音频时长）；LLM 类测 prefill ms + decode tok/s；
   预热后重复多次（首轮冷启动不计）

---

## 五、分模型经验表（独特坑，按模型）

| 模型 | 类型 | 最终精度/指标（数值见 PERF_SUMMARY） | 独特经验 |
|---|---|---|---|
| **Whisper**（base / large-v3-turbo） | 自回归 ASR | turbo W4F16 交付档，中英逐字无损、省 65% 内存；encoder 必须 `--disable_layer_group`；W8F16 上板硬重启。**base 有产物但无板卡实测数据**（旧文档的 base 耗时无出处，仓库内唯一 base 耗时是开发机 PyTorch baseline） | turbo 推荐 W4F16；校准数据 52 条可复用（`whisper/calib_data/`，本机产物、不入库）；KV 设备常驻 3.1× |
| **SenseVoice Small** | CTC ASR | FP16 交付档，F16 与 F32 结果逐字一致 | 无自回归所以量化宽容；4 prompt 向量固定可学习 → 自动语种识别；特征提取（CPU）在 F16 下占总耗时约 63%，是主要瓶颈 |
| **Moonshine** | 流式 ASR | FP16 交付档，约快于 F32 2.2×，token 100% 一致 | HF decoder 原地改 encoder_hidden_states（+=pos_emb）→ 循环必须传 clone；边界效应尾部 12 帧受 padding 影响；头文件勿用 `features.h`（glibc 同名冲突）；libsophon 0.5.1 的 bmrt_tensor 收 bm_shape_t 结构体 |
| **Zipformer** | 流式 Transducer | FP16 交付档，token 与 F32 ORT 完全一致；**RTF 必须区分 warm 与首轮**（旧文档的 0.024–0.071 无测量出处，已废） | 自定义流式协议 103→24（官方 context 全不匹配）；35 个 streaming state 的"2"是 stack layer 维非 batch；kaldi-native-fbank 冻结为 frontend（MAE 4.13e-6）；SplitMatMulPattern 规避 |
| **ChatTTS** | 生成式 TTS | GPT INT4 + decoder/vocos **BF16**，BM1684X 上 RTF<1、70/70（**旧表里的 RTF 2.5 是 BM1688/SE9-16 SDK V1.7 的数据，与本平台无关，不得引用**） | INT4 唯一验证模型；tokenizer 截断/prefill NaN/position 偏移/OOM 碎片化/共享 handle 七坑（详见历史版本）；参考音频克隆只在 Python/SAIL 路径，C++ 支持预计算音频 prompt，sherpa-onnx 路径仍只支持固定 `spk_emb` |
| **VITS-Melo** | TTS | **F16**（仓库内只有 F16 三件产物，无 F32 bmodel）；仅 2 条 smoke，未做批量回归 | 模型拆分（SDP 动态留 CPU，decoder 单独编译）；sid 必须传 1（0→静音）；NonZero/RandomNormalLike 不可编译 |
| **Eureka-Audio** | 音频分类 | whisper F16 + qwen3 W4BF16，准确率 **5-6/9**（ChatTTS 长指令集，与原版 PyTorch GPU 持平）。⚠️ **运行时资产与源权重已不在仓库，当前不可部署、不可复现** | whisper attention scale 只能乘 q；whisper bmodel 必须 F16（BF16 使 cosine 掉到 0.51），但文件名保留 `_bf16` 后缀——**不要按文件名判断精度**；C++ 裸 bmrt 板卡 reboot 最大坑（Python/sail 安全）；audio-embed 场景 llm_convert 有损（5-6/9 → 3/9，已弃） |
| **QwenLLM** | LLM 意图 | 当前交付 **v95e-soup / w8bf16 / seq2048**（仓库内 4 个变体全部 w8bf16）。历史上 1.7B w4bf16 是甜点、0.6B w4 失败、4B W4F16(AWQ) 可用但热降频严重——**1.7B/4B 的权重与产物已从仓库移除** | 0.6B w4 失败（误差指数放大）、1.7B 曾是甜点；thinking 模式必须关闭（`--no_think`）；0.6B 输出会包裹 markdown 代码块，JSON 解析需额外剥离 |
| **Qwen3-ASR** | LLM 类 ASR | W8BF16 为精度基线、W4F16 g64 为交付档，13/13 有效多语种音频；decode 64–65 tok/s（旧文档的"RTF 0.10"与"64-68 tok/s"无出处，已废） | encoder 全量 3000 帧一次过（chunk 编码长音频漏段）；mel 复制最后一帧而非补零；mask 必须 fp32（bf16 求和 3000→3008 破坏 cu_seqlens）；窗口 attention 帧-major 转置；STFT 帧数公式 `(len+2pad-n_fft)/hop+1`；npz 解析 ZIP64 占位；自定义方案 → 官方 llm_convert（体积 -51% 内存 -49% decode +83%） |
| **Qwen3-TTS** | 生成式 TTS | W8BF16 talker（回退）/ W4F16 talker（候选）+ CP F32 + cache F16 + codec F16，54 条 batch 全部成功；**RTF 引用必须带 SEQLEN**（当前 SEQLEN=192；旧值 ~1.85 / 2.51 / 2.85 分属不同配置与轮次） | CP 必须 F32（最硬结论）；head-major KV 写回；embedding_14 缺失段错误；F16 网络 input_dtypes 仍是 F32；mask SEQ 位保持可见；采样 seed 42 逐句复现 + NaN guard；SEQLEN=192 长句容量 |
| **HY-MT** | 翻译 | **W8BF16 交付档**，61 用例全部执行完成；W4BF16 g64 为速度档（16 项字符相似度 0.815 对 W8 的 0.964）；W4F16 g64 为实验档，**bmodel 与源权重均未保留**（板上日志已入库）。decode/prefill 各轮口径不同，见 PERF_SUMMARY 四 | BF16 KV cache（llm_convert 标准行为）；QK-Norm 顺序需 patch；W4 输出偏移明显但作速度档；61 用例回归脚本可复用于任意档位（`BIN=` 覆盖）；**W4F16 有单条 decode 离群到 1.35 tok/s**，评估速度档时必须计入 |

---

## 六、矛盾与边界（历史结论演进）

本节记录旧文档/旧结论与当前现实的差异，避免"过期经验"误导。

| 旧结论（时间） | 当前结论（2026-09） | 裁决依据 |
|---|---|---|
| "FP16 推荐，INT4 大模型首选"（05-19） | F16 仅编码器类主力；LLM 类 W4BF16 最佳/W8 兜底；INT4 仅 ChatTTS 验证 | 11 模型证据矩阵（2.2） |
| "多输入输出时加 --disable_layer_group"（05-19） | 边界是**单网络 >500MB**；逐层拆分小网络不用加；个别模型加了反而 SHA 失败 | whisper 1.3G panic vs Qwen3-TTS 21MB/层 OK vs Eureka 反例 |
| "每层 launch 后必须 bm_thread_sync"（08-08，Qwen3-ASR） | CPU 读结果前必须 sync；层间依赖驱动 FIFO 队列（llm_convert 产物验证稳定）；通用 deploy 产物保守逐层 sync。**`Qwen3-ASR/README.md` 曾长期沿用旧结论未同步，已修正** | HY-MT 61/61 与 llm_tpu 官方 demo 一致 |
| "禁止操作 net input_mems"（Eureka reboot） | d2d 写下一层 in0 是多数项目稳定路径（ASR/TTS/llm_tpu）；Eureka 案例作为**排查建议**保留：遇板卡 reboot 先隔离此项改独立 buffer | 6+ 项目对照 |
| "w4bf16 不可用"（08-08，0.6B 意图） | w4bf16 可用但**有条件**：必须 -g 64；≥1.7B 模型；离散任务；先噪声模拟验证 | ASR/HY-MT 成功 vs 0.6B 失败对照 |
| "自定义分块方案"（Qwen3-ASR 早期 1.83G） | 官方 llm_convert 优先（896MB）；自定义方案仅 audio-embed 等非标准输入时考虑 | 体积/内存/decode 全面对比 |
| **"板卡 glibc ≤2.30"**（sherpa 交付规范） | **板卡 glibc 实测 2.31**（`ldd --version` = Ubuntu GLIBC 2.31-0ubuntu9）；判据是"产物最高符号 ≤ 板卡 glibc"，cross-build 镜像产物为 ≤2.29 | 板上实测 + 镜像实测 |
| **"`sophgo/tpuc_dev:latest` 是干净镜像，不含 TPU-MLIR 工具"**（zipformer README） | **该说法错误**：本机该镜像的容器内 `model_transform.py`/`model_deploy.py`/`llm_convert.py` 齐全，`tpu_mlir` 版本同为 1.28.1。仓库仍统一用派生镜像 `sophon/tpuc_dev:v3.4-tpumlir-1.28.1` + 容器名 `sophon-tpumlir-v128`，理由是可复现（wheel 固定在 `0_Toolkits/`），不是因为 latest 不能用 | `docker exec` 两个容器分别实测 |
| **"推荐 Qwen3-1.7B / Qwen3-4B"**（PERF_SUMMARY 06-22） | 两者的权重目录与 bmodel 产物**已从仓库移除**，不可部署。当前 QwenLLM 交付为 **v95e-soup / w8bf16 / seq2048**；1.7B/4B 的测量值仅作为 §2.4 量化机理的对照证据保留 | 磁盘核查 + `QwenLLM/README.md` |
| **"Zipformer RTF 0.024–0.071"**（07-31 进入根 README，后被本库转抄） | **无测量出处**，已废。实测为 warm F16 0.0410 / F32 0.0513，首轮 0.0636–0.2264，两者必须分列 | `zipformer/README.md` 实测段；git 溯源 `7bf5ec5` 显示该数字进入时即为裸结论 |
| **"ChatTTS RTF 2.5"**（本库 §5 曾沿用） | 那是 **BM1688 / SE9-16 / SDK V1.7** 的数据，与 BM1684X 无关。BM1684X 实测 RTF 0.533、70/70 | `chatTTS/cpp/README.md`、`chatTTS/cpp/ISSUES.md` |
| **"Qwen3-ASR W4BF16 RTF 0.10"**（PERF_SUMMARY 06-22） | **无出处**，已删除。受控多轮中位数为 W8BF16 0.161 / W4F16 0.128；W4BF16 未做受控多轮测量 | `Qwen3-ASR/README.md` 表 + `test_outputs/bench_20260817/` |
| **"HY-MT W4F16 61/61、33.21 tok/s"作为头条指标** | 数值本身可从 `outputs/board_w4f16_20260817/hymt_w4f16_46.log` 复算 ✓，但**该档 bmodel 与 4G 源权重均未保留**，不可部署。头条改为 W8BF16。另发现该档有单条 decode 离群到 **1.35 tok/s**（`terminology_short`），此前未披露 | 磁盘核查 + 日志复算 |
| **"Eureka-Audio 准确率 ~90%"** | ~90% 属**早期短命令词集**，该测试集已不在仓库；当前仓库内验收集是 9 条 ChatTTS 合成长指令，实测 **5-6/9**，与原版 PyTorch GPU 基线持平。引用准确率必须带数据集 | `Eureka-Audio/README.md` |
| **`models_directory_standard.md` v1.0 的 §4/§5/§6** | 三节是带日期的执行流水账，其中"已删除"声明在后续端到端验证重新生成产物后**全部失效**（sensevoice/whisper 的 `.bmodel.json`+`.profile` 回来了、moonshine/sensevoice 的 raw onnx 回来了、Qwen3-ASR compile/ 又堆了 855MB）。已在 v2.0 移除，仅存于该处的两条技术结论迁入本库 §1.3 T10/T11 | 逐项磁盘核查 |
| 板卡地址 | 由部署环境变量提供（deploy 脚本为准），凭据受控保存于仓库外，不写入仓库。**"eth1 DHCP 每次重启会变"的说法已移除** | 实测 |
| 容器 transformers 4.51.1（05-25） | 按阶段：旧工具链 4.51.1；llm_convert 时代 4.57.6+ / 5.14 标准权重 | Qwen3-ASR/HY-MT |

**已知限制（非待办，归档时的既有状态）**

- **INT4 只在 ChatTTS 上验证过**，其余模型未做 INT4 实验；§2.5 的 INT4 行因此只对 ChatTTS 有效，不可外推。
- **W4F16 的质量结论不可跨模型泛化**：Qwen3-TTS / Qwen3-ASR / HY-MT 各自实测，结果互不适用。
- **两个模型的转换链路无法从本机复现**：ChatTTS 缺 HF `2Noise/ChatTTS` 权重（`python/.pretrained_models/*.ckpt` 是 143 字节占位），zipformer 缺 `assets/` 下的 checkpoint 与 icefall 源码；两者的 bmodel 产物在，但重导出 ONNX 需先取回上游资产。Eureka-Audio 另缺运行时资产与源权重，连部署都无法完成。
- **Qwen3-TTS 的 54 条加权 RTF 无逐条原始数据**；Qwen3-ASR 的 benchmark 日志不含音频文件名。两处的汇总数字因此不可独立复算，详见 `PERF_SUMMARY.md` 六.5。

---

## 附录：相关资源

- 算子支持与注意事项：`sophon_tpumlir_operators.md`
- 各模型移植记录：各模型目录 `.context/`（baseline / bmodel_info / operator_analysis）
- 板卡凭据受控保存于仓库外的受控文件，不随仓库提交
- 团队工作流：`.claude/subagents/`（5 阶段流水线：sail 验证先行）
