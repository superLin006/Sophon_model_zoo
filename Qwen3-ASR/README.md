# Qwen3-ASR-0.6B — BM1684X 移植

[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 的 Qwen3-asr-0.6B（30 语种 + 22 中文方言的语音识别 + 语种识别）完整移植到 Sophon BM1684X，中英文转写与原生基线一致。

## 当前采用方案：HF 标准权重 + llm_convert 单 bmodel

**结论：HF transformers 5.14 的 Qwen3ASR 权重（language_model 重命名为标准 Qwen3）直接走
LLM-TPU 官方 `llm_convert.py` 标准 Qwen3 转换**，比官方 `--qwen_asr` 方案（qwen_asr 包版权重，
thinker_config + mrope，架构不通用）更通用：**权重即 HF 原版，工具链即官方**。

**两个关键点（实测结论）**：
1. **默认交付使用 `w4f16 -g 64`**：646MB 单文件，13 条已覆盖多语种音频全部通过；`w8bf16` 是精度基线与回退档。`w4bf16 -g 64` 保留作 4bit 对照，默认 group 128 的量化误差会经 28 层放大，不能仅凭转换成功判断可用性
2. **KV bf16 是 llm_convert 标准行为**（block_cache 的 history_k/v 即 bf16），无需 hack

**encoder 用全量 3000 帧架构**（一个网络通吃离线+流式，对齐官方 Python/transformers 语义）：
mel pad 到 3000 帧一次过 encoder（跨块窗口上下文完整，20s 长音频无漏段），按真实帧数截取
audio tokens。流式每块全量重编码（固定 0.08s，1s 块下实时）。**chunk 独立编码（官方 bmodel 方案）
长音频会漏段，不采用。**

> 完整编译命令（含 W4F16 默认档、W4BF16 对照档、全量 encoder、合并单文件）见下文「**编译 bmodel**」一节，本节不重复，避免两份命令漂移。

### 档位与验收（2026-08-18 受控测试）

在全量 3000 帧 encoder + seq512 LLM 方案上：

| 档位 | 合并 bmodel 大小 | 有效测试 | RTF（受控多轮中位数） | 定位 |
|---|---:|---:|---:|---|
| **W4F16 g64** | **676,376,576 B（约 646MB）** | **13/13** | **0.128** | **默认部署档**（体积 -32%、RTF 最优） |
| W8BF16 | 987,545,600 B（约 942MB） | 13/13 | 0.161 | 精度基线 / 回退档 |
| W4BF16 g64 | 676,376,576 B（约 646MB） | 13/13（逐条结果见下文） | **未做受控多轮测量** | 既有 4bit 档，编译对照档 |

三档的 decode 速度一致（64–65 tok/s，与生成长度和 KV 长度无关）。

> **关于 W4BF16 的 RTF**：`PERF_SUMMARY.md` 早期版本给它标过「0.10（实测）」，但该数字在模型文档、`test_outputs/` 原始日志里都查无出处，已删除。W4BF16 只有 2026-08-10 的 13 条逐条单次结果（端到端 0.27–2.28s，RTF 0.040–0.54，短音频偏高），**没有做过受控多轮**，因此不与另两档的中位数并列比较。需要该档 RTF 时应按 `test_outputs/bench_20260817/` 的方式补测。

**定档结论**：默认部署 **W4F16 g64**，W8BF16 保留为精度基线与回退档。13 条有效多语种音频（中/英/日/德/法/西）上两档转写均全部成功，未观察到差异；但 **13 条不足以覆盖标称的 30 语种 + 22 中文方言**，因此在扩展到完整语种集之前，不宣布 W4F16 全面替代 W8BF16——遇到未覆盖语种的质量问题时回退 W8BF16。

W4F16 的 60 个网络已合并为 `models/BM1684X/qwen3_asr_merged_w4f16.bmodel`。首次测试发现 C++ 运行时原先把 LLM hidden/embedding 按 BF16 写入，而 W4F16 网络要求 F16；现已改为按 bmodel 的 input/output dtype 动态转换。修复后离线和流式均能正常转写，W8BF16 使用同一新版二进制回归正常。

受控测试中 60 秒音频失败属预期——超出当前 encoder 的 30 秒硬上限，不计入回归。W4F16 RTF 存在板卡运行抖动，表中用正式多轮的中位数，不将单次最低值当作最终性能结论。原始日志见 `test_outputs/bench_20260817/`（W8BF16 与 W4F16 各 4 轮，已入库）。

W4F16 的上板运行方式与 W4BF16 相同，仅替换 bmodel：

```bash
./qwen3_asr_bm1684x --bmodel models/BM1684X/qwen3_asr_merged_w4f16.bmodel \
    --model_dir . --audio test_data/test_zh.wav
./qwen3_asr_bm1684x --bmodel models/BM1684X/qwen3_asr_merged_w4f16.bmodel \
    --model_dir . --stream --audio test_data/test_zh.wav
```

官方 `--qwen_asr` 方案（qwen_asr 包版权重 + chunk encoder）曾做对照：其长音频会漏段且 20s 直接崩溃，**已弃用**。本方案与官方 Python/transformers 行为一致，转写更完整。



## 模型结构

`Qwen3ASRForConditionalGeneration`（单文件 model.safetensors，0.8B 参数 BF16）：

1. **Audio Encoder** `model.audio_tower`（18 层, d_model=896, 14 头, FFN 3584）：
   - 输入 log-mel (1,128,T)，16kHz，n_fft=400, hop=160（10ms/帧）
   - 3×Conv2d stride2（8x 时间下采样：3000 mel 帧 → 375）+ gelu + conv_out Linear → 896
   - 正弦位置编码（不可学习）→ 18 层**非因果窗口 attention**（n_window=50, chunk_len=100, n_window_infer=800）
   - 输出 390 帧（30 chunks × 13）
2. **Projector** `model.multi_modal_projector`：linear_1(896→896)→gelu→linear_2(896→1024)
3. **LLM** `model.language_model`：**标准 Qwen3-0.6B**（28 层, 1024 hidden, 16 头, KV 8 头, head_dim 128, FFN 3072, tie embeddings, vocab 151936）
   - 注意：5.14 的 Qwen3 带 **QK-Norm**（q_norm/k_norm 权重）
4. 生成：audio embeds 替换 `<|audio_pad|>`(151676) 占位符 → 标准自回归，输出 `language <NAME><asr_text>`，eos=[151643,151645]

## 移植方案（HF 标准权重 + llm_convert 单 bmodel）

**关键点**：HF 5.14 的 `Qwen3ASRForConditionalGeneration.language_model` 就是标准 Qwen3-0.6B，
把权重重命名为标准 Qwen3 布局（`model.language_model.*` → `model.*`，config `model_type=qwen3`）
后，**直接走 LLM-TPU 官方 `llm_convert.py` 标准 Qwen3 转换**（`--qwen_asr` 是官方 qwen_asr 包版权重
的专用分支，与 HF 权重不兼容）。audio embeds 通过 `<|audio_pad|>` 占位 token 走标准
`input_ids` + embedding 网络（非 inputs_embeds 注入），decode 侧 KV bf16 是 llm_convert 标准行为。

### 目录结构

```
Qwen3-ASR/
├── compile/                     # encoder 导出 + bmodel 编译
│   ├── make_models_llm_std.py   # HF 权重 → 标准 Qwen3 布局（生成 models_llm_std/）
│   ├── export_encoder_onnx.py   # encoder+projector 导出（--num_mel_frames 3000 全量）
│   ├── recompile_encoder.sh     # encoder bmodel 编译（W4BF16 + --disable_layer_group）
│   ├── recompile_encoder_3000.sh
│   ├── merge.sh                 # encoder + LLM 合并为单文件（固化 model_tool --combine）
│   └── tmp/                     # 编译中间物（不入库）
├── python/
│   └── infer_native.py          # 原生基线（输出格式参照）
├── cpp/                         # 纯 bmrt C++ 推理（交叉编译）
│   └── src/  (main / asr_engine / qwen_mel / tokenizer)
├── models/
│   ├── qwen3-asr-0.6b/          # HF 原始权重（不入库）+ 运行时小资产
│   │   └── prefix_ids.txt / suffix_ids.txt / mel_filters.npz / tokenizer.json
│   └── BM1684X/                 # 与 qwen3-asr-0.6b/ **平级**，不是它的子目录
│       ├── qwen3_asr_merged_w4g64.bmodel   # 646M（W4BF16 g64，对照档）
│       ├── qwen3_asr_merged_w4f16.bmodel   # 646M（W4F16 g64，默认部署档）
│       └── qwen3_asr_merged_w8bf16.bmodel  # 942M（精度回退档）
├── models_llm_std/              # llm_convert 用的标准 Qwen3 权重（重命名版，不入库）
├── models_llm_std_f16/          # W4F16 档用（config dtype 改 float16，不入库）
├── test_data/                   # 测试音频（16k mono）：13 个有效 + 2 个超 30s 上限的预期失败样本
├── test_outputs/                # 板上原始 bench 日志（已入库）
└── deploy_to_board.sh           # 一键部署 + 测试
```

### 编译权重准备（models_llm_std/）

`models_llm_std/` 是 llm_convert 编译用的**标准 Qwen3 权重**（HF 5.14 的 Qwen3ASR 权重重命名而来，
不入库，.gitignore 忽略）。**从 HF 原版权重（`models/qwen3-asr-0.6b/`）重建**：
```bash
cd compile && python make_models_llm_std.py   # 从 ../models 生成 ../models_llm_std
```
改动只有两处（脚本自动完成）：
1. **权重键去 `model.language_model.` 前缀**（`model.language_model.layers.*` → `model.layers.*`；
   audio_tower / multi_modal_projector 键丢弃，LLM 编译不需要）
2. **config.json 的 `model_type` 改 `qwen3`、`architectures` 改 `Qwen3ForCausalLM`**

### 编译 bmodel（一次性，容器 `sophon-tpumlir-v128`，TPU-MLIR v1.28.1）

容器内需 torch 2.4.1 + transformers 4.57.6（缺 qwen3 识别会 KeyError）。

> ⚠️ **所有命令先 `cd /workspace/Qwen3-ASR`**。`run_docker.sh` 把仓库根挂到 `/workspace` 并以此为 cwd，而下面的路径都是相对本模型目录的；不先 cd 会把产物落到仓库根（`.gitignore` 里的 `Qwen3-ASR/qwen3_asr_std*/` 规则即按此约定）。中间产物统一落 `compile/tmp/`，见 `.claude/standards/models_directory_standard.md` §3.1。

```bash
docker exec -it sophon-tpumlir-v128 bash
cd /workspace/Qwen3-ASR

# 0. 生成 llm_convert 用的标准 Qwen3 权重（若 models_llm_std/ 尚不存在）
cd compile && python make_models_llm_std.py && cd ..

# 1a. LLM 对照档 W4BF16 group64（646MB）：-g 64 必加，group 128 精度不足；KV bf16 自动
llm_convert.py -m models_llm_std -s 512 --quantize w4bf16 -g 64 -c bm1684x \
    --out_dir compile/tmp/qwen3_asr_std_w4g64_512

# 1b. LLM 默认部署档 W4F16 group64：config.json 的 dtype 必须为 float16，
#     权重目录用 make_models_llm_std.py 另行生成的 models_llm_std_f16
llm_convert.py -m models_llm_std_f16 -s 512 --quantize w4f16 -g 64 -c bm1684x \
    --out_dir compile/tmp/qwen3_asr_std_w4f16_512

# 2. 全量 encoder（110MB）：3000 帧输入 + W4BF16 + 必加 --disable_layer_group（离线/流式通吃）
cd compile
python export_encoder_onnx.py --model_path ../models --num_mel_frames 3000 \
    --out_dir ./tmp/onnx_enc3000
model_transform.py --model_name qwen3_asr_encoder \
    --model_def tmp/onnx_enc3000/qwen3_asr_encoder.onnx \
    --input_shapes [[1,128,3000]] --mlir tmp/encoder_full.mlir
model_deploy.py --mlir tmp/encoder_full.mlir --quantize W4BF16 --chip bm1684x \
    --disable_layer_group --model tmp/encoder_full.bmodel
cd ..

# 3. 合并单文件（646MB）：用已固化的 merge.sh，不要手敲 model_tool --combine
#    merge.sh 自带绝对路径默认输出、必填参数校验与产物存在性检查
bash compile/merge.sh \
    compile/tmp/encoder_full.bmodel \
    compile/tmp/qwen3_asr_std_w4g64_512 \
    qwen3_asr_merged_w4g64.bmodel
# W4F16 档同理，只换 LLM 目录与输出名：
bash compile/merge.sh \
    compile/tmp/encoder_full.bmodel \
    compile/tmp/qwen3_asr_std_w4f16_512 \
    qwen3_asr_merged_w4f16.bmodel
```

产物落在 `models/BM1684X/`（`merge.sh` 的默认输出路径）。encoder 的 `model_transform.py` / `model_deploy.py` 会把 npz/mlir/prototxt 等中间物写进**当前工作目录**，所以第 2 步必须在 `compile/` 下执行（其 `.gitignore` 已覆盖 `compile/*.npz` 等），不要在仓库根或 `models/` 下运行。

### 上板运行

```bash
bash cpp/build.sh                # sophon-cross-build 容器内交叉编译
BOARD_IP=<board_ip> bash deploy_to_board.sh --test
# 板上（单文件 bmodel）：
./qwen3_asr_bm1684x --bmodel models/BM1684X/qwen3_asr_merged_w4f16.bmodel \
    --model_dir . --audio test_data/test_zh.wav
```

## 上板测试结果（BM1684X SoC）

### 测试结果（test_data 全部 13 个音频，C++ bmrt 上板实测，2026-08-10，seq512）

| 音频 | 语种 | 时长 | 离线端到端 | 离线 RTF | 流式 Final |
|------|------|------|-----------|----------|------------|
| short_05s.wav | 中文 | 0.5s | 0.27s | 0.54 | ✅ `对。`（不吞） |
| short_1s.wav | 中文 | 1.0s | 0.31s | 0.31 | ✅ `对我做了介绍。` |
| test_ja.wav | 日语 | 4.7s | 0.45s | 0.096 | ✅ `すみません。駅はどこですか。` |
| test_de.wav | 德语 | 5.8s | 0.51s | 0.088 | ✅ `Entschuldigung, fährt der Zug nach Berlin.` |
| test_zh.wav | 中文 | 5.6s | 0.58s | 0.104 | ✅ `对我做了介绍啊，那么我想说的是呢，大家如果对我的研究感兴趣呢。` |
| test_en.wav | 英语 | 5.9s | 0.65s | 0.110 | ✅ `Mr. Quilter is the apostle of the middle classes...` |
| test_fr.wav | 法语 | 11.2s | 0.84s | 0.075 | ✅ `Bonjour les amis et bienvenue...` |
| en_long_1_16k.wav | 英语 | 10.5s | 0.93s | 0.089 | ✅ `Saffon Bremax is a high-performance AI inference chip...` |
| test_es.wav | 西语 | 16.0s | 0.98s | 0.061 | ✅ `Y hoy me gustaría hacer una excursión...` |
| test_en_p1.wav | 英语 | 28.8s | 1.14s | 0.040 | ✅ `Everyone, I'm Amanda, and I'm 25 years old...` |
| zh_long_1_16k.wav | 中文 | 19.9s | 1.53s | 0.077 | ✅ `语音合成技术经过多年发展，已经从早期的拼接式合成...` |
| test_en_p2.wav | 英语 | 28.8s | 1.85s | 0.064 | ✅ `Let's chat about this together...` |
| test_zh_18s.wav | 中文 | 18.1s | 2.28s | 0.126 | ✅ `如果说这个全国第一城市，上一次遭受的同类危机...` |

**Token 生成速度（多次运行稳定，波动 <1%）**：13~83 tokens 全覆盖 **64-65 tok/s**（与生成长度/KV 长度无关）。

**流式**：1s 起输出中间结果并逐步修订（`Every.` → `Everyone, I'm.` → ... → 完整），**Final 与离线逐字一致**；
每块 RTF 0.28-0.99（实时，含 28.8s 长音频）；0.5s 超短语音由 finish 兜底不吞。

**内存**：NPU heap 净占用 **~1.1GB**（646MB bmodel + 运行时 KV/激活；devmem 3.78GB 为全板卡统计，含其他进程）。

> **长音频限制**：encoder 固定 3000 mel 帧 = **30s 硬上限**（wav_to_mel 拒绝 >30s），KV（512 槽）不是瓶颈；超限明确报错（真实场景 VAD 切分一般 < 30s）。

## 关键经验与坑

1. **transformers 5.14**：processor 返回 float32 特征但模型权重 bf16，需显式转换；
   **mask 必须保持 fp32**（bf16 求和在 3000 处有舍入误差 3000→3008，破坏 cu_seqlens 窗口）
2. **audio encoder 默认 sdpa attention**（flash 风格，无法导出 ONNX）→ 导出用 eager 语义
3. **窗口 attention 的帧序**：原生是 transpose + cat(dim=1) 的**帧-major**（每行 = 一帧的 14 头），
   若 cat 在 head 维会完全错乱（cosine 0.36）
4. **encoder 输出打包**：原生 valid_indices 动态打包（nonzero 动态 shape TPU-MLIR 不支持）
   → 固定 3000 帧导出，host 侧按 `_get_audio_token_length` 公式截取
5. **mel pad 语义**：encoder 非因果窗口 attention，pad 帧会参与注意力 →
   mel **复制最后一帧**（≈静音延续）而非补零；**给 mel 补帧而非给音频补零**
6. **C++ stft 帧数**：`(len + 2*pad - n_fft)/hop + 1`，不能按 `(len+2*pad)/hop+1`
   （多算 3 帧越界读 heap 产生 NaN → encoder 全 NaN → 输出全 `!`）；
   reflect pad 右边界 = `a[n-2], a[n-3], ...`（不重复最后一个元素）
7. **C++ npz 解析**：numpy 2.x 的 npz 用 ZIP64 占位（csize=0xFFFFFFFF，实际在 extra field）；
   entry 名带 `.npy` 后缀
8. **bmrt 调用**：sail 的 SYSIO 模式 = `bmrt_launch_data`（host 内存直传）；
   普通 `bmrt_launch_tensor(_ex)` 也能用（LLM 的 launch_host 模式），
   但 **不要直接操作 `stages[0].input_mems`**（真实数据下会卡死板卡，Eureka 同款教训）
9. **decode mask**：mask 长度 SEQ+1，0..pos-1 可见、pos..SEQ-1 屏蔽无效槽位、
   **SEQ 位（本次新 key）保持 0 可见**；position_id = token_length（修正 Eureka 的 off-by-one）
10. **`bmrt_launch_tensor_ex` 是异步的**：最后两个参数是 `user_mem` / `user_stmode`（**不是 `is_sync`**）。sync 策略分层，详见知识库 §3.1：
    - **CPU 读结果前必须 `bm_thread_sync`**（无争议）
    - **层间是否 sync 取决于产物类型**：本模型走 `llm_convert` 标准产物（层融合、KV 独立），同线程 launch/d2d 进入驱动 FIFO 队列自然串行，**层间无需逐个 sync**（HY-MT 61/61 与官方 llm_tpu demo 一致）；`model_transform`/`model_deploy` 通用产物 + 手写 d2d 链则保守逐层 sync
    - 跨线程、或有 host 参与判断（s2d 覆盖等）时每步都要 sync
    - **排查线索**：读到 `0x7fff`（bf16 NaN）或上一次进程的残留值，说明在该 sync 的地方没 sync
    - ⚠️ 本条早期版本写作"每层 launch 后必须 bm_thread_sync"，那是 2026-08-08 的结论，**已在知识库 §6 被裁决推翻**；本模型正是 llm_convert 产物，不要再按旧结论加满 sync
11. **层间 d2d 必须直接写到下一层 in0**（`net_blocks_[i+1]->input_mems[0]`，QwenEngine 同款）；
    写到中转 buffer 不会生效（下一层读自己的 in0 是未初始化残留）→ layer1+ 全错
12. **各层 pos/mask 输入共享同一块 input_mem**（addr mode 1 下地址相同，已验证），只传一次即可；
    block 输入顺序 input_states(bf16)/position_ids(int32)/attention_mask(bf16)
13. **w4bf16 必须 `-g 64`**（llm_convert 默认 group 128）：0.6B 模型 4bit 量化误差（单层 ~0.05）
    经 28 层 attention/FFN 指数放大（layer1 就差 2.0）→ logits 翻转首 token；
    group 64 误差减半后精度恢复；w8bf16 也可（+210MB）。先本地噪声模拟验证量化方案再编译
14. **位置 0 输出爆炸**（hidden 值 5376+）：causal mask 下位置 0 只看自己，是模型固有行为
    （HF 原生同样），不影响生成（lm_head 用最后一个 token）
15. **优先用 1_third_party 现成库**：tokenizer 用 tokenizers-cpp（QwenLLM 同款，
    C API `tokenizers_decode` 支持 skip_special_token，需同时链接 libtokenizers_cpp.a + libtokenizers_c.a）；
    FFT 可考虑 kissfft/FFTW（aarch64 静态库）——本项目手写 radix-2 FFT 已达标（mel 0.1s）未替换

## 流式推理（--stream 模式）

流式（无 VAD）：音频持续灌入 + 模型持续吐字修订，`stream_finish` 定稿（音频流结束/静音检测后调用）。

```bash
./qwen3_asr_bm1684x --bmodel models/BM1684X/qwen3_asr_merged_w4f16.bmodel \
    --model_dir . --stream --audio test_data/test_zh.wav
```

实测（test_zh 5.6s，1s 块 + seq512 LLM）：
```
[ 0.3s] language English<asr_text>Every.                     (audio 1.0s | proc 0.28s | RTF 0.28)
[ 1.3s] language English<asr_text>Everyone, I'm.             (audio 2.0s | proc 0.99s | RTF 0.99)
[ 2.0s] language English<asr_text>Everyone, I'm Amanda, and I.  (audio 3.0s | proc 0.75s | RTF 0.75)
...（逐步修订增长）
[Final] language English<asr_text>Everyone, I'm Amanda, and I'm 25 years old...  ← 定稿与离线一致
```

**实时性**：每块 RTF 0.28-0.99 < 1 全程实时（含 28.8s 长音频）；中间结果随音频增长修订
（"延迟修正"行为），Final 与离线逐字一致。**1s 起输出**（官方 Python 实测 1s 可转写、0.5s 破碎），
更短语音由 finish 兜底不吞。

**机制**（对齐官方 Python/transformers 语义）：
- 每块全量 mel（pad 到 3000 帧）重编码（固定 0.08s）→ 按真实帧数截取 audio tokens（mel_frames_to_tokens）
- 全量 prefill + 短生成（32 token）→ 中间转写；`stream_finish()` 与 update 同一路径全量定稿
- **分段定稿（无限持续流式）**：audio tokens 将超 KV 上限（留 128 槽给生成，约每 28s 一段）时
  **自动定稿当前段（完整生成）→ 追加到历史（stream_partial_text_）→ 重置 mel/samples → 继续**；
  update/finish 返回"历史 + 当前段"，调用方无感知。实测 57.6s 音频自动切 3 段全部转写、
  段间无内容丢失。官方 demo（pipeline.py）是 KV 满直接 clear_history 丢进度，本方案更优
- **段边界静音对齐**：分段点双向搜索 ±1.5s 内最近静音（mel log10 帧均值 < -7 判静音），
  避免截断在词中间——说话有停顿时段边界自然衔接（实测拼接处插入 1s 静音后，段2 开头
  从 "And let's chat" 变为 "Let's chat"）；持续语音无静音则原样截断（可接受）

**C++ API**（asr_engine.h）：`init_stream() / stream_push(samples, n) / stream_update() / stream_finish()`

## 局限

- **音频长度**：**30s 硬上限**（encoder 固定 3000 mel 帧）；超限明确报错（真实场景 VAD 切分一般 < 30s）
- 流式：中间结果会随音频增长修订（"延迟修正"行为）；**<1s 不输出中间结果**（官方 Python 实测
  1s 可转写、0.5s 破碎），更短语音由 finish 兜底（不吞）；1s 块下全程实时（RTF < 1）
- **流式无 VAD**：音频持续灌入 + 模型持续吐字修订（`stream_update`），定稿由 `stream_finish`
  （音频流结束/静音检测后调用）；VAD 属于离线整段场景的句子边界判断，与流式无关
- 输入要求 **16k mono wav**（C++ 侧未实现重采样）