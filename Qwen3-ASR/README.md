# Qwen3-ASR-0.6B — BM1684X 移植

[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 的 Qwen3-asr-0.6B（30 语种 + 22 中文方言的语音识别 + 语种识别）完整移植到 Sophon BM1684X，中英文转写与原生基线一致。

## 🏆 当前推荐：LLM-TPU 官方工具链方案（--qwen_asr）

**结论：官方工具链全面优于自定义 ONNX 导出方案**（体积 -51%、内存 -49%、decode +83%），
采用 ModelScope `Qwen/Qwen3-ASR-0.6B`（qwen_asr 包版权重，thinker_config + mrope）：

```bash
# 编译（docker sophon-tpumlir，TPU-MLIR v1.28.1）
# 前置：pip3 install "transformers==4.57.6" "qwen-asr" "torch==2.4.1"（CPU index）
#       权重目录不能有 .bin 文件（LlmLoad 会误当 torch 权重）
llm_convert.py -m <qwen_asr_weights> -s 512 --max_input_length 256 \
    --quantize w4bf16 -c bm1684x --out_dir qwen3_asr_official --qwen_asr

# 上板（官方 C++ demo，板卡 python3.8 需重编 chat 模块）
#   g++ -O2 -std=c++17 -fPIC -shared -I <pybind11_include> -I /usr/include/python3.8 \
#       -I /opt/sophon/libsophon-current/include chat.cpp -o chat.cpython-38-aarch64-linux-gnu.so \
#       -L /opt/sophon/libsophon-current/lib -lbmrt -lbmlib
./qwen3_asr_demo <bmodel> <config_dir> <wav> [language]   # 官方 cpp_demo
```

**官方 bmodel 结构**（60 网络单文件）：`audio`（chunk encoder：100 mel 帧 → 13 embeds，内置）+ `embedding` + `block_i` + `block_cache_i`（**KV bf16**）+ `lm_head`。

**实测性能**（test_zh 5.6s）：audio 0.04s + prefill 0.06s + decode 64.5 tok/s → 端到端 ~0.5s（**RTF ~0.09**），FTL 97ms。

> 早期自定义方案（5.14 权重 + ONNX 导出 + 双 encoder bmodel）保留在 `compile/`、`cpp/` 作为备选与参考。



## 模型结构（探索结论）

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

## 移植方案（Eureka-Audio 同款路线）

**关键点：LLM 用 inputs_embeds 版 bmodel**（prefill 直接吃拼好的连续向量），
不能用 llm_convert.py 标准 token-id 版（audio embeds 是连续向量，无法走 embedding 查表；
且 Eureka 验证过 llm_convert.py 层融合版在 audio-embed 注入场景语义有损）。

### 目录结构

```
Qwen3-ASR/
├── compile/                     # 模型导出 + bmodel 编译
│   ├── export_qwen3_embeds.py   # LLM 导出为 inputs_embeds 版多网络 ONNX
│   ├── export_encoder_onnx.py   # encoder+projector 导出（固定 3000 帧窗口）
│   ├── gen_prefix_embeds.py     # 从原生 dump 的 input_ids 自动生成 prefix/suffix embeds
│   ├── verify_encoder_onnx.py   # ONNX 与原生数值校验（cosine）
│   ├── recompile_qwen3.sh       # LLM bmodel 编译（W4BF16）
│   └── recompile_encoder.sh     # encoder bmodel 编译（F16 + --disable_layer_group）
├── python/
│   ├── infer_native.py          # M0 原生基线 + dump（input_ids 布局 / mel / filter bank）
│   └── infer_board.py           # sail 上板推理
├── cpp/                         # 纯 bmrt C++ 推理（交叉编译）
│   └── src/  (main / asr_engine / qwen_mel / tokenizer)
├── models/                      # 权重（不入库）+ 编译好的 bmodel
│   ├── BM1684X/
│   │   ├── qwen3_asr_encoder_F16.bmodel           # 363M
│   │   └── qwen3_asr_llm_w4bf16_seq256_bm1684x.bmodel  # 1.1G（推荐：decode 35 tok/s，≤8s 音频）
│   ├── prefix_embeds.bin / suffix_embeds.bin / mel_filters.npz / tokenizer.json
├── test_data/                   # 测试音频（16k mono）
└── deploy_to_board.sh           # 一键部署 + 测试
```

### 编译 bmodel（主机，一次性）

```bash
# 环境：qwen3-asr conda env（transformers>=5.13.0, torch, onnx）
# docker sophon-tpumlir（TPU-MLIR v1.28.1）

# 1. LLM 导出 + 编译（28+28 层，W4BF16，不加 --disable_layer_group；seq 用 1024 性能最佳）
cd compile && python export_qwen3_embeds.py --model_path ../models --seq_length 256 --out_dir ./tmp/onnx_seq256
docker exec sophon-tpumlir bash -c 'cd /workspace/Qwen3-ASR/compile && bash recompile_qwen3.sh'

# 2. encoder 导出 + 编译（F16，必加 --disable_layer_group）
python export_encoder_onnx.py --model_path ../models
python verify_encoder_onnx.py  # cosine > 0.999
docker exec sophon-tpumlir bash -c 'cd /workspace/Qwen3-ASR/compile && bash recompile_encoder.sh'

# 3. 生成 prefix/suffix embeds（先跑原生推理得到 dump）
python infer_native.py --audio ../test_data/test_zh.wav --out_prefix baseline_zh
python gen_prefix_embeds.py --dump_json ../python/dump/baseline_zh_inputs.json
```

### 上板运行

```bash
bash deploy_to_board.sh --test   # 部署 + python/sail 测试

# C++（先交叉编译）
bash cpp/build.sh                # sophon-cross-build 容器内编译
bash deploy_to_board.sh          # 重新部署（含 C++ 二进制）
# 板上：
./qwen3_asr_bm1684x --encoder_bmodel models/BM1684X/qwen3_asr_encoder_F16.bmodel \
    --qwen3_bmodel models/BM1684X/qwen3_asr_llm_w4bf16_seq256_bm1684x.bmodel \
    --model_dir . --audio test_data/test_zh.wav
```

## 上板测试结果（BM1684X SoC，172.16.25.248）

### 三端输出对比

| 音频 | 原生 transformers (CPU) | python/sail 上板 | C++ bmrt 上板 |
|------|------------------------|------------------|---------------|
| test_zh (5.6s) | `language Chinese<asr_text>对我做了介绍啊。那么我想说的是呢，大家如果对我的研究感兴趣呢。` | `language Chinese<asr_text>对我做了介绍啊，那么我想说的是呢，大家如果对我的研究感兴趣呢。` | `language Chinese对我做了介绍啊，那么我想说的是呢，大家如果对我的研究感兴趣呢。` |
| test_en (5.9s) | `language English<asr_text>Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.` | 同左（逐字一致） | `language EnglishMr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.`（`<asr_text>` 被 C++ tokenizer 跳过） |

语种识别 Chinese/English 全部正确；转写仅 w4bf16 量化导致个别标点差异（原生"啊。" vs 上板"啊，"）。

### 性能（C++ 版，板上实测，test_zh 5.6s / test_en 5.9s）

| 版本 | mel | encoder | prefill | decode | 端到端 | RTF |
|------|-----|---------|---------|--------|--------|-----|
| 初版（朴素 DFT mel + seq2048） | 5.15s | 0.08s | 2.61s | 8.1 tok/s | 10.44s | 1.88 |
| FFT mel + seq2048 | 0.09s | 0.08s | 2.57s | 8.1 tok/s | 5.33s | 0.95 |
| FFT mel + seq1024 + 基础 prefill | 0.10s | 0.08s | 1.02s | 14.1 tok/s | 2.68s | 0.48 |
| FFTW mel + seq512 + prefill 优化 | 0.09s | 0.08s | 0.17s | 23.0 tok/s | 1.26s | 0.23 |
| **FFTW mel + seq256（自定义方案）** | 0.09s | 0.08s | 0.08s | 35.3 tok/s | 0.85s | 0.15 |
| **官方工具链（当前推荐，--qwen_asr）** | 0.09s | **0.04s** | **0.06s** | **64.5 tok/s** | **~0.5s** | **~0.09** |

**优化历程**
1. **mel 提速 57 倍**（5.15s → 0.09s）：朴素 DFT → FFT（最终用 **FFTW3**，1_third_party 现成库，chatTTS 同款）
2. **seq 重编**（2048 → 512/1024）：decode 8.1 → 22.7 tok/s（KV 传输量 ∝ seq：448MB/步 → 112MB/步）
3. **prefill ping-pong 优化**（2.61s → 0.17s）：hidden device 内接力 + pos/mask 复用 + KV d2d 免 host 中转；
   seq512 的 mask 仅 1MB（seq 越小 attention 无效计算越少）
4. **tokenizer 换现成 tokenizers-cpp**（1_third_party，QwenLLM 同款），删除手写 BPE
5. **seq 缩减到 256**（decode 23 → 35.3 tok/s）：KV 传输 112MB → 56MB/步 + attention 计算减半；
   代价是音频上限降到 ~8s（prefix 9 + audio 104 + 6 + 生成 ≤ 256）——短音频/流式场景最优

> **多核编译验证结论**：`model_deploy.py --num_core 8` 对 LLM block 不生效（bmodel core num 仍为 1，
> LLM-TPU 的"多芯"是 --num_device 多卡方案，非单卡多核）；BM1684X 单卡 LLM matmul 多核切分收益有限。

**与仓库其他 ASR 模型对比（RTF 越小越好，<1 即实时）**

| 模型 | RTF | 端到端（~5.6s 音频） | 特点 |
|------|-----|---------------------|------|
| SenseVoice Small F16 | 0.0094 | 53ms | CTC 非自回归，最快 |
| Moonshine streaming-small | 0.045 | 296ms | 轻量英文流式 |
| Whisper turbo W4F16 | 0.343 | 2.4s | 多语言，自回归 |
| **Qwen3-ASR-0.6B（本移植）** | **0.09** | **~0.5s** | LLM 类 ASR，30 语种 + 22 方言 + 语种识别（官方工具链，FTL 97ms） |

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
10. **优先用 1_third_party 现成库**：tokenizer 用 tokenizers-cpp（QwenLLM 同款，
    C API `tokenizers_decode` 支持 skip_special_token，需同时链接 libtokenizers_cpp.a + libtokenizers_c.a）；
    FFT 可考虑 kissfft/FFTW（aarch64 静态库）——本项目手写 radix-2 FFT 已达标（mel 0.1s）未替换

## 流式推理（--stream 模式）

对齐官方（qwen-asr 的 streaming API：1s chunk + 5s 重编码窗口 + 延迟修正）：

```bash
# 板上：按 2s 块喂音频（2s 更新的处理耗时 ~1.7s ≤ 2s → 实时）
./qwen3_asr_bm1684x --encoder_bmodel models/BM1684X/qwen3_asr_encoder_F16.bmodel \
    --qwen3_bmodel models/BM1684X/qwen3_asr_llm_w4bf16_seq256_bm1684x.bmodel \
    --model_dir . --stream \
    --stream_encoder models/BM1684X/qwen3_asr_encoder_w500_F16.bmodel \
    --audio test_data/test_zh.wav
```

实测（test_zh 5.6s，1s 块 + seq256 + 官方 32 token + mel 增量）：
```
[ 1.1s] !!!!!...   (proc 1.04s | RTF 1.04)   ← 早期块满 32 token（语义未完整，未定稿可修订）
[ 2.1s] !!!!!...   (proc 1.04s | RTF 1.04)
[ 2.9s] language Chinese<asr_text>对我做了介绍啊。那么...   (proc 0.72s | RTF 0.72)   ← EOS 早停，消化积压
[Final] language Chinese<asr_text>对我做了介绍啊，那么...  ← 定稿与基线一致
```

**实时性实测（官方配置：1s 块 + 32 token）**：**平均 RTF 0.64（zh）/ 0.65（en）< 1 实时**——
早期块（语义未完整）满 32 token 略超（RTF 1.04），后续块 EOS 早停快（0.72-0.84s）消化积压，
流水线稳态实时（真实流式无累积延迟）。mel 增量：只算新帧 FFT + 全局重裁剪（长音频收益大）。

**机制**（对齐官方）：
- 音频按 chunk 累积（100 mel 帧 = 1s），update 时用**窗口 encoder（500 帧 = 5s，`qwen3_asr_encoder_w500_F16.bmodel`）**重编码最近 5s（跨 chunk 上下文 + 未固化修正）
- 累积 audio tokens（旧帧固定 + 新窗口替换）→ LLM prefill + 短生成（32 token）→ 中间转写
- `stream_finish()` 用**离线 encoder 全量重编码**定稿（精度与离线版一致）

**C++ API**（asr_engine.h）：`init_stream() / stream_push(samples, n) / stream_update() / stream_finish()`

## 局限

- 离线：音频 ≤ 30s（encoder 固定 3000 mel 帧）；长音频需分块（块间无上下文，Eureka 同款限制）
- 流式：中间结果会随音频增长修订（官方同款"延迟修正"行为）；<3s 不输出；2s 更新频率下实时
- 输入要求 16k mono wav（C++ 侧未实现重采样）
