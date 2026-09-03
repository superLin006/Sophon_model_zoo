# Eureka-Audio — BM1684X 移植

把 Eureka-Audio（whisper-large-v3 encoder + Qwen3-1.7B，语音指令 → 动作 JSON）移植到算能 BM1684X SoC，端到端在板卡上推理。

> ## ⚠️ 当前状态：产物在，但不可部署、不可复现
>
> **两个 bmodel 产物完整保留**（whisper encoder 1.4G + qwen3 2.7G，共 ~4.1GB），历史上 Python·sail 与 C++ 两条路径都在板上跑通过。但：
>
> - **4 个运行时资产已不在本机**：`prefix_embeds.bin`、`suffix_embeds.bin`、`mel_filters.npz`、`tokenizer.json`
> - **源权重目录 `Eureka-Audio-Instruct` 已不在本机**（只剩 HF 的 `transformers_modules` 代码缓存）
>
> 因此 `deploy_to_board.sh` 会在 `require_file` 处直接中止，§2 的导出与资产生成命令也无法执行。**要恢复可部署状态，必须先取回 `Eureka-Audio-Instruct` 权重**，再按 §2.1 / §2.3 重新导出 ONNX 并生成那 4 个资产；bmodel 本身不需要重编（除非要改量化档位）。
>
> 以下按"资产齐备时"的正确流程编写，供恢复后直接使用。

## 验收结论

- **测试集**：`test_audios/intent/long_01~09.wav` —— 9 条 ChatTTS 合成的口语化长指令，贴近真实使用场景（而非短命令词）。期望动作见 `python/benchmark_board.py` 的 `EXPECTED` / `TEXTS`。
- **准确率：约 5-6/9**，与原版 PyTorch（GPU）基线持平。难点在合成音偏噪、口语化语义弱，**不是移植回归**。

  > ⚠️ **口径警告**：早期文档与汇总表里的「~90%」来自**另一套短命令词测试集**，该测试集已不在仓库。引用准确率必须带数据集，否则 90% 与 5-6/9 看起来互相矛盾。`PERF_SUMMARY.md` 已统一为 5-6/9 + 数据集标注。

- **已放弃的路线**：试过 `llm_convert.py` 层融合重编 qwen3（w4bf16/w8bf16），decode 提到 22-30 tok/s，但 **audio-embed 注入场景下语义有损**——同音频准确率从 5-6/9 降到 **3/9**。已放弃，保留下方 `model_deploy` 通用编译版。这条是知识库 §1.2「路线 B 的边界」的实证来源。
- **两个版本均可用**：C++（`cpp/`）和 Python（`python/`），输出一致。

## 性能（BM1684X SoC，whisper F16 + qwen3 W4BF16，C++ 版）

9 句平均：

| 阶段 | 耗时 |
|------|------|
| whisper encoder（mel → audio_embeds） | 0.64 s/case |
| qwen3 prefill（~120 token，28 层） | 0.67 s/case |
| qwen3 decode | 16.4 tok/s（约 16 token/句） |
| **端到端** | **~2.3 s/case** |
| 模型加载（一次性，预热后） | ~8 s |

固定开销（whisper + prefill）约 1.3s，decode 每输出 token 约 61ms；快的句子（10 tok）约 1.9s，慢的（26 tok）约 2.9s。

decode 性能历程：初版 host KV 每步传 112MB → 6.2 tok/s；优化为 KV 常驻 device + IO buffer 复用后 → **16 tok/s（2.6×）**。实测瓶颈是 28 层 block_cache 的 launch + 计算固有开销（~2.2ms/层/token），数据层优化已到顶；与 QwenLLM 独立版曾达到的 29 tok/s 的差距来自 `llm_convert.py` 的层融合编译优化（通用 `model_deploy` 流程达不到）。

> ⚠️ **文件名陷阱**：`whisper_encoder_b1_bf16.bmodel` 实际是 **F16** 量化。BF16 对 attention 精度损失过大（cosine 仅 0.51，F16 为 0.99），编译时改用了 F16 但**输出文件名保留了 `_bf16` 后缀**——改名会牵动 `deploy_to_board.sh` 与 C++ 默认参数。**不要按文件名判断精度。**

---

## 1. 目录结构

```
Eureka-Audio/
├── compile/                    # 模型导出 + bmodel 编译
│   ├── export_whisper_encoder.py   # 导出 whisper encoder + audio_adaptor 为 ONNX
│   ├── export_qwen3_embeds.py      # 导出 qwen3（inputs_embeds 版）为 ONNX
│   ├── gen_prefix_embeds.py        # 生成 prefix/suffix_embeds.bin（系统提示词的 embed）
│   ├── recompile_whisper.sh        # 编译 whisper bmodel（F16，文件名保留 _bf16）
│   ├── recompile_qwen3.sh          # 编译 qwen3 bmodel（W4BF16）
│   └── verify_onnx.py              # 校验 ONNX 与 PyTorch 数值一致
├── cpp/                        # C++ 推理（纯 bmrt），产物 cpp/build/eureka_audio_bm1684x
├── python/                     # Python 推理（sophon.sail，建议先用这个验证 bmodel）
│   ├── infer_board.py              # 单音频推理
│   └── benchmark_board.py          # 批量 + 准确率（含 EXPECTED / TEXTS）
├── eureka_infer/               # 原版 PyTorch 推理封装（对照基线用）
├── models/BM1684X/             # 编译好的 bmodel（不入 git）
│   ├── whisper_encoder_b1_bf16.bmodel                  # 1.4G，实为 F16 量化
│   └── qwen3_1.7b_embeds_w4bf16_seq512_bm1684x.bmodel  # 2.7G
├── test_audios/                # qa_example.wav、asr_example.wav、intent/long_01~09.wav（入库）
├── deploy_to_board.sh
└── requirements.txt
```

**运行时还需要 4 个资产**（当前缺失，见顶部状态说明）：`prefix_embeds.bin`、`suffix_embeds.bin`、`mel_filters.npz`、`tokenizer.json`。它们来自 `Eureka-Audio-Instruct` 权重目录，其中 prefix/suffix 由 `compile/gen_prefix_embeds.py` 生成。

---

## 2. 编译模型（主机，一次性）

> 已有编译好的 bmodel 可跳过本节。环境：主机有 GPU + `sophon-eureka-audio` conda 环境（含 torch/transformers），TPU-MLIR 容器 `sophon-tpumlir-v128`（镜像 `sophon/tpuc_dev:v3.4-tpumlir-1.28.1`，TPU-MLIR v1.28.1）。
>
> **前置：`../Eureka-Audio-Instruct` 权重目录必须存在**，否则 §2.1 与 §2.3 都会失败。

### 2.1 导出 ONNX

```bash
cd compile
conda run -n sophon-eureka-audio python export_whisper_encoder.py --model_path ../../Eureka-Audio-Instruct
conda run -n sophon-eureka-audio python export_qwen3_embeds.py    --model_path ../../Eureka-Audio-Instruct
```

### 2.2 编译 bmodel（容器内）

```bash
docker exec sophon-tpumlir-v128 bash -c "cd /workspace/Eureka-Audio/compile && bash recompile_whisper.sh"
docker exec sophon-tpumlir-v128 bash -c "cd /workspace/Eureka-Audio/compile && bash recompile_qwen3.sh"
```

产物在 `models/BM1684X/`。注意 qwen3 这一档**加了 `--disable_layer_group` 反而 SHA 校验失败**，是知识库 §1.4「个别模型加了反而失败」的反例；脚本里已按实测配置，不要照通用规则去加。

### 2.3 生成 prefix/suffix embeds

系统提示词的 token embedding 离线算好存成 `.bin`（C++/Python 直接读，板卡上不需要 embed_tokens 权重）：

```bash
conda run -n sophon-eureka-audio python gen_prefix_embeds.py \
  --model_path ../../Eureka-Audio-Instruct \
  --output_path ../../Eureka-Audio-Instruct/prefix_embeds.bin
```

> ⚠️ **关键坑（已修复，改 prompt 必须重新理解）**：
> - whisper encoder 手写 forward 的 attention scale 只能乘到 q（不能 q、k 都乘），否则等价除以 D 而非 √D，softmax 过平导致语义失真（cosine 0.52）。
> - whisper bmodel 必须用 **F16** 量化（不是 BF16），BF16 对 attention 精度损失过大（cosine 0.51）。`recompile_whisper.sh` 已用 F16（输出文件名仍叫 `..._bf16.bmodel`，未改名）。
> - suffix 文本必须为：
>
>   ```text
>   <|fim_prefix|>
>   </think>
>
>   ```
>
>   `audio_end` 后**无换行**，且 `enable_thinking=False` 时 Qwen3 模板会追加 `<think>\n\n</think>`。漏了首 token 直接 EOS。

---

## 3. 部署到板卡

**统一用部署脚本**（自带逐文件 md5 校验，见 `.claude/standards/board_deploy_workflow.md`）：

```bash
BOARD_IP=<board_ip> bash Eureka-Audio/deploy_to_board.sh --test
```

可用环境变量：`BOARD_USER`、`BOARD_PORT`、`BOARD_DIR`（默认 `/data/eureka_audio`）、`ASSET_DIR`（默认 `../Eureka-Audio-Instruct`，即 4 个运行时资产的来源目录）、`BINARY`、`TEST_AUDIO`（默认 `test_audios/qa_example.wav`）、`BOARD_PASS`。

脚本上传后的板上布局：

```text
/data/eureka_audio/
├── eureka_audio_bm1684x
├── prefix_embeds.bin  suffix_embeds.bin  mel_filters.npz  tokenizer.json   # 平铺在根
├── models/BM1684X/
│   ├── whisper_encoder_b1_bf16.bmodel
│   └── qwen3_1.7b_embeds_w4bf16_seq512_bm1684x.bmodel
└── test_data/test.wav            # 由 test_audios/qa_example.wav 重命名而来
```

> ⚠️ **脚本只上传 1 条测试音频，也不上传 `python/*.py`**。要跑 9 条意图 benchmark（Python 路径）需另行上传：
>
> ```bash
> scp Eureka-Audio/python/*.py root@<board_ip>:/data/eureka_audio/
> scp -r Eureka-Audio/test_audios/intent root@<board_ip>:/data/eureka_audio/test_data/intent
> ```
>
> 板上目录名统一用 **`test_data/intent`**。早期文档里出现过的 `intent_wav` 是同一批音频的第三个名字，已废弃。

**不要手写裸 scp 部署 bmodel**：4.1GB 大文件传输会静默损坏（大小对、内容错），跳过 md5 校验会让"板卡行为诡异"极难定位。

C++ 二进制单独编译：

```bash
bash Eureka-Audio/cpp/build.sh   # 产物 cpp/build/eureka_audio_bm1684x
```

---

## 4. 在板卡上运行

> **运行前先预热**：eMMC 冷读 2.7G bmodel 要 ~60s，进程会进 D(disk sleep) 像卡死。先 `cat models/BM1684X/*.bmodel >/dev/null` 把 bmodel 读进 page cache，后续加载秒级。

### Python（建议先用这个验证 bmodel）

```bash
cd /data/eureka_audio
TMPDIR=/data/tmp python3 infer_board.py \
  --whisper models/BM1684X/whisper_encoder_b1_bf16.bmodel \
  --qwen3   models/BM1684X/qwen3_1.7b_embeds_w4bf16_seq512_bm1684x.bmodel \
  --model_dir . --audio test_data/test.wav

# 批量 + 准确率（9 条意图）
TMPDIR=/data/tmp python3 benchmark_board.py \
  --whisper models/BM1684X/whisper_encoder_b1_bf16.bmodel \
  --qwen3   models/BM1684X/qwen3_1.7b_embeds_w4bf16_seq512_bm1684x.bmodel \
  --model_dir . --audio_dir test_data/intent
```

依赖（板卡自带，已实测）：`numpy 1.23.2` / `scipy 1.10.1` / `tokenizers 0.20.3` / `sophon.sail`。**导入方式是 `import sophon.sail as sail`，裸 `import sail` 会失败**（板上由 `sophon-arm-pcie` 提供 `sophon/sail.*.so`）。无需 torch/librosa，mel 用纯 numpy 算。

`TMPDIR=/data/tmp` 是必需的：板卡根分区 overlay 空间小，默认 `/tmp` 在根分区上。

### C++

```bash
cd /data/eureka_audio
# 单个音频
./eureka_audio_bm1684x \
  --whisper_bmodel models/BM1684X/whisper_encoder_b1_bf16.bmodel \
  --qwen3_bmodel   models/BM1684X/qwen3_1.7b_embeds_w4bf16_seq512_bm1684x.bmodel \
  --model_dir . --audio test_data/test.wav --max_new_tokens 64

# 批量目录（bmodel 只加载一次，循环推理 + 分段性能汇总）
./eureka_audio_bm1684x \
  --whisper_bmodel models/BM1684X/whisper_encoder_b1_bf16.bmodel \
  --qwen3_bmodel   models/BM1684X/qwen3_1.7b_embeds_w4bf16_seq512_bm1684x.bmodel \
  --model_dir . --audio_dir test_data/intent
```

输出示例：

```
[test_data/intent/long_01.wav]
  Output: {"action":"open_whiteboard","params":{}}
  Perf:   whisper=0.63s  prefill=0.67s(127tok)  decode=1.58s(26tok, 16.4 tok/s)
```

---

## 5. 音频要求

- 输入 16k 单声道 WAV（PCM16/32 或 IEEE float，`scipy.io.wavfile` 读）。
- mp3 需在主机转 wav：`conda run -n sophon-eureka-audio python` + `librosa.load(mp3, sr=16000, mono=True)` → `soundfile.write(..., subtype='PCM_16')`。
- 音频 token 数按实际时长截取：`real_frames = len(wav)//1280`，只取前 real_frames 个 audio embed（不能用 pad 到 30s 的全 375 个，否则静音淹没语义、输出乱码）。
- KV cache 上限 SEQLEN=512：prefix(68) + audio + suffix(10) 后剩余额度给 decode。音频太长（audio token 多）会挤占 decode 空间。

---

## 6. 已知问题与排查经验

- **C++ 用裸 bmrt 必须用 whisper/sail 范式**：每个 net 用 `bmrt_tensor` 独立分配 IO + 全程 host↔device 往返（`bm_memcpy_s2d/d2s`）+ KV cache 存 host。**禁止**直接操作 net 内置的 `stages[0].input_mems`、**禁止** net 间 device 端 d2d 直传——那样在真实数据下会踩 bmrt 内部状态导致**板卡硬重启**（零数据不崩、真实数据崩，极难定位）。范本见 `../whisper/cpp/src/whisper_inference.cpp`。

  > 注：知识库 §6 后来把"禁止操作 input_mems"软化为**排查建议**——d2d 写下一层 in0 在多数项目（Qwen3-ASR / Qwen3-TTS / llm_tpu）是稳定路径。但 Eureka 这个案例确实踩过板卡 reboot，遇到重启仍应优先隔离此项。
- **device memory**：`bm_get_stat` 实测 9070MB 可用，两个 bmodel（1.4+2.7=4.1G）共存够用。ion debug 接口报的 npu heap 3.86G 是单分区，不是 bmrt 上限。
- **精度排查方法**：逐级 cosine 对比 mel → encoder → audio_embeds，再分 ONNX vs bmodel，定位是导出 bug 还是量化损失。对照基准用原版 PyTorch（`eureka_infer.api.EurekaAudio`）。
- **板卡地址**由 `BOARD_IP` 等环境变量提供，不写入仓库。网络瞬断会自动恢复；判断是否真的重启过看 `uptime`（开机时间短 = 刚重启）。