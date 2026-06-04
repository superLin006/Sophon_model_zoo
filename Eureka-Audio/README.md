# Eureka-Audio 部署到算能 BM1684X

把 Eureka-Audio（whisper-large-v3 encoder + Qwen3-1.7B，语音指令 → 动作 JSON）移植到算能 BM1684X SoC，端到端在板卡上推理。

- **测试集**：`test_audios/intent/long_01~09.wav`——ChatTTS 合成的口语化长指令，贴近真实使用场景（而非短命令词）。期望动作见 `python/benchmark_board.py` 的 `EXPECTED`/`TEXTS`。
- **准确率**：本方案在该 ChatTTS 长句集上约 5-6/9，与原版 PyTorch（GPU）持平；难点在合成音偏噪、口语化语义弱，非移植回归。
- **decode 提速实验结论**：试过 `llm_convert.py` 层融合重编 qwen3（w4bf16/w8bf16），decode 提到 22-30 tok/s，但 audio-embed 注入场景下语义有损（同音频准确率 5-6/9→3/9），**已放弃，保留下方 model_deploy 通用编译版**。
- **两个版本均可用**：C++（`cpp/`）和 Python（`python/`），输出一致。
- **性能**（BM1684X SoC，F16 whisper + W4BF16 qwen3，C++ 版）：

  实测 9 句平均（C++ 版）：

  | 阶段 | 耗时 |
  |------|------|
  | whisper encoder（mel→audio_embeds） | 0.64 s/case |
  | qwen3 prefill（~120 token，28 层） | 0.67 s/case |
  | qwen3 decode | 16.4 tok/s（约 16 token/句）|
  | **端到端** | **2.3 s/case** |
  | 模型加载（一次性，预热后） | ~8 s |

  > 固定开销（whisper+prefill）~1.3s，decode 每输出 token ~61ms；快的句子(10 tok)~1.9s，慢的(26 tok)~2.9s。

  > decode 性能历程：初版 host KV 每步传 112MB → 6.2 tok/s；优化后 KV 常驻 device + IO buffer 复用 → **16 tok/s（2.6倍）**。
  > 实测瓶颈是 28 层 block_cache 的 launch+计算固有开销（~2.2ms/层/token），数据层优化已到顶；与 QwenLLM 独立版 29 tok/s 的差距来自 llm_convert.py 的层融合编译优化（通用 model_deploy 流程达不到）。

---

## 1. 目录结构

```
Eureka-Audio/
├── compile/                    # 模型导出 + bmodel 编译
│   ├── export_whisper_encoder.py   # 导出 whisper encoder + audio_adaptor 为 ONNX
│   ├── export_qwen3_embeds.py      # 导出 qwen3（inputs_embeds 版）为 ONNX
│   ├── gen_prefix_embeds.py        # 生成 prefix/suffix_embeds.bin（系统提示词的 embed）
│   ├── recompile_whisper.sh        # 编译 whisper bmodel（已改 F16，见下）
│   ├── recompile_qwen3.sh          # 编译 qwen3 bmodel（W4BF16）
│   └── verify_onnx.py              # 校验 ONNX 与 PyTorch 数值一致
├── cpp/                        # C++ 推理（纯 bmrt）
│   ├── src/                        # main / eureka_audio / whisper_mel / tokenizer
│   └── build.sh                    # docker 交叉编译
├── python/                     # Python 推理（sophon.sail，推荐先用这个验证）
│   ├── infer_board.py              # 单音频推理
│   └── benchmark_board.py          # 批量 + 准确率
├── models/BM1684X/             # 编译好的 bmodel（不入 git）
│   ├── whisper_encoder_b1_bf16.bmodel   # 1.4G（F16 量化，名字保留 _bf16 后缀）
│   └── qwen3_1.7b_embeds_w4bf16_seq512_bm1684x.bmodel  # 2.7G
└── (运行时需要) prefix_embeds.bin / suffix_embeds.bin / mel_filters.npz / tokenizer.json
```

---

## 2. 编译模型（主机，一次性）

> 已有编译好的 bmodel 可跳过本节。环境：主机有 GPU + `eureka-audio` conda 环境（含 torch/transformers），TPU-MLIR docker 容器 `sophon-tpumlir`（镜像 `sophgo/tpuc_dev`，TPU-MLIR v1.28.1）。

### 2.1 导出 ONNX

```bash
cd compile
conda run -n eureka-audio python export_whisper_encoder.py --model_path ../../Eureka-Audio-Instruct
conda run -n eureka-audio python export_qwen3_embeds.py    --model_path ../../Eureka-Audio-Instruct
```

### 2.2 编译 bmodel（docker 内）

```bash
docker start sophon-tpumlir
docker exec sophon-tpumlir bash -c "cd /workspace/Eureka-Audio/compile && bash recompile_whisper.sh"
docker exec sophon-tpumlir bash -c "cd /workspace/Eureka-Audio/compile && bash recompile_qwen3.sh"
```

产物在 `models/BM1684X/`。

### 2.3 生成 prefix/suffix embeds

系统提示词的 token embedding 离线算好存成 .bin（C++/Python 直接读，板卡上不需要 embed_tokens 权重）：

```bash
conda run -n eureka-audio python gen_prefix_embeds.py \
  --model_path ../../Eureka-Audio-Instruct \
  --output_path ../../Eureka-Audio-Instruct/prefix_embeds.bin
```

> ⚠️ **关键坑（已修复，改 prompt 必须重新理解）**：
> - whisper encoder 手写 forward 的 attention scale 只能乘到 q（不能 q、k 都乘），否则等价除以 D 而非 √D，softmax 过平导致语义失真（cosine 0.52）。
> - whisper bmodel 必须用 **F16** 量化（不是 BF16），BF16 对 attention 精度损失过大（cosine 0.51）。`recompile_whisper.sh` 已用 F16（输出文件名仍叫 `..._bf16.bmodel`，未改名）。
> - suffix 文本必须是 `<|audio_end|><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`：audio_end 后**无换行**，且 `enable_thinking=False` 时 Qwen3 模板会追加 `<think>\n\n</think>`。漏了首 token 直接 EOS。

---

## 3. 部署到板卡

### 3.1 上传文件

```bash
# 板卡 IP 是 eth1 DHCP 分配（每次重启会变，用 `ip addr show eth1` 查），SSH 端口 22
BOARD=root@<板卡IP>
ssh $BOARD "mkdir -p /data/eureka_audio/models/BM1684X"
scp models/BM1684X/*.bmodel $BOARD:/data/eureka_audio/models/BM1684X/
scp ../Eureka-Audio-Instruct/{prefix_embeds.bin,suffix_embeds.bin,mel_filters.npz,tokenizer.json} $BOARD:/data/eureka_audio/
scp python/*.py $BOARD:/data/eureka_audio/
# C++ 二进制（见 3.2 编译后）
scp cpp/build/eureka_audio_bm1684x $BOARD:/data/eureka_audio/
```

### 3.2 C++ 交叉编译（主机）

```bash
bash cpp/build.sh   # docker 内交叉编译，产物 cpp/build/eureka_audio_bm1684x
```

---

## 4. 在板卡上运行

> **运行前先预热**：eMMC 冷读 2.7G bmodel 要 ~60s，进程会进 D(disk sleep) 像卡死。先 `cat models/BM1684X/*.bmodel >/dev/null` 把 bmodel 读进 page cache，后续加载秒级。

### Python（推荐先验证）

```bash
cd /data/eureka_audio
TMPDIR=/data/tmp python3 infer_board.py \
  --whisper models/BM1684X/whisper_encoder_b1_bf16.bmodel \
  --qwen3   models/BM1684X/qwen3_1.7b_embeds_w4bf16_seq512_bm1684x.bmodel \
  --model_dir . --audio qa_example.wav

# 批量 + 准确率
TMPDIR=/data/tmp python3 benchmark_board.py ... --audio_dir intent_wav
```

依赖（板卡自带）：`numpy / scipy / tokenizers / sophon.sail`（无需 torch/librosa，mel 用纯 numpy 算）。

### C++

```bash
cd /data/eureka_audio
# 单个音频
./eureka_audio_bm1684x \
  --whisper_bmodel models/BM1684X/whisper_encoder_b1_bf16.bmodel \
  --qwen3_bmodel   models/BM1684X/qwen3_1.7b_embeds_w4bf16_seq512_bm1684x.bmodel \
  --model_dir . --audio qa_example.wav --max_new_tokens 64

# 批量目录（bmodel 只加载一次，循环推理 + 分段性能汇总）
./eureka_audio_bm1684x ... --audio_dir intent_wav
```

输出示例：

```
[intent_wav/long_01.wav]
  Output: {"action":"open_whiteboard","params":{}}
  Perf:   whisper=0.63s  prefill=0.67s(127tok)  decode=1.58s(26tok, 16.4 tok/s)
```

---

## 5. 音频要求

- 输入 16k 单声道 WAV（PCM16/32 或 IEEE float，`scipy.io.wavfile` 读）。
- mp3 需在主机转 wav：`conda run -n eureka-audio python` + `librosa.load(mp3, sr=16000, mono=True)` → `soundfile.write(..., subtype='PCM_16')`。
- 音频 token 数按实际时长截取：`real_frames = len(wav)//1280`，只取前 real_frames 个 audio embed（不能用 pad 到 30s 的全 375 个，否则静音淹没语义、输出乱码）。
- KV cache 上限 SEQLEN=512：prefix(68)+audio+suffix(10) 后剩余额度给 decode。音频太长（audio token 多）会挤占 decode 空间。

---

## 6. 已知问题与排查经验

- **C++ 用裸 bmrt 必须用 whisper/sail 范式**：每个 net 用 `bmrt_tensor` 独立分配 IO + 全程 host↔device 往返（`bm_memcpy_s2d/d2s`）+ KV cache 存 host。**禁止**直接操作 net 内置的 `stages[0].input_mems`、**禁止** net 间 device 端 d2d 直传——那样在真实数据下会踩 bmrt 内部状态导致**板卡硬重启**（零数据不崩、真实数据崩，极难定位）。范本见 `../whisper/cpp/src/whisper_inference.cpp`。
- **device memory**：`bm_get_stat` 实测 9070MB 可用，两个 bmodel（1.4+2.7=4.1G）共存够用。ion debug 接口报的 npu heap 3.86G 是单分区，不是 bmrt 上限。
- **精度排查方法**：逐级 cosine 对比 mel → encoder → audio_embeds，再分 ONNX vs bmodel，定位是导出 bug 还是量化损失。对照基准用原版 PyTorch（`eureka_infer.api.EurekaAudio`）。
