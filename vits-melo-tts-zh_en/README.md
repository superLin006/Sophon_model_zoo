# VITS-MeloTTS 中英双语 — BM1684X

VITS-MeloTTS 中英双语语音合成在 Sophon BM1684X 上的三段式推理实现：Part A 文本编码与时长预测、Part B（MAS 对齐，留在 CPU）、Part C1 Flow 与 Part C2 Decoder。

**交付精度：F16**（仓库内只有 F16 三件 bmodel，无 F32 产物）。性能与限制见下文，权威数值登记在 [`PERF_SUMMARY.md` 第二节](../PERF_SUMMARY.md)。

## 目录

```text
vits-melo-tts-zh_en/
├── python/                  # ONNX 准备、拆分、bmodel 编译和验证（详见 python/README.md）
├── cpp/                     # BMRuntime C++ 推理，产物 cpp/build/
├── models/BM1684X/          # 本地生成的 Part A/C1/C2 bmodel（F16）
├── models/onnx/             # 原始/转换后的 ONNX、词典与 FST 资源（生成中间文件不提交）
├── test_data/               # token/tone 输入样本（入库）
├── deploy_to_board.sh       # 上传并部署到板卡
└── .context/                # 移植过程记录（baseline / operator_analysis / bmodel_info）
```

> 本模型的 `compile/tmp/` 只用于转换中间产物，已加入 `.gitignore`，不属于交付目录；最终 bmodel 才会复制到 `models/BM1684X/`。

## 模型来源与导出

原始单体 ONNX 来自上游 **sherpa-onnx 对 [myshell-ai/MeloTTS](https://github.com/myshell-ai/MeloTTS) 的导出**（单女声 zh_en 模型）。仓库现在提供真实的 `python/export_onnx.py`，可使用 ModelScope 的 `myshell-ai/MeloTTS-Chinese` checkpoint 重新导出；导出后的单体模型和中间文件只用于转换，不作为最终交付文件。上游自带的 `README.md` 与 `LICENSE`、词典和 FST 资源保留在 `models/onnx/vits-melo-tts-zh_en/`。

> `export_onnx.py` 通过 `MELOTTS_CHECKPOINT` 和 `MELOTTS_CONFIG` 指定权重与配置，并依赖 MeloTTS 源码目录；完整命令见 [python/README.md](python/README.md)。

## 快速开始

详细的 ONNX 准备、bmodel 编译和输入格式说明见 [python/README.md](python/README.md)。

编译 bmodel（公共 TPU-MLIR 容器）：

```bash
docker exec sophon-tpumlir-v128 bash /workspace/vits-melo-tts-zh_en/python/gen_bmodel.sh F16
```

C++ 交叉编译：

```bash
bash vits-melo-tts-zh_en/cpp/build.sh
# 产物：vits-melo-tts-zh_en/cpp/build/vits_melo_tts_bm1684
```

部署前设置板卡信息，避免把认证信息写入脚本。追加 `--test` 会上传标准 token/tone 样本并运行中英文 smoke test：

```bash
BOARD_IP=<board_ip> \
  bash vits-melo-tts-zh_en/deploy_to_board.sh F16 --test-stream
# 句内 C2 窗口流式：
BOARD_IP=<board_ip> \
  bash vits-melo-tts-zh_en/deploy_to_board.sh F16 --test-window
```

当前 C++ 运行时需要以下模型文件：

```text
vits_part_a_F16.bmodel     # 20M，L=256
vits_part_c1_F16.bmodel    # 53M，T_mel=1024
vits_part_c2_F16.bmodel    # 34M，T_mel=1024
vits_part_c2_stream_W128_R16_F16.bmodel  # 33M，输入 mel=160，中心输出 128 帧
```

板端运行（legacy 参数：`<tokens.bin> <tones.bin> <n_tokens> <model_dir> <out.wav> [F16|F32]`；当前 `n_tokens` 最大 256）：

```bash
./vits_melo_tts_bm1684 test_data/test_zh_tokens.bin    test_data/test_zh_tones.bin    61 models output_zh_F16.wav    F16
./vits_melo_tts_bm1684 test_data/test_en_zh_tokens.bin test_data/test_en_zh_tones.bin 53 models output_en_zh_F16.wav F16
```

句段级流式模式按 manifest 顺序逐段推理；每段 C2 完成后立即触发 PCM callback，可写追加式 WAV 或纯 S16LE PCM。它不是 token/mel/decoder 内部增量：

```text
# tokens.bin tones.bin seq_len label
test_zh_tokens.bin test_zh_tones.bin 61 zh
test_en_zh_tokens.bin test_en_zh_tones.bin 53 en
```

```bash
./vits_melo_tts_bm1684 --stream-manifest test_data/stream_manifest.txt \
  --model-dir models --wav-out stream.wav --precision F16

./vits_melo_tts_bm1684 --stream-manifest test_data/stream_manifest.txt \
  --model-dir models --wav-out window_stream.wav --precision F16 --window-stream
# 或：--pcm-s16le-out window_stream.pcm
```

`--window-stream` 是句内 C2 音频流式：Part A、CPU MAS、C1 仍按句完成，C2 使用
128 个有效 mel 帧、左右 16 帧上下文和 32 帧 overlap-add 窗口；每个窗口完成后
立即触发 callback。新增 window bmodel 只作为显式模式使用，旧三件套和句段级
`--stream-manifest` 行为不变。它仍不是 token 级或 decoder state 级增量。

## 性能与验收状态

单链路实测（详见 [python/README.md](python/README.md)）：

| 阶段 | 执行位置 | 耗时 |
|---|---|---:|
| Part A（文本编码 + 时长预测） | TPU | 3.9ms（L=256） |
| MAS 对齐 | **CPU** | 8.2ms（61 token）/ 127.5ms（256 token） |
| Part C（Flow + Decoder） | TPU | 169.0ms（T_mel=1024） |
| **合计** | — | **182.4ms（61 token）→ RTF 0.068** |

> **板卡验收**：L=256、T_mel=1024 的中文 256 token 输入生成 10.83s 音频，耗时 300.8ms、RTF 0.0278；中英混合 256 token 输入生成 11.60s 音频，耗时 309.9ms、RTF 0.0267。两次均正常完成，板卡 uptime 未变化。

## 已知限制

- **L 固定 256，T_mel 固定 1024**，最多生成约 11.9s 音频；当预测时长超过 1024 帧时，C++ 会在 CPU 侧先截断再进入 TPU，并打印警告；更长文本应在前端分句后分别合成
- **SDP 分支被替换为零常量**（只用 DP 做确定性时长预测）：SDP 含 21 个 `NonZero`，输出 shape 依赖运行时数值，TPU-MLIR 无法静态推断
- **Flow 的 `RandomNormalLike` 分支被绕过**：TPU-MLIR v1.28.1 未实现该算子；`noise_scale=0` 时其贡献为零，因此绕过不影响输出
- **MAS 留在 CPU**：含 `Range` 算子，输出 shape 依赖运行时决定的 T_mel，无法静态编译；CPU 侧仅约 8ms
- **窗口流式只窗口化 C2**：C1 Flow 含全局注意力，仍按完整 mel 序列运行；window C2 的输出经 32 帧 overlap-add，和 legacy 整段波形不保证字节一致，但需通过边界误差和听感验收
- BM1684X SDK 无 `BM_INT64`，Part A 的 token/tone 输入需在 C++ 侧 cast 为 int32 再上传

测试输出和模型产物均为本地生成文件，保存规则见 `.claude/standards/bmodel_output_management.md`。
