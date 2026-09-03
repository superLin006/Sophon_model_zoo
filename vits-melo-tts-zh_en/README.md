# VITS-MeloTTS 中英双语 — BM1684X

VITS-MeloTTS 中英双语语音合成在 Sophon BM1684X 上的三段式推理实现：Part A 文本编码与时长预测、Part B（MAS 对齐，留在 CPU）、Part C1 Flow 与 Part C2 Decoder。

**交付精度：F16**（仓库内只有 F16 三件 bmodel，无 F32 产物）。性能与限制见下文，权威数值登记在 [`PERF_SUMMARY.md` 第二节](../PERF_SUMMARY.md)。

## 目录

```text
vits-melo-tts-zh_en/
├── python/                  # ONNX 准备、拆分、bmodel 编译和验证（详见 python/README.md）
├── cpp/                     # BMRuntime C++ 推理，产物 cpp/build/
├── models/BM1684X/          # 本地生成的 Part A/C1/C2 bmodel（F16）
├── models/onnx/             # 上游导出的三段 ONNX + 随附的上游 README/LICENSE/词典
├── test_data/               # token/tone 输入样本（入库）
├── deploy_to_board.sh       # 上传并部署到板卡
└── .context/                # 移植过程记录（baseline / operator_analysis / bmodel_info）
```

> 本模型**没有 `compile/` 目录**。转换中间产物由 `python/gen_bmodel.sh` 在其自身工作目录内处理，不落 `compile/tmp/`；早期文档里的 `compile/tmp/` 条目是不存在的目录，已删除。

## 模型来源（本仓不做 PyTorch → ONNX 导出）

三段 ONNX 来自上游 **sherpa-onnx 对 [myshell-ai/MeloTTS](https://github.com/myshell-ai/MeloTTS) 的导出**（单女声 zh_en 模型）。完整说明见 [python/README.md](python/README.md) 的「原始 model.onnx 来源」一节；上游自带的 `README.md` 与 `LICENSE` 已随 ONNX 一并保留在 `models/onnx/vits-melo-tts-zh_en/`，词典与 FST 资源在同目录 `dict/`、`*.fst`、`lexicon.txt`、`tokens.txt`。

> ⚠️ **这是 `.claude/standards/models_directory_standard.md` §2 的一条已知例外**：该标准要求非 LLM 部件必须有 python onnx 导出脚本、禁止 placeholder，而本模型的 `python/export_onnx.py` 是**来源说明占位**，仓库内的 Python 工具只负责把上游 ONNX 改造成 TPU 兼容图（静态 shape 拆分、算子替换）并校验，不重复下载或重新导出原始权重。

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
  bash vits-melo-tts-zh_en/deploy_to_board.sh F16 --test
```

当前 C++ 运行时需要以下模型文件：

```text
vits_part_a_F16.bmodel     # 19M
vits_part_c1_F16.bmodel    # 48M
vits_part_c2_F16.bmodel    # 33M
```

板端运行（参数：`<tokens.bin> <tones.bin> <n_tokens> <model_dir> <out.wav> [F16|F32]`）：

```bash
./vits_melo_tts_bm1684 test_data/test_zh_tokens.bin    test_data/test_zh_tones.bin    61 models output_zh_F16.wav    F16
./vits_melo_tts_bm1684 test_data/test_en_zh_tokens.bin test_data/test_en_zh_tones.bin 53 models output_en_zh_F16.wav F16
```

## 性能与验收状态

单链路实测（详见 [python/README.md](python/README.md)）：

| 阶段 | 执行位置 | 耗时 |
|---|---|---:|
| Part A（文本编码 + 时长预测） | TPU | 6ms |
| MAS 对齐 | **CPU** | 8ms |
| Part C（Flow + Decoder） | TPU | 305ms |
| **合计** | — | **319ms → RTF ~0.12** |

> ⚠️ **验收状态：只跑过 2 条 smoke test**（中文 61 token「今天天气真好，我们去公园散步吧。」、中英混 53 token「你好world，这是一个test句子。」），**没有做批量回归**，因此没有通过率与统计意义上的 RTF。上面的 0.12 是单链路实测值，不是多样本统计量。`PERF_SUMMARY.md` 因此只把它登记为「跑通（无批量回归）」，不给通过率。

## 已知限制

- **T_mel 固定 512**，最多生成约 6s 音频；推理时 `z_p` 会 pad 到 512，输出按实际有效帧截取
- **SDP 分支被替换为零常量**（只用 DP 做确定性时长预测）：SDP 含 21 个 `NonZero`，输出 shape 依赖运行时数值，TPU-MLIR 无法静态推断
- **Flow 的 `RandomNormalLike` 分支被绕过**：TPU-MLIR v1.28.1 未实现该算子；`noise_scale=0` 时其贡献为零，因此绕过不影响输出
- **MAS 留在 CPU**：含 `Range` 算子，输出 shape 依赖运行时决定的 T_mel，无法静态编译；CPU 侧仅约 8ms
- **`sid` 必须传 1**：传 0 会输出静音
- BM1684X SDK 无 `BM_INT64`，Part A 的 token/tone 输入需在 C++ 侧 cast 为 int32 再上传

测试输出和模型产物均为本地生成文件，保存规则见 `.claude/standards/bmodel_output_management.md`。
