# QwenLLM — BM1684X 意图识别

Qwen3-0.6B dispatch 模型用于语音助手意图识别（ASR 文本 → action + params JSON）。**当前正式交付档为 `v95e-soup` / W8BF16 / seq2048**；其他 v95、v95c、v95d 变体保留用于质量对比，不作为默认部署目标。

性能数据登记在 [`PERF_SUMMARY.md` 第三节](../PERF_SUMMARY.md)。

## 目录结构

```text
QwenLLM/
├── compile/
│   └── compile_v95e.sh       # v95e-soup → bmodel，中间物落 compile/tmp/
├── cpp/                      # 纯 C++ bmruntime 推理（qwen_demo）
├── demo/                     # Python/sail benchmark 用的 pybind demo 与 config/
├── models/BM1684X/           # 最终 bmodel（本地生成，不入库）
├── Qwen3-0.6B-dispatch-v95e-soup/  # v95e 原始权重（本地生成，不入库）
├── scripts/                  # 部署入口 + 历史档位的下载/编译脚本
├── benchmark_intent.py       # 板卡意图 benchmark
└── requirements.txt          # 本地不需要 Python 转换环境
```

`LLM-TPU/` 是本地使用的第三方源码（sophgo 官方 demo），已被 gitignore，不属于本模型交付内容。

**`scripts/` 分两类，不要混用**：

| 脚本 | 状态 | 说明 |
|---|---|---|
| `deploy_to_board.sh` | **当前入口** | 部署 v95e bmodel + config + benchmark |
| `compile_qwen3_small.sh` / `compile_qwen3_4b.sh` | 历史档位，**非交付范围** | 编译原始 Qwen3-0.6B / 1.7B / 4B-AWQ（w4bf16 / W4F16-AWQ，seq2048） |
| `download_qwen3_small.sh` / `download_qwen3_4b.sh` | 历史档位配套 | 从 ModelScope 下载上述原始权重 |

> 历史档位的权重目录（`Qwen3-0.6B/`、`Qwen3-1.7B/`、`Qwen3-4B-AWQ/`）**当前不在本机**，因此那两组 compile 脚本直接运行会失败；需先跑配套的 `download_*.sh` 取回权重。它们保留是因为其测量结果构成量化机理的关键对照证据（0.6B w4bf16 失败 vs 1.7B 成功），见知识库 §2.2 与 §2.4。当前交付只有 w8bf16 产物，**仓库内无任何 w4 档位 bmodel**。

## 当前正式流程

### 1. 准备 v95e 权重

将完整的 HuggingFace/Safetensors 权重放在：

```text
QwenLLM/Qwen3-0.6B-dispatch-v95e-soup/
```

目录至少需要 `config.json` 和 `model.safetensors`。权重、tokenizer 和微调 provenance 不提交 Git。

### 2. 编译 bmodel

```bash
bash QwenLLM/compile/compile_v95e.sh
```

默认参数：量化 `w8bf16`、序列长度 `2048`、芯片 `bm1684x`，最终产物
`QwenLLM/models/BM1684X/qwen3_0.6b_dispatch_v95e_soup_w8bf16_seq2048.bmodel`（772MB）。
中间产物落 `compile/tmp/`。可用 `QWEN_MODEL_DIR` / `SEQ_LEN` / `QUANTIZE` / `OUT_DIR` / `TPUMLIR_CONTAINER` 覆盖。

脚本要求公共 TPU-MLIR 容器已启动，并且**权重目录必须位于仓库内**（脚本会把宿主路径映射成容器内的 `/workspace/QwenLLM/...`；`run_docker.sh` 只创建 `/workspace` 这一个挂载）：

```bash
./3_docker/build_tpumlir.sh
./3_docker/run_docker.sh       # 容器名 sophon-tpumlir-v128
```

### 3. 构建 C++ 推理程序

> ⚠️ **QwenLLM 的 `cpp/build.sh` 不会自己起 docker**（与仓库里另外 9 个模型不同，见 `3_docker/README.md` 的「两种 build.sh 风格」）。它只检查 `aarch64-linux-gnu-g++` 是否存在，**必须在 cross-build 容器内执行**。
>
> 在宿主机直接 `bash QwenLLM/cpp/build.sh` 有两种结局：宿主没装交叉编译器时报错退出（安全）；宿主装了（如 WSL 的 gcc 15）则**编译成功但产物不可上板**，运行报 `GLIBC_2.xx not found`。

正确做法（将仓库显式挂载到容器内）：

```bash
docker run --rm \
  -v "$PWD":/workspace/repo \
  sophon-cross-build bash -lc \
  'cd /workspace/repo/QwenLLM/cpp && rm -rf build && ./build.sh'
# 产物：QwenLLM/cpp/build/qwen_demo
# 需要连同 libbmrt/libbmlib 一起打包时：在上面的命令中追加 build.sh install
```

上板前用 `file QwenLLM/cpp/build/qwen_demo` 确认输出含 `ARM aarch64`。

### 4. 部署与 smoke test

推荐使用 SSH key。需要密码时由环境变量提供，不要写入脚本：

```bash
BOARD_IP=<board_ip> \
  bash QwenLLM/scripts/deploy_to_board.sh --test
```

可覆盖默认模型或部署目录：

```bash
BOARD_IP=<board_ip> \
BMODEL=QwenLLM/models/BM1684X/another.bmodel \
BOARD_DIR=/data/qwenllm \
  bash QwenLLM/scripts/deploy_to_board.sh
```

部署脚本会上传 bmodel、tokenizer 配置、benchmark 和 Python demo，并对上传文件执行 md5 校验；`--test` 才会在板卡编译 Python demo 并运行 10 条意图用例。

板上目录为平铺布局：`/data/qwenllm/{qwen_demo, benchmark_intent.py, config/, models/, python_demo/}`。

## Benchmark

### Python 路径（`benchmark_intent.py`，10 条意图用例）

```bash
cd /data/qwenllm
python3 benchmark_intent.py \
  -m models/qwen3_0.6b_dispatch_v95e_soup_w8bf16_seq2048.bmodel \
  -c config \
  -n qwen3-0.6b-dispatch-v95e-soup \
  --no_think
```

指标包括模型加载耗时、FTL、Prefill 吞吐、Decode TPS 和端到端延迟。`--quick` 只跑前 3 条。结果 JSON 写入当前工作目录，**不提交仓库**。

> ⚠️ **`--test` 依赖板卡 pybind11**：板卡无 pybind11 时 `python_demo` 编译失败。此时 bmodel/config 仍已上传成功（用不带 `--test` 的部署即可），改用下面的 C++ `qwen_demo` 验证。

### C++ 路径（`qwen_demo`）

```bash
cd /data/qwenllm
./qwen_demo . models/qwen3_0.6b_dispatch_v95e_soup_w8bf16_seq2048.bmodel
# 参数: <model_dir> [bmodel_filename]
#   model_dir 需同时含 config/tokenizer.json；bmodel 在子目录时用第二个参数给相对路径
```

`qwen_demo` 跑内置的两条演示 prompt（短回复 / 中等回复），不从 stdin 读输入。**2026-09-03 板卡实测**：Prefill 约 485ms，Decode 约 45 token/s。C++ 输出路径现已按 token ID 累积后调用 `tok.Decode()`，适配 Qwen 的 byte-level BPE；不要用 `IdToToken()` 逐 token 拼接文本。修改后需重新交叉编译并上板复验文本输出。Python benchmark 使用 `transformers` 的 `AutoTokenizer.decode()`，不受该 C++ 显示路径影响。

## 模型版本管理

正式版本和实验版本必须通过目录名与 provenance 文件区分。当前默认版本为 `Qwen3-0.6B-dispatch-v95e-soup`。

`v95-normal-2e`、`v95-r4`、`v95-recall`、`v95c-locked7k`、`v95c-soup25`、`v95d-soup` 等版本可继续用于质量对比，但部署脚本不会自动扫描它们。本机 `models/` 下现有 4 个已编译变体（`v95_normal_2e`、`v95c_soup25`、`v95d_soup`、`v95e_soup`），**全部为 w8bf16**。选定版本的 bmodel 与测试结果需登记到 `PERF_SUMMARY.md`（v95e 已登记）。

## 验收要求

- bmodel 只保留最终部署文件，编译中间物放在 `compile/tmp/`。
- 模型目录、tokenizer 和 benchmark 配置必须与 bmodel 一起部署。
- 上传后必须逐文件 md5 校验。
- 生产结果必须关闭 thinking（`--no_think`），并记录测试用例、量化方式、seq_len、板卡型号和测试日期。
- benchmark 结果 JSON 留在板卡工作目录，不提交仓库。
