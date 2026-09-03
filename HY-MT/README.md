# HY-MT1.5-1.8B — BM1684X 移植

**状态：已完成板卡验证并定档。** 交付档 **W8BF16 / seq512**，61 用例全量回归通过；W4BF16 g64 为速度对照档；W4F16 g64 为实验档，**bmodel 产物与源权重均未保留**（板上验证日志已入库）。

性能与验收数据的权威来源是 [`PERF_SUMMARY.md` 第四节](../PERF_SUMMARY.md)，其中每个数字都标注了用例集、统计口径与原始日志出处。本文件只给流程、命令与已知限制。

## 资源

- **原始权重**：HY-MT1.5-1.8B（约 4GB safetensors），通过编译脚本的第一个参数指定容器内路径。**当前本机不再保留**，重新编译前需先取回
- **原生 baseline 环境**：conda 环境 **`sophon-hy-mt`**（Python 3.10 / PyTorch 2.6 / Transformers 4.56.1，`requirements.txt` 锁定；transformers 4.56.1 是 HY-MT1.5-1.8B 官方权重要求的版本）
- **转换环境**：容器 `sophon-tpumlir-v128`（镜像 `sophon/tpuc_dev:v3.4-tpumlir-1.28.1`，TPU-MLIR 1.28.1）
- **目标芯片**：BM1684X

> 环境名以 `sophon-hy-mt` 为准（与仓库其余模型的 `sophon-*` 命名一致，也是本机实测可用的环境）。文档与 `requirements.txt` 历史上写作 `hy-mt-sophon`，已统一。

## 原生 baseline

需要 **CUDA GPU**（`requirements.txt` 装的是 `torch==2.6.0` 的 cu124 构建，命令也显式指定 `CUDA_VISIBLE_DEVICES`）；无 GPU 的机器上跑不了这一步，但**不影响 bmodel 编译与板卡验证**——baseline 只用于产出逐层对照参考。

```bash
export MODEL_PATH=<model_path>
CUDA_VISIBLE_DEVICES=0 conda run -n sophon-hy-mt python \
  python/infer_native.py --model_path "$MODEL_PATH"
```

baseline 使用 greedy decoding，并保存 embedding、代表层 hidden state、logits 和首层 KV cache，用于后续逐层校验 bmodel。

## 转换

TPU-MLIR 1.28.1 需要先应用本项目的 Hunyuan 适配补丁（`compile/patch_tpumlir_hymt.py`）。补丁修正动态 RoPE、QK-Norm 权重名以及 Hunyuan 特有的 `RoPE -> QK-Norm` 顺序。**三个编译脚本都已内置调用该补丁**，不需要单独执行。

> ⚠️ **权重路径必须用 `/workspace/...`**。`3_docker/run_docker.sh` 只创建 `/workspace`（= 仓库根）这一个挂载，容器内没有 `/models`。脚本原先把 `MODEL_PATH` 默认值写成 `/models`，缺参数时会静默走到 `llm_convert.py` 才失败；现已改为必填参数，缺失或目录不存在时立即报错并打印用法。

### 探针（block-0 对照）

```bash
docker exec sophon-tpumlir-v128 bash \
  /workspace/HY-MT/compile/probe_converter.sh \
  /workspace/HY-MT/compile/tmp/HY-MT1.5-1.8B-f16 \
  /workspace/HY-MT/compile/tmp/probe
```

block-0 F32 MLIR 与 PyTorch 的 cosine 为 0.9999995。

### W8BF16（交付档）

```bash
docker exec sophon-tpumlir-v128 bash \
  /workspace/HY-MT/compile/compile_w8bf16.sh \
  /workspace/HY-MT/compile/tmp/HY-MT1.5-1.8B-f16 512 \
  /workspace/HY-MT/models/BM1684X/w8bf16_seq512
```

### W4BF16 group64（速度档）

```bash
docker exec sophon-tpumlir-v128 bash \
  /workspace/HY-MT/compile/compile_w4bf16_g64.sh \
  /workspace/HY-MT/compile/tmp/HY-MT1.5-1.8B-f16 512 \
  /workspace/HY-MT/models/BM1684X/w4bf16_g64_seq512
```

### W4F16 group64（实验档，无固化脚本）

W4F16 使用同一补丁和 `-g 64`，但模型 config 的 `dtype` 需改为 `float16`；权重可用硬链接副本，避免复制 4GB safetensors。**这一档没有对应的 `compile_*.sh`**，只有手册命令：

```bash
docker exec sophon-tpumlir-v128 bash -lc \
  'python3 /workspace/HY-MT/compile/patch_tpumlir_hymt.py && \
   llm_convert.py -m /workspace/HY-MT/compile/tmp/HY-MT1.5-1.8B-f16 \
   -s 512 --quantize w4f16 -g 64 -c bm1684x \
   --out_dir /workspace/HY-MT/compile/tmp/w4f16_seq512'
```

> ⚠️ 该档产物输出到 `compile/tmp/`（可再生成目录，已清空），**从未提升到 `models/BM1684X/`**，因此当前无可部署 bmodel。板上验证日志保留在 `outputs/board_w4f16_20260817/hymt_w4f16_46.log`，测量结论可查证；要重新产出需先取回源权重。

## 板端部署

部署入口按版本目录提取根部合并 bmodel 和 `config/`，不会上传 block 编译中间目录。默认部署交付档 W8BF16；速度档用 `MODEL_VARIANT` 覆盖：

```bash
BOARD_IP=<board_ip> bash HY-MT/deploy_to_board.sh --test
MODEL_VARIANT=w4bf16_g64_seq512 BOARD_DIR=/data/hymt_w4g64 \
  BOARD_IP=<board_ip> bash HY-MT/deploy_to_board.sh --test
```

支持 `BOARD_USER`、`BOARD_PORT`、`BOARD_DIR`、`MODEL_DIR`、`BINARY`、`TEST_PROMPT` 和 `BOARD_PASS` 环境变量。上传后脚本会逐项执行 md5 校验。

脚本上传三样东西：`hymt_demo`、合并后的 bmodel、`config/`（tokenizer 等）。**板上目录是平铺布局**（不带 `models/` 子目录）：

```text
/data/hymt/
├── hymt_demo
├── models_w8bf16_seq512_bm1684x_1dev_static_<时间戳>.bmodel
└── config/
```

> ⚠️ **回归脚本不由部署脚本上传**。`test_data/board_*.sh` 需要另行 scp：
> ```bash
> scp HY-MT/test_data/board_*.sh HY-MT/test_data/analyze_board_logs.py \
>   root@<board_ip>:/data/hymt_w4g64/
> ```

## 板端运行

```bash
cd /data/hymt
./hymt_demo . \
  'Translate the following segment into Chinese, without additional explanation.\n\nIt’s on the house.' \
  128
# 参数: <model_dir> <prompt> <max_new_tokens>
```

`<model_dir>` 传 `.`（平铺布局）或板上绝对路径均可。二进制在仓库内的位置是 `cpp/build-aarch64-v2/hymt_demo`——**注意不是 `cpp/build/`**，HY-MT 是本仓库两个产物目录例外之一（另一个是 zipformer 的 `cpp/build-aarch64/`）。

### 回归脚本

`test_data/` 下有四个脚本，用途与规模各不相同：

| 脚本 | 用例数 | 用途 |
|---|---|---|
| `board_regression.sh` | 5 | 快速冒烟 |
| `board_regression_extended.sh` | 16 | 短/中/长三档质量对照（字符相似度判定） |
| `board_regression_full.sh` | **61** | 全量回归：16 项 + 45 项（生活短句、日中/中日、术语专名、格式标签、数字单位、科技、财经新闻、长文本、标点、API 风格） |
| `board_perf_stability.sh` | 3 × REPEATS | 短/中/长重复运行的性能稳定性 |

```bash
REPEATS=1 ./board_regression_extended.sh /data/hymt_w4g64
REPEATS=1 ./board_regression_full.sh     /data/hymt_w4g64
REPEATS=5 ./board_perf_stability.sh      /data/hymt_w4g64
```

`board_regression_full.sh` 可用 `BIN=` 复测任意版本的二进制：

```bash
BIN=/data/hymt/hymt_demo bash HY-MT/test_data/board_regression_full.sh /data/hymt
```

单用例失败不中断整套（记录 `[FAIL]` 后继续），收尾行打印实际用例数与失败数。日志用 `test_data/analyze_board_logs.py` 后处理。

## 验证结果

数值、口径与原始日志的完整对照见 [`PERF_SUMMARY.md` 第四节](../PERF_SUMMARY.md)。要点：

- **W8BF16 seq512**：合并 bmodel 1.98 GiB；block-0 output cosine 0.99999951；61 用例全部执行完成，prefill 中位 ~206ms（区间 204.6–210.4ms），decode 中位 23.37 tok/s（区间 22.20–23.95）；5/5 代表用例与 PyTorch greedy baseline 逐字一致
- **W4BF16 g64**：1.26 GiB；16 项质量对照 3/16、平均字符相似度 0.815（W8 同集为 9/16、0.964）；61 项 decode 中位 34.35 tok/s
- **W4F16 g64**：1.26 GiB；61 项 decode 中位 33.21 tok/s，与 W4BF16 输出 48/61 完全一致——说明 W4F16 没有恢复 W8 的输出轨迹，因此仅作速度档

**「61/61」的准确含义是 61 个用例全部执行完成，不是 61 条通过质量判定。** 回归脚本只在进程失败时打印 `[FAIL]`，日志不含逐条质量判定；质量口径是 16 项的字符相似度（9/16 对 3/16）。

### 已知问题

- **W4 两档共同的质量偏移**：格式标签、术语和长文本。W8 侧已知 3 处瑕疵——术语用例输出 "edge computing" 未用参考词 "edge inference"；格式用例 XML 标签嵌套错位；names_products 措辞非最优。分析见 `outputs/board_extended/w8_full_61cases_analysis.md`（原始日志 `w8_full_61cases.log`，已入库）。
- **W4F16 有一条严重离群**：`terminology_short` 单条 decode 仅 **1.35 tok/s**，是该档中位数 33.21 的约 1/25，把 decode 均值拉到 31.18、prefill 均值拉到 242.95ms（中位数仍为 205.98ms）。W8 最慢 19.09、W4BF16 最慢 33.73，均无同类异常。评估 W4F16 作为速度档时必须计入这条。
- **同一档位有多个 decode 数值，来自不同用例集与统计量**，引用时必须带口径：W8 的 22.2–24.0 是 61 项 min–max 区间、23.06 是 16 项均值、23.31 与 23.37 是两轮 61 项中位数；W4BF16 的 34.41 是 16 项均值、34.35 是 61 项中位数。这些不是笔误，也不可互相替换。
- **已入库的历史日志里，收尾行写的是 `ALL DONE (1x46 cases)`**，而实际跑了 61 个用例——那是脚本硬编码的汇总字符串少报了 15 条，现已改为动态计数。读取旧日志时以 `===== <name> repeat=` 用例块计数为准。
- KV cache 当前为 BF16（llm_convert 标准行为）。改成 FP16 仍是 16 bit，不会降低 cache 带宽或容量。

## 目录

```text
HY-MT/
├── compile/                  # patch_tpumlir_hymt.py + 三个编译脚本 + block-0 探针
│   └── tmp/                  # 中间产物与权重副本（可再生成，不入库）
├── cpp/                      # C++ 推理；产物在 build-aarch64-v2/
├── models/BM1684X/
│   ├── w8bf16_seq512/        # 交付档：合并 bmodel + config/
│   └── w4bf16_g64_seq512/    # 速度档：合并 bmodel + config/
├── outputs/
│   ├── baseline*/            # PyTorch 原生 baseline 参考（npz，不入库）
│   ├── board_extended/       # 61 用例、16 项扩展、稳定性日志 + 分析（log 已入库）
│   └── board_w4f16_20260817/ # W8/W4BF16/W4F16 三档同轮对照日志（已入库）
├── python/infer_native.py    # 原生 baseline（需 GPU）
├── test_data/                # 四个回归脚本 + 日志分析脚本
├── deploy_to_board.sh
└── .context/                 # baseline / operator_analysis / bmodel_info
```
