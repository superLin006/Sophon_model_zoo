# HY-MT1.5-1.8B — BM1684X 移植

当前状态：原生 PyTorch baseline 与 TPU-MLIR 转换适配探索。

## 资源

- 原始权重：`/home/xh/itc_project/RK_model_zoo/models/HY-MT1.5-1.8B`
- 原生环境：`/home/xh/miniconda3/envs/hy-mt-sophon`（PyTorch 2.6 + Transformers 4.56.1）
- 转换环境：`sophon-tpumlir-v128`（TPU-MLIR 1.28.1）
- 目标芯片：BM1684X

## 原生 baseline

```bash
CUDA_VISIBLE_DEVICES=0 /home/xh/miniconda3/envs/hy-mt-sophon/bin/python \
  python/infer_native.py \
  --model_path /home/xh/itc_project/RK_model_zoo/models/HY-MT1.5-1.8B
```

baseline 使用 greedy decoding，并保存 embedding、代表层 hidden state、logits 和首层 KV cache，
用于后续逐层校验 bmodel。

## 转换探针

TPU-MLIR 1.28.1 需要先应用本项目的 Hunyuan 适配补丁。补丁修正动态 RoPE、
QK-Norm 权重名以及 Hunyuan 特有的 `RoPE -> QK-Norm` 顺序。

```bash
docker exec sophon-tpumlir-v128 bash \
  /workspace/HY-MT/compile/probe_converter.sh /models \
  /workspace/HY-MT/compile/tmp/probe
```

block-0 F32 MLIR 与 PyTorch 的 cosine 为 0.9999995。完整 W8BF16 转换入口：

```bash
docker exec sophon-tpumlir-v128 bash \
  /workspace/HY-MT/compile/compile_w8bf16.sh /models 512 \
  /workspace/HY-MT/models/BM1684X/w8bf16_seq512
```

W4F16 使用同一 patch 和 `-g 64`，但模型 config 的 `dtype` 需要设置为 `float16`；权重可使用硬链接副本，避免复制 4GB safetensors：

```bash
docker exec sophon-tpumlir-v128 bash -lc \
  'python3 /workspace/HY-MT/compile/patch_tpumlir_hymt.py && \
   llm_convert.py -m /workspace/HY-MT/compile/tmp/HY-MT1.5-1.8B-f16 \
   -s 512 --quantize w4f16 -g 64 -c bm1684x \
   --out_dir /workspace/HY-MT/compile/tmp/w4f16_seq512'
```

## 板端部署位置

板卡上的两个版本相互独立，可直接交给测试同事：

- 精度优先 W8BF16：`/data/hymt`
- 速度优先 W4BF16 group-size 64：`/data/hymt_w4g64`

两个目录均包含 `hymt_demo`、tokenizer 配置、bmodel 和回归脚本。快速运行：

```bash
cd /data/hymt_w4g64
./hymt_demo /data/hymt_w4g64 \
  'Translate the following segment into Chinese, without additional explanation.\n\nIt’s on the house.' \
  128
```

完整短/中/长 16 项回归和重复性能测试：

```bash
REPEATS=1 ./board_regression_extended.sh /data/hymt_w4g64
REPEATS=5 ./board_perf_stability.sh /data/hymt_w4g64
```

## 当前验证结果

- W8BF16 seq512 bmodel：1.98 GiB
- block-0 output cosine：0.99999951
- BM1684X 板上 5/5 用例与 PyTorch greedy baseline 逐字一致
- static prefill：约 205 ms
- decode：23.0–23.5 token/s

扩展的 16 项短/中/长测试结果：

| 版本 | 大小 | 质量/回归结果 | 平均字符相似度 | prefill 均值 | decode 均值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| W8BF16 | 1.98 GiB | 9/16 | 0.964 | 205.83 ms | 23.06 token/s |
| W4BF16 g64 | 1.26 GiB | 3/16 | 0.815 | 203.45 ms | 34.41 token/s |
| **W4F16 g64** | **1.26 GiB** | **W4 档，61/61 进程成功** | — | **205.98 ms** | **33.21 token/s（中位数）** |

W4F16 g64 与 W4BF16 g64 的模型体积相同；61 条回归中两者输出有 48/61 条完全一致，说明 W4F16 没有恢复 W8BF16 的输出轨迹。W4F16 的 decode 中位数为 33.21 token/s，略低于 W4BF16 的 34.35 token/s，但比 W8BF16 的 23.31 token/s 快约 42.5%。格式标签、术语和长文本仍存在 W4 档共同的质量偏移，因此 W4F16 仅作为速度档，默认交付继续采用 W8BF16。KV cache 当前为 BF16，改成 FP16 仍是 16 bit，不会降低 cache 带宽或容量。

W4 的短/中/长代表用例各重复 3 次，输出均确定一致。decode 均值分别为
34.88、34.03、33.64 token/s，样本标准差分别为 0.04、0.74、0.55 token/s。
长上下文相对短句下降约 3.6%，表明 KV cache 访问有影响，但当前主要收益仍来自减小权重带宽。

### 61 用例全量回归（2026-08-14）

`test_data/board_regression_full.sh` 在原有 16 项基础上扩展 45 项（生活短句、日中/中日、
术语专名、格式标签、数字单位、科技、财经新闻、长文本、标点、API 风格），W8BF16 板上
61/61 全部通过，prefill ~205 ms 恒定、decode 22.2–24.0 token/s。分析见
`outputs/board_extended/w8_full_61cases_analysis.md`（原始日志 `w8_full_61cases.log`）。
已知 3 处瑕疵：术语用例输出 "edge computing" 未用参考词 "edge inference"、
格式用例 XML 标签嵌套错位、names_products 措辞非最优。W4BF16 已部署于板卡
`/data/hymt_w4g64`，可用 `BIN=/data/hymt/hymt_demo_async bash board_regression_full.sh /data/hymt` 复测任意版本。

纯 C++ 推理程序：`cpp/build-aarch64-v2/hymt_demo`。

```bash
./hymt_demo /data/hymt \
  'Translate the following segment into Chinese, without additional explanation.\n\nIt’s on the house.' \
  128
```

板上回归脚本：`test_data/board_regression.sh`。
