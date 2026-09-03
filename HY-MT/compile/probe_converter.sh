#!/bin/bash
set -euo pipefail

if [[ -z "${1:-}" ]]; then
    cat >&2 <<'USAGE'
用法: probe_converter.sh <容器内权重目录> [输出目录]

仅生成 block-0 的 bf16 MLIR 用于与 PyTorch 对照（--only_mlir -s 64），不产出 bmodel。
权重目录必须在容器内可见：3_docker/run_docker.sh 只把仓库根挂载到 /workspace，
因此传入 /workspace/HY-MT/compile/tmp/<权重目录>。
历史默认值 /models/HY-MT1.5-1.8B 不是 run_docker.sh 创建的挂载点，直接使用会失败。
USAGE
    exit 2
fi
MODEL_PATH="$1"
OUT_DIR="${2:-/workspace/HY-MT/compile/tmp/probe}"

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "[FAIL] 权重目录不存在: ${MODEL_PATH}" >&2
    echo "       容器内唯一挂载是 /workspace（= 仓库根），见上方用法说明。" >&2
    exit 1
fi

python3 /workspace/HY-MT/compile/patch_tpumlir_hymt.py
llm_convert.py \
  -m "${MODEL_PATH}" \
  -s 64 \
  --quantize bf16 \
  -c bm1684x \
  --only_mlir \
  --out_dir "${OUT_DIR}"

