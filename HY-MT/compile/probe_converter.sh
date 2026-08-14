#!/bin/bash
set -euo pipefail

MODEL_PATH="${1:-/models/HY-MT1.5-1.8B}"
OUT_DIR="${2:-/workspace/HY-MT/compile/tmp/probe}"

python3 /workspace/HY-MT/compile/patch_tpumlir_hymt.py
llm_convert.py \
  -m "${MODEL_PATH}" \
  -s 64 \
  --quantize bf16 \
  -c bm1684x \
  --only_mlir \
  --out_dir "${OUT_DIR}"

