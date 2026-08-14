#!/bin/bash
set -euo pipefail

MODEL_PATH="${1:-/models}"
SEQ_LEN="${2:-512}"
OUT_DIR="${3:-/workspace/HY-MT/models/BM1684X/w8bf16_seq${SEQ_LEN}}"

python3 /workspace/HY-MT/compile/patch_tpumlir_hymt.py
llm_convert.py \
  -m "${MODEL_PATH}" \
  -s "${SEQ_LEN}" \
  --quantize w8bf16 \
  -c bm1684x \
  --out_dir "${OUT_DIR}"

