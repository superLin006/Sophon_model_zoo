#!/bin/bash
# 合并 encoder + LLM 为单一 bmodel（model_tool --combine）。
# 对应 README 构建流程的最终一步，固化命令避免手工敲错：
#   model_tool --combine tmp/encoder_full.bmodel qwen3_asr_std_<档位>_512/*.bmodel \
#       -o models/BM1684X/qwen3_asr_merged_<档位>.bmodel
#
# 用法（docker sophon/tpuc_dev:v3.4-tpumlir-1.28.1 内，/workspace = 仓库根）:
#   bash /workspace/Qwen3-ASR/compile/merge.sh \
#       <encoder.bmodel> <LLM bmodel 目录> <输出名如 qwen3_asr_merged_w4f16.bmodel>
set -euo pipefail
ENC="${1:?用法: merge.sh <encoder.bmodel> <llm_bmodel_dir> <输出名> [输出路径]}"
LLM_DIR="${2:?}"
NAME="${3:?}"

[ -f "$ENC" ] || { echo "[FAIL] encoder bmodel 不存在: $ENC"; exit 1; }
if [ ! -d "$LLM_DIR" ] || ! ls "$LLM_DIR"/*.bmodel >/dev/null 2>&1; then
  echo "[FAIL] LLM bmodel 目录无产物: $LLM_DIR"
  exit 1
fi

OUT="${4:-/workspace/Qwen3-ASR/models/BM1684X/$NAME}"
mkdir -p "$(dirname "$OUT")"

model_tool --combine "$ENC" "$LLM_DIR"/*.bmodel -o "$OUT" >/dev/null 2>&1 || \
  { echo "[FAIL] model_tool --combine 失败"; exit 1; }
[ -f "$OUT" ] || { echo "[FAIL] 合并产物缺失: $OUT"; exit 1; }

echo "[OK] merged -> $OUT ($(ls -lh "$OUT" | awk '{print $5}'))"