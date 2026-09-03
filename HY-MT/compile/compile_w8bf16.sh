#!/bin/bash
set -euo pipefail

if [[ -z "${1:-}" ]]; then
    cat >&2 <<'USAGE'
用法: compile_w8bf16.sh <容器内权重目录> [seq_len=512] [输出目录]

权重目录必须在容器内可见。3_docker/run_docker.sh 只把仓库根挂载到 /workspace，
因此应把 HY-MT1.5-1.8B 权重放在仓库内（例如 HY-MT/compile/tmp/），
并传入 /workspace/HY-MT/compile/tmp/<权重目录>。

历史默认值 /models 不是 run_docker.sh 创建的挂载点，直接使用会失败；
确实需要挂载仓库外权重时，请自行 docker run 并追加 -v <宿主权重目录>:/models。
USAGE
    exit 2
fi
MODEL_PATH="$1"
SEQ_LEN="${2:-512}"
OUT_DIR="${3:-/workspace/HY-MT/models/BM1684X/w8bf16_seq${SEQ_LEN}}"

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "[FAIL] 权重目录不存在: ${MODEL_PATH}" >&2
    echo "       容器内唯一挂载是 /workspace（= 仓库根），见上方用法说明。" >&2
    exit 1
fi

python3 /workspace/HY-MT/compile/patch_tpumlir_hymt.py
llm_convert.py \
  -m "${MODEL_PATH}" \
  -s "${SEQ_LEN}" \
  --quantize w8bf16 \
  -c bm1684x \
  --out_dir "${OUT_DIR}"

# L6：产物存在性检查（llm_convert 失败时尽早报错）
if ! ls "${OUT_DIR}"/*.bmodel >/dev/null 2>&1; then
    echo "[FAIL] bmodel 未生成: ${OUT_DIR}"
    exit 1
fi
echo "[OK] ${OUT_DIR}/*.bmodel"
