#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${QWEN_MODEL_DIR:-${QWEN_DIR}/Qwen3-0.6B-dispatch-v95e-soup}"
SEQ_LEN="${SEQ_LEN:-2048}"
QUANTIZE="${QUANTIZE:-w8bf16}"
CONTAINER="${TPUMLIR_CONTAINER:-sophon-tpumlir-v128}"
TMP_DIR="${TMP_DIR:-${QWEN_DIR}/compile/tmp/v95e_soup_${QUANTIZE}_seq${SEQ_LEN}}"
OUT_DIR="${OUT_DIR:-${QWEN_DIR}/models/BM1684X}"
BMODEL_NAME="${BMODEL_NAME:-qwen3_0.6b_dispatch_v95e_soup_${QUANTIZE}_seq${SEQ_LEN}.bmodel}"

[[ -f "${MODEL_DIR}/config.json" ]] || {
    printf '权重目录不存在或缺少 config.json: %s\n' "${MODEL_DIR}" >&2
    exit 1
}

case "${MODEL_DIR}" in
    "${QWEN_DIR}"/*) CONTAINER_MODEL_DIR="/workspace/QwenLLM/${MODEL_DIR#${QWEN_DIR}/}" ;;
    *)
        printf 'QWEN_MODEL_DIR 必须位于仓库目录内，或请先将权重挂载到容器 /workspace 下。\n' >&2
        exit 1
        ;;
esac

mkdir -p "${TMP_DIR}" "${OUT_DIR}"
docker start "${CONTAINER}" >/dev/null

docker exec "${CONTAINER}" bash -lc "
    set -euo pipefail
    llm_convert.py \\
      -m '${CONTAINER_MODEL_DIR}' \\
      -s '${SEQ_LEN}' \\
      --quantize '${QUANTIZE}' \\
      -c bm1684x \\
      --out_dir '/workspace/QwenLLM/compile/tmp/v95e_soup_${QUANTIZE}_seq${SEQ_LEN}'
"

mapfile -t BMODELS < <(find "${TMP_DIR}" -maxdepth 1 -type f -name '*.bmodel' -print)
if [[ "${#BMODELS[@]}" -ne 1 ]]; then
    printf '期望得到 1 个最终 bmodel，实际得到 %s 个: %s\n' "${#BMODELS[@]}" "${TMP_DIR}" >&2
    exit 1
fi

cp "${BMODELS[0]}" "${OUT_DIR}/${BMODEL_NAME}"
printf '完成: %s\n' "${OUT_DIR}/${BMODEL_NAME}"
printf '中间产物: %s\n' "${TMP_DIR}"
