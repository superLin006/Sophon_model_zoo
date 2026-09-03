#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf '用法: BOARD_IP=<ip> [BOARD_USER=root] [BOARD_PORT=22] [BOARD_DIR=/data/chattts] %s [--test]\n' "$0"
}

RUN_TEST=0
for arg in "$@"; do
    case "$arg" in
        --test) RUN_TEST=1 ;;
        *) usage >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOARD_IP="${BOARD_IP:?请设置 BOARD_IP}"
BOARD_USER="${BOARD_USER:-root}"
BOARD_PORT="${BOARD_PORT:-22}"
BOARD_DIR="${BOARD_DIR:-/data/chattts}"
MODEL_DIR="${MODEL_DIR:-${SCRIPT_DIR}/models}"
BMODEL_DIR="${BMODEL_DIR:-${MODEL_DIR}/BM1684X}"
BINARY="${BINARY:-${SCRIPT_DIR}/cpp/build/chattts}"
SPK_EMB="${SPK_EMB:-}"
REMOTE="${BOARD_USER}@${BOARD_IP}"

if [[ -n "${BOARD_PASS:-}" ]]; then
    command -v sshpass >/dev/null || { printf '未找到 sshpass，或改用 SSH key。\n' >&2; exit 1; }
    SSH=(sshpass -p "${BOARD_PASS}" ssh -o StrictHostKeyChecking=no -p "${BOARD_PORT}")
    SCP=(sshpass -p "${BOARD_PASS}" scp -o StrictHostKeyChecking=no -P "${BOARD_PORT}")
else
    SSH=(ssh -p "${BOARD_PORT}")
    SCP=(scp -P "${BOARD_PORT}")
fi

require_file() {
    [[ -f "$1" ]] || { printf '文件不存在: %s\n' "$1" >&2; exit 1; }
}

require_dir() {
    [[ -d "$1" ]] || { printf '目录不存在: %s\n' "$1" >&2; exit 1; }
}

BMODELS=(
    "${BMODEL_DIR}/chattts-llama_int4_1dev_1024_bm1684x.bmodel"
    "${BMODEL_DIR}/decoder_1-768-1024_bm1684x.bmodel"
    "${BMODEL_DIR}/vocos_1-100-2048_bm1684x.bmodel"
)
ASSET_DIR="${MODEL_DIR}/asset"

require_file "$BINARY"
for file in "${BMODELS[@]}"; do require_file "$file"; done
require_dir "$ASSET_DIR"
if [[ -n "$SPK_EMB" ]]; then require_file "$SPK_EMB"; fi

"${SSH[@]}" "$REMOTE" "mkdir -p '${BOARD_DIR}/models/BM1684X'"
"${SCP[@]}" "$BINARY" "${REMOTE}:${BOARD_DIR}/"
"${SCP[@]}" "${BMODELS[@]}" "${REMOTE}:${BOARD_DIR}/models/BM1684X/"
"${SCP[@]}" -r "$ASSET_DIR" "${REMOTE}:${BOARD_DIR}/models/"
if [[ -n "$SPK_EMB" ]]; then
    "${SCP[@]}" "$SPK_EMB" "${REMOTE}:${BOARD_DIR}/"
fi

check_md5() {
    local local_file="$1"
    local remote_file="$2"
    local local_md5 remote_md5
    local_md5="$(md5sum "$local_file" | cut -d' ' -f1)"
    remote_md5="$("${SSH[@]}" "$REMOTE" "md5sum -- '${remote_file}' | cut -d' ' -f1")"
    [[ "$local_md5" == "$remote_md5" ]] || {
        printf 'md5 不一致: %s (local=%s board=%s)\n' "$local_file" "$local_md5" "$remote_md5" >&2
        exit 1
    }
    printf 'md5 OK: %s\n' "$(basename "$local_file")"
}

check_md5 "$BINARY" "${BOARD_DIR}/$(basename "$BINARY")"
for file in "${BMODELS[@]}"; do
    check_md5 "$file" "${BOARD_DIR}/models/BM1684X/$(basename "$file")"
done
while IFS= read -r -d '' file; do
    relative="${file#"${MODEL_DIR}/"}"
    check_md5 "$file" "${BOARD_DIR}/models/${relative}"
done < <(find "$ASSET_DIR" -type f -print0)
if [[ -n "$SPK_EMB" ]]; then
    check_md5 "$SPK_EMB" "${BOARD_DIR}/$(basename "$SPK_EMB")"
fi

if [[ "$RUN_TEST" -eq 1 ]]; then
    test_output="${BOARD_DIR}/chattts_smoke.wav"
    "${SSH[@]}" "$REMOTE" "cd '${BOARD_DIR}' && chmod +x './$(basename "$BINARY")' && ./$(basename "$BINARY") --model-dir models --text '你好，这是 ChatTTS 的板卡 smoke test。' --output '${test_output}'"
fi

printf '部署完成: %s\n' "${REMOTE}:${BOARD_DIR}"
