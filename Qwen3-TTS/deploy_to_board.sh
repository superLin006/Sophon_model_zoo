#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf '用法: BOARD_IP=<ip> [BOARD_USER=root] [BOARD_PORT=22] [BOARD_DIR=/data/Qwen3-TTS] %s [--test]\n' "$0"
}

if [[ $# -gt 1 || ( $# -eq 1 && $1 != "--test" ) ]]; then
    usage >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOARD_IP="${BOARD_IP:?请设置 BOARD_IP}"
BOARD_USER="${BOARD_USER:-root}"
BOARD_PORT="${BOARD_PORT:-22}"
BOARD_DIR="${BOARD_DIR:-/data/Qwen3-TTS}"
MODEL_DIR="${MODEL_DIR:-${SCRIPT_DIR}/models/BM1684X}"
BINARY="${BINARY:-${SCRIPT_DIR}/cpp/build/qwen3_tts_bm1684x}"
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

BMODEL_NAMES=(
    qwen3_tts_talker_w8s192.bmodel
    qwen3_tts_cp_allf32.bmodel
    codec_decoder.bmodel
)
require_file "$BINARY"
for name in "${BMODEL_NAMES[@]}"; do require_file "${MODEL_DIR}/${name}"; done
require_file "${SCRIPT_DIR}/assets/tokenizer.json"

"${SSH[@]}" "$REMOTE" "mkdir -p '${BOARD_DIR}/models' '${BOARD_DIR}/assets'"
"${SCP[@]}" "$BINARY" "${REMOTE}:${BOARD_DIR}/"
for name in "${BMODEL_NAMES[@]}"; do
    "${SCP[@]}" "${MODEL_DIR}/${name}" "${REMOTE}:${BOARD_DIR}/models/"
done
"${SCP[@]}" "${SCRIPT_DIR}/assets/tokenizer.json" "${REMOTE}:${BOARD_DIR}/assets/"

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
for name in "${BMODEL_NAMES[@]}"; do
    check_md5 "${MODEL_DIR}/${name}" "${BOARD_DIR}/models/${name}"
done
check_md5 "${SCRIPT_DIR}/assets/tokenizer.json" "${BOARD_DIR}/assets/tokenizer.json"

if [[ "${1:-}" == "--test" ]]; then
    "${SSH[@]}" "$REMOTE" "cd '${BOARD_DIR}' && chmod +x './$(basename "$BINARY")' && ./$(basename "$BINARY") --talker_bmodel models/qwen3_tts_talker_w8s192.bmodel --cp_bmodel models/qwen3_tts_cp_allf32.bmodel --codec_bmodel models/codec_decoder.bmodel --model_dir assets --text '你好。' --speaker Vivian --language Chinese --sample --seed 42 --out test.wav"
fi

printf '部署完成: %s\n' "${REMOTE}:${BOARD_DIR}"
