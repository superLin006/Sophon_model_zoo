#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf '用法: BOARD_IP=<ip> [BOARD_USER=root] [BOARD_PORT=22] [BOARD_DIR=/data/zipformer] %s [--test]\n' "$0"
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
BOARD_DIR="${BOARD_DIR:-/data/zipformer}"
MODEL_DIR="${MODEL_DIR:-${SCRIPT_DIR}/models/BM1684X}"
CONFIG_DIR="${CONFIG_DIR:-${SCRIPT_DIR}/configs}"
BINARY="${BINARY:-${SCRIPT_DIR}/cpp/build-aarch64/zipformer_cli}"
TEST_AUDIO="${TEST_AUDIO:-${SCRIPT_DIR}/test_data/test_en.wav}"
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

BMODELS=(
    "${MODEL_DIR}/zipformer_encoder_f16.bmodel"
    "${MODEL_DIR}/zipformer_decoder_f16.bmodel"
    "${MODEL_DIR}/zipformer_joiner_f16.bmodel"
)
require_file "$BINARY"
for file in "${BMODELS[@]}"; do require_file "$file"; done
require_file "${CONFIG_DIR}/tensor_manifest.json"
require_file "${CONFIG_DIR}/tokens.txt"
if [[ "$RUN_TEST" -eq 1 ]]; then require_file "$TEST_AUDIO"; fi

"${SSH[@]}" "$REMOTE" "mkdir -p '${BOARD_DIR}/models' '${BOARD_DIR}/configs' '${BOARD_DIR}/test_data'"
"${SCP[@]}" "$BINARY" "${REMOTE}:${BOARD_DIR}/"
"${SCP[@]}" "${BMODELS[@]}" "${REMOTE}:${BOARD_DIR}/models/"
"${SCP[@]}" "${CONFIG_DIR}/tensor_manifest.json" "${CONFIG_DIR}/tokens.txt" "${REMOTE}:${BOARD_DIR}/configs/"
if [[ "$RUN_TEST" -eq 1 ]]; then "${SCP[@]}" "$TEST_AUDIO" "${REMOTE}:${BOARD_DIR}/test_data/"; fi

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
for file in "${BMODELS[@]}"; do check_md5 "$file" "${BOARD_DIR}/models/$(basename "$file")"; done
check_md5 "${CONFIG_DIR}/tensor_manifest.json" "${BOARD_DIR}/configs/tensor_manifest.json"
check_md5 "${CONFIG_DIR}/tokens.txt" "${BOARD_DIR}/configs/tokens.txt"
if [[ "$RUN_TEST" -eq 1 ]]; then
    check_md5 "$TEST_AUDIO" "${BOARD_DIR}/test_data/$(basename "$TEST_AUDIO")"
    "${SSH[@]}" "$REMOTE" "cd '${BOARD_DIR}' && chmod +x './$(basename "$BINARY")' && ./$(basename "$BINARY") configs/tensor_manifest.json test_data/$(basename "$TEST_AUDIO") models/zipformer_encoder_f16.bmodel models/zipformer_decoder_f16.bmodel models/zipformer_joiner_f16.bmodel configs/tokens.txt"
fi

printf '部署完成: %s\n' "${REMOTE}:${BOARD_DIR}"
