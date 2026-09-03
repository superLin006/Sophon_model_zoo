#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf '用法: BOARD_IP=<ip> [BOARD_USER=root] [BOARD_PORT=22] [BOARD_DIR=/data/vits_melo_tts] %s [F16|F32] [--test]\n' "$0"
}

PREC=F16
RUN_TEST=0
for arg in "$@"; do
    case "$arg" in
        F16|F32) PREC="$arg" ;;
        --test) RUN_TEST=1 ;;
        *) usage >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOARD_IP="${BOARD_IP:?请设置 BOARD_IP}"
BOARD_USER="${BOARD_USER:-root}"
BOARD_PORT="${BOARD_PORT:-22}"
BOARD_DIR="${BOARD_DIR:-/data/vits_melo_tts}"
MODEL_DIR="${MODEL_DIR:-${SCRIPT_DIR}/models/BM1684X}"
BINARY="${BINARY:-${SCRIPT_DIR}/cpp/build/vits_melo_tts_bm1684}"
TEST_DIR="${TEST_DIR:-${SCRIPT_DIR}/test_data}"
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
    "${MODEL_DIR}/vits_part_a_${PREC}.bmodel"
    "${MODEL_DIR}/vits_part_c1_${PREC}.bmodel"
    "${MODEL_DIR}/vits_part_c2_${PREC}.bmodel"
)
TEST_FILES=(
    test_zh_tokens.bin
    test_zh_tones.bin
    test_en_zh_tokens.bin
    test_en_zh_tones.bin
)

require_file "$BINARY"
for file in "${BMODELS[@]}"; do require_file "$file"; done
if [[ "$RUN_TEST" -eq 1 ]]; then
    for file in "${TEST_FILES[@]}"; do require_file "${TEST_DIR}/${file}"; done
fi

"${SSH[@]}" "$REMOTE" "mkdir -p '${BOARD_DIR}/models' '${BOARD_DIR}/test_data'"
"${SCP[@]}" "$BINARY" "${REMOTE}:${BOARD_DIR}/"
"${SCP[@]}" "${BMODELS[@]}" "${REMOTE}:${BOARD_DIR}/models/"
if [[ "$RUN_TEST" -eq 1 ]]; then
    "${SCP[@]}" "${TEST_FILES[@]/#/${TEST_DIR}/}" "${REMOTE}:${BOARD_DIR}/test_data/"
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
    check_md5 "$file" "${BOARD_DIR}/models/$(basename "$file")"
done
if [[ "$RUN_TEST" -eq 1 ]]; then
    for file in "${TEST_FILES[@]}"; do
        check_md5 "${TEST_DIR}/${file}" "${BOARD_DIR}/test_data/${file}"
    done
fi

if [[ "$RUN_TEST" -eq 1 ]]; then
    "${SSH[@]}" "$REMOTE" "cd '${BOARD_DIR}' && chmod +x './$(basename "$BINARY")' && ./$(basename "$BINARY") test_data/test_zh_tokens.bin test_data/test_zh_tones.bin 61 models output_zh_${PREC}.wav '${PREC}'"
    "${SSH[@]}" "$REMOTE" "cd '${BOARD_DIR}' && ./$(basename "$BINARY") test_data/test_en_zh_tokens.bin test_data/test_en_zh_tones.bin 53 models output_en_zh_${PREC}.wav '${PREC}'"
fi

printf '部署完成: %s（精度=%s）\n' "${REMOTE}:${BOARD_DIR}" "$PREC"
