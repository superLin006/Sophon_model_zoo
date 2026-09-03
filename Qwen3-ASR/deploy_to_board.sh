#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf '用法: BOARD_IP=<ip> [BOARD_USER=root] [BOARD_PORT=22] [BOARD_DIR=/data/qwen3_asr] %s [--test]\n' "$0"
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
BOARD_DIR="${BOARD_DIR:-/data/qwen3_asr}"
MODEL_DIR="${MODEL_DIR:-${SCRIPT_DIR}/models/qwen3-asr-0.6b}"
BMODEL="${BMODEL:-${SCRIPT_DIR}/models/BM1684X/qwen3_asr_merged_w4f16.bmodel}"
BINARY="${BINARY:-${SCRIPT_DIR}/cpp/build/qwen3_asr_bm1684x}"
TEST_AUDIO_DIR="${TEST_AUDIO_DIR:-${SCRIPT_DIR}/test_data}"
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

require_file "$BMODEL"
require_file "$BINARY"
ASSETS=(
    "${MODEL_DIR}/prefix_ids.txt"
    "${MODEL_DIR}/suffix_ids.txt"
    "${MODEL_DIR}/mel_filters.npz"
    "${MODEL_DIR}/tokenizer.json"
)
for file in "${ASSETS[@]}"; do require_file "$file"; done

TEST_AUDIO_FILES=()
if [[ "$RUN_TEST" -eq 1 ]]; then
    while IFS= read -r -d '' file; do TEST_AUDIO_FILES+=("$file"); done < <(find "$TEST_AUDIO_DIR" -maxdepth 1 -type f -name '*.wav' -print0)
    ((${#TEST_AUDIO_FILES[@]} > 0)) || { printf '未找到测试音频: %s\n' "$TEST_AUDIO_DIR" >&2; exit 1; }
fi

"${SSH[@]}" "$REMOTE" "mkdir -p '${BOARD_DIR}/models/BM1684X' '${BOARD_DIR}/test_data'"
"${SCP[@]}" "$BMODEL" "${REMOTE}:${BOARD_DIR}/models/BM1684X/"
"${SCP[@]}" "${ASSETS[@]}" "${REMOTE}:${BOARD_DIR}/"
"${SCP[@]}" "$BINARY" "${REMOTE}:${BOARD_DIR}/"
if [[ "$RUN_TEST" -eq 1 ]]; then
    "${SCP[@]}" "${TEST_AUDIO_FILES[@]}" "${REMOTE}:${BOARD_DIR}/test_data/"
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

check_md5 "$BMODEL" "${BOARD_DIR}/models/BM1684X/$(basename "$BMODEL")"
for file in "${ASSETS[@]}"; do check_md5 "$file" "${BOARD_DIR}/$(basename "$file")"; done
check_md5 "$BINARY" "${BOARD_DIR}/$(basename "$BINARY")"
if [[ "$RUN_TEST" -eq 1 ]]; then
    for file in "${TEST_AUDIO_FILES[@]}"; do
        check_md5 "$file" "${BOARD_DIR}/test_data/$(basename "$file")"
    done
    "${SSH[@]}" "$REMOTE" "cd '${BOARD_DIR}' && chmod +x './$(basename "$BINARY")' && ./$(basename "$BINARY") --bmodel 'models/BM1684X/$(basename "$BMODEL")' --model_dir . --audio_dir test_data"
fi

printf '部署完成: %s\n' "${REMOTE}:${BOARD_DIR}"
