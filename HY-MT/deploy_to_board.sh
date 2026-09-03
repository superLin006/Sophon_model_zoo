#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf '用法: BOARD_IP=<ip> [MODEL_VARIANT=w8bf16_seq512] [BOARD_DIR=/data/hymt] %s [--test]\n' "$0"
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
MODEL_VARIANT="${MODEL_VARIANT:-w8bf16_seq512}"
BOARD_DIR="${BOARD_DIR:-/data/hymt}"
MODEL_DIR="${MODEL_DIR:-${SCRIPT_DIR}/models/BM1684X/${MODEL_VARIANT}}"
BINARY="${BINARY:-${SCRIPT_DIR}/cpp/build-aarch64-v2/hymt_demo}"
TEST_PROMPT="${TEST_PROMPT:-Translate the following segment into Chinese, without additional explanation.\\n\\nIt’s on the house.}"
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

require_file "$BINARY"
require_dir "${MODEL_DIR}/config"
BMODEL="$(find "$MODEL_DIR" -maxdepth 1 -type f -name '*.bmodel' -print -quit)"
[[ -n "$BMODEL" ]] || { printf '版本目录根部未找到 bmodel: %s\n' "$MODEL_DIR" >&2; exit 1; }

CONFIG_FILES=()
while IFS= read -r -d '' file; do CONFIG_FILES+=("$file"); done < <(find "${MODEL_DIR}/config" -type f -print0)
((${#CONFIG_FILES[@]} > 0)) || { printf 'config 目录为空: %s\n' "${MODEL_DIR}/config" >&2; exit 1; }

"${SSH[@]}" "$REMOTE" "mkdir -p '${BOARD_DIR}/config'"
"${SCP[@]}" "$BINARY" "$BMODEL" "${REMOTE}:${BOARD_DIR}/"
"${SCP[@]}" -r "${MODEL_DIR}/config" "${REMOTE}:${BOARD_DIR}/"

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
check_md5 "$BMODEL" "${BOARD_DIR}/$(basename "$BMODEL")"
for file in "${CONFIG_FILES[@]}"; do
    relative="${file#"${MODEL_DIR}/"}"
    check_md5 "$file" "${BOARD_DIR}/${relative}"
done

if [[ "$RUN_TEST" -eq 1 ]]; then
    "${SSH[@]}" "$REMOTE" "cd '${BOARD_DIR}' && chmod +x './$(basename "$BINARY")' && ./$(basename "$BINARY") . '${TEST_PROMPT}' 128"
fi

printf '部署完成: %s（版本=%s）\n' "${REMOTE}:${BOARD_DIR}" "$MODEL_VARIANT"
