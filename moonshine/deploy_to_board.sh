#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf '用法: BOARD_IP=<ip> [BOARD_USER=root] [BOARD_PORT=22] [BOARD_DIR=/data/moonshine] [BMODEL_DIR=models/BM1684X] [ASSET_DIR=models] %s [F16|F32] [--test]\n' "$0"
}

PRECISION=F16
RUN_TEST=0
for arg in "$@"; do
    case "$arg" in
        F16|F32) PRECISION="$arg" ;;
        --test) RUN_TEST=1 ;;
        *) usage >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOARD_IP="${BOARD_IP:?请设置 BOARD_IP}"
BOARD_USER="${BOARD_USER:-root}"
BOARD_PORT="${BOARD_PORT:-22}"
BOARD_DIR="${BOARD_DIR:-/data/moonshine}"
# bmodel 与 tokens.txt 在仓库内不同层：bmodel 在 models/BM1684X/（产物布局见
# .claude/standards/models_directory_standard.md §1），tokens.txt 在 models/。
# 因此拆成两个变量；旧的单一 MODEL_DIR 默认值无法同时满足，会在 require_file 处失败。
BMODEL_DIR="${BMODEL_DIR:-${SCRIPT_DIR}/models/BM1684X}"
ASSET_DIR="${ASSET_DIR:-${SCRIPT_DIR}/models}"
BINARY="${BINARY:-${SCRIPT_DIR}/cpp/build/moonshine_bm1684}"
TEST_AUDIO="${TEST_AUDIO:-${SCRIPT_DIR}/test_data/0.wav}"
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
    "${BMODEL_DIR}/moonshine_encoder_${PRECISION}.bmodel"
    "${BMODEL_DIR}/moonshine_decoder_${PRECISION}.bmodel"
)
require_file "$BINARY"
require_file "${ASSET_DIR}/tokens.txt"
for file in "${BMODELS[@]}"; do require_file "$file"; done
if [[ "$RUN_TEST" -eq 1 ]]; then require_file "$TEST_AUDIO"; fi

"${SSH[@]}" "$REMOTE" "mkdir -p '${BOARD_DIR}/models' '${BOARD_DIR}/test_data'"
"${SCP[@]}" "$BINARY" "${REMOTE}:${BOARD_DIR}/"
"${SCP[@]}" "${BMODELS[@]}" "${ASSET_DIR}/tokens.txt" "${REMOTE}:${BOARD_DIR}/models/"
if [[ "$RUN_TEST" -eq 1 ]]; then
    "${SCP[@]}" "$TEST_AUDIO" "${REMOTE}:${BOARD_DIR}/test_data/test.wav"
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
check_md5 "${ASSET_DIR}/tokens.txt" "${BOARD_DIR}/models/tokens.txt"

if [[ "$RUN_TEST" -eq 1 ]]; then
    "${SSH[@]}" "$REMOTE" "cd '${BOARD_DIR}' && chmod +x './$(basename "$BINARY")' && ./$(basename "$BINARY") models test_data/test.wav '${PRECISION}'"
fi

printf '部署完成: %s（精度=%s）\n' "${REMOTE}:${BOARD_DIR}" "$PRECISION"
