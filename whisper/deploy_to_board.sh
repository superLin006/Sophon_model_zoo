#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf '用法: BOARD_IP=<ip> [MODEL_VARIANT=base] [PRECISION=F16] [BOARD_DIR=/data/whisper] %s [--test]\n' "$0"
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
BOARD_DIR="${BOARD_DIR:-/data/whisper}"
MODEL_VARIANT="${MODEL_VARIANT:-base}"
PRECISION="${PRECISION:-F16}"
MODEL_DIR="${MODEL_DIR:-${SCRIPT_DIR}/models/BM1684X}"
BINARY="${BINARY:-${SCRIPT_DIR}/cpp/build/whisper_bm1684}"
TEST_AUDIO="${TEST_AUDIO:-${SCRIPT_DIR}/test_data/test_zh.wav}"
REMOTE="${BOARD_USER}@${BOARD_IP}"

case "$MODEL_VARIANT" in
    base) MODEL_NAME=base; MODEL_DIR_NAME=BM1684X ;;
    turbo) MODEL_NAME=turbo; MODEL_DIR_NAME=BM1684X_turbo ;;
    turbo_w4f16) MODEL_NAME=turbo; MODEL_DIR_NAME=BM1684X_turbo_w4f16; PRECISION=W4F16 ;;
    *) printf '不支持的 MODEL_VARIANT: %s\n' "$MODEL_VARIANT" >&2; exit 2 ;;
esac

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

LOCAL_MODEL_DIR="${MODEL_DIR%/}"
if [[ "$MODEL_DIR" == "${SCRIPT_DIR}/models/BM1684X" ]]; then
    LOCAL_MODEL_DIR="${SCRIPT_DIR}/models/${MODEL_DIR_NAME}"
fi
BMODEL_PREFIX="whisper_${MODEL_NAME}"
BMODELS=(
    "${LOCAL_MODEL_DIR}/${BMODEL_PREFIX}_encoder_${PRECISION}.bmodel"
    "${LOCAL_MODEL_DIR}/${BMODEL_PREFIX}_decoder_${PRECISION}.bmodel"
)
for file in "${BMODELS[@]}"; do require_file "$file"; done

ASSETS=()
while IFS= read -r -d '' file; do ASSETS+=("$file"); done < <(find "$LOCAL_MODEL_DIR" -maxdepth 1 -type f ! -name '*.bmodel' -print0)
((${#ASSETS[@]} > 0)) || { printf '未找到 Whisper 运行时资产: %s\n' "$LOCAL_MODEL_DIR" >&2; exit 1; }
if [[ "$RUN_TEST" -eq 1 ]]; then require_file "$TEST_AUDIO"; fi

REMOTE_MODEL_DIR="${BOARD_DIR}/models/${MODEL_VARIANT}"
"${SSH[@]}" "$REMOTE" "mkdir -p '${REMOTE_MODEL_DIR}' '${BOARD_DIR}/test_data'"
"${SCP[@]}" "$BINARY" "${REMOTE}:${BOARD_DIR}/"
"${SCP[@]}" "${BMODELS[@]}" "${ASSETS[@]}" "${REMOTE}:${REMOTE_MODEL_DIR}/"
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
for file in "${BMODELS[@]}" "${ASSETS[@]}"; do
    check_md5 "$file" "${REMOTE_MODEL_DIR}/$(basename "$file")"
done
if [[ "$RUN_TEST" -eq 1 ]]; then
    check_md5 "$TEST_AUDIO" "${BOARD_DIR}/test_data/$(basename "$TEST_AUDIO")"
    "${SSH[@]}" "$REMOTE" "cd '${BOARD_DIR}' && chmod +x './$(basename "$BINARY")' && ./$(basename "$BINARY") models/${MODEL_VARIANT} test_data/$(basename "$TEST_AUDIO") zh '${PRECISION}' '${MODEL_NAME}'"
fi

printf '部署完成: %s（模型=%s，精度=%s）\n' "${REMOTE}:${BOARD_DIR}" "$MODEL_VARIANT" "$PRECISION"
