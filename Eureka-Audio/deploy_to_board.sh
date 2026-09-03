#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf '用法: BOARD_IP=<ip> [BOARD_USER=root] [BOARD_PORT=22] [BOARD_DIR=/data/eureka_audio] %s [--test]\n' "$0"
}

if [[ $# -gt 1 || ( $# -eq 1 && $1 != "--test" ) ]]; then
    usage >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOARD_IP="${BOARD_IP:?请设置 BOARD_IP}"
BOARD_USER="${BOARD_USER:-root}"
BOARD_PORT="${BOARD_PORT:-22}"
BOARD_DIR="${BOARD_DIR:-/data/eureka_audio}"
MODEL_DIR="${MODEL_DIR:-${SCRIPT_DIR}/models/BM1684X}"
ASSET_DIR="${ASSET_DIR:-${SCRIPT_DIR}/../Eureka-Audio-Instruct}"
BINARY="${BINARY:-${SCRIPT_DIR}/cpp/build/eureka_audio_bm1684x}"
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
    "${MODEL_DIR}/whisper_encoder_b1_bf16.bmodel"
    "${MODEL_DIR}/qwen3_1.7b_embeds_w4bf16_seq512_bm1684x.bmodel"
)
ASSETS=(
    prefix_embeds.bin
    suffix_embeds.bin
    mel_filters.npz
    tokenizer.json
)

require_file "$BINARY"
for file in "${BMODELS[@]}"; do require_file "$file"; done
for name in "${ASSETS[@]}"; do require_file "${ASSET_DIR}/${name}"; done

TEST_AUDIO="${TEST_AUDIO:-${SCRIPT_DIR}/test_audios/qa_example.wav}"
if [[ "${1:-}" == "--test" ]]; then require_file "$TEST_AUDIO"; fi

"${SSH[@]}" "$REMOTE" "mkdir -p '${BOARD_DIR}/models/BM1684X' '${BOARD_DIR}/test_data'"

"${SCP[@]}" "$BINARY" "${REMOTE}:${BOARD_DIR}/"
for file in "${BMODELS[@]}"; do
    "${SCP[@]}" "$file" "${REMOTE}:${BOARD_DIR}/models/BM1684X/"
done
for name in "${ASSETS[@]}"; do
    "${SCP[@]}" "${ASSET_DIR}/${name}" "${REMOTE}:${BOARD_DIR}/"
done
if [[ "${1:-}" == "--test" ]]; then
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
    check_md5 "$file" "${BOARD_DIR}/models/BM1684X/$(basename "$file")"
done
for name in "${ASSETS[@]}"; do
    check_md5 "${ASSET_DIR}/${name}" "${BOARD_DIR}/${name}"
done

if [[ "${1:-}" == "--test" ]]; then
    "${SSH[@]}" "$REMOTE" "cd '${BOARD_DIR}' && chmod +x './$(basename "$BINARY")' && ./$(basename "$BINARY") --whisper_bmodel models/BM1684X/whisper_encoder_b1_bf16.bmodel --qwen3_bmodel models/BM1684X/qwen3_1.7b_embeds_w4bf16_seq512_bm1684x.bmodel --model_dir . --audio test_data/test.wav --max_new_tokens 64"
fi

printf '部署完成: %s\n' "${REMOTE}:${BOARD_DIR}"
