#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf '用法: BOARD_IP=<ip> [BOARD_USER=root] [BOARD_PORT=22] [BOARD_DIR=/data/qwenllm] %s [--test]\n' "$0"
    printf '可用 BMODEL 覆盖默认的 v95e-soup bmodel 路径。\n'
}

if [[ $# -gt 1 || ( $# -eq 1 && $1 != "--test" ) ]]; then
    usage >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BOARD_IP="${BOARD_IP:?请设置 BOARD_IP}"
BOARD_USER="${BOARD_USER:-root}"
BOARD_PORT="${BOARD_PORT:-22}"
BOARD_DIR="${BOARD_DIR:-/data/qwenllm}"
BMODEL="${BMODEL:-${QWEN_DIR}/models/BM1684X/qwen3_0.6b_dispatch_v95e_soup_w8bf16_seq2048.bmodel}"
CONFIG_DIR="${CONFIG_DIR:-${QWEN_DIR}/demo/config}"
DEMO_DIR="${DEMO_DIR:-${QWEN_DIR}/demo/python_demo}"
BENCHMARK="${BENCHMARK:-${QWEN_DIR}/benchmark_intent.py}"
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
require_file "$BENCHMARK"
for name in tokenizer.json tokenizer_config.json vocab.json config.json generation_config.json; do
    [[ -f "${CONFIG_DIR}/${name}" ]] || {
        printf '配置文件不存在: %s\n' "${CONFIG_DIR}/${name}" >&2
        exit 1
    }
done
[[ -f "${DEMO_DIR}/CMakeLists.txt" ]] || { printf 'Python demo 不存在: %s\n' "${DEMO_DIR}" >&2; exit 1; }

SCP_RECURSIVE=("${SCP[@]}" -r)

BMODEL_NAME="$(basename "$BMODEL")"
"${SSH[@]}" "$REMOTE" "mkdir -p '${BOARD_DIR}/models' '${BOARD_DIR}/config'"
"${SCP_RECURSIVE[@]}" "$DEMO_DIR" "${REMOTE}:${BOARD_DIR}/"
"${SCP[@]}" "$BMODEL" "${REMOTE}:${BOARD_DIR}/models/"
"${SCP[@]}" "$BENCHMARK" "${REMOTE}:${BOARD_DIR}/"
for name in tokenizer.json tokenizer_config.json vocab.json config.json generation_config.json; do
    "${SCP[@]}" "${CONFIG_DIR}/${name}" "${REMOTE}:${BOARD_DIR}/config/"
done

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

check_md5 "$BMODEL" "${BOARD_DIR}/models/${BMODEL_NAME}"
for name in tokenizer.json tokenizer_config.json vocab.json config.json generation_config.json; do
    check_md5 "${CONFIG_DIR}/${name}" "${BOARD_DIR}/config/${name}"
done

if [[ "${1:-}" == "--test" ]]; then
    "${SSH[@]}" "$REMOTE" "cd '${BOARD_DIR}/python_demo' && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4 && cp build/chat*.so ."
    "${SSH[@]}" "$REMOTE" "cd '${BOARD_DIR}' && python3 benchmark_intent.py -m 'models/${BMODEL_NAME}' -c config -n qwen3-0.6b --no_think"
fi

printf '部署完成: %s\n' "${REMOTE}:${BOARD_DIR}"
