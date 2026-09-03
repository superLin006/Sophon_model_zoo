#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMAGE="${CROSS_BUILD_IMAGE:-sophon-cross-build}"
BUILD_DIR="${SCRIPT_DIR}/build"

if [[ "${1:-cross}" != "cross" ]]; then
    printf '用法: %s [cross]\n' "$0" >&2
    exit 2
fi

docker image inspect "$IMAGE" >/dev/null 2>&1 || {
    printf '交叉编译镜像不存在: %s\n请先构建 3_docker/Dockerfile.cross-build\n' "$IMAGE" >&2
    exit 1
}

docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "${REPO_ROOT}:/workspace" \
    -w /workspace \
    "$IMAGE" bash -lc '
        set -euo pipefail
        cmake -S chatTTS/cpp -B chatTTS/cpp/build \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_TOOLCHAIN_FILE=chatTTS/cpp/cmake/aarch64.cmake \
          -DTHIRD_PARTY_DIR=/workspace/1_third_party
        cmake --build chatTTS/cpp/build -j"${JOBS:-$(nproc)}"
    '

printf '构建完成:\n  %s\n  %s\n' \
    "${BUILD_DIR}/chattts" \
    "${BUILD_DIR}/chattts_bench"
