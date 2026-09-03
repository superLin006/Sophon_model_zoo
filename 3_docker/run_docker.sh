#!/usr/bin/env bash
set -euo pipefail

# 启动公共 TPU-MLIR 转换容器
# 用法: TPUMLIR_IMAGE=<image> ./3_docker/run_docker.sh [容器名]

CONTAINER_NAME=${1:-sophon-tpumlir-v128}
IMAGE=${TPUMLIR_IMAGE:-sophon/tpuc_dev:v3.4-tpumlir-1.28.1}
WORKSPACE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

container_image() {
    docker inspect --format '{{.Config.Image}}' "$CONTAINER_NAME"
}

if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    CURRENT_IMAGE=$(container_image)
    if [[ "$CURRENT_IMAGE" != "$IMAGE" ]]; then
        printf '[ERROR] 容器 %s 使用镜像 %s，期望 %s；请换容器名或先手动删除旧容器。\n' \
            "$CONTAINER_NAME" "$CURRENT_IMAGE" "$IMAGE" >&2
        exit 2
    fi
    echo "[INFO] 容器 ${CONTAINER_NAME} 已在运行，进入容器..."
    docker exec -it "$CONTAINER_NAME" bash
    exit 0
fi

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    CURRENT_IMAGE=$(container_image)
    if [[ "$CURRENT_IMAGE" != "$IMAGE" ]]; then
        printf '[ERROR] 容器 %s 使用镜像 %s，期望 %s；请换容器名或先手动删除旧容器。\n' \
            "$CONTAINER_NAME" "$CURRENT_IMAGE" "$IMAGE" >&2
        exit 2
    fi
    echo "[INFO] 启动已有容器 ${CONTAINER_NAME}..."
    docker start -i "$CONTAINER_NAME"
    exit 0
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    printf '[ERROR] 镜像不存在：%s\n请先运行 ./3_docker/build_tpumlir.sh\n' "$IMAGE" >&2
    exit 2
fi

echo "[INFO] 创建容器 ${CONTAINER_NAME}，镜像 ${IMAGE}，挂载 ${WORKSPACE_DIR} -> /workspace"
docker run --privileged \
    --name "$CONTAINER_NAME" \
    -v "$WORKSPACE_DIR":/workspace \
    -w /workspace \
    -it "$IMAGE"
