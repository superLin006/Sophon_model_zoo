#!/bin/bash
# 启动 TPU-MLIR 转换容器
# 用法: ./docker/run_docker.sh [容器名]

CONTAINER_NAME=${1:-sophon-tpumlir}
IMAGE="sophgo/tpuc_dev:latest"
WORKSPACE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# 检查容器是否已在运行
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[INFO] 容器 ${CONTAINER_NAME} 已在运行，进入容器..."
    docker exec -it "${CONTAINER_NAME}" bash
    exit 0
fi

# 检查容器是否已存在（停止状态）
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[INFO] 启动已有容器 ${CONTAINER_NAME}..."
    docker start -i "${CONTAINER_NAME}"
    exit 0
fi

# 新建容器
echo "[INFO] 创建新容器 ${CONTAINER_NAME}，挂载 ${WORKSPACE_DIR} -> /workspace"
docker run --privileged \
    --name "${CONTAINER_NAME}" \
    -v "${WORKSPACE_DIR}":/workspace \
    -w /workspace \
    -it "${IMAGE}"
