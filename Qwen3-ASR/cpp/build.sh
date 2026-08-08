#!/bin/bash
# 在 sophon-cross-build 容器内交叉编译 qwen3_asr_bm1684x（aarch64）
#
# 用法（从任意目录执行）:
#   bash Qwen3-ASR/cpp/build.sh
#
# 产物:
#   Qwen3-ASR/cpp/build/qwen3_asr_bm1684x

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"   # Sophon_model_zoo/

echo "================================================================"
echo "  Qwen3-ASR BM1684X 交叉编译"
echo "  仓库根目录: ${REPO_ROOT}"
echo "================================================================"

docker run --rm \
    -v "${REPO_ROOT}:/repo" \
    sophon-cross-build \
    bash -c '
set -e

echo "[Build] 配置..."
rm -rf /repo/Qwen3-ASR/cpp/build
mkdir -p /repo/Qwen3-ASR/cpp/build
cd /repo/Qwen3-ASR/cpp/build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
    -DSOPHON_SDK=/repo/0_Toolkits/soc-sdk-sp4 \
    -DNLOHMANN_DIR=/repo/1_third_party \
    -DCMAKE_VERBOSE_MAKEFILE=OFF \
    2>&1

echo "[Build] 编译..."
make -j$(nproc) 2>&1

echo "[Build] 完成"
ls -lh /repo/Qwen3-ASR/cpp/build/qwen3_asr_bm1684x
'

echo ""
echo "================================================================"
echo "  产物: ${REPO_ROOT}/Qwen3-ASR/cpp/build/qwen3_asr_bm1684x"
echo "================================================================"
