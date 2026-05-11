#!/bin/bash
# 在 Ubuntu 20.04 容器内交叉编译 whisper_bm1684（aarch64）
# 产物与服务器（Ubuntu 20.04 + glibc 2.31）完全兼容
#
# 用法（从任意目录执行）:
#   bash whisper/cpp/build.sh
#
# 产物:
#   whisper/cpp/build/whisper_bm1684

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"   # Sophon_model_zoo/

echo "================================================================"
echo "  Whisper BM1684X 交叉编译"
echo "  仓库根目录: ${REPO_ROOT}"
echo "================================================================"

docker run --rm \
    -v "${REPO_ROOT}:/repo" \
    sophon-cross-build \
    bash -c '
set -e

echo "[Build] 配置..."
rm -rf /repo/whisper/cpp/build
mkdir -p /repo/whisper/cpp/build
cd /repo/whisper/cpp/build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
    -DSOPHON_SDK=/repo/0_Toolkits/soc-sdk-sp4 \
    -DFFTW_DIR=/repo/1_third_party/fftw \
    -DCMAKE_VERBOSE_MAKEFILE=OFF \
    2>&1

echo "[Build] 编译..."
make -j$(nproc) 2>&1

echo "[Build] 完成"
ls -lh /repo/whisper/cpp/build/whisper_bm1684
'

echo ""
echo "================================================================"
echo "  产物: ${REPO_ROOT}/whisper/cpp/build/whisper_bm1684"
echo ""
echo "  上传到服务器:"
echo "    sshpass -p '1' scp whisper/cpp/build/whisper_bm1684 \\"
echo "      root@172.16.40.75:/home/xiehuan_ai/whisper/"
echo "================================================================"
