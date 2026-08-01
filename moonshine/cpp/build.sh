#!/bin/bash
# 在 sophon-cross-build 容器内交叉编译 moonshine_bm1684（aarch64）
# 用法（从仓库根目录执行）:
#   bash moonshine/cpp/build.sh
# 产物:
#   moonshine/cpp/build/moonshine_bm1684

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"   # Sophon_model_zoo/

echo "================================================================"
echo "  Moonshine BM1684X 交叉编译"
echo "  仓库根目录: ${REPO_ROOT}"
echo "================================================================"

docker run --rm \
    -v "${REPO_ROOT}:/repo" \
    sophon-cross-build \
    bash -c '
set -e

echo "[Build] 配置..."
rm -rf /repo/moonshine/cpp/build
mkdir -p /repo/moonshine/cpp/build
cd /repo/moonshine/cpp/build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
    -DSOPHON_SDK=/repo/0_Toolkits/soc-sdk-sp4 \
    -DCMAKE_VERBOSE_MAKEFILE=OFF \
    2>&1

echo "[Build] 编译..."
make -j$(nproc) 2>&1

echo "[Build] 完成"
ls -lh /repo/moonshine/cpp/build/moonshine_bm1684
'

echo ""
echo "================================================================"
echo "  产物: ${REPO_ROOT}/moonshine/cpp/build/moonshine_bm1684"
echo ""
echo "  上传到板卡:"
echo "    sshpass -p 1 scp moonshine/cpp/build/moonshine_bm1684 \\"
echo "      root@172.16.25.248:/root/moonshine/"
echo "================================================================"
