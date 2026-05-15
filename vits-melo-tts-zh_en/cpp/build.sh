#!/bin/bash
# 交叉编译 vits_melo_tts_bm1684（aarch64 BM1684X）
# 使用 Docker sophon-cross-build（GCC 9.4，GLIBC 2.17 兼容）
#
# 用法（从仓库根目录执行）:
#   bash vits-melo-tts-zh_en/cpp/build.sh
#
# 产物:
#   vits-melo-tts-zh_en/cpp/build/vits_melo_tts_bm1684

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "================================================================"
echo "  VITS-MeloTTS BM1684X 交叉编译"
echo "  仓库根目录: ${REPO_ROOT}"
echo "================================================================"

docker run --rm \
    -v "${REPO_ROOT}:/repo" \
    sophon-cross-build \
    bash -c '
set -e

echo "[Build] 配置..."
rm -rf /repo/vits-melo-tts-zh_en/cpp/build
mkdir -p /repo/vits-melo-tts-zh_en/cpp/build
cd /repo/vits-melo-tts-zh_en/cpp/build

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
ls -lh /repo/vits-melo-tts-zh_en/cpp/build/vits_melo_tts_bm1684
'

echo ""
echo "================================================================"
echo "  产物: ${REPO_ROOT}/vits-melo-tts-zh_en/cpp/build/vits_melo_tts_bm1684"
echo "================================================================"
