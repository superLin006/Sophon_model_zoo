#!/bin/bash
# Qwen3 BM1684X 纯 C++ Demo 交叉编译脚本
#
# ⚠️  必须在 Ubuntu 20.04 / sophon-cross-build docker 下编译
#     板卡为 Ubuntu 20.04，最高支持 glibc 2.31
#     Ubuntu 22+ / 24+ 的 GCC 产物需要 GLIBC_2.34+，板上无法运行
#
# 用法:
#   ./build.sh              # 编译，产物在 build/
#   ./build.sh install      # 编译 + 安装到 dist/（含 .so）

# 注意：3rdparty/bm1684x/libsophon/aarch64/*.so 和 prebuilt/libtokenizers_c.a
# 因 .gitignore 未纳入版本控制，需手动从以下来源获取：
#   .so  → 板卡 /opt/sophon/libsophon-current/lib/，或 llm-sdk/3rdparty/bm1684x/libsophon/aarch64/
#   .a   → 从 llm-sdk/3rdparty/bm1684x/prebuilt/libtokenizers_c.a 复制到此处同名路径

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
DIST_DIR="${SCRIPT_DIR}/dist"

# 检查交叉编译工具链
if ! command -v aarch64-linux-gnu-g++ &>/dev/null; then
    echo "❌ 未找到 aarch64-linux-gnu-g++，请安装："
    echo "   sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu"
    exit 1
fi

echo "✅ 使用工具链: $(aarch64-linux-gnu-g++ --version | head -1)"

cmake -S "${SCRIPT_DIR}" \
      -B "${BUILD_DIR}" \
      -DCMAKE_TOOLCHAIN_FILE="${SCRIPT_DIR}/../../llm-sdk/cmake/toolchains/sophon_linux.cmake" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="${DIST_DIR}"

cmake --build "${BUILD_DIR}" -j"$(nproc)"

if [[ "${1}" == "install" ]]; then
    cmake --install "${BUILD_DIR}"
    echo ""
    echo "✅ 安装完成，产物目录: ${DIST_DIR}"
    echo "   上板部署: scp -r ${DIST_DIR}/* root@<board_ip>:/data/qwen_demo/"
fi

echo ""
echo "✅ 编译完成: ${BUILD_DIR}/qwen_demo"
