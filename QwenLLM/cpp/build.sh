#!/bin/bash
# Qwen3 BM1684X 纯 C++ Demo 交叉编译脚本
#
# 推荐在 3_docker/ 的 cross-build docker 中运行（Ubuntu 20.04，glibc 2.31）：
#   cd /workspace/QwenLLM/cpp && ./build.sh
#
# 用法:
#   ./build.sh              # 编译，产物在 build/qwen_demo
#   ./build.sh install      # 编译 + 安装到 dist/（含 .so，方便 scp 到板卡）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
DIST_DIR="${SCRIPT_DIR}/dist"

if ! command -v aarch64-linux-gnu-g++ &>/dev/null; then
    echo "❌ 未找到 aarch64-linux-gnu-g++，请在 3_docker/cross-build docker 中运行"
    exit 1
fi

# 依赖检查
TOKENIZERS_DIR="${SCRIPT_DIR}/../../1_third_party/tokenizers-cpp"
if [[ ! -f "${TOKENIZERS_DIR}/aarch64-linux/libtokenizers_cpp.a" ]]; then
    echo "❌ 未找到 1_third_party/tokenizers-cpp/aarch64-linux/libtokenizers_cpp.a"
    echo "   该文件已预编译入库，请确认 git clone 完整"
    exit 1
fi

cmake -S "${SCRIPT_DIR}" \
      -B "${BUILD_DIR}" \
      -DCMAKE_BUILD_TYPE=Release

cmake --build "${BUILD_DIR}" -j"$(nproc)"

if [[ "${1}" == "install" ]]; then
    mkdir -p "${DIST_DIR}/lib"
    cp "${BUILD_DIR}/qwen_demo" "${DIST_DIR}/"
    SOPHON_SDK="${SCRIPT_DIR}/../../0_Toolkits/soc-sdk-sp4"
    cp "${SOPHON_SDK}/lib/libbmrt.so"   "${DIST_DIR}/lib/"
    cp "${SOPHON_SDK}/lib/libbmrt.so.1.0" "${DIST_DIR}/lib/" 2>/dev/null || true
    cp "${SOPHON_SDK}/lib/libbmlib.so"  "${DIST_DIR}/lib/"
    cp "${SOPHON_SDK}/lib/libbmlib.so.0" "${DIST_DIR}/lib/" 2>/dev/null || true
    echo ""
    echo "✅ 安装完成: ${DIST_DIR}"
    echo "   上板部署: scp -r ${DIST_DIR}/* root@<board_ip>:/data/qwen_demo/"
fi

echo ""
echo "✅ 编译完成: ${BUILD_DIR}/qwen_demo"
