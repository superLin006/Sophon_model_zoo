#!/bin/bash
# 交叉编译 qwen3_tts_bm1684x（aarch64，sophon-cross-build 容器）
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

docker run --rm -v "${REPO_ROOT}:/repo" sophon-cross-build bash -c '
set -e
rm -rf /repo/Qwen3-TTS/cpp/build
mkdir -p /repo/Qwen3-TTS/cpp/build
cd /repo/Qwen3-TTS/cpp/build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
    -DSOPHON_SDK=/repo/0_Toolkits/soc-sdk-sp4 \
    -DTOKENIZERS_DIR=/repo/1_third_party/tokenizers-cpp
make -j$(nproc)
ls -lh /repo/Qwen3-TTS/cpp/build/qwen3_tts_bm1684x
'
echo "产物: ${REPO_ROOT}/Qwen3-TTS/cpp/build/qwen3_tts_bm1684x"
