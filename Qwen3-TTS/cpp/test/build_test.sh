#!/bin/bash
# 交叉编译 bm_test_cp（板卡单测工具）
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

docker run --rm -v "${REPO_ROOT}:/repo" sophon-cross-build bash -c '
set -e
cd /repo/Qwen3-TTS/cpp/test
aarch64-linux-gnu-g++ -O2 -std=c++17 -I/repo/0_Toolkits/soc-sdk-sp4/include \
    bm_test_cp.cpp -o bm_test_cp \
    -L/repo/0_Toolkits/soc-sdk-sp4/lib -lbmrt -lbmlib -lpthread
ls -lh bm_test_cp
'
echo "done"
