#!/bin/bash
# 部署 Qwen3-ASR-0.6B 到 BM1684X 板卡（单文件 bmodel + C++ 推理）并测试
# 用法: bash deploy_to_board.sh [--test]
set -e

BOARD_IP=172.16.25.248
BOARD_USER=root
BOARD_PASS=1
BOARD_DIR=/data/qwen3_asr

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCP="sshpass -p ${BOARD_PASS} scp -o StrictHostKeyChecking=no"
SSH="sshpass -p ${BOARD_PASS} ssh -o StrictHostKeyChecking=no ${BOARD_USER}@${BOARD_IP}"

echo "=== Creating remote directory ==="
${SSH} "mkdir -p ${BOARD_DIR}/models/BM1684X ${BOARD_DIR}/test_data"

echo "=== Uploading bmodels ==="
${SCP} ${REPO_ROOT}/models/BM1684X/qwen3_asr_merged_w4g64.bmodel \
       ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/models/BM1684X/
echo "=== Uploading model resources ==="
${SCP} ${REPO_ROOT}/models/prefix_ids.txt ${REPO_ROOT}/models/suffix_ids.txt \
       ${REPO_ROOT}/models/mel_filters.npz ${REPO_ROOT}/models/tokenizer.json \
       ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/

echo "=== Uploading C++ binary (if built) ==="
if [ -f ${REPO_ROOT}/cpp/build/qwen3_asr_bm1684x ]; then
    ${SCP} ${REPO_ROOT}/cpp/build/qwen3_asr_bm1684x ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/
    echo "  C++ binary uploaded"
else
    echo "  (skip: run bash cpp/build.sh first)"
fi

echo "=== Uploading test audio ==="
${SCP} ${REPO_ROOT}/test_data/*.wav ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/test_data/

echo ""
echo "=== Upload done. Files on board: ==="
${SSH} "ls -lh ${BOARD_DIR}/ && ls -lh ${BOARD_DIR}/models/BM1684X/"

if [[ "$1" == "--test" ]]; then
    echo ""
    echo "=== Running test on board (all test audios, offline) ==="
    ${SSH} "cd ${BOARD_DIR} && timeout 400 ./qwen3_asr_bm1684x \
        --bmodel models/BM1684X/qwen3_asr_merged_w4g64.bmodel \
        --model_dir . --audio_dir test_data"
fi

echo ""
echo "Done! To run manually on board:"
echo "  ssh root@${BOARD_IP}  (pass: 1)"
echo "  cd /data/qwen3_asr"
echo "  ./qwen3_asr_bm1684x --bmodel models/BM1684X/qwen3_asr_merged_w4g64.bmodel \\"
echo "      --model_dir . --audio test_data/test_zh.wav"
echo "  # 流式：加 --stream"
