#!/bin/bash
# 部署 Qwen3-ASR-0.6B 到 BM1684X 板卡（python/sail 版）并测试
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
${SSH} "mkdir -p ${BOARD_DIR}/models/BM1684X ${BOARD_DIR}/python ${BOARD_DIR}/test_data"

echo "=== Uploading bmodels ==="
${SCP} ${REPO_ROOT}/models/BM1684X/qwen3_asr_llm_w4bf16_seq256_bm1684x.bmodel \
       ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/models/BM1684X/
${SCP} ${REPO_ROOT}/models/BM1684X/qwen3_asr_encoder_F16.bmodel \
       ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/models/BM1684X/
${SCP} ${REPO_ROOT}/models/BM1684X/qwen3_asr_encoder_w500_F16.bmodel \
       ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/models/BM1684X/ 2>/dev/null || echo "  (skip: stream encoder w500)"

echo "=== Uploading model resources ==="
${SCP} ${REPO_ROOT}/models/prefix_embeds.bin ${REPO_ROOT}/models/suffix_embeds.bin \
       ${REPO_ROOT}/models/mel_filters.npz ${REPO_ROOT}/models/tokenizer.json \
       ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/

echo "=== Uploading inference script ==="
${SCP} ${REPO_ROOT}/python/infer_board.py ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/python/

echo "=== Uploading C++ binary (if built) ==="
if [ -f ${REPO_ROOT}/cpp/build/qwen3_asr_bm1684x ]; then
    ${SCP} ${REPO_ROOT}/cpp/build/qwen3_asr_bm1684x ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/
    echo "  C++ binary uploaded"
else
    echo "  (skip: run bash cpp/build.sh first)"
fi

echo "=== Uploading test audio ==="
${SCP} ${REPO_ROOT}/test_data/test_zh.wav ${REPO_ROOT}/test_data/test_en.wav \
       ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/test_data/

echo ""
echo "=== Upload done. Files on board: ==="
${SSH} "ls -lh ${BOARD_DIR}/ && ls -lh ${BOARD_DIR}/models/BM1684X/"

if [[ "$1" == "--test" ]]; then
    echo ""
    echo "=== Running test on board ==="
    ${SSH} "cd ${BOARD_DIR}/python && python3 infer_board.py \
        --encoder models/BM1684X/qwen3_asr_encoder_F16.bmodel \
        --qwen3   models/BM1684X/qwen3_asr_llm_w4bf16_seq1024_bm1684x.bmodel \
        --model_dir .. --audio ../test_data/test_zh.wav"
fi

echo ""
echo "Done! To run manually on board:"
echo "  ssh root@${BOARD_IP}  (pass: 1)"
echo "  cd /data/qwen3_asr/python"
echo "  python3 infer_board.py --encoder ../models/BM1684X/qwen3_asr_encoder_F16.bmodel \\"
echo "      --qwen3 ../models/BM1684X/qwen3_asr_llm_w4bf16_seq1024_bm1684x.bmodel \\"
echo "      --model_dir .. --audio ../test_data/test_zh.wav"
