#!/bin/bash
# 部署 Eureka-Audio 到 BM1684X 板卡并测试
# 用法: bash deploy_to_board.sh [--test]

set -e

BOARD_IP=172.16.40.75
BOARD_USER=root
BOARD_PASS=1
BOARD_DIR=/data/eureka_audio

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCP="sshpass -p ${BOARD_PASS} scp -o StrictHostKeyChecking=no"
SSH="sshpass -p ${BOARD_PASS} ssh -o StrictHostKeyChecking=no ${BOARD_USER}@${BOARD_IP}"

echo "=== Creating remote directory ==="
${SSH} "mkdir -p ${BOARD_DIR}/models/BM1684X"

echo "=== Uploading binary ==="
${SCP} ${REPO_ROOT}/cpp/eureka_audio_bm1684x ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/

echo "=== Uploading bmodels ==="
${SCP} ${REPO_ROOT}/models/BM1684X/whisper_encoder_b1_bf16.bmodel \
       ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/models/BM1684X/
${SCP} ${REPO_ROOT}/models/BM1684X/qwen3_1.7b_embeds_w4bf16_seq512_bm1684x.bmodel \
       ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/models/BM1684X/

echo "=== Uploading model resources ==="
${SCP} ${REPO_ROOT}/../Eureka-Audio-Instruct/prefix_embeds.bin \
       ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/
${SCP} ${REPO_ROOT}/../Eureka-Audio-Instruct/mel_filters.npz \
       ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/

echo "=== Uploading test audio ==="
${SCP} ${REPO_ROOT}/test_audios/test_en.wav \
       ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/ 2>/dev/null || true
ls ${REPO_ROOT}/test_audios/ 2>/dev/null && \
  ${SCP} $(ls ${REPO_ROOT}/test_audios/*.wav 2>/dev/null | head -1) \
         ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/test.wav 2>/dev/null || true

echo ""
echo "=== Upload done. Files on board: ==="
${SSH} "ls -lh ${BOARD_DIR}/ && ls -lh ${BOARD_DIR}/models/BM1684X/"

if [[ "$1" == "--test" ]]; then
    echo ""
    echo "=== Running test on board ==="
    ${SSH} "cd ${BOARD_DIR} && chmod +x eureka_audio_bm1684x && \
        ./eureka_audio_bm1684x \
          --whisper_bmodel models/BM1684X/whisper_encoder_b1_bf16.bmodel \
          --qwen3_bmodel   models/BM1684X/qwen3_1.7b_embeds_w4bf16_seq512_bm1684x.bmodel \
          --model_dir      . \
          --audio          test.wav \
          --max_new_tokens 64"
fi

echo ""
echo "Done! To run manually on board:"
echo "  ssh ${BOARD_USER}@${BOARD_IP}"
echo "  cd ${BOARD_DIR}"
echo "  ./eureka_audio_bm1684x \\"
echo "      --whisper_bmodel models/BM1684X/whisper_encoder_b1_bf16.bmodel \\"
echo "      --qwen3_bmodel   models/BM1684X/qwen3_1.7b_embeds_w4bf16_seq512_bm1684x.bmodel \\"
echo "      --model_dir      . \\"
echo "      --audio          test.wav"
