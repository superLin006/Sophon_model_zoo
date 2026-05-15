#!/bin/bash
# Deploy VITS-MeloTTS three-stage TPU inference to BM1684X board
# Usage: bash vits-melo-tts-zh_en/deploy_to_board.sh [F32|F16]
#
# Default precision: F32 (safer for first run); use F16 for faster inference

set -e

BOARD_IP=172.16.25.195
BOARD_PORT=26666
BOARD_USER=linaro
BOARD_PASS=linaro
PREC=${1:-F32}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_DIR="/home/linaro/vits_tts"

SSH="sshpass -p ${BOARD_PASS} ssh -p ${BOARD_PORT} -o StrictHostKeyChecking=no ${BOARD_USER}@${BOARD_IP}"
SCP="sshpass -p ${BOARD_PASS} scp -P ${BOARD_PORT} -o StrictHostKeyChecking=no"

echo "================================================================"
echo "  Deploy VITS-MeloTTS to ${BOARD_IP}:${BOARD_PORT}  [${PREC}]"
echo "================================================================"

echo "[1/4] Creating remote directories..."
$SSH "mkdir -p ${REMOTE_DIR}/models"

echo "[2/4] Uploading binary..."
$SCP "${REPO_ROOT}/vits-melo-tts-zh_en/cpp/build/vits_melo_tts_bm1684" \
     "${BOARD_USER}@${BOARD_IP}:${REMOTE_DIR}/"
$SSH "chmod +x ${REMOTE_DIR}/vits_melo_tts_bm1684"

echo "[3/4] Uploading bmodels (${PREC})..."
$SCP "${REPO_ROOT}/vits-melo-tts-zh_en/models/BM1684X/vits_part_a_${PREC}.bmodel" \
     "${BOARD_USER}@${BOARD_IP}:${REMOTE_DIR}/models/"
$SCP "${REPO_ROOT}/vits-melo-tts-zh_en/models/BM1684X/vits_part_c_${PREC}.bmodel" \
     "${BOARD_USER}@${BOARD_IP}:${REMOTE_DIR}/models/"

echo "[4/4] Uploading test data..."
$SCP "${REPO_ROOT}/vits-melo-tts-zh_en/test_data/test_zh_tokens.bin" \
     "${BOARD_USER}@${BOARD_IP}:${REMOTE_DIR}/"
$SCP "${REPO_ROOT}/vits-melo-tts-zh_en/test_data/test_zh_tones.bin" \
     "${BOARD_USER}@${BOARD_IP}:${REMOTE_DIR}/"
$SCP "${REPO_ROOT}/vits-melo-tts-zh_en/test_data/test_en_zh_tokens.bin" \
     "${BOARD_USER}@${BOARD_IP}:${REMOTE_DIR}/"
$SCP "${REPO_ROOT}/vits-melo-tts-zh_en/test_data/test_en_zh_tones.bin" \
     "${BOARD_USER}@${BOARD_IP}:${REMOTE_DIR}/"

echo ""
echo "================================================================"
echo "  Deployment complete. Test commands on board:"
echo ""
echo "  cd ${REMOTE_DIR}"
echo "  # Chinese test (seq_len=61):"
echo "  ./vits_melo_tts_bm1684 test_zh_tokens.bin test_zh_tones.bin 61 models output_zh_${PREC}.wav ${PREC}"
echo ""
echo "  # Chinese+English test (seq_len=53):"
echo "  ./vits_melo_tts_bm1684 test_en_zh_tokens.bin test_en_zh_tones.bin 53 models output_en_zh_${PREC}.wav ${PREC}"
echo "================================================================"
