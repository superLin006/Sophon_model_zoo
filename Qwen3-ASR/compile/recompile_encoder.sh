#!/bin/bash
# 编译 Qwen3-ASR Audio Encoder（含 projector）为 bmodel
# **必加 --disable_layer_group**（whisper 先例：大模型不加会 TPU kernel panic）
set -e

ONNX_DIR="./tmp/onnx"  # 导出: python export_encoder_onnx.py（离线 3000 帧）/ --num_mel_frames 500 --out_dir ./tmp/onnx_enc_w500（流式窗口）
TMP_DIR="./tmp/encoder_recompile"
OUT_DIR="../models/BM1684X"

mkdir -p "${TMP_DIR}" "${OUT_DIR}"

echo "=== Compiling qwen3_asr_encoder (F16, --disable_layer_group) ==="
model_transform.py \
  --model_name qwen3_asr_encoder \
  --model_def "${ONNX_DIR}/qwen3_asr_encoder.onnx" \
  --input_shapes [[1,128,3000]] \
  --mlir "${TMP_DIR}/qwen3_asr_encoder.mlir"

model_deploy.py \
  --mlir "${TMP_DIR}/qwen3_asr_encoder.mlir" \
  --quantize F16 \
  --chip bm1684x \
  --disable_layer_group \
  --model "${TMP_DIR}/qwen3_asr_encoder.bmodel"

echo "=== Merging ==="
model_tool --combine \
  "${TMP_DIR}/qwen3_asr_encoder.bmodel" \
  -o "${OUT_DIR}/qwen3_asr_encoder_F16.bmodel"

echo ""
echo "Done: ${OUT_DIR}/qwen3_asr_encoder_F16.bmodel"
