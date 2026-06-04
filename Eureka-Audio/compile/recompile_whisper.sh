#!/bin/bash
# 重新编译 whisper_encoder_b1_bf16.bmodel，加 --disable_layer_group
# 防止 TPU kernel panic
set -e

ONNX_DIR="./tmp/onnx"
TMP_DIR="./tmp/whisper_recompile"
OUT_DIR="../models/BM1684X"

mkdir -p "${TMP_DIR}" "${OUT_DIR}"

echo "=== Compiling whisper_encoder (--disable_layer_group) ==="
model_transform.py \
  --model_name whisper_encoder \
  --model_def "${ONNX_DIR}/whisper_encoder_b1.onnx" \
  --input_shapes [[1,128,3000]] \
  --mlir "${TMP_DIR}/whisper_encoder.mlir"

model_deploy.py \
  --mlir "${TMP_DIR}/whisper_encoder.mlir" \
  --quantize F16 \
  --chip bm1684x \
  --disable_layer_group \
  --model "${TMP_DIR}/whisper_encoder.bmodel"

echo "=== Compiling audio_adaptor (--disable_layer_group) ==="
model_transform.py \
  --model_name audio_adaptor \
  --model_def "${ONNX_DIR}/audio_adaptor.onnx" \
  --input_shapes [[1500,1280]] \
  --mlir "${TMP_DIR}/audio_adaptor.mlir"

model_deploy.py \
  --mlir "${TMP_DIR}/audio_adaptor.mlir" \
  --quantize F16 \
  --chip bm1684x \
  --disable_layer_group \
  --model "${TMP_DIR}/audio_adaptor.bmodel"

echo "=== Merging ==="
model_tool --combine \
  "${TMP_DIR}/whisper_encoder.bmodel" \
  "${TMP_DIR}/audio_adaptor.bmodel" \
  -o "${OUT_DIR}/whisper_encoder_b1_bf16.bmodel"

echo ""
echo "Done: ${OUT_DIR}/whisper_encoder_b1_bf16.bmodel"
