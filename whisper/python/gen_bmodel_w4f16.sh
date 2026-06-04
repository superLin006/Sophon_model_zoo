#!/bin/bash
# Whisper base W4F16 量化试验（权重 int4 + 激活 F16）
# 在 sophon-tpumlir 容器内执行：
#   docker exec sophon-tpumlir bash /workspace/whisper/python/gen_bmodel_w4f16.sh
set -e

MODEL_NAME="base"
ONNX_DIR="/workspace/whisper/models/onnx"
WORK_DIR="/tmp/whisper_w4f16"
BMODEL_DIR="/workspace/whisper/models/BM1684X"
chip="bm1684x"
quantize="W4F16"

n_state=512
n_audio_ctx=1500
padding_size=448

mkdir -p "${BMODEL_DIR}" "${WORK_DIR}"
echo "=== Whisper base ${quantize} ==="

# 1. Encoder
echo "[1/2] Encoder..."
cd "${WORK_DIR}"; rm -rf encoder && mkdir encoder && cd encoder
model_transform.py \
    --model_name "whisper_${MODEL_NAME}_encoder" \
    --model_def "${ONNX_DIR}/whisper_${MODEL_NAME}_encoder_sim.onnx" \
    --input_shapes "[[1,80,3000]]" \
    --mlir "enc.mlir"
model_deploy.py \
    --mlir "enc.mlir" \
    --quantize ${quantize} \
    --q_group_size 64 \
    --chip ${chip} \
    --model "${BMODEL_DIR}/whisper_${MODEL_NAME}_encoder_${quantize}.bmodel"
echo "[1/2] Encoder done"

# 2. Decoder
echo "[2/2] Decoder..."
cd "${WORK_DIR}"; rm -rf decoder && mkdir decoder && cd decoder
SELF_KV=""; CROSS_KV=""
for i in 0 1 2 3 4 5; do
    SELF_KV="${SELF_KV},[1,${padding_size},${n_state}]"
    CROSS_KV="${CROSS_KV},[1,${n_audio_ctx},${n_state}]"
done
INPUT_SHAPES="[[1,1],[1,${n_audio_ctx},${n_state}],[1,1,${n_state}],[1,1,1,$((padding_size+1))]${SELF_KV}${SELF_KV}${CROSS_KV}${CROSS_KV}]"
model_transform.py \
    --model_name "whisper_${MODEL_NAME}_decoder" \
    --model_def "${ONNX_DIR}/whisper_${MODEL_NAME}_decoder_sim.onnx" \
    --input_shapes "${INPUT_SHAPES}" \
    --mlir "dec.mlir"
model_deploy.py \
    --mlir "dec.mlir" \
    --quantize ${quantize} \
    --q_group_size 64 \
    --chip ${chip} \
    --disable_layer_group \
    --model "${BMODEL_DIR}/whisper_${MODEL_NAME}_decoder_${quantize}.bmodel"
echo "[2/2] Decoder done"

ls -lh "${BMODEL_DIR}/whisper_${MODEL_NAME}_"*"_${quantize}.bmodel"
echo "=== W4F16 转换完成 ==="
