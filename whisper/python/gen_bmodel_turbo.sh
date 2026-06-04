#!/bin/bash
# Whisper large-v3-turbo bmodel 转换（F16）
# turbo 维度: n_state=1280, encoder 32层, decoder 4层, n_head=20, mel=128, n_audio_ctx=1500, n_text_ctx=448
# 在 sophon-tpumlir 容器内执行：
#   docker exec sophon-tpumlir bash /workspace/whisper/python/gen_bmodel_turbo.sh [F16|F32]
set -e

MODEL_NAME="large-v3-turbo"
ONNX_DIR="/workspace/whisper/models/onnx"
WORK_DIR="/tmp/whisper_turbo_compile"
BMODEL_DIR="/workspace/whisper/models/BM1684X"
chip="bm1684x"
quantize="${1:-F16}"

# turbo 维度
n_mels=128
n_state=1280
n_audio_ctx=1500
padding_size=448
dec_layers=4   # turbo decoder 只有 4 层

mkdir -p "${BMODEL_DIR}" "${WORK_DIR}"
echo "=== Whisper ${MODEL_NAME} ${quantize} (n_state=${n_state}, dec_layers=${dec_layers}) ==="

# W4F16/W4BF16 是 per-group 权重量化，需指定 group size（与 base 的 gen_bmodel_w4f16.sh 一致用 64）
QG=""
case "${quantize}" in
    W4F16|W4BF16) QG="--q_group_size 64" ;;
esac

# 1. Encoder：输入 mel [1,128,3000]
echo "[1/2] Encoder (32 层，较慢)..."
cd "${WORK_DIR}"; rm -rf encoder && mkdir encoder && cd encoder
model_transform.py \
    --model_name "whisper_turbo_encoder" \
    --model_def "${ONNX_DIR}/whisper_${MODEL_NAME}_encoder_sim.onnx" \
    --input_shapes "[[1,${n_mels},3000]]" \
    --mlir "enc.mlir"
# turbo encoder 1.3G(>500MB) 属大模型：必须加 --disable_layer_group，
# 否则 TPU-MLIR v1.28.1 的 layer_group 优化在大模型上有 bug，推理时 kernel panic 重启板卡。
# （base encoder 仅 46M，小模型不触发，故 gen_bmodel.sh 里 base encoder 没加。）
model_deploy.py \
    --mlir "enc.mlir" \
    --quantize ${quantize} \
    ${QG} \
    --chip ${chip} \
    --disable_layer_group \
    --model "${BMODEL_DIR}/whisper_turbo_encoder_${quantize}.bmodel"
echo "[1/2] Encoder done"

# 2. Decoder：4 层 KV
echo "[2/2] Decoder (4 层)..."
cd "${WORK_DIR}"; rm -rf decoder && mkdir decoder && cd decoder
SELF_KV=""; CROSS_KV=""
for ((i=0; i<dec_layers; i++)); do
    SELF_KV="${SELF_KV},[1,${padding_size},${n_state}]"
    CROSS_KV="${CROSS_KV},[1,${n_audio_ctx},${n_state}]"
done
# 输入顺序: token, audio_features, pos_emb, mask, past_self_k×N, past_self_v×N, cross_k×N, cross_v×N
INPUT_SHAPES="[[1,1],[1,${n_audio_ctx},${n_state}],[1,1,${n_state}],[1,1,1,$((padding_size+1))]${SELF_KV}${SELF_KV}${CROSS_KV}${CROSS_KV}]"
model_transform.py \
    --model_name "whisper_turbo_decoder" \
    --model_def "${ONNX_DIR}/whisper_${MODEL_NAME}_decoder_sim.onnx" \
    --input_shapes "${INPUT_SHAPES}" \
    --mlir "dec.mlir"
model_deploy.py \
    --mlir "dec.mlir" \
    --quantize ${quantize} \
    ${QG} \
    --chip ${chip} \
    --disable_layer_group \
    --model "${BMODEL_DIR}/whisper_turbo_decoder_${quantize}.bmodel"
echo "[2/2] Decoder done"

ls -lh "${BMODEL_DIR}/whisper_turbo_"*"_${quantize}.bmodel"
echo "=== turbo 转换完成 ==="
