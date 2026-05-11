#!/bin/bash
# Whisper BM1684 bmodel 转换脚本
# 在 TPU-MLIR Docker 容器内执行（从 whisper/ 目录）:
#   docker run --rm \
#     -v $(pwd):/workspace \
#     -v $(pwd)/../../0_Toolkits:/toolkits \
#     sophgo/tpuc_dev:latest bash /workspace/python/gen_bmodel.sh

set -e

# 安装 tpu_mlir（若尚未安装）
if ! python3 -c "import tpu_mlir" 2>/dev/null; then
    WHL=$(ls /toolkits/tpu_mlir*.whl 2>/dev/null | head -1)
    if [ -z "$WHL" ]; then
        echo "[Error] tpu_mlir whl not found in /toolkits/"
        exit 1
    fi
    echo "[Setup] 安装 tpu_mlir: $WHL"
    pip install "$WHL" -q
fi

MODEL_NAME="base"
ONNX_DIR="/workspace/models/onnx"
BMODEL_DIR="/workspace/models/BM1684X"
WORK_DIR="/tmp/whisper_compile"

mkdir -p "${BMODEL_DIR}" "${WORK_DIR}"

chip="bm1684x"
quantize="F32"
n_state=512      # whisper-base
n_audio_ctx=1500
padding_size=448

echo "================================================================"
echo "  Whisper BM1684 bmodel 转换"
echo "  Model: whisper-${MODEL_NAME}  Chip: ${chip}  Quant: ${quantize}"
echo "================================================================"

# ----------------------------------------------------------------
# 1. Encoder
# ----------------------------------------------------------------
echo ""
echo "[1/2] 转换 Encoder..."
cd "${WORK_DIR}"
mkdir -p encoder && cd encoder

model_transform.py \
    --model_name "whisper_${MODEL_NAME}_encoder" \
    --model_def "${ONNX_DIR}/whisper_${MODEL_NAME}_encoder_sim.onnx" \
    --input_shapes "[[1,80,3000]]" \
    --mlir "whisper_${MODEL_NAME}_encoder.mlir"

model_deploy.py \
    --mlir "whisper_${MODEL_NAME}_encoder.mlir" \
    --quantize ${quantize} \
    --chip ${chip} \
    --model "${BMODEL_DIR}/whisper_${MODEL_NAME}_encoder_${quantize}.bmodel"

echo "[1/2] Encoder 转换完成"

# ----------------------------------------------------------------
# 2. Decoder（单步，含 embedding/mask，与 MTK 方案对齐）
# ----------------------------------------------------------------
echo ""
echo "[2/2] 转换 Decoder..."
cd "${WORK_DIR}"
mkdir -p decoder && cd decoder

# 输入顺序: token, audio_features, pos_emb, self_attn_mask,
#           past_self_k_0..5, past_self_v_0..5, cross_k_0..5, cross_v_0..5
# past_self_k/v: [1, padding_size, n_state] = [1, 448, 512]
# cross_k/v:     [1, n_audio_ctx, n_state]  = [1, 1500, 512]
SELF_KV=""
CROSS_KV=""
for i in 0 1 2 3 4 5; do
    SELF_KV="${SELF_KV},[1,${padding_size},${n_state}]"
    CROSS_KV="${CROSS_KV},[1,${n_audio_ctx},${n_state}]"
done

INPUT_SHAPES="[[1,1],[1,${n_audio_ctx},${n_state}],[1,1,${n_state}],[1,1,1,$((padding_size+1))]${SELF_KV}${SELF_KV}${CROSS_KV}${CROSS_KV}]"

model_transform.py \
    --model_name "whisper_${MODEL_NAME}_decoder" \
    --model_def "${ONNX_DIR}/whisper_${MODEL_NAME}_decoder_sim.onnx" \
    --input_shapes "${INPUT_SHAPES}" \
    --mlir "whisper_${MODEL_NAME}_decoder.mlir"

model_deploy.py \
    --mlir "whisper_${MODEL_NAME}_decoder.mlir" \
    --quantize ${quantize} \
    --chip ${chip} \
    --disable_layer_group \
    --model "${BMODEL_DIR}/whisper_${MODEL_NAME}_decoder_${quantize}.bmodel"

echo "[2/2] Decoder 转换完成"

# ----------------------------------------------------------------
# 结果
# ----------------------------------------------------------------
echo ""
echo "================================================================"
echo "  转换完成！bmodel 文件:"
ls -lh "${BMODEL_DIR}/"*.bmodel
echo ""
echo "  下一步: scp 到 BM1684 服务器 (172.16.40.75)"
echo "    scp ${BMODEL_DIR}/*.bmodel root@172.16.40.75:/root/whisper/models/"
echo "================================================================"
