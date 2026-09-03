#!/bin/bash
# Whisper bmodel 转换脚本（支持 F32 / F16）
# 在 TPU-MLIR Docker 容器内执行（从 whisper/ 目录）:
#   docker run --rm \
#     -v $(pwd):/workspace \
#     -v $(pwd)/../../0_Toolkits:/toolkits \
#     sophgo/tpuc_dev:latest bash /workspace/python/gen_bmodel.sh [F32|F16]

set -e

# 使用公共镜像已安装的 TPU-MLIR；仅在工具不存在时从挂载 wheel 补装。
if ! command -v model_transform.py >/dev/null 2>&1 || ! command -v model_deploy.py >/dev/null 2>&1; then
    WHL=$(find /toolkits -maxdepth 1 -type f -name 'tpu_mlir*.whl' -print -quit 2>/dev/null)
    if [ -z "$WHL" ]; then
        echo "[Error] model_transform.py/model_deploy.py 不存在，且 /toolkits 无 tpu_mlir wheel"
        exit 1
    fi
    python -m pip install "$WHL" -q --no-deps
fi

MODEL_ROOT="/workspace/whisper"
MODEL_NAME="base"
ONNX_DIR="${MODEL_ROOT}/models/onnx"
WORK_DIR="/tmp/whisper_compile"

chip="bm1684x"
quantize="${1:-F16}"   # 默认 F16（与 deploy_to_board.sh 默认一致），可传 F32

if [ "${quantize}" != "F32" ] && [ "${quantize}" != "F16" ]; then
    echo "[Error] quantize 参数只支持 F32 或 F16，收到: ${quantize}"
    exit 1
fi

BMODEL_DIR="${MODEL_ROOT}/models/BM1684X"
OUTPUT_DIR="${WORK_DIR}/output"
mkdir -p "${BMODEL_DIR}" "${WORK_DIR}"
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

n_state=512
n_audio_ctx=1500
padding_size=448

echo "================================================================"
echo "  Whisper BM1684X bmodel 转换"
echo "  Model: whisper-${MODEL_NAME}  Chip: ${chip}  Quant: ${quantize}"
echo "================================================================"

# ----------------------------------------------------------------
# 1. Encoder
# ----------------------------------------------------------------
echo ""
echo "[1/2] 转换 Encoder..."
cd "${WORK_DIR}"
rm -rf encoder && mkdir encoder && cd encoder

model_transform.py \
    --model_name "whisper_${MODEL_NAME}_encoder" \
    --model_def "${ONNX_DIR}/whisper_${MODEL_NAME}_encoder_sim.onnx" \
    --input_shapes "[[1,80,3000]]" \
    --mlir "whisper_${MODEL_NAME}_encoder.mlir"

model_deploy.py \
    --mlir "whisper_${MODEL_NAME}_encoder.mlir" \
    --quantize ${quantize} \
    --chip ${chip} \
    --model "${OUTPUT_DIR}/whisper_${MODEL_NAME}_encoder_${quantize}.bmodel"

mv "${OUTPUT_DIR}/whisper_${MODEL_NAME}_encoder_${quantize}.bmodel" \
   "${BMODEL_DIR}/whisper_${MODEL_NAME}_encoder_${quantize}.bmodel"

echo "[1/2] Encoder 转换完成"

# ----------------------------------------------------------------
# 2. Decoder
# ----------------------------------------------------------------
echo ""
echo "[2/2] 转换 Decoder..."
cd "${WORK_DIR}"
rm -rf decoder && mkdir decoder && cd decoder

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
    --model "${OUTPUT_DIR}/whisper_${MODEL_NAME}_decoder_${quantize}.bmodel"

mv "${OUTPUT_DIR}/whisper_${MODEL_NAME}_decoder_${quantize}.bmodel" \
   "${BMODEL_DIR}/whisper_${MODEL_NAME}_decoder_${quantize}.bmodel"

echo "[2/2] Decoder 转换完成"

# ----------------------------------------------------------------
echo ""
echo "================================================================"
echo "  转换完成！bmodel 文件:"
ls -lh "${BMODEL_DIR}/whisper_${MODEL_NAME}_"*"_${quantize}.bmodel"
echo "================================================================"
