#!/bin/bash
# Moonshine streaming-small bmodel 转换脚本（支持 F32 / F16）
# 在 TPU-MLIR Docker 容器内执行（从 Sophon_model_zoo/ 根目录）:
#   docker run --rm \
#     -v $(pwd)/moonshine:/workspace \
#     -v $(pwd)/0_Toolkits:/toolkits \
#     sophgo/tpuc_dev:latest bash /workspace/python/gen_bmodel.sh [F32|F16]

set -e

# 安装 tpu_mlir（强制从本地 whl 安装，避免走网络）
WHL=$(ls /toolkits/tpu_mlir*.whl 2>/dev/null | head -1)
if [ -z "$WHL" ]; then
    echo "[Error] tpu_mlir whl not found in /toolkits/"
    exit 1
fi
pip install "$WHL" -q --no-deps 2>/dev/null || pip install "$WHL" -q

MODEL_NAME="moonshine"
ONNX_DIR="/workspace/models/onnx"
WORK_DIR="/tmp/moonshine_compile"

chip="bm1684x"
quantize="${1:-F32}"   # 默认 F32，可传 F16
if [ "${quantize}" != "F32" ] && [ "${quantize}" != "F16" ]; then
    echo "[Error] quantize 参数只支持 F32 或 F16，收到: ${quantize}"
    exit 1
fi

BMODEL_DIR="/workspace/models/BM1684X"
mkdir -p "${BMODEL_DIR}" "${WORK_DIR}"

# 固定 shape: 10s 音频 -> T=2000 帧 -> T_enc=500
T=2000
T_ENC=500
MAX_DEC_LEN=128
HID_DEC=512
N_LAYER=10

echo "================================================================"
echo "  Moonshine BM1684X bmodel 转换"
echo "  Model: streaming-small  Chip: ${chip}  Quant: ${quantize}"
echo "================================================================"

# ----------------------------------------------------------------
# 1. Encoder（方案 B: 输入 x_frames [1,2000,80]，C++ 已做 CMVN+Asinh）
# ----------------------------------------------------------------
echo ""
echo "[1/2] 转换 Encoder..."
cd "${WORK_DIR}"
rm -rf encoder && mkdir encoder && cd encoder

model_transform.py \
    --model_name "${MODEL_NAME}_encoder" \
    --model_def "${ONNX_DIR}/moonshine_encoder_sim.onnx" \
    --input_shapes "[[1,${T},80]]" \
    --mlir "${MODEL_NAME}_encoder.mlir"

model_deploy.py \
    --mlir "${MODEL_NAME}_encoder.mlir" \
    --quantize ${quantize} \
    --chip ${chip} \
    --model "${BMODEL_DIR}/${MODEL_NAME}_encoder_${quantize}.bmodel"

echo "[1/2] Encoder 转换完成"

# ----------------------------------------------------------------
# 2. Decoder（单步: 23 输入 / 21 输出，必须 --disable_layer_group）
# 输入顺序与 ONNX 图一致:
#   token [1,1] int64, encoder_out [1,500,620] f32, cache_len [1] int64,
#   past_k_0..9 [1,128,512] f32, past_v_0..9 [1,128,512] f32
# ----------------------------------------------------------------
echo ""
echo "[2/2] 转换 Decoder..."
cd "${WORK_DIR}"
rm -rf decoder && mkdir decoder && cd decoder

KV_SHAPES=""
for i in $(seq 0 $((N_LAYER-1))); do
    KV_SHAPES="${KV_SHAPES},[1,${MAX_DEC_LEN},${HID_DEC}]"
done

# token + encoder_out + cache_len + past_k x10 + past_v x10
INPUT_SHAPES="[[1,1],[1,${T_ENC},620],[1]${KV_SHAPES}${KV_SHAPES}]"

model_transform.py \
    --model_name "${MODEL_NAME}_decoder" \
    --model_def "${ONNX_DIR}/moonshine_decoder_sim.onnx" \
    --input_shapes "${INPUT_SHAPES}" \
    --mlir "${MODEL_NAME}_decoder.mlir"

model_deploy.py \
    --mlir "${MODEL_NAME}_decoder.mlir" \
    --quantize ${quantize} \
    --chip ${chip} \
    --disable_layer_group \
    --model "${BMODEL_DIR}/${MODEL_NAME}_decoder_${quantize}.bmodel"

echo "[2/2] Decoder 转换完成"

# ----------------------------------------------------------------
# 3. bmrt_test 校验 IO 规格
# ----------------------------------------------------------------
echo ""
echo "[3] bmrt_test 校验..."
bmrt_test --bmodel "${BMODEL_DIR}/${MODEL_NAME}_encoder_${quantize}.bmodel" 2>&1 | \
    grep -E "input|output|shape|dtype" | head -10 || true
bmrt_test --bmodel "${BMODEL_DIR}/${MODEL_NAME}_decoder_${quantize}.bmodel" 2>&1 | \
    grep -E "input|output|shape|dtype" | head -60 || true

echo ""
echo "================================================================"
echo "  转换完成！bmodel 文件:"
ls -lh "${BMODEL_DIR}/${MODEL_NAME}_"*"_${quantize}.bmodel"
echo "================================================================"
