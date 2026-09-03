#!/bin/bash
# Moonshine streaming-small bmodel 转换脚本（支持 F32 / F16）
# 在 TPU-MLIR Docker 容器内执行，/workspace = 仓库根（由 3_docker/run_docker.sh 挂载）:
#   docker exec sophon-tpumlir-v128 bash /workspace/moonshine/python/gen_bmodel.sh [F32|F16]
#
# 注意：下方 MODEL_ROOT 写死 /workspace/moonshine，因此必须让 /workspace 对应仓库根。
# 早期文档里的私有挂载（-v $(pwd)/moonshine:/workspace）与此矛盾——那样 MODEL_ROOT 会
# 解析成 moonshine/moonshine 而不存在，必然失败。

set -e

# 使用公共镜像已安装的 TPU-MLIR；仅在工具不存在时从仓库内 wheel 补装。
if ! command -v model_transform.py >/dev/null 2>&1 || ! command -v model_deploy.py >/dev/null 2>&1; then
    WHL=$(find /workspace/0_Toolkits -maxdepth 1 -type f -name 'tpu_mlir*.whl' -print -quit 2>/dev/null)
    if [ -z "$WHL" ]; then
        echo "[Error] model_transform.py/model_deploy.py 不存在，且 /workspace/0_Toolkits 无 tpu_mlir wheel"
        exit 1
    fi
    python -m pip install "$WHL" -q --no-deps
fi

# 从仓库根目录挂载到 /workspace 时的模型路径。
MODEL_ROOT="/workspace/moonshine"
MODEL_NAME="moonshine"
ONNX_DIR="${MODEL_ROOT}/models/onnx"
WORK_DIR="/tmp/moonshine_compile"
OUTPUT_DIR="${WORK_DIR}/output"

chip="bm1684x"
quantize="${1:-F16}"   # 默认 F16（与 deploy_to_board.sh 默认一致），可传 F32
if [ "${quantize}" != "F32" ] && [ "${quantize}" != "F16" ]; then
    echo "[Error] quantize 参数只支持 F32 或 F16，收到: ${quantize}"
    exit 1
fi

BMODEL_DIR="${MODEL_ROOT}/models/BM1684X"
mkdir -p "${BMODEL_DIR}" "${WORK_DIR}"
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

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
    --model "${OUTPUT_DIR}/${MODEL_NAME}_encoder_${quantize}.bmodel"

mv "${OUTPUT_DIR}/${MODEL_NAME}_encoder_${quantize}.bmodel" \
   "${BMODEL_DIR}/${MODEL_NAME}_encoder_${quantize}.bmodel"

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
    --model "${OUTPUT_DIR}/${MODEL_NAME}_decoder_${quantize}.bmodel"

mv "${OUTPUT_DIR}/${MODEL_NAME}_decoder_${quantize}.bmodel" \
   "${BMODEL_DIR}/${MODEL_NAME}_decoder_${quantize}.bmodel"

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
