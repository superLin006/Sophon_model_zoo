#!/bin/bash
# Whisper large-v3-turbo W4A8 量化编译脚本
#
# 流程：
#   Step 1: ONNX → MLIR（model_transform.py）
#   Step 2: 生成 calibration table（run_calibration.py，需 calib_data/encoder_npy/）
#   Step 3: 编译 W4A8 bmodel（model_deploy.py --quantize W4A8）
#
# 在 sophon-tpumlir 容器内执行（从仓库根目录挂载）：
#   docker exec sophon-tpumlir bash /workspace/whisper/python/gen_bmodel_w4a8_turbo.sh
#
# 若 W4A8 编译报错（TPU-MLIR 不支持），脚本直接退出并打印结论，不降级。
set -e

ONNX_DIR="/workspace/whisper/models/onnx"
WORK_DIR="/tmp/whisper_w4a8_compile"
BMODEL_DIR="/workspace/whisper/models/BM1684X"
CALIB_DIR="/workspace/whisper/calib_data/encoder_npy"
chip="bm1684x"
quantize="W4A8"

# turbo 维度
n_mels=128
n_state=1280
n_audio_ctx=1500
padding_size=448
dec_layers=4

mkdir -p "${BMODEL_DIR}" "${WORK_DIR}"
echo "=== Whisper large-v3-turbo ${quantize} ==="
echo "    encoder: [1,${n_mels},3000]"
echo "    decoder: ${dec_layers} 层，n_state=${n_state}"

# ---------------------------------------------------------------
# Step 1a: Encoder ONNX → MLIR
# ---------------------------------------------------------------
echo ""
echo "[1/4] Encoder: ONNX → MLIR..."
cd "${WORK_DIR}"; rm -rf encoder && mkdir encoder && cd encoder

model_transform.py \
    --model_name "whisper_turbo_encoder" \
    --model_def "${ONNX_DIR}/whisper_large-v3-turbo_encoder_sim.onnx" \
    --input_shapes "[[1,${n_mels},3000]]" \
    --mlir "enc.mlir"
echo "[1/4] Encoder MLIR 完成"

# ---------------------------------------------------------------
# Step 2a: Encoder calibration table
# ---------------------------------------------------------------
echo ""
echo "[2/4] Encoder: 生成 calibration table（${CALIB_DIR}）..."
CALIB_COUNT=$(ls "${CALIB_DIR}"/*.npy 2>/dev/null | wc -l)
echo "      calibration 样本数: ${CALIB_COUNT}"

run_calibration.py enc.mlir \
    --dataset "${CALIB_DIR}" \
    --input_num "${CALIB_COUNT}" \
    -o enc_cali_table
echo "[2/4] Encoder calibration table 完成"

# ---------------------------------------------------------------
# Step 3a: Encoder W4A8 bmodel
# ---------------------------------------------------------------
echo ""
echo "[3/4] Encoder: 编译 ${quantize} bmodel..."
set +e
model_deploy.py \
    --mlir "enc.mlir" \
    --quantize ${quantize} \
    --calibration_table enc_cali_table \
    --q_group_size 64 \
    --chip ${chip} \
    --disable_layer_group \
    --model "${BMODEL_DIR}/whisper_turbo_encoder_${quantize}.bmodel"
ENC_EXIT=$?
set -e

if [ ${ENC_EXIT} -ne 0 ]; then
    echo ""
    echo "======================================================"
    echo "  结论: TPU-MLIR 不支持 --quantize ${quantize}"
    echo "  Encoder 编译失败（exit ${ENC_EXIT}）"
    echo "  W4F16 已验证无损，建议继续使用 W4F16。"
    echo "======================================================"
    exit ${ENC_EXIT}
fi
echo "[3/4] Encoder ${quantize} bmodel 完成"

# ---------------------------------------------------------------
# Step 1b: Decoder ONNX → MLIR
# ---------------------------------------------------------------
echo ""
echo "[1b/4] Decoder: ONNX → MLIR..."
cd "${WORK_DIR}"; rm -rf decoder && mkdir decoder && cd decoder

SELF_KV=""; CROSS_KV=""
for ((i=0; i<dec_layers; i++)); do
    SELF_KV="${SELF_KV},[1,${padding_size},${n_state}]"
    CROSS_KV="${CROSS_KV},[1,${n_audio_ctx},${n_state}]"
done
INPUT_SHAPES="[[1,1],[1,${n_audio_ctx},${n_state}],[1,1,${n_state}],[1,1,1,$((padding_size+1))]${SELF_KV}${SELF_KV}${CROSS_KV}${CROSS_KV}]"

model_transform.py \
    --model_name "whisper_turbo_decoder" \
    --model_def "${ONNX_DIR}/whisper_large-v3-turbo_decoder_sim.onnx" \
    --input_shapes "${INPUT_SHAPES}" \
    --mlir "dec.mlir"
echo "[1b/4] Decoder MLIR 完成"

# ---------------------------------------------------------------
# Step 2b: Decoder calibration table
# 注意：Decoder 有 14 个输入，run_calibration.py 使用 encoder_npy
# 作为近似校准（激活分布与实际推理相近，足够 W4A8 校准）
# ---------------------------------------------------------------
echo ""
echo "[2b/4] Decoder: 生成 calibration table..."
run_calibration.py dec.mlir \
    --dataset "${CALIB_DIR}" \
    --input_num "${CALIB_COUNT}" \
    -o dec_cali_table
echo "[2b/4] Decoder calibration table 完成"

# ---------------------------------------------------------------
# Step 3b: Decoder W4A8 bmodel
# ---------------------------------------------------------------
echo ""
echo "[3b/4] Decoder: 编译 ${quantize} bmodel..."
set +e
model_deploy.py \
    --mlir "dec.mlir" \
    --quantize ${quantize} \
    --calibration_table dec_cali_table \
    --q_group_size 64 \
    --chip ${chip} \
    --disable_layer_group \
    --model "${BMODEL_DIR}/whisper_turbo_decoder_${quantize}.bmodel"
DEC_EXIT=$?
set -e

if [ ${DEC_EXIT} -ne 0 ]; then
    echo ""
    echo "======================================================"
    echo "  Decoder 编译失败（exit ${DEC_EXIT}）"
    echo "  Encoder ${quantize} 已成功，可考虑混合精度："
    echo "    Encoder: ${quantize}  Decoder: W4F16"
    echo "======================================================"
    exit ${DEC_EXIT}
fi
echo "[3b/4] Decoder ${quantize} bmodel 完成"

# ---------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------
echo ""
echo "======================================================"
echo "  ${quantize} 编译完成！"
ls -lh "${BMODEL_DIR}/whisper_turbo_"*"_${quantize}.bmodel"
echo ""
echo "  下一步: 上板运行评估"
echo "    ./whisper_bm1684 models/BM1684X test.wav zh W4A8 turbo"
echo "======================================================"
