#!/bin/bash
# 编译 VITS-MeloTTS Part A + Part C bmodel（三段式方案）
#
# 在 TPU-MLIR Docker 内从仓库根目录执行：
#   docker run --rm \
#     -v $(pwd):/repo \
#     -v $(pwd)/0_Toolkits:/repo/0_Toolkits \
#     sophgo/tpuc_dev:latest \
#     bash /repo/vits-melo-tts-zh_en/python/gen_bmodel.sh [F32|F16]
#
# 前提：已运行 make_tpu_model.py 和 make_split_models.py 生成子图 ONNX
#
# 注意: T_MEL_FIXED=256 是 BM1684X 的硬上限，更大值会导致 TPU DDR 溢出崩溃

set -e

QUANTIZE="${1:-F32}"
ONNX_DIR="/repo/vits-melo-tts-zh_en/models/onnx/vits-melo-tts-zh_en"
BMODEL_DIR="/repo/vits-melo-tts-zh_en/models/BM1684X"
WORK_DIR="/tmp/vits_compile"
CHIP="bm1684x"
L_MAX=128
Z_DIM=192
T_MEL_FIXED=256

if [ "${QUANTIZE}" != "F32" ] && [ "${QUANTIZE}" != "F16" ]; then
    echo "[Error] 只支持 F32 或 F16，收到: ${QUANTIZE}"
    exit 1
fi

# 安装 tpu_mlir
WHL=$(ls /repo/0_Toolkits/tpu_mlir*.whl 2>/dev/null | head -1)
if [ -z "$WHL" ]; then
    echo "[Error] tpu_mlir whl not found in 0_Toolkits/"
    exit 1
fi
pip install "$WHL" -q --no-deps 2>/dev/null || pip install "$WHL" -q

mkdir -p "${BMODEL_DIR}" "${WORK_DIR}"

echo "================================================================"
echo "  VITS-MeloTTS BM1684X bmodel 编译  [${QUANTIZE}]  T_MEL=${T_MEL_FIXED}"
echo "================================================================"

# ── Part A: enc_p + DP ──────────────────────────────────────────────
echo ""
echo "[Part A] 编译 enc_p + DP ..."
mkdir -p "${WORK_DIR}/part_a" && cd "${WORK_DIR}/part_a"

model_transform.py \
    --model_name vits_part_a \
    --model_def "${ONNX_DIR}/part_a_encoder.onnx" \
    --input_shapes "[[1,${L_MAX}],[1],[1,${L_MAX}]]" \
    --mlir vits_part_a.mlir

model_deploy.py \
    --mlir vits_part_a.mlir \
    --quantize ${QUANTIZE} \
    --chip ${CHIP} \
    --model "${BMODEL_DIR}/vits_part_a_${QUANTIZE}.bmodel"

echo "[Part A] 完成：$(ls -lh ${BMODEL_DIR}/vits_part_a_${QUANTIZE}.bmodel | awk '{print $5, $9}')"

# ── Part C: Flow + Decoder ──────────────────────────────────────────
echo ""
echo "[Part C] 编译 Flow + Decoder ..."
mkdir -p "${WORK_DIR}/part_c" && cd "${WORK_DIR}/part_c"

model_transform.py \
    --model_name vits_part_c \
    --model_def "${ONNX_DIR}/part_c_flow_decoder.onnx" \
    --input_shapes "[[1,${Z_DIM},${T_MEL_FIXED}],[1,1,${T_MEL_FIXED}]]" \
    --mlir vits_part_c.mlir

model_deploy.py \
    --mlir vits_part_c.mlir \
    --quantize ${QUANTIZE} \
    --chip ${CHIP} \
    --model "${BMODEL_DIR}/vits_part_c_${QUANTIZE}.bmodel"

echo "[Part C] 完成：$(ls -lh ${BMODEL_DIR}/vits_part_c_${QUANTIZE}.bmodel | awk '{print $5, $9}')"

echo ""
echo "================================================================"
echo "  全部完成 [${QUANTIZE}]"
echo "  Part A: ${BMODEL_DIR}/vits_part_a_${QUANTIZE}.bmodel"
echo "  Part C: ${BMODEL_DIR}/vits_part_c_${QUANTIZE}.bmodel"
echo "================================================================"
