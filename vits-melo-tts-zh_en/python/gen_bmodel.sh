#!/bin/bash
# 编译 VITS-MeloTTS Part A + Part C bmodel（三段式方案）
#
# 在 TPU-MLIR Docker 内从仓库根目录执行：
#   docker exec sophon-tpumlir-v128 bash /workspace/vits-melo-tts-zh_en/python/gen_bmodel.sh [F32|F16]
#
# 前提：已准备 models/onnx/vits-melo-tts-zh_en/ 下的三段 ONNX

# 注意: T_MEL_FIXED=1024 与 C++ 运行时及拆分 ONNX 的固定形状保持一致

set -e

QUANTIZE="${1:-F16}"   # 默认 F16（与 deploy_to_board.sh 默认一致），可传 F32
MODEL_ROOT="/workspace/vits-melo-tts-zh_en"
ONNX_DIR="${MODEL_ROOT}/models/onnx/vits-melo-tts-zh_en"
BMODEL_DIR="${MODEL_ROOT}/models/BM1684X"
WORK_DIR="${MODEL_ROOT}/compile/tmp/vits_${QUANTIZE}"
CHIP="bm1684x"
L_MAX=256
Z_DIM=192
T_MEL_FIXED=1024

PART_A_MODEL="${WORK_DIR}/vits_part_a_${QUANTIZE}.bmodel"
PART_C1_MODEL="${WORK_DIR}/vits_part_c1_${QUANTIZE}.bmodel"
PART_C2_MODEL="${WORK_DIR}/vits_part_c2_${QUANTIZE}.bmodel"
FINAL_PART_A_MODEL="${BMODEL_DIR}/vits_part_a_${QUANTIZE}.bmodel"
FINAL_PART_C1_MODEL="${BMODEL_DIR}/vits_part_c1_${QUANTIZE}.bmodel"
FINAL_PART_C2_MODEL="${BMODEL_DIR}/vits_part_c2_${QUANTIZE}.bmodel"

if [ "${QUANTIZE}" != "F32" ] && [ "${QUANTIZE}" != "F16" ]; then
    echo "[Error] 只支持 F32 或 F16，收到: ${QUANTIZE}"
    exit 1
fi

# 使用公共镜像已安装的 TPU-MLIR；仅在工具不存在时从挂载 wheel 补装。
if ! command -v model_transform.py >/dev/null 2>&1 || ! command -v model_deploy.py >/dev/null 2>&1; then
    WHL=""
    for toolkit_dir in /toolkits /workspace/0_Toolkits; do
        if [ -z "$WHL" ] && [ -d "$toolkit_dir" ]; then
            WHL=$(find "$toolkit_dir" -maxdepth 1 -type f -name 'tpu_mlir*.whl' -print -quit 2>/dev/null)
        fi
    done
    if [ -z "$WHL" ]; then
        echo "[Error] model_transform.py/model_deploy.py 不存在，且未找到 tpu_mlir wheel"
        exit 1
    fi
    python -m pip install "$WHL" -q --no-deps
fi

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
    --model "${PART_A_MODEL}"

cp "${PART_A_MODEL}" "${FINAL_PART_A_MODEL}"
echo "[Part A] 完成：$(ls -lh "${FINAL_PART_A_MODEL}" | awk '{print $5, $9}')"

# ── Part C1: Flow ────────────────────────────────────────────────────
echo ""
echo "[Part C1] 编译 Flow ..."
mkdir -p "${WORK_DIR}/part_c1" && cd "${WORK_DIR}/part_c1"

model_transform.py \
    --model_name vits_part_c1 \
    --model_def "${ONNX_DIR}/part_c1_flow.onnx" \
    --input_shapes "[[1,${Z_DIM},${T_MEL_FIXED}],[1,1,${T_MEL_FIXED}]]" \
    --mlir vits_part_c1.mlir

model_deploy.py \
    --mlir vits_part_c1.mlir \
    --quantize ${QUANTIZE} \
    --chip ${CHIP} \
    --model "${PART_C1_MODEL}"

cp "${PART_C1_MODEL}" "${FINAL_PART_C1_MODEL}"
echo "[Part C1] 完成：$(ls -lh "${FINAL_PART_C1_MODEL}" | awk '{print $5, $9}')"

# ── Part C2: Decoder ─────────────────────────────────────────────────
echo ""
echo "[Part C2] 编译 Decoder ..."
mkdir -p "${WORK_DIR}/part_c2" && cd "${WORK_DIR}/part_c2"

model_transform.py \
    --model_name vits_part_c2 \
    --model_def "${ONNX_DIR}/part_c2_decoder.onnx" \
    --input_shapes "[[1,${Z_DIM},${T_MEL_FIXED}]]" \
    --mlir vits_part_c2.mlir

model_deploy.py \
    --mlir vits_part_c2.mlir \
    --quantize ${QUANTIZE} \
    --chip ${CHIP} \
    --model "${PART_C2_MODEL}"

cp "${PART_C2_MODEL}" "${FINAL_PART_C2_MODEL}"
echo "[Part C2] 完成：$(ls -lh "${FINAL_PART_C2_MODEL}" | awk '{print $5, $9}')"

echo ""
echo "================================================================"
echo "  全部完成 [${QUANTIZE}]"
echo "  Part A: ${BMODEL_DIR}/vits_part_a_${QUANTIZE}.bmodel"
echo "  Part C1: ${BMODEL_DIR}/vits_part_c1_${QUANTIZE}.bmodel"
echo "  Part C2: ${BMODEL_DIR}/vits_part_c2_${QUANTIZE}.bmodel"
echo "================================================================"
