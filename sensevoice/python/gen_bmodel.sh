#!/bin/bash
# SenseVoice Small bmodel 转换脚本（支持 F32 / F16）
# 在 TPU-MLIR Docker 容器内执行（从 sensevoice/ 目录）:
#   docker run --rm \
#     -v $(pwd):/workspace \
#     -v $(pwd)/../../0_Toolkits:/toolkits \
#     sophgo/tpuc_dev:latest bash /workspace/python/gen_bmodel.sh [F32|F16]

set -e

# 安装 tpu_mlir（强制从本地 whl 安装）
WHL=$(ls /toolkits/tpu_mlir*.whl 2>/dev/null | head -1)
if [ -z "$WHL" ]; then
    echo "[Error] tpu_mlir whl not found in /toolkits/"
    exit 1
fi
pip install "$WHL" -q --no-deps 2>/dev/null || pip install "$WHL" -q

MODEL_NAME="sensevoice_small"
ONNX_DIR="/workspace/models/onnx"
BMODEL_DIR="/workspace/models/BM1684X"
WORK_DIR="/tmp/sensevoice_compile"

chip="bm1684x"
quantize="${1:-F32}"

if [ "${quantize}" != "F32" ] && [ "${quantize}" != "F16" ]; then
    echo "[Error] 只支持 F32 或 F16，收到: ${quantize}"
    exit 1
fi

mkdir -p "${BMODEL_DIR}" "${WORK_DIR}"

echo "================================================================"
echo "  SenseVoice Small BM1684X bmodel 转换"
echo "  Chip: ${chip}  Quant: ${quantize}"
echo "================================================================"

cd "${WORK_DIR}"
rm -rf sensevoice && mkdir sensevoice && cd sensevoice

model_transform.py \
    --model_name "${MODEL_NAME}" \
    --model_def "${ONNX_DIR}/${MODEL_NAME}_sim.onnx" \
    --input_shapes "[[1,166,560]]" \
    --mlir "${MODEL_NAME}.mlir"

model_deploy.py \
    --mlir "${MODEL_NAME}.mlir" \
    --quantize ${quantize} \
    --chip ${chip} \
    --model "${BMODEL_DIR}/${MODEL_NAME}_${quantize}.bmodel"

echo ""
echo "================================================================"
echo "  转换完成！"
ls -lh "${BMODEL_DIR}/${MODEL_NAME}_${quantize}.bmodel"
echo "================================================================"
