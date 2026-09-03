#!/bin/bash
# SenseVoice Small bmodel 转换脚本（支持 F32 / F16）
# 在 TPU-MLIR Docker 容器内执行（从 sensevoice/ 目录）:
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

MODEL_ROOT="/workspace/sensevoice"
MODEL_NAME="sensevoice_small"
ONNX_DIR="${MODEL_ROOT}/models/onnx"
BMODEL_DIR="${MODEL_ROOT}/models/BM1684X"
WORK_DIR="/tmp/sensevoice_compile"
OUTPUT_DIR="${WORK_DIR}/output"

chip="bm1684x"
quantize="${1:-F16}"   # 默认 F16（与 deploy_to_board.sh 默认一致），可传 F32

if [ "${quantize}" != "F32" ] && [ "${quantize}" != "F16" ]; then
    echo "[Error] 只支持 F32 或 F16，收到: ${quantize}"
    exit 1
fi

mkdir -p "${BMODEL_DIR}" "${WORK_DIR}"
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

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
    --model "${OUTPUT_DIR}/${MODEL_NAME}_${quantize}.bmodel"

mv "${OUTPUT_DIR}/${MODEL_NAME}_${quantize}.bmodel" \
   "${BMODEL_DIR}/${MODEL_NAME}_${quantize}.bmodel"

echo ""
echo "================================================================"
echo "  转换完成！"
ls -lh "${BMODEL_DIR}/${MODEL_NAME}_${quantize}.bmodel"
echo "================================================================"
