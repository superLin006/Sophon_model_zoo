#!/bin/bash
# 在 sophgo/tpuc_dev:latest 容器内将 Qwen3-4B-AWQ 转换为 bmodel (seq_len=2048)
# 用法（从宿主机运行）: bash compile_qwen3_bmodel.sh
# 预计耗时: ~2-4 小时

set -e
# 脚本在 scripts/ 下，QwenLLM 根目录是上一级
QWEN_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WEIGHTS_DIR="${QWEN_DIR}/Qwen3-4B-AWQ"
OUTPUT_DIR="${QWEN_DIR}/models"

if [ ! -f "${WEIGHTS_DIR}/config.json" ]; then
    echo "[ERROR] 权重目录不存在或未下载完成: ${WEIGHTS_DIR}"
    echo "       请先运行: bash download_qwen3_4b.sh"
    exit 1
fi

echo "[INFO] TPU-MLIR 容器内转换 Qwen3-4B-AWQ, seq_len=2048"
echo "[INFO] 权重: ${WEIGHTS_DIR}"
echo "[INFO] 输出: ${OUTPUT_DIR}/qwen3_4b/"

docker start sophon-tpumlir
docker exec sophon-tpumlir bash -c "
    set -e
    echo '[docker] 固定 transformers==4.51.1（与 PyTorch 2.1.0+cpu 兼容）...'
    pip3 install "transformers==4.51.1" -q 2>/dev/null || true

    echo '[docker] TPU-MLIR 版本:'
    llm_convert.py --version

    echo '[docker] 开始转换...'
    llm_convert.py \
        -m /workspace/QwenLLM/Qwen3-4B-AWQ \
        -s 2048 \
        --quantize w4f16 \
        -c bm1684x \
        --out_dir /workspace/QwenLLM/models/qwen3_4b

    echo '[docker] 转换完成，输出文件:'
    ls -lh /workspace/QwenLLM/models/qwen3_4b/*.bmodel 2>/dev/null || \
    ls -lh /workspace/QwenLLM/models/qwen3_4b/
"

echo "[INFO] bmodel 文件:"
ls -lh "${OUTPUT_DIR}/qwen3_4b/"*.bmodel 2>/dev/null || ls -lh "${OUTPUT_DIR}/qwen3_4b/"

echo ""
echo "[INFO] 下一步：scp bmodel 到板卡并测试"
echo "  BMODEL=\$(ls ${OUTPUT_DIR}/qwen3_4b/*.bmodel | head -1)"
echo "  sshpass -p 1 scp \"\${BMODEL}\" root@172.16.40.75:/data/sophon-llm/qwen3/"
