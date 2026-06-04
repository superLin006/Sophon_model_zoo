#!/bin/bash
# 将 Qwen3-0.6B / Qwen3-1.7B 编译为 bmodel（PTQ w4bf16, seq_len=2048）
# 用法（从宿主机运行）:
#   bash compile_qwen3_small_bmodel.sh 0.6b
#   bash compile_qwen3_small_bmodel.sh 1.7b
#   bash compile_qwen3_small_bmodel.sh all   # 依次编译两个
# 预计耗时: 0.6B ~30-60 min, 1.7B ~1-2 h
# 注: docker /workspace 对应宿主机 /home/xh/itc_project/Sophon_model_zoo/

set -e
# 脚本在 scripts/ 下，QwenLLM 根目录是上一级
QWEN_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${1:-all}"

compile_model() {
    local label="$1"        # 0.6B | 1.7B
    local weights_name="$2" # Qwen3-0.6B | Qwen3-1.7B
    local out_name="$3"     # qwen3_0.6b | qwen3_1.7b

    local weights_dir="${QWEN_DIR}/${weights_name}"
    local out_dir="${QWEN_DIR}/models/${out_name}"

    if [ ! -f "${weights_dir}/config.json" ]; then
        echo "[ERROR] 权重目录不存在: ${weights_dir}"
        echo "       请先运行: bash download_qwen3_small.sh ${label,,}"
        exit 1
    fi

    echo "========================================================"
    echo "[INFO] 编译 Qwen3-${label}  seq_len=2048  w4bf16"
    echo "[INFO] 权重: ${weights_dir}"
    echo "[INFO] 输出: ${out_dir}"
    echo "========================================================"

    docker start sophon-tpumlir

    docker exec sophon-tpumlir bash -c "
        set -e
        echo '[docker] 固定 transformers==4.51.1...'
        pip3 install 'transformers==4.51.1' -q 2>/dev/null || true

        echo '[docker] TPU-MLIR 版本:'
        llm_convert.py --version

        echo '[docker] 开始转换 Qwen3-${label}...'
        llm_convert.py \
            -m /workspace/QwenLLM/${weights_name} \
            -s 2048 \
            --quantize w4bf16 \
            -c bm1684x \
            --out_dir /workspace/QwenLLM/models/${out_name}

        echo '[docker] 完成，输出文件:'
        ls -lh /workspace/QwenLLM/models/${out_name}/*.bmodel 2>/dev/null || \
        ls -lh /workspace/QwenLLM/models/${out_name}/
    "

    echo "[INFO] 本地 bmodel:"
    ls -lh "${out_dir}"/*.bmodel 2>/dev/null || ls "${out_dir}/"
}

case "${TARGET}" in
    0.6b|0.6B)
        compile_model "0.6B" "Qwen3-0.6B" "qwen3_0.6b"
        ;;
    1.7b|1.7B)
        compile_model "1.7B" "Qwen3-1.7B" "qwen3_1.7b"
        ;;
    all)
        compile_model "0.6B" "Qwen3-0.6B" "qwen3_0.6b"
        compile_model "1.7B" "Qwen3-1.7B" "qwen3_1.7b"
        echo "[INFO] 全部编译完成"
        ;;
    *)
        echo "用法: $0 [0.6b|1.7b|all]"
        exit 1
        ;;
esac

echo ""
echo "下一步: bash $(dirname "$0")/deploy_to_board.sh"
