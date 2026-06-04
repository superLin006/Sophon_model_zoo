#!/bin/bash
# 从 ModelScope 下载 Qwen3-0.6B 和 Qwen3-1.7B 权重（base 模型即 chat 模型）
# 用法: bash download_qwen3_small_weights.sh [0.6b|1.7b|all]
# 默认下载全部

set -e
# 脚本在 scripts/ 下，权重下载到 QwenLLM 根目录（compile 脚本从那里读）
QWEN_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${1:-all}"

download_model() {
    local model_id="$1"
    local local_name="$2"
    local target_dir="${QWEN_DIR}/${local_name}"

    if [ -d "${target_dir}" ] && [ "$(ls -A ${target_dir} 2>/dev/null | wc -l)" -gt 5 ]; then
        echo "[INFO] 已存在，跳过: ${target_dir}"
        du -sh "${target_dir}"
        return 0
    fi

    echo "[INFO] 下载 ${model_id} → ${target_dir}"
    conda run -n sophon-llm python3 -c "
from modelscope import snapshot_download
path = snapshot_download('${model_id}', cache_dir='${QWEN_DIR}', local_dir='${target_dir}')
print('[DONE] 下载完成:', path)
"
    echo "[INFO] 大小:"
    du -sh "${target_dir}"
}

case "${TARGET}" in
    0.6b|0.6B)
        download_model "Qwen/Qwen3-0.6B" "Qwen3-0.6B"
        ;;
    1.7b|1.7B)
        download_model "Qwen/Qwen3-1.7B" "Qwen3-1.7B"
        ;;
    all)
        download_model "Qwen/Qwen3-0.6B" "Qwen3-0.6B"
        download_model "Qwen/Qwen3-1.7B" "Qwen3-1.7B"
        echo "[INFO] 全部下载完成"
        ;;
    *)
        echo "用法: $0 [0.6b|1.7b|all]"
        exit 1
        ;;
esac
