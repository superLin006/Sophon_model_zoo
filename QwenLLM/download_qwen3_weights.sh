#!/bin/bash
# 从 ModelScope 下载 Qwen3-4B-AWQ 权重
# 下载后权重位于: QwenLLM/Qwen3-4B-AWQ/
# 容器内路径: /workspace/QwenLLM/Qwen3-4B-AWQ/

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}/Qwen3-4B-AWQ"

if [ -d "${TARGET_DIR}" ] && [ "$(ls -A ${TARGET_DIR} 2>/dev/null | wc -l)" -gt 5 ]; then
    echo "[INFO] 权重目录已存在: ${TARGET_DIR}"
    ls -lh "${TARGET_DIR}"
    exit 0
fi

echo "[INFO] 开始从 ModelScope 下载 Qwen3-4B-AWQ..."
echo "[INFO] 目标目录: ${TARGET_DIR}"
conda run -n sophon-llm python3 -c "
from modelscope import snapshot_download
path = snapshot_download('Qwen/Qwen3-4B-AWQ', cache_dir='${SCRIPT_DIR}', local_dir='${TARGET_DIR}')
print('[DONE] 下载完成:', path)
"
echo "[INFO] 权重大小:"
du -sh "${TARGET_DIR}"
