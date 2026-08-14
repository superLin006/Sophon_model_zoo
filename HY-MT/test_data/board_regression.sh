#!/bin/bash
set -euo pipefail

MODEL_DIR="${1:-/data/hymt}"
BIN="${MODEL_DIR}/hymt_demo"

run_case() {
    local name="$1"
    local prompt="$2"
    echo "===== ${name} ====="
    "${BIN}" "${MODEL_DIR}" "${prompt}" 128
}

run_case en_zh_short 'Translate the following segment into Chinese, without additional explanation.\n\nIt’s on the house.'
run_case zh_en_short '将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释：\n\n这个方案需要在周五之前完成。'
run_case ja_zh_short 'Translate the following segment into Chinese, without additional explanation.\n\nこの電車は東京駅に止まりますか。'
run_case terminology '参考下面的翻译：\n算能 翻译成 Sophon\n\n将以下文本翻译为英语，注意只需要输出翻译后的结果，不要额外解释：\n算能的处理器适合边缘推理。'
run_case formatting '将以下<source></source>之间的文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释，原文中的<sn></sn>标签表示标签内文本包含格式信息，需要在译文中相应的位置尽量保留该标签。输出格式为：<target>str</target>\n\n<source>Install <sn>transformers==4.56.1</sn> before running the model.</source>'

