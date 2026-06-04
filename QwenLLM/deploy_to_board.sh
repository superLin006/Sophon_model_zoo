#!/bin/bash
# 将 QwenLLM 工作目录同步到板子并编译 python_demo
# 用法: ./deploy_to_board.sh

BOARD_IP="172.16.40.75"
BOARD_USER="root"
BOARD_PASS="1"
BOARD_DIR="/data/sophon-llm"

SSH="sshpass -p ${BOARD_PASS} ssh -o StrictHostKeyChecking=no ${BOARD_USER}@${BOARD_IP}"
SCP="sshpass -p ${BOARD_PASS} scp -o StrictHostKeyChecking=no -r"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "====== 1. 创建板子目录结构 ======"
$SSH "mkdir -p ${BOARD_DIR}/qwen3/python_demo ${BOARD_DIR}/qwen3/config"

echo "====== 2. 同步 config 和 python_demo 源码 ======"
${SCP} ${SCRIPT_DIR}/qwen3/python_demo/ ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/qwen3/python_demo/
${SCP} ${SCRIPT_DIR}/qwen3/config/      ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/qwen3/config/

echo "====== 3. 同步 benchmark 脚本 ======"
sshpass -p ${BOARD_PASS} scp -o StrictHostKeyChecking=no \
    ${SCRIPT_DIR}/benchmark_intent.py ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/

echo "====== 4. 同步 bmodel 文件（大文件，需要时间）======"
QWEN3_BMODEL=$(ls ${SCRIPT_DIR}/qwen3/qwen3_4b_seq2048/*.bmodel 2>/dev/null | head -1)
if [ -n "${QWEN3_BMODEL}" ]; then
    echo "  -> qwen3-4b (seq2048)..."
    sshpass -p ${BOARD_PASS} rsync -av \
        -e "ssh -o StrictHostKeyChecking=no" \
        "${QWEN3_BMODEL}" ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/qwen3/
fi

for size_dir in qwen3_0.6b_seq2048 qwen3_1.7b_seq2048; do
    SMALL_BMODEL=$(ls ${SCRIPT_DIR}/qwen3/${size_dir}/*.bmodel 2>/dev/null | head -1)
    if [ -n "${SMALL_BMODEL}" ]; then
        echo "  -> ${size_dir}..."
        sshpass -p ${BOARD_PASS} rsync -av \
            -e "ssh -o StrictHostKeyChecking=no" \
            "${SMALL_BMODEL}" ${BOARD_USER}@${BOARD_IP}:${BOARD_DIR}/qwen3/
    fi
done

echo "====== 5. 在板子上编译 qwen3 python_demo ======"
$SSH "cd ${BOARD_DIR}/qwen3/python_demo && \
      mkdir -p build && \
      cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4 && \
      cp *.cpython*.so .. && \
      echo 'qwen3 python_demo 编译完成'"

echo ""
echo "====== 部署完成！======"
echo ""
echo "在板子上运行 benchmark（用 nohup 防止 SSH 断开中断）："
echo ""
echo "  # qwen3-0.6b"
echo "  BMODEL=\$(ls ${BOARD_DIR}/qwen3/qwen3-0.6b*.bmodel | head -1)"
echo "  nohup python3 ${BOARD_DIR}/benchmark_intent.py \\"
echo "    -m \"\${BMODEL}\" -c ${BOARD_DIR}/qwen3/config -n qwen3-0.6b --no_think \\"
echo "    > ${BOARD_DIR}/result_0.6b.log 2>&1 &"
echo ""
echo "  # qwen3-1.7b"
echo "  BMODEL=\$(ls ${BOARD_DIR}/qwen3/qwen3-1.7b*.bmodel | head -1)"
echo "  nohup python3 ${BOARD_DIR}/benchmark_intent.py \\"
echo "    -m \"\${BMODEL}\" -c ${BOARD_DIR}/qwen3/config -n qwen3-1.7b --no_think \\"
echo "    > ${BOARD_DIR}/result_1.7b.log 2>&1 &"
echo ""
echo "  # qwen3-4b"
echo "  BMODEL=\$(ls ${BOARD_DIR}/qwen3/qwen3-4b*.bmodel | head -1)"
echo "  nohup python3 ${BOARD_DIR}/benchmark_intent.py \\"
echo "    -m \"\${BMODEL}\" -c ${BOARD_DIR}/qwen3/config -n qwen3-4b --no_think \\"
echo "    > ${BOARD_DIR}/result_4b.log 2>&1 &"
