#!/bin/bash
# 部署 Qwen3-TTS 到 BM1684X（172.16.25.248, root/1）
# 3 文件配置：talker_w8s192 + cp_allf32（内含 F16 16槽 cache）+ codec_decoder
# 运行时不需要 --cp_cache_bmodel（cache 网络已合并进 cp_allf32）
set -e

IP=172.16.25.248
USER=root
PASS=1
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOARD_PATH=/data/Qwen3-TTS

sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no $USER@$IP "mkdir -p $BOARD_PATH/models $BOARD_PATH/assets"

echo "[deploy] 上传 bmodel（3 文件）..."
for f in qwen3_tts_talker_w8s192.bmodel qwen3_tts_cp_allf32.bmodel codec_decoder.bmodel; do
    sshpass -p "$PASS" scp -o StrictHostKeyChecking=no \
        "$ROOT_DIR/models/BM1684X/$f" \
        $USER@$IP:$BOARD_PATH/models/
done

echo "[deploy] 上传二进制 + 资产..."
sshpass -p "$PASS" scp -o StrictHostKeyChecking=no \
    "$ROOT_DIR/cpp/build/qwen3_tts_bm1684x" \
    $USER@$IP:$BOARD_PATH/
sshpass -p "$PASS" scp -o StrictHostKeyChecking=no \
    "$ROOT_DIR/assets/tokenizer.json" \
    $USER@$IP:$BOARD_PATH/assets/

echo "[deploy] md5 校验（scp 崩传检测）..."
for f in qwen3_tts_talker_w8s192.bmodel qwen3_tts_cp_allf32.bmodel codec_decoder.bmodel qwen3_tts_bm1684x; do
    local_md5=$(md5sum "$ROOT_DIR/models/BM1684X/$f" 2>/dev/null | cut -d' ' -f1)
    [ "$f" = qwen3_tts_bm1684x ] && local_md5=$(md5sum "$ROOT_DIR/cpp/build/$f" | cut -d' ' -f1)
    board_md5=$(sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no $USER@$IP \
        "md5sum $BOARD_PATH/$f 2>/dev/null | cut -d' ' -f1")
    if [ "$local_md5" = "$board_md5" ]; then
        echo "  $f OK"
    else
        echo "  $f MISMATCH (local=$local_md5 board=$board_md5)"; exit 1
    fi
done

echo "[deploy] 完成，验证..."
sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no $USER@$IP "ls -lh $BOARD_PATH/models/ $BOARD_PATH/assets/"
