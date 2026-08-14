#!/bin/bash
# Qwen3-TTS 最终 3 bmodel 一键重建脚本（v1.0）
# 前置：models/onnx/ 下已有全部 ONNX（python export_talker.py 生成）
# 产物：qwen3_tts_talker_w8s192.bmodel（W8BF16 + SEQLEN=128）
#       qwen3_tts_cp_allf32.bmodel（CP 全 F32，40 网络）
#       codec_decoder.bmodel（F16）
# 用法: docker run --rm -v <repo>:/workspace sophon/tpuc_dev:v3.4-tpumlir-1.28.1 \
#         bash /workspace/Qwen3-TTS/python/gen_final.sh
set -e
ONNX=/workspace/Qwen3-TTS/models/onnx
OUT=/workspace/Qwen3-TTS/models/BM1684X
CHIP=bm1684x
LOG="$ONNX/gen_final_build.log"
cd "$ONNX"   # 重要：model_deploy 的中间产物（npz/mlir/json）写入 cwd，必须 cd 到 onnx 目录，避免散落到仓库根
: > "$LOG"
tf() { model_transform.py --model_name "$1" --model_def "$2" --input_shapes "$3" --mlir "$4" >>"$LOG" 2>&1 || { echo "[FAIL] transform $1"; tail -10 "$LOG"; exit 1; }; }
dpl() { model_deploy.py --mlir "$1" --quantize "$2" --chip $CHIP $3 --model "$4" >>"$LOG" 2>&1 || { echo "[FAIL] deploy $1"; tail -10 "$LOG"; exit 1; }; }

# ================= talker（W8BF16 + SEQLEN=128）=================
TALKER=""
for i in $(seq 0 27); do
  tf talker_block_$i "$ONNX/talker_block_$i.onnx" '[[1,192,1024],[3,1,192],[1,1,192,192]]' "$ONNX/talker_block_${i}_w8s192.mlir"
  dpl "$ONNX/talker_block_${i}_w8s192.mlir" W8BF16 "" "$OUT/talker_block_${i}_w8s192.bmodel"
  TALKER="$TALKER $OUT/talker_block_${i}_w8s192.bmodel"
  tf talker_block_cache_$i "$ONNX/talker_block_cache_$i.onnx" \
     '[[1,1,1024],[3,1,1],[1,1,1,193],[1,8,192,128],[1,8,192,128]]' "$ONNX/talker_block_cache_${i}_w8s192.mlir"
  dpl "$ONNX/talker_block_cache_${i}_w8s192.mlir" W8BF16 "" "$OUT/talker_block_cache_${i}_w8s192.bmodel"
  TALKER="$TALKER $OUT/talker_block_cache_${i}_w8s192.bmodel"
  echo "[talker] $i OK"
done
tf embedding_code "$ONNX/embedding_code.onnx" '[[1,192]]' "$ONNX/embedding_code_s192.mlir"
dpl "$ONNX/embedding_code_s192.mlir" F16 "" "$OUT/embedding_code_s192.bmodel"
tf embedding_text "$ONNX/embedding_text.onnx" '[[1,192]]' "$ONNX/embedding_text_s192.mlir"
dpl "$ONNX/embedding_text_s192.mlir" F16 "" "$OUT/embedding_text_s192.bmodel"
tf codec_head "$ONNX/codec_head.onnx" '[[1,1,1024]]' "$ONNX/codec_head.mlir"
dpl "$ONNX/codec_head.mlir" F16 "" "$OUT/codec_head.bmodel"
model_tool --combine $TALKER $OUT/embedding_code_s192.bmodel $OUT/embedding_text_s192.bmodel $OUT/codec_head.bmodel \
  -o "$OUT/qwen3_tts_talker_w8s192.bmodel" >/dev/null 2>&1
echo "[combine] talker done"

# ================= code_predictor（全 F32，40 网络）=================
CP=""
for i in $(seq 0 4); do
  tf cp_block_$i "$ONNX/cp_block_$i.onnx" '[[1,2,1024],[1,2],[1,1,2,2]]' "$ONNX/cp_block_${i}_f32.mlir"
  dpl "$ONNX/cp_block_${i}_f32.mlir" F32 "" "$OUT/cp_block_${i}_f32.bmodel"
  CP="$CP $OUT/cp_block_${i}_f32.bmodel"
  tf cp_block_cache_$i "$ONNX/cp_block_cache_$i.onnx" \
     '[[1,1,1024],[1,1],[1,1,1,17],[1,8,16,128],[1,8,16,128]]' "$ONNX/cp_block_cache_${i}_f16s16.mlir"
  dpl "$ONNX/cp_block_cache_${i}_f16s16.mlir" F16 "" "$OUT/cp_block_cache_${i}_f16s16.bmodel"
  CP="$CP $OUT/cp_block_cache_${i}_f16s16.bmodel"
  echo "[cp] block $i OK"
done
for g in $(seq 0 14); do
  tf cp_lm_head_$g "$ONNX/cp_lm_head_$g.onnx" '[[1,1,1024]]' "$ONNX/cp_lm_head_${g}_f32.mlir"
  dpl "$ONNX/cp_lm_head_${g}_f32.mlir" F32 "" "$OUT/cp_lm_head_${g}_f32.bmodel"
  CP="$CP $OUT/cp_lm_head_${g}_f32.bmodel"
done
# 注意：15 个 embedding（0..14），导出时勿漏 embedding_14（历史 bug）
for e in $(seq 0 14); do
  tf cp_embedding_$e "$ONNX/cp_embedding_$e.onnx" '[[1,1]]' "$ONNX/cp_embedding_${e}_f32.mlir"
  dpl "$ONNX/cp_embedding_${e}_f32.mlir" F32 "" "$OUT/cp_embedding_${e}_f32.bmodel"
  CP="$CP $OUT/cp_embedding_${e}_f32.bmodel"
done
model_tool --combine $CP -o "$OUT/qwen3_tts_cp_allf32.bmodel" >/dev/null 2>&1
echo "[combine] cp done"

# ================= codec decoder（F16）=================
# 注：codec_decoder_T325.onnx 需从模型 speech_tokenizer 导出（独立脚本，见 python/export_codec.py）
#     若 codec_decoder.bmodel 已存在则跳过（它不依赖本仓库其他产物，无需每次重编）
if [ -f "$OUT/codec_decoder.bmodel" ]; then
  echo "[codec] skip (exists: $(ls -lh "$OUT/codec_decoder.bmodel" | awk '{print $5}'))"
else
  tf codec_decoder "$ONNX/codec_decoder_T325.onnx" '[[1,16,325]]' "$ONNX/codec_decoder_T325.mlir"
  model_deploy.py --mlir "$ONNX/codec_decoder_T325.mlir" --quantize F16 --chip $CHIP \
    --disable_layer_group --model "$OUT/codec_decoder.bmodel" >/dev/null 2>&1
  echo "[codec] done"
fi

echo "=== 最终产物 ==="
ls -lh "$OUT/qwen3_tts_talker_w8s192.bmodel" "$OUT/qwen3_tts_cp_allf32.bmodel" "$OUT/codec_decoder.bmodel"
