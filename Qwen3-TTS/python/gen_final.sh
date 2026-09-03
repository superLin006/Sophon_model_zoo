#!/bin/bash
# Qwen3-TTS 最终 3 bmodel 一键重建脚本（v1.1）
# 前置：models/onnx/ 下已有全部 ONNX（python export_talker.py / export_codec.py 生成）
# 产物（models/BM1684X/）：
#       qwen3_tts_talker_w8s192.bmodel（W8BF16 + SEQLEN=192：28 block + 28 cache + 2 embedding + codec_head）
#       qwen3_tts_cp_allf32.bmodel（code_predictor 40 网络：5 prefill F32 + 5 cache F16/16槽 + 15 lm_head F32 + 15 embedding F32）
#       codec_decoder.bmodel（codec 解码 F16）
# 转换中间产物（mlir/npz/json）全部写入 compile/tmp/，不污染 models/onnx/ 与 models/BM1684X/。
# 用法: docker run --rm -v <repo>:/workspace sophon/tpuc_dev:v3.4-tpumlir-1.28.1 \
#         bash /workspace/Qwen3-TTS/python/gen_final.sh
set -e
REPO=/workspace
ONNX=$REPO/Qwen3-TTS/models/onnx
OUT=$REPO/Qwen3-TTS/models/BM1684X
WORK=$REPO/Qwen3-TTS/compile/tmp/gen_final
COMP=$WORK/components
CHIP=bm1684x
LOG="$WORK/gen_final_build.log"

mkdir -p "$WORK" "$OUT" "$COMP"
cd "$WORK"   # model_deploy 的 cwd 相对产物（npz/json 等）一律落在 WORK，不污染 onnx 目录
: > "$LOG"
tf() { model_transform.py --model_name "$1" --model_def "$2" --input_shapes "$3" --mlir "$4" >>"$LOG" 2>&1 || { echo "[FAIL] transform $1"; tail -10 "$LOG"; exit 1; }; }
dpl() { model_deploy.py --mlir "$1" --quantize "$2" --chip $CHIP $3 --model "$4" >>"$LOG" 2>&1 || { echo "[FAIL] deploy $1"; tail -10 "$LOG"; exit 1; }; }

# ================= talker（W8BF16 + SEQLEN=192，28 层）=================
TALKER=""
for i in $(seq 0 27); do
  tf talker_block_$i "$ONNX/talker/talker_block_$i.onnx" '[[1,192,1024],[3,1,192],[1,1,192,192]]' "$WORK/talker_block_${i}_w8s192.mlir"
  dpl "$WORK/talker_block_${i}_w8s192.mlir" W8BF16 "" "$COMP/talker_block_${i}_w8s192.bmodel"
  TALKER="$TALKER $COMP/talker_block_${i}_w8s192.bmodel"
  tf talker_block_cache_$i "$ONNX/talker/talker_block_cache_$i.onnx" \
     '[[1,1,1024],[3,1,1],[1,1,1,193],[1,8,192,128],[1,8,192,128]]' "$WORK/talker_block_cache_${i}_w8s192.mlir"
  dpl "$WORK/talker_block_cache_${i}_w8s192.mlir" W8BF16 "" "$COMP/talker_block_cache_${i}_w8s192.bmodel"
  TALKER="$TALKER $COMP/talker_block_cache_${i}_w8s192.bmodel"
  echo "[talker] $i OK"
done
tf embedding_code "$ONNX/embedding/embedding_code.onnx" '[[1,192]]' "$WORK/embedding_code_s192.mlir"
dpl "$WORK/embedding_code_s192.mlir" F16 "" "$COMP/embedding_code_s192.bmodel"
tf embedding_text "$ONNX/embedding/embedding_text.onnx" '[[1,192]]' "$WORK/embedding_text_s192.mlir"
dpl "$WORK/embedding_text_s192.mlir" F16 "" "$COMP/embedding_text_s192.bmodel"
tf codec_head "$ONNX/embedding/codec_head.onnx" '[[1,1,1024]]' "$WORK/codec_head.mlir"
dpl "$WORK/codec_head.mlir" F16 "" "$COMP/codec_head.bmodel"
model_tool --combine $TALKER $COMP/embedding_code_s192.bmodel $COMP/embedding_text_s192.bmodel $COMP/codec_head.bmodel \
  -o "$OUT/qwen3_tts_talker_w8s192.bmodel" >/dev/null 2>&1
echo "[combine] talker done"

# ================= code_predictor（40 网络）=================
CP=""
for i in $(seq 0 4); do
  tf cp_block_$i "$ONNX/cp/cp_block_$i.onnx" '[[1,2,1024],[1,2],[1,1,2,2]]' "$WORK/cp_block_${i}_f32.mlir"
  dpl "$WORK/cp_block_${i}_f32.mlir" F32 "" "$COMP/cp_block_${i}_f32.bmodel"
  CP="$CP $COMP/cp_block_${i}_f32.bmodel"
  tf cp_block_cache_$i "$ONNX/cp/cp_block_cache_$i.onnx" \
     '[[1,1,1024],[1,1],[1,1,1,17],[1,8,16,128],[1,8,16,128]]' "$WORK/cp_block_cache_${i}_f16s16.mlir"
  dpl "$WORK/cp_block_cache_${i}_f16s16.mlir" F16 "" "$COMP/cp_block_cache_${i}_f16s16.bmodel"
  CP="$CP $COMP/cp_block_cache_${i}_f16s16.bmodel"
  echo "[cp] block $i OK"
done
for g in $(seq 0 14); do
  tf cp_lm_head_$g "$ONNX/cp/cp_lm_head_$g.onnx" '[[1,1,1024]]' "$WORK/cp_lm_head_${g}_f32.mlir"
  dpl "$WORK/cp_lm_head_${g}_f32.mlir" F32 "" "$COMP/cp_lm_head_${g}_f32.bmodel"
  CP="$CP $COMP/cp_lm_head_${g}_f32.bmodel"
done
# 注意：15 个 embedding（0..14），导出时勿漏 embedding_14（历史 bug）
for e in $(seq 0 14); do
  tf cp_embedding_$e "$ONNX/cp/cp_embedding_$e.onnx" '[[1,1]]' "$WORK/cp_embedding_${e}_f32.mlir"
  dpl "$WORK/cp_embedding_${e}_f32.mlir" F32 "" "$COMP/cp_embedding_${e}_f32.bmodel"
  CP="$CP $COMP/cp_embedding_${e}_f32.bmodel"
done
model_tool --combine $CP -o "$OUT/qwen3_tts_cp_allf32.bmodel" >/dev/null 2>&1
echo "[combine] cp done"

# ================= codec decoder（F16）=================
# 注：codec_decoder_T325.onnx 由 python/export_codec.py 从模型 speech_tokenizer 导出
#     若 codec_decoder.bmodel 已存在则跳过（它不依赖本仓库其他产物，无需每次重编）
if [ -f "$OUT/codec_decoder.bmodel" ]; then
  echo "[codec] skip (exists: $(ls -lh "$OUT/codec_decoder.bmodel" | awk '{print $5}'))"
else
  tf codec_decoder "$ONNX/codec/codec_decoder_T325.onnx" '[[1,16,325]]' "$WORK/codec_decoder_T325.mlir"
  model_deploy.py --mlir "$WORK/codec_decoder_T325.mlir" --quantize F16 --chip $CHIP \
    --disable_layer_group --model "$OUT/codec_decoder.bmodel" >/dev/null 2>&1
  echo "[codec] done"
fi

echo "=== 最终产物 ==="
ls -lh "$OUT/qwen3_tts_talker_w8s192.bmodel" "$OUT/qwen3_tts_cp_allf32.bmodel" "$OUT/codec_decoder.bmodel"
echo "=== 中间产物目录（可随时清理）==="
du -sh "$WORK"