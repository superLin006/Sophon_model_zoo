#!/bin/bash
# 重新编译 qwen3 bmodel（不加 --disable_layer_group）
# whisper encoder 才需要这个参数，qwen3 加了会触发 SHA 校验失败
set -e

ONNX_DIR="./tmp/onnx"
TMP_DIR="./tmp"
OUT_DIR="../models/BM1684X"
SEQ=512; HIDDEN=2048; N_LAYERS=28; N_KV_HEADS=8; HEAD_DIM=128; VOCAB=151936

mkdir -p "${TMP_DIR}" "${OUT_DIR}"

echo "=== Compiling embedding_cache ==="
model_transform.py \
  --model_name embedding_cache \
  --model_def "${ONNX_DIR}/embedding_cache.onnx" \
  --input_shapes "[[1,1]]" \
  --mlir "${TMP_DIR}/embedding_cache.mlir"
model_deploy.py \
  --mlir "${TMP_DIR}/embedding_cache.mlir" \
  --quantize BF16 --chip bm1684x \
  --model "${TMP_DIR}/embedding_cache.bmodel"

echo "=== Compiling lm_head ==="
model_transform.py --model_name lm_head \
  --model_def "${ONNX_DIR}/lm_head.onnx" \
  --input_shapes "[[1,1,${HIDDEN}]]" \
  --mlir "${TMP_DIR}/lm_head.mlir"
model_deploy.py --mlir "${TMP_DIR}/lm_head.mlir" \
  --quantize BF16 --chip bm1684x \
  --model "${TMP_DIR}/lm_head.bmodel"

echo "=== Compiling greedy_head ==="
model_transform.py --model_name greedy_head \
  --model_def "${ONNX_DIR}/greedy_head.onnx" \
  --input_shapes "[[1,${VOCAB}]]" \
  --mlir "${TMP_DIR}/greedy_head.mlir"
model_deploy.py --mlir "${TMP_DIR}/greedy_head.mlir" \
  --quantize BF16 --chip bm1684x \
  --model "${TMP_DIR}/greedy_head.bmodel"

# block（28层）— 不加 --disable_layer_group
echo "=== Compiling ${N_LAYERS} blocks (W4BF16) ==="
for i in $(seq 0 $((N_LAYERS - 1))); do
  echo -n "  block_${i} ..."
  model_transform.py \
    --model_name "block_${i}" \
    --model_def "${ONNX_DIR}/block/block_${i}.onnx" \
    --input_shapes "[[1,${SEQ},${HIDDEN}],[1,${SEQ}],[1,1,${SEQ},${SEQ}]]" \
    --mlir "${TMP_DIR}/block_${i}.mlir"
  model_deploy.py \
    --mlir "${TMP_DIR}/block_${i}.mlir" \
    --quantize W4BF16 --chip bm1684x \
    --model "${TMP_DIR}/block_${i}.bmodel"
  echo " done"
done

# block_cache（28层）— 不加 --disable_layer_group
echo "=== Compiling ${N_LAYERS} block_cache (W4BF16) ==="
for i in $(seq 0 $((N_LAYERS - 1))); do
  echo -n "  block_cache_${i} ..."
  model_transform.py \
    --model_name "block_cache_${i}" \
    --model_def "${ONNX_DIR}/cache/block_cache_${i}.onnx" \
    --input_shapes "[[1,1,${HIDDEN}],[1,1],[1,1,1,$((SEQ+1))],[1,${SEQ},${N_KV_HEADS},${HEAD_DIM}],[1,${SEQ},${N_KV_HEADS},${HEAD_DIM}]]" \
    --mlir "${TMP_DIR}/block_cache_${i}.mlir"
  model_deploy.py \
    --mlir "${TMP_DIR}/block_cache_${i}.mlir" \
    --quantize W4BF16 --chip bm1684x \
    --model "${TMP_DIR}/block_cache_${i}.bmodel"
  echo " done"
done

# 合并
echo "=== Merging all Qwen3 bmodels ==="
BLOCK_LIST=""
for i in $(seq 0 $((N_LAYERS - 1))); do
  BLOCK_LIST="${BLOCK_LIST} ${TMP_DIR}/block_${i}.bmodel"
done
CACHE_LIST=""
for i in $(seq 0 $((N_LAYERS - 1))); do
  CACHE_LIST="${CACHE_LIST} ${TMP_DIR}/block_cache_${i}.bmodel"
done

model_tool --combine \
  "${TMP_DIR}/embedding_cache.bmodel" \
  ${BLOCK_LIST} \
  ${CACHE_LIST} \
  "${TMP_DIR}/lm_head.bmodel" \
  "${TMP_DIR}/greedy_head.bmodel" \
  -o "${OUT_DIR}/qwen3_1.7b_embeds_w4bf16_seq${SEQ}_bm1684x.bmodel"

echo ""
echo "Done: ${OUT_DIR}/qwen3_1.7b_embeds_w4bf16_seq${SEQ}_bm1684x.bmodel"
