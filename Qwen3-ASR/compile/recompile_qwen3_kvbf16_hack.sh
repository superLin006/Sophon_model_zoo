#!/bin/bash
# 编译 Qwen3-ASR 的 LLM（inputs_embeds 版）为 bmodel
# 流程复刻 Eureka-Audio/compile/recompile_qwen3.sh
# 注意：block 系列 W4BF16 量化；**不加 --disable_layer_group**（qwen3 加了会 SHA 校验失败；
#       且 Eureka 验证过 llm_convert.py 层融合版在 audio-embed 注入场景语义有损，已废弃）
set -e

# 默认编译 seq256（短音频/流式最优，decode 35 tok/s）；长音频改 SEQ=512/1024 并同步改导出
ONNX_DIR="./tmp/onnx_kvbf16_2"  # 导出: python export_qwen3_embeds.py --seq_length 256 --out_dir ./tmp/onnx_seq256
TMP_DIR="./tmp"
OUT_DIR="../models/BM1684X"
SEQ=256; HIDDEN=1024; N_LAYERS=28; N_KV_HEADS=8; HEAD_DIM=128; VOCAB=151936

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

echo "=== Compiling ${N_LAYERS} blocks (W4BF16) ==="
for i in $(seq 0 $((N_LAYERS - 1))); do
  echo -n "  block_${i} ..."
  model_transform.py \
    --model_name "block_${i}" \
    --model_def "${ONNX_DIR}/block/block_${i}.onnx" \
    --input_shapes "[[1,${SEQ},${HIDDEN}],[1,${SEQ}],[1,1,${SEQ},${SEQ}]]" \
    --mlir "${TMP_DIR}/block_${i}.mlir"
  # MLIR hack: KV 输入/输出 f32 → bf16（BM1684X LLM 输入 bf16 化，官方 llm_convert 同效）
  sed -i "s/tensor<1x${SEQ}x${N_KV_HEADS}x${HEAD_DIM}xf32>/tensor<1x${SEQ}x${N_KV_HEADS}x${HEAD_DIM}xbf16>/g" "${TMP_DIR}/block_${i}.mlir"
  model_deploy.py \
    --mlir "${TMP_DIR}/block_${i}.mlir" \
    --quantize W4BF16 --chip bm1684x \
    --model "${TMP_DIR}/block_${i}.bmodel"
  echo " done"
done

echo "=== Compiling ${N_LAYERS} block_cache (W4BF16) ==="
for i in $(seq 0 $((N_LAYERS - 1))); do
  echo -n "  block_cache_${i} ..."
  model_transform.py \
    --model_name "block_cache_${i}" \
    --model_def "${ONNX_DIR}/cache/block_cache_${i}.onnx" \
    --input_shapes "[[1,1,${HIDDEN}],[1,1],[1,1,1,$((SEQ+1))],[1,${SEQ},${N_KV_HEADS},${HEAD_DIM}],[1,${SEQ},${N_KV_HEADS},${HEAD_DIM}]]" \
    --mlir "${TMP_DIR}/block_cache_${i}.mlir"
  # MLIR hack: KV 输入/输出 f32 → bf16（BM1684X LLM 输入 bf16 化，官方 llm_convert 同效）
  sed -i "s/tensor<1x${SEQ}x${N_KV_HEADS}x${HEAD_DIM}xf32>/tensor<1x${SEQ}x${N_KV_HEADS}x${HEAD_DIM}xbf16>/g" "${TMP_DIR}/block_cache_${i}.mlir"
  model_deploy.py \
    --mlir "${TMP_DIR}/block_cache_${i}.mlir" \
    --quantize W4BF16 --chip bm1684x \
    --model "${TMP_DIR}/block_cache_${i}.bmodel"
  echo " done"
done

echo "=== Merging all bmodels ==="
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
  -o "${OUT_DIR}/qwen3_asr_llm_w4bf16_seq${SEQ}_bm1684x.bmodel"

echo ""
echo "Done: ${OUT_DIR}/qwen3_asr_llm_w4bf16_seq${SEQ}_bm1684x.bmodel"
