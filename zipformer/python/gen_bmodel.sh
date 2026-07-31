#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MANIFEST=${MANIFEST:-"$ROOT/configs/tensor_manifest.json"}
NETWORK=encoder
QUANTIZE=F32
ONNX=
OUTPUT=
DRY_RUN=0
DECODER_DISABLE_LAYER_GROUP=0

usage() { printf '%s\n' "Usage: $0 --network encoder|decoder|joiner --onnx FILE --output FILE [--quantize F32|F16] [--manifest FILE] [--decoder-disable-layer-group] [--dry-run]"; }
while (($#)); do
  case "$1" in
    --network) NETWORK=$2; shift 2;;
    --quantize) QUANTIZE=$2; shift 2;;
    --onnx) ONNX=$2; shift 2;;
    --output) OUTPUT=$2; shift 2;;
    --manifest) MANIFEST=$2; shift 2;;
    --decoder-disable-layer-group) DECODER_DISABLE_LAYER_GROUP=1; shift;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2;;
  esac
done
[[ "$NETWORK" == encoder || "$NETWORK" == decoder || "$NETWORK" == joiner ]] || { printf 'invalid network\n' >&2; exit 2; }
[[ -n "$ONNX" ]] || { printf '%s\n' '--onnx is required' >&2; exit 2; }
[[ -f "$MANIFEST" ]] || { printf 'manifest not found: %s\n' "$MANIFEST" >&2; exit 2; }
[[ -f "$ONNX" ]] || { printf 'ONNX not found: %s\n' "$ONNX" >&2; exit 2; }
[[ -n "$OUTPUT" ]] || { printf '%s\n' '--output is required' >&2; exit 2; }
case "$QUANTIZE" in F16|F32) ;; *) printf 'invalid quantize: %s\n' "$QUANTIZE" >&2; exit 2;; esac

SHAPES=$(python3 "$ROOT/python/tpumlir_args.py" --manifest "$MANIFEST" --network "$NETWORK" --onnx "$ONNX")
MLIR="${OUTPUT%.*}.mlir"
TRANSFORM=(model_transform.py --model_name "zipformer_${NETWORK}" --model_def "$ONNX" --input_shapes "$SHAPES" --mlir "$MLIR")
DEPLOY=(model_deploy.py --mlir "$MLIR" --quantize "$QUANTIZE" --chip bm1684x --model "$OUTPUT")
if [[ "$NETWORK" == encoder || ( "$NETWORK" == decoder && "$DECODER_DISABLE_LAYER_GROUP" == 1 ) ]]; then
  DEPLOY+=(--disable_layer_group)
fi
printf 'validated %s against manifest; input_shapes=%s\n' "$NETWORK" "$SHAPES"
printf '%q ' "${TRANSFORM[@]}"; printf '\n'
printf '%q ' "${DEPLOY[@]}"; printf '\n'
(( DRY_RUN )) && exit 0
mkdir -p "$(dirname "$OUTPUT")"
"${TRANSFORM[@]}"
"${DEPLOY[@]}"
