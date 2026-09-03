#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

DOCKER_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$DOCKER_DIR/.." && pwd)
WHEEL=${1:-$REPO_ROOT/0_Toolkits/tpu_mlir-1.28.1-py3-none-any.whl}
TAG=${TPUMLIR_IMAGE:-sophon/tpuc_dev:v3.4-tpumlir-1.28.1}

[[ -f "$WHEEL" ]] || { printf 'wheel not found: %s\n' "$WHEEL" >&2; exit 2; }
[[ "$(basename "$WHEEL")" == tpu_mlir-1.28.1-py3-none-any.whl ]] || {
  printf 'unexpected wheel name: %s\n' "$(basename "$WHEEL")" >&2
  exit 2
}

BUILD_CONTEXT=$(mktemp -d)
trap 'rm -rf "$BUILD_CONTEXT"' EXIT
cp "$WHEEL" "$BUILD_CONTEXT/"

docker build \
  --tag "$TAG" \
  --file "$DOCKER_DIR/Dockerfile.tpumlir" \
  "$BUILD_CONTEXT"

docker image inspect "$TAG" --format 'built {{.RepoTags}} size={{.Size}} bytes'
