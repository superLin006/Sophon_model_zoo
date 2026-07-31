#!/usr/bin/env bash
set -euo pipefail

CPP_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$CPP_DIR/../.." && pwd)
MODE=${1:-host}
JOBS=${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '2')}

case "$MODE" in
  host)
    BUILD_DIR=${BUILD_DIR:-$CPP_DIR/build-host}
    cmake -S "$CPP_DIR" -B "$BUILD_DIR" \
      -DZIPFORMER_WITH_BM=OFF \
      -DZIPFORMER_BUILD_FBANK=OFF \
      -DZIPFORMER_BUILD_TESTS=ON
    cmake --build "$BUILD_DIR" -j"$JOBS"
    ctest --test-dir "$BUILD_DIR" --output-on-failure
    ;;
  cross)
    IMAGE=${CROSS_BUILD_IMAGE:-sophon-cross-build}
    docker image inspect "$IMAGE" >/dev/null 2>&1 || {
      printf 'cross-build image not found: %s\nBuild it with: docker build -t sophon-cross-build -f 3_docker/Dockerfile.cross-build .\n' "$IMAGE" >&2
      exit 2
    }
    docker run --rm \
      --user "$(id -u):$(id -g)" \
      -v "$REPO_ROOT":/workspace \
      -w /workspace \
      "$IMAGE" bash -lc '
        set -euo pipefail
        cmake -S zipformer/cpp -B zipformer/cpp/build-aarch64 \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
          -DZIPFORMER_WITH_BM=ON \
          -DZIPFORMER_BUILD_TESTS=OFF \
          -DSOPHON_SDK=/workspace/0_Toolkits/soc-sdk-sp4 \
          -DKALDI_FBANK_DIR=/workspace/1_third_party/kaldi_native_fbank
        cmake --build zipformer/cpp/build-aarch64 --target zipformer_cli -j'"$JOBS"'
      '
    ;;
  *)
    printf 'usage: %s [host|cross]\n' "$0" >&2
    exit 2
    ;;
esac
