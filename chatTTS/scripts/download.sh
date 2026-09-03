#!/usr/bin/env bash
set -euo pipefail

pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/../models"

required_files=(
    "${MODEL_DIR}/BM1684X/chattts-llama_int4_1dev_1024_bm1684x.bmodel"
    "${MODEL_DIR}/BM1684X/decoder_1-768-1024_bm1684x.bmodel"
    "${MODEL_DIR}/BM1684X/vocos_1-100-2048_bm1684x.bmodel"
    "${MODEL_DIR}/BM1688/chattts-llama_int4_1dev_1024_bm1688.bmodel"
    "${MODEL_DIR}/BM1688/decoder_1-768-1024_bm1688.bmodel"
    "${MODEL_DIR}/BM1688/vocos_1-100-2048_bm1688.bmodel"
    "${MODEL_DIR}/asset/DVAE_full.pt"
    "${MODEL_DIR}/asset/homophones_map.json"
    "${MODEL_DIR}/asset/tokenizer.pt"
    "${MODEL_DIR}/asset/tokenizer/vocab.txt"
)

all_files_exist=true
for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        all_files_exist=false
        break
    fi
done

if [[ "$all_files_exist" == true ]]; then
    printf '模型和运行时资产已存在，无需重复下载。\n'
    exit 0
fi

mkdir -p "${MODEL_DIR}/BM1684X" "${MODEL_DIR}/BM1688"
pushd "${MODEL_DIR}/BM1684X" >/dev/null
python3 -m dfss --url=open@sophgo.com:sophon-demo/ChatTTS/chattts-llama_int4_1dev_1024_bm1684x.bmodel
python3 -m dfss --url=open@sophgo.com:sophon-demo/ChatTTS/decoder_1-768-1024_bm1684x.bmodel
python3 -m dfss --url=open@sophgo.com:sophon-demo/ChatTTS/vocos_1-100-2048_bm1684x.bmodel
popd >/dev/null

pushd "${MODEL_DIR}/BM1688" >/dev/null
python3 -m dfss --url=open@sophgo.com:sophon-demo/ChatTTS/chattts-llama_int4_1dev_1024_bm1688.bmodel
python3 -m dfss --url=open@sophgo.com:sophon-demo/ChatTTS/decoder_1-768-1024_bm1688.bmodel
python3 -m dfss --url=open@sophgo.com:sophon-demo/ChatTTS/vocos_1-100-2048_bm1688.bmodel
popd >/dev/null

pushd "$MODEL_DIR" >/dev/null
python3 -m dfss --url=open@sophgo.com:sophon-demo/ChatTTS/asset.tar.gz
tar xvf asset.tar.gz
rm asset.tar.gz
popd >/dev/null

printf '模型和运行时资产下载完成。\n'