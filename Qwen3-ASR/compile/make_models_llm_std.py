#!/usr/bin/env python3
"""
从 HF 原版 Qwen3ASR 权重生成 llm_convert 编译用的标准 Qwen3 权重目录（models_llm_std/）。

为什么需要：llm_convert.py 的标准 Qwen3 转换要求标准权重前缀（model.layers.*），
而 transformers 5.14 的 Qwen3ASR 权重前缀是 model.language_model.layers.*。
改动只有两处：
  1. 权重键去掉 `model.language_model.` 前缀（audio_tower / multi_modal_projector 丢弃，编译不需要）
  2. config.json 的 model_type 改为 qwen3、architectures 改为 Qwen3ForCausalLM
其余（tokenizer 等）原样复制。

用法（qwen3-asr conda env）:
  cd Qwen3-ASR/compile && python make_models_llm_std.py
"""
import os
import json
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, default='../models', help='HF 原版 Qwen3ASR 权重目录')
parser.add_argument('--dst', type=str, default='../models_llm_std', help='输出标准 Qwen3 权重目录')
args = parser.parse_args()

SRC = os.path.abspath(args.src)
DST = os.path.abspath(args.dst)
PREFIX = 'model.language_model.'          # 去掉的前缀
PREFIX_LEN = len(PREFIX)

def main():
    print(f'源（HF 原版）: {SRC}')
    print(f'目标（标准 Qwen3）: {DST}')

    # ── 1. 权重重命名（safetensors 流式读写，避免整包载入内存）──
    from safetensors import safe_open
    from safetensors.torch import save_file
    src_path = os.path.join(SRC, 'model.safetensors')
    dst_path = os.path.join(DST, 'model.safetensors')
    assert os.path.exists(src_path), f'缺少 {src_path}（需先下载 HF 原版权重）'
    os.makedirs(DST, exist_ok=True)

    renamed, dropped = 0, 0
    with safe_open(src_path, framework='pt') as f:
        tensors = {}
        for k in f.keys():
            if k.startswith(PREFIX):
                tensors['model.' + k[PREFIX_LEN:]] = f.get_tensor(k)
                renamed += 1
            else:
                dropped += 1   # audio_tower / multi_modal_projector 等
        print(f'重命名 {renamed} 个权重键，丢弃 {dropped} 个（audio_tower/projector）')
        save_file(tensors, dst_path)
    print(f'权重: {dst_path} ({os.path.getsize(dst_path)/1e9:.2f} GB)')

    # ── 2. config.json：model_type=qwen3 + Qwen3ForCausalLM ──
    cfg = json.load(open(os.path.join(SRC, 'config.json')))
    cfg['model_type'] = 'qwen3'
    cfg['architectures'] = ['Qwen3ForCausalLM']
    # 去掉 ASR 特有字段（无碍，但保持干净）
    for k in ['audio_tower_config', 'multi_modal_projector_config']:
        cfg.pop(k, None)
    json.dump(cfg, open(os.path.join(DST, 'config.json'), 'w'), indent=2, ensure_ascii=False)
    print('config.json: model_type=qwen3, architectures=Qwen3ForCausalLM')

    # ── 3. tokenizer 等原样复制 ──
    for name in ['tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json',
                 'generation_config.json', 'chat_template.jinja', 'special_tokens_map.json',
                 'added_tokens.json']:
        src = os.path.join(SRC, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(DST, name))
            print(f'复制 {name}')

    print(f'\n完成：{DST}/')
    print('下一步: llm_convert.py -m models_llm_std -s 512 --quantize w4bf16 -g 64 -c bm1684x ...')

if __name__ == '__main__':
    main()
