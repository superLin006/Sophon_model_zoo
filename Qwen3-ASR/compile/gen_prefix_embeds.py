#!/usr/bin/env python3
"""
从 M0 原生推理 dump 的 input_ids 自动提取 prefix/suffix token 布局，
离线生成 prefix_embeds.bin / suffix_embeds.bin（板卡直接读取，无需 embed 权重）。

思路（对齐 Eureka 教训：手写模板差一个 \n 都语义偏移）：
  以原生 processor 生成的 input_ids 为准，audio_token(151676) 连续段即为音频占位，
  段前 = prefix（system/user 前缀），段后 = suffix（im_end + assistant 生成提示）。

用法（qwen3-asr conda env）:
  python gen_prefix_embeds.py \
    --model_path ../models \
    --dump_json ../python/dump/baseline_zh_inputs.json \
    --output_dir ../models
"""
import os
import json
import argparse
import torch
import numpy as np

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='../models')
parser.add_argument('--dump_json', default='../python/dump/baseline_zh_inputs.json',
                    help='M0 infer_native.py 的 dump（含完整 input_ids）')
parser.add_argument('--output_dir', default='../models')
args = parser.parse_args()

from transformers import AutoModelForMultimodalLM

with open(args.dump_json, encoding='utf-8') as f:
    dump = json.load(f)

ids = dump['input_ids']
AUDIO_TOKEN = 151676

# 定位 audio token 段
audio_positions = [i for i, x in enumerate(ids) if x == AUDIO_TOKEN]
assert audio_positions, 'no audio tokens found in input_ids'
s, e = audio_positions[0], audio_positions[-1] + 1
prefix_ids = ids[:s]
suffix_ids = ids[e:]
print(f'input_ids len={len(ids)}  audio段 [{s}:{e}) 共 {e-s} 个 audio token')
print(f'prefix {len(prefix_ids)} ids: {prefix_ids}')
print(f'suffix {len(suffix_ids)} ids: {suffix_ids}')

# 加载 embed_tokens（只需 embedding 权重）
print(f'Loading model from {args.model_path} ...')
model = AutoModelForMultimodalLM.from_pretrained(
    args.model_path, dtype=torch.bfloat16, torch_dtype=torch.bfloat16)
embed_tokens = model.model.language_model.embed_tokens
del model

with torch.no_grad():
    prefix_embeds = embed_tokens(torch.tensor([prefix_ids])).float().numpy()[0]
    suffix_embeds = embed_tokens(torch.tensor([suffix_ids])).float().numpy()[0]

os.makedirs(args.output_dir, exist_ok=True)
prefix_embeds.tofile(os.path.join(args.output_dir, 'prefix_embeds.bin'))
suffix_embeds.tofile(os.path.join(args.output_dir, 'suffix_embeds.bin'))
print(f'Saved prefix {prefix_embeds.shape} → {args.output_dir}/prefix_embeds.bin')
print(f'Saved suffix {suffix_embeds.shape} → {args.output_dir}/suffix_embeds.bin')

# 存 ids 便于调试
with open(os.path.join(args.output_dir, 'prefix_ids.txt'), 'w') as f:
    f.write(' '.join(map(str, prefix_ids)))
with open(os.path.join(args.output_dir, 'suffix_ids.txt'), 'w') as f:
    f.write(' '.join(map(str, suffix_ids)))
print('Saved *_ids.txt')
