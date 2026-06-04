#!/usr/bin/env python3
"""
离线生成 prefix_embeds.bin，供 C++ main.cpp 直接读取。

把系统提示词通过 embed_tokens 转成 float32 embeds，存为二进制文件。
C++ 直接 fread 即可，避免在嵌入式设备上运行完整 tokenizer+embedding。

用法：
  python gen_prefix_embeds.py \
    --model_path ../../Eureka-Audio-Instruct \
    --system_prompt "把用户的语音指令归类为..." \
    --output_path ../../Eureka-Audio-Instruct/prefix_embeds.bin
"""

import os
import sys
import argparse
import struct
import torch
import numpy as np

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='../../Eureka-Audio-Instruct')
parser.add_argument('--output_path', default=None,
                    help='默认输出到 model_path/prefix_embeds.bin')
parser.add_argument('--system_prompt', default=
    '把用户的语音指令归类为以下动作之一：open_whiteboard, close_window, set_volume, '
    'open_camera, set_pen, draw_shape, set_tool, save_file, screenshot。\n'
    '输出JSON：{"action":"动作名","params":{}}。只输出JSON，不要解释。')
args = parser.parse_args()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

out_path = args.output_path or os.path.join(args.model_path, 'prefix_embeds.bin')

# ── 加载模型（只需要 tokenizer + embed_tokens） ────────────────────────────
print(f'Loading model ...')
from eureka_infer.api import EurekaAudio
wrapper = EurekaAudio(model_path=args.model_path, device='cpu')
model      = wrapper.model
tokenizer  = wrapper.tokenizer
processor  = wrapper.processor
embed_tokens = model.backbone.model.embed_tokens

# ── 构建对话模板（无音频，仅 prefix 部分） ─────────────────────────────────
# 格式：<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n
# 音频占位符会在 C++ 侧注入，这里不生成
# ── 构建 prefix/suffix 文本（与原版 processor 的 token 序列严格对齐）──────────
# 原版 case dump 出的真实 token：
#   prefix: <|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n<|audio_start|>
#   audio:  [音频 embed 注入]
#   suffix: <|audio_end|><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
#   注意：audio_end 后直接接 im_end，中间【没有】换行（之前误加 \n 导致语义偏移）
prefix_text = f'<|im_start|>system\n{args.system_prompt}<|im_end|>\n<|im_start|>user\n<|audio_start|>'
suffix_text = '<|audio_end|><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

print(f'Prefix text ({len(prefix_text)} chars):\n{repr(prefix_text[:200])}...')
print(f'Suffix text: {repr(suffix_text)}')

# Tokenize
prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
print(f'Prefix token count: {len(prefix_ids)},  last 3 IDs: {prefix_ids[-3:]}')
print(f'Suffix token count: {len(suffix_ids)},  IDs: {suffix_ids}')

# Embed
with torch.no_grad():
    prefix_embeds = embed_tokens(torch.tensor([prefix_ids], dtype=torch.long))[0].float().numpy()
    suffix_embeds = embed_tokens(torch.tensor([suffix_ids], dtype=torch.long))[0].float().numpy()

# 保存 prefix
prefix_embeds.tofile(out_path)
print(f'Saved prefix {prefix_embeds.shape} → {out_path}  ({os.path.getsize(out_path)/1024:.1f} KB)')

# 保存 suffix
suffix_path = out_path.replace('prefix_embeds.bin', 'suffix_embeds.bin')
suffix_embeds.tofile(suffix_path)
print(f'Saved suffix {suffix_embeds.shape} → {suffix_path}  ({os.path.getsize(suffix_path)/1024:.1f} KB)')

# 同时保存 token_ids.txt 方便调试
ids_path = out_path.replace('.bin', '_ids.txt')
with open(ids_path, 'w') as f:
    f.write(' '.join(str(x) for x in prefix_ids) + '\n')
print(f'Prefix Token IDs → {ids_path}')

suf_ids_path = suffix_path.replace('.bin', '_ids.txt')
with open(suf_ids_path, 'w') as f:
    f.write(' '.join(str(x) for x in suffix_ids) + '\n')
print(f'Suffix Token IDs → {suf_ids_path}')
