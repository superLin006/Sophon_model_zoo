#!/usr/bin/env python3
"""
M0: Qwen3-ASR-0.6B 原生 CPU 推理基线 + dump。

跑通 test_zh/test_en，并 dump：
  1. 完整 input_ids（prefix / audio 占位符×N / suffix 布局）→ inputs_zh.json
  2. 输出文本（language <NAME><asr_text> 格式）→ outputs_zh.json
  3. mel 特征数值 + filter bank 矩阵 → mel_zh.npz / mel_filters.npz
  4. encoder 输出帧数实测（375 or 800）

用法（sophon-qwen3-asr conda env）:
  python infer_native.py [--audio ../test_data/test_zh.wav]
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
parser.add_argument('--model_dir', default=os.path.join(PROJECT_ROOT, 'models', 'qwen3-asr-0.6b'))
parser.add_argument('--audio', default=os.path.join(PROJECT_ROOT, 'test_data', 'test_zh.wav'))
parser.add_argument('--out_prefix', default='baseline_zh')
parser.add_argument('--dump_dir', default='dump')
args = parser.parse_args()
DUMP_DIR = os.path.abspath(args.dump_dir)

from transformers import AutoProcessor, AutoModelForMultimodalLM

print(f'Loading model from {args.model_dir} ...')
processor = AutoProcessor.from_pretrained(args.model_dir)
model = AutoModelForMultimodalLM.from_pretrained(
    args.model_dir, dtype=torch.bfloat16, torch_dtype=torch.bfloat16)
model.eval()

# ── 1. 构建输入 ────────────────────────────────────────────────────────────
inputs = processor.apply_transcription_request(audio=args.audio)

# dump input_ids 布局
ids = inputs['input_ids'][0].tolist()
tokens = processor.tokenizer.convert_ids_to_tokens(ids)
print(f'\n=== input_ids: len={len(ids)} ===')
for i, (tid, tok) in enumerate(zip(ids, tokens)):
    print(f'  [{i:4d}] {tid:7d} {tok!r}')

feat = inputs['input_features']
feat_mask = inputs['input_features_mask']
print(f'\ninput_features: {feat.shape}  dtype={feat.dtype}')
print(f'input_features_mask: {feat_mask.shape}  sum={feat_mask.sum().item()}')

# encoder 输出帧数实测（走 audio_tower 前向）
mel = feat.to(torch.bfloat16)
enc_out = model.model.audio_tower(input_features=mel, input_features_mask=feat_mask.to(torch.bfloat16))
print(f'audio_tower last_hidden_state: {enc_out.last_hidden_state.shape}')

# ── 2. 生成 ────────────────────────────────────────────────────────────────
# 注意：transformers 5.14 的 processor 返回 float32 特征，但模型权重是 bf16，
#       Qwen3ASR 的 forward 不会自动转 dtype → 需显式转换（否则 conv 报 dtype 不匹配）
#       mask 保持 fp32（bf16 求和有精度误差：3000 会算成 3008，破坏 cu_seqlens 窗口）
inputs['input_features'] = inputs['input_features'].to(torch.bfloat16)
# inputs['input_features_mask'] 保持 fp32

print('\n=== generate ===')
with torch.no_grad():
    out_ids = model.generate(**inputs, max_new_tokens=512)

prompt_len = inputs['input_ids'].shape[1]
gen = out_ids[0][prompt_len:].tolist()
print(f'generated tokens: {len(gen)}')
print('tokens:', processor.tokenizer.convert_ids_to_tokens(gen))

raw = processor.decode(gen)
parsed = processor.parse_output(raw)
print(f'\nraw     : {raw!r}')
print(f'parsed  : {parsed}')

# ── 3. 保存 dump ───────────────────────────────────────────────────────────
os.makedirs(DUMP_DIR, exist_ok=True)
out = {
    'audio': args.audio,
    'input_ids': ids,
    'input_tokens': tokens,
    'input_features_shape': list(feat.shape),
    'input_features_mask_sum': int(feat_mask.sum().item()),
    'generated_ids': gen,
    'generated_tokens': processor.tokenizer.convert_ids_to_tokens(gen),
    'raw': raw,
    'parsed': parsed,
}
with open(os.path.join(DUMP_DIR, f'{args.out_prefix}_inputs.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

# mel 特征（float32）与 mask
np.savez(os.path.join(DUMP_DIR, f'{args.out_prefix}_mel.npz'),
         input_features=feat.numpy().astype(np.float32),
         input_features_mask=feat_mask.numpy().astype(np.float32))

# mel filter bank 矩阵（128, n_fft//2+1），供板卡前端使用
mel_filters = processor.feature_extractor.mel_filters
np.savez(os.path.join(DUMP_DIR, 'mel_filters.npz'), mel_filters=mel_filters.astype(np.float32))
print(f'\nSaved {DUMP_DIR}/{args.out_prefix}_inputs.json, {DUMP_DIR}/{args.out_prefix}_mel.npz, {DUMP_DIR}/mel_filters.npz')
print(f'filter bank shape: {mel_filters.shape}')
