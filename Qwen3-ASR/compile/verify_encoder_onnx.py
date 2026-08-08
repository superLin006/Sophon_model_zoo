#!/usr/bin/env python3
"""
校验 qwen3_asr_encoder.onnx 与原生 PyTorch 前向数值一致（cosine 相似度）。

用法（qwen3-asr conda env）:
  python verify_encoder_onnx.py --model_path ../models
"""
import argparse
import numpy as np
import torch
import onnxruntime as ort

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='../models')
parser.add_argument('--onnx', default='./tmp/onnx/qwen3_asr_encoder.onnx')
args = parser.parse_args()

from transformers import AutoModelForMultimodalLM

print('Loading model ...')
# 强制 eager attention：原生默认 sdpa（flash 风格内核，CPU 上数值路径与 eager 不同，
# 且 sdpa 无法导出 ONNX）。我们的导出实现是 eager 语义，故原生基线也用 eager 对比。
model = AutoModelForMultimodalLM.from_pretrained(
    args.model_path, dtype=torch.bfloat16, torch_dtype=torch.bfloat16,
    attn_implementation='eager').eval()
enc = model.model.audio_tower
proj = model.model.multi_modal_projector
# audio_tower 有独立嵌套 config，需逐层强制 eager
for layer in enc.layers:
    layer.self_attn.config._attn_implementation = 'eager'
print('attn impl:', enc.layers[0].self_attn.config._attn_implementation)

# 随机 mel（模拟真实输入：-1.5 到 1.5 之间）
# 注意：mel 转 bf16（conv 权重是 bf16），但 mask 保持 fp32——
# bf16 求和在 3000 处有舍入误差(3000→3008)会破坏 cu_seqlens 窗口划分
torch.manual_seed(0)
mel = (torch.rand(1, 128, 3000) * 3 - 1.5).to(torch.bfloat16)
mask = torch.ones(1, 3000, dtype=torch.float32)

# 原生前向（bf16 计算，与导出 fp16 会有量化差异，阈值放宽）
with torch.no_grad():
    out = enc(input_features=mel, input_features_mask=mask)
    ref = proj(out.last_hidden_state).float().numpy()      # [390, 1024]

# ONNX 前向
sess = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])
onnx_out = sess.run(['audio_embeds'], {'mel': mel.half().numpy()})[0]
print(f'ref shape {ref.shape}  onnx shape {onnx_out.shape}')

# 余弦相似度
def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def mse(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))

c = cos(ref, onnx_out)
m = mse(ref, onnx_out)
print(f'cosine = {c:.6f}   mse = {m:.6e}   max_abs_diff = {np.abs(ref - onnx_out).max():.4f}')
if c > 0.999:
    print('PASS: cosine > 0.999')
else:
    print('FAIL: cosine <= 0.999')
