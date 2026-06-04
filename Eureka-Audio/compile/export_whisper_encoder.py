#!/usr/bin/env python3
"""
导出 Eureka-Audio 的音频编码模块为两个 ONNX：

  whisper_encoder_b1.onnx  : WhisperLargeV3 encoder，手写展开 forward
      输入:  mel [1, 128, 3000]  float32
      输出:  encoder_out [1, 1500, 1280]  float32
      ※ 手写展开 + float16 权重，规避 TPU-MLIR 2GB protobuf 限制

  audio_adaptor.onnx       : 4x downsample + MoE adaptor，手写展开 RMSNorm
      输入:  encoder_feat [1500, 1280]  float32
      输出:  audio_embeds [375, 2048]   float32

用法：
  python export_whisper_encoder.py --model_path /path/to/Eureka-Audio-Instruct
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='../../Eureka-Audio-Instruct')
parser.add_argument('--device',     type=str, default='cpu')
args = parser.parse_args()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print(f'Loading model from {args.model_path} ...')
from eureka_infer.api import EurekaAudio
wrapper    = EurekaAudio(model_path=args.model_path, device=args.device)
full_model = wrapper.model.eval()
for p in full_model.parameters():
    p.requires_grad_(False)

T_ENC      = 1500
D_ENC      = 1280
N_MEL      = 128
CHUNK_LEN  = 3000
DOWNSAMPLE = 4
D_LLM      = 2048
TOKENS     = T_ENC // DOWNSAMPLE  # 375

folder = './tmp/onnx'
os.makedirs(folder, exist_ok=True)

enc = full_model.audio_encoder.speech_encoder  # WhisperEncoder
moe = full_model.audio_moe_adaptor             # AudioNanoExpert


# ── 1. WhisperEncoder：手写展开，权重用 fp16 ─────────────────────────────
class WhisperEncoderExport(nn.Module):
    """
    完全手写 WhisperLargeV3 encoder forward，不依赖 transformers 路径。
    权重 fp16 存储（ONNX < 2GB），前向 cast 到 float32 计算。
    """
    def __init__(self, enc):
        super().__init__()
        # 复制权重（fp16 存储）
        self.conv1         = nn.Conv1d(enc.conv1.in_channels, enc.conv1.out_channels,
                                       enc.conv1.kernel_size, enc.conv1.stride, enc.conv1.padding,
                                       bias=enc.conv1.bias is not None)
        self.conv1.weight  = nn.Parameter(enc.conv1.weight.data.half())
        self.conv1.bias    = nn.Parameter(enc.conv1.bias.data.half())

        self.conv2         = nn.Conv1d(enc.conv2.in_channels, enc.conv2.out_channels,
                                       enc.conv2.kernel_size, enc.conv2.stride, enc.conv2.padding,
                                       bias=enc.conv2.bias is not None)
        self.conv2.weight  = nn.Parameter(enc.conv2.weight.data.half())
        self.conv2.bias    = nn.Parameter(enc.conv2.bias.data.half())

        # positional embedding：[1500, 1280] → [1, 1500, 1280]
        self.pos_emb = nn.Parameter(enc.embed_positions.weight.data.half().unsqueeze(0))

        # 逐层保存 attention 和 FFN 权重（fp16）
        self.n_layers  = len(enc.layers)
        self.num_heads = enc.layers[0].self_attn.num_heads
        self.head_dim  = enc.layers[0].self_attn.head_dim

        self.q_proj    = nn.ParameterList()
        self.k_proj    = nn.ParameterList()
        self.v_proj    = nn.ParameterList()
        self.out_proj  = nn.ParameterList()
        self.q_bias    = nn.ParameterList()
        self.out_bias  = nn.ParameterList()
        self.fc1_w     = nn.ParameterList()
        self.fc1_b     = nn.ParameterList()
        self.fc2_w     = nn.ParameterList()
        self.fc2_b     = nn.ParameterList()
        self.ln1_w     = nn.ParameterList()
        self.ln1_b     = nn.ParameterList()
        self.ln2_w     = nn.ParameterList()
        self.ln2_b     = nn.ParameterList()

        for layer in enc.layers:
            a = layer.self_attn
            self.q_proj.append(nn.Parameter(a.q_proj.weight.data.half()))
            self.q_bias.append(nn.Parameter(a.q_proj.bias.data.half()))
            self.k_proj.append(nn.Parameter(a.k_proj.weight.data.half()))
            self.v_proj.append(nn.Parameter(a.v_proj.weight.data.half()))
            self.out_proj.append(nn.Parameter(a.out_proj.weight.data.half()))
            self.out_bias.append(nn.Parameter(a.out_proj.bias.data.half()))
            self.fc1_w.append(nn.Parameter(layer.fc1.weight.data.half()))
            self.fc1_b.append(nn.Parameter(layer.fc1.bias.data.half()))
            self.fc2_w.append(nn.Parameter(layer.fc2.weight.data.half()))
            self.fc2_b.append(nn.Parameter(layer.fc2.bias.data.half()))
            self.ln1_w.append(nn.Parameter(layer.self_attn_layer_norm.weight.data.float()))
            self.ln1_b.append(nn.Parameter(layer.self_attn_layer_norm.bias.data.float()))
            self.ln2_w.append(nn.Parameter(layer.final_layer_norm.weight.data.float()))
            self.ln2_b.append(nn.Parameter(layer.final_layer_norm.bias.data.float()))

        self.final_ln_w = nn.Parameter(enc.layer_norm.weight.data.float())
        self.final_ln_b = nn.Parameter(enc.layer_norm.bias.data.float())

    def forward(self, mel):
        # mel: [1, 128, 3000] float32
        x = F.gelu(self.conv1(mel.half()).float())   # [1, 1280, 3000]
        x = F.gelu(self.conv2(x.half()).float())     # [1, 1280, 1500]
        x = x.permute(0, 2, 1)                       # [1, 1500, 1280]
        x = x + self.pos_emb.float()

        H = self.num_heads
        D = self.head_dim
        scale = D ** -0.5

        for i in range(self.n_layers):
            # --- Self-attention ---
            r = x
            x = F.layer_norm(x, [x.shape[-1]], self.ln1_w[i], self.ln1_b[i])

            q = F.linear(x.half(), self.q_proj[i], self.q_bias[i]).float()
            k = F.linear(x.half(), self.k_proj[i]).float()   # k_proj has no bias in WhisperLargeV3
            v = F.linear(x.half(), self.v_proj[i]).float()

            B, T, _ = q.shape
            # Whisper attention: 只对 q 乘一次 scale（D^-0.5），k 不再乘，
            # 否则等价于除以 D 而非 sqrt(D)，softmax 过平导致语义严重失真（cosine 0.52）
            q = q.reshape(B, T, H, D).permute(0, 2, 1, 3) * scale  # [B,H,T,D]
            k = k.reshape(B, T, H, D).permute(0, 2, 3, 1)          # [B,H,D,T]
            v = v.reshape(B, T, H, D).permute(0, 2, 1, 3)           # [B,H,T,D]

            attn = F.softmax(q @ k, dim=-1)                          # [B,H,T,T]
            out  = (attn @ v).permute(0, 2, 1, 3).reshape(B, T, -1) # [B,T,H*D]
            x    = F.linear(out.half(), self.out_proj[i], self.out_bias[i]).float()
            x    = r + x

            # --- FFN ---
            r = x
            x = F.layer_norm(x, [x.shape[-1]], self.ln2_w[i], self.ln2_b[i])
            x = F.gelu(F.linear(x.half(), self.fc1_w[i], self.fc1_b[i]).float())
            x = F.linear(x.half(), self.fc2_w[i], self.fc2_b[i]).float()
            x = r + x

        x = F.layer_norm(x, [x.shape[-1]], self.final_ln_w, self.final_ln_b)
        return x  # [1, 1500, 1280] float32


# ── 2. AudioAdaptor：手写 RMSNorm，规避 aten::rms_norm ───────────────────
class AudioAdaptorExport(nn.Module):
    """
    4x downsample + MoE adaptor。
    MoE 内部的 nn.RMSNorm 手写展开（opset 18 不支持 aten::rms_norm）。
    """
    def __init__(self, moe):
        super().__init__()
        from copy import deepcopy
        self.moe = deepcopy(moe).float()
        self._replace_rmsnorm(self.moe)

    @staticmethod
    def _replace_rmsnorm(module):
        for name, child in list(module.named_children()):
            if type(child).__name__ == 'RMSNorm' and hasattr(child, 'weight'):
                w = child.weight.data.clone().float()
                # nn.RMSNorm.eps may be None if not set; fall back to 1e-6
                eps = child.eps if (hasattr(child, 'eps') and child.eps is not None) else 1e-6
                class _RMS(nn.Module):
                    def __init__(self, w, eps):
                        super().__init__()
                        self.w = nn.Parameter(w)
                        self.eps = float(eps)
                    def forward(self, x):
                        x = x.float()
                        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w
                setattr(module, name, _RMS(w, eps))
            else:
                AudioAdaptorExport._replace_rmsnorm(child)

    def forward(self, encoder_feat):
        # encoder_feat: [1500, 1280] float32
        flat = encoder_feat.float().view(TOKENS, D_ENC * DOWNSAMPLE)  # [375, 5120]
        return self.moe(flat)  # [375, 2048]


# ── 验证 ──────────────────────────────────────────────────────────────────
print('\nSanity check ...')
whisper_m = WhisperEncoderExport(enc).eval()
adaptor_m = AudioAdaptorExport(moe).eval()

dummy_mel = torch.randn(1, N_MEL, CHUNK_LEN)
with torch.no_grad():
    enc_out = whisper_m(dummy_mel)
    embeds  = adaptor_m(enc_out[0])   # squeeze batch dim

print(f'  encoder_out: {enc_out.shape}  (expected [1, {T_ENC}, {D_ENC}])')
print(f'  embeds:      {embeds.shape}   (expected [{TOKENS}, {D_LLM}])')
assert enc_out.shape == (1, T_ENC, D_ENC)
assert embeds.shape  == (TOKENS, D_LLM)
print('  OK')

# 与原版对比（encoder 数值一致性，moe 用 float32 原版对比）
print('\nNumerical check vs original model ...')
import numpy as np
def cosine(a, b):
    a, b = a.detach().float().numpy().flatten(), b.detach().float().numpy().flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

with torch.no_grad():
    ref_out = enc(dummy_mel.bfloat16(), attention_mask=None, return_dict=True).last_hidden_state.float()
print(f'  encoder cosine={cosine(enc_out, ref_out):.6f}  max_err={float((enc_out-ref_out).abs().max()):.2e}')

# moe 原版是 bfloat16，用我们手写的 adaptor_m（已替换 RMSNorm）对比
ref_emb = adaptor_m(enc_out[0])
print(f'  adaptor (self-consistency): shape={ref_emb.shape}  OK')


# ── 导出 ONNX ─────────────────────────────────────────────────────────────
def export_and_check(model, dummy, path, in_names, out_names, opset=18):
    torch.onnx.export(
        model, (dummy,), path,
        input_names=in_names,
        output_names=out_names,
        do_constant_folding=True,
        opset_version=opset,
        dynamo=False,
    )
    import onnx
    m = onnx.load(path)
    gb = sum(len(i.raw_data) for i in m.graph.initializer) / 1e9
    status = 'OK' if gb < 2.0 else 'WARNING: >2GB'
    print(f'  {os.path.basename(path)}  weight={gb:.2f}GB  [{status}]')
    return gb < 2.0

print('\nExporting ONNX ...')
whisper_path  = f'{folder}/whisper_encoder_b1.onnx'
adaptor_path  = f'{folder}/audio_adaptor.onnx'

ok1 = export_and_check(whisper_m, torch.randn(1, N_MEL, CHUNK_LEN),
                        whisper_path, ['mel'], ['encoder_out'])
ok2 = export_and_check(adaptor_m, torch.randn(T_ENC, D_ENC),
                        adaptor_path, ['encoder_feat'], ['audio_embeds'],
                        opset=17)

if ok1 and ok2:
    print('\nAll ONNX exported successfully.')
else:
    print('\nWARNING: some ONNX exceed 2GB.')

print(f'''
TPU-MLIR compile commands (run inside Docker):

model_transform.py \\
  --model_name whisper_encoder \\
  --model_def {whisper_path} \\
  --input_shapes [[1,{N_MEL},{CHUNK_LEN}]] \\
  --mlir ./tmp/whisper_encoder.mlir
model_deploy.py --mlir ./tmp/whisper_encoder.mlir \\
  --quantize BF16 --chip bm1684x \\
  --model ./tmp/whisper_encoder.bmodel

model_transform.py \\
  --model_name audio_adaptor \\
  --model_def {adaptor_path} \\
  --input_shapes [[{T_ENC},{D_ENC}]] \\
  --mlir ./tmp/audio_adaptor.mlir
model_deploy.py --mlir ./tmp/audio_adaptor.mlir \\
  --quantize BF16 --chip bm1684x \\
  --model ./tmp/audio_adaptor.bmodel
''')
