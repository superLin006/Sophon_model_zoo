#!/usr/bin/env python3
"""
导出 Qwen3-ASR-0.6B 的 Audio Encoder + Projector 为单个 ONNX。

输入:  mel [1, 128, 3000]  float32  （30s @16k，10ms/帧；固定长度，mel 帧须 3000 的倍数语义为 100）
输出:  audio_embeds [390, 1024]  float32
        （30 chunks × 13 帧/chunk = 390 帧，逐帧过 projector）

实现说明（对照 transformers 5.14 源码 Qwen3ASREncoder.forward）：
  - 固定 T=3000 且全有效（无 pad）：cu_seqlens 为常数 [0,104,208,312,390]
    （n_window_ratio=8, window_aftercnn=13*8=104, 390=104*3+78）
  - 跳过动态 valid_indices 打包（nonzero 动态 shape，TPU-MLIR 不支持）→ 全有效时等价于直接 reshape
  - attention 走 eager 路径（非 flash）：按 cu_seqlens 静态 split 窗口，窗口内非因果 attention
  - 权重 fp16 存储（防 ONNX 2GB protobuf），计算 fp16（TPU-MLIR F16 量化对齐）
  - 引用真实子模块（conv/attention/FFN/ln_post/projector），仅重写 forward 控制流

用法（sophon-qwen3-asr conda env）:
  python export_encoder_onnx.py --model_path ../models
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='../models')
parser.add_argument('--out_dir', type=str, default='./tmp/onnx')
parser.add_argument('--num_mel_frames', type=int, default=3000)
args = parser.parse_args()

from transformers import AutoModelForMultimodalLM

T_MEL = args.num_mel_frames          # 3000 mel 帧 = 30s（离线）；500 = 5s（流式窗口，对齐官方 unfixed_chunk_num=4+1）
assert T_MEL % 100 == 0, 'T must be multiple of 100'
NUM_CHUNKS = T_MEL // 100
TOKENS_PER_CHUNK = 13                # 100 帧 3 次 stride2 conv 后
T_ENC = NUM_CHUNKS * TOKENS_PER_CHUNK
# cu_seqlens 窗口划分（T 全有效时）：window_aftercnn = 13*8 = 104，窗口 = 104 重复 + 余数
WINDOW = 104
n_win = T_ENC // WINDOW
rem = T_ENC % WINDOW
WINDOWS = [WINDOW] * n_win + ([rem] if rem else [])
assert sum(WINDOWS) == T_ENC
print(f'WINDOWS={WINDOWS}')

print(f'Loading model from {args.model_path} ...')
full_model = AutoModelForMultimodalLM.from_pretrained(
    args.model_path, dtype=torch.bfloat16, torch_dtype=torch.bfloat16).eval()
for p in full_model.parameters():
    p.requires_grad_(False)

enc = full_model.model.audio_tower       # Qwen3ASREncoder
proj = full_model.model.multi_modal_projector
attn0 = enc.layers[0].self_attn
N_HEADS = attn0.num_heads
HEAD_DIM = attn0.head_dim
D_MODEL = enc.config.d_model             # 896
OUT_DIM = enc.config.output_dim          # 1024
SCALING = attn0.scaling
CONV_CH = enc.config.downsample_hidden_size  # 480
print(f'chunks={NUM_CHUNKS} T_ENC={T_ENC} heads={N_HEADS} head_dim={HEAD_DIM} '
      f'd_model={D_MODEL} out={OUT_DIM} scaling={SCALING:.4f}')

folder = args.out_dir
os.makedirs(folder, exist_ok=True)


class EncoderExport(nn.Module):
    """引用真实子模块权重（fp16），重写 forward：固定窗口 + 无动态打包。"""
    def __init__(self, enc, proj):
        super().__init__()
        self.conv2d1 = enc.conv2d1.half()
        self.conv2d2 = enc.conv2d2.half()
        self.conv2d3 = enc.conv2d3.half()
        self.conv_out = enc.conv_out.half()
        # 正弦位置编码（buffer，persistent=False 不在 state dict，但模块实例在）
        self.pos_emb = enc.positional_embedding
        self.layers = enc.layers.half()
        self.ln_post = enc.ln_post.half()
        self.proj_linear_1 = proj.linear_1.half()
        self.proj_linear_2 = proj.linear_2.half()
        self.num_heads = N_HEADS
        self.head_dim = HEAD_DIM
        self.scaling = SCALING

    def window_attn(self, attn, h):
        """单层非因果窗口 attention（对齐原生 eager 路径：transpose + cat dim=1，帧-major）。

        注意：原生 split 后 attention 输出 transpose(1,2)（帧移到 dim1），cat(dim=1)，
        最后 reshape(seq_len, -1) 的每行 = 一帧的 14 头拼接。若 cat 在 head 维（dim=2）
        则 reshape 后行序为 head-major，完全错乱（cosine 只有 0.36）。"""
        seq_len = h.shape[0]
        q = attn.q_proj(h).reshape(seq_len, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
        k = attn.k_proj(h).reshape(seq_len, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
        v = attn.v_proj(h).reshape(seq_len, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
        outs = []
        for qs, ks, vs in zip(torch.split(q, WINDOWS, dim=2),
                              torch.split(k, WINDOWS, dim=2),
                              torch.split(v, WINDOWS, dim=2)):
            w = torch.matmul(qs, ks.transpose(-1, -2)) * self.scaling
            w = F.softmax(w.float().to(torch.float32), dim=-1).to(qs.dtype)
            outs.append(torch.matmul(w, vs).transpose(1, 2).contiguous())
        o = torch.cat(outs, dim=1)
        return attn.out_proj(o.reshape(seq_len, -1).contiguous())

    def layer(self, layer, h):
        residual = h
        h = layer.self_attn_layer_norm(h)
        h = self.window_attn(layer.self_attn, h)
        h = residual + h
        residual = h
        h = layer.final_layer_norm(h)
        h = layer.fc2(layer.activation_fn(layer.fc1(h)))
        h = residual + h
        return h

    def forward(self, mel):
        # mel: [1, 128, 3000] → chunks [30, 1, 128, 100]
        chunked = (mel.view(1, mel.shape[1], NUM_CHUNKS, T_MEL // NUM_CHUNKS)
                     .permute(0, 2, 1, 3)
                     .reshape(NUM_CHUNKS, 1, mel.shape[1], T_MEL // NUM_CHUNKS))
        conv_out = F.gelu(self.conv2d1(chunked))
        conv_out = F.gelu(self.conv2d2(conv_out))
        conv_out = F.gelu(self.conv2d3(conv_out))          # [30, 480, 13]
        total_chunks, conv_ch, freq_bins, time_steps = conv_out.size()
        conv_out = self.conv_out(
            conv_out.permute(0, 3, 1, 2).contiguous()
                    .view(total_chunks, time_steps, conv_ch * freq_bins)
        )                                                  # [30, 13, 896]
        conv_out = conv_out + self.pos_emb.positional_embedding[:time_steps]
        h = conv_out.reshape(-1, D_MODEL)                  # [390, 896]
        for layer in self.layers:
            h = self.layer(layer, h)
        h = self.ln_post(h)                                # [390, 896]
        h = self.proj_linear_1(h)
        h = self.proj_linear_2(F.gelu(h))                  # [390, 1024]
        return h


model = EncoderExport(enc, proj)
# pos_emb buffer 转 fp16（正弦位置编码 buffer 在 .half() 时未随 layers 转；
# 注意不能整体 .float()——那会把 half 权重转回 float32 导致与 half 输入不匹配）
model.pos_emb.half()

dummy = torch.randn(1, 128, T_MEL, dtype=torch.float16)
torch.onnx.export(
    model, (dummy,),
    f'{folder}/qwen3_asr_encoder.onnx',
    input_names=['mel'],
    output_names=['audio_embeds'],
    do_constant_folding=True,
    opset_version=17,
    dynamo=False,
)
print(f'Exported {folder}/qwen3_asr_encoder.onnx')
