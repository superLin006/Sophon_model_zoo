#!/usr/bin/env python
"""
Moonshine streaming-small ONNX 导出脚本(方案 B,正式版)
========================================================
目标芯片: BM1684X (TPU-MLIR v1.28.1), 目标精度 F32 + F16
固定输入: 10s 音频 -> C++ 侧 CMVN+Asinh -> x_frames [1,2000,80] -> encoder_out [1,500,620]

拆分方式(与 operator_analysis.md 一致):
- encoder:  x_frames [1,2000,80] f32 -> encoder_out [1,500,620] f32
  (CMVN/Asinh 在 C++ 侧做; 模型内 silu(Linear) -> CausalConv1d x2 -> 10 层 sliding-window attn -> final_norm)
- decoder:  单步自回归, 23 输入 / 21 输出
  输入: token [1,1] int64, encoder_out [1,500,620] f32, cache_len [1] int64,
        past_k_0..9 [1,128,512] f32, past_v_0..9 [1,128,512] f32 (存旋转后 K)
  输出: logits [1,1,32768] f32, new_k_0..9 [1,1,512] f32, new_v_0..9 [1,1,512] f32

关键实现点(详见 operator_analysis.md):
- sliding window mask 用 transformers 语义预计算为常量 buffer
- HF CausalConv1d 内部自带 left_pad(F.pad(4,0)), wrapper 外层不重复 pad
- RoPE 模型内计算 cos/sin(位置=cache_len); causal mask 模型内生成
- MLP GLU 用显式 Slice 替代 chunk(避免 Split num_outputs 与 onnxsim 0.6.3 冲突)
- KV dummy 用列表推导式(不能用 [t]*n 共享引用)
- torch 2.11 Conv 导出缺 kernel_shape, 需 fix_conv_kernel_shape
- 导出后 onnxsim 简化 + opset 统一为 17

额外产物:
- models/log_k.npy + models/log_k.txt: encoder.embedder.comp.log_k 标量(C++ 预处理需要)

用法:
    python export_onnx.py              # 全量导出(encoder + decoder)
    python export_onnx.py --encoder    # 只导出 encoder
    python export_onnx.py --decoder    # 只导出 decoder

环境: sophon-moonshine (conda), torch 2.11.0+cpu, onnx 1.21, onnxsim 0.6.3
"""
import argparse
import os
from collections import Counter

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxsim
from transformers import MoonshineStreamingForConditionalGeneration
from transformers.cache_utils import EncoderDecoderCache, DynamicCache

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "moonshine-streaming-small")
ONNX_DIR = os.path.join(ROOT, "models", "onnx")
AUDIO_0 = os.path.join(ROOT, "test_data", "0.wav")

FRAME_LEN = 80          # 16000 * 5ms / 1000
T = 2000                # 10s -> 2000 帧
T_ENC = 500             # (2000-1)//4+1
MAX_DEC_LEN = 128
N_LAYER, HID, N_HEAD, HEAD_DIM, ROT_DIM = 10, 512, 8, 64, 32


# ============================================================
# Encoder(方案 B)
# ============================================================

def make_sliding_mask(t_enc, left, right, dtype=torch.float32):
    """transformers 语义 sliding window mask(与 HF create_bidirectional_mask 逐元素一致):
    keep: (dist>=0 & dist<left) | (dist<0 & -dist<right), dist = q - k
    返回 [1,1,T,T]: 参与=0.0, 不参与=finfo.min
    """
    q = torch.arange(t_enc)[:, None]
    k = torch.arange(t_enc)[None, :]
    dist = q - k
    keep = ((dist >= 0) & (dist < left)) | ((dist < 0) & (-dist < right))
    mask = torch.where(keep, torch.zeros((), dtype=dtype),
                       torch.tensor(torch.finfo(dtype).min, dtype=dtype))
    return mask.unsqueeze(0).unsqueeze(0)


class EncoderWrapperB(nn.Module):
    """方案 B: 输入 x_frames [1,2000,80](C++ 已做 CMVN+Asinh), 输出 [1,500,620]"""
    def __init__(self, encoder, t_enc=T_ENC):
        super().__init__()
        self.enc = encoder
        for i, (l, r) in enumerate(encoder.config.sliding_windows):
            self.register_buffer(f"mask_{i}", make_sliding_mask(t_enc, l, r), persistent=False)

    def forward(self, x_frames):
        enc = self.enc
        x = F.silu(enc.embedder.linear(x_frames))                 # [1,T,620]
        x = x.transpose(1, 2)                                     # [1,620,T]
        # 注意: HF MoonshineStreamingCausalConv1d.forward 内部自带 left_pad,
        # 不要再外层 F.pad(否则双重 padding -> 输出 503 帧)
        x, _ = enc.embedder.conv1(x); x = F.silu(x)               # [1,1240,1000]
        x, _ = enc.embedder.conv2(x)                              # [1,620,500]
        x = x.transpose(1, 2)                                     # [1,500,620]
        for i, layer in enumerate(enc.layers):
            x = layer(x, attention_mask=getattr(self, f"mask_{i}"))
        return enc.final_norm(x)


# ============================================================
# Decoder(单步)
# ============================================================

class DecoderWrapper(nn.Module):
    """单步 decoder: Sophon 风格(embed gather / pos_emb+proj / RoPE / causal mask 全部进模型)"""
    def __init__(self, model, max_dec_len=MAX_DEC_LEN, t_enc=T_ENC):
        super().__init__()
        dec = model.model.decoder
        self.embed_tokens = dec.embed_tokens        # Embedding(32768, 512)
        self.pos_emb = dec.pos_emb                  # Embedding(4096, 620)
        self.proj = dec.proj                        # Linear(620, 512, bias=False)
        self.layers = dec.layers
        self.norm = dec.norm
        self.proj_out = model.proj_out              # Linear(512, 32768, bias=False)
        self.max_dec_len = max_dec_len
        self.scaling = HEAD_DIM ** -0.5
        with torch.no_grad():
            self.inv_freq = dec.rotary_emb.inv_freq.detach().float()   # [16] 常量
        self.register_buffer("pos_idx", torch.arange(t_enc, dtype=torch.long), persistent=False)   # [500]
        self.register_buffer("mask_arange",
                             torch.arange(max_dec_len + 1, dtype=torch.long).view(1, 1, 1, -1),
                             persistent=False)                                                      # [1,1,1,129]

    def apply_rope(self, q, k, cos, sin):
        """HF apply_rotary_pos_emb 等价(interleaved), 显式 stack/flatten 替代 repeat_interleave
        q,k: [1,8,1,64]; cos,sin: [1,1,32]"""
        half = ROT_DIM // 2
        cos_i = torch.stack([cos[..., :half], cos[..., :half]], dim=-1).flatten(-2)   # [1,1,32]
        sin_i = torch.stack([sin[..., :half], sin[..., :half]], dim=-1).flatten(-2)
        q_rot, q_pass = q[..., :ROT_DIM], q[..., ROT_DIM:]
        k_rot, k_pass = k[..., :ROT_DIM], k[..., ROT_DIM:]
        q_half = torch.stack([-q_rot[..., 1::2], q_rot[..., 0::2]], dim=-1).flatten(-2)
        k_half = torch.stack([-k_rot[..., 1::2], k_rot[..., 0::2]], dim=-1).flatten(-2)
        q_embed = q_rot * cos_i + q_half * sin_i
        k_embed = k_rot * cos_i + k_half * sin_i
        return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)

    def forward(self, token, encoder_out, cache_len, *kv):
        """token [1,1] int64, encoder_out [1,500,620] f32, cache_len [1] int64
        kv: past_k_0..9, past_v_0..9 每个 [1,128,512] f32
        返回: logits [1,1,32768], new_k_0..9, new_v_0..9 每个 [1,1,512]"""
        n = N_LAYER
        past_k = kv[0:n]
        past_v = kv[n:2 * n]

        # 1. encoder 侧预处理: pos_emb(arange(T_enc)) + proj 620->512 (每步重算, 与 HF 一致)
        pe = self.pos_emb(self.pos_idx).unsqueeze(0)      # [1,500,620]
        enc = self.proj(encoder_out + pe)                 # [1,500,512]

        # 2. token embedding (Gather)
        x = self.embed_tokens(token)                      # [1,1,512]

        # 3. RoPE: 由 cache_len(位置) 在模型内计算 cos/sin
        pos = cache_len.to(torch.float32).unsqueeze(-1)   # [1,1]
        freqs = pos * self.inv_freq.unsqueeze(0)          # [1,16]
        emb = torch.cat([freqs, freqs], dim=-1)           # [1,32]
        cos = emb.cos().unsqueeze(1)                      # [1,1,32]
        sin = emb.sin().unsqueeze(1)

        # 4. causal mask: 过去位置 0..cache_len-1 有效 + 尾部当前 token 槽(max_dec_len)有效
        #    (当前 token 的 k/v 拼接在缓存末尾, 而非缓存开头!)
        #    ★ 不能用 `lt | eq`(Or 算子): TPU-MLIR 1.28.1 convert_or_op 只认节点
        #    输出, 常量输入(无论 initializer 还是 Constant 节点)都报 operand not
        #    found。用嵌套 Where 表达同样的布尔语义(实测逐位等价)。
        lt = self.mask_arange < cache_len.view(1, 1, 1, 1)        # [1,1,1,129] bool 动态
        tail = self.mask_arange == self.max_dec_len               # 常量 bool(尾部槽恒有效)
        tail_mask = torch.where(tail, torch.zeros((), dtype=encoder_out.dtype),
                                torch.tensor(-1e9, dtype=encoder_out.dtype))
        mask = torch.where(lt, torch.zeros((), dtype=encoder_out.dtype), tail_mask)

        new_keys, new_values = [], []
        for i, layer in enumerate(self.layers):
            # ---- self-attn ----
            h = layer.input_layernorm(x)
            q = layer.self_attn.q_proj(h).view(1, 1, N_HEAD, HEAD_DIM).transpose(1, 2)   # [1,8,1,64]
            k = layer.self_attn.k_proj(h).view(1, 1, N_HEAD, HEAD_DIM).transpose(1, 2)
            v = layer.self_attn.v_proj(h).view(1, 1, N_HEAD, HEAD_DIM).transpose(1, 2)
            q, k = self.apply_rope(q, k, cos, sin)
            k_new = k.transpose(1, 2).reshape(1, 1, -1)   # [1,1,512] 旋转后 K(与 HF cache 格式一致)
            v_new = v.transpose(1, 2).reshape(1, 1, -1)   # [1,1,512]
            k_full = torch.cat([past_k[i], k_new], dim=1)  # [1,129,512]
            v_full = torch.cat([past_v[i], v_new], dim=1)
            kf = k_full.view(1, self.max_dec_len + 1, N_HEAD, HEAD_DIM).transpose(1, 2)
            vf = v_full.view(1, self.max_dec_len + 1, N_HEAD, HEAD_DIM).transpose(1, 2)
            w = F.softmax((q * self.scaling) @ kf.transpose(-2, -1) + mask,
                          dim=-1, dtype=torch.float32).to(q.dtype)
            o = (w @ vf).transpose(1, 2).reshape(1, 1, -1)
            x = x + layer.self_attn.o_proj(o)

            # ---- cross-attn (无 mask: 固定 10s 全有效) ----
            h = layer.post_attention_layernorm(x)
            qc = layer.encoder_attn.q_proj(h).view(1, 1, N_HEAD, HEAD_DIM).transpose(1, 2)
            kc = layer.encoder_attn.k_proj(enc).view(1, T_ENC, N_HEAD, HEAD_DIM).transpose(1, 2)
            vc = layer.encoder_attn.v_proj(enc).view(1, T_ENC, N_HEAD, HEAD_DIM).transpose(1, 2)
            wc = F.softmax((qc * self.scaling) @ kc.transpose(-2, -1),
                           dim=-1, dtype=torch.float32).to(qc.dtype)
            oc = (wc @ vc).transpose(1, 2).reshape(1, 1, -1)
            x = x + layer.encoder_attn.o_proj(oc)

            # ---- MLP (silu GLU, 显式 Slice 替代 chunk 避免 Split 算子) ----
            y = layer.mlp.fc1(layer.final_layernorm(x))   # [1,1,4096]
            hidden, gate = y[..., :2048], y[..., 2048:]
            x = x + layer.mlp.fc2(F.silu(gate) * hidden)

            new_keys.append(k_new)
            new_values.append(v_new)

        x = self.norm(x)
        logits = self.proj_out(x)                        # [1,1,32768]
        return (logits, *new_keys, *new_values)          # 21 个输出


# ============================================================
# ONNX 修复工具
# ============================================================

def fix_conv_kernel_shape(model):
    """torch 2.11 导出 Conv 不写 kernel_shape, TPU-MLIR model_transform 会报错。
    从权重 initializer 推断 kernel_shape 补上(whisper 已验证的修复)。"""
    init_map = {init.name: init for init in model.graph.initializer}
    fixed = 0
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        if any(a.name == "kernel_shape" for a in node.attribute):
            continue
        weight_name = node.input[1]
        if weight_name not in init_map:
            continue
        kernel_shape = list(init_map[weight_name].dims[2:])
        node.attribute.append(onnx.helper.make_attribute("kernel_shape", kernel_shape))
        fixed += 1
    print(f"  [fix] Conv 补 kernel_shape {fixed} 个")
    return model


def export_onnx(module, dummy_inputs, path, input_names, output_names):
    """导出 + onnxsim 简化 + opset 统一 17 + Conv 修复 + checker 校验"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(module, dummy_inputs, path,
                          input_names=input_names, output_names=output_names,
                          opset_version=17, do_constant_folding=True)
    print(f"  -> {path} ({os.path.getsize(path)/1e6:.1f} MB)")

    m = onnx.load(path)
    m2, ok = onnxsim.simplify(m)
    if not ok:
        raise RuntimeError(f"onnxsim 失败: {path}")
    if m2.opset_import[0].version > 17:          # onnxsim 可能升到 18, 统一回 17
        m2.opset_import[0].version = 17
        print("  [fix] opset -> 17")
    m2 = fix_conv_kernel_shape(m2)
    # TPU-MLIR convert_or_op 不处理常量输入, 图中不允许出现 Or 节点
    or_nodes = [n for n in m2.graph.node if n.op_type == "Or"]
    assert not or_nodes, f"图中存在 Or 节点(TPU-MLIR 1.28.1 无法转换): {or_nodes}"
    onnx.checker.check_model(m2)
    print("  checker: OK")

    sim_path = path.replace(".onnx", "_sim.onnx")
    onnx.save(m2, sim_path)
    for raw_path in (path, path + ".data"):
        if os.path.exists(raw_path):
            os.remove(raw_path)
    ops = Counter(n.op_type for n in m2.graph.node)
    print(f"  sim 算子({len(m2.graph.node)} 节点): {dict(sorted(ops.items()))}")
    print(f"  sim 输入: {[i.name + str([d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m2.graph.input]}")
    print(f"  sim 输出: {[o.name + str([d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m2.graph.output]}")
    return sim_path


# ============================================================
# 音频加载与 HF ground truth
# ============================================================

def load_audio_padded(path, n_samples=160000):
    """加载音频(非16k重采样), 尾部补零到 n_samples"""
    import torchaudio
    audio, sr = sf.read(path, dtype="float32")
    if sr != 16000:
        audio = torchaudio.functional.resample(
            torch.from_numpy(audio).unsqueeze(0), sr, 16000).squeeze(0).numpy()
    out = np.zeros(n_samples, np.float32)
    out[:len(audio)] = audio
    return out


def hf_embedder_frontend(enc, input_values):
    """复刻 HF embedder 前两步(CMVN+Asinh), 得到方案 B 的输入 x_frames(C++ 预处理目标)"""
    x = input_values.reshape(input_values.shape[0], -1, FRAME_LEN)
    x = enc.embedder.cmvn(x)
    x = enc.embedder.comp(x)
    return x


def hf_decoder_loop(model, enc_out, max_steps=72):
    """HF 官方 decoder 循环(DynamicCache), 逐步 logits + token
    注意: HF decoder.forward 内 `encoder_hidden_states += pos_emb` 原地修改
    传入 tensor, 必须每步传干净副本(否则从第 2 步起双重 pos_emb)"""
    past = EncoderDecoderCache(DynamicCache(), DynamicCache())
    logits_list, ids = [], []
    next_ids = torch.tensor([[1]], dtype=torch.long)
    for _ in range(max_steps):
        out = model.model.decoder(input_ids=next_ids,
                                  encoder_hidden_states=enc_out.clone(),
                                  past_key_values=past, use_cache=True)
        logits = model.proj_out(out.last_hidden_state[:, -1, :])
        nid = int(logits.argmax(-1).item())
        logits_list.append(logits[0].float())
        ids.append(nid)
        if nid == 2:
            break
        next_ids = torch.tensor([[nid]], dtype=torch.long)
    return torch.stack(logits_list), ids


# ============================================================
# 主流程
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Moonshine streaming-small ONNX 导出(方案 B)")
    ap.add_argument("--encoder", action="store_true", help="只导出 encoder")
    ap.add_argument("--decoder", action="store_true", help="只导出 decoder")
    ap.add_argument("--no-verify", action="store_true", help="跳过导出前的数值验证")
    args = ap.parse_args()
    do_enc = args.encoder or not args.decoder
    do_dec = args.decoder or not args.encoder

    print(f"[Load] {MODEL_DIR}")
    model = MoonshineStreamingForConditionalGeneration.from_pretrained(MODEL_DIR)
    model.eval()
    model.config._attn_implementation = "eager"   # 强制 eager attention 路径

    # ---- 测试音频: 0.wav 尾部补零到 10s(与 C++ 部署输入一致) ----
    audio = load_audio_padded(AUDIO_0)
    input_values = torch.from_numpy(audio).unsqueeze(0)      # [1,160000]
    attn_mask = torch.ones(1, 160000, dtype=torch.long)

    with torch.no_grad():
        enc = model.model.encoder
        hf_out = enc(input_values=input_values, attention_mask=attn_mask).last_hidden_state
        print(f"[Enc] HF encoder_out: {tuple(hf_out.shape)}")
        x_frames = hf_embedder_frontend(enc, input_values)   # [1,2000,80]
        wrapB = EncoderWrapperB(enc).eval()
        outB = wrapB(x_frames)
        diff = (outB - hf_out).abs().max().item()
        print(f"[Verify] EncoderWrapperB vs HF: max_abs={diff:.3e}")
        assert diff < 1e-4, f"encoder wrapper 与 HF 差异过大: {diff}"
        del hf_out   # 释放内存

        # ---- log_k 导出(C++ 预处理需要) ----
        log_k = float(enc.embedder.comp.log_k.detach())
        np.save(os.path.join(ROOT, "models", "log_k.npy"), np.float32(log_k))
        with open(os.path.join(ROOT, "models", "log_k.txt"), "w") as f:
            f.write(f"{log_k:.10f}\n")
        print(f"[Asset] log_k = {log_k:.10f} -> models/log_k.npy + log_k.txt")

        if do_enc:
            print("\n[Export] encoder(方案 B)...")
            path = os.path.join(ONNX_DIR, "moonshine_encoder.onnx")
            export_onnx(wrapB, x_frames, path, ["x_frames"], ["encoder_out"])

        if do_dec:
            print("\n[Export] decoder(单步)...")
            wrapD = DecoderWrapper(model).eval()
            dummy = (torch.tensor([[1]], dtype=torch.long), outB,
                     torch.tensor([0], dtype=torch.long),
                     *[torch.zeros(1, MAX_DEC_LEN, HID) for _ in range(2 * N_LAYER)])
            input_names = ["token", "encoder_out", "cache_len"]
            input_names += [f"past_k_{i}" for i in range(N_LAYER)]
            input_names += [f"past_v_{i}" for i in range(N_LAYER)]
            output_names = ["logits"]
            output_names += [f"new_k_{i}" for i in range(N_LAYER)]
            output_names += [f"new_v_{i}" for i in range(N_LAYER)]
            path = os.path.join(ONNX_DIR, "moonshine_decoder.onnx")
            export_onnx(wrapD, dummy, path, input_names, output_names)

            if not args.no_verify:
                print("\n[Verify] decoder 逐步自回归 (wrapper vs HF 修正循环)...")
                hf_logits, hf_ids = hf_decoder_loop(model, outB)
                cache_k = [torch.zeros(1, MAX_DEC_LEN, HID) for _ in range(N_LAYER)]
                cache_v = [torch.zeros(1, MAX_DEC_LEN, HID) for _ in range(N_LAYER)]
                wrap_logits, wrap_ids = [], []
                nid = 1
                for step in range(72):
                    outs = wrapD(torch.tensor([[nid]], dtype=torch.long), outB,
                                 torch.tensor([step], dtype=torch.long),
                                 *(cache_k + cache_v))
                    logits, nk, nv = outs[0], outs[1:1 + N_LAYER], outs[1 + N_LAYER:]
                    nid = int(logits.argmax(-1).item())
                    wrap_logits.append(logits[0, 0].float())
                    wrap_ids.append(nid)
                    if nid == 2:
                        break
                    for i in range(N_LAYER):
                        cache_k[i][:, step:step + 1] = nk[i]
                        cache_v[i][:, step:step + 1] = nv[i]
                wrap_logits = torch.stack(wrap_logits)
                n = min(len(hf_ids), len(wrap_ids))
                d = (wrap_logits[:n] - hf_logits[:n]).abs().max().item()
                ok = wrap_ids == hf_ids
                print(f"  HF {len(hf_ids)} 步 / wrapper {len(wrap_ids)} 步")
                print(f"  logits max_abs={d:.3e}, token 序列一致: {ok}")
                print(f"  tokens: {wrap_ids}")
                assert ok, f"decoder wrapper token 序列与 HF 不一致"
                assert d < 1e-3, f"decoder logits 差异过大: {d}"

    print("\n[Done] 导出完成: ")
    for f in sorted(os.listdir(ONNX_DIR)):
        print(f"  models/onnx/{f} ({os.path.getsize(os.path.join(ONNX_DIR, f))/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
