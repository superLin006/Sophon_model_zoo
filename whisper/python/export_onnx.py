"""
Whisper BM1684 ONNX 导出脚本
目标芯片: BM1684 (FP32)
转换链路: PyTorch → ONNX → (Docker) → bmodel

拆分方式:
  - encoder:   mel [1, 80, 3000] → audio_features [1, 1500, n_state]
  - decoder:   单步自回归，含 KV Cache 输入输出（与 MTK 方案一致）

与 MTK 版本的关键区别（Sophon 算子更强，更多逻辑放进模型）:
  MTK C++ 侧做:  token embedding lookup、position embedding lookup、attention mask 生成
  Sophon 模型内: 全部放进来（Gather/Where/Concat 均支持）

Decoder 输入:
  token           [1, 1]                int64   当前 token id（模型内做 Embedding）
  audio_features  [1, 1500, n_state]    float32 encoder 输出（cross attention 用）
  cache_len       [1]                   int32   当前已填充的 KV Cache 长度（用于 position + mask）
  past_self_k_*   [1, 448, n_state]     float32 x n_layer  自注意力 KV Cache
  past_self_v_*   [1, 448, n_state]     float32 x n_layer
  cross_k_*       [1, 1500, n_state]    float32 x n_layer  交叉注意力 KV（首次后固定不变）
  cross_v_*       [1, 1500, n_state]    float32 x n_layer

Decoder 输出:
  logits          [1, 1, n_vocab]       float32
  new_self_k_*    [1, 1, n_state]       float32 x n_layer  本步新增 KV，C++ 写入 cache 对应位置
  new_self_v_*    [1, 1, n_state]       float32 x n_layer
  new_cross_k_*   [1, 1500, n_state]    float32 x n_layer  仅首步有效，后续忽略
  new_cross_v_*   [1, 1500, n_state]    float32 x n_layer

用法:
    python export_onnx.py --model base
    python export_onnx.py --model small
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper
import onnx
import onnxsim

PADDING_SIZE = 448   # max KV cache length (n_text_ctx)


# ============================================================
# Encoder
# ============================================================

class EncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, mel):
        # BM1684 不支持 Erf，手动展开 AudioEncoder.forward 中的 F.gelu 为 tanh 近似
        enc = self.encoder
        x = F.gelu(enc.conv1(mel), approximate='tanh')
        x = F.gelu(enc.conv2(x), approximate='tanh')
        x = x.permute(0, 2, 1)
        x = (x + enc.positional_embedding).to(x.dtype)
        for block in enc.blocks:
            x = block(x)
        x = enc.ln_post(x)
        return x


def export_encoder(model, output_path: str):
    print(f"\n[Encoder] 导出 -> {output_path}")
    wrapper = EncoderWrapper(model.encoder)
    wrapper.eval()

    dummy_mel = torch.zeros(1, model.dims.n_mels, 3000)

    torch.onnx.export(
        wrapper,
        dummy_mel,
        output_path,
        input_names=["mel"],
        output_names=["audio_features"],
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"[Encoder] 导出完成")
    return output_path


# ============================================================
# Decoder（单步，含完整 KV Cache 管理，与 MTK 方案对齐）
# ============================================================

class DecoderWrapper(nn.Module):
    """
    单步 decoder，Sophon 版本将 MTK 在 C++ 侧做的三件事移入模型：
      1. token embedding lookup（Gather）
      2. position embedding lookup（Gather，按 cache_len 索引）
      3. attention mask 生成（Where + 广播）

    KV Cache 策略（与 MTK 完全一致）：
      - past_self_k/v: [1, PADDING_SIZE, n_state]，固定形状，C++ 侧管理写入
      - 本步只输出 new_self_k/v [1, 1, n_state]，C++ 按 cache_len 写入对应位置
      - cross_k/v: 首步计算，后续透传（C++ 侧复用）
    """
    def __init__(self, decoder, n_layer, n_head, padding_size):
        super().__init__()
        self.decoder = decoder
        self.n_layer = n_layer
        self.n_head = n_head
        self.padding_size = padding_size

    def forward(self, token, audio_features, pos_emb, self_attn_mask, *kv_args):
        """
        token:          [1, 1]                        int64
        audio_features: [1, n_audio_ctx, n_state]     float32
        pos_emb:        [1, 1, n_state]               float32  当前步 positional embedding，C++ 按步切片
        self_attn_mask: [1, 1, 1, padding_size+1]     float32  C++ 生成（有效位=0，无效位=-1e9）
        kv_args:        past_self_k x n_layer, past_self_v x n_layer,
                        cross_k x n_layer, cross_v x n_layer
        """
        n = self.n_layer
        past_self_k  = list(kv_args[0:n])
        past_self_v  = list(kv_args[n:2*n])
        cross_k      = list(kv_args[2*n:3*n])
        cross_v      = list(kv_args[3*n:4*n])

        dec = self.decoder

        # 1. Token embedding（Gather，Sophon 支持）
        x = dec.token_embedding(token)   # [1, 1, n_state]

        # 2. Position embedding（C++ 侧按步切片后传入，与 MTK 一致）
        x = x + pos_emb
        x = x.to(audio_features.dtype)

        mask_full = self_attn_mask

        new_self_keys, new_self_values = [], []
        new_cross_keys, new_cross_values = [], []

        for i, block in enumerate(dec.blocks):
            n_head = block.attn.n_head
            scale = (x.shape[-1] // n_head) ** -0.25

            # --- Self attention ---
            attn_ln_x = block.attn_ln(x)
            q   = block.attn.query(attn_ln_x)
            k_new = block.attn.key(attn_ln_x)    # [1, 1, n_state]
            v_new = block.attn.value(attn_ln_x)  # [1, 1, n_state]
            new_self_keys.append(k_new)
            new_self_values.append(v_new)

            # 拼接 past cache + 当前新 k/v，然后在末尾 padding 一个占位（与 mask 对应）
            # past_self_k[i]: [1, PADDING_SIZE, n_state]
            # cat: [1, PADDING_SIZE+1, n_state]（末尾是 k_new，但 mask 只看 padding_size 位置）
            k_full = torch.cat([past_self_k[i], k_new], dim=1)  # [1, PADDING_SIZE+1, n_state]
            v_full = torch.cat([past_self_v[i], v_new], dim=1)

            head_dim = x.shape[-1] // n_head
            q_ = q.reshape(1, 1, n_head, head_dim).permute(0, 2, 1, 3) * scale
            k_ = k_full.reshape(1, self.padding_size+1, n_head, head_dim).permute(0, 2, 3, 1) * scale
            v_ = v_full.reshape(1, self.padding_size+1, n_head, head_dim).permute(0, 2, 1, 3)

            qk = q_ @ k_  # [1, n_head, 1, PADDING_SIZE+1]
            qk = qk + mask_full
            w  = F.softmax(qk.float(), dim=-1).to(q.dtype)
            attn_out = (w @ v_).permute(0, 2, 1, 3).flatten(start_dim=2)
            attn_out = block.attn.out(attn_out)
            x = x + attn_out

            # --- Cross attention ---
            cross_ln_x = block.cross_attn_ln(x)
            q_c = block.cross_attn.query(cross_ln_x)
            # cross k/v 首步由模型计算并输出，后续由 C++ 传入缓存值
            k_c = block.cross_attn.key(audio_features)
            v_c = block.cross_attn.value(audio_features)
            new_cross_keys.append(k_c)
            new_cross_values.append(v_c)

            # 实际 attention 用传入的 cross_k/v（C++ 首步后固定复用）
            n_audio_ctx = audio_features.shape[1]
            q_c_ = q_c.reshape(1, 1, n_head, head_dim).permute(0, 2, 1, 3) * scale
            k_c_ = cross_k[i].reshape(1, n_audio_ctx, n_head, head_dim).permute(0, 2, 3, 1) * scale
            v_c_ = cross_v[i].reshape(1, n_audio_ctx, n_head, head_dim).permute(0, 2, 1, 3)
            qk_c = q_c_ @ k_c_
            w_c  = F.softmax(qk_c.float(), dim=-1).to(q_c.dtype)
            cross_out = (w_c @ v_c_).permute(0, 2, 1, 3).flatten(start_dim=2)
            cross_out = block.cross_attn.out(cross_out)
            x = x + cross_out

            # FFN
            x = x + block.mlp(block.mlp_ln(x))

        x = dec.ln(x)
        logits = (x @ dec.token_embedding.weight.T).float()  # [1, 1, n_vocab]

        return tuple(
            [logits]
            + new_self_keys + new_self_values
            + new_cross_keys + new_cross_values
        )


def export_decoder(model, output_path: str):
    print(f"\n[Decoder] 导出 -> {output_path}")
    dims = model.dims
    n_layer    = dims.n_text_layer
    n_state    = dims.n_text_state
    n_head     = dims.n_text_head
    n_audio_ctx = dims.n_audio_ctx

    wrapper = DecoderWrapper(model.decoder, n_layer, n_head, PADDING_SIZE)
    wrapper.eval()

    dummy_token         = torch.zeros(1, 1, dtype=torch.long)
    dummy_audio         = torch.zeros(1, n_audio_ctx, n_state)
    dummy_pos_emb       = torch.zeros(1, 1, n_state)
    dummy_self_attn_mask = torch.zeros(1, 1, 1, PADDING_SIZE + 1)

    dummy_past_self_k = [torch.zeros(1, PADDING_SIZE, n_state) for _ in range(n_layer)]
    dummy_past_self_v = [torch.zeros(1, PADDING_SIZE, n_state) for _ in range(n_layer)]
    dummy_cross_k     = [torch.zeros(1, n_audio_ctx, n_state)  for _ in range(n_layer)]
    dummy_cross_v     = [torch.zeros(1, n_audio_ctx, n_state)  for _ in range(n_layer)]

    inputs = (
        dummy_token, dummy_audio, dummy_pos_emb, dummy_self_attn_mask,
        *dummy_past_self_k, *dummy_past_self_v,
        *dummy_cross_k, *dummy_cross_v,
    )

    input_names = ["token", "audio_features", "pos_emb", "self_attn_mask"]
    for i in range(n_layer): input_names.append(f"past_self_k_{i}")
    for i in range(n_layer): input_names.append(f"past_self_v_{i}")
    for i in range(n_layer): input_names.append(f"cross_k_{i}")
    for i in range(n_layer): input_names.append(f"cross_v_{i}")

    output_names = ["logits"]
    for i in range(n_layer): output_names.append(f"new_self_k_{i}")
    for i in range(n_layer): output_names.append(f"new_self_v_{i}")
    for i in range(n_layer): output_names.append(f"new_cross_k_{i}")
    for i in range(n_layer): output_names.append(f"new_cross_v_{i}")

    torch.onnx.export(
        wrapper,
        inputs,
        output_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"[Decoder] 导出完成")
    return output_path


# ============================================================
# ONNX 验证 + 简化
# ============================================================

def _fix_conv_kernel_shape(model):
    """新版 torch.onnx exporter 可能漏写 Conv 的 kernel_shape 属性，从 weight 推导补全。"""
    init_map = {init.name: init for init in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        if any(attr.name == "kernel_shape" for attr in node.attribute):
            continue
        if len(node.input) < 2:
            continue
        weight_name = node.input[1]
        if weight_name not in init_map:
            continue
        kernel_shape = list(init_map[weight_name].dims[2:])
        node.attribute.append(onnx.helper.make_attribute("kernel_shape", kernel_shape))
    return model


def verify_and_simplify(onnx_path: str):
    print(f"\n[验证] {os.path.basename(onnx_path)}")
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print(f"  ✅ ONNX 格式验证通过")

    print(f"  简化中 (onnxsim)...")
    model_sim, ok = onnxsim.simplify(model)
    if ok:
        model_sim = _fix_conv_kernel_shape(model_sim)
        sim_path = onnx_path.replace(".onnx", "_sim.onnx")
        onnx.save(model_sim, sim_path)
        orig_size = os.path.getsize(onnx_path) / 1024 / 1024
        sim_size  = os.path.getsize(sim_path)  / 1024 / 1024
        print(f"  ✅ 简化完成: {orig_size:.1f}MB → {sim_size:.1f}MB -> {sim_path}")
        return sim_path
    else:
        print(f"  ⚠️  简化失败，使用原始 ONNX")
        return onnx_path


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Whisper BM1684 ONNX 导出")
    parser.add_argument("--model", default="base",
                        choices=["tiny", "base", "small", "medium"])
    parser.add_argument("--output_dir", default="../models/onnx")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Whisper BM1684 ONNX 导出")
    print(f"  模型: whisper-{args.model}  精度: FP32  芯片: BM1684")
    print(f"{'='*60}")

    print(f"\n[加载模型] whisper-{args.model} ...")
    model = whisper.load_model(args.model)
    model.eval()

    # BM1684 不支持 Erf，将所有 nn.GELU 改为 tanh 近似
    for m in model.modules():
        if isinstance(m, nn.GELU):
            m.approximate = 'tanh'

    dims = model.dims
    print(f"  n_audio_state={dims.n_audio_state}, n_audio_layer={dims.n_audio_layer}")
    print(f"  n_text_state={dims.n_text_state},   n_text_layer={dims.n_text_layer}")
    print(f"  n_vocab={dims.n_vocab}")

    prefix = os.path.join(args.output_dir, f"whisper_{args.model}")

    enc_path = export_encoder(model, f"{prefix}_encoder.onnx")
    verify_and_simplify(enc_path)

    dec_path = export_decoder(model, f"{prefix}_decoder.onnx")
    verify_and_simplify(dec_path)

    print(f"\n{'='*60}")
    print(f"  导出完成，共 2 个 ONNX 文件:")
    print(f"  1. whisper_{args.model}_encoder_sim.onnx")
    print(f"  2. whisper_{args.model}_decoder_sim.onnx")
    print(f"\n  下一步: 在 Docker 容器内运行 gen_bmodel.sh 转换为 bmodel")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
