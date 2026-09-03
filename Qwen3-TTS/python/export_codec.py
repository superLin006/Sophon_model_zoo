#!/usr/bin/env python
# codec decoder 单图 ONNX 导出（Qwen3TTSTokenizerV2Model 的 decoder 部分）。
#
# 输入  codes [1, 16, T]   —— 16 路 codebook 的离散 id（T 帧，12.5Hz 帧率）
# 输出  wav   [1, 1, T*1920] —— 24kHz 波形（decode_upsample_rate=1920，Clamp[-1,1]）
#
# 与 gen_final.sh 中 codec_decoder_T325.onnx 的编译参数对齐：
#   model_transform: --input_shapes [[1,16,325]]
#   model_deploy:    --quantize F16 --disable_layer_group
#
# 用法（sophon-qwen3-tts conda 环境）:
#   python export_codec.py                    # 默认导出 T=325，含 onnxruntime 数值自检
#   python export_codec.py --seq-len 500 --skip-check
import argparse
import os

import numpy as np
import torch
from torch import nn

from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 模型权重目录：必须通过 --model-path 或环境变量 QWEN3_TTS_MODEL 提供，禁止硬编码本机路径
_DEF_MODEL = os.environ.get("QWEN3_TTS_MODEL", "")
_DEF_OUT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "models", "onnx"))


class CodecDecoder(nn.Module):
    """codes [B, Q, T] -> decoder -> wav [B, 1, T*1920]。

    复制 Qwen3TTSTokenizerV2Decoder.forward 的完整前向，唯一区别：
    pre_transformer 不再让 transformers 4.57 用 vmap（_vmap_for_bhqkv）自动构造因果
    mask（静态 torch.onnx.export 不支持），而是传入预先算好的 attention_mask dict。

    export_talker.py 对 talker 层用同样的思路（预计算 triu mask 传 attention_mask）。
    """

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, codes):            # [B, Q, T]
        d = self.decoder
        if codes.shape[1] != d.config.num_quantizers:
            raise ValueError(f"Expected {d.config.num_quantizers} codebooks, got {codes.shape[1]}")
        T = codes.shape[-1]
        hidden = d.quantizer.decode(codes)
        hidden = d.pre_conv(hidden).transpose(1, 2)
        # 因果加倍 mask：float -inf / 0 相加（与 transformers 4.57 create_causal_mask 语义一致）
        causal = torch.triu(
            torch.full((1, 1, T, T), float("-inf"), device=hidden.device, dtype=hidden.dtype),
            diagonal=1,
        )
        masks = {"full_attention": causal}
        # 滑窗层（decoder_config.sliding_window=72）额外需要 sliding_attention mask
        layer_types = getattr(d.config, "layer_types", None) or ["full_attention"] * d.config.num_hidden_layers
        if "sliding_attention" in layer_types:
            window = int(d.config.sliding_window)
            q = torch.arange(T, device=hidden.device).unsqueeze(1)
            kv = torch.arange(T, device=hidden.device).unsqueeze(0)
            masked = (kv < q - (window - 1)) | (kv > q)
            masks["sliding_attention"] = torch.where(
                masked, torch.tensor(float("-inf"), dtype=hidden.dtype, device=hidden.device),
                torch.zeros((T, T), dtype=hidden.dtype, device=hidden.device),
            ).unsqueeze(0).unsqueeze(0)
        hidden = d.pre_transformer(
            inputs_embeds=hidden,
            attention_mask=masks,
        ).last_hidden_state
        hidden = hidden.permute(0, 2, 1)
        for blocks in d.upsample:
            for block in blocks:
                hidden = block(hidden)
        wav = hidden
        for block in d.decoder:
            wav = block(wav)
        return wav.clamp(min=-1, max=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=_DEF_MODEL,
                    help="Qwen3-TTS-12Hz-0.6B 权重目录（必填，或环境变量 QWEN3_TTS_MODEL；需含 speech_tokenizer/ 子目录）")
    ap.add_argument("--out", default=_DEF_OUT,
                    help="onnx 输出目录（默认 <repo>/Qwen3-TTS/models/onnx）")
    ap.add_argument("--seq-len", type=int, default=325,
                    help="codec 帧数（12.5Hz，325 帧 ≈ 26s，受板端 4GB DDR 限制）")
    ap.add_argument("--skip-check", action="store_true", help="跳过 onnxruntime 数值自检")
    args = ap.parse_args()
    if not args.model_path:
        ap.error("--model-path 或环境变量 QWEN3_TTS_MODEL 不能为空（禁止硬编码本机路径）")

    os.makedirs(args.out, exist_ok=True)
    # codec onnx 归入 codec/ 子目录（与 export_talker.py 的 onnx 分类一致）
    out_codec = os.path.join(args.out, "codec")
    os.makedirs(out_codec, exist_ok=True)
    tok_dir = os.path.join(args.model_path, "speech_tokenizer")
    assert os.path.isdir(tok_dir), f"缺少 speech_tokenizer 目录: {tok_dir}"

    print(f"[codec] 加载 speech_tokenizer: {tok_dir}")
    tok = Qwen3TTSTokenizer.from_pretrained(tok_dir, local_files_only=True)
    decoder = tok.model.decoder
    for p in decoder.parameters():
        p.requires_grad = False
    decoder.eval()

    wrapper = CodecDecoder(decoder)
    onnx_path = os.path.join(out_codec, f"codec_decoder_T{args.seq_len}.onnx")
    dummy = torch.randint(0, 2048, (1, 16, args.seq_len), dtype=torch.long)

    # 结构冒烟：wrapper（预制 dict mask）必须能跑通且输出形状正确。
    # 注：不能直接调用 decoder(dummy) 做地基对照——transformers 4.57 在本 torch 版本下
    # 自动构造 mask 走 _vmap_for_bhqkv，vmap 内含 .item() 会报错。wrapper 用预制 mask
    # 绕过 vmap，其 mask 语义与 create_causal_mask 等价（causal additive -inf）。
    with torch.inference_mode():
        out_wrap = wrapper(dummy)
    exp_len = args.seq_len * 1920
    assert out_wrap.shape == (1, 1, exp_len), f"输出形状 {out_wrap.shape} != {(1, 1, exp_len)}"
    assert torch.isfinite(out_wrap).all(), "输出含 NaN/Inf"
    assert float(out_wrap.max()) <= 1.0 and float(out_wrap.min()) >= -1.0, "输出超出 Clamp 范围"
    print(f"[codec] wrapper 冒烟 OK: {out_wrap.shape}, range=({float(out_wrap.min()):.2f}, {float(out_wrap.max()):.2f})")

    print(f"[codec] torch.onnx.export -> {onnx_path}（可能需 1-3 分钟）")
    torch.onnx.export(
        wrapper,
        (dummy,),
        onnx_path,
        input_names=["codes"],
        output_names=["wav"],
        do_constant_folding=True,
        opset_version=15,
        dynamo=False,
    )
    print(f"[codec] 导出完成: {onnx_path}")

    if not args.skip_check:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        sess = ort.InferenceSession(onnx_path, so)
        with torch.inference_mode():
            ref = wrapper(dummy).numpy()
        out = sess.run(None, {"codes": dummy.numpy()})[0]
        max_diff = float(np.abs(ref - out).max())
        ok = max_diff < 1e-4
        print(f"[codec] 自检 max|Δ|={max_diff:.3e} {'PASS' if ok else 'FAIL'}")
        if not ok:
            raise SystemExit("[codec] 自检失败")

    print(f"[codec] done: {onnx_path}")


if __name__ == "__main__":
    main()