#!/usr/bin/env python3
"""Export the streaming (103→24) Zipformer protocol to three ONNX graphs.

The vendored streaming model definition lives in this repository
(zipformer/python/streaming_zipformer.py); only the icefall checkout path is
external and injected through the CLI.  This is intentionally not the official
icefall/32 export protocol.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import torch


MODEL_DIMS = dict(vocab_size=6254, decoder_dim=512, encoder_dim=256, joiner_dim=512)
STATE_NAMES = ["cached_len", "cached_avg", "cached_key", "cached_val", "cached_val2", "cached_conv1", "cached_conv2"]


def _safe_relative_position_matmul(self, x_proj, pos, attention_dim, num_heads,
                                   out_proj_weight, out_proj_bias,
                                   cached_key, cached_val):
    """Compute relative-position attention without MatMul-then-Slice fusion."""
    seq_len, bsz, _ = x_proj.size()
    head_dim = attention_dim // num_heads
    pos_dim = self.pos_dim
    q = x_proj[..., 0:attention_dim]
    k = x_proj[..., attention_dim:2 * attention_dim]
    value_dim = attention_dim // 2
    v = x_proj[..., 2 * attention_dim:2 * attention_dim + value_dim]
    p = x_proj[..., 2 * attention_dim + value_dim:]
    left_context_len = cached_key.shape[0]
    k = torch.cat([cached_key, k], dim=0)
    v = torch.cat([cached_val, v], dim=0)
    cached_key = k[-left_context_len:, ...]
    cached_val = v[-left_context_len:, ...]
    kv_len = k.shape[0]

    q = q.reshape(seq_len, bsz, num_heads, head_dim)
    p = p.reshape(seq_len, bsz, num_heads, pos_dim)
    k = k.reshape(kv_len, bsz, num_heads, head_dim)
    v = v.reshape(kv_len, bsz * num_heads, head_dim // 2).transpose(0, 1)
    q = q.permute(1, 2, 0, 3)
    p = p.permute(1, 2, 0, 3)
    k = k.permute(1, 2, 3, 0)
    seq_len2 = 2 * seq_len - 1 + left_context_len
    pos = pos.reshape(1, seq_len2, num_heads, pos_dim).permute(0, 2, 3, 1)

    position_rows = []
    for i in range(seq_len):
        start = self.time1 - 1 - i
        p_row = p[:, :, i:i + 1, :].reshape(bsz * num_heads, 1, pos_dim)
        pos_row = pos[:, :, :, start:start + kv_len].expand(
            bsz, -1, -1, -1).reshape(bsz * num_heads, pos_dim, kv_len)
        position_rows.append(torch.matmul(p_row, pos_row).reshape(
            bsz, num_heads, 1, kv_len))
    pos_weights = torch.cat(position_rows, dim=2)

    attn_output_weights = torch.matmul(q, k) + pos_weights
    attn_output_weights = attn_output_weights.view(
        bsz * num_heads, seq_len, kv_len)
    attn_output_weights = torch.softmax(attn_output_weights, dim=-1)
    attn_output = torch.bmm(attn_output_weights, v)
    attn_output = attn_output.transpose(0, 1).contiguous().view(
        seq_len, bsz, attention_dim // 2)
    attn_output = torch.nn.functional.linear(
        attn_output, out_proj_weight, out_proj_bias)
    return attn_output, attn_output_weights, cached_key, cached_val


def install_safe_position_matmul(streaming):
    """Install the BM1684X-safe expression before constructing EncoderStreaming."""
    streaming.RelPositionMultiheadAttentionStreaming.streaming_multi_head_attention_forward_streaming = (
        _safe_relative_position_matmul
    )


def _install_icefall_stubs() -> None:
    """Provide only the icefall runtime API needed by the old model sources."""
    icefall = ModuleType("icefall")
    utils = ModuleType("icefall.utils")
    dist = ModuleType("icefall.dist")
    utils.torch_autocast = lambda *a, **k: contextlib.nullcontext()
    def make_pad_mask(lengths, max_len=0):
        n = int(max_len or lengths.max())
        return torch.arange(n, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)
    def subsequent_chunk_mask(size, chunk_size, num_left_chunks=-1, device=torch.device("cpu")):
        out = torch.zeros(size, size, dtype=torch.bool, device=device)
        for i in range(size):
            start = 0 if num_left_chunks < 0 else max(0, (i // chunk_size - num_left_chunks) * chunk_size)
            out[i, start:min(size, (i // chunk_size + 1) * chunk_size)] = True
        return out
    utils.make_pad_mask, utils.subsequent_chunk_mask = make_pad_mask, subsequent_chunk_mask
    dist.get_rank = lambda: 0
    icefall.utils = utils
    sys.modules.update({"icefall": icefall, "icefall.utils": utils, "icefall.dist": dist})


def load_streaming_module(icefall_root: Path, streaming_model_file: Path):
    """Import the vendored streaming (103→24) model definition.

    The model source lives inside this repository (zipformer/python/streaming_zipformer.py);
    only the icefall checkout path is external, injected via CLI.
    """
    egs = icefall_root / "egs/librispeech/ASR/pruned_transducer_stateless7_streaming"
    if not (egs / "zipformer.py").is_file():
        raise SystemExit(f"icefall source is not the expected checkout: {egs}")
    _install_icefall_stubs()
    sys.path[:0] = [str(egs), str(icefall_root)]
    spec = importlib.util.spec_from_file_location("zipformer_streaming_model_export", streaming_model_file)
    if spec is None or spec.loader is None:
        raise SystemExit("cannot import vendored streaming zipformer model")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def audit_and_build(streaming, checkpoint: Path):
    """Audit each model prefix before allowing the streaming builder's legacy load."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    # Build the unwrapped models exactly as build_streaming_models does.
    encoder = streaming.Zipformer(num_features=80, output_downsampling_factor=2,
        encoder_dims=(256,)*5, attention_dim=(192,)*5, encoder_unmasked_dims=(192,)*5,
        zipformer_downsampling_factors=(1,2,4,8,2), nhead=(4,)*5,
        feedforward_dim=(768,)*5, num_encoder_layers=(2,)*5,
        cnn_module_kernels=(31,)*5, pos_dim=4, num_left_chunks=4,
        short_chunk_size=50, decode_chunk_size=32)
    decoder = streaming.Decoder(vocab_size=6254, decoder_dim=512, blank_id=0, context_size=2)
    joiner = streaming.Joiner(encoder_dim=256, decoder_dim=512, joiner_dim=512, vocab_size=6254)
    models = {"encoder": encoder, "decoder": decoder, "joiner": joiner}
    reports = {}
    for prefix, model in models.items():
        incoming = {k[len(prefix)+1:]: v for k,v in state.items() if k.startswith(prefix + ".")}
        result = model.load_state_dict(incoming, strict=False)
        missing = list(result.missing_keys)
        unexpected = list(result.unexpected_keys)
        # Training-only counters may be absent; model parameters may not.
        bad_missing = [k for k in missing if not k.endswith("count")]
        reports[prefix] = {"missing": missing, "unexpected": unexpected, "bad_missing": bad_missing}
        if bad_missing or unexpected:
            raise RuntimeError("checkpoint audit failed: " + json.dumps(reports, indent=2))
        print(f"{prefix}: missing={missing} unexpected={unexpected}")
    install_safe_position_matmul(streaming)
    enc = streaming.EncoderStreaming(encoder).eval()
    # Sophon supports Gather: retain the original token-ID Decoder graph, including
    # embedding -> conv -> relu.  Do not use DecoderNPU (CPU embedding workaround).
    dec = DecoderSophon(decoder).eval()
    joi = streaming.JoinerStreaming(joiner).eval()
    return enc, dec, joi, reports


class SophonBatchedStateEncoder(torch.nn.Module):
    """Expose streaming states behind a singleton external batch axis.

    The leading 2 inside each state is the number of layers, not runtime batch.
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x, *states):
        if len(states) != len(STATE_NAMES) * 5:
            raise ValueError(f"expected 35 states, got {len(states)}")
        result = self.encoder(x, *(state.squeeze(0) for state in states))
        return (result[0],) + tuple(state.unsqueeze(0) for state in result[1:])


def wrap_encoder(encoder, state_envelope):
    if state_envelope == "sophon":
        return SophonBatchedStateEncoder(encoder).eval()
    if state_envelope == "streaming":
        return encoder
    raise ValueError(f"unknown state envelope: {state_envelope}")


def encoder_dummy_inputs(encoder, state_envelope):
    states = encoder.get_init_state()
    if state_envelope == "sophon":
        states = [state.unsqueeze(0) for state in states]
    return [torch.randn(1, 103, 80)] + states


class DecoderSophon(torch.nn.Module):
    """Original token Decoder with only the singleton output axis removed."""
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
    def forward(self, token_ids):
        return self.decoder(token_ids, need_pad=False).squeeze(1)


def export_graph(model, args, path: Path, input_names, output_names):
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        torch.onnx.export(model, args, str(path), opset_version=17,
            input_names=input_names, output_names=output_names,
            do_constant_folding=True, dynamo=False)
    import onnx
    graph = onnx.load(str(path)); onnx.checker.check_model(graph)
    # Some old exporters omit this attribute; the streaming definition expects explicit Conv shapes.
    for node in graph.graph.node:
        if node.op_type == "Conv" and not any(a.name == "kernel_shape" for a in node.attribute):
            # Initializers are looked up by name and are always present for these Conv nodes.
            weights = next((x for x in graph.graph.initializer if x.name == node.input[1]), None)
            if weights is not None:
                from onnx import helper
                node.attribute.append(helper.make_attribute("kernel_shape", list(weights.dims[2:])))
    onnx.checker.check_model(graph); onnx.save(graph, str(path))
    print(f"wrote {path} ({path.stat().st_size} bytes)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--icefall-root", "--source-root", dest="icefall_root", required=True)
    p.add_argument("--streaming-model", default=str(Path(__file__).resolve().parent / "streaming_zipformer.py"),
                   help="vendored streaming (103→24) model definition inside this repo")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", "--output", dest="output_dir", required=True)
    p.add_argument("--state-envelope", choices=("sophon", "streaming"), default="sophon",
                   help="encoder state IO envelope (default: explicit Sophon batch-1 envelope)")
    a = p.parse_args()
    ck, out = Path(a.checkpoint), Path(a.output_dir)
    if not ck.is_file(): raise SystemExit(f"checkpoint not found: {ck}")
    streaming = load_streaming_module(Path(a.icefall_root), Path(a.streaming_model))
    raw_enc, dec, joi, report = audit_and_build(streaming, ck)
    enc = wrap_encoder(raw_enc, a.state_envelope)
    states = encoder_dummy_inputs(raw_enc, a.state_envelope)
    enc_names = ["x"] + [f"{kind}_{i}" for kind in STATE_NAMES for i in range(5)]
    enc_out = ["encoder_out"] + [f"new_{kind}_{i}" for kind in STATE_NAMES for i in range(5)]
    export_graph(enc, tuple(states), out / "encoder_103_24_256.onnx", enc_names, enc_out)
    export_graph(dec, (torch.tensor([[0, 0]], dtype=torch.int64),), out / "decoder_sophon_tokens.onnx", ["token_ids"], ["decoder_out"])
    export_graph(joi, (torch.randn(1, 256), torch.randn(1, 512)), out / "joiner_streaming.onnx", ["enc_out", "dec_out"], ["logit"])
    (out / "load_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

if __name__ == "__main__": main()
