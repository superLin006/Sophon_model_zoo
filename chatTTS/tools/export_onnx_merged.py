"""Merged ONNX exporter for the CPU/GPU (onnxruntime) ChatTTS backend.

Produces 4 self-contained onnx graphs instead of the 44 fragmented ones the
Sophon/bmodel toolchain needs:

    gpt_prefill.onnx   embed_text + 20 LlamaDecoderLayer + final norm + lm_head_code
                       in : input_ids[1,L] int64, position_ids[1,L] int64,
                            attention_mask[1,1,L,L] f32, spk_emb[1,768] f32,
                            spk_pos[1] int64
                       out: logits[1,626,4] f32, hidden[1,768] f32 (last token),
                            past_k_0..19[1,L,12,64] f32, past_v_0..19[1,L,12,64] f32

    gpt_decode.onnx    embed_code_cache + 20 LlamaDecoderLayer(cache) + lm_head_code
                       in : vq_ids[1,1,4] int64, position_ids[1,1] int64,
                            attention_mask[1,1,1,L+1] f32,
                            history_k_0..19[1,L,12,64] f32, history_v_0..19 ...
                       out: logits[1,626,4] f32, hidden[1,768] f32,
                            new_k_0..19[1,1,12,64] f32, new_v_0..19[1,1,12,64] f32

    decoder.onnx       DVAE decode: in [1,768,L] f32 -> mel [1,100,?] f32
    vocos.onnx         reuse the existing vocos_1-100-2048.onnx (not re-exported here)

The graphs mirror, layer-for-layer, the proven control flow in
sherpa-onnx/csrc/sophon/chattts/gpt_engine.cpp (prefill / decode). The attention
in this repo's modeling_llama.py uses a NON-standard KV layout [b, seq, heads,
dim] and takes mask + position_ids explicitly, so we replicate the exporter's
per-layer call style rather than going through LlamaModel.forward.

Usage:
    python export_onnx_merged.py --gpt --decoder \
        --source_dir /path/to/ChatTTS-ONNX --out_dir ./onnx_merged
    # then copy the existing vocos onnx alongside:
    #   cp ../models/onnx/vocos_1-100-2048.onnx ./onnx_merged/vocos.onnx
"""

from dataclasses import asdict
import argparse
import os

import torch

from config import Config
from gpt import GPT
from dvae import DVAE

# ChatTTS runs fine on CPU; force it so export is deterministic and dependency-light.
torch.cuda.is_available = lambda: False

SEQ_LENGTH = 1024  # max sequence length the graphs are compiled for

# onnxruntime shipped with sherpa-onnx supports ONNX IR version <= 9, but torch
# 2.11's exporter emits IR v10. Downgrade in place after export, preserving the
# existing external-data layout (don't let onnx re-shard self-contained models).
ORT_MAX_IR_VERSION = 9


def _downgrade_ir(path):
    import onnx
    m = onnx.load(path, load_external_data=False)
    if m.ir_version > ORT_MAX_IR_VERSION:
        m.ir_version = ORT_MAX_IR_VERSION
        # save_as_external_data=False keeps whatever layout torch produced:
        # big graphs already have a sidecar .data referenced by the proto; small
        # graphs stay self-contained. We only touch the version field.
        onnx.save(m, path)
        print(f"  [ir] {os.path.basename(path)} -> ir_version={ORT_MAX_IR_VERSION}")


def _load_gpt_state_from_hf(snapshot_dir):
    """Assemble the GPT state_dict from the new-layout HF ChatTTS snapshot.

    The legacy GPT.pt that exporter.py expects bundled everything in one file.
    The current HF repo (2Noise/ChatTTS) splits it:
      asset/gpt/model.safetensors  -> Llama backbone (keys: layers.N.* , norm.* ,
                                       embed_tokens.* which the model deletes)
      asset/Embed.safetensors      -> ChatTTS heads/embeddings (keys already match:
                                       emb_text, emb_code.N, head_text, head_code.N)
    We add the "gpt." prefix to the backbone keys and merge. rotary_emb.inv_freq is
    a registered buffer rebuilt at init, so strict=False is fine.
    """
    from safetensors.torch import load_file

    backbone = load_file(os.path.join(snapshot_dir, "asset", "gpt", "model.safetensors"))
    embed = load_file(os.path.join(snapshot_dir, "asset", "Embed.safetensors"))

    state = {}
    for k, v in backbone.items():
        if k == "embed_tokens.weight":
            continue  # model._build_llama deletes embed_tokens; emb_text comes from Embed
        state["gpt." + k] = v
    state.update(embed)  # emb_text / emb_code.N / head_text / head_code.N — keys already aligned
    return state


def _build_gpt(source_dir):
    cfg = Config()
    gpt_model = GPT(gpt_config=asdict(cfg.gpt), use_flash_attn=False).eval()

    legacy_pt = os.path.join(source_dir, asdict(cfg.path)["gpt_ckpt_path"])
    if os.path.exists(legacy_pt):
        # Old single-file GPT.pt layout (what the original exporter.py assumed).
        gpt_model.from_pretrained(legacy_pt)
    else:
        # New split HF snapshot layout (gpt/model.safetensors + Embed.safetensors).
        state = _load_gpt_state_from_hf(source_dir)
        missing, unexpected = gpt_model.load_state_dict(state, strict=False)
        # Only buffers (rotary_emb.inv_freq) may be "missing"; anything else is a bug.
        bad = [k for k in missing if "rotary_emb.inv_freq" not in k]
        if bad:
            raise RuntimeError(f"GPT weights missing real params: {bad[:8]} ...")
        if unexpected:
            raise RuntimeError(f"GPT weights had unexpected keys: {unexpected[:8]} ...")
        print(f"[gpt] loaded HF split safetensors "
              f"({len(state)} tensors, {len(missing)} buffer(s) auto-filled)")

    for p in gpt_model.parameters():
        p.requires_grad = False
    return gpt_model


class GptPrefill(torch.nn.Module):
    """One forward = full prefill: text embed -> 20 layers -> norm -> lm_head_code.

    Speaker embedding is injected by overwriting the hidden vector at spk_pos
    (matches gpt_engine.cpp prefill()'s "inject speaker embedding" step).
    Returns code logits + the last real-token hidden, plus the per-layer KV cache
    so gpt_decode can continue from it.
    """

    def __init__(self, gpt_model):
        super().__init__()
        self.emb_text = gpt_model.emb_text
        self.layers = gpt_model.gpt.layers
        self.norm = gpt_model.gpt.norm
        self.head_code = gpt_model.head_code
        self.num_vq = gpt_model.num_vq
        self.num_layers = len(self.layers)

    def forward(self, input_ids, position_ids, attention_mask, spk_emb, spk_pos, last_pos):
        # input_ids: [1, L] int64 -> hidden [1, L, 768]
        hidden = self.emb_text(input_ids)

        # Inject speaker embedding at spk_pos (scatter the [1,768] vector into row spk_pos).
        idx = spk_pos.view(1, 1, 1).expand(1, 1, hidden.shape[-1])  # [1,1,768]
        hidden = hidden.scatter(1, idx, spk_emb.unsqueeze(1))

        present_k, present_v = [], []
        for i, layer in enumerate(self.layers):
            hidden, past_kv = layer(
                hidden_states=hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
            )
            k, v = past_kv
            present_k.append(k)  # [1, L, 12, 64]
            present_v.append(v)
        hidden = self.norm(hidden)  # final RMSNorm

        # The input is padded to L; the true last token is at row last_pos
        # (= tok_len-1), NOT row L-1. Gather it explicitly. This mirrors
        # gpt_engine.cpp prefill() which slices row (tok_len-1).
        gidx = last_pos.view(1, 1, 1).expand(1, 1, hidden.shape[-1])  # [1,1,768]
        last_hidden = hidden.gather(1, gidx).squeeze(1)  # [1, 768]
        logits = torch.stack(
            [self.head_code[i](last_hidden) for i in range(self.num_vq)], dim=2
        )  # [1, 626, 4]
        return (logits, last_hidden, *present_k, *present_v)


def _make_dynamic_decode(gpt_model, max_pos=1024):
    """Patch every layer's rotary_emb so the decode graph supports a DYNAMIC KV
    length.

    The stock attention (modeling_llama.py) does, in the cache path:
        cos, sin = rotary_emb(seq_len=kv_seq_len-1)   # truncates the table to
                                                       # the current history len
        cos[position_ids]                              # indexes with ABSOLUTE pos
    With a fixed 1024 history that's fine (table is 1024, pos<=1023). But with a
    dynamic (shorter) history the table shrinks while position_ids stays absolute,
    so cos[position_ids] goes out of bounds.

    Fix: make rotary_emb.forward ignore the requested seq_len and always return
    the full pre-computed table (built to max_position_embeddings >= max_pos at
    construction). Then cos[position_ids] indexes the full table — never OOB — and
    the KV length is free to be dynamic. We only patch the model used for the
    decode export, so the fixed-1024 prefill/Sophon paths are untouched.
    """
    for layer in gpt_model.gpt.layers:
        rot = layer.self_attn.rotary_emb
        # ensure the cached table covers max_pos
        if rot.max_seq_len_cached < max_pos:
            rot._set_cos_sin_cache(max_pos, rot.inv_freq.device,
                                   torch.get_default_dtype())

        def full_forward(x, seq_len=None, _rot=rot):
            n = _rot.max_seq_len_cached
            return (_rot.cos_cached[:, :, :n, ...].to(dtype=x.dtype),
                    _rot.sin_cached[:, :, :n, ...].to(dtype=x.dtype))

        rot.forward = full_forward
    return gpt_model


class GptDecode(torch.nn.Module):
    """One forward = one decode step: code embed -> 20 cache layers -> norm -> lm_head."""

    def __init__(self, gpt_model):
        super().__init__()
        self.emb_code = gpt_model.emb_code
        self.layers = gpt_model.gpt.layers
        self.norm = gpt_model.gpt.norm
        self.head_code = gpt_model.head_code
        self.num_vq = gpt_model.num_vq
        self.num_layers = len(self.layers)

    def forward(self, vq_ids, position_ids, attention_mask, *history):
        # vq_ids: [1, 1, num_vq] int64 -> hidden [1, 1, 768]
        code_emb = [self.emb_code[i](vq_ids[:, :, i]) for i in range(self.num_vq)]
        hidden = torch.stack(code_emb, 2).sum(2)

        history_k = history[: self.num_layers]
        history_v = history[self.num_layers :]

        new_k, new_v = [], []
        for i, layer in enumerate(self.layers):
            hidden, past_kv = layer(
                hidden_states=hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=(history_k[i], history_v[i]),
                use_cache=True,
            )
            k, v = past_kv
            new_k.append(k)  # [1, 1, 12, 64]
            new_v.append(v)
        hidden = self.norm(hidden)
        last_hidden = hidden[:, -1, :]  # [1, 768]
        logits = torch.stack(
            [self.head_code[i](last_hidden) for i in range(self.num_vq)], dim=2
        )
        return (logits, last_hidden, *new_k, *new_v)


def export_gpt_prefill(gpt_model, out_dir):
    H = gpt_model.model_dim          # 768
    NL = len(gpt_model.gpt.layers)   # 20
    cfg0 = gpt_model.gpt.layers[0].self_attn
    HEADS = cfg0.num_heads           # 12
    DIM = cfg0.head_dim              # 64

    model = GptPrefill(gpt_model).eval()
    # Trace at a representative real length N; the seq dim is made DYNAMIC below so
    # prefill computes N×N attention instead of a wasted 1024×1024. (Validated: at
    # N=16 prefill drops from ~1620ms to ~39ms.) The engine no longer pads ids to
    # 1024; it feeds the true token count and copies the [1,N,..] KV into its cache.
    N = 32
    input_ids = torch.zeros(1, N, dtype=torch.int64)
    position_ids = torch.arange(N, dtype=torch.int64).unsqueeze(0)
    attention_mask = (
        -10000.0 * torch.ones(1, 1, N, N, dtype=torch.float32).triu(1)
    )
    spk_emb = torch.zeros(1, H, dtype=torch.float32)
    spk_pos = torch.zeros(1, dtype=torch.int64)
    last_pos = torch.tensor([N - 1], dtype=torch.int64)

    k_names = [f"past_k_{i}" for i in range(NL)]
    v_names = [f"past_v_{i}" for i in range(NL)]
    out_names = ["logits", "hidden"] + k_names + v_names

    dyn = {"input_ids": {1: "N"}, "position_ids": {1: "N"},
           "attention_mask": {2: "N", 3: "N"}}
    for nm in k_names + v_names:
        dyn[nm] = {1: "N"}

    path = os.path.join(out_dir, "gpt_prefill.onnx")
    torch.onnx.export(
        model,
        (input_ids, position_ids, attention_mask, spk_emb, spk_pos, last_pos),
        path,
        input_names=["input_ids", "position_ids", "attention_mask", "spk_emb",
                     "spk_pos", "last_pos"],
        output_names=out_names,
        dynamic_axes=dyn,
        do_constant_folding=True,
        opset_version=15,
    )
    _downgrade_ir(path)
    print(f"[prefill] exported {path}  (heads={HEADS} dim={DIM} layers={NL}, dynamic N)")


def export_gpt_decode(gpt_model, out_dir):
    H = gpt_model.model_dim
    NL = len(gpt_model.gpt.layers)
    cfg0 = gpt_model.gpt.layers[0].self_attn
    HEADS = cfg0.num_heads
    DIM = cfg0.head_dim
    NVQ = gpt_model.num_vq

    # Patch rotary so the KV/history length can be DYNAMIC (see _make_dynamic_decode).
    _make_dynamic_decode(gpt_model, max_pos=SEQ_LENGTH)

    model = GptDecode(gpt_model).eval()
    # Trace at a representative history length P (NOT 1024); P is made dynamic below.
    P = 48
    vq_ids = torch.zeros(1, 1, NVQ, dtype=torch.int64)
    position_ids = torch.tensor([[P]], dtype=torch.int64)
    attention_mask = torch.zeros(1, 1, 1, P + 1, dtype=torch.float32)
    history_k = [torch.zeros(1, P, HEADS, DIM) for _ in range(NL)]
    history_v = [torch.zeros(1, P, HEADS, DIM) for _ in range(NL)]

    hk_names = [f"history_k_{i}" for i in range(NL)]
    hv_names = [f"history_v_{i}" for i in range(NL)]
    nk_names = [f"new_k_{i}" for i in range(NL)]
    nv_names = [f"new_v_{i}" for i in range(NL)]
    in_names = ["vq_ids", "position_ids", "attention_mask"] + hk_names + hv_names
    out_names = ["logits", "hidden"] + nk_names + nv_names

    # Dynamic: history seq dim (axis 1) = P, and the mask's last dim = P+1.
    dyn = {"attention_mask": {3: "Pp1"}}
    for nm in hk_names + hv_names:
        dyn[nm] = {1: "P"}

    path = os.path.join(out_dir, "gpt_decode.onnx")
    torch.onnx.export(
        model,
        (vq_ids, position_ids, attention_mask, *history_k, *history_v),
        path,
        input_names=in_names,
        output_names=out_names,
        dynamic_axes=dyn,
        do_constant_folding=True,
        opset_version=15,
        dynamo=False,  # TorchScript exporter: handles varargs + dynamic_axes
    )
    _downgrade_ir(path)
    print(f"[decode] exported {path} (dynamic P)")


def export_decoder(source_dir, out_dir):
    cfg = Config()
    decoder = DVAE(
        decoder_config=asdict(cfg.decoder),
        dim=cfg.decoder.idim,
    ).eval()
    # IMPORTANT: this is the GPT-hidden -> mel "Decoder" (idim=384, hidden=512),
    # NOT the mel<->mel DVAE (DVAE_full.pt, idim=512/hidden=256). The Sophon
    # decoder bmodel takes [1,768,T]; DVAE.forward(decode) reshapes 768->384 inside
    # (view(B,2,384,T)->permute->flatten). So we load the Decoder weights, which
    # match Config().decoder (idim=384). Legacy layout: asset/Decoder.pt;
    # new HF layout: asset/Decoder.safetensors.
    legacy = os.path.join(source_dir, asdict(cfg.path)["decoder_ckpt_path"])
    hf_st = os.path.join(source_dir, "asset", "Decoder.safetensors")
    if os.path.exists(legacy):
        print(f"[decoder] loading {legacy}")
        sd = torch.load(legacy, weights_only=True, mmap=True)
    elif os.path.exists(hf_st):
        print(f"[decoder] loading {hf_st}")
        from safetensors.torch import load_file
        sd = load_file(hf_st)
    else:
        raise FileNotFoundError(
            f"no Decoder checkpoint found (tried {legacy} and {hf_st})")
    # Key remap: the new-layout Decoder renames the ConvNeXt LayerScale parameter
    # from `decoder_block.N.gamma` (what dvae.py's ConvNeXtBlock registers) to
    # `decoder_block.N.weight`. They are the SAME [hidden] tensor. Rename so it
    # loads into `.gamma` — dropping it would disable LayerScale and corrupt output.
    import re
    # Match ONLY the LayerScale weight that hangs directly off a block, i.e.
    # "...decoder_block.<N>.weight" — NOT submodule weights like
    # "...decoder_block.<N>.norm.weight" / ".dwconv.weight" / ".pwconv1.weight".
    block_weight = re.compile(r"(decoder_block\.\d+)\.weight$")
    remapped = {}
    for k, v in sd.items():
        m = block_weight.search(k)
        if m and v.dim() == 1:
            remapped[k[: m.end() - len(".weight")] + ".gamma"] = v
        else:
            remapped[k] = v
    sd = remapped

    # Still lacks out_conv / vq_layer / coef(may differ) -> strict=False.
    missing, unexpected = decoder.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if not k.startswith("vq_layer")
                    and "encoder" not in k]
    if any(k.startswith("decoder.") or k == "out_conv.weight" for k in real_missing):
        raise RuntimeError(f"[decoder] real params missing: {real_missing[:8]}")
    # After remap, the only acceptable unexpected keys are non-param extras (coef etc.)
    bad_unexp = [k for k in unexpected if k.startswith("decoder.")]
    if bad_unexp:
        raise RuntimeError(f"[decoder] unmapped decoder keys: {bad_unexp[:8]}")
    if unexpected:
        print(f"[decoder] ignoring {len(unexpected)} non-param keys "
              f"(e.g. {unexpected[:3]})")
    # Sanity: gamma actually populated and not the 1e-6 init constant
    g0 = decoder.decoder.decoder_block[0].gamma
    print(f"[decoder] LayerScale gamma loaded: shape={tuple(g0.shape)} "
          f"mean={float(g0.mean()):.4g}")
    for p in decoder.parameters():
        p.requires_grad = False

    class Dec(torch.nn.Module):
        def __init__(self, d):
            super().__init__()
            self.d = d

        def forward(self, x):
            return self.d(x, mode="decode")

    model = Dec(decoder).eval()
    rand_input = torch.rand(1, 768, SEQ_LENGTH)
    path = os.path.join(out_dir, "decoder.onnx")
    torch.onnx.export(
        model,
        (rand_input,),
        path,
        input_names=["dvae_input"],
        output_names=["mel"],
        dynamic_axes={"dvae_input": {2: "T"}, "mel": {2: "T_out"}},
        do_constant_folding=True,
        opset_version=15,
    )
    _downgrade_ir(path)
    print(f"[decoder] exported {path}")


def export_vocos(source_dir, out_dir):
    """Export vocos with a DYNAMIC time dim (vs the shipped fixed [1,100,2048]).

    iSTFT stays out of the graph (handled by csrc/chattts/istft.cpp), matching the
    Sophon export: outputs are mag/x/y each [1, 513, T]. Making T dynamic lets the
    engine feed the real mel length instead of padding to 2048 — a big CPU win
    since vocos runs once per stream batch.
    """
    from vocos import Vocos
    from vocos.pretrained import instantiate_class

    cfg = Config()
    feature_extractor = instantiate_class(
        args=(), init=asdict(cfg.vocos.feature_extractor))
    backbone = instantiate_class(args=(), init=asdict(cfg.vocos.backbone))
    head = instantiate_class(args=(), init=asdict(cfg.vocos.head))
    vocos = Vocos(feature_extractor=feature_extractor, backbone=backbone,
                  head=head).eval()

    legacy = os.path.join(source_dir, asdict(cfg.path)["vocos_ckpt_path"])
    hf_st = os.path.join(source_dir, "asset", "Vocos.safetensors")
    if os.path.exists(legacy):
        print(f"[vocos] loading {legacy}")
        sd = torch.load(legacy, weights_only=True, mmap=True)
    elif os.path.exists(hf_st):
        print(f"[vocos] loading {hf_st}")
        from safetensors.torch import load_file
        sd = load_file(hf_st)
    else:
        raise FileNotFoundError(f"no Vocos checkpoint (tried {legacy}, {hf_st})")
    vocos.load_state_dict(sd, strict=False)
    for p in vocos.parameters():
        p.requires_grad = False

    class VocosNoISTFT(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.v = v

        def forward(self, mel):
            x = self.v.backbone(mel)
            x = self.v.head.out(x).transpose(1, 2)
            mag, p = x.chunk(2, dim=1)
            mag = torch.clip(torch.exp(mag), max=1e2)
            return mag, torch.cos(p), torch.sin(p)

    model = VocosNoISTFT(vocos).eval()
    rand_input = torch.rand(1, 100, 2048)  # trace length; T made dynamic below
    path = os.path.join(out_dir, "vocos.onnx")
    torch.onnx.export(
        model, (rand_input,), path,
        input_names=["mel"], output_names=["mag", "x", "y"],
        dynamic_axes={"mel": {2: "T"}, "mag": {2: "T"}, "x": {2: "T"},
                      "y": {2: "T"}},
        do_constant_folding=True, opset_version=15, dynamo=False,
    )
    _downgrade_ir(path)
    print(f"[vocos] exported {path} (dynamic T)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt", action="store_true")
    ap.add_argument("--decoder", action="store_true")
    ap.add_argument("--vocos", action="store_true")
    ap.add_argument("--int8", action="store_true",
                    help="also emit <name>.int8.onnx via dynamic quantization "
                         "(QUInt8) for the models exported in this run; ~4x "
                         "smaller, ~2x faster on CPU, slight quality loss")
    ap.add_argument("--source_dir", required=True, help="path to ChatTTS-ONNX repo (holds asset/*.pt)")
    ap.add_argument("--out_dir", default="./onnx_merged")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    produced = []  # base names exported this run, for optional int8 quantization

    if args.gpt:
        m = _build_gpt(args.source_dir)
        export_gpt_prefill(m, args.out_dir)
        export_gpt_decode(m, args.out_dir)
        produced += ["gpt_prefill", "gpt_decode"]

    if args.decoder:
        export_decoder(args.source_dir, args.out_dir)
        produced += ["decoder"]

    if args.vocos:
        export_vocos(args.source_dir, args.out_dir)
        produced += ["vocos"]

    if args.int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        for name in produced:
            src = os.path.join(args.out_dir, name + ".onnx")
            dst = os.path.join(args.out_dir, name + ".int8.onnx")
            quantize_dynamic(model_input=src, model_output=dst,
                             weight_type=QuantType.QUInt8)
            print(f"[int8] {name}.int8.onnx")

    print("Done. Outputs in", args.out_dir)
