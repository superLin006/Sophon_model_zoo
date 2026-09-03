#!/usr/bin/env python
# M2: talker + code_predictor per-layer ONNX 导出（ChatTTS 模式）。
# 用法:
#   python export_talker.py            # 全量导出
#   python export_talker.py --quick    # 仅导出 layer 0 并做单层数值验证

import argparse
import os

import torch
from torch import nn
from transformers.cache_utils import DynamicCache

from qwen_tts import Qwen3TTSModel


# 模型权重目录：必须通过 --model-path 或环境变量 QWEN3_TTS_MODEL 提供，禁止硬编码本机路径
MODEL_PATH = os.environ.get("QWEN3_TTS_MODEL", "")

OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "models", "onnx"))
SEQLEN = 192          # talker prefill 最大序列长度（192 帧=15.36s，decode attention 增 50%）
CP_PREFILL = 2        # code_predictor prefill 固定 2 token
CP_HIST = 16          # code_predictor decode 最大历史长度（2 + 14）


def main():
    global MODEL_PATH, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true", help="覆盖已有 ONNX，执行完整重导出")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--model-path", dest="model_path", default=MODEL_PATH,
                    help="Qwen3-TTS-12Hz-0.6B 权重目录（必填，或通过环境变量 QWEN3_TTS_MODEL 提供）")
    ap.add_argument("--out", dest="out", default=None,
                    help="onnx 输出目录（默认 <repo>/Qwen3-TTS/models/onnx）")
    args = ap.parse_args()
    if not args.model_path:
        ap.error("--model-path 或环境变量 QWEN3_TTS_MODEL 不能为空（禁止硬编码本机路径）")
    MODEL_PATH = args.model_path
    if args.out:
        OUT = os.path.abspath(args.out)
    dt = torch.float32 if args.dtype == "float32" else torch.bfloat16

    os.makedirs(OUT, exist_ok=True)
    # onnx 按用途分子目录，避免 100 个文件平铺（talker 56 / cp 40 / embedding 3 / codec 1）
    OUT_TALKER = os.path.join(OUT, "talker")
    OUT_CP = os.path.join(OUT, "cp")
    OUT_EMB = os.path.join(OUT, "embedding")
    OUT_CODEC = os.path.join(OUT, "codec")
    for d in (OUT_TALKER, OUT_CP, OUT_EMB, OUT_CODEC):
        os.makedirs(d, exist_ok=True)

    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH, device_map="cpu", dtype=dt,
        attn_implementation="eager", local_files_only=True,
    )
    m = tts.model
    talker = m.talker
    for p in m.parameters():
        p.requires_grad = False
    talker.model.config._attn_implementation = "eager"
    talker.code_predictor.config._attn_implementation = "eager"

    layers = talker.model.layers
    rotary = talker.model.rotary_emb
    norm = talker.model.norm
    cfg = talker.config
    L = cfg.num_hidden_layers          # 28
    H = cfg.hidden_size                 # 1024
    KV = cfg.num_key_value_heads        # 8
    D = cfg.head_dim                    # 128

    # ---- talker block ----
    class TalkerBlock(nn.Module):
        def __init__(self, layer, layer_idx, is_last):
            super().__init__()
            self.layer = layer
            self.layer_idx = layer_idx
            self.is_last = is_last

        def forward(self, hidden_states, position_ids, attention_mask):
            pos_emb = rotary(hidden_states, position_ids)
            seq = hidden_states.shape[1]
            cache_position = torch.arange(seq, dtype=torch.long, device=hidden_states.device)
            cache = DynamicCache()
            out = self.layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=None,
                position_embeddings=pos_emb,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
            )
            h = out[0]
            if self.is_last:
                h = norm(h)
            k = cache.layers[self.layer_idx].keys
            v = cache.layers[self.layer_idx].values
            return h, k, v

    class TalkerBlockCache(nn.Module):
        def __init__(self, layer, layer_idx, is_last):
            super().__init__()
            self.layer = layer
            self.layer_idx = layer_idx
            self.is_last = is_last

        def forward(self, hidden_states, position_ids, attention_mask, history_k, history_v):
            pos_emb = rotary(hidden_states, position_ids)
            past = history_k.shape[2]
            cache_position = torch.tensor([past], dtype=torch.long, device=hidden_states.device)
            cache = DynamicCache()
            cache.update(history_k, history_v, self.layer_idx)
            out = self.layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=None,
                position_embeddings=pos_emb,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
            )
            h = out[0]
            if self.is_last:
                h = norm(h)
            k = cache.layers[self.layer_idx].keys[:, :, -1:, :]
            v = cache.layers[self.layer_idx].values[:, :, -1:, :]
            return h, k, v

    def export_block(i):
        is_last = i == L - 1
        pid = torch.zeros(3, 1, SEQLEN, dtype=torch.long)
        pid[:] = torch.arange(SEQLEN, dtype=torch.long)
        mask = torch.full((1, 1, SEQLEN, SEQLEN), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        blk = TalkerBlock(layers[i], i, is_last).eval()
        torch.onnx.export(
            blk,
            (torch.zeros(1, SEQLEN, H, dtype=dt), pid, mask),
            f"{OUT_TALKER}/talker_block_{i}.onnx",
            input_names=["input_states", "position_ids", "attention_mask"],
            output_names=["hidden_states", "past_k", "past_v"],
            do_constant_folding=True, opset_version=15, dynamo=False,
        )

        pid = torch.zeros(3, 1, 1, dtype=torch.long)
        mask = torch.zeros(1, 1, 1, SEQLEN + 1)
        blkc = TalkerBlockCache(layers[i], i, is_last).eval()
        torch.onnx.export(
            blkc,
            (torch.zeros(1, 1, H, dtype=dt), pid, mask,
             torch.zeros(1, KV, SEQLEN, D, dtype=dt), torch.zeros(1, KV, SEQLEN, D, dtype=dt)),
            f"{OUT_TALKER}/talker_block_cache_{i}.onnx",
            input_names=["input_states", "position_ids", "attention_mask", "history_k", "history_v"],
            output_names=["hidden_states", "past_k", "past_v"],
            do_constant_folding=True, opset_version=15, dynamo=False,
        )
        print(f"[talker] exported block {i}")

    # ---- embedding / codec_head ----
    def export_embedding_code():
        class Emb(nn.Module):
            def forward(self, x):
                return talker.model.codec_embedding(x)
        torch.onnx.export(
            Emb().eval(), (torch.zeros(1, SEQLEN, dtype=torch.long),),
            f"{OUT_EMB}/embedding_code.onnx",
            input_names=["input_ids"], output_names=["input_embed"],
            do_constant_folding=True, opset_version=15, dynamo=False,
        )
        print("[talker] exported embedding_code")

    def export_embedding_text():
        # text_embedding 是 151936×2048（fp32 = 1.25GB），直接 fp32 导出会触发 torch.onnx
        # 2GiB protobuf 形状推断上限（_jit_pass_onnx_node_shape_type_inference 崩溃）。
        # 因此先降到 bf16（0.6GB）再导出——这是经板卡验证的 workaround，_sim 版 onnx/bmodel 均正常；
        # 精度损失经量化决策认可（相对误差 ~1e-2，TTS 可接受）。备用方案：dynamo=True 导出（未采用）。
        # 注意：bf16 Sigmoid 在 ORT CPU 下无法运行（NO_IMPLEMENTED），属已知限制，不影响 TPU-MLIR 编译。
        talker.model.text_embedding.to(torch.bfloat16)
        talker.text_projection.to(torch.bfloat16)
        class Emb(nn.Module):
            def forward(self, x):
                return talker.text_projection(talker.model.text_embedding(x))
        torch.onnx.export(
            Emb().eval(), (torch.zeros(1, SEQLEN, dtype=torch.long),),
            f"{OUT_EMB}/embedding_text.onnx",
            input_names=["input_ids"], output_names=["input_embed"],
            do_constant_folding=True, opset_version=15, dynamo=False,
        )
        print("[talker] exported embedding_text")

    def export_codec_head():
        class Head(nn.Module):
            def forward(self, x):
                return talker.codec_head(x)
        torch.onnx.export(
            Head().eval(), (torch.zeros(1, 1, H, dtype=dt),),
            f"{OUT_EMB}/codec_head.onnx",
            input_names=["hidden_states"], output_names=["logits"],
            do_constant_folding=True, opset_version=15, dynamo=False,
        )
        print("[talker] exported codec_head")

    if os.environ.get('TALKER_ONLY'):
        print('[talker_only] skip CP export')
    else:
        # ---- code_predictor ----
        cp = talker.code_predictor
        cp_layers = cp.model.layers
        cp_rotary = cp.model.rotary_emb
        cp_norm = cp.model.norm
        CP_L = cp.config.num_hidden_layers       # 5
        CP_KV = cp.config.num_key_value_heads    # 8
        CP_D = cp.config.head_dim                # 128

        class CpBlock(nn.Module):
            def __init__(self, layer, layer_idx, is_last):
                super().__init__()
                self.layer = layer
                self.layer_idx = layer_idx
                self.is_last = is_last

            def forward(self, hidden_states, position_ids, attention_mask):
                pos_emb = cp_rotary(hidden_states, position_ids)
                seq = hidden_states.shape[1]
                cache_position = torch.arange(seq, dtype=torch.long, device=hidden_states.device)
                cache = DynamicCache()
                out = self.layer(
                    hidden_states, attention_mask=attention_mask, position_ids=None,
                    position_embeddings=pos_emb, past_key_values=cache,
                    use_cache=True, cache_position=cache_position,
                )
                h = out[0]
                if self.is_last:
                    h = cp_norm(h)
                k = cache.layers[self.layer_idx].keys
                v = cache.layers[self.layer_idx].values
                return h, k, v

        class CpBlockCache(nn.Module):
            def __init__(self, layer, layer_idx, is_last):
                super().__init__()
                self.layer = layer
                self.layer_idx = layer_idx
                self.is_last = is_last

            def forward(self, hidden_states, position_ids, attention_mask, history_k, history_v):
                pos_emb = cp_rotary(hidden_states, position_ids)
                past = history_k.shape[2]
                cache_position = torch.tensor([past], dtype=torch.long, device=hidden_states.device)
                cache = DynamicCache()
                cache.update(history_k, history_v, self.layer_idx)
                out = self.layer(
                    hidden_states, attention_mask=attention_mask, position_ids=None,
                    position_embeddings=pos_emb, past_key_values=cache,
                    use_cache=True, cache_position=cache_position,
                )
                h = out[0]
                if self.is_last:
                    h = cp_norm(h)
                k = cache.layers[self.layer_idx].keys[:, :, -1:, :]
                v = cache.layers[self.layer_idx].values[:, :, -1:, :]
                return h, k, v

        def export_cp_block(i):
            is_last = i == CP_L - 1
            pid = torch.arange(CP_PREFILL, dtype=torch.long).unsqueeze(0)
            mask = torch.full((1, 1, CP_PREFILL, CP_PREFILL), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            b = CpBlock(cp_layers[i], i, is_last).eval()
            torch.onnx.export(
                b, (torch.zeros(1, CP_PREFILL, H, dtype=dt), pid, mask),
                f"{OUT_CP}/cp_block_{i}.onnx",
                input_names=["input_states", "position_ids", "attention_mask"],
                output_names=["hidden_states", "past_k", "past_v"],
                do_constant_folding=True, opset_version=15, dynamo=False,
            )
            # CP cache 16 槽：history 16 槽 + mask 17（与 C++/gen_final.sh 对齐，勿改回 15 槽）
            pid = torch.tensor([[CP_HIST]], dtype=torch.long)
            mask = torch.zeros(1, 1, 1, CP_HIST + 1)
            bc = CpBlockCache(cp_layers[i], i, is_last).eval()
            torch.onnx.export(
                bc,
                (torch.zeros(1, 1, H, dtype=dt), pid, mask,
                 torch.zeros(1, CP_KV, CP_HIST, CP_D, dtype=dt),
                 torch.zeros(1, CP_KV, CP_HIST, CP_D, dtype=dt)),
                f"{OUT_CP}/cp_block_cache_{i}.onnx",
                input_names=["input_states", "position_ids", "attention_mask", "history_k", "history_v"],
                output_names=["hidden_states", "past_k", "past_v"],
                do_constant_folding=True, opset_version=15, dynamo=False,
            )
            print(f"[cp] exported block {i}")

        if args.quick:
            # 单层数值验证：比对 wrapper 与原始模型 forward 的输出
            i = 0
            export_block(i)
            torch.manual_seed(0)
            hid = torch.randn(1, SEQLEN, H, dtype=dt)
            pid = torch.zeros(3, 1, SEQLEN, dtype=torch.long)
            pid[:] = torch.arange(SEQLEN, dtype=torch.long)
            mask = torch.full((1, 1, SEQLEN, SEQLEN), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            with torch.no_grad():
                w = TalkerBlock(layers[i], i, False).eval()
                h1, k1, v1 = w(hid, pid, mask)
                # 参考：手动复制模型 forward 的单层逻辑
                pos_emb = rotary(hid, pid)
                cache = DynamicCache()
                cache_position = torch.arange(SEQLEN, dtype=torch.long)
                out = layers[i](hid, attention_mask=mask, position_ids=None,
                                position_embeddings=pos_emb, past_key_values=cache,
                                use_cache=True, cache_position=cache_position)
                h2 = out[0]
                k2 = cache.layers[i].keys
                v2 = cache.layers[i].values
            for a, b, n in [(h1, h2, "hidden"), (k1, k2, "k"), (v1, v2, "v")]:
                d = (a.float() - b.float()).abs().max().item()
                print(f"[verify layer {i}] {n} max_diff={d:.6e} {'PASS' if d < 1e-4 else 'FAIL'}")

            # decode 验证
            past = 8
            hidc = torch.randn(1, 1, H, dtype=dt)
            pidc = torch.zeros(3, 1, 1, dtype=torch.long)
            pidc[:] = past
            maskc = torch.zeros(1, 1, 1, past + 1)
            hk = torch.randn(1, KV, past, D, dtype=dt)
            hv = torch.randn(1, KV, past, D, dtype=dt)
            with torch.no_grad():
                wc = TalkerBlockCache(layers[i], i, False).eval()
                hc1, kc1, vc1 = wc(hidc, pidc, maskc, hk, hv)
                pos_embc = rotary(hidc, pidc)
                cachec = DynamicCache()
                cachec.update(hk, hv, i)
                cache_positionc = torch.tensor([past], dtype=torch.long)
                outc = layers[i](hidc, attention_mask=maskc, position_ids=None,
                                 position_embeddings=pos_embc, past_key_values=cachec,
                                 use_cache=True, cache_position=cache_positionc)
                hc2 = outc[0]
                kc2 = cachec.layers[i].keys[:, :, -1:, :]
                vc2 = cachec.layers[i].values[:, :, -1:, :]
            for a, b, n in [(hc1, hc2, "hidden"), (kc1, kc2, "new_k"), (vc1, vc2, "new_v")]:
                d = (a.float() - b.float()).abs().max().item()
                print(f"[verify cache layer {i}] {n} max_diff={d:.6e} {'PASS' if d < 1e-4 else 'FAIL'}")
            return

        def export_cp_heads():
            n_heads = cp.config.num_code_groups - 1  # 15
            n_emb = n_heads                           # 15（embedding_14 必须导出，历史 bug：n_heads-1 缺失致 C++ 段错误）
            for g in range(n_heads):
                class Head(nn.Module):
                    def __init__(self, gi):
                        super().__init__()
                        self.gi = gi
                    def forward(self, x):
                        return cp.lm_head[self.gi](x)
                torch.onnx.export(
                    Head(g).eval(), (torch.zeros(1, 1, H, dtype=dt),),
                    f"{OUT_CP}/cp_lm_head_{g}.onnx",
                    input_names=["hidden_states"], output_names=["logits"],
                    do_constant_folding=True, opset_version=15, dynamo=False,
                )
            for e in range(n_emb):
                class Emb(nn.Module):
                    def __init__(self, ei):
                        super().__init__()
                        self.ei = ei
                    def forward(self, x):
                        return cp.model.codec_embedding[self.ei](x)
                torch.onnx.export(
                    Emb(e).eval(), (torch.zeros(1, 1, dtype=torch.long),),
                    f"{OUT_CP}/cp_embedding_{e}.onnx",
                    input_names=["input_ids"], output_names=["input_embed"],
                    do_constant_folding=True, opset_version=15, dynamo=False,
                )
            print(f"[cp] exported {n_heads} lm_heads + {n_emb} embeddings")


    import pathlib
    r = os.environ.get('TALKER_RANGE')
    if r:
        a, b = map(int, r.split(':'))
        rng = range(a, b)
    else:
        rng = range(L)
    for i in rng:
        if not args.force and pathlib.Path(f"{OUT_TALKER}/talker_block_{i}.onnx").exists() and \
           pathlib.Path(f"{OUT_TALKER}/talker_block_cache_{i}.onnx").exists():
            print(f"[talker] skip block {i} (exists)")
            continue
        export_block(i)
        print(f"[talker] done block {i}", flush=True)
    if args.force or not pathlib.Path(f"{OUT_EMB}/embedding_code.onnx").exists():
        export_embedding_code()
    if args.force or not pathlib.Path(f"{OUT_EMB}/embedding_text.onnx").exists():
        export_embedding_text()
    if args.force or not pathlib.Path(f"{OUT_EMB}/codec_head.onnx").exists():
        export_codec_head()
        print("[talker] exported embeddings/head", flush=True)
    if not os.environ.get('TALKER_ONLY'):
        for i in range(CP_L):
            if not args.force and pathlib.Path(f"{OUT_CP}/cp_block_{i}.onnx").exists() and \
               pathlib.Path(f"{OUT_CP}/cp_block_cache_{i}.onnx").exists():
                print(f"[cp] skip block {i} (exists)")
                continue
            export_cp_block(i)
        if args.force or not pathlib.Path(f"{OUT_CP}/cp_lm_head_0.onnx").exists() or not pathlib.Path(f"{OUT_CP}/cp_embedding_0.onnx").exists():
            export_cp_heads()


if __name__ == "__main__":
    main()
