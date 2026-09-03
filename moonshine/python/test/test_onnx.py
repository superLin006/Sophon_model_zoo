#!/usr/bin/env python
"""
Moonshine streaming-small ONNX 精度验证(与 HF baseline 对比)
=============================================================
验证内容(对 test_data/0.wav 与 8k.wav, 均尾部补零到 10s 固定 shape):
1. x_frames 的 C++ 构造公式(numpy 分帧 + CMVN eps=1e-6 + asinh(exp(log_k)*x))
   vs HF embedder 输出 —— 验证 log_k 资产与预处理公式
2. encoder ORT vs HF: max_abs < 1e-4
3. decoder ORT 逐步自回归 vs HF 修正循环: token 序列 100% 一致, logits max_abs
4. (参考)与 baseline debug npy 对比: baseline 为未填充音频, 且旧循环存在
   双重 pos_emb bug, 只作参考, 不作 pass/fail 依据

用法:
    python test/test_onnx.py            # 跑 0.wav + 8k.wav
    python test/test_onnx.py --name 0   # 只跑单个

环境: sophon-moonshine (conda)
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio
import onnxruntime as ort
from transformers import MoonshineStreamingForConditionalGeneration
from transformers import AutoProcessor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT + "/python")
from export_onnx import (MODEL_DIR, ONNX_DIR, hf_decoder_loop,
                         N_LAYER, HID, MAX_DEC_LEN, FRAME_LEN, T)

DEBUG_DIR = os.path.join(ROOT, "python", "test", "outputs", "debug")
N_SAMPLES = 160000
T_ENC = 500
EPS = 1e-6

FAIL = []


def check(name, cond, info=""):
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {name} {info}")
    if not cond:
        FAIL.append(name)


def load_audio_padded(path, n_samples=N_SAMPLES):
    audio, sr = sf.read(path, dtype="float32")
    if sr != 16000:
        audio = torchaudio.functional.resample(
            torch.from_numpy(audio).unsqueeze(0), sr, 16000).squeeze(0).numpy()
    out = np.zeros(n_samples, np.float32)
    out[:len(audio)] = audio
    return out


def cmvn_asinh_numpy(audio, log_k, frame_len=FRAME_LEN, eps=EPS):
    """C++ 侧预处理公式的 numpy 复刻:
    1) 分帧 N//80; 2) 帧内 CMVN(eps=1e-6); 3) asinh(exp(log_k)*x)"""
    x = audio.reshape(-1, frame_len).astype(np.float32)
    mean = x.mean(axis=-1, keepdims=True)
    centered = x - mean
    rms = np.sqrt((centered ** 2).mean(axis=-1, keepdims=True) + eps)
    x = centered / rms
    return np.arcsinh(np.exp(log_k) * x).astype(np.float32)   # [2000, 80]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="", help="只跑单个音频名(如 0 / 8k)")
    args = ap.parse_args()
    names = [args.name] if args.name else ["0", "8k"]

    print(f"[Load] {MODEL_DIR}")
    model = MoonshineStreamingForConditionalGeneration.from_pretrained(MODEL_DIR)
    model.eval()
    model.config._attn_implementation = "eager"
    processor = AutoProcessor.from_pretrained(MODEL_DIR)

    log_k = float(np.load(os.path.join(ROOT, "models", "log_k.npy")))
    print(f"[Asset] log_k = {log_k:.10f}")

    enc_ort = ort.InferenceSession(os.path.join(ONNX_DIR, "moonshine_encoder_sim.onnx"),
                                   providers=["CPUExecutionProvider"])
    dec_ort = ort.InferenceSession(os.path.join(ONNX_DIR, "moonshine_decoder_sim.onnx"),
                                   providers=["CPUExecutionProvider"])
    enc_in = [i.name for i in enc_ort.get_inputs()]
    dec_in = [i.name for i in dec_ort.get_inputs()]
    assert enc_in == ["x_frames"], enc_in
    assert dec_in[:3] == ["token", "encoder_out", "cache_len"], dec_in[:3]
    assert dec_in[3:13] == [f"past_k_{i}" for i in range(N_LAYER)], dec_in[3:13]
    assert dec_in[13:] == [f"past_v_{i}" for i in range(N_LAYER)], dec_in[13:]
    print(f"[ORT] encoder 输入 {enc_in}, decoder 输入 {len(dec_in)} 个")

    for name in names:
        wav = os.path.join(ROOT, "test_data", f"{name}.wav")
        print(f"\n{'='*70}\n[{name}] {os.path.basename(wav)}")
        audio = load_audio_padded(wav)
        input_values = torch.from_numpy(audio).unsqueeze(0)

        # ---- 1. x_frames: C++ 公式(numpy) vs HF embedder ----
        with torch.no_grad():
            enc = model.model.encoder
            x_frames_hf = enc.embedder.cmvn(
                input_values.reshape(1, -1, FRAME_LEN))
            x_frames_hf = enc.embedder.comp(x_frames_hf).numpy()[0]     # [2000,80]
        x_frames_np = cmvn_asinh_numpy(audio, log_k)
        d = np.abs(x_frames_np - x_frames_hf).max()
        check(f"x_frames numpy(C++公式) vs HF embedder", d < 1e-6,
              f"max_abs={d:.3e}")

        # ---- 2. encoder: ORT vs HF ----
        with torch.no_grad():
            hf_out = enc(input_values=input_values,
                         attention_mask=torch.ones(1, N_SAMPLES, dtype=torch.long)
                         ).last_hidden_state.numpy()
        ort_out = enc_ort.run(None, {"x_frames": x_frames_np[None]})[0]  # [1,500,620]
        d = np.abs(ort_out - hf_out).max()
        check(f"encoder ORT vs HF", d < 1e-4, f"max_abs={d:.3e} (p99={np.percentile(np.abs(hf_out),99):.3f})")

        # 参考: 与 baseline debug npy(未填充)对比 —— 不同输入(填充 vs 未填充),
        # 不是 ONNX 转换错误。实测尾部补零影响 encoder 最后 ~12 帧(右窗口 4 帧
        # 跨层传播), 如 frame331 max_abs≈0.29; 前 320 帧 ≤6.4e-3。C++ 对比时
        # 应只比前 T_enc-12 帧, 或与同条件(填充后)HF 输出对比。
        base_enc = np.load(os.path.join(DEBUG_DIR, f"{name}_encoder_output.npy"))
        t_base = base_enc.shape[1]
        if t_base <= ort_out.shape[1]:
            d0 = np.abs(ort_out[0, :t_base - 12] - base_enc[0, :t_base - 12]).max()
            d12 = np.abs(ort_out[0, t_base - 12:t_base] - base_enc[0, t_base - 12:t_base]).max()
            print(f"  [Ref] vs baseline npy(未填充): 前 {t_base-12} 帧 max_abs={d0:.3e}, "
                  f"最后 12 帧(边界效应) max_abs={d12:.3e}")

        # ---- 3. decoder: ORT 逐步 vs HF 修正循环 ----
        with torch.no_grad():
            hf_logits, hf_ids = hf_decoder_loop(model, torch.from_numpy(ort_out))
        # ORT 逐步(模拟 C++ 调用)
        cache_k = [np.zeros((1, MAX_DEC_LEN, HID), np.float32) for _ in range(N_LAYER)]
        cache_v = [np.zeros((1, MAX_DEC_LEN, HID), np.float32) for _ in range(N_LAYER)]
        ort_ids, ort_logits, nid = [], [], 1
        for step in range(72):
            feed = {"token": np.array([[nid]], np.int64),
                    "encoder_out": ort_out,
                    "cache_len": np.array([step], np.int64)}
            for i in range(N_LAYER):
                feed[f"past_k_{i}"] = cache_k[i]
                feed[f"past_v_{i}"] = cache_v[i]
            outs = dec_ort.run(None, feed)
            logits, nk, nv = outs[0], outs[1:1 + N_LAYER], outs[1 + N_LAYER:]
            nid = int(logits.argmax(-1).reshape(-1)[-1])
            ort_ids.append(nid)
            ort_logits.append(logits[0, 0])
            if nid == 2:
                break
            for i in range(N_LAYER):
                cache_k[i][:, step:step + 1] = nk[i]
                cache_v[i][:, step:step + 1] = nv[i]
        ort_logits = np.stack(ort_logits)
        n = min(len(hf_ids), len(ort_ids))
        d = np.abs(ort_logits[:n] - hf_logits[:n].numpy()).max()
        tok_ok = ort_ids == hf_ids
        tok_rate = np.mean(np.array(ort_ids[:n]) == np.array(hf_ids[:n]))
        check(f"decoder ORT vs HF 修正循环 token", tok_ok,
              f"{len(ort_ids)} 步, 一致率={tok_rate:.2%}")
        check(f"decoder ORT vs HF logits", d < 1e-3, f"max_abs={d:.3e}")
        text = processor.batch_decode([ort_ids], skip_special_tokens=True)[0].strip()
        print(f"  [Text] {text!r}")

        # 参考: baseline debug npy(未填充 + 旧循环, 仅参考)
        base_toks = np.load(os.path.join(DEBUG_DIR, f"{name}_decoder_tokens.npy"))
        same = list(ort_ids) == list(base_toks)
        n_common = sum(1 for a, b in zip(ort_ids, base_toks) if a == b)
        print(f"  [Ref] baseline tokens(未填充): {'一致' if same else f'前 {n_common}/{len(base_toks)} 个一致'}")
        base_txt = processor.batch_decode([list(base_toks)], skip_special_tokens=True)[0].strip()
        if not same:
            print(f"  [Ref] baseline text: {base_txt!r}")

    print(f"\n{'='*70}\n结果: {'全部 PASS' if not FAIL else f'FAIL: {FAIL}'}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
