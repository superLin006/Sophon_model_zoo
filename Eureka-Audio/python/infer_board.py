#!/usr/bin/env python3
"""
Eureka-Audio BM1684X 板上推理（sophon.sail 版本）

完整流程：WAV → mel(numpy) → whisper_encoder+audio_adaptor → qwen3 prefill+decode → JSON

为什么用 Python/sail 而非 C++/bmrt：
  纯 bmrt 裸调用在真实数据下会触发 TPU 异常导致板卡重启；
  sophon.sail 的 process() 接口处理了底层细节，稳定可靠（已板上验证）。

依赖（板卡自带）：numpy, scipy, tokenizers, sophon.sail
用法：
  python3 infer_board.py \
    --whisper models/BM1684X/whisper_encoder_b1_bf16.bmodel \
    --qwen3   models/BM1684X/qwen3_1.7b_embeds_w4bf16_seq512_bm1684x.bmodel \
    --model_dir .  --audio qa_example.wav
"""
import os
import sys
import time
import argparse
import numpy as np
import sophon.sail as sail

# ── 配置 ──────────────────────────────────────────────────────────────────────
SEQ      = 512        # bmodel 编译时的 SEQ_LENGTH
HIDDEN   = 2048
N_LAYERS = 28
N_KV     = 8
HEAD_DIM = 128
EOS_ID   = 151645     # <|im_end|>
N_FFT    = 400
HOP      = 160
SR       = 16000

# ── mel 前处理（纯 numpy，移植自 eureka_infer/utils/audio.py）─────────────────
def hann_window(n=N_FFT):
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float32)

def stft(audio, n_fft=N_FFT, hop=HOP, window=None):
    if window is None:
        window = np.ones(n_fft, dtype=np.float32)
    pad = n_fft // 2
    audio = np.pad(audio, (pad, pad), mode='reflect')
    n_frames = 1 + (len(audio) - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(
        audio, shape=(n_frames, n_fft),
        strides=(audio.strides[0] * hop, audio.strides[0])).copy()
    frames *= window
    mag = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
    return mag.T

def log_mel(audio, filters):
    mag = stft(audio, window=hann_window())
    mel = np.dot(filters, mag)
    log = np.log10(np.clip(mel, 1e-10, None))
    log = np.maximum(log, log.max() - 8.0)
    log = (log + 4.0) / 4.0
    return log.astype(np.float32)[:, :-1]

def split_and_pad_mel(mel, chunk=3000):
    C, T = mel.shape
    nch = (T + chunk - 1) // chunk
    out = []
    for i in range(nch):
        s, e = i * chunk, min((i + 1) * chunk, T)
        seg = mel[:, s:e]
        if e - s < chunk:
            zp = np.zeros((C, chunk - (e - s)), dtype=seg.dtype)
            lp = np.log10(np.clip(zp, 1e-10, None))
            lp = np.maximum(lp, lp.max() - 8.0)
            lp = (lp + 4.0) / 4.0
            seg = np.concatenate([seg, lp], axis=-1)
        out.append(seg[None])
    return np.concatenate(out, axis=0)   # [n_chunks, 128, 3000]

def load_wav_16k_mono(path):
    """读 WAV → 16k 单声道 float32（用 scipy.io.wavfile，支持 PCM16/32 和 IEEE float）"""
    from scipy.io import wavfile
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        a = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        a = data.astype(np.float32) / 2147483648.0
    elif data.dtype in (np.float32, np.float64):
        a = data.astype(np.float32)
    elif data.dtype == np.uint8:
        a = (data.astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f'unsupported wav dtype {data.dtype}')
    if a.ndim == 2:            # 立体声 → 单声道
        a = a.mean(axis=1)
    if sr != SR:
        from scipy.signal import resample_poly
        a = resample_poly(a, SR, sr).astype(np.float32)
    # pad 到 1280 倍数（一个 encoder frame = 80ms = 1280 samples @16k）
    rem = len(a) % 1280
    if rem:
        a = np.concatenate([a, np.zeros(1280 - rem, dtype=np.float32)])
    return a

def wav_to_mel_chunks(path, mel_npz):
    a = load_wav_16k_mono(path)
    # 实际音频对应的 audio token 数：每 1280 samples (80ms) = 1 个 audio token
    real_frames = len(a) // 1280
    with np.load(mel_npz) as f:
        filt = f['mel_128']
    mel = log_mel(a, filt)            # [128, T]
    return split_and_pad_mel(mel), real_frames   # [n_chunks,128,3000], int

# ── 推理 ──────────────────────────────────────────────────────────────────────
def run(engine, net, feed):
    out = engine.process(net, feed)
    return [out[n] for n in engine.get_output_names(net)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--whisper', required=True)
    ap.add_argument('--qwen3', required=True)
    ap.add_argument('--model_dir', default='.')
    ap.add_argument('--audio', required=True)
    ap.add_argument('--max_new_tokens', type=int, default=64)
    ap.add_argument('--device', type=int, default=0)
    args = ap.parse_args()

    md = args.model_dir
    prefix = np.fromfile(os.path.join(md, 'prefix_embeds.bin'), dtype=np.float32).reshape(-1, HIDDEN)
    suffix = np.fromfile(os.path.join(md, 'suffix_embeds.bin'), dtype=np.float32).reshape(-1, HIDDEN)
    plen, slen = prefix.shape[0], suffix.shape[0]

    print(f'[1/4] mel from {args.audio} ...', flush=True)
    mel, real_frames = wav_to_mel_chunks(args.audio, os.path.join(md, 'mel_filters.npz'))
    n_chunks = mel.shape[0]
    print(f'      mel chunks={n_chunks}  real audio tokens={real_frames}', flush=True)

    # ── whisper encoder + audio_adaptor ──
    print('[2/4] whisper encoder ...', flush=True)
    t0 = time.time()
    w = sail.Engine(args.whisper, args.device, sail.IOMode.SYSIO)
    audio_embeds = []
    for c in range(n_chunks):
        enc = run(w, 'whisper_encoder', {'mel': mel[c:c+1].astype(np.float32)})[0]
        ad  = run(w, 'audio_adaptor', {'encoder_feat': enc.squeeze(0)})[0]
        audio_embeds.append(ad)
    audio = np.concatenate(audio_embeds, axis=0)   # [n_chunks*375, 2048]
    # 只保留实际音频对应的 token，丢弃 pad 静音部分（对齐原版 real_frames 截取）
    audio = audio[:real_frames]
    del w
    alen = audio.shape[0]
    tlen = plen + alen + slen
    print(f'      audio tokens={alen}  tlen={tlen}  ({time.time()-t0:.1f}s)', flush=True)
    assert tlen <= SEQ, f'tlen {tlen} > SEQ {SEQ}'

    # ── qwen3 prefill ──
    print('[3/4] qwen3 prefill ...', flush=True)
    q = sail.Engine(args.qwen3, args.device, sail.IOMode.SYSIO)
    embeds = np.zeros((1, SEQ, HIDDEN), dtype=np.float32)
    embeds[0, :plen] = prefix
    embeds[0, plen:plen+alen] = audio
    embeds[0, plen+alen:tlen] = suffix
    mask = np.full((1, 1, SEQ, SEQ), -1e9, dtype=np.float32)
    for i in range(tlen):
        mask[0, 0, i, :i+1] = 0.0
    pos = np.arange(SEQ, dtype=np.int32).reshape(1, SEQ)
    # KV layout [1, SEQ, N_KV, HEAD]（单 token 连续）
    past_k = [np.zeros((1, SEQ, N_KV, HEAD_DIM), dtype=np.float32) for _ in range(N_LAYERS)]
    past_v = [np.zeros((1, SEQ, N_KV, HEAD_DIM), dtype=np.float32) for _ in range(N_LAYERS)]

    h = embeds
    for i in range(N_LAYERS):
        h, pk, pv = run(q, f'block_{i}', {'hidden_states': h, 'position_ids': pos, 'attention_mask': mask})
        past_k[i][:, :tlen, :, :] = pk[:, :tlen, :, :]
        past_v[i][:, :tlen, :, :] = pv[:, :tlen, :, :]
    logits = run(q, 'lm_head', {'hidden_states': h[:, tlen-1:tlen, :]})[0]
    cur = int(np.argmax(logits.flatten()))

    # ── decode loop ──
    print('[4/4] decode ...', flush=True)
    t1 = time.time()
    result = []
    token_length = tlen
    for _ in range(args.max_new_tokens):
        if cur == EOS_ID:
            break
        result.append(cur)
        if token_length >= SEQ:
            break
        dh = run(q, 'embedding_cache', {'input_ids': np.array([[cur]], dtype=np.int32)})[0]
        dpos = np.array([[token_length - 1]], dtype=np.int32)
        dmask = np.zeros((1, 1, 1, SEQ + 1), dtype=np.float32)
        dmask[0, 0, 0, token_length-1:SEQ] = -1e9
        for i in range(N_LAYERS):
            dh, nk, nv = run(q, f'block_cache_{i}', {
                'hidden_states': dh, 'position_ids': dpos, 'attention_mask': dmask,
                'past_k': past_k[i], 'past_v': past_v[i]})
            past_k[i][:, token_length-1, :, :] = nk[:, 0, :, :]
            past_v[i][:, token_length-1, :, :] = nv[:, 0, :, :]
        logits = run(q, 'lm_head', {'hidden_states': dh})[0]
        cur = int(np.argmax(logits.flatten()))
        token_length += 1

    dt = time.time() - t1
    # ── 解码 ──
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(os.path.join(md, 'tokenizer.json'))
    text = tk.decode(result)
    print('\n========================================')
    print(f'[Output] {text}')
    print(f'[Perf]   {len(result)} tokens, {dt:.2f}s, {len(result)/dt:.1f} tok/s')
    print('========================================')

if __name__ == '__main__':
    main()
