#!/usr/bin/env python3
"""
Qwen3-ASR-0.6B BM1684X 板上推理（sophon.sail 版本）。

完整流程：WAV → log-mel(128 slaney) → encoder bmodel → audio_embeds(390,1024)
        → 拼 prefix+audio+suffix → LLM prefill(28层) → decode → 文本

依赖（板卡自带）：numpy, scipy, tokenizers, sophon.sail
用法：
  python3 infer_board.py \
    --encoder models/BM1684X/qwen3_asr_encoder_F16.bmodel \
    --qwen3   models/BM1684X/qwen3_asr_llm_w4bf16_seq2048_bm1684x.bmodel \
    --model_dir . --audio test_data/test_zh.wav
"""
import os
import sys
import time
import argparse
import numpy as np
import sophon.sail as sail

# ── 配置（与 bmodel 编译一致）─────────────────────────────────────────────────
SEQ      = 2048
HIDDEN   = 1024
N_LAYERS = 28
N_KV     = 8
HEAD_DIM = 128
VOCAB    = 151936
EOS_IDS  = (151643, 151645)
AUDIO_TOKEN = 151676
N_FFT    = 400
HOP      = 160
SR       = 16000
N_MEL    = 128
T_MEL_MAX = 3000        # 30s，encoder bmodel 固定输入
T_ENC    = 390          # encoder 输出帧数（30 chunks × 13）
AUDIO_TOKEN_MAX = 390   # 最长音频的 audio token 数


# ── mel 前处理（Qwen3ASRFeatureExtractor 的 numpy 移植，公式逐行对齐）──────────
def hann_window(n=N_FFT):
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float32)

def stft_magnitude(audio, n_fft=N_FFT, hop=HOP):
    """torch.stft(center=True, pad_mode='reflect') 的 numpy 等价：输出 [n_fft//2+1, T]"""
    window = hann_window(n_fft)
    pad = n_fft // 2
    audio = np.pad(audio, (pad, pad), mode='reflect')
    n_frames = len(audio) // hop + 1
    # 取前 n_frames 帧（与 torch.stft 相同；最后丢弃一帧）
    frames = np.lib.stride_tricks.as_strided(
        audio, shape=(n_frames, n_fft),
        strides=(audio.strides[0] * hop, audio.strides[0])).copy()
    frames = frames[:n_frames]
    frames *= window
    mag = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
    return mag.T[:, :-1]        # 丢弃最后一帧

def log_mel(audio, filters):
    """filters: [n_mels, n_fft//2+1]（slaney 128，dump 自原生 feature_extractor）"""
    mag = stft_magnitude(audio)
    mel = np.dot(filters, mag)
    log = np.log10(np.clip(mel, 1e-10, None))
    max_val = log.max(axis=(0, 1), keepdims=True)
    log = np.maximum(log, max_val - 8.0)
    log = (log + 4.0) / 4.0
    return log.astype(np.float32)

def load_wav_16k_mono(path):
    """读 WAV → 16k 单声道 float32（不支持重采样，板卡上要求 16k 输入）"""
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
    if a.ndim == 2:
        a = a.mean(axis=1)
    if sr != SR:
        raise ValueError(f'sample rate {sr} != {SR}, resample before use')
    return a

def wav_to_mel(path, filters):
    a = load_wav_16k_mono(path)
    mel = log_mel(a, filters)             # [128, T_real]
    T_real = mel.shape[1]
    if T_real > T_MEL_MAX:
        raise ValueError(f'audio too long: {T_real} mel frames > {T_MEL_MAX} (30s)')
    # mel 补帧到 3000：**复制最后一帧**而非补零。
    # 原因：encoder 是非因果窗口 attention，pad 帧会作为 key/value 参与窗口内注意力的权重分配。
    # 补零帧(log(1e-10) 归一化后=-1.5)与真实帧差异大，污染明显；复制尾部帧 ≈ 静音延续，污染温和。
    if T_real < T_MEL_MAX:
        tail = np.repeat(mel[:, -1:], T_MEL_MAX - T_real, axis=1)   # numpy repeat：repeats 在前，axis 在后
        mel = np.concatenate([mel, tail], axis=1)
    return mel[None].astype(np.float32), T_real


# ── 推理辅助 ──────────────────────────────────────────────────────────────────
def run(engine, net, feed):
    out = engine.process(net, feed)
    return [out[n] for n in engine.get_output_names(net)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder', required=True, help='encoder bmodel')
    ap.add_argument('--qwen3', required=True, help='LLM bmodel')
    ap.add_argument('--model_dir', default='.')
    ap.add_argument('--audio', required=True)
    ap.add_argument('--max_new_tokens', type=int, default=256)
    ap.add_argument('--device', type=int, default=0)
    args = ap.parse_args()

    md = args.model_dir
    prefix = np.fromfile(os.path.join(md, 'prefix_embeds.bin'), dtype=np.float32).reshape(-1, HIDDEN)
    suffix = np.fromfile(os.path.join(md, 'suffix_embeds.bin'), dtype=np.float32).reshape(-1, HIDDEN)
    plen, slen = prefix.shape[0], suffix.shape[0]
    with np.load(os.path.join(md, 'mel_filters.npz')) as f:
        filters = f['mel_filters'].T   # dump 为 [201, 128]（n_fft//2+1, n_mels），转置成 [128, 201]

    print(f'[1/5] mel from {args.audio} ...', flush=True)
    mel, T_real = wav_to_mel(args.audio, filters)
    print(f'      mel {T_real} frames ({T_real/100:.2f}s)', flush=True)

    # ── encoder：mel → audio_embeds ──
    print('[2/5] encoder ...', flush=True)
    t0 = time.time()
    e = sail.Engine(args.encoder, args.device, sail.IOMode.SYSIO)
    audio = run(e, 'qwen3_asr_encoder', {'mel': mel})[0]      # [390, 1024]
    # 截取真实音频对应帧（与 processor._get_audio_token_length 一致）：
    #   每完整 chunk(100 mel 帧) → 13 帧；末尾不完整 chunk 按 3 次 stride2 conv 折算
    n_chunks = (T_real + 99) // 100
    t = T_real % 100
    feat = ((t - 1) // 2 + 1) if t > 0 else 13   # 第一次 conv
    feat = ((feat - 1) // 2 + 1)                 # 第二次 conv
    feat = ((feat - 1) // 2 + 1)                 # 第三次 conv
    real_tokens = feat + (n_chunks - 1) * 13
    audio = audio[:real_tokens]
    del e
    alen = audio.shape[0]
    tlen = plen + alen + slen
    print(f'      audio tokens={alen}  tlen={tlen}  ({time.time()-t0:.1f}s)', flush=True)
    assert tlen <= SEQ, f'tlen {tlen} > SEQ {SEQ}'

    # ── qwen3 prefill ──
    print('[3/5] qwen3 prefill ...', flush=True)
    q = sail.Engine(args.qwen3, args.device, sail.IOMode.SYSIO)
    embeds = np.zeros((1, SEQ, HIDDEN), dtype=np.float32)
    embeds[0, :plen] = prefix
    embeds[0, plen:plen+alen] = audio
    embeds[0, plen+alen:tlen] = suffix
    mask = np.full((1, 1, SEQ, SEQ), -1e9, dtype=np.float32)
    for i in range(tlen):
        mask[0, 0, i, :i+1] = 0.0
    pos = np.arange(SEQ, dtype=np.int32).reshape(1, SEQ)
    past_k = [np.zeros((1, SEQ, N_KV, HEAD_DIM), dtype=np.float32) for _ in range(N_LAYERS)]
    past_v = [np.zeros((1, SEQ, N_KV, HEAD_DIM), dtype=np.float32) for _ in range(N_LAYERS)]

    h = embeds
    for i in range(N_LAYERS):
        h, pk, pv = run(q, f'block_{i}', {'hidden_states': h, 'position_ids': pos, 'attention_mask': mask})
        past_k[i][:, :tlen, :, :] = pk[:, :tlen, :, :]
        past_v[i][:, :tlen, :, :] = pv[:, :tlen, :, :]
    logits = run(q, 'lm_head', {'hidden_states': h[:, tlen-1:tlen, :]})[0]
    cur = int(np.argmax(logits.flatten()))

    # ── decode loop（修正 Eureka off-by-one：dpos=tlen+k、KV 写 slot tlen+k）──
    print('[4/5] decode ...', flush=True)
    t1 = time.time()
    result = []
    token_length = tlen
    for _ in range(args.max_new_tokens):
        if cur in EOS_IDS:
            break
        result.append(cur)
        if token_length >= SEQ:
            break
        dh = run(q, 'embedding_cache', {'input_ids': np.array([[cur]], dtype=np.int32)})[0]
        dpos = np.array([[token_length]], dtype=np.int32)
        # mask 长度 SEQ+1 = 2049：0..SEQ-1 是 past KV 槽位，SEQ(2048) 是本次新 key 位
        # 屏蔽 token_length..SEQ-1（无效 past 槽位），保留 SEQ 位（新 key 自身）可见
        dmask = np.zeros((1, 1, 1, SEQ + 1), dtype=np.float32)
        dmask[0, 0, 0, token_length:SEQ] = -1e9
        for i in range(N_LAYERS):
            dh, nk, nv = run(q, f'block_cache_{i}', {
                'hidden_states': dh, 'position_ids': dpos, 'attention_mask': dmask,
                'past_k': past_k[i], 'past_v': past_v[i]})
            past_k[i][:, token_length, :, :] = nk[:, 0, :, :]
            past_v[i][:, token_length, :, :] = nv[:, 0, :, :]
        logits = run(q, 'lm_head', {'hidden_states': dh})[0]
        cur = int(np.argmax(logits.flatten()))
        token_length += 1

    dt = time.time() - t1
    # ── 解码 ──
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(os.path.join(md, 'tokenizer.json'))
    text = tk.decode(result, skip_special_tokens=True)
    print('\n========================================')
    print(f'[Output] {text}')
    print(f'[Perf]   {len(result)} tokens, {dt:.2f}s, {len(result)/dt:.1f} tok/s')
    print('========================================')

if __name__ == '__main__':
    main()
