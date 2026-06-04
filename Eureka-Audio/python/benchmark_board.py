#!/usr/bin/env python3
"""
Eureka-Audio BM1684X 批量意图识别 benchmark（sophon.sail）

一次加载 bmodel，循环跑所有测试音频，对比期望 action 算准确率。
用法：
  python3 benchmark_board.py \
    --whisper models/BM1684X/whisper_encoder_b1_bf16.bmodel \
    --qwen3   models/BM1684X/qwen3_1.7b_embeds_w4bf16_seq512_bm1684x.bmodel \
    --model_dir . --audio_dir intent_wav
"""
import os
import re
import sys
import json
import time
import argparse
import numpy as np
import sophon.sail as sail

# 复用 infer_board.py 的 mel 前处理
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer_board import (wav_to_mel_chunks, run,
                         SEQ, HIDDEN, N_LAYERS, N_KV, HEAD_DIM, EOS_ID)

# 期望答案，对应 test_audios/intent/long_01.wav ~ long_09.wav（ChatTTS 合成的口语化长指令）
EXPECTED = {
    1: "open_whiteboard", 2: "set_pen",   3: "close_window", 4: "set_volume",
    5: "open_camera",     6: "draw_shape", 7: "set_tool",     8: "save_file",
    9: "screenshot",
}
TEXTS = {
    1: "帮我把白板打开一下，我现在要开始讲课了",
    2: "我想用马克笔写字，麻烦把笔迹大小调到十二号",
    3: "这个窗口我用不到了，帮我把当前的窗口关掉吧",
    4: "声音有点小，能不能帮我把音量往上调一点",
    5: "我需要看一下画面，请帮我把摄像头打开",
    6: "在白板的中间，帮我画一个大一点的圆形",
    7: "我写错了，现在切换到橡皮擦模式",
    8: "这份内容很重要，帮我把当前的文件保存下来",
    9: "这个画面不错，帮我截一张图",
}


def whisper_encode(w, mel, real_frames):
    audio_embeds = []
    for c in range(mel.shape[0]):
        enc = run(w, 'whisper_encoder', {'mel': mel[c:c+1].astype(np.float32)})[0]
        ad = run(w, 'audio_adaptor', {'encoder_feat': enc.squeeze(0)})[0]
        audio_embeds.append(ad)
    audio = np.concatenate(audio_embeds, axis=0)
    return audio[:real_frames]   # 只取实际音频 token，丢弃 pad 静音


def qwen3_generate(q, prefix, suffix, audio, max_new_tokens=64):
    plen, slen, alen = prefix.shape[0], suffix.shape[0], audio.shape[0]
    tlen = plen + alen + slen
    if tlen > SEQ:
        return None, tlen
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
    result = []
    token_length = tlen
    for _ in range(max_new_tokens):
        if cur == EOS_ID:
            break
        result.append(cur)
        if token_length >= SEQ:
            break
        dh = run(q, 'embedding_cache', {'input_ids': np.array([[cur]], dtype=np.int32)})[0]
        dpos = np.array([[token_length-1]], dtype=np.int32)
        dmask = np.zeros((1, 1, 1, SEQ+1), dtype=np.float32)
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
    return result, tlen


def extract_action(text):
    m = re.search(r'"action"\s*:\s*"([^"]+)"', text)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--whisper', required=True)
    ap.add_argument('--qwen3', required=True)
    ap.add_argument('--model_dir', default='.')
    ap.add_argument('--audio_dir', default='intent_wav')
    ap.add_argument('--device', type=int, default=0)
    args = ap.parse_args()

    md = args.model_dir
    prefix = np.fromfile(os.path.join(md, 'prefix_embeds.bin'), dtype=np.float32).reshape(-1, HIDDEN)
    suffix = np.fromfile(os.path.join(md, 'suffix_embeds.bin'), dtype=np.float32).reshape(-1, HIDDEN)
    mel_npz = os.path.join(md, 'mel_filters.npz')
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(os.path.join(md, 'tokenizer.json'))

    # 先把所有音频转 mel（whisper 阶段），再统一过 qwen3
    cases = sorted([f for f in os.listdir(args.audio_dir) if f.endswith('.wav')])
    print(f'Found {len(cases)} audio files\n', flush=True)

    print('=== Stage A: whisper encode all ===', flush=True)
    w = sail.Engine(args.whisper, args.device, sail.IOMode.SYSIO)
    audio_feats = {}
    for f in cases:
        cid = int(re.search(r'(\d+)', f).group(1))
        mel, real_frames = wav_to_mel_chunks(os.path.join(args.audio_dir, f), mel_npz)
        audio_feats[cid] = whisper_encode(w, mel, real_frames)
        print(f'  long_{cid:02d}: {audio_feats[cid].shape[0]} audio tokens', flush=True)
    del w

    print('\n=== Stage B: qwen3 generate all ===', flush=True)
    q = sail.Engine(args.qwen3, args.device, sail.IOMode.SYSIO)
    correct, total = 0, 0
    t0 = time.time()
    print(f'\n{"ID":<4}{"期望":<16}{"预测":<16}{"结果":<6}{"输出"}')
    print('-' * 80)
    for cid in sorted(audio_feats):
        result, tlen = qwen3_generate(q, prefix, suffix, audio_feats[cid])
        if result is None:
            print(f'{cid:<4}{"":<16}{"":<16}{"SKIP":<6}tlen={tlen} > SEQ')
            continue
        text = tk.decode(result)
        pred = extract_action(text)
        exp = EXPECTED.get(cid, '?')
        ok = (pred == exp)
        correct += ok
        total += 1
        mark = 'OK' if ok else 'X'
        print(f'{cid:<4}{exp:<16}{str(pred):<16}{mark:<6}{text[:40]}', flush=True)

    dt = time.time() - t0
    print('-' * 80)
    print(f'\n准确率: {correct}/{total} = {100*correct/max(total,1):.1f}%')
    print(f'qwen3 总耗时: {dt:.1f}s, 平均 {dt/max(total,1):.1f}s/case')


if __name__ == '__main__':
    main()
