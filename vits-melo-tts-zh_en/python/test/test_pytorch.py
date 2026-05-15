#!/usr/bin/env python3
"""
test_pytorch.py - Baseline ONNX inference for vits-melo-tts-zh_en
Uses onnxruntime as the baseline (sherpa-onnx provides pre-exported ONNX).
Implements text frontend: lexicon.txt + tokens.txt + jieba segmentation.
"""

import os
import json
import struct
import wave
import numpy as np
import onnxruntime
import jieba

# ===================== Paths =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
MODEL_DIR = os.path.join(PROJECT_DIR, "models", "onnx", "vits-melo-tts-zh_en")

MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")
LEXICON_PATH = os.path.join(MODEL_DIR, "lexicon.txt")
TOKENS_PATH = os.path.join(MODEL_DIR, "tokens.txt")

OUTPUT_BASELINE = os.path.join(BASE_DIR, "outputs", "baseline")
OUTPUT_DEBUG = os.path.join(BASE_DIR, "outputs", "debug")

os.makedirs(OUTPUT_BASELINE, exist_ok=True)
os.makedirs(OUTPUT_DEBUG, exist_ok=True)

# ===================== Load tokens.txt =====================
def load_tokens(tokens_path):
    """Returns dict: token_str -> id"""
    token2id = {}
    with open(tokens_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                token = parts[0]
                token_id = int(parts[1])
                token2id[token] = token_id
    # Map " " (space) to "_" id (per sherpa-onnx convention)
    if "_" in token2id and " " not in token2id:
        token2id[" "] = token2id["_"]
    # Map punctuation variants
    punct_pairs = [(",", "，"), (".", "。"), ("!", "！"), ("?", "？")]
    for a, b in punct_pairs:
        if a in token2id and b not in token2id:
            token2id[b] = token2id[a]
        if b in token2id and a not in token2id:
            token2id[a] = token2id[b]
    if "、" not in token2id and "，" in token2id:
        token2id["、"] = token2id["，"]
    return token2id


# ===================== Load lexicon.txt =====================
def load_lexicon(lexicon_path, token2id):
    """
    Lexicon format per line:
      word phone1 phone2 ... phoneN tone1 tone2 ... toneN
    Returns dict: word -> (token_ids: list[int], tones: list[int])
    """
    word2ids = {}
    with open(lexicon_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            word = parts[0].lower()
            rest = parts[1:]
            if len(rest) % 2 != 0:
                continue
            n = len(rest) // 2
            phones = rest[:n]
            tones_str = rest[n:]
            token_ids = []
            tones = []
            valid = True
            for ph, t_str in zip(phones, tones_str):
                if ph not in token2id:
                    valid = False
                    break
                try:
                    tone = int(t_str)
                except ValueError:
                    valid = False
                    break
                token_ids.append(token2id[ph])
                tones.append(tone)
            if valid and token_ids:
                word2ids[word] = (token_ids, tones)
    # Special aliases (per sherpa-onnx melo-tts-lexicon.cc)
    if "母" in word2ids and "呣" not in word2ids:
        word2ids["呣"] = word2ids["母"]
    if "恩" in word2ids and "嗯" not in word2ids:
        word2ids["嗯"] = word2ids["恩"]
    return word2ids


# ===================== Text Frontend =====================
# Punctuation mapping (Chinese -> ASCII)
PUNCT_MAP = {
    "：": ",", "、": ",", "；": ",",
    "。": ".", "？": "?", "！": "!",
    "，": ",",
}

SENTENCE_BREAK = {".", "!", "?", ",", "。", "！", "？", "，"}


def word_to_ids(word, word2ids, token2id):
    """Convert a word to (token_ids, tones). Handles OOV via char-by-char fallback."""
    # Direct lookup (lowercase)
    key = word.lower()
    if key in word2ids:
        return word2ids[key]

    # Direct lookup as single token (e.g. punctuation)
    if key in token2id:
        return ([token2id[key]], [0])

    # Char-by-char fallback
    all_ids = []
    all_tones = []
    for ch in word:
        ch_key = ch.lower()
        if ch_key in word2ids:
            ids, tones = word2ids[ch_key]
            all_ids.extend(ids)
            all_tones.extend(tones)
        elif ch_key in token2id:
            all_ids.append(token2id[ch_key])
            all_tones.append(0)
        # else: silently skip OOV char
    return (all_ids, all_tones)


def text_to_token_ids(text, word2ids, token2id):
    """
    Convert text to (token_ids, tones).
    Uses jieba for Chinese segmentation + char-by-char fallback.
    Returns: (token_ids: np.ndarray[int64], tones: np.ndarray[int64])
    """
    # Normalize punctuation
    normalized = []
    for ch in text:
        if ch in PUNCT_MAP:
            normalized.append(PUNCT_MAP[ch])
        else:
            normalized.append(ch)
    text_norm = "".join(normalized)

    # Jieba segmentation
    seg_words = list(jieba.cut(text_norm))

    all_ids = []
    all_tones = []

    for word in seg_words:
        ids, tones = word_to_ids(word, word2ids, token2id)
        all_ids.extend(ids)
        all_tones.extend(tones)

    if not all_ids:
        raise ValueError(f"No tokens found for text: {text!r}")

    # add_blank=1: interleave blank(0) between every token: [0, t0, 0, t1, 0, ...]
    # mirrors sherpa-onnx OfflineTtsImpl::AddBlank()
    n = len(all_ids)
    blanked_ids = [0] * (2 * n + 1)
    blanked_tones = [0] * (2 * n + 1)
    for i, (tid, tone) in enumerate(zip(all_ids, all_tones)):
        blanked_ids[2 * i + 1] = tid
        blanked_tones[2 * i + 1] = tone

    return np.array(blanked_ids, dtype=np.int64), np.array(blanked_tones, dtype=np.int64)


# ===================== Write WAV =====================
def save_wav(audio: np.ndarray, path: str, sample_rate: int = 44100):
    """Save float32 audio [-1, 1] to 16-bit PCM WAV."""
    audio_clipped = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


# ===================== Main =====================
def run_inference(text, name, sess, word2ids, token2id, sample_rate, speaker_id):
    print(f"\n[{name}] Text: {text!r}")

    token_ids, tones = text_to_token_ids(text, word2ids, token2id)
    print(f"[{name}] Token count: {len(token_ids)}")
    print(f"[{name}] Token ids: {token_ids.tolist()}")
    print(f"[{name}] Tones:     {tones.tolist()}")

    # Save debug inputs
    np.save(os.path.join(OUTPUT_DEBUG, f"{name}_input_tokens.npy"), token_ids)
    np.save(os.path.join(OUTPUT_DEBUG, f"{name}_input_tones.npy"), tones)

    # Build inputs
    x = token_ids.reshape(1, -1)          # [1, T]
    x_tones = tones.reshape(1, -1)        # [1, T]
    x_lengths = np.array([token_ids.shape[0]], dtype=np.int64)  # [1]
    sid = np.array([speaker_id], dtype=np.int64)                # [1]
    noise_scale = np.array([0.667], dtype=np.float32)
    length_scale = np.array([1.0], dtype=np.float32)
    noise_scale_w = np.array([0.8], dtype=np.float32)

    # Run inference
    outputs = sess.run(
        ["y"],
        {
            "x": x,
            "x_lengths": x_lengths,
            "tones": x_tones,
            "sid": sid,
            "noise_scale": noise_scale,
            "length_scale": length_scale,
            "noise_scale_w": noise_scale_w,
        }
    )
    audio_out = outputs[0]  # shape: [1, 1, T] or [1, T]
    print(f"[{name}] Audio output shape: {audio_out.shape}")

    # Save debug audio array
    np.save(os.path.join(OUTPUT_DEBUG, f"{name}_audio_output.npy"), audio_out)

    # Flatten to 1D
    audio_flat = audio_out.flatten()
    duration_sec = len(audio_flat) / sample_rate
    print(f"[{name}] Audio duration: {duration_sec:.3f}s ({len(audio_flat)} samples @ {sample_rate}Hz)")

    # Save WAV
    wav_path = os.path.join(OUTPUT_BASELINE, f"{name}.wav")
    save_wav(audio_flat, wav_path, sample_rate)
    print(f"[{name}] WAV saved: {wav_path}")

    return {
        "name": name,
        "text": text,
        "token_count": int(len(token_ids)),
        "token_ids": token_ids.tolist(),
        "tones": tones.tolist(),
        "audio_shape": list(audio_out.shape),
        "n_samples": int(len(audio_flat)),
        "duration_sec": round(duration_sec, 3),
        "sample_rate": sample_rate,
        "wav_path": wav_path,
    }


def main():
    print("=" * 60)
    print("vits-melo-tts-zh_en Baseline Test (ONNX)")
    print("=" * 60)

    # Load resources
    print("\nLoading tokens.txt ...")
    token2id = load_tokens(TOKENS_PATH)
    print(f"  Total tokens: {len(token2id)}")

    print("Loading lexicon.txt ...")
    word2ids = load_lexicon(LEXICON_PATH, token2id)
    print(f"  Total lexicon entries: {len(word2ids)}")

    # Load ONNX session
    print(f"\nLoading ONNX model: {MODEL_PATH}")
    sess_opts = onnxruntime.SessionOptions()
    sess_opts.log_severity_level = 3
    sess = onnxruntime.InferenceSession(MODEL_PATH, sess_opts)

    # Read metadata
    meta = sess.get_modelmeta().custom_metadata_map
    sample_rate = int(meta.get("sample_rate", 44100))
    speaker_id = int(meta.get("speaker_id", 1))
    print(f"  sample_rate: {sample_rate}, speaker_id: {speaker_id}")
    print(f"  language: {meta.get('language')}")
    print(f"  model_type: {meta.get('model_type')}")

    # Test cases
    test_cases = [
        ("test_zh", "今天天气真好，我们去公园散步吧。"),
        ("test_en_zh", "你好world，这是一个test句子。"),
    ]

    results = []
    for name, text in test_cases:
        result = run_inference(text, name, sess, word2ids, token2id, sample_rate, speaker_id)
        results.append(result)

    # Save summary
    summary = {
        "model_path": MODEL_PATH,
        "onnxruntime_version": onnxruntime.__version__,
        "sample_rate": sample_rate,
        "speaker_id": speaker_id,
        "metadata": meta,
        "results": results,
    }
    summary_path = os.path.join(OUTPUT_BASELINE, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSummary saved: {summary_path}")
    print("\n=== BASELINE TEST COMPLETE ===")
    for r in results:
        print(f"  [{r['name']}] tokens={r['token_count']}, duration={r['duration_sec']:.3f}s -> {r['wav_path']}")


if __name__ == "__main__":
    main()
