"""
Moonshine streaming-small PyTorch Baseline 测试
================================================
生成 ground truth,供后续 ONNX / bmodel 精度对比使用。

流程(对 test_data/ 下每个 wav):
1. soundfile 加载音频,非 16k 用 torchaudio 重采样
2. AutoProcessor 预处理 -> input_values [1, N] f32 + attention_mask [1, N]
3. model.generate() -> 官方 baseline 文本
4. 手动 encoder (model.model.encoder) -> encoder_output [1, T_enc, 620]
5. 手动 decoder 循环 (greedy + KV cache) -> 逐步 logits [steps, vocab] 与 token ids
   (与 generate() 结果交叉验证)
6. 保存 baseline/result.json + debug/*.npy (供 C++ 对比)

用法:
    python test_pytorch.py [audio.wav]     # 跑单个文件
    python test_pytorch.py                 # 自动跑 test_data/ 下所有 wav

环境: sophon-moonshine (conda)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from transformers import AutoProcessor, MoonshineStreamingForConditionalGeneration

ROOT         = Path(__file__).resolve().parent.parent.parent   # moonshine/
MODEL_DIR    = ROOT / "models" / "moonshine-streaming-small"
TEST_DATA    = ROOT / "test_data"
OUTPUT_DIR   = Path(__file__).resolve().parent / "outputs"
BASELINE_DIR = OUTPUT_DIR / "baseline"
DEBUG_DIR    = OUTPUT_DIR / "debug"
for d in (BASELINE_DIR, DEBUG_DIR):
    d.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
SOS_ID, EOS_ID, PAD_ID = 1, 2, 0

# ── 加载模型 ──────────────────────────────────────────────────────────────

print(f"[Load] Loading model from {MODEL_DIR} ...")
t0 = time.time()
model = MoonshineStreamingForConditionalGeneration.from_pretrained(MODEL_DIR)
processor = AutoProcessor.from_pretrained(MODEL_DIR)
model.eval()
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"[Load] Done in {time.time()-t0:.1f}s. Params: {n_params:.1f}M")


def load_audio(path: Path) -> np.ndarray:
    """加载音频并重采样到 16kHz(soundfile + torchaudio)"""
    audio, sr = sf.read(str(path), dtype="float32")
    if sr != SAMPLE_RATE:
        print(f"[Audio] {path.name}: {sr}Hz -> {SAMPLE_RATE}Hz resample")
        audio = torchaudio.functional.resample(
            torch.from_numpy(audio).unsqueeze(0), sr, SAMPLE_RATE
        ).squeeze(0).numpy()
    return audio


@torch.no_grad()
def run_one(audio_path: Path) -> dict:
    name = audio_path.stem
    print(f"\n[Run] {audio_path.name}")

    # 1. 音频加载 + processor 预处理
    audio = load_audio(audio_path)
    n_samples = len(audio)
    seconds = n_samples / SAMPLE_RATE
    max_len = int(seconds * 6) + 10          # max_new_tokens 经验值

    inputs = processor(audio, return_tensors="pt", sampling_rate=SAMPLE_RATE)
    input_values  = inputs.input_values       # [1, N] float32
    attention_mask = inputs.attention_mask    # [1, N] int64
    print(f"[Pre] input_values {tuple(input_values.shape)}, "
          f"attention_mask {tuple(attention_mask.shape)}, dur={seconds:.2f}s, "
          f"max_new_tokens={max_len}")

    # 2. generate() baseline
    t0 = time.time()
    generated_ids = model.generate(
        input_values=input_values,
        attention_mask=attention_mask,
        max_new_tokens=max_len,
    )
    gen_time = time.time() - t0
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(f"[Gen] {text!r} ({gen_time:.2f}s, {generated_ids.shape[1]-1} new tokens)")

    # 3. 手动 encoder
    enc = model.model.encoder(input_values=input_values, attention_mask=attention_mask)
    enc_out       = enc.last_hidden_state          # [1, T_enc, 620]
    enc_attn_mask = enc.attention_mask             # [1, T_enc] (int32) 或 None
    t_enc = enc_out.shape[1]
    print(f"[Enc] encoder_output {tuple(enc_out.shape)}, "
          f"encoder_attention_mask {tuple(enc_attn_mask.shape) if enc_attn_mask is not None else None}")

    # 4. 手动 decoder 循环(greedy + KV cache,与 generate 交叉验证)
    #    注意: HF decoder.forward 内 `encoder_hidden_states += pos_emb` 会原地修改
    #    传入的 tensor(实测: 28 步后 enc_out 被累加 28 次 pos_emb, 保存的
    #    encoder_output.npy 全帧污染)。必须每步传 enc_out.clone(), 否则从
    #    第 2 步起是双重 pos_emb。bmodel 是纯函数无此问题, 但 debug npy
    #    ground truth 必须以本修正循环为准。
    step_logits, step_ids = [], []
    past_key_values = None
    next_ids = torch.tensor([[SOS_ID]], dtype=torch.long)
    t0 = time.time()
    for step in range(max_len):
        dec = model.model.decoder(
            input_ids=next_ids,
            encoder_hidden_states=enc_out.clone(),
            encoder_attention_mask=enc_attn_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = dec.past_key_values
        logits = model.proj_out(dec.last_hidden_state[:, -1, :])   # [1, vocab]
        next_id = int(logits.argmax(dim=-1).item())
        step_logits.append(logits[0].float().numpy())              # [vocab] f32
        step_ids.append(next_id)
        if next_id == EOS_ID:
            break
        next_ids = torch.tensor([[next_id]], dtype=torch.long)
    dec_time = time.time() - t0
    manual_text = processor.batch_decode([step_ids], skip_special_tokens=True)[0].strip()
    print(f"[Dec] {manual_text!r} ({dec_time:.2f}s, {len(step_ids)} steps)")

    match = manual_text == text
    print(f"[Match] generate vs manual decoder: {'OK' if match else 'MISMATCH!'}")

    # 5. 保存 debug 中间输出(供 C++ 对比)
    np.save(DEBUG_DIR / f"{name}_input_values.npy", input_values.numpy())
    np.save(DEBUG_DIR / f"{name}_attention_mask.npy", attention_mask.numpy())
    np.save(DEBUG_DIR / f"{name}_encoder_output.npy", enc_out.numpy())
    if enc_attn_mask is not None:
        np.save(DEBUG_DIR / f"{name}_encoder_attention_mask.npy", enc_attn_mask.numpy())
    np.save(DEBUG_DIR / f"{name}_decoder_logits.npy", np.stack(step_logits))   # [steps, vocab]
    np.save(DEBUG_DIR / f"{name}_decoder_tokens.npy", np.asarray(step_ids, dtype=np.int64))
    print(f"[Debug] saved to {DEBUG_DIR}: "
          f"input_values/attention_mask/encoder_output/decoder_logits({len(step_logits)}x{model.config.vocab_size})/decoder_tokens")

    # 6. baseline 记录
    record = {
        "file": audio_path.name,
        "duration_s": round(seconds, 2),
        "n_samples": n_samples,
        "input_shape": list(input_values.shape),
        "attention_mask_shape": list(attention_mask.shape),
        "t_enc": t_enc,
        "encoder_shape": list(enc_out.shape),
        "text_generate": text,
        "text_manual": manual_text,
        "text_match": match,
        "tokens_generate": generated_ids[0].tolist(),
        "tokens_manual": step_ids,
        "decoder_steps": len(step_ids),
        "generate_time_s": round(gen_time, 3),
        "decoder_loop_time_s": round(dec_time, 3),
        "max_new_tokens": max_len,
        "model": "moonshine-streaming-small",
        "dtype": "F32",
        "note": ("MoonshineStreamingForConditionalGeneration(streaming 变体): encoder 预处理 = "
                 "reshape 到 frame_len=80(5ms@16k) -> frame CMVN -> asinh(k=0.75) -> Linear(80->620)+silu "
                 "-> causal conv1(k5 s2) -> causal conv2(k5 s2); T=N//80, T_enc=(T-1)//4+1; "
                 "10s(160000 samples)->T=2000,T_enc=500; 10 层 encoder 每层带 sliding window mask "
                 "([[16,4]x2,[16,0]x6,[16,4]x2]); decoder 对 encoder_hidden_states 加学习 pos_emb 并 proj 620->512"),
    }
    with open(BASELINE_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    with open(BASELINE_DIR / f"{name}.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"[Save] {BASELINE_DIR / name}.json")
    return record


def main():
    vocab_size = model.config.vocab_size
    if len(sys.argv) >= 2:
        run_one(Path(sys.argv[1]))
        return

    wavs = sorted(TEST_DATA.glob("*.wav"))
    if not wavs:
        print(f"[Error] No wav files in {TEST_DATA}")
        sys.exit(1)

    results = [run_one(wav) for wav in wavs]
    with open(BASELINE_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[Done] {len(results)} files. Baseline: {BASELINE_DIR}")
    print(f"[Info] vocab_size={vocab_size}, hidden=512(dec)/620(enc), layers=10/10")


if __name__ == "__main__":
    main()
