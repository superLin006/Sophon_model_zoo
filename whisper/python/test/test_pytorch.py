"""
Whisper Base PyTorch Baseline 测试
使用官方 whisper.transcribe() 生成 ground truth，供后续 ONNX / bmodel 精度对比使用
同时保存 mel、encoder_out 等中间输出到 debug/ 供 C++ 对比

用法:
    python test_pytorch.py [audio.wav] [language]
    python test_pytorch.py                         # 自动跑 test_data/ 下所有 wav

环境: sophon-whisper（需确保 ffmpeg 在 PATH 中）
"""

import sys
import json
import time
import numpy as np
import torch
from pathlib import Path

ROOT         = Path(__file__).parent.parent.parent  # whisper/
OUTPUT_DIR   = Path(__file__).parent / "outputs"
BASELINE_DIR = OUTPUT_DIR / "baseline"
DEBUG_DIR    = OUTPUT_DIR / "debug"
for d in [BASELINE_DIR, DEBUG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── 加载模型 ──────────────────────────────────────────────────────────────

import whisper

print("[Load] Loading whisper base...")
model = whisper.load_model("base")
model.eval()
print(f"[Load] Done. Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")


# ── 推理 ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_one(audio_path: str, language: str = "zh") -> dict:
    name = Path(audio_path).stem
    print(f"\n[Run] {audio_path}  lang={language}")
    t0 = time.time()

    # 1. 官方 transcribe（ground truth）
    result = model.transcribe(audio_path, language=language)
    text = result["text"].strip()

    elapsed = time.time() - t0
    print(f"[Result] {text}")
    print(f"[Time]   {elapsed*1000:.1f} ms")

    # 2. 保存 debug 中间输出（供 C++ 对比）
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio)        # [80, 3000]
    mel_input = mel.unsqueeze(0)                    # [1, 80, 3000]
    encoder_out = model.encoder(mel_input)          # [1, 1500, 512]

    np.save(DEBUG_DIR / f"{name}_mel.npy",         mel_input.numpy())
    np.save(DEBUG_DIR / f"{name}_encoder_out.npy", encoder_out.numpy())
    print(f"[Debug]  mel {mel_input.shape}, encoder_out {encoder_out.shape}")

    # 3. 保存 baseline 结果
    record = {
        "file":          audio_path,
        "language":      language,
        "text":          text,
        "elapsed_ms":    round(elapsed * 1000, 1),
        "mel_shape":     list(mel_input.shape),
        "encoder_shape": list(encoder_out.shape),
    }
    with open(BASELINE_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    with open(BASELINE_DIR / f"{name}.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")

    return record


def main():
    if len(sys.argv) >= 2:
        run_one(sys.argv[1], sys.argv[2] if len(sys.argv) >= 3 else "zh")
        return

    test_data = ROOT / "test_data"
    wavs = sorted(test_data.glob("*.wav"))
    if not wavs:
        print(f"[Error] No wav files in {test_data}")
        sys.exit(1)

    results = []
    for wav in wavs:
        lang = "en" if "en" in wav.stem else "zh"
        results.append(run_one(str(wav), lang))

    with open(BASELINE_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[Done] {len(results)} files. Saved to {BASELINE_DIR}")


if __name__ == "__main__":
    main()
