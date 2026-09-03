"""
SenseVoice Small PyTorch Baseline 测试
使用 FunASR 官方接口生成 ground truth，供后续 ONNX / bmodel 精度对比使用
同时保存 speech_features、encoder_out 等中间输出到 debug/ 供 C++ 对比

用法:
    python test_pytorch.py [audio.wav] [language]
    python test_pytorch.py                         # 自动跑 test_data/ 下所有 wav

语言选项: zh / en / ja / ko / yue / auto
环境: sophon-sensevoice（测试脚本用 soundfile 读取 WAV，不依赖 ffmpeg）
"""

import sys
import json
import time
import numpy as np
import soundfile as sf
import torch
from pathlib import Path

ROOT         = Path(__file__).parent.parent.parent  # sensevoice/
OUTPUT_DIR   = Path(__file__).parent / "outputs"
BASELINE_DIR = OUTPUT_DIR / "baseline"
DEBUG_DIR    = OUTPUT_DIR / "debug"
for d in [BASELINE_DIR, DEBUG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── 加载模型 ──────────────────────────────────────────────────────────────

from funasr import AutoModel

print("[Load] Loading SenseVoice Small via FunASR...")
model = AutoModel(
    model="iic/SenseVoiceSmall",
    trust_remote_code=True,
    device="cpu",
    disable_update=True,
)
print("[Load] Done.")

# ── 注册 hook 捕获中间输出 ────────────────────────────────────────────────

_captured = {}

def _encoder_hook(module, input, output):
    _captured["speech_features"] = input[0].detach().cpu().numpy()
    _captured["encoder_out"]     = output[0].detach().cpu().numpy()

model.model.encoder.register_forward_hook(_encoder_hook)


# ── 推理 ──────────────────────────────────────────────────────────────────

def run_one(audio_path: str, language: str = "zh") -> dict:
    name = Path(audio_path).stem
    print(f"\n[Run] {audio_path}  lang={language}")
    _captured.clear()
    t0 = time.time()

    # 1. 官方 generate（ground truth）
    audio, sample_rate = sf.read(audio_path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    result = model.generate(input=audio, fs=sample_rate, language=language, use_itn=False)
    text = result[0]["text"] if result else ""

    elapsed = time.time() - t0
    print(f"[Result] {text}")
    print(f"[Time]   {elapsed*1000:.1f} ms")

    # 2. 保存 debug 中间输出（供 C++ 和 ONNX 对比）
    speech_features = _captured.get("speech_features")
    encoder_out     = _captured.get("encoder_out")

    if speech_features is not None:
        np.save(DEBUG_DIR / f"{name}_speech_features.npy", speech_features)
        np.save(DEBUG_DIR / f"{name}_encoder_out.npy",     encoder_out)
        print(f"[Debug]  speech_features {speech_features.shape}, encoder_out {encoder_out.shape}")
    else:
        print("[Debug]  WARNING: hook did not capture features")

    # 3. 保存 baseline 结果
    record = {
        "file":                  audio_path,
        "language":              language,
        "text":                  text,
        "elapsed_ms":            round(elapsed * 1000, 1),
        "speech_features_shape": list(speech_features.shape) if speech_features is not None else None,
        "encoder_shape":         list(encoder_out.shape)     if encoder_out is not None else None,
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
