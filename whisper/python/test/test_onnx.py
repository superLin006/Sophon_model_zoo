"""
Whisper Base ONNX 精度验证
对比 ONNX 推理结果与 PyTorch baseline，验证导出正确性

用法:
    python test_onnx.py [audio.wav] [language]
    python test_onnx.py                          # 自动跑 test_data/ 下所有 wav

前置条件:
    先运行 test_pytorch.py 生成 baseline
    ONNX 文件: models/onnx/whisper_base_encoder_sim.onnx
                models/onnx/whisper_base_decoder_sim.onnx
"""

import sys
import json
import numpy as np
from pathlib import Path

ROOT       = Path(__file__).parent.parent.parent  # whisper/
ONNX_DIR   = ROOT / "models" / "onnx"
OUTPUT_DIR = Path(__file__).parent / "outputs"
BASELINE_DIR = OUTPUT_DIR / "baseline"
ONNX_OUT_DIR = OUTPUT_DIR / "onnx"
DEBUG_DIR    = OUTPUT_DIR / "debug"
ONNX_OUT_DIR.mkdir(parents=True, exist_ok=True)

import onnxruntime as ort
import whisper

# ── 加载 ONNX session ────────────────────────────────────────────────────
encoder_path = ONNX_DIR / "whisper_base_encoder_sim.onnx"
decoder_path = ONNX_DIR / "whisper_base_decoder_sim.onnx"

print(f"[Load] Encoder: {encoder_path}")
encoder_sess = ort.InferenceSession(str(encoder_path))
print(f"[Load] Decoder: {decoder_path}")
decoder_sess = ort.InferenceSession(str(decoder_path))


def compare(name: str, py_val: np.ndarray, onnx_val: np.ndarray) -> float:
    diff = np.mean(np.abs(py_val - onnx_val))
    max_diff = np.max(np.abs(py_val - onnx_val))
    status = "✅ PASS" if diff < 1e-3 else ("⚠️  WARN" if diff < 1e-2 else "❌ FAIL")
    print(f"  [{status}] {name}: mean_abs_diff={diff:.6f}  max_diff={max_diff:.6f}")
    return diff


def run_one(audio_path: str, language: str = "zh") -> dict:
    name = Path(audio_path).stem
    print(f"\n[Run] {audio_path}  lang={language}")

    # ── Mel 特征（与 test_pytorch.py 完全一致）────────────────────────────
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).unsqueeze(0).numpy()  # [1, 80, 3000]

    # ── Encoder 对比 ──────────────────────────────────────────────────────
    encoder_out_onnx = encoder_sess.run(None, {"mel": mel})[0]  # [1, 1500, 512]

    py_encoder = np.load(DEBUG_DIR / f"{name}_encoder_out.npy")
    print("[Encoder accuracy]")
    enc_diff = compare("encoder_output", py_encoder, encoder_out_onnx)

    # ── 保存 ONNX encoder 输出 ────────────────────────────────────────────
    np.save(DEBUG_DIR / f"{name}_encoder_out_onnx.npy", encoder_out_onnx)

    # ── 读取 baseline 结果 ────────────────────────────────────────────────
    baseline_path = BASELINE_DIR / f"{name}.json"
    baseline_text = ""
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline_text = json.load(f)["text"]

    # ── 保存对比报告 ──────────────────────────────────────────────────────
    record = {
        "file": audio_path,
        "language": language,
        "encoder_mean_abs_diff": round(float(enc_diff), 8),
        "baseline_text": baseline_text,
        "pass": bool(enc_diff < 1e-2),
    }
    with open(ONNX_OUT_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    diff_report = (
        f"Encoder mean_abs_diff: {enc_diff:.6f} ({'PASS' if enc_diff < 1e-3 else 'WARN/FAIL'})\n"
        f"Baseline text: {baseline_text}\n"
    )
    with open(ONNX_OUT_DIR / f"diff_vs_baseline_{name}.txt", "w") as f:
        f.write(diff_report)

    return record


def main():
    # 检查 baseline 是否已生成
    if not BASELINE_DIR.exists() or not list(BASELINE_DIR.glob("*.json")):
        print("[Error] baseline/ is empty. Run test_pytorch.py first.")
        sys.exit(1)

    if len(sys.argv) >= 2:
        audio = sys.argv[1]
        lang  = sys.argv[2] if len(sys.argv) >= 3 else "zh"
        run_one(audio, lang)
    else:
        import whisper as _w
        test_data = ROOT / "test_data"
        wavs = sorted(test_data.glob("*.wav"))
        if not wavs:
            print(f"[Error] No wav files found in {test_data}")
            sys.exit(1)

        results = []
        for wav in wavs:
            lang = "en" if "en" in wav.stem else "zh"
            r = run_one(str(wav), lang)
            results.append(r)

        all_pass = all(r["pass"] for r in results)
        print(f"\n[Summary] {'✅ ALL PASS' if all_pass else '❌ SOME FAILED'}")
        for r in results:
            print(f"  {Path(r['file']).name}: encoder_diff={r['encoder_mean_abs_diff']:.6f}")


if __name__ == "__main__":
    main()
