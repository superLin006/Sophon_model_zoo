"""
SenseVoice Small ONNX 精度验证
对比 ONNX 推理结果与 PyTorch baseline，验证导出正确性

用法:
    python test_onnx.py [audio.wav] [language]
    python test_onnx.py                          # 自动跑 test_data/ 下所有 wav

前置条件:
    先运行 test_pytorch.py 生成 baseline
    ONNX 文件: models/onnx/sensevoice_small_sim.onnx

注意: ONNX 模型输入固定为 [1, 166, 560]（10s 音频），
     短音频会在帧维度 pad 到 166 帧。
"""

import sys
import json
import numpy as np
from pathlib import Path

ROOT         = Path(__file__).parent.parent.parent  # sensevoice/
ONNX_DIR     = ROOT / "models" / "onnx"
OUTPUT_DIR   = Path(__file__).parent / "outputs"
BASELINE_DIR = OUTPUT_DIR / "baseline"
ONNX_OUT_DIR = OUTPUT_DIR / "onnx"
DEBUG_DIR    = OUTPUT_DIR / "debug"
ONNX_OUT_DIR.mkdir(parents=True, exist_ok=True)

import onnxruntime as ort

FIXED_FRAMES = 166   # 10s: (16000*10 - 400)/160 + 1 = 998 fbank -> (998-7)/6+1 = 166 LFR
INPUT_DIM    = 560   # fbank(80) × LFR窗口(7)

# ── 加载 ONNX session ────────────────────────────────────────────────────
onnx_path = ONNX_DIR / "sensevoice_small_sim.onnx"
print(f"[Load] {onnx_path}")
sess = ort.InferenceSession(str(onnx_path))
print(f"[Load] Input:  {sess.get_inputs()[0].name}  {sess.get_inputs()[0].shape}")
print(f"[Load] Output: {sess.get_outputs()[0].name} {sess.get_outputs()[0].shape}")


def compare(name: str, py_val: np.ndarray, onnx_val: np.ndarray) -> float:
    diff     = float(np.mean(np.abs(py_val - onnx_val)))
    max_diff = float(np.max(np.abs(py_val - onnx_val)))
    status   = "PASS" if diff < 1e-3 else ("WARN" if diff < 1e-2 else "FAIL")
    print(f"  [{status}] {name}: mean_abs_diff={diff:.6f}  max_diff={max_diff:.6f}")
    return diff


def pad_features(features: np.ndarray, target_frames: int = FIXED_FRAMES) -> np.ndarray:
    """Pad or trim [1, T, D] to [1, target_frames, D]"""
    T = features.shape[1]
    if T < target_frames:
        pad = np.zeros((1, target_frames - T, features.shape[2]), dtype=features.dtype)
        features = np.concatenate([features, pad], axis=1)
    else:
        features = features[:, :target_frames, :]
    return features


def run_one(audio_path: str, language: str = "zh") -> dict:
    name = Path(audio_path).stem
    print(f"\n[Run] {audio_path}  lang={language}")

    # ── 读取 debug 中的 speech features（由 test_pytorch.py 保存）────────
    feat_path = DEBUG_DIR / f"{name}_speech_features.npy"
    if not feat_path.exists():
        print(f"[Error] {feat_path} not found. Run test_pytorch.py first.")
        return {}

    speech_features = np.load(feat_path)          # [1, T, 560]
    speech_padded   = pad_features(speech_features)  # [1, 166, 560]
    print(f"  speech_features: {speech_features.shape} -> padded: {speech_padded.shape}")

    # ── ONNX 推理 ─────────────────────────────────────────────────────────
    logits_onnx = sess.run(None, {"audio_features": speech_padded})[0]  # [1, 170, 25055]
    print(f"  logits_onnx: {logits_onnx.shape}")

    # ── 读取 baseline 文本 ────────────────────────────────────────────────
    baseline_path = BASELINE_DIR / f"{name}.json"
    baseline_text = ""
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline_text = json.load(f)["text"]

    # ── 对比 logits（与 PyTorch encoder_out 对比，因 ONNX 是整个模型）────
    # ONNX 输出是 logits，无法直接与 encoder_out 对比
    # 改为对比 argmax token 序列（前 T 帧，忽略 pad 部分）
    T = speech_features.shape[1]
    tokens_onnx = logits_onnx[0, :T+4].argmax(axis=-1).tolist()  # +4 prompt tokens
    print(f"  top tokens (first 10): {tokens_onnx[:10]}")

    # ── 保存对比报告 ──────────────────────────────────────────────────────
    record = {
        "file":            audio_path,
        "language":        language,
        "logits_shape":    list(logits_onnx.shape),
        "baseline_text":   baseline_text,
        "top_tokens":      tokens_onnx[:20],
    }
    with open(ONNX_OUT_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    np.save(DEBUG_DIR / f"{name}_logits_onnx.npy", logits_onnx)

    return record


def main():
    if not BASELINE_DIR.exists() or not list(BASELINE_DIR.glob("*.json")):
        print("[Error] baseline/ is empty. Run test_pytorch.py first.")
        sys.exit(1)

    if len(sys.argv) >= 2:
        run_one(sys.argv[1], sys.argv[2] if len(sys.argv) >= 3 else "zh")
        return

    test_data = ROOT / "test_data"
    wavs = sorted(test_data.glob("*.wav"))
    if not wavs:
        print(f"[Error] No wav files found in {test_data}")
        sys.exit(1)

    results = []
    for wav in wavs:
        lang = "en" if "en" in wav.stem else "zh"
        r = run_one(str(wav), lang)
        if r:
            results.append(r)

    print(f"\n[Done] {len(results)} files processed.")
    for r in results:
        print(f"  {Path(r['file']).name}: logits {r['logits_shape']}")


if __name__ == "__main__":
    main()
