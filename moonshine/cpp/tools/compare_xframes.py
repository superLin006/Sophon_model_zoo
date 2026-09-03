#!/usr/bin/env python
"""对比 C++ 特征提取(feat_dump 产物)与 Python(numpy) 参考
用法: python compare_xframes.py <cpp_x_frames.bin> <wav_name>
  1) 用与 make_ref.py 完全相同的公式生成参考 x_frames
  2) 与 C++ 二进制对比, 输出 mean_abs / max_abs / 不一致元素数
通过标准: mean_abs_diff < 0.01 (目标 ~2e-7, 即 float32 1-2 ulp)
"""
import os
import sys

import numpy as np
import soundfile as sf

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
FRAME_LEN, N_SAMPLES, EPS = 80, 160000, 1e-6
LOG_K = float(np.load(os.path.join(ROOT, "models", "log_k.npy")))


def main():
    cpp_bin, name = sys.argv[1], sys.argv[2]
    audio, sr = sf.read(os.path.join(ROOT, "test_data", f"{name}.wav"), dtype="float32")
    if sr != 16000:
        import torch
        import torchaudio
        audio = torchaudio.functional.resample(
            torch.from_numpy(audio).unsqueeze(0), sr, 16000).squeeze(0).numpy()
    out = np.zeros(N_SAMPLES, np.float32)
    out[: len(audio)] = audio

    x = out.reshape(-1, FRAME_LEN)
    mean = x.mean(axis=-1, keepdims=True)
    centered = x - mean
    rms = np.sqrt((centered ** 2).mean(axis=-1, keepdims=True) + EPS)
    ref = np.arcsinh(np.exp(LOG_K) * centered / rms).astype(np.float32)

    cpp = np.fromfile(cpp_bin, np.float32).reshape(-1, FRAME_LEN)
    assert cpp.shape == ref.shape, (cpp.shape, ref.shape)

    d = np.abs(cpp - ref)
    print(f"[{name}.wav] shape={ref.shape}")
    print(f"  mean_abs_diff = {d.mean():.3e}  (pass if < 0.01)")
    print(f"  max_abs_diff  = {d.max():.3e}   (目标 ~2e-7)")
    n_bad = int((d > 0.01).sum())
    print(f"  元素数 > 0.01 : {n_bad} / {ref.size}")
    ok = d.mean() < 0.01 and n_bad == 0
    print(f"  结果: {'PASS' if ok else 'FAIL'}")
    if not ok:
        idx = np.unravel_index(np.argmax(d), d.shape)
        print(f"  最大差异位置 frame={idx[0]} j={idx[1]}: cpp={cpp[idx]:.7f} ref={ref[idx]:.7f}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
