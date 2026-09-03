"""Speaker similarity between a reference wav and its cloned output.

Pairs ref_dir/*.wav with gen_dir/<same stem>.wav, embeds both with SpeechBrain's
ECAPA-TDNN (the usual metric in zero-shot TTS papers), and reports cosine
similarity. x86/CPU only -- it does not need a Sophon card.

    pip install speechbrain soundfile torch torchaudio
    python eval_speaker_sim.py --ref-dir ../test_data/refs --gen-dir out/clone --min 0.60
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

EVAL_SR = 16000


@torch.inference_mode()
def embed(model, path: Path) -> np.ndarray:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = torch.from_numpy(wav.mean(axis=1))
    if sr != EVAL_SR:
        x = torchaudio.functional.resample(x, sr, EVAL_SR)
    return model.encode_batch(x.unsqueeze(0)).squeeze().float().numpy()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref-dir", required=True, help="reference voices")
    ap.add_argument("--gen-dir", required=True, help="cloned outputs, named like the refs")
    ap.add_argument("--min", type=float, default=0.60, help="pass threshold on mean cosine")
    ap.add_argument("--ext", default="wav")
    args = ap.parse_args()

    refs = sorted(Path(args.ref_dir).glob(f"*.{args.ext}"))
    if not refs:
        print(f"no *.{args.ext} under {args.ref_dir}", file=sys.stderr)
        return 2

    from speechbrain.inference.speaker import EncoderClassifier

    model = EncoderClassifier.from_hparams(
        "speechbrain/spkrec-ecapa-voxceleb", savedir=".pretrained_models"
    )

    scores = []
    for ref in refs:
        gen = Path(args.gen_dir) / f"{ref.stem}.{args.ext}"
        if not gen.exists():
            print(f"SKIP {ref.stem}: {gen} missing")
            continue
        s = cosine(embed(model, ref), embed(model, gen))
        scores.append(s)
        print(f"{ref.stem:>24}  cosine={s:.3f}")

    if not scores:
        print("nothing scored", file=sys.stderr)
        return 2
    scores = np.array(scores)
    mean = scores.mean()
    print(f"\npairs={len(scores)} mean={mean:.3f} min={scores.min():.3f} "
          f"max={scores.max():.3f}")
    ok = mean >= args.min
    print(f"{'PASS' if ok else 'FAIL'} (threshold mean>={args.min})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
