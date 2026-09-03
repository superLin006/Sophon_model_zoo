import argparse
import os
import time

import numpy as np
import torch
import torchaudio

os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")

import ChatTTS

SAMPLE_RATE = 24000


def main():
    ap = argparse.ArgumentParser(description="ChatTTS voice cloning on Sophon (SAIL)")
    ap.add_argument("--ref", required=True, help="reference wav to clone")
    ap.add_argument("--ref-text", default="",
                    help="transcript of the reference wav (required to clone)")
    ap.add_argument("--text", default="希望大家使用愉快，音质和韵律都会跟着参考音频走。")
    ap.add_argument("--model-path", default="../models")
    ap.add_argument("--output", default="clone.wav")
    ap.add_argument("--prompt-out", default="",
                    help="also save [frames,4] audio prompt for the C++ demo")
    ap.add_argument("--max-ref-sec", type=float, default=5.0,
                    help="0 disables trimming; the GPT bmodel budget is SEQLEN=1024")
    ap.add_argument("--temp", type=float, default=0.0001)
    ap.add_argument("--speed", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1222)
    ap.add_argument("--tpu-id", type=int, default=0)
    ap.add_argument("--no-clone", action="store_true",
                    help="same text/seed without a prompt, for A/B listening")
    args = ap.parse_args()

    chat = ChatTTS.Chat()
    chat.load(local_path=args.model_path, tpu_id=args.tpu_id)
    torch.set_num_threads(4)

    spk_smp = txt_smp = None
    if not args.no_clone:
        if not args.ref_text:
            raise SystemExit(
                "--ref-text is required: the reference audio codes must be paired "
                "with their transcript, otherwise ChatTTS EOSes early. "
                "Use --no-clone for the plain-voice path."
            )
        t_enc = time.time()
        spk_smp = chat.load_ref_prompt(args.ref, max_sec=args.max_ref_sec)
        txt_smp = args.ref_text
        n_codes = chat.speaker.decode_prompt(spk_smp).size(1)
        print(f"[clone] ref encoded: {n_codes} codes "
              f"(~{n_codes / 47.0:.1f}s) in {time.time() - t_enc:.2f}s")
        if args.prompt_out:
            frames = chat.save_prompt(spk_smp, args.prompt_out)
            print(f"[clone] C++ prompt saved: {args.prompt_out} ({frames} frames)")

    torch.manual_seed(args.seed)
    t0 = time.time()
    wavs = chat.infer(
        args.text,
        skip_refine_text=True,
        use_decoder=True,
        params_infer_code=ChatTTS.Chat.InferCodeParams(
            prompt=f"[speed_{args.speed}]",
            temperature=args.temp,
            spk_smp=spk_smp,
            txt_smp=txt_smp,
        ),
    )
    cost = time.time() - t0

    audio = np.asarray(wavs[0], dtype=np.float32)
    sec = audio.size / SAMPLE_RATE
    tag = "baseline" if args.no_clone else "clone"
    print(f"[{tag}] audio={sec:.2f}s infer={cost:.2f}s RTF={cost / sec:.3f}")
    torchaudio.save(args.output, torch.from_numpy(audio.reshape(1, -1)), SAMPLE_RATE)
    print(f"[{tag}] saved: {args.output}")


if __name__ == "__main__":
    main()
