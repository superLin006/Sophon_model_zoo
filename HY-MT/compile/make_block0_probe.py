#!/usr/bin/env python3
"""Build fixed-sequence block-0 inputs from the native baseline dump."""

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seq_len", type=int, default=64)
    args = parser.parse_args()

    ref = np.load(args.reference)
    embedding = ref["embedding"]
    valid = embedding.shape[1]
    if valid > args.seq_len:
        raise ValueError(f"input length {valid} exceeds seq_len {args.seq_len}")

    states = np.zeros((1, args.seq_len, embedding.shape[-1]), dtype=np.float32)
    states[:, :valid] = embedding
    positions = np.arange(args.seq_len, dtype=np.int32)[None, :]
    mask = np.full((1, 1, args.seq_len, args.seq_len), -10000.0, dtype=np.float32)
    mask[0, 0][np.tril_indices(args.seq_len)] = 0.0
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, input_states=states, position_ids=positions, attention_mask=mask)
    print(f"wrote {args.output}: valid={valid}, seq_len={args.seq_len}")


if __name__ == "__main__":
    main()

