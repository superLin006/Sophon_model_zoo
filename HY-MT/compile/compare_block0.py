#!/usr/bin/env python3
"""Compare block-0 MLIR output with the native PyTorch reference."""

import argparse
import json
import numpy as np


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    a = actual.astype(np.float64).reshape(-1)
    e = expected.astype(np.float64).reshape(-1)
    diff = a - e
    denom = np.linalg.norm(a) * np.linalg.norm(e)
    return {
        "cosine": float(np.dot(a, e) / denom),
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--actual", required=True)
    args = parser.parse_args()
    ref = np.load(args.reference)
    out = np.load(args.actual)
    valid = ref["layer0"].shape[1]
    print("actual keys:", list(out.keys()))
    result = {}
    key_name = "k_cache" if "k_cache" in out else "model.layers.0.self_attn.key_layernorm"
    for actual_name, reference_name in [
        ("output_states", "layer0"),
        (key_name, "layer0_key"),
        ("v_cache", "layer0_value"),
    ]:
        actual = out[actual_name][:, :valid]
        expected = ref[reference_name]
        # PyTorch cache layout is [B, heads, seq, dim].
        if reference_name.startswith("layer0_"):
            expected = expected.transpose(0, 2, 1, 3)
        result[actual_name] = metrics(actual, expected)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
