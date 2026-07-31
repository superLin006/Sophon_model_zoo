#!/usr/bin/env python3
"""Run deterministic, non-golden ONNX protocol smoke tests."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Any
import numpy as np

def shape_of(value: Any) -> list[int]:
    return [int(dim) if isinstance(dim, int) and dim > 0 else 1 for dim in value.shape]

def make_input(value: Any, seed: int, integer_values: list[int] | None = None) -> np.ndarray:
    shape = shape_of(value)
    if "int64" in value.type:
        values = integer_values or [0]
        return np.resize(np.asarray(values, dtype=np.int64), shape)
    if "int32" in value.type:
        return np.resize(np.asarray(integer_values or [0], dtype=np.int32), shape)
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)

def run_decoder(path: Path) -> dict[str, Any]:
    import onnxruntime as ort
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    runs = []
    for context in ([0, 0], [1, 2], [6253, 42]):
        start = time.perf_counter()
        output = session.run(None, {input_info.name: make_input(input_info, 1, context)})[0]
        runs.append({"context": context, "shape": list(output.shape), "finite": bool(np.isfinite(output).all()), "seconds": time.perf_counter() - start, "output": output})
    difference = float(np.max(np.abs(runs[0]["output"] - runs[1]["output"])))
    difference = max(difference, float(np.max(np.abs(runs[1]["output"] - runs[2]["output"]))))
    return {"runs": [{key: value for key, value in item.items() if key != "output"} for item in runs], "output_changed": difference > 0.0, "max_difference": difference}

def run_joiner(path: Path) -> dict[str, Any]:
    import onnxruntime as ort
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    runs = []
    for seed in (7, 19):
        feed = {item.name: make_input(item, seed + index) for index, item in enumerate(session.get_inputs())}
        start = time.perf_counter()
        output = session.run(None, feed)[0]
        runs.append({"seed": seed, "shape": list(output.shape), "finite": bool(np.isfinite(output).all()), "seconds": time.perf_counter() - start, "output": output})
    difference = float(np.max(np.abs(runs[0]["output"] - runs[1]["output"])))
    return {"runs": [{key: value for key, value in item.items() if key != "output"} for item in runs], "output_changed": difference > 0.0, "max_difference": difference}

def run_encoder(path: Path) -> dict[str, Any]:
    import onnxruntime as ort
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    first_feed = {item.name: make_input(item, 17) for item in inputs}
    first_feed[inputs[0].name] = make_input(inputs[0], 17)
    start = time.perf_counter()
    first = session.run(None, first_feed)
    first_time = time.perf_counter() - start
    output_by_name = {item.name: value for item, value in zip(session.get_outputs(), first)}
    second_feed = {item.name: make_input(item, 23) for item in inputs}
    mapped = 0
    for item in inputs[1:]:
        output_name = "new_" + item.name
        if output_name not in output_by_name:
            raise RuntimeError(f"missing state output for {item.name}")
        state = output_by_name[output_name]
        expected_shape = tuple(1 if not isinstance(dim, int) else dim for dim in item.shape)
        type_name = item.type.replace("tensor(", "").replace(")", "")
        expected_dtype = {"float": np.float32, "double": np.float64, "int64": np.int64, "int32": np.int32}[type_name]
        if tuple(state.shape) != expected_shape or state.dtype != np.dtype(expected_dtype):
            raise RuntimeError(f"state mismatch {item.name}: {state.shape}/{state.dtype}")
        second_feed[item.name] = state
        mapped += 1
    start = time.perf_counter()
    second = session.run(None, second_feed)
    second_time = time.perf_counter() - start
    changed = sum(not np.array_equal(a, b) for a, b in zip(first[1:], second[1:]))
    return {"runs": [{"chunk": 1, "seconds": first_time, "outputs": [list(x.shape) for x in first]}, {"chunk": 2, "seconds": second_time, "outputs": [list(x.shape) for x in second]}], "state_mapped": mapped, "state_count": len(inputs) - 1, "state_changed_count": changed, "finite": all(bool(np.isfinite(x).all()) for x in first + second)}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()
    root = Path(args.dir)
    result = {"decoder": run_decoder(next(root.glob("decoder*.onnx"))), "joiner": run_joiner(next(root.glob("joiner*.onnx"))), "encoder": run_encoder(next(root.glob("encoder*.onnx")))}
    print(json.dumps(result, indent=2, allow_nan=False))
    valid = result["decoder"]["output_changed"] and result["joiner"]["output_changed"] and result["encoder"]["state_mapped"] == result["encoder"]["state_count"] and result["encoder"]["state_changed_count"] > 0 and result["encoder"]["finite"]
    return 0 if valid else 1

if __name__ == "__main__":
    raise SystemExit(main())
