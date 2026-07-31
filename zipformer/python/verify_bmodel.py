#!/usr/bin/env python3
"""Numerically verify a TPU-MLIR bmodel against the same ONNX graph."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "tensor_manifest.json"

def compare_stats(first: np.ndarray, second: np.ndarray, top_k: int = 5) -> dict[str, Any]:
    result: dict[str, Any] = {
        "shape_a": list(first.shape), "shape_b": list(second.shape),
        "dtype_a": str(first.dtype), "dtype_b": str(second.dtype),
        "nan_a": int(np.isnan(first).sum()), "nan_b": int(np.isnan(second).sum()),
        "inf_a": int(np.isinf(first).sum()), "inf_b": int(np.isinf(second).sum()),
    }
    if first.shape != second.shape:
        result["error"] = "shape mismatch"
        return result
    if first.size == 0:
        result.update({"mae": 0.0, "max_abs": 0.0, "relative_l2": 0.0,
                       "cosine": None, "topk": [], "argmax_a": None, "argmax_b": None})
        return result
    left, right = first.astype(np.float64), second.astype(np.float64)
    delta = np.abs(left - right)
    result["mae"] = float(np.nanmean(delta))
    result["max_abs"] = float(np.nanmax(delta))
    result["relative_l2"] = float(
        np.linalg.norm(np.nan_to_num(delta)) /
        (np.linalg.norm(np.nan_to_num(right)) + 1e-12)
    )
    x, y = left.ravel(), right.ravel()
    nx, ny = np.linalg.norm(np.nan_to_num(x)), np.linalg.norm(np.nan_to_num(y))
    result["cosine"] = None if nx == 0 or ny == 0 else float(
        np.dot(np.nan_to_num(x), np.nan_to_num(y)) / (nx * ny)
    )
    indices = np.argsort(np.nan_to_num(delta).ravel())[-top_k:][::-1]
    result["topk"] = [
        {"index": int(index), "abs": float(delta.ravel()[index])}
        for index in indices
    ]
    result["argmax_a"] = int(np.nanargmax(first))
    result["argmax_b"] = int(np.nanargmax(second))
    return result


def load_manifest(path: Path, network: str) -> dict[str, Any]:
    data = json.loads(path.read_text()); net = data.get("networks", {}).get(network)
    if not isinstance(net, dict): raise ValueError(f"manifest has no valid network {network!r}")
    for kind in ("inputs", "outputs"):
        items = net.get(kind, []); names = [x.get("name") for x in items]; indexes = [x.get("index") for x in items]
        if any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names): raise ValueError(f"manifest {network} {kind} names are not unique")
        if sorted(indexes) != list(range(len(indexes))): raise ValueError(f"manifest {network} {kind} indices must be contiguous")
    return net

def dtype_from_manifest(item: Mapping[str, Any]) -> np.dtype:
    try: return np.dtype(item["dtype"])
    except (KeyError, TypeError) as exc: raise ValueError(f"invalid manifest dtype for {item.get('name')}") from exc

def validate_tensor(value: np.ndarray, item: Mapping[str, Any], where: str) -> np.ndarray:
    value = np.asarray(value); shape = tuple(int(x) for x in item["shape"]); dtype = dtype_from_manifest(item)
    if value.shape != shape: raise ValueError(f"{where}: {item['name']} shape {value.shape} != manifest {shape}")
    if value.dtype != dtype: raise ValueError(f"{where}: {item['name']} dtype {value.dtype} != manifest {dtype}")
    return value

def tensor_map(values: Mapping[str, np.ndarray], items: Sequence[Mapping[str, Any]], where: str) -> dict[str, np.ndarray]:
    """Resolve explicit runner aliases without falling back to output order."""
    result, used = {}, set()
    structural_suffixes = (
        "_Transpose", "_Unsqueeze", "_Squeeze", "_Gemm",
        "_Transpose_f32", "_Unsqueeze_f32", "_Squeeze_f32", "_Gemm_f32",
    )
    for item in items:
        name, index = item["name"], int(item["index"])
        aliases = (name, str(index), f"output_{index}") + tuple(
            name + suffix for suffix in structural_suffixes
        )
        found = [key for key in aliases if key in values]
        if len(found) != 1:
            state = "missing" if not found else "ambiguous"
            raise ValueError(
                f"{where}: {state} tensor {name!r} (aliases={found})"
            )
        key = found[0]
        if key in used:
            raise ValueError(f"{where}: tensor alias {key!r} used twice")
        used.add(key)
        result[name] = validate_tensor(values[key], item, where)
    extras = set(values) - used
    if extras:
        raise ValueError(f"{where}: unexpected tensor names: {sorted(extras)}")
    return result

def deterministic_inputs(items: Sequence[Mapping[str, Any]], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed); result = {}
    for item in items:
        dtype = dtype_from_manifest(item); shape = tuple(item["shape"])
        result[item["name"]] = (rng.integers(0, 6254, size=shape, dtype=dtype) if np.issubdtype(dtype, np.integer) else rng.standard_normal(shape).astype(dtype))
    return result

def zero_states(items: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    return {x["name"]: np.zeros(tuple(x["shape"]), dtype=dtype_from_manifest(x)) for x in items}

def metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    reference, actual = np.asarray(reference), np.asarray(actual)
    result = compare_stats(reference, actual)
    result["shape"] = list(actual.shape)
    result["finite"] = bool(np.isfinite(reference).all() and np.isfinite(actual).all())
    if reference.shape == actual.shape and reference.size == 0:
        result["cosine"] = 1.0
    elif reference.shape == actual.shape:
        ref_zero, actual_zero = np.all(reference == 0), np.all(actual == 0)
        if ref_zero and actual_zero: result["cosine"] = 1.0
        elif ref_zero or actual_zero: result["cosine"] = 0.0
    return result

def passed(stat: Mapping[str, Any], args: argparse.Namespace) -> bool:
    return bool(stat.get("finite") and "error" not in stat and stat.get("mae", float("inf")) <= args.mae and stat.get("max_abs", float("inf")) <= args.max_abs and stat.get("relative_l2", float("inf")) <= args.relative_l2 and stat.get("cosine") is not None and stat["cosine"] >= 1 - args.cosine_tol)

def run_runner(runner: str, model: Path, inputs: Mapping[str, np.ndarray], work_dir: Path, tag: str, invoke: Callable[..., Any] = subprocess.run) -> dict[str, np.ndarray]:
    work_dir.mkdir(parents=True, exist_ok=True); ip, op = work_dir / f"{tag}_input.npz", work_dir / f"{tag}_output.npz"; np.savez(ip, **inputs)
    command = [runner, "--model", str(model), "--input", str(ip), "--output", str(op)]
    try:
        invoke(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        details = "\n".join(
            value.strip()[-2000:]
            for value in (exc.stdout, exc.stderr)
            if value and value.strip()
        )
        suffix = f"\nrunner output:\n{details}" if details else ""
        raise RuntimeError(f"TPU-MLIR runner failed with exit code {exc.returncode}{suffix}") from exc
    if not op.exists(): raise RuntimeError(f"TPU-MLIR runner did not create {op}")
    with np.load(op, allow_pickle=False) as archive: return {k: np.array(archive[k], copy=True) for k in archive.files}

def ort_session(path: Path):
    import onnxruntime as ort
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

def ort_run(session: Any, input_items: Sequence[Mapping[str, Any]], output_items: Sequence[Mapping[str, Any]], feeds: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    expected = {x["name"] for x in input_items}; actual = {x.name for x in session.get_inputs()}
    if actual != expected: raise RuntimeError(f"ONNX input names differ from manifest: {sorted(actual ^ expected)}")
    for item in input_items: validate_tensor(feeds[item["name"]], item, "ONNX input")
    values = session.run(None, dict(feeds)); names = [x.name for x in session.get_outputs()]
    expected_outputs = [x["name"] for x in output_items]
    if names != expected_outputs: raise RuntimeError(f"ONNX output names differ from manifest: {names} != {expected_outputs}")
    return {item["name"]: validate_tensor(value, item, "ONNX output") for item, value in zip(output_items, values)}

def compare_run(network: str, onnx: Path, bmodel: Path, manifest: Path, work_dir: Path, runner: str, args: argparse.Namespace, session: Any = None, invoke: Callable[..., Any] = subprocess.run) -> tuple[bool, dict[str, Any]]:
    net = load_manifest(manifest, network)
    ins, outs = net["inputs"], net["outputs"]
    session = session or ort_session(onnx)
    reports, ok = [], True
    if network == "encoder":
        ref_feed = deterministic_inputs([ins[0]], 17) | zero_states(ins[1:])
        actual_feed = dict(ref_feed)
        ref_state, actual_state = None, None
        for chunk in range(2):
            if chunk == 1:
                ref_feed = deterministic_inputs([ins[0]], 23) | {x["name"]: ref_state["new_" + x["name"]] for x in ins[1:]}
                actual_feed = deterministic_inputs([ins[0]], 23) | {x["name"]: actual_state["new_" + x["name"]] for x in ins[1:]}
            ref = ort_run(session, ins, outs, ref_feed)
            actual = tensor_map(run_runner(runner, bmodel, actual_feed, work_dir, f"{network}_{chunk}", invoke), outs, f"bmodel {network} run {chunk}")
            stats = {x["name"]: metrics(ref[x["name"]], actual[x["name"]]) for x in outs}
            ok = ok and all(passed(stat, args) for stat in stats.values())
            reports.append({"chunk": chunk + 1, "propagated_state": chunk == 1, "tensors": stats})
            ref_state, actual_state = ref, actual
    else:
        if network == "decoder":
            contexts = ([0, 0], [1, 2], [6253, 42])
            feeds = [{ins[0]["name"]: np.asarray(c, dtype=np.int64).reshape(tuple(ins[0]["shape"]))} for c in contexts]
        else:
            feeds = [deterministic_inputs(ins, seed) for seed in (7, 19, 31)]
        for index, feed in enumerate(feeds):
            ref = ort_run(session, ins, outs, feed)
            actual = tensor_map(run_runner(runner, bmodel, feed, work_dir, f"{network}_{index}", invoke), outs, f"bmodel {network} run {index}")
            stats = {x["name"]: metrics(ref[x["name"]], actual[x["name"]]) for x in outs}
            ok = ok and all(passed(stat, args) for stat in stats.values())
            reports.append({"run": index, "tensors": stats})
    return bool(ok), {"network": network, "runs": reports}

def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--network", choices=("encoder", "decoder", "joiner"), required=True); p.add_argument("--onnx", type=Path, required=True); p.add_argument("--bmodel", type=Path, required=True); p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST); p.add_argument("--work-dir", type=Path, required=True); p.add_argument("--runner", default="model_runner.py"); p.add_argument("--mae", type=float, default=1e-4); p.add_argument("--max-abs", type=float, default=1e-4); p.add_argument("--relative-l2", type=float, default=1e-3); p.add_argument("--cosine-tol", type=float, default=1e-4); args = p.parse_args(argv)
    try:
        for path in (args.onnx, args.bmodel, args.manifest):
            if not path.is_file(): raise FileNotFoundError(path)
        ok, report = compare_run(args.network, args.onnx, args.bmodel, args.manifest, args.work_dir, args.runner, args)
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0 if ok else 1
    except Exception as exc: print(f"verify_bmodel: FAIL: {exc}", file=sys.stderr); return 1

if __name__ == "__main__": raise SystemExit(main())
