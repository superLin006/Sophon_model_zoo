#!/usr/bin/env python3
"""Manifest-driven TPU-MLIR argument and ONNX protocol checks."""
import argparse
import json
from pathlib import Path


def load_network(manifest_path, network):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    try:
        spec = manifest["networks"][network]
    except KeyError as exc:
        raise ValueError(f"unknown network: {network}") from exc
    for direction in ("inputs", "outputs"):
        tensors = spec[direction]
        indices = [item["index"] for item in tensors]
        expected = list(range(len(tensors)))
        if indices != expected:
            raise ValueError(f"{network}.{direction} indices must be contiguous from zero: {indices}")
    return manifest, spec


def input_shapes(manifest_path, network):
    _, spec = load_network(manifest_path, network)
    return [item["shape"] for item in spec["inputs"]]


def _onnx_dtype(onnx, elem_type):
    names = {
        "FLOAT": "float32", "DOUBLE": "float64", "FLOAT16": "float16",
        "BFLOAT16": "bfloat16", "INT8": "int8", "UINT8": "uint8",
        "INT16": "int16", "UINT16": "uint16", "INT32": "int32",
        "UINT32": "uint32", "INT64": "int64", "UINT64": "uint64",
        "BOOL": "bool",
    }
    name = onnx.TensorProto.DataType.Name(elem_type)
    return names.get(name, name.lower())


def _value_info(onnx, value):
    dims = []
    for dim in value.type.tensor_type.shape.dim:
        if not dim.HasField("dim_value") or dim.dim_value <= 0:
            raise ValueError(f"ONNX tensor {value.name!r} has dynamic/invalid shape")
        dims.append(dim.dim_value)
    return value.name, dims, _onnx_dtype(onnx, value.type.tensor_type.elem_type)


def inspect_onnx(model_path, manifest_path, network):
    """Require a valid static ONNX graph matching all manifest inputs and outputs."""
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("ONNX inspection requires the TPU-MLIR image with onnx installed") from exc
    _, spec = load_network(manifest_path, network)
    model = onnx.load(str(model_path), load_external_data=False)
    onnx.checker.check_model(model)
    initializers = {item.name for item in model.graph.initializer}
    actual_inputs = [_value_info(onnx, value) for value in model.graph.input if value.name not in initializers]
    actual_outputs = [_value_info(onnx, value) for value in model.graph.output]
    expected_inputs = [(item["name"], item["shape"], item["dtype"]) for item in spec["inputs"]]
    expected_outputs = [(item["name"], item["shape"], item["dtype"]) for item in spec["outputs"]]
    if actual_inputs != expected_inputs:
        raise ValueError(
            f"{network}: ONNX inputs do not match manifest (name, shape, dtype); "
            f"expected {expected_inputs[:2]}..., got {actual_inputs[:2]}..."
        )
    if actual_outputs != expected_outputs:
        raise ValueError(
            f"{network}: ONNX outputs do not match manifest (name, shape, dtype); "
            f"expected {expected_outputs[:2]}..., got {actual_outputs[:2]}..."
        )
    return {"inputs": actual_inputs, "outputs": actual_outputs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--network", required=True, choices=("encoder", "decoder", "joiner"))
    parser.add_argument("--onnx", type=Path)
    args = parser.parse_args()
    shapes = input_shapes(args.manifest, args.network)
    if args.onnx:
        inspect_onnx(args.onnx, args.manifest, args.network)
    print(json.dumps(shapes, separators=(",", ":")))


if __name__ == "__main__":
    main()
