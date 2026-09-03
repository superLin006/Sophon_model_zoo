"""Zipformer tensor layout: the single source of truth for the manifest."""
import argparse
import json
from pathlib import Path

SAMPLE_RATE = 16000
N_MELS = 80
SEGMENT = 103
OFFSET = 96
VOCAB_SIZE = 6254
BLANK_ID = 0
UNK_ID = 2
CONTEXT_SIZE = 2
MODEL_ID = "csukuangfj/k2fsa-zipformer-bilingual-zh-en-t"
SHORT_NAME = "zipformer_bilingual"
DTYPE = "float32"

FRONTEND = {
    "sample_rate": SAMPLE_RATE,
    "n_mels": N_MELS,
    "segment": SEGMENT,
    "offset": OFFSET,
    "frame_shift_ms": 10.0,
    "frame_length_ms": 25.0,
    "dither": 0.0,
    "preemphasis_coefficient": 0.97,
    "window_type": "povey",
    "snip_edges": False,
    "remove_dc_offset": True,
    "low_frequency": 20.0,
    "high_frequency": -400.0,
    "use_energy": False,
    "use_log_fbank": True,
    "use_power": True,
    "tail_padding_seconds": 1.03,
}

_DIMS = (128, 64, 32, 16, 64)


def _external_shape(shape):
    """Add the real batch-1 axis outside the streaming layer/cache dimensions."""
    return (1,) + tuple(shape)


def _state_specs():
    specs = []
    for group, shapes in (
        ("cached_len", [(2, 1)] * 5),
        ("cached_avg", [(2, 1, 256)] * 5),
        ("cached_key", [(2, n, 1, 192) for n in _DIMS]),
        ("cached_val", [(2, n, 1, 96) for n in _DIMS]),
        ("cached_val2", [(2, n, 1, 96) for n in _DIMS]),
        ("cached_conv1", [(2, 1, 256, 30)] * 5),
        ("cached_conv2", [(2, 1, 256, 30)] * 5),
    ):
        specs.extend({"name": f"{group}_{i}", "shape": list(_external_shape(shape)),
                      "inner_shape": list(shape),
                      "element_count": _element_count(_external_shape(shape)),
                      "inner_element_count": _element_count(shape), "dtype": DTYPE}
                     for i, shape in enumerate(shapes))
    return specs

def state_specs():
    return [dict(s) for s in _state_specs()]

def _tensor(name, index, shape, dtype=DTYPE, notes=None, runtime_dtype=None):
    tensor = {"name": name, "index": index, "shape": list(shape),
              "element_count": _element_count(shape), "dtype": dtype}
    if runtime_dtype is not None:
        tensor["runtime_dtype"] = runtime_dtype
    if notes is not None:
        tensor["notes"] = notes
    return tensor

def build_manifest():
    states = state_specs()
    enc_inputs = [_tensor("x", 0, (1, SEGMENT, N_MELS))]
    enc_inputs += [_tensor(s["name"], i + 1, s["shape"]) for i, s in enumerate(states)]
    enc_outputs = [_tensor("encoder_out", 0, (1, 24, 256))]
    enc_outputs += [_tensor("new_" + s["name"], i + 1, s["shape"]) for i, s in enumerate(states)]
    return {
        "schema_version": 1,
        "model": {"id": MODEL_ID, "short_name": SHORT_NAME, "family": "zipformer"},
        "state_layout": {
            "envelope": "sophon_external_batch1",
            "external_batch": 1,
            "inner_semantics": "streaming stack layer axis remains the leading dimension 2; it is not batch",
            "unwrap": "state.squeeze(0) before EncoderStreaming; state.unsqueeze(0) after EncoderStreaming",
        },
        "networks": {
            "encoder": {"inputs": enc_inputs, "outputs": enc_outputs},
            "decoder": {"inputs": [_tensor("token_ids", 0, (1, 2), dtype="int64",
                                           runtime_dtype="int32",
                                           notes="ONNX Gather uses int64; BM1684X runtime lowers token IDs to int32")],
                        "outputs": [_tensor("decoder_out", 0, (1, 512))]},
            "joiner": {"inputs": [_tensor("enc_out", 0, (1, 256)), _tensor("dec_out", 1, (1, 512))],
                       "outputs": [_tensor("logit", 0, (1, VOCAB_SIZE))]},
        },
        "frontend": dict(FRONTEND),
        "decoding": {"vocab_size": VOCAB_SIZE, "blank_id": BLANK_ID,
                      "unk_id": UNK_ID, "context_size": CONTEXT_SIZE},
    }

def validate_manifest(manifest):
    expected = build_manifest()
    assert manifest.get("schema_version") == expected["schema_version"]
    assert manifest.get("model") == expected["model"]
    assert manifest.get("frontend") == expected["frontend"]
    assert manifest.get("decoding") == expected["decoding"]
    assert manifest.get("state_layout") == expected["state_layout"]
    for network in ("encoder", "decoder", "joiner"):
        actual = manifest.get("networks", {}).get(network)
        wanted = expected["networks"][network]
        assert actual is not None, network
        for direction in ("inputs", "outputs"):
            got = actual.get(direction)
            want = wanted[direction]
            assert got == want, f"{network}.{direction} does not match layout"
            for tensor in got:
                assert len(tensor["shape"]) > 0
                assert all(isinstance(x, int) and x > 0 for x in tensor["shape"])
                assert tensor["dtype"] in ("float32", "int32", "int64")
                if "runtime_dtype" in tensor:
                    assert tensor["runtime_dtype"] in ("float32", "int32", "int64")
                assert tensor["index"] >= 0
                assert tensor["element_count"] == _element_count(tensor["shape"])
                assert tensor["shape"] and tensor["element_count"] > 0
    assert len(manifest["networks"]["encoder"]["inputs"]) == 36
    assert len(manifest["networks"]["encoder"]["outputs"]) == 36
    assert len(state_specs()) == 35
    for inp, out in zip(manifest["networks"]["encoder"]["inputs"][1:],
                         manifest["networks"]["encoder"]["outputs"][1:]):
        assert out["name"] == "new_" + inp["name"]
        assert out["shape"] == inp["shape"]
        assert out["dtype"] == inp["dtype"]
    return True

def _element_count(shape):
    result = 1
    for dim in shape:
        result *= dim
    return result

def load_manifest(path):
    with Path(path).open(encoding="utf-8") as f:
        manifest = json.load(f)
    validate_manifest(manifest)
    return manifest

def main():
    parser = argparse.ArgumentParser(description="Validate the Zipformer tensor manifest")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).parents[1] / "configs" / "tensor_manifest.json")
    args = parser.parse_args()
    validate_manifest(json.loads(args.manifest.read_text(encoding="utf-8")))
    print(f"valid: {args.manifest} (35 states, 36 encoder inputs, 36 encoder outputs)")

if __name__ == "__main__":
    main()
