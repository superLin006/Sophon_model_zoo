#!/usr/bin/env python3
"""F32 BM1684X Zipformer reference inference through sophon.sail."""
from __future__ import annotations

import argparse
import json
import time
import wave
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


class ManifestError(ValueError):
    pass


LOGICAL_GRAPHS = ("encoder", "decoder", "joiner")


def load_manifest(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    for network in LOGICAL_GRAPHS:
        spec = manifest.get("networks", {}).get(network)
        if not spec:
            raise ManifestError(f"manifest missing network {network!r}")
        for kind in ("inputs", "outputs"):
            if not spec.get(kind) or any(
                not all(k in item for k in ("name", "shape", "dtype"))
                for item in spec[kind]
            ):
                raise ManifestError(f"manifest {network}.{kind} must contain name/shape/dtype")
    return manifest


def normalize_dtype(value: Any) -> str:
    """Normalize numpy, sail enum, and common sail dtype spellings."""
    text = str(value).lower().replace("numpy.", "").replace("bm_", "")
    aliases = {
        "fp32": "float32", "f32": "float32", "float": "float32",
        "int64_t": "int64", "i64": "int64", "long": "int64",
    }
    text = aliases.get(text, text)
    try:
        return np.dtype(text).name
    except TypeError:
        if "float32" in text:
            return "float32"
        if "int64" in text:
            return "int64"
        if "int32" in text:
            return "int32"
        return text


def _dtype(value: str) -> np.dtype:
    try:
        return np.dtype(value)
    except TypeError as exc:
        raise ManifestError(f"unsupported manifest dtype {value!r}") from exc


class EngineAdapter:
    """Validate logical manifest names while the backend may use actual graph names."""

    def __init__(self, engine: Any, manifest: Mapping[str, Any]):
        self.engine = engine
        self.manifest = manifest
        self.graphs = list(engine.get_graph_names())
        self.io_names = {}
        for logical in LOGICAL_GRAPHS:
            if logical not in self.graphs:
                raise ManifestError(f"backend has no logical graph {logical!r}: {self.graphs}")
            self._check_io(logical)

    def _names(self, logical: str, kind: str) -> list[str]:
        return list(getattr(self.engine, f"get_{kind}_names")(logical))

    @staticmethod
    def _resolve_backend_names(actual: list[str], expected: list[str], kind: str) -> dict[str, str]:
        if actual == expected:
            return dict(zip(expected, actual))
        suffixes = (
            "_Transpose", "_Unsqueeze", "_Squeeze", "_Gemm",
            "_Transpose_f32", "_Unsqueeze_f32", "_Squeeze_f32", "_Gemm_f32",
        ) if kind == "output" else ()
        mapping = {}
        used = set()
        for name in expected:
            matches = [candidate for candidate in actual
                       if candidate == name or any(candidate == name + suffix for suffix in suffixes)]
            if len(matches) != 1 or matches[0] in used:
                raise ManifestError(
                    f"cannot map backend {kind} for {name!r}: matches={matches}"
                )
            mapping[name] = matches[0]
            used.add(matches[0])
        if used != set(actual):
            raise ManifestError(f"unexpected backend {kind} names: {sorted(set(actual) - used)}")
        return mapping

    def _check_io(self, logical: str) -> None:
        spec = self.manifest["networks"][logical]
        for kind in ("input", "output"):
            actual = self._names(logical, kind)
            expected = [item["name"] for item in spec[kind + "s"]]
            name_map = self._resolve_backend_names(actual, expected, kind)
            self.io_names[(logical, kind)] = name_map
            shape_fn = getattr(self.engine, f"get_{kind}_shape", None)
            dtype_fn = getattr(self.engine, f"get_{kind}_dtype", None)
            for item in spec[kind + "s"]:
                name = item["name"]
                backend_name = name_map[name]
                if shape_fn is not None and list(shape_fn(logical, backend_name)) != item["shape"]:
                    raise ManifestError(f"{logical}/{name} shape differs from manifest")
                runtime_dtype = item.get("runtime_dtype", item["dtype"])
                if dtype_fn is not None and normalize_dtype(dtype_fn(logical, backend_name)) != normalize_dtype(runtime_dtype):
                    raise ManifestError(f"{logical}/{name} dtype differs from manifest runtime dtype")

    def process(self, logical: str, tensors: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        spec = self.manifest["networks"][logical]
        checked = {}
        for item in spec["inputs"]:
            name = item["name"]
            if name not in tensors:
                raise ManifestError(f"missing {logical} input {name}")
            array = np.asarray(tensors[name])
            if list(array.shape) != item["shape"] or normalize_dtype(array.dtype) != normalize_dtype(item["dtype"]):
                raise ManifestError(f"bad {logical}/{name}: shape={array.shape}, dtype={array.dtype}")
            runtime_dtype = _dtype(item.get("runtime_dtype", item["dtype"]))
            checked[name] = array.astype(runtime_dtype, copy=False)
        backend_inputs = {
            self.io_names[(logical, "input")][name]: value
            for name, value in checked.items()
        }
        result = self.engine.process(logical, backend_inputs)
        output_map = self.io_names[(logical, "output")]
        if isinstance(result, Mapping):
            raw_outputs = dict(result)
            outputs = {name: raw_outputs[backend_name]
                       for name, backend_name in output_map.items()}
            extras = set(raw_outputs) - set(output_map.values())
            if extras:
                raise ManifestError(f"unexpected {logical} outputs: {sorted(extras)}")
        else:
            backend_names = self._names(logical, "output")
            raw_outputs = dict(zip(backend_names, result))
            outputs = {name: raw_outputs[backend_name]
                       for name, backend_name in output_map.items()}
        for item in spec["outputs"]:
            name = item["name"]
            array = np.asarray(outputs[name])
            if list(array.shape) != item["shape"] or normalize_dtype(array.dtype) != normalize_dtype(item["dtype"]):
                raise ManifestError(f"bad {logical} output {name}: shape={array.shape}, dtype={array.dtype}")
            outputs[name] = array
        return outputs


def read_wav(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        if wav.getframerate() != sample_rate or wav.getsampwidth() != 2:
            raise ValueError(f"WAV must be 16-bit PCM at {sample_rate} Hz")
        channels = wav.getnchannels()
        if channels < 1:
            raise ValueError("WAV has no channels")
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    samples = samples.reshape(-1, channels).astype(np.float32) / 32768.0
    return samples.mean(axis=1)


def load_features(path: str | Path, n_mels: int) -> np.ndarray:
    features = np.load(path, allow_pickle=False)
    if features.dtype != np.float32:
        raise ValueError(f"features must have dtype float32, got {features.dtype}")
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D [frames, mel], got shape {features.shape}")
    if features.shape[0] < 1:
        raise ValueError("features must contain at least one frame")
    if features.shape[1] != n_mels:
        raise ValueError(f"features must have {n_mels} mel bins, got {features.shape[1]}")
    if not np.isfinite(features).all():
        raise ValueError("features contain non-finite values")
    return features


def kaldifeat_fbank(samples: np.ndarray, frontend: Mapping[str, Any], kaldifeat_module=None, torch_module=None) -> np.ndarray:
    try:
        if kaldifeat_module is None:
            import kaldifeat as kaldifeat_module
        if torch_module is None:
            import torch as torch_module
    except ImportError as exc:
        raise RuntimeError("kaldifeat frontend requires torch and kaldifeat; refusing inconsistent fallback") from exc
    options = kaldifeat_module.FbankOptions()
    frame = options.frame_opts
    mel = options.mel_opts
    frame_values = {
        "samp_freq": "sample_rate", "frame_shift_ms": "frame_shift_ms",
        "frame_length_ms": "frame_length_ms", "dither": "dither",
        "preemph_coeff": "preemphasis_coefficient", "window_type": "window_type",
        "snip_edges": "snip_edges", "remove_dc_offset": "remove_dc_offset",
    }
    for destination, source in frame_values.items():
        setattr(frame, destination, frontend[source])
    mel.low_freq = frontend["low_frequency"]
    mel.high_freq = frontend["high_frequency"]
    mel.num_bins = frontend["n_mels"]
    options.use_energy = frontend["use_energy"]
    options.use_log_fbank = frontend["use_log_fbank"]
    options.use_power = frontend["use_power"]
    tensor = torch_module.from_numpy(np.asarray(samples, dtype=np.float32)).float()
    return np.asarray(kaldifeat_module.Fbank(options)(tensor))


def make_chunks(features: np.ndarray, segment=103, offset=96, tail_padding_seconds=1.03,
                sample_rate=16000, frame_shift_ms=10.0) -> list[np.ndarray]:
    if features.ndim != 2:
        raise ValueError("features must be [frames, mel]")
    padding = int(round(tail_padding_seconds * 1000.0 / frame_shift_ms))
    padded = np.pad(features, ((0, padding), (0, 0)))
    count = max(1, int(np.ceil(max(0, len(padded) - segment) / offset)) + 1)
    chunks = []
    for index in range(count):
        chunk = padded[index * offset:index * offset + segment]
        chunks.append(np.pad(chunk, ((0, max(0, segment - len(chunk))), (0, 0))))
    return chunks


def load_tokens(path: str | Path, vocab_size: int) -> dict[int, str]:
    tokens = {}
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.strip().split()
            if not fields or line.lstrip().startswith("#"):
                continue
            if len(fields) < 2:
                raise ValueError(f"tokens.txt line {line_number} must be 'symbol id'")
            symbol, raw_id = fields[0], fields[-1]
            try:
                token_id = int(raw_id)
            except ValueError as exc:
                raise ValueError(f"invalid token ID on line {line_number}") from exc
            if token_id in tokens:
                raise ValueError(f"duplicate token ID {token_id}")
            tokens[token_id] = symbol
    missing = [token_id for token_id in range(vocab_size) if token_id not in tokens]
    if missing:
        raise ValueError(f"tokens.txt missing joiner IDs, first missing: {missing[:5]}")
    return tokens


def token_text(ids: Sequence[int], tokens: Mapping[int, str], vocab_size: int) -> str:
    pieces = [tokens[int(token)] for token in ids if 0 <= int(token) < vocab_size and int(token) in tokens]
    text = "".join(pieces).replace("▁", " ")
    return text.lstrip()


def greedy_decode(adapter, chunks, manifest, dump_dir=None, dump_io=False):
    encoder = manifest["networks"]["encoder"]
    decoder = manifest["networks"]["decoder"]
    joiner = manifest["networks"]["joiner"]
    states = {
        item["name"]: np.zeros(item["shape"], dtype=_dtype(item["dtype"]))
        for item in encoder["inputs"] if item["name"] != "x"
    }
    blank = manifest["decoding"]["blank_id"]
    unk = manifest["decoding"]["unk_id"]
    vocab_size = manifest["decoding"]["vocab_size"]
    context = [blank, blank]
    decoder_output = adapter.process("decoder", {"token_ids": np.asarray([context], dtype=np.int64)})
    decoder_value = decoder_output[decoder["outputs"][0]["name"]]
    output_ids, latencies = [], []
    dump_path = Path(dump_dir) if dump_dir else None
    if dump_path:
        dump_path.mkdir(parents=True, exist_ok=True)
        if dump_io:
            np.save(dump_path / "decoder_initial_decoder_out.npy", decoder_value)
    for chunk_index, chunk in enumerate(chunks):
        started = time.perf_counter()
        encoder_output = adapter.process("encoder", {"x": chunk[None].astype(np.float32), **states})
        encoder_value = encoder_output[encoder["outputs"][0]["name"]]
        new_states = {}
        for item in encoder["outputs"]:
            if item["name"].startswith("new_"):
                new_states[item["name"][4:]] = encoder_output[item["name"]]
                if dump_path:
                    np.save(dump_path / f"chunk_{chunk_index:04d}_{item['name']}.npy", encoder_output[item["name"]])
        states = new_states
        if dump_path:
            np.save(dump_path / f"chunk_{chunk_index:04d}_encoder_out.npy", encoder_value)
        for frame_index, frame in enumerate(encoder_value[0]):
            joiner_output = adapter.process("joiner", {
                "enc_out": frame[None].astype(np.float32),
                "dec_out": decoder_value.astype(np.float32),
            })
            logits = joiner_output[joiner["outputs"][0]["name"]]
            if dump_path and dump_io:
                np.save(dump_path / f"chunk_{chunk_index:04d}_frame_{frame_index:04d}_joiner_logit.npy", logits)
            token = int(np.argmax(logits.reshape(-1)))
            if token in (blank, unk) or token >= vocab_size:
                continue
            output_ids.append(token)
            context = [context[-1], token]
            decoder_output = adapter.process("decoder", {"token_ids": np.asarray([context], dtype=np.int64)})
            decoder_value = decoder_output[decoder["outputs"][0]["name"]]
            if dump_path and dump_io:
                np.save(dump_path / f"chunk_{chunk_index:04d}_frame_{frame_index:04d}_decoder_out.npy", decoder_value)
        latencies.append(time.perf_counter() - started)
    print("chunk latency(ms): " + ", ".join(f"{value * 1000:.2f}" for value in latencies))
    return output_ids


def make_combined_engines(engines: Sequence[Any], logical_names=LOGICAL_GRAPHS):
    mapping = {}
    for logical, engine in zip(logical_names, engines):
        actual = list(engine.get_graph_names())
        expected = f"zipformer_{logical}"
        if expected not in actual:
            raise ManifestError(f"{logical} bmodel must expose {expected!r}, found {actual}")
        mapping[logical] = (engine, expected)

    class Combined:
        def get_graph_names(self):
            return list(mapping)

        def _target(self, logical):
            return mapping[logical]

        def get_input_names(self, logical):
            engine, actual = self._target(logical)
            return engine.get_input_names(actual)

        def get_output_names(self, logical):
            engine, actual = self._target(logical)
            return engine.get_output_names(actual)

        def get_input_shape(self, logical, name):
            engine, actual = self._target(logical)
            return engine.get_input_shape(actual, name)

        def get_output_shape(self, logical, name):
            engine, actual = self._target(logical)
            return engine.get_output_shape(actual, name)

        def get_input_dtype(self, logical, name):
            engine, actual = self._target(logical)
            return engine.get_input_dtype(actual, name)

        def get_output_dtype(self, logical, name):
            engine, actual = self._target(logical)
            return engine.get_output_dtype(actual, name)

        def process(self, logical, tensors):
            engine, actual = self._target(logical)
            return engine.process(actual, tensors)

    return Combined()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--decoder", required=True)
    parser.add_argument("--joiner", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wav")
    source.add_argument("--features-npy")
    parser.add_argument("--manifest", default=str(Path(__file__).parents[1] / "configs/tensor_manifest.json"))
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--dump-dir")
    parser.add_argument("--dump-io", action="store_true")
    return parser


def main():
    args = build_argument_parser().parse_args()
    manifest = load_manifest(args.manifest)
    try:
        import sophon.sail as sail
    except ImportError as exc:
        raise RuntimeError("sophon.sail is required on BM1684X; no emulation is provided") from exc
    engines = [sail.Engine(path, 0, sail.IOMode.SYSIO) for path in (args.encoder, args.decoder, args.joiner)]
    adapter = EngineAdapter(make_combined_engines(engines), manifest)
    frontend = manifest["frontend"]
    if args.features_npy:
        features = load_features(args.features_npy, frontend["n_mels"])
    else:
        samples = read_wav(args.wav, frontend["sample_rate"])
        samples = np.pad(samples, (0, int(round(frontend["tail_padding_seconds"] * frontend["sample_rate"]))))
        features = kaldifeat_fbank(samples, frontend)
    chunks = make_chunks(features, frontend["segment"], frontend["offset"], 0,
                         frontend["sample_rate"], frontend["frame_shift_ms"])
    tokens = load_tokens(args.tokens, manifest["decoding"]["vocab_size"])
    ids = greedy_decode(adapter, chunks, manifest, args.dump_dir, args.dump_io)
    print("tokens:", ids)
    print("text:", token_text(ids, tokens, manifest["decoding"]["vocab_size"]))


if __name__ == "__main__":
    main()
