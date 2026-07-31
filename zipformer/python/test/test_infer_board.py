import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from infer_board import (EngineAdapter, ManifestError, build_argument_parser,
                         greedy_decode, kaldifeat_fbank, load_features,
                         load_manifest, load_tokens, make_chunks,
                         make_combined_engines, read_wav, token_text)

ROOT = Path(__file__).parents[2]


class FakeEngine:
    def __init__(self, manifest, logical, tokens):
        self.manifest = manifest
        self.logical = logical
        self.actual = "zipformer_" + logical
        self.tokens = iter(tokens)
        self.calls = []
        self.encoder_calls = 0

    def get_graph_names(self):
        return [self.actual]

    def _spec(self):
        return self.manifest["networks"][self.logical]

    def _check_graph(self, graph):
        if graph != self.actual:
            raise AssertionError(f"logical name leaked to backend: {graph}")

    def get_input_names(self, graph):
        self._check_graph(graph)
        return [x["name"] for x in self._spec()["inputs"]]

    def get_output_names(self, graph):
        self._check_graph(graph)
        return [x["name"] for x in self._spec()["outputs"]]

    def get_input_shape(self, graph, name):
        self._check_graph(graph)
        return next(x["shape"] for x in self._spec()["inputs"] if x["name"] == name)

    def get_output_shape(self, graph, name):
        self._check_graph(graph)
        return next(x["shape"] for x in self._spec()["outputs"] if x["name"] == name)

    def get_input_dtype(self, graph, name):
        self._check_graph(graph)
        return "BM_FLOAT32" if name != "token_ids" else "Dtype.BM_INT32"

    def get_output_dtype(self, graph, name):
        self._check_graph(graph)
        return "BM_FLOAT32"

    def process(self, graph, tensors):
        self._check_graph(graph)
        self.calls.append(tensors)
        spec = self._spec()
        if self.logical == "joiner":
            output = np.zeros(spec["outputs"][0]["shape"], np.float32)
            output.reshape(-1)[next(self.tokens)] = 10
            return {"logit": output}
        if self.logical == "decoder":
            return {"decoder_out": np.zeros((1, 512), np.float32)}
        self.encoder_calls += 1
        value = float(self.encoder_calls)
        return {item["name"]: np.full(item["shape"], value, np.float32) for item in spec["outputs"]}


class InferBoardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_manifest(ROOT / "configs/tensor_manifest.json")

    def _adapter(self, joiner_tokens):
        engines = [
            FakeEngine(self.manifest, "encoder", [0] * 48),
            FakeEngine(self.manifest, "decoder", []),
            FakeEngine(self.manifest, "joiner", joiner_tokens),
        ]
        return EngineAdapter(make_combined_engines(engines), self.manifest), engines

    def test_f16_output_suffixes_are_explicitly_mapped(self):
        expected = ["encoder_out", "new_cached_len_0"]
        actual = ["encoder_out_Transpose_f32", "new_cached_len_0_Unsqueeze_f32"]
        self.assertEqual(
            EngineAdapter._resolve_backend_names(actual, expected, "output"),
            {"encoder_out": actual[0], "new_cached_len_0": actual[1]},
        )
        with self.assertRaises(ManifestError):
            EngineAdapter._resolve_backend_names(["encoder_out_f32"], ["encoder_out"],
                                                 "output")

    def test_manifest_dtype_and_actual_graph_are_checked(self):
        adapter, engines = self._adapter([0] * 48)
        with self.assertRaises(ManifestError):
            adapter.process("decoder", {"token_ids": np.zeros((1, 2), np.int32)})
        adapter.process("decoder", {"token_ids": np.zeros((1, 2), np.int64)})
        self.assertEqual(engines[1].calls[-1]["token_ids"].dtype, np.dtype("int32"))

    def test_chunk_window_hop_and_padding(self):
        chunks = make_chunks(np.ones((104, 80), np.float32), segment=103, offset=96,
                             tail_padding_seconds=0)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(np.all(chunks[1][:8] == 1) and np.all(chunks[1][8:] == 0))

    def test_state_continues_for_two_chunks_and_dump_contains_states(self):
        adapter, engines = self._adapter([0, 2, 7, 7] + [0] * 44)
        with tempfile.TemporaryDirectory() as directory:
            ids = greedy_decode(adapter, [np.zeros((103, 80), np.float32)] * 2,
                                self.manifest, directory)
            self.assertEqual(ids, [7, 7])
            self.assertEqual(len(engines[0].calls), 2)
            self.assertTrue(np.all(engines[0].calls[1]["cached_len_0"] == 1))
            self.assertTrue((Path(directory) / "chunk_0000_encoder_out.npy").exists())
            self.assertTrue((Path(directory) / "chunk_0001_new_cached_len_0.npy").exists())

    def test_decoder_runs_initially_and_only_after_nonblank(self):
        adapter, engines = self._adapter([0, 2] + [0] * 46)
        greedy_decode(adapter, [np.zeros((103, 80), np.float32)], self.manifest)
        self.assertEqual(len(engines[1].calls), 1)

    def test_tokens_txt_and_sentencepiece_marker(self):
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as stream:
            for token_id in range(6254):
                symbol = "▁hello" if token_id == 1 else ("▁x" if token_id == 3 else "x")
                stream.write(f"{symbol} {token_id}\n")
            path = stream.name
        try:
            tokens = load_tokens(path, 6254)
        finally:
            Path(path).unlink()
        self.assertEqual(token_text([1, 6254, 3], tokens, 6254), "hello x")

    def test_kaldifeat_options_put_frequencies_in_mel_options(self):
        captured = []

        class Options:
            def __init__(self):
                self.frame_opts = type("Frame", (), {})()
                self.mel_opts = type("Mel", (), {})()

        class FakeKaldifeat:
            FbankOptions = Options

            @staticmethod
            def Fbank(options):
                captured.append(options)
                return lambda tensor: np.zeros((1, 80), np.float32)

        class FakeTorch:
            @staticmethod
            def from_numpy(array):
                return type("Tensor", (), {"float": lambda self: self})()

        kaldifeat_fbank(np.zeros(160, np.float32), self.manifest["frontend"],
                        FakeKaldifeat, FakeTorch)
        self.assertEqual(captured[0].mel_opts.low_freq, 20.0)
        self.assertEqual(captured[0].mel_opts.high_freq, -400.0)

    def test_cli_requires_exactly_one_feature_source(self):
        parser = build_argument_parser()
        common = ["--encoder", "e", "--decoder", "d", "--joiner", "j",
                  "--tokens", "t"]
        self.assertEqual(parser.parse_args(common + ["--wav", "a.wav"]).wav, "a.wav")
        self.assertEqual(parser.parse_args(common + ["--features-npy", "a.npy"]).features_npy,
                         "a.npy")
        with self.assertRaises(SystemExit):
            parser.parse_args(common)
        with self.assertRaises(SystemExit):
            parser.parse_args(common + ["--wav", "a.wav", "--features-npy", "a.npy"])

    def test_precomputed_features_are_strictly_validated(self):
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as stream:
            path = Path(stream.name)
        try:
            expected = np.ones((103, 80), np.float32)
            np.save(path, expected, allow_pickle=False)
            np.testing.assert_array_equal(load_features(path, 80), expected)
            for bad in (
                np.ones((103, 80), np.float64),
                np.ones((80,), np.float32),
                np.ones((0, 80), np.float32),
                np.ones((103, 79), np.float32),
            ):
                np.save(path, bad, allow_pickle=False)
                with self.assertRaises(ValueError):
                    load_features(path, 80)
            bad = np.ones((103, 80), np.float32)
            bad[0, 0] = np.nan
            np.save(path, bad, allow_pickle=False)
            with self.assertRaises(ValueError):
                load_features(path, 80)
        finally:
            path.unlink(missing_ok=True)

    def test_multichannel_wav_is_averaged(self):
        import wave
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as stream:
            path = stream.name
        try:
            with wave.open(path, "wb") as wav:
                wav.setnchannels(2)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(np.array([[1000, -1000], [2000, 2000]], dtype="<i2").tobytes())
            np.testing.assert_allclose(read_wav(path), [0, 2000 / 32768], rtol=0, atol=1e-7)
        finally:
            Path(path).unlink()


if __name__ == "__main__":
    unittest.main()
