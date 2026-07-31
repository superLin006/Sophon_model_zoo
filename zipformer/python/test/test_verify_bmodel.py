import sys
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from verify_bmodel import compare_run, compare_stats, tensor_map

class FakeValue:
    def __init__(self, name): self.name = name

class FakeSession:
    def __init__(self): self.calls = []
    def get_inputs(self): return [FakeValue("x"), FakeValue("state")]
    def get_outputs(self): return [FakeValue("y"), FakeValue("new_state")]
    def run(self, unused, feed):
        self.calls.append(feed)
        return [feed["x"][:, :1, :1], feed["state"] + 1]

def manifest(path):
    data = {"networks": {"encoder": {"inputs": [
        {"name":"x", "index":0, "shape":[1,2,1], "dtype":"float32"},
        {"name":"state", "index":1, "shape":[1,1], "dtype":"float32"}], "outputs": [
        {"name":"y", "index":0, "shape":[1,1,1], "dtype":"float32"},
        {"name":"new_state", "index":1, "shape":[1,1], "dtype":"float32"}]}}}
    path.write_text(json.dumps(data))

def args(**kw):
    base = dict(mae=1e-4, max_abs=1e-4, relative_l2=1e-3, cosine_tol=1e-4)
    base.update(kw); return Namespace(**base)

class VerifyBmodelTest(unittest.TestCase):
    def test_tensor_comparison_diagnostics(self):
        equal = compare_stats(np.ones((2,), np.float32), np.ones((2,), np.float32))
        self.assertEqual(equal["mae"], 0.0)
        self.assertEqual(equal["argmax_a"], equal["argmax_b"])
        mismatch = compare_stats(np.zeros((2,), np.float32), np.zeros((1,), np.float32))
        self.assertEqual(mismatch["error"], "shape mismatch")

    def test_encoder_maps_first_new_states_to_second_cached_state(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); mp = root / "manifest.json"; manifest(mp); session = FakeSession()
            runner_states = []
            def runner(command, check, **kwargs):
                inp, out = Path(command[command.index("--input") + 1]), Path(command[command.index("--output") + 1])
                with np.load(inp) as z:
                    runner_states.append(np.array(z["state"], copy=True))
                    np.savez(out, y=z["x"][:, :1, :1], new_state=z["state"] + 1)
            ok, report = compare_run("encoder", root/"x.onnx", root/"x.bmodel", mp, root/"work", "runner", args(), session, runner)
            self.assertTrue(ok); self.assertEqual(len(session.calls), 2)
            self.assertTrue(np.array_equal(runner_states[1], runner_states[0] + 1))
            self.assertEqual(len(report["runs"]), 2)

    def test_missing_output_fails_exactly(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            tensor_map({"y": np.zeros((1,1,1), np.float32)}, [
                {"name":"y", "index":0, "shape":[1,1,1], "dtype":"float32"},
                {"name":"new_state", "index":1, "shape":[1,1], "dtype":"float32"}], "mock")

    def test_only_explicit_f32_and_f16_output_suffixes_are_mapped(self):
        items = [{"name": "y", "index": 0, "shape": [1, 1], "dtype": "float32"}]
        for name in ("y", "y_Transpose", "y_Squeeze", "y_Gemm_f32"):
            mapped = tensor_map({name: np.zeros((1, 1), np.float32)}, items, "mock")
            self.assertIn("y", mapped)
        for name in ("y_f32", "prefix_y", "y_Gemm_f16"):
            with self.assertRaisesRegex(ValueError, "missing"):
                tensor_map({name: np.zeros((1, 1), np.float32)}, items, "mock")
        with self.assertRaisesRegex(ValueError, "unexpected"):
            tensor_map({"y": np.zeros((1, 1), np.float32),
                        "extra": np.zeros((1, 1), np.float32)}, items, "mock")

    def test_zero_cosine_semantics(self):
        from verify_bmodel import metrics
        self.assertEqual(metrics(np.zeros(3, np.float32), np.zeros(3, np.float32))["cosine"], 1.0)
        self.assertEqual(metrics(np.zeros(3, np.float32), np.ones(3, np.float32))["cosine"], 0.0)

    def test_runner_failure_includes_captured_diagnostics(self):
        import subprocess
        from verify_bmodel import run_runner
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            def runner(command, **kwargs):
                raise subprocess.CalledProcessError(
                    7, command, output="runner stdout", stderr="specific runner failure"
                )
            with self.assertRaisesRegex(RuntimeError, "specific runner failure"):
                run_runner("runner", root / "model.bmodel",
                           {"x": np.zeros((1,), np.float32)}, root, "failed", runner)

    def test_threshold_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); mp = root / "manifest.json"; manifest(mp); session = FakeSession()
            def runner(command, check, **kwargs):
                inp, out = Path(command[command.index("--input") + 1]), Path(command[command.index("--output") + 1])
                with np.load(inp) as z: np.savez(out, y=z["x"][:, :1, :1] + 1, new_state=z["state"] + 1)
            ok, _ = compare_run("encoder", root/"x.onnx", root/"x.bmodel", mp, root/"work", "runner", args(), session, runner)
            self.assertFalse(ok)

if __name__ == "__main__": unittest.main()
