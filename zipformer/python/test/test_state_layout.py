import sys
import json
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from state_layout import build_manifest, load_manifest, state_specs, validate_manifest


class StateLayoutTest(unittest.TestCase):
    def test_state_order_and_count(self):
        states = state_specs()
        self.assertEqual(len(states), 35)
        self.assertEqual([s["name"] for s in states[:5]], [f"cached_len_{i}" for i in range(5)])
        self.assertEqual([s["name"] for s in states[-5:]], [f"cached_conv2_{i}" for i in range(5)])
        self.assertEqual(states[10]["name"], "cached_key_0")
        self.assertEqual(states[20]["name"], "cached_val2_0")
        self.assertEqual(states[25]["name"], "cached_conv1_0")

    def test_key_shapes(self):
        states = {s["name"]: s["shape"] for s in state_specs()}
        self.assertEqual(states["cached_key_0"], [1, 2, 128, 1, 192])
        self.assertEqual(states["cached_key_4"], [1, 2, 64, 1, 192])
        self.assertEqual(states["cached_val_2"], [1, 2, 32, 1, 96])
        self.assertEqual(states["cached_conv1_3"], [1, 2, 1, 256, 30])

    def test_manifest_roundtrip_and_validation(self):
        manifest = build_manifest()
        self.assertTrue(validate_manifest(manifest))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(load_manifest(path), manifest)

    def test_checked_in_manifest_is_generated_manifest(self):
        path = Path(__file__).parents[1] / "configs" / "tensor_manifest.json"
        checked = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(checked, build_manifest())

    def test_protocol_names_and_decoder_dtype(self):
        manifest = build_manifest()["networks"]
        self.assertEqual(manifest["encoder"]["inputs"][0]["name"], "x")
        self.assertEqual(manifest["encoder"]["outputs"][1]["name"], "new_cached_len_0")
        self.assertEqual(manifest["decoder"]["inputs"][0]["name"], "token_ids")
        self.assertEqual(manifest["decoder"]["inputs"][0]["dtype"], "int64")
        self.assertEqual([x["name"] for x in manifest["joiner"]["inputs"]], ["enc_out", "dec_out"])
        self.assertEqual(manifest["joiner"]["outputs"][0]["name"], "logit")

        manifest = build_manifest()
        self.assertEqual(manifest["model"]["id"], "csukuangfj/k2fsa-zipformer-bilingual-zh-en-t")
        self.assertEqual(manifest["model"]["short_name"], "zipformer_bilingual")
        frontend = manifest["frontend"]
        self.assertEqual(frontend["frame_shift_ms"], 10.0)
        self.assertEqual(frontend["frame_length_ms"], 25.0)
        self.assertEqual(frontend["dither"], 0.0)
        self.assertEqual(frontend["preemphasis_coefficient"], 0.97)
        self.assertEqual(frontend["window_type"], "povey")
        self.assertFalse(frontend["snip_edges"])
        self.assertTrue(frontend["remove_dc_offset"])
        self.assertEqual(frontend["low_frequency"], 20.0)
        self.assertEqual(frontend["high_frequency"], -400.0)
        self.assertFalse(frontend["use_energy"])
        self.assertTrue(frontend["use_log_fbank"])
        self.assertTrue(frontend["use_power"])
        self.assertEqual(frontend["tail_padding_seconds"], 1.03)

        encoder = build_manifest()["networks"]["encoder"]
        self.assertEqual(len(encoder["inputs"]), 36)
        self.assertEqual(len(encoder["outputs"]), 36)
        self.assertEqual(encoder["inputs"][0]["shape"], [1, 103, 80])
        self.assertEqual(encoder["outputs"][0]["shape"], [1, 24, 256])
        for input_tensor, output_tensor in zip(encoder["inputs"][1:], encoder["outputs"][1:]):
            self.assertEqual(output_tensor["shape"], input_tensor["shape"])


if __name__ == "__main__":
    unittest.main()
