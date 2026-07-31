#!/usr/bin/env python3
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))
from tpumlir_args import input_shapes, inspect_onnx, load_network

SCRIPT = Path(__file__).parents[2] / "python" / "gen_bmodel.sh"
MANIFEST = Path(__file__).parents[2] / "configs" / "tensor_manifest.json"


class ManifestArgsTest(unittest.TestCase):
    def test_shapes_follow_index_order(self):
        shapes = input_shapes(MANIFEST, "encoder")
        self.assertEqual(len(shapes), 36)
        self.assertEqual(shapes[0], [1, 103, 80])

    def test_decoder_integer_dtype_fixture_keeps_shapes(self):
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        data["networks"]["decoder"]["inputs"][0]["dtype"] = "int32"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(input_shapes(path, "decoder"), [[1, 2]])
            _, spec = load_network(path, "decoder")
            self.assertEqual(spec["inputs"][0]["dtype"], "int32")
            self.assertEqual(spec["inputs"][0]["name"], "token_ids")

        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
        data["networks"]["joiner"]["inputs"][1]["index"] = 7
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contiguous"):
                load_network(path, "joiner")

    def test_complete_onnx_protocol_is_checked(self):
        import onnx
        from onnx import TensorProto, helper

        spec = json.loads(MANIFEST.read_text(encoding="utf-8"))["networks"]["joiner"]
        inputs = [helper.make_tensor_value_info(t["name"], TensorProto.FLOAT, t["shape"]) for t in spec["inputs"]]
        outputs = [helper.make_tensor_value_info(t["name"], TensorProto.FLOAT, t["shape"]) for t in spec["outputs"]]
        nodes = [helper.make_node("Identity", [inputs[0].name], [output.name]) for output in outputs]
        with tempfile.TemporaryDirectory() as directory:
            valid = Path(directory) / "valid.onnx"
            onnx.save(helper.make_model(helper.make_graph(nodes, "joiner", inputs, outputs)), valid)
            report = inspect_onnx(valid, MANIFEST, "joiner")
            self.assertEqual(len(report["inputs"]), 2)
            self.assertEqual(len(report["outputs"]), 1)

            wrong = Path(directory) / "wrong.onnx"
            bad_outputs = [helper.make_tensor_value_info("wrong_logit", TensorProto.FLOAT, spec["outputs"][0]["shape"])]
            bad_nodes = [helper.make_node("Identity", [inputs[0].name], ["wrong_logit"])]
            onnx.save(helper.make_model(helper.make_graph(bad_nodes, "joiner", inputs, bad_outputs)), wrong)
            with self.assertRaisesRegex(ValueError, "outputs do not match"):
                inspect_onnx(wrong, MANIFEST, "joiner")

    def test_dry_run_uses_chip_and_network_policy(self):
        import onnx
        from onnx import TensorProto, helper

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "joiner.onnx"
            spec = json.loads(MANIFEST.read_text(encoding="utf-8"))["networks"]["joiner"]
            tensors = spec["inputs"]
            inputs = [helper.make_tensor_value_info(t["name"], TensorProto.FLOAT, t["shape"]) for t in tensors]
            outputs = [helper.make_tensor_value_info(t["name"], TensorProto.FLOAT, t["shape"]) for t in spec["outputs"]]
            nodes = [helper.make_node("Identity", [inputs[0].name], [output.name]) for output in outputs]
            onnx.save(helper.make_model(helper.make_graph(nodes, "test", inputs, outputs)), model)
            result = subprocess.run(
                ["bash", str(SCRIPT), "--network", "joiner", "--onnx", str(model),
                 "--output", str(Path(directory) / "joiner.bmodel"), "--dry-run"],
                check=True, capture_output=True, text=True,
            )
            self.assertIn("--model_name zipformer_joiner", result.stdout)
            self.assertIn("--chip bm1684x", result.stdout)
            self.assertNotIn("--processor", result.stdout)
            self.assertIn("--quantize F32", result.stdout)
            self.assertNotIn("--disable_layer_group", result.stdout)

            encoder = Path(directory) / "encoder.onnx"
            spec = json.loads(MANIFEST.read_text(encoding="utf-8"))["networks"]["encoder"]
            tensors = spec["inputs"]
            inputs = [helper.make_tensor_value_info(t["name"], TensorProto.FLOAT, t["shape"]) for t in tensors]
            outputs = [helper.make_tensor_value_info(t["name"], TensorProto.FLOAT, t["shape"]) for t in spec["outputs"]]
            nodes = [helper.make_node("Identity", [inputs[0].name], [output.name]) for output in outputs]
            onnx.save(helper.make_model(helper.make_graph(nodes, "test", inputs, outputs)), encoder)
            result = subprocess.run(
                ["bash", str(SCRIPT), "--network", "encoder", "--onnx", str(encoder),
                 "--output", str(Path(directory) / "encoder.bmodel"), "--dry-run"],
                check=True, capture_output=True, text=True,
            )
            self.assertIn("--model_name zipformer_encoder", result.stdout)
            self.assertIn("--disable_layer_group", result.stdout)

        for network in ("encoder", "decoder", "joiner"):
            _, spec = load_network(MANIFEST, network)
            self.assertEqual([x["index"] for x in spec["inputs"]], list(range(len(spec["inputs"]))))


if __name__ == "__main__":
    unittest.main()
