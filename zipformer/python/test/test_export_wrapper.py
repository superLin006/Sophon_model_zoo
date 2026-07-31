import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import unittest

try:
    import torch
except ModuleNotFoundError:
    raise unittest.SkipTest("wrapper tests require torch")

from export_onnx import SophonBatchedStateEncoder


class _FakeEncoder(torch.nn.Module):
    def forward(self, x, *states):
        return (x + 3.0,) + tuple(state + float(i + 1) for i, state in enumerate(states))


class BatchedStateWrapperTest(unittest.TestCase):
    def test_envelope_round_trip_preserves_inner_semantics(self):
        wrapper = SophonBatchedStateEncoder(_FakeEncoder()).eval()
        x = torch.randn(1, 103, 80)
        states = [torch.full((1, 2, 1), float(i)) for i in range(35)]
        result = wrapper(x, *states)
        self.assertEqual(len(result), 36)
        torch.testing.assert_close(result[0], x + 3.0)
        for i, state in enumerate(result[1:]):
            self.assertEqual(tuple(state.shape), tuple(states[i].shape))
            torch.testing.assert_close(state, states[i] + float(i + 1))

    def test_rejects_wrong_state_count(self):
        with self.assertRaisesRegex(ValueError, "expected 35 states"):
            SophonBatchedStateEncoder(_FakeEncoder())(torch.zeros(1, 103, 80))


if __name__ == "__main__":
    unittest.main()
