"""CPU tests for the explicit DSA score reference."""
import unittest

try:
    import torch
except ImportError:
    torch = None

from m0a.echo_score_reference import weighted_relu_scores, verify_native_topk


@unittest.skipIf(torch is None, "PyTorch is tested in the server CPU environment")
class ScoreReferenceTests(unittest.TestCase):
    def test_weighted_relu_matches_manual_scores(self):
        query = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        keys = torch.tensor([[2.0, 1.0], [-1.0, 3.0], [0.0, 0.0]])
        weights = torch.tensor([0.5, 2.0])
        scores = weighted_relu_scores(query, keys, weights)
        self.assertEqual(scores.tolist(), [5.0, 12.0, 0.0])
        self.assertTrue(verify_native_topk(scores, [1, 0], 2))
        with self.assertRaisesRegex(ValueError, "disagree"):
            verify_native_topk(scores, [0, 2], 2)

    def test_shape_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "mismatch"):
            weighted_relu_scores(torch.zeros((2, 3)), torch.zeros((4, 2)),
                                 torch.ones(2))


if __name__ == "__main__":
    unittest.main()
