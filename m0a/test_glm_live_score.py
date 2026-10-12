import importlib.util
import unittest

from m0a.glm_live_score import verify_rows


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable")
class GLMScoreTest(unittest.TestCase):
    def test_paged_full_scores_reject_wrong_native_selection(self):
        import torch
        key = torch.zeros(1, 128, 1, 2)
        key[0, 0, 0] = torch.tensor([4., 0.])
        key[0, 1, 0] = torch.tensor([3., 0.])
        key[0, 2, 0] = torch.tensor([1., 0.])
        query = torch.tensor([[[1., 0.]]])
        weights = torch.tensor([[1.]])
        block_table = torch.tensor([[0]], dtype=torch.int32)
        selected = torch.tensor([[[0, 1]]], dtype=torch.int32)
        result = verify_rows(query, weights, key, block_table, selected,
                             query_start=2, rows=[0])
        self.assertEqual(result[2].tolist(), [4., 3., 1.])
        selected[0, 0] = torch.tensor([0, 2])
        with self.assertRaisesRegex(ValueError, "disagree"):
            verify_rows(query, weights, key, block_table, selected,
                        query_start=2, rows=[0])


if __name__ == "__main__":
    unittest.main()
