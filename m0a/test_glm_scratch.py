import importlib.util
import unittest

from m0a.glm_scratch import ScratchKV


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable")
class ScratchTest(unittest.TestCase):
    def test_prefetch_budget_remap_and_guaranteed_recall(self):
        import torch
        source_key = torch.arange(12, dtype=torch.float32).reshape(6, 1, 2)
        source_rope = source_key + 100
        scratch = ScratchKV(source_key, source_rope,
                            torch.empty(192, 16, 1, 2), torch.empty(192, 16, 1, 2))
        result = scratch.prepare([0, 2, 4, -1], {i: float(i) for i in range(6)},
                                 [2, 3], prefetch=True)
        self.assertEqual(result.prefetched, (2, 3))
        self.assertEqual(result.recalled, (0, 4))
        self.assertEqual(result.useful_prefetch, (2,))
        self.assertEqual(result.wasted_prefetch, (3,))
        self.assertEqual(result.remapped[-1], -1)
        self.assertEqual(len(set(result.remapped[:-1])), 3)
        scratch.assert_payload_exact([0, 2, 4])
        self.assertEqual(scratch.block_table().shape, (1, 192))


if __name__ == "__main__":
    unittest.main()
