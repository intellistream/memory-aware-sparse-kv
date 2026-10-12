import importlib.util
import random
import unittest

from m0a.echo_replay import approximate_prefill_topk


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy unavailable")
class GLMPolicyTest(unittest.TestCase):
    def test_vectorized_prefill_bins_match_reference(self):
        from m0a.glm_prefetch_policy import prefill_candidates
        rng = random.Random(72)
        scores = [rng.random() * 10 for _ in range(4000)]
        scalar = set(approximate_prefill_topk(list(enumerate(scores)), 2048, 5.0))
        vectorized = prefill_candidates(scores, 5.0, 2048)
        self.assertTrue(set(vectorized).issubset(scalar))
        self.assertLessEqual(len(vectorized), 512)
        self.assertEqual(vectorized, sorted(vectorized))

    def test_decode_budget_does_not_sort_by_exact_score(self):
        from m0a.glm_prefetch_policy import decode_candidates
        scores = [1.0] * 513 + [100.0]
        candidates = decode_candidates(scores, 0.5)
        self.assertEqual(candidates, list(range(512)))
        self.assertNotIn(513, candidates)


if __name__ == "__main__":
    unittest.main()
