"""Correctness checks for the non-timed KV reference and mode gates."""
import unittest

try:
    import torch
except ImportError:
    torch = None

from m0a.echo_runtime import Capabilities, Mode, ReferenceKVPool


@unittest.skipIf(torch is None, "PyTorch is tested in the server CPU environment")
class EchoRuntimeTests(unittest.TestCase):
    def test_modes_fail_closed_without_native_capabilities(self):
        with self.assertRaisesRegex(RuntimeError, "guaranteed recall"):
            Capabilities().validate(Mode.OFFLOAD_NO_PREFETCH, timed=False)
        with self.assertRaisesRegex(RuntimeError, "complete native"):
            Capabilities(exact_guaranteed_recall=True).validate(
                Mode.OFFLOAD_ECHO_PREFETCH, timed=False)
        with self.assertRaisesRegex(RuntimeError, "fused NPU"):
            Capabilities(exact_guaranteed_recall=True, native_full_scores=True).validate(
                Mode.OFFLOAD_ECHO_PREFETCH, timed=True)
        Capabilities().validate(Mode.NO_OFFLOAD, timed=False)

    def test_prefetch_and_demand_preserve_native_payloads(self):
        host = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        pool = ReferenceKVPool(host, torch.empty((3, 4)))
        scores = {0: 3.0, 1: 2.0, 2: 4.0, 3: 1.0}
        self.assertEqual(pool.guaranteed_recall([0, 1], scores), [0, 1])
        self.assertEqual(pool.prefetch([2], scores, 1), [2])
        self.assertEqual(pool.guaranteed_recall([1, 2], scores), [])
        self.assertEqual(pool.assert_ready([1, 2]), [1, 2])
        self.assertEqual(pool.prefetched_units, 1)
        self.assertEqual(pool.synchronous_units, 2)
        self.assertFalse(pool.timed_performance_eligible)

    def test_capacity_and_missing_score_rejected(self):
        host = torch.zeros((3, 2))
        pool = ReferenceKVPool(host, torch.empty((1, 2)))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            pool.guaranteed_recall([0, 1], {0: 1.0, 1: 2.0})
        with self.assertRaisesRegex(ValueError, "lacks"):
            pool.prefetch([0], {}, 1)


if __name__ == "__main__":
    unittest.main()
