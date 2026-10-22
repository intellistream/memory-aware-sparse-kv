"""Independent numerical/causal checks for the CPU attention component."""
import os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
import unittest
import torch
from .deepseek_v4_cpu import (pool_projected, pool_incremental, hadamard, rope,
    gathered_attention, masked_attention_reference, index_scores, stable_topk,
    TensorLRU, int8_reference, AttentionComponent)


class DeepSeekCPUChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)

    def test_stream_vs_batch_c4_partial_blocks(self):
        for length in (1, 3, 4, 7, 8, 13, 20):
            kv, gates, ape = torch.randn(length, 16), torch.randn(length, 16), torch.randn(4, 16)
            torch.testing.assert_close(pool_projected(kv, gates, ape, 4), pool_incremental(kv, gates, ape, 4))

    def test_stream_vs_batch_c128(self):
        kv, gates, ape = torch.randn(267, 8), torch.randn(267, 8), torch.randn(128, 8)
        torch.testing.assert_close(pool_projected(kv, gates, ape, 128), pool_incremental(kv, gates, ape, 128))

    def test_c4_halves_and_first_window_padding(self):
        kv = torch.zeros(8, 4)
        kv[:4, :2], kv[:4, 2:], kv[4:, 2:] = 10, 2, 4
        out = pool_projected(kv, torch.zeros_like(kv), torch.zeros(4,4), 4)
        torch.testing.assert_close(out, torch.tensor([[2., 2.], [7., 7.]]))

    def test_future_tokens_do_not_change_compressed_prefix(self):
        kv, gates, ape = torch.randn(16, 8), torch.randn(16, 8), torch.randn(4, 8)
        a = pool_projected(kv, gates, ape, 4)
        kv[8:] = 1000
        gates[8:] = 100
        b = pool_projected(kv, gates, ape, 4)
        torch.testing.assert_close(a[:2], b[:2])

    def test_indexer_relu_then_weighted_head_sum(self):
        q, k, w = torch.randn(3, 4), torch.randn(7, 4), torch.randn(3)
        actual = index_scores(q, k, w)
        expected = torch.tensor([sum(max(0, float(torch.dot(q[h], key)))*float(w[h]) for h in range(3)) for key in k])
        torch.testing.assert_close(actual, expected)

    def test_stable_topk_tie_and_short_context(self):
        self.assertEqual(stable_topk(torch.tensor([1., 2., 2.]), 5).tolist(), [1, 2, 0])

    def test_selection_masks_future_cache_entries(self):
        component = object.__new__(AttentionComponent)
        component.ratio, component.cfg = 4, {'index_topk': 1}
        state = dict(prompt_tokens=8, index_q=torch.randn(1, 3, 4),
                     index_k=torch.randn(10,4), index_weights=torch.ones(1,3))
        a, score = component.select(state, 0)
        state['index_k'][2:] = 10000
        b, new_score = component.select(state, 0)
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(score, new_score)
        self.assertEqual(len(score), 2)

    def test_gather_equals_independent_dense_mask(self):
        q, kv, sink = torch.randn(3, 8), torch.randn(17, 8), torch.randn(3)
        ids = torch.tensor([2, 5, 7, 11])
        actual, _ = gathered_attention(q, kv[ids], sink)
        expected = masked_attention_reference(q, kv, ids, sink)
        torch.testing.assert_close(actual, expected)

    def test_attention_sink_is_zero_value_probability(self):
        out, prob = gathered_attention(torch.zeros(1,2), torch.tensor([[2.,4.]]), torch.zeros(1))
        torch.testing.assert_close(out, torch.tensor([[1.,2.]]))
        torch.testing.assert_close(prob, torch.tensor([.5]))

    def test_missing_required_kv_changes_output(self):
        q, kv, sink = torch.ones(1,2), torch.tensor([[8.,8.],[-1.,-1.]]), torch.zeros(1)
        a, _ = gathered_attention(q, kv, sink)
        b, _ = gathered_attention(q, kv[1:], sink)
        self.assertGreater(float((a-b).abs().max()), 1)

    def test_cache_payload_copy_capacity_and_recall(self):
        host = torch.randn(6,8).bfloat16()
        cache = TensorLRU(2, host)
        cache.insert(5)
        cache.insert(4)
        actual = cache.recall([1,0])
        torch.testing.assert_close(actual, host[[1,0]])
        self.assertEqual(cache.peak, 2)
        self.assertEqual(cache.demand, 2)
        host[1] = 0
        self.assertTrue(torch.count_nonzero(cache.resident[1]))

    def test_cache_rejects_oversized_working_set(self):
        with self.assertRaises(ValueError):
            TensorLRU(1, torch.zeros(3,2)).recall([0,1])

    def test_hadamard_preserves_dot_product(self):
        a, b = torch.randn(3,128), torch.randn(3,128)
        torch.testing.assert_close((hadamard(a)*hadamard(b)).sum(-1), (a*b).sum(-1))

    def test_rope_inverse_and_nonrotary_part(self):
        x, positions, freq = torch.randn(7,3,8), torch.arange(7), torch.tensor([1.,.1])
        y = rope(x, positions, freq)
        torch.testing.assert_close(y[..., :4], x[..., :4])
        torch.testing.assert_close(rope(y, positions, freq, inverse=True), x)

    def test_zero_quantization_and_error_bound(self):
        self.assertTrue(torch.isfinite(int8_reference(torch.zeros(3,8))).all())
        x = torch.randn(3,8)
        error = (int8_reference(x)-x).abs()
        self.assertTrue((error <= x.abs().amax(-1,keepdim=True)/254+1e-6).all())


if __name__ == '__main__':
    unittest.main()
