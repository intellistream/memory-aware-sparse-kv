"""Static checks for the opt-in, top-k-only NPU score probe."""
import unittest

from m0a.echo_diagnostic_source import enable_topk_score_capture


SOURCE = '''
def probe(self, trace_prefill, trace_metadata, layer_name):
    topk_idxs, _ = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer(
        query=None,
        return_value=False,
    )
    if trace_prefill:
        record_prefill_selected(
            topk_idxs,
            trace_metadata,
            layer_name=layer_name,
            rank=self.tp_rank,
        )
    return topk_idxs
'''


class ScoreProbeTests(unittest.TestCase):
    def test_opt_in_patch_exposes_only_topk_values(self):
        changed = enable_topk_score_capture(
            SOURCE, kernel_source="// M0A_SPARSE_VALUES_WRITTEN_V1")
        self.assertIn("topk_idxs, topk_values =", changed)
        self.assertIn("return_value=True,", changed)
        self.assertIn("topk_values=topk_values,", changed)
        self.assertEqual(changed.count("return topk_idxs"), 1)

    def test_changed_call_shape_fails_closed(self):
        with self.assertRaises(ValueError):
            enable_topk_score_capture(SOURCE.replace("return_value=False", "return_value=True"),
                                      kernel_source="// M0A_SPARSE_VALUES_WRITTEN_V1")
        with self.assertRaises(ValueError):
            enable_topk_score_capture(SOURCE.replace("record_prefill_selected", "other_hook"),
                                      kernel_source="// M0A_SPARSE_VALUES_WRITTEN_V1")
        with self.assertRaisesRegex(ValueError, "does not prove"):
            enable_topk_score_capture(SOURCE, kernel_source="sparseValues")


if __name__ == "__main__":
    unittest.main()
