"""CPU checks for the score-based ECHO reference and its evidence contract."""
import unittest

from m0a.echo_replay import ThresholdEMA, approximate_prefill_topk, replay


def row(sequence, phase, scores, selected, **extra):
    data = {
        "schema_version": 1, "score_coverage": "full",
        "score_source_kind": "synthetic_fixture",
        "score_source_sha256": "0" * 64,
        "native_topk_verified": True,
        "run_id": "run", "request_id": "request", "rank": 0,
        "layer": "model.layers.2.self_attn.attn",
        "phase": phase, "sequence": sequence, "position": sequence + 10,
        "context_len": 100, "k": 2, "scores": scores,
        "selected_ids": selected,
    }
    data.update(extra)
    return data


class EchoReplayTests(unittest.TestCase):
    def test_decode_ema_and_guaranteed_recall(self):
        rows = [
            row(0, "decode", [[0, 3.0], [1, 2.0], [2, 0.0]], [0, 1]),
            row(1, "decode", [[0, 0.0], [1, 3.0], [2, 2.5]], [1, 2]),
        ]
        result = replay(rows, capacity_units=3, prefetch_budget_units=2)
        self.assertEqual(result["rows"][1]["predicted_kth_score"], 2.0)
        self.assertEqual(result["totals"]["no_prefetch"]["synchronous_recall_units"], 3)
        self.assertEqual(result["totals"]["echo_prefetch"]["synchronous_recall_units"], 2)
        self.assertEqual(result["totals"]["echo_prefetch"]["prefetch_hit_units"], 1)
        for item in result["rows"]:
            self.assertEqual(item["no_prefetch"]["selected_ids_sha256"],
                             item["echo_prefetch"]["selected_ids_sha256"])
            self.assertTrue(item["echo_prefetch"]["all_native_selected_ready"])
        self.assertFalse(result["online_performance_validated"])
        self.assertFalse(result["eligible_for_empirical_conclusion"])

    def test_prefill_threshold_is_frozen_within_chunk(self):
        scores = [[0, 10.0], [1, 9.0], [2, 1.0]]
        rows = [
            row(0, "prefill", scores, [0, 1], prefill_chunk=0, query_block=0),
            row(1, "prefill", scores, [0, 1], prefill_chunk=0, query_block=1),
            row(2, "prefill", scores, [0, 1], prefill_chunk=1, query_block=0),
            row(3, "prefill", scores, [0, 1], prefill_chunk=1, query_block=1),
        ]
        result = replay(rows, capacity_units=3, prefetch_budget_units=2)
        self.assertIsNone(result["rows"][0]["predicted_kth_score"])
        self.assertIsNone(result["rows"][1]["predicted_kth_score"])
        self.assertEqual(result["rows"][2]["predicted_kth_score"], 9.0)
        self.assertEqual(result["rows"][2]["echo_prefetch"]["predicted_candidates"],
                         len(approximate_prefill_topk([(x, y) for x, y in scores], 2, 9.0)))
        self.assertEqual(result["rows"][3]["echo_prefetch"]["predicted_candidates"], 0)

    def test_incomplete_scores_and_invalid_selection_fail_closed(self):
        valid = row(0, "decode", [[0, 2.0], [1, 1.0]], [0, 1])
        for change in ({"score_coverage": "topk_only"},
                       {"score_source_kind": "topk_only"},
                       {"native_topk_verified": False},
                       {"selected_ids": [2]},
                       {"scores": [[0, float("nan")], [1, 1.0]]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replay([dict(valid, **change)], capacity_units=2, prefetch_budget_units=1)

    def test_native_selection_must_fit_capacity(self):
        data = row(0, "decode", [[0, 2.0], [1, 1.0]], [0, 1])
        with self.assertRaisesRegex(ValueError, "exceeds capacity"):
            replay([data], capacity_units=1, prefetch_budget_units=1)

    def test_declared_native_scores_do_not_upgrade_cpu_replay_to_empirical_evidence(self):
        data = row(0, "decode", [[0, 2.0], [1, 1.0]], [0, 1],
                   score_source_kind="native_indexer_full")
        result = replay([data], capacity_units=2, prefetch_budget_units=1)
        self.assertFalse(result["eligible_for_empirical_conclusion"])
        self.assertFalse(result["online_performance_validated"])

    def test_ema_formula(self):
        state = ThresholdEMA()
        state.update(4.0)
        state.update(2.0)
        self.assertEqual(state.value, 3.0)


if __name__ == "__main__":
    unittest.main()
