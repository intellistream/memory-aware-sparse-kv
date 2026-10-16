"""Causality, task isolation, deterministic ranking and boundary regression tests."""
import copy
import unittest

from m0a.history_fixture import fixture
from m0a.history_prefetch import AccessHistoryPredictor, PredictionContext
from m0a.history_replay import STRATEGIES, replay


def context(query, stage="a", eligible=(0, 1, 2, 3, 4, 5, 6, 7), step=None):
    return PredictionContext(query, 2 * query if step is None else step, stage, eligible)


def train(predictor, selections):
    for query, selected in enumerate(selections):
        predictor.predict(context(query), budget=2)
        predictor.observe(query=query, completion_step=2 * query + 1, selected_blocks=selected)


def blocks(candidates):
    return [c.block for c in candidates]


class HistoryPredictorTests(unittest.TestCase):
    def test_frequency_and_recency_are_distinct(self):
        frequency = AccessHistoryPredictor(signal="frequency")
        recency = AccessHistoryPredictor(signal="recency")
        for predictor in (frequency, recency):
            train(predictor, [(0,), (0,), (1,)])
        self.assertEqual(blocks(frequency.predict(context(3), budget=1)), [0])
        self.assertEqual(blocks(recency.predict(context(3), budget=1)), [1])

    def test_region_counts_each_completed_query_once(self):
        predictor = AccessHistoryPredictor(signal="region", region_blocks=4)
        train(predictor, [(0, 1, 2), (4,), (5,)])
        candidates = predictor.predict(context(3), budget=8)
        self.assertEqual(blocks(candidates)[:4], [5, 4, 6, 7])
        self.assertEqual(candidates[0].region_accesses, 2)
        self.assertEqual(candidates[-1].region_accesses, 1)
        self.assertIsNone(candidates[2].last_completed_query)

    def test_stage_reset_and_global_retention(self):
        for scope, expected in (("global", [0]), ("stage", [])):
            predictor = AccessHistoryPredictor(scope=scope)
            train(predictor, [(0,), (0,)])
            self.assertEqual(blocks(predictor.predict(context(2, "b"), budget=1)), expected)

    def test_budget_and_eligibility_apply_to_every_signal(self):
        for signal in ("frequency", "recency", "region", "last_selection"):
            predictor = AccessHistoryPredictor(signal=signal)
            train(predictor, [(0, 1), (0, 2)])
            self.assertEqual(blocks(predictor.predict(context(2, eligible=(2,)), budget=10)), [2])
            predictor.observe(query=2, completion_step=5, selected_blocks=(2,))
            self.assertEqual(predictor.predict(context(3), budget=0), [])

    def test_current_and_future_observations_are_rejected(self):
        predictor = AccessHistoryPredictor()
        with self.assertRaises(ValueError):
            predictor.observe(query=0, completion_step=1, selected_blocks=(0,))
        predictor.predict(context(0), budget=1)
        with self.assertRaises(ValueError):
            predictor.observe(query=0, completion_step=0, selected_blocks=(0,))
        predictor.observe(query=0, completion_step=10, selected_blocks=(0,))
        with self.assertRaises(ValueError):
            predictor.predict(context(1), budget=1)

    def test_batch_has_no_intra_batch_feedback(self):
        predictor = AccessHistoryPredictor()
        self.assertEqual(predictor.predict(context(0, step=0), budget=1), [])
        self.assertEqual(predictor.predict(context(1, step=0), budget=1), [])
        with self.assertRaises(ValueError):
            predictor.observe(query=1, completion_step=1, selected_blocks=(1,))
        predictor.observe(query=0, completion_step=1, selected_blocks=(0,))
        with self.assertRaises(ValueError):
            predictor.predict(context(2, step=2), budget=1)
        predictor.observe(query=1, completion_step=1, selected_blocks=(1,))
        self.assertEqual(blocks(predictor.predict(context(2, step=2), budget=1)), [1])

    def test_invalid_ids_and_stage_occurrence_reuse(self):
        for selected in ((-1,), (True,), (0, 0)):
            predictor = AccessHistoryPredictor()
            predictor.predict(context(0), budget=1)
            with self.assertRaises(ValueError):
                predictor.observe(query=0, completion_step=1, selected_blocks=selected)
        predictor = AccessHistoryPredictor(scope="stage")
        train(predictor, [(0,)])
        predictor.predict(context(1, "b"), budget=1)
        predictor.observe(query=1, completion_step=3, selected_blocks=(1,))
        with self.assertRaises(ValueError):
            predictor.predict(context(2, "a"), budget=1)

    def test_ties_are_deterministic_independent_of_input_order(self):
        a, b = AccessHistoryPredictor(), AccessHistoryPredictor()
        train(a, [(0, 1)])
        train(b, [(1, 0)])
        self.assertEqual(blocks(a.predict(context(1, eligible=(1, 0)), budget=2)),
                         blocks(b.predict(context(1, eligible=(0, 1)), budget=2)))


class HistoryReplayTests(unittest.TestCase):
    def test_fixture_and_stage_switch_behavior(self):
        result = replay(fixture(), budget_blocks=2)
        self.assertEqual(len(result["rows"]), 96 * len(STRATEGIES))
        self.assertFalse(result["online_performance_validated"])
        self.assertFalse(result["capacity_or_transfer_simulated"])
        rows = [r for r in result["rows"] if r["task_id"] == "switching" and r["phase"] == "decode"]
        by_key = {(r["query"], r["strategy"]): r for r in rows}
        self.assertEqual(by_key[(9, "stage_frequency")]["candidate_hits"], 2)
        self.assertEqual(by_key[(9, "global_frequency")]["candidate_hits"], 0)
        self.assertEqual(by_key[(8, "global_frequency")]["previous_query_jaccard"], 0)
        self.assertEqual(by_key[(8, "stage_frequency")]["candidate_count"], 0)

    def test_mutating_target_or_future_selection_cannot_change_prediction(self):
        rows = fixture()[:24]
        changed = copy.deepcopy(rows)
        for row in changed[10:]:
            row["selected_blocks"] = [14, 15]
        before, after = (replay(data, budget_blocks=2)["rows"] for data in (rows, changed))
        for a, b in zip(before, after):
            if a["query"] <= 11 and a["strategy"] != "oracle_candidate_upper_bound":
                self.assertEqual(a["candidate_blocks"], b["candidate_blocks"])

    def test_lane_namespace_isolation_and_phase_cold_start(self):
        rows = replay(fixture(), budget_blocks=2)["rows"]
        for row in rows:
            if row["query"] == 0 and row["strategy"] != "oracle_candidate_upper_bound":
                self.assertEqual(row["candidate_count"], 0)
        base = fixture()[24:48]
        for field, value in (("task_id", "new-task"), ("context_version", "v2"),
                             ("request_id", "req2"), ("rank", 1), ("layer", "layer.1")):
            duplicate = [dict(r, **{field: value}) for r in base]
            result = replay(base + duplicate, budget_blocks=2)
            first = [r for r in result["rows"] if r["query"] == 0 and r["strategy"] == "global_frequency"]
            self.assertEqual([r["candidate_count"] for r in first], [0, 0])

    def test_bad_trace_fails_closed(self):
        cases = []
        base = fixture()[:24]
        cases.extend((base[1:], base + [base[-1]], list(reversed(base))))
        for patch in ({"completion_step": 0}, {"phase": "unknown"}, {"selected_blocks": [16]},
                      {"eligible_blocks": [True]}, {"source_kind": "unknown"}, {"batch": 20}):
            data = copy.deepcopy(base)
            data[0].update(patch)
            cases.append(data)
        for data in cases:
            with self.subTest(first=data[0]), self.assertRaises(ValueError):
                replay(data, budget_blocks=2)

    def test_zero_budget_and_oracle_only_available_blocks(self):
        rows = fixture()[24:48]
        rows[0]["eligible_blocks"] = []
        result = replay(rows, budget_blocks=2)
        first = [r for r in result["rows"] if r["query"] == 0]
        self.assertTrue(all(r["candidate_count"] == 0 for r in first))
        self.assertTrue(all(r["candidate_count"] == 0 for r in replay(rows, budget_blocks=0)["rows"]))


if __name__ == "__main__":
    unittest.main()
