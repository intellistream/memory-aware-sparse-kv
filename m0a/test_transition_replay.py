"""Adversarial CPU-only checks for losslessness, byte accounting and split isolation."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from model_profiles import load_profile
from transition_fixture import fixture
from transition_replay import (read_trace, replay_event, run_replay, validate_config, validate_sidecar)
from working_set import (LRU, TransitionTable, candidates, feature, identity, lane_key, validate_snapshot)


class WorkingSetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sidecar, self.config, self.trace = fixture(self.root, 1, 1)
        self.snapshots, self.groups = validate_sidecar(self.sidecar, self.root)
        self.old, self.cur = self.sidecar["snapshots"][:2]
        self.lane = self.cur["lanes"][0]
        self.event = self.sidecar["events"][0]

    def rows(self):
        return read_trace(self.root/"traces", self.snapshots, self.sidecar["events"], synthetic=True)

    def simulate(self, strategy="transition", granularity="native", config=None, table=None):
        cfg = config or self.config
        return replay_event(strategy, granularity, self.event, self.old, self.cur,
                            self.rows()[self.event["event_id"]], LRU(cfg["capacity_bytes"]), cfg,
                            table or TransitionTable(), TransitionTable(), "task_switch")

    def test_cold_start_and_exact_readiness(self):
        result = self.simulate()
        self.assertEqual(result["attention_readiness_assertions"], 32)
        self.assertEqual(set(result["table_sources"].values()), {"cold"})
        self.assertEqual(result["candidate_native_units"], 512)
        self.assertGreater(result["synchronous_recall_bytes"], 0)

    def test_prefix_rewrite_invalidates_suffix_and_version_identity(self):
        self.cur["token_ids"][200] = 10000
        self.assertEqual(candidates(self.old, self.cur, self.lane), set(range(200)))
        cache = LRU(1024)
        cache.put(identity(self.old)+(0, "model.layers.0", "kv_token_position", 1), 256, set(range(128, 256)))
        cache.migrate(self.old, self.cur, {lane_key(self.lane): candidates(self.old, self.cur, self.lane)})
        entry = next(iter(cache.entries.values()))
        self.assertEqual(entry.members, set(range(128, 200)))
        self.assertEqual(next(iter(cache.entries))[:5], identity(self.cur))

    def test_no_reuse_across_model_tokenizer_or_session(self):
        for field in ("session_id", "model_id", "model_revision", "tokenizer_sha256"):
            cur = copy.deepcopy(self.cur)
            cur[field] += "changed"
            self.assertEqual(candidates(self.old, cur, cur["lanes"][0]), set())

    def test_new_inserted_content_is_not_candidate(self):
        self.cur["token_ids"].insert(10, 999)
        self.assertEqual(candidates(self.old, self.cur, self.lane), set(range(10)))
        self.assertNotIn(512, candidates(self.old, self.cur, self.lane))

    def test_version_reuse_rewrite_rejected(self):
        self.cur["context_version"] = self.old["context_version"]
        self.cur["token_ids"][4] = 888
        with self.assertRaisesRegex(ValueError, "context_version"):
            validate_sidecar(self.sidecar, self.root)

    def test_compressor_support_is_required_and_hash_checked(self):
        lane = copy.deepcopy(self.lane)
        lane["kv_kind"] = "compressed_kv_token_position"
        snapshot = copy.deepcopy(self.cur)
        snapshot["lanes"] = [lane]
        with self.assertRaisesRegex(ValueError, "support intervals"):
            validate_snapshot(snapshot, self.root)
        lane["generated_ids"] = [0, 1]
        lane["support_intervals"] = {"0": [0, 8, 7], "1": [124, 132, 131]}
        source = self.root/"synthetic_compressor.py"
        source.write_text("# Synthetic test contract: compressor support spans eight tokens.\n")
        lane["support_evidence"] = {"path": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                                    "start_line": 1, "end_line": 1, "explanation": "Synthetic test only; overlapping support."}
        validate_snapshot(snapshot, self.root)
        self.assertEqual(feature(snapshot, lane, 1)[0], "mixed")
        old = copy.deepcopy(snapshot)
        old["sequence"] = 1
        snapshot["token_ids"][130] = 999
        self.assertEqual(candidates(old, snapshot, lane), {0})
        source.write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            validate_snapshot(snapshot, self.root)

    def test_table_is_bounded_and_has_no_absolute_ids(self):
        table = TransitionTable(max_tables=3)
        units = candidates(self.old, self.cur, self.lane)
        for i in range(20):
            table.update(self.cur, self.lane, units, {260}, "state", str(i))
        self.assertLessEqual(len(table.tables), 3)
        for entry in table.tables.values():
            self.assertLessEqual(len(entry["cells"]), 640)
            self.assertTrue(all(len(k) == 3 and 0 <= k[2] < 16 for k in entry["cells"]))

    def test_predict_before_update_and_eight_observation_fallback(self):
        table = TransitionTable()
        units = candidates(self.old, self.cur, self.lane)
        self.assertEqual(table.rank(self.cur, self.lane, units, "s", "e")[1], "cold")
        for i in range(7):
            table.update(self.cur, self.lane, units, {260, 261}, "s", "e")
        self.assertEqual(table.rank(self.cur, self.lane, units, "s", "e")[1], "event")
        table.update(self.cur, self.lane, units, {260, 261}, "s", "e")
        ranking, source = table.rank(self.cur, self.lane, units, "s", "e")
        self.assertEqual(source, "previous_state+event")
        self.assertTrue(set(ranking[:2]) <= set(range(256, 264)))
        self.assertEqual(table.rank(self.cur, self.lane, units, "unseen", "e")[1], "event")

    def test_wrong_prediction_only_changes_cache_and_transfers(self):
        cfg = dict(self.config, capacity_bytes=256, prefetch_budget_bytes=128)
        table = TransitionTable()
        table.update(self.cur, self.lane, set(range(512)), set(range(480, 512)), self.event["previous_state"], self.event["event_type"])
        rows = self.rows()[self.event["event_id"]]
        before = copy.deepcopy(rows)
        result = replay_event("transition", "native", self.event, self.old, self.cur, rows,
                              LRU(cfg["capacity_bytes"]), cfg, table, TransitionTable(), "task_switch")
        self.assertEqual(result["precision"], 0)
        self.assertEqual(result["useful_prefetch_bytes"], 0)
        self.assertEqual(result["wasted_prefetch_bytes"], result["prefetched_bytes"])
        self.assertEqual(result["attention_readiness_assertions"], 32)
        self.assertEqual(rows, before)

    def test_page_mixes_regions_without_rounding_candidates(self):
        self.cur["token_ids"][200] = 777
        self.cur["regions"][1]["end"] = 160
        self.cur["regions"][2]["start"] = 160
        result = self.simulate(granularity="page128")
        self.assertEqual(result["candidate_native_units"], 200)
        self.assertLessEqual(result["predicted_native_units"], 72)
        self.assertEqual(result["prefetched_bytes"], 256)
        self.assertEqual(result["attention_readiness_assertions"], 32)

    def test_capacity_rejects_entire_attention_footprint(self):
        with self.assertRaisesRegex(ValueError, "complete single-attention"):
            self.simulate(granularity="page128", config=dict(self.config, capacity_bytes=256))
        with self.assertRaisesRegex(ValueError, "complete single-attention"):
            self.simulate(config=dict(self.config, capacity_bytes=8))

    def test_zero_and_subpage_budget(self):
        for budget in (0, 255):
            result = self.simulate(granularity="page128", config=dict(self.config, prefetch_budget_bytes=budget))
            self.assertEqual(result["prefetched_bytes"], 0)
        with self.assertRaisesRegex(ValueError, "byte sizes"):
            validate_config({"schema_version": 1, "capacity_bytes": 512, "prefetch_budget_bytes": 0})

    def test_lru_protects_complete_demand_under_pressure(self):
        cache = LRU(4)
        cache.put("a", 2, {1})
        cache.put("b", 2, {2})
        cache.touch("a")
        cache.put("c", 2, {3}, prefetch=True)
        self.assertNotIn("b", cache.entries)
        cache.put("d", 2, {4}, protected={"a"})
        self.assertEqual(set(cache.entries), {"a", "d"})
        with self.assertRaisesRegex(ValueError, "complete attention"):
            cache.put("e", 2, {5}, protected={"a", "d"})

    def test_split_rejects_session_episode_pair_and_repeat_aliases(self):
        for field in ("episode_id", "pair_id", "source_trace_id"):
            data = copy.deepcopy(self.sidecar)
            train = data["events"][0]
            evaluation = next(e for e in data["events"] if e["split"] == "eval")
            evaluation[field] = train[field]
            with self.assertRaisesRegex(ValueError, "leakage"):
                validate_sidecar(data, self.root)
        data = copy.deepcopy(self.sidecar)
        data["events"][1]["split"] = "eval"
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_sidecar(data, self.root)
        data = copy.deepcopy(self.sidecar)
        evaluation = next(e for e in data["events"] if e["split"] == "eval")
        for s in data["snapshots"]:
            if s["snapshot_id"] in {evaluation["snapshot_id"], evaluation["previous_snapshot_id"]}:
                s["session_id"] = self.old["session_id"]
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_sidecar(data, self.root)

    def test_trace_requires_window_and_rejects_duplicates(self):
        path = self.root/"traces/rank0.jsonl"
        lines = path.read_text().splitlines()
        path.write_text("\n".join(lines[1:])+"\n")
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.rows()
        path.write_text("\n".join(lines+[lines[0]])+"\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.rows()

    def test_generation_is_local_not_recall(self):
        result = self.simulate(strategy="demand_only")
        self.assertEqual(result["synchronous_recall_bytes"], 8)
        result = self.simulate(strategy="demand_only", granularity="page128")
        self.assertEqual(result["synchronous_recall_bytes"], 256)

    def test_all_strategies_equal_budgets_and_frozen_eval(self):
        report = run_replay(self.root/"traces", self.sidecar, self.config, self.root)
        self.assertEqual(report["evaluation_table_updates"], 0)
        self.assertEqual(report["training_events"], 8)
        for r in report["results"]:
            self.assertLessEqual(r["prefetched_bytes"], 256)
            self.assertEqual(r["prefetched_bytes"], r["useful_prefetch_bytes"]+r["wasted_prefetch_bytes"])
            self.assertEqual(r["attention_readiness_assertions"], 32)
            self.assertLessEqual(r["oracle_sync_lower_bound_bytes"], r["synchronous_recall_bytes"])
        pairs = report["summary"]["native"]["paired_differences"]["transition"]
        self.assertEqual(pairs["complete_pairs"], 4)
        self.assertEqual(pairs["event_minus_control_sync_bytes"]["n"], 1)
        self.assertIsNone(pairs["group_bootstrap_95pct"])

    def test_eval_selected_sets_do_not_modify_training(self):
        before = run_replay(self.root/"traces", self.sidecar, self.config, self.root)
        lines = []
        eval_ids = {e["request_id"]+"-abc12345" for e in self.sidecar["events"] if e["split"] == "eval"}
        for row in self.trace:
            row = copy.deepcopy(row)
            if row["request_id"] in eval_ids and row["prompt_position"] >= 512:
                row["raw_selected_ids"][:4] = [440, 441, 442, 443]
                row["logical_block_ids"][:4] = [3]*4
            lines.append(json.dumps(row))
        (self.root/"traces/rank0.jsonl").write_text("\n".join(lines)+"\n")
        after = run_replay(self.root/"traces", self.sidecar, self.config, self.root)
        for a, b in zip(before["results"], after["results"]):
            if a["strategy"] != "oracle_lookahead":
                self.assertEqual(a["prediction_sha256"], b["prediction_sha256"])
                self.assertEqual(a["table_sources"], b["table_sources"])
        self.assertEqual(before["training_table_cells"], after["training_table_cells"])


    def test_real_trace_rejects_partial_profile(self):
        data = copy.deepcopy(self.sidecar)
        data["synthetic"] = False
        with self.assertRaisesRegex(ValueError, "layer coverage"):
            run_replay(self.root/"traces", data, self.config, self.root)

    def test_generated_inventory_cannot_be_future_selected_subset(self):
        cur = copy.deepcopy(self.cur)
        cur["lanes"][0]["generated_ids"] = [260, 261, 262, 263]
        with self.assertRaisesRegex(ValueError, "inventory must be complete"):
            validate_snapshot(cur, self.root)

    def test_compressed_end_to_end_without_ratio_position_guess(self):
        source = self.root/"synthetic_compressor.py"
        source.write_text("# Synthetic support contract, not the deployed kernel.\n")
        profile = load_profile("deepseek_v4")
        evidence = {"path": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "start_line": 1, "end_line": 1, "explanation": "Synthetic support with overlaps and nonuniform positions."}
        intervals = {"0": [0, 8, 7], "1": [124, 132, 131], "2": [256, 264, 263],
                     "3": [508, 512, 511], "4": [512, 520, 519], "5": [528, 536, 535]}
        for snapshot in (self.old, self.cur):
            snapshot.update(model_id=profile["model_id"], model_revision=profile["model_revision"])
            snapshot["lanes"] = [{"rank": 0, "layer": "model.layers.2", "kv_kind": "compressed_kv_token_position",
                                  "generated_ids": [0, 1, 2, 3], "support_evidence": evidence,
                                  "support_intervals": {k: v for k, v in intervals.items() if v[1] <= len(snapshot["token_ids"])}}]
        self.cur["token_ids"][130] = 10000
        rows = []
        for position in range(511, 544):
            ids = [0] if position == 511 else [1, 2]
            if position >= 519:
                ids.append(4)
            ids += [-1]*(512-len(ids))
            rows.append({"schema_version": 1, "run_id": self.event["run_id"], "request_id": self.event["request_id"],
                         "rank": 0, "layer": "model.layers.2", "prompt_position": position, "context_len": position+1,
                         "request_context_len": 544, "raw_index_unit": "compressed_kv_token_position", "invalid_sentinel": -1,
                         "compression_ratio": 4, "compressed_block_size": 128, "selected_width": 512,
                         "raw_selected_ids": ids, "logical_compressed_block_ids": [u//128 if u >= 0 else -1 for u in ids],
                         "timestamp_ns": position+1})
        (self.root/"traces/rank0.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
        data = {"schema_version": 1, "synthetic": True, "snapshots": [self.old, self.cur], "events": [self.event]}
        cfg = dict(self.config, kv_unit_bytes={"compressed_kv_token_position": 2})
        report = run_replay(self.root/"traces", data, cfg, self.root)
        self.assertTrue(all(r["candidate_native_units"] == 1 and r["attention_readiness_assertions"] == 32 for r in report["results"]))


    def test_chained_restore_reuses_generated_post_window_prefix(self):
        post = copy.deepcopy(self.cur)
        post.update(snapshot_id="post-window", generated_tokens=544)
        post["lanes"][0]["generated_ids"] = list(range(544))
        second = copy.deepcopy(post)
        second.update(snapshot_id="second-window", context_version="v3", sequence=3, token_ids=list(range(576)))
        second["regions"][-1]["end"] = 576
        first_event = copy.deepcopy(self.event)
        first_event["post_window_snapshot_id"] = post["snapshot_id"]
        second_event = copy.deepcopy(first_event)
        second_event.pop("post_window_snapshot_id")
        second_event.update(event_id="second-event", request_id="second-request", previous_snapshot_id=post["snapshot_id"],
                            snapshot_id=second["snapshot_id"], event_position=544, resume_position=544, pair_id="second-pair")
        frames = []
        template = self.trace[0]
        for event, start, length in ((first_event, 512, 544), (second_event, 544, 576)):
            for pos in range(start-1, min(start+32, length)):
                row = copy.deepcopy(template)
                raw = [0, 1, 2, 3] if pos == start-1 else [260, 261, 262, 263, pos]
                raw += [-1]*(2048-len(raw))
                row.update(request_id=event["request_id"], prompt_position=pos, context_len=pos+1, request_context_len=length,
                           raw_selected_ids=raw, logical_block_ids=[u//128 if u >= 0 else -1 for u in raw])
                frames.append(row)
        (self.root/"traces/rank0.jsonl").write_text("".join(json.dumps(r)+"\n" for r in frames))
        data = {"schema_version": 1, "synthetic": True, "snapshots": [self.old, self.cur, post, second], "events": [first_event, second_event]}
        report = run_replay(self.root/"traces", data, self.config, self.root)
        first = [r for r in report["results"] if r["event_id"] == first_event["event_id"]]
        later = [r for r in report["results"] if r["event_id"] == second_event["event_id"]]
        self.assertTrue(all(r["candidate_native_units"] == 512 for r in first))
        self.assertTrue(all(r["candidate_native_units"] == 544 for r in later))
        self.assertTrue(all(r["attention_readiness_assertions"] == 32 for r in later))


    def test_multiple_ranks_do_not_satisfy_eight_event_threshold(self):
        table = TransitionTable()
        units = set(range(512))
        for event_id in range(8):
            for rank in range(8):
                lane = dict(self.lane, rank=rank)
                table.update(self.cur, lane, units, {260}, "s", "e", observation_id=str(event_id))
            key = table.key(self.cur, self.lane, "s", "e")
            self.assertEqual(table.tables[key]["observations"], event_id+1)
            expected = "event" if event_id < 7 else "previous_state+event"
            self.assertEqual(table.rank(self.cur, self.lane, units, "s", "e")[1], expected)


if __name__ == "__main__":
    unittest.main()
