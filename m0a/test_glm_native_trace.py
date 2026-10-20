"""Fail-closed native capture and offline evidence checks; no NPU imports."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from m0a.glm_agent_cases import sha_file
from m0a.glm_native_trace import (audit_archive, audit_repeats, decode_history,
                                digest_json, targets, validate_request)
from m0a.run_glm_native_trace import write_json, write_trace


CASE = {"case_id": "test:event", "event_position": 4096, "prompt_tokens": 8192,
        "domain": "test", "episode_id": "episode", "event_type": "tool_result",
        "prompt_token_ids_sha256": "a" * 64, "forced_continuation_token_ids": list(range(32))}


def fixture():
    rows = []
    for layer in range(6):
        for position in sorted(targets(CASE)):
            decode = position >= CASE["prompt_tokens"]
            start, count = (position, 1) if decode else (0, 8192)
            offset = min(layer, 2) * 100
            ids = list(range(offset, offset + 2048))
            rows.append({"schema_version": 1, "case_id": CASE["case_id"],
                "layer": f"model.layers.{layer}.self_attn.attn", "position": position,
                "phase": "decode" if decode else "prefill", "query_start": start,
                "query_rows": count, "context_len": start + count, "row_in_batch": position - start,
                "candidate_range": [0, start], "capture_site": "attention_entry",
                "indexer_source_layer": min(layer, 2), "raw_selected_ids": ids,
                "indexer_selected_sha256": digest_json(ids),
                "actual_seq_lengths_query": [count], "actual_seq_lengths_key": [start + count]})
    return rows


def archive(root):
    cases = root / "cases.json"
    write_json(cases, {"cases": [CASE]})
    run = root / "run"
    run.mkdir()
    rows = validate_request(fixture(), CASE)
    requests = []
    for repeat in range(2):
        path = run / f"repeat-{repeat}.jsonl.gz"
        write_trace(path, rows)
        requests.append({"case_id": CASE["case_id"], "repeat": repeat, "trace": path.name,
                         "sha256": sha_file(path), "output_tokens": list(range(32)), "rows": rows})
    write_trace(run / "greedy.jsonl.gz", rows)
    audit = audit_repeats(requests, repeats=2)
    audit.update(requests=[{k: v for k, v in r.items() if k != "rows"} for r in requests],
                 greedy_checks=[{"case_id": CASE["case_id"], "trace": "greedy.jsonl.gz",
                                 "trace_off_tokens": list(range(32)), "trace_on_tokens": list(range(32)),
                                 "trace_off_repeats": [list(range(32)), list(range(32))],
                                 "native_greedy_repeat_stable": True, "greedy_output_equal": True}],
                 all_greedy_outputs_equal=True, all_native_greedy_repeats_stable=True)
    write_json(run / "audit.json", audit)
    history = decode_history(rows, CASE, run_id="run", repeat=0)
    (run / "history-decode.jsonl").write_text("".join(json.dumps(r) + "\n" for r in history))
    manifest = {"status": "passed", "case_ids": [CASE["case_id"]], "repeats": 2, "run_id": "run",
                "cases_sha256": sha_file(cases), "artifacts": {p.name: sha_file(p) for p in run.iterdir()}}
    write_json(run / "manifest.json", manifest)
    return run, cases


class NativeTraceTests(unittest.TestCase):
    def test_predecessor_visibility_follows_batch_not_position(self):
        rows = validate_request(fixture(), CASE)
        self.assertEqual(len(rows), 390)
        keyed = {r["position"]: r for r in rows if r["layer"] == "model.layers.0.self_attn.attn"}
        self.assertEqual(keyed[4096]["predecessor_position"], 4095)
        self.assertFalse(keyed[4096]["predecessor_visible_before_batch"])
        self.assertTrue(keyed[8192]["predecessor_visible_before_batch"])
        self.assertEqual(keyed[8192]["predecessor_position"], 8191)

    def test_decode_export_is_contiguous_and_excludes_current_kv(self):
        rows = decode_history(fixture(), CASE, run_id="run", repeat=0)
        self.assertEqual(len(rows), 186)
        self.assertEqual([r["query"] for r in rows[:31]], list(range(31)))
        self.assertEqual(rows[0]["eligible_blocks"][-1], 8191)
        self.assertTrue(all(r["phase"] == "decode" for r in rows))

    def test_missing_or_duplicate_rows_rejected(self):
        rows = fixture()
        for invalid in (rows[1:], rows + [rows[0]]):
            with self.assertRaises(ValueError):
                validate_request(invalid, CASE)

    def test_causal_ids_and_capture_metadata_rejected(self):
        mutations = [lambda r: r.update(candidate_range=[0, 8192]),
                     lambda r: r.update(indexer_source_layer=2),
                     lambda r: r.update(capture_site="after_attention"),
                     lambda r: r.update(actual_seq_lengths_key=[9999]),
                     lambda r: r["raw_selected_ids"].__setitem__(0, 9999),
                     lambda r: r["raw_selected_ids"].__setitem__(0, 1),
                     lambda r: r.update(indexer_selected_sha256="bad")]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                rows = fixture()
                mutate(rows[0])
                with self.assertRaises(ValueError):
                    validate_request(rows, CASE)

    def test_shared_layer_must_match_source(self):
        rows = fixture()
        row = next(r for r in rows if r["layer"] == "model.layers.3.self_attn.attn")
        row["raw_selected_ids"][0] = 3000
        row["indexer_selected_sha256"] = digest_json(row["raw_selected_ids"])
        with self.assertRaisesRegex(ValueError, "Shared layer"):
            validate_request(rows, CASE)

    def test_optional_full_score_reference_is_independently_checked(self):
        rows = fixture()
        for row in rows:
            selected = set(row['raw_selected_ids'])
            row['full_scores'] = [2.0 if b in selected else 0.0 for b in range(row['position'] + 1)]
            row['score_source_kind'] = 'native_inputs_full_reference_checked_against_topk'
        validate_request(rows, CASE)
        rows[0]['full_scores'][3000] = 99
        with self.assertRaisesRegex(ValueError, 'differs from native'):
            validate_request(rows, CASE)

    def test_stability_preserves_changed_set_and_order_separately(self):
        rows = fixture()
        altered = copy.deepcopy(rows)
        altered[0]["raw_selected_ids"].reverse()
        requests = [{"case_id": "case", "repeat": i, "rows": r} for i, r in enumerate((rows, altered))]
        report = audit_repeats(requests, repeats=2)
        self.assertTrue(report["all_selected_sets_stable"])
        self.assertFalse(report["cases"][0]["order_stable"])
        altered[1]["raw_selected_ids"][0] = 3000
        report = audit_repeats(requests, repeats=2)
        self.assertFalse(report["all_selected_sets_stable"])
        self.assertEqual(len(report["cases"][0]["unstable_set_rows"]), 1)
        with self.assertRaisesRegex(ValueError, "repetition"):
            audit_repeats(requests[:1], repeats=2)

    def test_archive_recomputes_raw_rows_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            run, cases = archive(Path(directory))
            result = audit_archive(run, cases)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["history_rows"], 186)
            self.assertEqual(result["fixed_requests"], 2)
            with (run / "repeat-0.jsonl.gz").open("ab") as output:
                output.write(b"modified")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                audit_archive(run, cases)

    def test_resealed_forged_stability_report_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run, cases = archive(Path(directory))
            audit = json.loads((run / "audit.json").read_text())
            audit["all_selected_sets_stable"] = False
            write_json(run / "audit.json", audit)
            manifest = json.loads((run / "manifest.json").read_text())
            manifest["artifacts"]["audit.json"] = sha_file(run / "audit.json")
            write_json(run / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "stability differs"):
                audit_archive(run, cases)

    def test_native_greedy_instability_retained_even_when_trace_pair_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            run, cases = archive(Path(directory))
            audit = json.loads((run / "audit.json").read_text())
            audit["greedy_checks"][0]["trace_off_repeats"][1][0] = 999
            audit["greedy_checks"][0]["native_greedy_repeat_stable"] = False
            audit["all_native_greedy_repeats_stable"] = False
            write_json(run / "audit.json", audit)
            manifest = json.loads((run / "manifest.json").read_text())
            manifest["artifacts"]["audit.json"] = sha_file(run / "audit.json")
            manifest["status"] = "unstable"
            write_json(run / "manifest.json", manifest)
            result = audit_archive(run, cases)
            self.assertEqual(result["status"], "unstable")
            self.assertTrue(result["all_greedy_outputs_equal"])
            self.assertFalse(result["all_native_greedy_repeats_stable"])


if __name__ == "__main__":
    unittest.main()
