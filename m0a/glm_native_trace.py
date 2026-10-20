"""Read-only native GLM indexer/attention capture with explicit batch boundaries."""
from __future__ import annotations

import hashlib
import gzip
import json
import re
import math
from pathlib import Path

from .history_prefetch import require


def digest_json(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def targets(case):
    boundary, prompt = case["event_position"], case["prompt_tokens"]
    require(0 < boundary and boundary + 32 < prompt, "Invalid sampled event window")
    decode_queries = case.get('capture_decode_queries', 31)
    require(type(decode_queries) is int and 0 <= decode_queries <= 255, 'Invalid decode capture window')
    return set(range(boundary - 1, boundary + 32)) | set(range(prompt - 1, prompt + decode_queries))


def selected_ids(values, position):
    require(len(values) == 2048 and all(type(v) is int and -1 <= v <= position for v in values),
            "Invalid native width, padding or causal index")
    valid = [v for v in values if v != -1]
    require(len(valid) == len(set(valid)), "Duplicate native selected ID")
    return valid


def layer_number(name):
    match = re.fullmatch(r"model\.layers\.([0-5])\.self_attn\.attn", name)
    require(match is not None, "Unexpected pinned GLM layer name")
    return int(match[1])


def validate_request(rows, case):
    expected = {(layer, pos) for layer in range(6) for pos in targets(case)}
    seen = set()
    by_key = {}
    for row in rows:
        layer, position = layer_number(row["layer"]), row["position"]
        key = (layer, position)
        require(key in expected and key not in seen, "Unexpected/duplicate native row")
        seen.add(key)
        by_key[key] = row
        require(row["case_id"] == case["case_id"], "Case identity mismatch")
        require(row["phase"] == ("prefill" if position < case["prompt_tokens"] else "decode"),
                "Wrong native phase")
        start, count = row["query_start"], row["query_rows"]
        require(type(start) is int and start >= 0 and type(count) is int and count > 0
                and start <= position < start + count == row["context_len"], "Invalid query batch")
        require(row["row_in_batch"] == position - start, "Incorrect capture row offset")
        require(row["schema_version"] == 1 and row["actual_seq_lengths_query"] == [count]
                and row["actual_seq_lengths_key"] == [start + count], "Sequence lengths disagree with batch")
        require(row["phase"] != "decode" or count == 1, "Decode is not a single query")
        require(row["candidate_range"] == [0, start], "Candidate universe contains current-batch KV")
        require(row["capture_site"] == "attention_entry" and row["indexer_source_layer"] == min(layer, 2),
                "Capture site or indexer reuse mapping mismatch")
        selected_ids(row["raw_selected_ids"], position)
        require(row["indexer_selected_sha256"] == digest_json(row["raw_selected_ids"]),
                "Indexer/attention selected set mismatch")
        if 'full_scores' in row:
            scores = row['full_scores']
            selected = set(selected_ids(row['raw_selected_ids'], position))
            require(row.get('score_source_kind') == 'native_inputs_full_reference_checked_against_topk'
                    and len(scores) == position + 1 and all(math.isfinite(s) for s in scores),
                    'Incomplete/nonfinite native score reference')
            floor = min(scores[i] for i in selected)
            require(all(s <= floor + 1e-4 for i, s in enumerate(scores) if i not in selected),
                    'Full score reference differs from native selection')
    require(seen == expected, "Missing native layer/query rows, including predecessors")
    for (layer, position), row in by_key.items():
        source = by_key[(min(layer, 2), position)]
        require(row["raw_selected_ids"] == source["raw_selected_ids"], "Shared layer selection mismatch")
    enriched = []
    for key, row in sorted(by_key.items()):
        layer, position = key
        predecessor = by_key.get((layer, position - 1))
        enriched.append(dict(row,
            predecessor_position=position - 1 if predecessor else None,
            predecessor_selected_sha256=digest_json(predecessor["raw_selected_ids"]) if predecessor else None,
            predecessor_visible_before_batch=bool(predecessor and predecessor["query_start"] < row["query_start"]),
            selected_blocks_128=sorted({v // 128 for v in row["raw_selected_ids"] if v >= 0})))
    return enriched


def audit_repeats(requests, *, repeats):
    """Preserve unstable results rather than treating them as stable or dropping them."""
    require(type(repeats) is int and repeats >= 2, "At least two repeats required")
    grouped = {}
    for request in requests:
        grouped.setdefault(request["case_id"], []).append(request)
    reports = []
    for case_id, group in sorted(grouped.items()):
        require(sorted(r["repeat"] for r in group) == list(range(repeats)), "Missing/duplicate repetition")
        group.sort(key=lambda r: r["repeat"])
        reference = {(r["layer"], r["position"]): r for r in group[0]["rows"]}
        unstable_set, unstable_order = set(), set()
        for request in group[1:]:
            current = {(r["layer"], r["position"]): r for r in request["rows"]}
            require(current.keys() == reference.keys(), "Repeat coverage differs")
            for key, row in current.items():
                other = reference[key]
                require((row["query_start"], row["query_rows"], row["candidate_range"]) ==
                        (other["query_start"], other["query_rows"], other["candidate_range"]),
                        "Repeat execution/candidate shape differs")
                if set(row["raw_selected_ids"]) != set(other["raw_selected_ids"]):
                    unstable_set.add(key)
                if row["raw_selected_ids"] != other["raw_selected_ids"]:
                    unstable_order.add(key)
        reports.append({"case_id": case_id, "repeats": repeats, "rows_per_repeat": len(reference),
                        "set_stable": not unstable_set, "order_stable": not unstable_order,
                        "unstable_set_rows": sorted(unstable_set), "unstable_order_rows": sorted(unstable_order)})
    require(reports, "No repeated requests")
    return {"all_selected_sets_stable": all(r["set_stable"] for r in reports), "cases": reports}


class NativeCapture:
    def __init__(self, *, capture_full_scores=False):
        self.capture_full_scores = capture_full_scores
        self.case = None
        self.rows = []
        self.indexer_rows = {}

    def begin(self, case):
        self.case, self.rows, self.indexer_rows = case, [], {}
        self.indexer_scores = {}

    def _batch(self, metadata, query_rows, seq_q, seq_key):
        require(metadata.block_table.shape[0] == 1, "Capture requires one request")
        q_lengths = seq_q.detach().cpu().reshape(-1).tolist()
        k_lengths = seq_key.detach().cpu().reshape(-1).tolist()
        require(len(q_lengths) == len(k_lengths) == 1 and q_lengths[0] == query_rows,
                "Unsupported packed query layout")
        return int(k_lengths[0]) - query_rows, int(k_lengths[0]), q_lengths, k_lengths

    def install(self):
        from vllm_ascend.device.device_op import DeviceOperator
        self.operator = DeviceOperator
        self.saved_indexer = DeviceOperator.__dict__["indexer_select_post_process"]
        self.saved_attention = DeviceOperator.__dict__["execute_sparse_flash_attention_process"]
        native_indexer = DeviceOperator.indexer_select_post_process
        native_attention = DeviceOperator.execute_sparse_flash_attention_process
        capture = self

        def indexer(impl, q, scale, shape, weights, cache, metadata, seq_q, seq_key, c8, lightning):
            output = native_indexer(impl, q, scale, shape, weights, cache, metadata, seq_q, seq_key, c8, lightning)
            if capture.case is not None:
                start, _, _, _ = capture._batch(metadata, int(q.shape[0]), seq_q, seq_key)
                layer = layer_number(str(impl.layer_name))
                positions = sorted(p for p in targets(capture.case) if start <= p < start + q.shape[0])
                if capture.capture_full_scores and positions:
                    require(not c8 and scale is None, 'Full score capture requires pinned BF16 indexer')
                    from .glm_live_score import verify_rows
                    verified = verify_rows(q, weights, cache[2], metadata.block_table, output,
                                           query_start=start, rows=[p - start for p in positions])
                    for pos, scores in verified.items():
                        capture.indexer_scores[(layer, pos)] = scores.detach().cpu().tolist()
                for pos in sorted(targets(capture.case)):
                    if start <= pos < start + q.shape[0]:
                        key = (layer, pos)
                        require(key not in capture.indexer_rows, "Duplicate indexer capture")
                        capture.indexer_rows[key] = output[pos - start].detach().cpu().reshape(-1).tolist()
            return output  # Exact native tensor object; no replacement or in-place writes.

        def attention(impl, q, rope, cache, topk, metadata, seq_q, seq_key, **kwargs):
            if capture.case is not None:
                start, end, q_lengths, k_lengths = capture._batch(metadata, int(q.shape[0]), seq_q, seq_key)
                name = str(impl.layer_name)
                layer = layer_number(name)
                for pos in sorted(targets(capture.case)):
                    if start <= pos < end:
                        ids = topk[pos - start].detach().cpu().reshape(-1).tolist()
                        source = capture.indexer_rows.get((min(layer, 2), pos))
                        require(source is not None and ids == source,
                                f"Native indexer/attention handoff mismatch at layer {layer}, position {pos}")
                        capture.rows.append({"schema_version": 1, "case_id": capture.case["case_id"],
                            "layer": name, "position": pos, "phase": "prefill" if pos < capture.case["prompt_tokens"] else "decode",
                            "query_start": start, "query_rows": int(q.shape[0]), "context_len": end,
                            "row_in_batch": pos - start, "candidate_range": [0, start],
                            "capture_site": "attention_entry", "indexer_source_layer": min(layer, 2),
                            "indexer_selected_sha256": digest_json(source), "raw_selected_ids": ids,
                            "actual_seq_lengths_query": q_lengths, "actual_seq_lengths_key": k_lengths})
                        if capture.capture_full_scores:
                            capture.rows[-1]['full_scores'] = capture.indexer_scores[(min(layer, 2), pos)]
                            capture.rows[-1]['score_source_kind'] = 'native_inputs_full_reference_checked_against_topk'
            return native_attention(impl, q, rope, cache, topk, metadata, seq_q, seq_key, **kwargs)

        DeviceOperator.indexer_select_post_process = staticmethod(indexer)
        DeviceOperator.execute_sparse_flash_attention_process = classmethod(
            lambda _cls, *args, **kwargs: attention(*args, **kwargs))

    def restore(self):
        self.operator.indexer_select_post_process = self.saved_indexer
        self.operator.execute_sparse_flash_attention_process = self.saved_attention


def decode_history(rows, case, *, run_id, repeat):
    """Only contiguous decode rows enter history replay; sampled prefill stays diagnostic."""
    enriched = validate_request(rows, case)
    output = []
    for row in enriched:
        if row["phase"] != "decode":
            continue
        query = row["position"] - case["prompt_tokens"]
        output.append({"schema_version": 1, "source_kind": "native_trace",
            "task_id": case["domain"] + ":" + case["episode_id"], "run_id": run_id,
            "request_id": f"{case['case_id']}:repeat-{repeat}", "context_version": case["prompt_token_ids_sha256"],
            "rank": 0, "layer": row["layer"], "phase": "decode", "query": query, "batch": query,
            "prediction_step": query * 2, "completion_step": query * 2 + 1,
            "stage": "decode-window", "event_type": case["event_type"], "block_count": row["position"] + 1,
            "eligible_blocks": list(range(*row["candidate_range"])),
            "selected_blocks": sorted(selected_ids(row["raw_selected_ids"], row["position"]))})
    return output


def audit_archive(root, cases_path):
    """Recompute from persisted raw traces, not the collector's pass flag."""
    from .glm_agent_cases import sha_file
    manifest = json.loads((root / "manifest.json").read_text())
    require(manifest["status"] in {"passed", "unstable"}, "Incomplete capture")
    require(sha_file(cases_path) == manifest["cases_sha256"], "Sealed cases changed")
    cases = {c["case_id"]: c for c in json.loads(cases_path.read_text())["cases"]}
    ids = manifest["case_ids"]
    require(ids and len(ids) == len(set(ids)) and all(c in cases for c in ids), "Invalid case coverage")
    artifacts = manifest["artifacts"]
    for name, digest in artifacts.items():
        require(Path(name).name == name and name not in {".", ".."}, "Unsafe artifact path")
        require(sha_file(root / name) == digest, "Artifact hash mismatch: " + name)
    require({"audit.json", "history-decode.jsonl"} <= artifacts.keys(), "Missing audit artifacts")
    reported = json.loads((root / "audit.json").read_text())
    requests, history = [], []
    expected = {(case_id, repeat) for case_id in ids for repeat in range(manifest["repeats"])}
    seen, trace_names = set(), set()
    for record in reported["requests"]:
        key, name = (record["case_id"], record["repeat"]), record["trace"]
        require(key in expected and key not in seen and name not in trace_names, "Invalid repeated request coverage")
        seen.add(key)
        trace_names.add(name)
        require(name in artifacts and artifacts[name] == record["sha256"], "Trace is not sealed")
        with gzip.open(root / name, "rt", encoding="utf-8") as stream:
            raw = [json.loads(line) for line in stream]
        case = cases[key[0]]
        rows = validate_request(raw, case)
        require(rows == raw, "Derived predecessor/block metadata changed")
        require(record["output_tokens"] == case["forced_continuation_token_ids"], "Fixed output changed")
        requests.append(dict(record, rows=rows))
        if key[1] == 0:
            history.extend(decode_history(rows, case, run_id=manifest["run_id"], repeat=0))
    require(seen == expected, "Missing repeated requests")
    actual_history = [json.loads(line) for line in (root / "history-decode.jsonl").read_text().splitlines()]
    require(actual_history == history, "History export differs from native decode")
    result = audit_repeats(requests, repeats=manifest["repeats"])
    for field in ("cases", "all_selected_sets_stable"):
        # Tuples become lists in serialized reports.
        require(json.loads(json.dumps(result[field])) == reported[field], "Reported stability differs")
    checks = reported["greedy_checks"]
    require([c["case_id"] for c in checks] == ids, "Missing/duplicate greedy checks")
    for check in checks:
        name = check["trace"]
        require(name in artifacts, "Greedy trace is not sealed")
        with gzip.open(root / name, "rt", encoding="utf-8") as stream:
            validate_request([json.loads(line) for line in stream], cases[check["case_id"]])
        require(len(check["trace_off_tokens"]) == len(check["trace_on_tokens"]) == 32, "Incomplete greedy output")
        require(check["greedy_output_equal"] == (check["trace_off_tokens"] == check["trace_on_tokens"]),
                "Greedy comparison differs")
        if "trace_off_repeats" in check:
            outputs = check["trace_off_repeats"]
            require(len(outputs) == manifest["repeats"] and all(len(t) == 32 for t in outputs)
                    and outputs[0] == check["trace_off_tokens"], "Incomplete native greedy repetitions")
            require(check["native_greedy_repeat_stable"] == all(t == outputs[0] for t in outputs),
                    "Native greedy stability differs")
    equal = all(c["greedy_output_equal"] for c in checks)
    require(equal == reported["all_greedy_outputs_equal"], "Reported greedy summary differs")
    native_stable = (all(c["native_greedy_repeat_stable"] for c in checks)
                     if all("trace_off_repeats" in c for c in checks) else None)
    require(native_stable == reported.get("all_native_greedy_repeats_stable"), "Reported native greedy summary differs")
    status = "passed" if equal and result["all_selected_sets_stable"] and native_stable is not False else "unstable"
    require(manifest["status"] == status, "Reported status differs")
    return dict(result, status=status, all_greedy_outputs_equal=equal, all_native_greedy_repeats_stable=native_stable,
                fixed_requests=len(requests), history_rows=len(history),
                manifest_sha256=sha_file(root / "manifest.json"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Offline audit of native GLM trace artifacts")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_archive(args.run_dir, args.cases), indent=2))
