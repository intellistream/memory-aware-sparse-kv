"""Import audited, fixed-continuation GLM decode traces for diagnostics only.

No prefill completion order or cross-event KV identity is inferred. Explicit
case IDs keep the chosen diagnostic scope visible and avoid silent sampling.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .history_prefetch import integer, require


def import_decode(run_dir, case_ids, *, block_tokens):
    run_dir = Path(run_dir)
    require(integer(block_tokens, 1), "Invalid block width")
    require(case_ids and len(set(case_ids)) == len(case_ids), "Explicit distinct case IDs required")
    audit_bytes = (run_dir / "independent-audit.json").read_bytes()
    measurement_bytes = (run_dir / "measurements.jsonl").read_bytes()
    audit = json.loads(audit_bytes)
    require(audit.get("passed") is True, "Archive audit did not pass")
    selected = [m for m in map(json.loads, measurement_bytes.splitlines())
                if m["mode"] == "no_offload" and m["repeat"] == 0 and m["case_id"] in case_ids]
    require(len(selected) == len(case_ids) and {m["case_id"] for m in selected} == set(case_ids),
            "Missing/duplicate no-offload repeat-zero case")
    result, sources = [], []
    for measurement in selected:
        basename = Path(measurement["native_selected_trace"]).name
        path = run_dir / "selected" / basename
        compressed = path.read_bytes()
        digest = hashlib.sha256(compressed).hexdigest()
        require(digest == audit["selected_trace_sha256"].get(basename), "Trace differs from audit")
        raw = gzip.decompress(compressed)
        require(hashlib.sha256(raw).hexdigest() == measurement["native_selected_set_sha256"],
                "Selected set digest mismatch")
        rows = [json.loads(line) for line in raw.splitlines()]
        require(len(rows) == 378, "Expected six layers of 32 prefill + 31 decode rows")
        lanes = defaultdict(list)
        seen = set()
        for row in rows:
            key = (row["layer"], row["position"])
            require(key not in seen, "Duplicate native query")
            seen.add(key)
            require(row["phase"] in {"prefill", "decode"}, "Unknown native phase")
            if row["phase"] == "decode":
                lanes[row["layer"]].append(row)
        require(len(lanes) == 6, "Incomplete GLM layer coverage")
        for layer, decode in sorted(lanes.items()):
            positions = list(range(measurement["prompt_tokens"], measurement["prompt_tokens"] + 31))
            require([r["position"] for r in decode] == positions, "Incomplete/reordered decode positions")
            for query, row in enumerate(decode):
                position = row["position"]
                native = row["selected_ids"]
                require(len(native) == 2048 and all(type(v) is int and -1 <= v <= position for v in native),
                        "Invalid native width or non-causal KV token")
                valid = [v for v in native if v != -1]
                require(len(valid) == len(set(valid)), "Duplicate native token IDs")
                result.append({"schema_version": 1, "source_kind": "native_trace",
                               "task_id": measurement["domain"] + ":" + measurement["episode_id"],
                               "run_id": audit["run_id"], "request_id": measurement["case_id"],
                               "context_version": measurement["prompt_token_ids_sha256"],
                               "rank": 0, "layer": layer, "phase": "decode", "query": query,
                               "batch": query, "prediction_step": 2 * query, "completion_step": 2 * query + 1,
                               "stage": "observed-decode-window", "event_type": measurement["event_type"],
                               "block_count": (position + block_tokens) // block_tokens,
                               # Only blocks fully materialized before this decode query.
                               "eligible_blocks": list(range(position // block_tokens)),
                               "selected_blocks": sorted({v // block_tokens for v in valid})})
        sources.append({"case_id": measurement["case_id"], "trace": basename, "sha256": digest})
    manifest = {"schema_version": 1, "run_id": audit["run_id"], "block_tokens": block_tokens,
                "case_ids": case_ids, "sources": sources,
                "audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
                "measurements_sha256": hashlib.sha256(measurement_bytes).hexdigest(),
                "limitations": ["GLM tiny fixed-continuation replay, not autonomous Agent",
                                "decode only; no cross-event trajectory or prefill visibility inferred",
                                "rank 0 is the single-device archive namespace, not physical NPU ID",
                                "only fully generated blocks are eligible; first query cold-starts",
                                "periodic continuation/control artifacts remain; no generalization claim"],
                "rows": len(result)}
    return result, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument("--block-tokens", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, manifest = import_decode(args.run_dir, args.case_id, block_tokens=args.block_tokens)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    trace = args.output_dir / "access.jsonl"
    with trace.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")
    manifest["output_sha256"] = hashlib.sha256(trace.read_bytes()).hexdigest()
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
