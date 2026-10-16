"""Prediction-only replay: visibility batches are explicit, never inferred.

No selected-set trace can prove transfer overlap, latency or device residency.
All strategies share the same candidate budget and eligible block universe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

from .history_prefetch import AccessHistoryPredictor, PredictionContext, block_ids, integer, require


STRATEGIES = ("none", "last_selection", "global_frequency", "stage_frequency",
              "global_recency", "stage_recency", "global_region", "stage_region",
              "oracle_candidate_upper_bound")
LANE_FIELDS = ("task_id", "run_id", "request_id", "context_version", "rank", "layer", "phase")


def validate_rows(rows):
    require(bool(rows), "No access rows")
    lanes = defaultdict(list)
    source_kinds = set()
    for row in rows:
        require(row.get("schema_version") == 1, "Unsupported history trace schema")
        require(row.get("source_kind") in {"synthetic_fixture", "native_trace"}, "Missing provenance")
        source_kinds.add(row["source_kind"])
        for field in LANE_FIELDS + ("stage", "event_type"):
            if field == "rank":
                require(integer(row.get(field)), "Invalid rank")
            else:
                require(isinstance(row.get(field), str) and bool(row[field]), f"Missing {field}")
        require(row["phase"] in {"prefill", "decode"}, "Unknown phase")
        for field in ("query", "batch", "prediction_step", "completion_step", "block_count"):
            require(integer(row.get(field)), f"Invalid {field}")
        require(row["completion_step"] > row["prediction_step"], "Invalid visibility interval")
        eligible, selected = block_ids(row["eligible_blocks"]), block_ids(row["selected_blocks"])
        require(all(b < row["block_count"] for b in eligible + selected), "Block outside KV namespace")
        lanes[tuple(row[f] for f in LANE_FIELDS)].append(row)
    require(len(source_kinds) == 1, "Synthetic and native traces cannot be mixed")
    for lane in lanes.values():
        require([r["query"] for r in lane] == list(range(len(lane))), "Missing/duplicate/reordered query")
        batches = defaultdict(list)
        for row in lane:
            batches[row["batch"]].append(row)
        require(list(batches) == list(range(len(batches))), "Missing/reordered visibility batch")
        require([r["batch"] for r in lane] == sorted(r["batch"] for r in lane), "Interleaved batch")
        previous_completion = -1
        for batch in batches.values():
            require(len({(r["prediction_step"], r["completion_step"], r["stage"], r["event_type"])
                         for r in batch}) == 1, "Inconsistent batch visibility/stage")
            require(batch[0]["prediction_step"] > previous_completion, "Overlapping visibility batches")
            previous_completion = batch[0]["completion_step"]
            require(lane[0]["phase"] != "decode" or len(batch) == 1, "Decode batch must have one query")
    return lanes


def summarize(records):
    groups = defaultdict(list)
    for record in records:
        key = (record["strategy"], record["phase"], record["event_type"], record["stage_window"])
        groups[key].append(record)
    result = []
    for key, items in sorted(groups.items()):
        selected = sum(r["selected_count"] for r in items)
        candidates = sum(r["candidate_count"] for r in items)
        hits = sum(r["candidate_hits"] for r in items)
        overlap = [r["previous_query_jaccard"] for r in items if r["previous_query_jaccard"] is not None]
        transitions = [r["previous_query_jaccard"] for r in items
                       if r["stage_changed"] and r["previous_query_jaccard"] is not None]
        task_rows = defaultdict(list)
        for row in items:
            task_rows[row["task_id"]].append(row)
        per_task = []
        for task_id, task_items in sorted(task_rows.items()):
            total = sum(r["selected_count"] for r in task_items)
            per_task.append({"task_id": task_id, "rows": len(task_items),
                             "coverage": sum(r["candidate_hits"] for r in task_items) / total if total else None})
        result.append(dict(zip(("strategy", "phase", "event_type", "stage_window"), key),
                           queries=len(items), candidate_count=candidates, selected_count=selected,
                           candidate_hits=hits, candidate_precision=hits / candidates if candidates else None,
                           native_set_coverage=hits / selected if selected else None,
                           candidate_waste=candidates - hits,
                           mean_previous_query_jaccard=sum(overlap) / len(overlap) if overlap else None,
                           mean_stage_transition_jaccard=sum(transitions) / len(transitions) if transitions else None,
                           prediction_cpu_ns=sum(r["prediction_cpu_ns"] for r in items),
                           per_task=per_task))
    return result


def replay(rows, *, budget_blocks, region_blocks=4, boundary_queries=4):
    require(integer(budget_blocks) and integer(region_blocks, 1) and integer(boundary_queries, 1),
            "Invalid replay budget/region/boundary width")
    lanes = validate_rows(rows)
    records = []
    for lane_key, lane in sorted(lanes.items()):
        predictors = {}
        for strategy in STRATEGIES[:-1]:
            scope, signal = strategy.split("_", 1) if strategy.startswith(("global_", "stage_")) else ("global", strategy)
            predictors[strategy] = AccessHistoryPredictor(scope=scope, signal=signal, region_blocks=region_blocks)
        batches = defaultdict(list)
        for row in lane:
            batches[row["batch"]].append(row)
        stage, offset, previous = None, 0, set()
        for batch in batches.values():
            predictions = {}
            # Freeze every predictor until all queries in this batch are planned.
            # The planners receive only PredictionContext, never the trace row.
            for row in batch:
                context = PredictionContext(row["query"], row["prediction_step"], row["stage"],
                                            tuple(row["eligible_blocks"]))
                for strategy, predictor in predictors.items():
                    start = time.perf_counter_ns()
                    candidates = predictor.predict(context, budget=budget_blocks)
                    predictions[(row["query"], strategy)] = ([c.block for c in candidates], time.perf_counter_ns() - start)
            for row in batch:
                selected = set(row["selected_blocks"])
                changed = row["stage"] != stage
                if changed:
                    stage, offset = row["stage"], 0
                union = previous | selected
                common = dict(zip(LANE_FIELDS, lane_key), query=row["query"], stage=stage,
                              event_type=row["event_type"], stage_offset=offset,
                              stage_window="boundary" if offset < boundary_queries else "steady",
                              selected_count=len(selected), stage_changed=changed,
                              previous_query_jaccard=len(previous & selected) / len(union)
                              if union and row["query"] > 0 else None,
                              eligible_selected_count=len(selected & set(row["eligible_blocks"])))
                # Privileged diagnostic is built only after every causal prediction.
                oracle = sorted(selected & set(row["eligible_blocks"]))[:budget_blocks]
                for strategy in STRATEGIES:
                    ids, elapsed = ((oracle, 0) if strategy == STRATEGIES[-1] else
                                    predictions[(row["query"], strategy)])
                    records.append(dict(common, strategy=strategy, candidate_blocks=ids,
                                        candidate_count=len(ids), candidate_hits=len(selected & set(ids)),
                                        prediction_cpu_ns=elapsed))
                previous, offset = selected, offset + 1
            for row in batch:
                for predictor in predictors.values():
                    predictor.observe(query=row["query"], completion_step=row["completion_step"],
                                      selected_blocks=row["selected_blocks"])
    return {"schema_version": 1, "source_kind": rows[0]["source_kind"],
            "budget_blocks": budget_blocks, "region_blocks": region_blocks,
            "boundary_queries": boundary_queries,
            "evaluation_kind": "candidate_prediction_only",
            "online_performance_validated": False, "agent_generalization_validated": False,
            "capacity_or_transfer_simulated": False,
            "summary": summarize(records), "rows": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget-blocks", type=int, required=True)
    parser.add_argument("--region-blocks", type=int, default=4)
    parser.add_argument("--boundary-queries", type=int, default=4)
    args = parser.parse_args()
    source = args.trace.read_bytes()
    rows = [json.loads(line) for line in source.splitlines() if line.strip()]
    result = replay(rows, budget_blocks=args.budget_blocks, region_blocks=args.region_blocks,
                    boundary_queries=args.boundary_queries)
    result["input_sha256"] = hashlib.sha256(source).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    detail = result.pop("rows")
    (args.output_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in detail), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"queries": len(rows), "predictions": len(detail),
                      "output_dir": str(args.output_dir), "online_performance_validated": False}))


if __name__ == "__main__":
    main()
