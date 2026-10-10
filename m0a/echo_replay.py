#!/usr/bin/env python3
"""CPU reference for ECHO's score-based prefetch mechanisms.

The input must contain complete native indexer scores. A selected-set-only
trace cannot be upgraded to this format. This module models transfer counts,
not Ascend overlap, DMA time, or end-to-end serving performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from collections import defaultdict
from pathlib import Path


SCHEMA_VERSION = 1
PHASES = {"prefill", "decode"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _float_order_bin(value: float) -> int:
    """Use the leading eight bits of a sortable float32 representation."""
    bits = struct.unpack(">I", struct.pack(">f", float(value)))[0]
    sortable = (~bits & 0xFFFFFFFF) if bits & 0x80000000 else bits ^ 0x80000000
    return sortable >> 24


def approximate_prefill_topk(scores: list[tuple[int, float]], k: int,
                             predicted_kth: float) -> list[int]:
    """ECHO-style one-pass, 256-bin subset of top-k for one query.

    Only whole bins above the threshold bin are returned. Their size cannot
    exceed k; the native indexer remains authoritative for guaranteed recall.
    """
    require(k > 0, "k must be positive")
    bins: dict[int, list[int]] = defaultdict(list)
    for native_id, score in scores:
        bins[_float_order_bin(score - predicted_kth)].append(native_id)
    size = 0
    accepted = set()
    for bin_id in range(255, -1, -1):
        count = len(bins[bin_id])
        if count and size + count > k:
            break
        if count:
            accepted.add(bin_id)
            size += count
    return [native_id for native_id, score in scores
            if _float_order_bin(score - predicted_kth) in accepted]


class ThresholdEMA:
    def __init__(self, alpha: float = 0.5):
        require(0.0 < alpha < 1.0, "EMA alpha must be between zero and one")
        self.alpha = alpha
        self.value: float | None = None

    def update(self, native_kth_score: float) -> None:
        require(math.isfinite(native_kth_score), "Non-finite native kth score")
        self.value = (native_kth_score if self.value is None else
                      self.alpha * self.value + (1.0 - self.alpha) * native_kth_score)


class ScorePool:
    """Per-layer bounded residency with score priority and guaranteed recall."""

    def __init__(self, capacity_units: int):
        require(capacity_units > 0, "Capacity must be positive")
        self.capacity_units = capacity_units
        self.resident: dict[int, tuple[float, int]] = {}
        self.clock = 0

    def _evict_for(self, protected: set[int]) -> None:
        if len(self.resident) < self.capacity_units:
            return
        victims = [(priority, age, native_id)
                   for native_id, (priority, age) in self.resident.items()
                   if native_id not in protected]
        require(victims, "Native selected set exceeds device cache capacity")
        self.resident.pop(min(victims)[2])

    def insert(self, native_id: int, priority: float, protected: set[int]) -> bool:
        if native_id in self.resident:
            return False
        self._evict_for(protected)
        self.clock += 1
        self.resident[native_id] = (priority, self.clock)
        return True

    def touch_selected(self, selected: set[int], scores: dict[int, float]) -> None:
        for native_id in selected:
            require(native_id in self.resident, "Guaranteed recall left native KV absent")
            self.clock += 1
            self.resident[native_id] = (scores[native_id], self.clock)


def validate_score_row(row: dict) -> tuple[list[tuple[int, float]], list[int]]:
    require(row.get("schema_version") == SCHEMA_VERSION, "Unsupported score schema")
    require(row.get("score_coverage") == "full", "Complete native scores are required")
    require(row.get("score_source_kind") in ("native_indexer_full", "synthetic_fixture"),
            "Score source is not identified")
    require(isinstance(row.get("score_source_sha256"), str) and
            re.fullmatch(r"[0-9a-f]{64}", row["score_source_sha256"]) is not None,
            "Score source SHA-256 is missing")
    require(row.get("native_topk_verified") is True,
            "Full score source was not checked against native top-k")
    require(row.get("phase") in PHASES, "Unknown phase")
    for key in ("run_id", "request_id", "layer"):
        require(isinstance(row.get(key), str) and bool(row[key]), f"Missing {key}")
    for key in ("rank", "sequence", "position", "k"):
        require(type(row.get(key)) is int and row[key] >= 0, f"Invalid {key}")
    require(type(row.get("context_len")) is int and row["k"] > 0 and
            row["position"] < row["context_len"],
            "Invalid k or query position")
    if row["phase"] == "prefill":
        require(type(row.get("prefill_chunk")) is int and row["prefill_chunk"] >= 0,
                "Prefill chunk is required")
        require(type(row.get("query_block")) is int and row["query_block"] >= 0,
                "Prefill query block is required")
    scores_raw = row.get("scores")
    selected = row.get("selected_ids")
    require(isinstance(scores_raw, list) and isinstance(selected, list),
            "Scores and selected IDs must be lists")
    scores: list[tuple[int, float]] = []
    seen: set[int] = set()
    for pair in scores_raw:
        require(isinstance(pair, list) and len(pair) == 2, "Invalid score pair")
        native_id, score = pair
        require(type(native_id) is int and native_id >= 0 and native_id not in seen,
                "Duplicate or invalid native score ID")
        require(type(score) in (int, float) and math.isfinite(score),
                "Non-finite native score")
        seen.add(native_id)
        scores.append((native_id, float(score)))
    require(scores and len(scores) >= min(row["k"], len(selected)) and
            [native_id for native_id, _ in scores] == sorted(seen),
            "Empty or incomplete score vector")
    require(all(type(native_id) is int and native_id in seen for native_id in selected) and
            len(selected) == len(set(selected)) and
            len(selected) == min(row["k"], len(scores)),
            "Native selected IDs are absent, duplicate, or wider than k")
    require(bool(selected), "Native selected set is empty")
    score_map = dict(scores)
    if len(scores) > len(selected):
        kth = min(score_map[native_id] for native_id in selected)
        nonselected = max(score for native_id, score in scores
                          if native_id not in set(selected))
        require(nonselected <= kth + 1e-4,
                "Full score vector disagrees with native top-k selection")
    return scores, selected


def _row_result(row: dict, scores: list[tuple[int, float]], selected: list[int],
                candidate_ids: list[int], pool: ScorePool, budget_units: int) -> dict:
    score_map = dict(scores)
    selected_set = set(selected)
    require(len(selected_set) <= pool.capacity_units,
            "Native selected set exceeds device cache capacity")
    prefetched: list[int] = []
    for native_id in candidate_ids:
        if len(prefetched) >= budget_units:
            break
        if native_id not in pool.resident:
            # Prediction cannot protect the future native selected set.
            pool.insert(native_id, score_map[native_id], set())
            prefetched.append(native_id)
    prefetched_resident = set(prefetched).intersection(pool.resident)
    synchronous: list[int] = []
    for native_id in selected:
        if native_id not in pool.resident:
            pool.insert(native_id, score_map[native_id], selected_set)
            synchronous.append(native_id)
    pool.touch_selected(selected_set, score_map)
    hit = selected_set.intersection(prefetched_resident)
    return {
        "run_id": row["run_id"], "request_id": row["request_id"],
        "rank": row["rank"], "layer": row["layer"], "phase": row["phase"],
        "sequence": row["sequence"], "position": row["position"],
        "prefill_chunk": row.get("prefill_chunk"), "query_block": row.get("query_block"),
        "selected_ids_sha256": hashlib.sha256(json.dumps(selected, separators=(",", ":")).encode()).hexdigest(),
        "predicted_candidates": len(candidate_ids),
        "prefetched_units": len(prefetched),
        "prefetch_hit_units": len(hit),
        "prefetch_wasted_current_query_units": len(prefetched) - len(hit),
        "synchronous_recall_units": len(synchronous),
        "native_selected_units": len(selected),
        "all_native_selected_ready": selected_set.issubset(pool.resident),
    }


def replay(rows: list[dict], *, capacity_units: int, prefetch_budget_units: int,
           alpha: float = 0.5) -> dict:
    require(capacity_units > 0 and prefetch_budget_units >= 0,
            "Invalid capacity or prefetch budget")
    lanes: dict[tuple[str, str, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        validate_score_row(row)
        lanes[(row["run_id"], row["request_id"], row["rank"], row["layer"])].append(row)
    require(lanes, "No score rows")
    source_kinds = {row["score_source_kind"] for row in rows}
    require(len(source_kinds) == 1, "Synthetic and native score sources cannot be mixed")
    source_kind = next(iter(source_kinds))
    results: list[dict] = []
    for lane_key, lane_rows in sorted(lanes.items()):
        lane_rows.sort(key=lambda row: row["sequence"])
        sequences = [row["sequence"] for row in lane_rows]
        require(sequences == sorted(set(sequences)), "Duplicate lane sequence")
        positions = [row["position"] for row in lane_rows]
        require(positions == sorted(set(positions)), "Lane positions are not strictly causal")
        states = {mode: ScorePool(capacity_units) for mode in ("no_prefetch", "echo_prefetch")}
        ema = {phase: ThresholdEMA(alpha) for phase in PHASES}
        blocks: dict[int, set[int]] = defaultdict(set)
        for row in lane_rows:
            if row["phase"] == "prefill":
                blocks[row["prefill_chunk"]].add(row["query_block"])
        overlap_blocks = {(chunk, block) for chunk, members in blocks.items()
                          for block in members if block < max(members)}
        previous_prefill_chunk: int | None = None
        previous_prefill_kth: float | None = None
        for row in lane_rows:
            scores, selected = validate_score_row(row)
            require(len(selected) <= capacity_units, "Selected set exceeds capacity")
            if row["phase"] == "prefill":
                chunk = row["prefill_chunk"]
                if previous_prefill_chunk is not None and chunk != previous_prefill_chunk:
                    require(chunk > previous_prefill_chunk, "Prefill chunks are out of order")
                    require(previous_prefill_kth is not None, "Missing previous prefill threshold")
                    ema["prefill"].update(previous_prefill_kth)
                previous_prefill_chunk = chunk
            state = ema[row["phase"]]
            predicted = state.value
            if predicted is None:
                candidates: list[int] = []
            elif row["phase"] == "decode":
                candidates = [native_id for native_id, score in scores if score > predicted]
            elif (row["prefill_chunk"], row["query_block"]) in overlap_blocks:
                candidates = approximate_prefill_topk(scores, row["k"], predicted)
            else:
                candidates = []
            baseline = _row_result(row, scores, selected, [], states["no_prefetch"], 0)
            echo = _row_result(row, scores, selected, candidates, states["echo_prefetch"],
                               prefetch_budget_units)
            kth = min(dict(scores)[native_id] for native_id in selected)
            if row["phase"] == "decode":
                state.update(kth)
            else:
                previous_prefill_kth = kth
            require(baseline["selected_ids_sha256"] == echo["selected_ids_sha256"] and
                    baseline["all_native_selected_ready"] and echo["all_native_selected_ready"],
                    "Native selection or guaranteed recall changed")
            results.append({
                "identity": list(lane_key), "phase": row["phase"],
                "sequence": row["sequence"], "position": row["position"],
                "prefill_chunk": row.get("prefill_chunk"),
                "query_block": row.get("query_block"),
                "predicted_kth_score": predicted, "native_kth_score": kth,
                "no_prefetch": baseline, "echo_prefetch": echo,
            })
    totals: dict[str, dict[str, int]] = {}
    for mode in ("no_prefetch", "echo_prefetch"):
        totals[mode] = {
            key: sum(item[mode][key] for item in results)
            for key in ("synchronous_recall_units", "prefetched_units",
                        "prefetch_hit_units", "prefetch_wasted_current_query_units")
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_level": ("CPU mechanism replay of declared native scores"
                           if source_kind == "native_indexer_full" else
                           "synthetic CPU mechanism test"),
        "score_source_kind": source_kind,
        "eligible_for_empirical_conclusion": False,
        "score_coverage": "full",
        "row_count": len(results),
        "lane_count": len(lanes),
        "capacity_units": capacity_units,
        "prefetch_budget_units": prefetch_budget_units,
        "ema_alpha": alpha,
        "totals": totals,
        "rows": results,
        "online_performance_validated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity-units", type=int, required=True)
    parser.add_argument("--prefetch-budget-units", type=int, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Output already exists")
    raw = args.scores.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    report = replay(rows, capacity_units=args.capacity_units,
                    prefetch_budget_units=args.prefetch_budget_units)
    report["input_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
