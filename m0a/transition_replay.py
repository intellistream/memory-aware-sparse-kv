#!/usr/bin/env python3
"""Lossless offline working-set replay; never changes serving or native selection."""
from __future__ import annotations

import argparse
from bisect import bisect_left
import hashlib
import json
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

try:
    from .contracts import EVENT_TYPES, load_jsonl, resolve_api_request_id, validate_trace
    from .model_profiles import load_profile
    from .working_set import (LRU, RankCaches, TransitionTable, candidates, identity, integer,
                              lane_key, require, support, support_units, text_id, validate_snapshot)
except ImportError:
    from contracts import EVENT_TYPES, load_jsonl, resolve_api_request_id, validate_trace
    from model_profiles import load_profile
    from working_set import (LRU, RankCaches, TransitionTable, candidates, identity, integer,
                             lane_key, require, support, support_units, text_id, validate_snapshot)

STRATEGIES = ("demand_only", "sequential_selected_set", "region_recency",
              "transition", "transition_sequential", "shuffled_events", "oracle_lookahead")
METRICS = ("precision", "recall", "prefetched_bytes", "useful_prefetch_bytes",
           "wasted_prefetch_bytes", "synchronous_recall_bytes", "cache_pollution_bytes",
           "prefetch_evicted_native_units", "planning_cpu_ns", "oracle_sync_lower_bound_bytes")


def validate_config(config):
    require(config.get("schema_version") == 1, "Unsupported cache config schema")
    require(integer(config.get("capacity_bytes"), 1), "Explicit positive capacity_bytes required")
    require(integer(config.get("prefetch_budget_bytes")), "Explicit nonnegative prefetch_budget_bytes required")
    sizes = config.get("kv_unit_bytes", {})
    require(isinstance(sizes, dict) and sizes and all(integer(v, 1) for v in sizes.values()), "Explicit KV unit byte sizes required")
    require(integer(config.get("max_tables", 256), 1), "Invalid max_tables")
    require(integer(config.get("bootstrap_draws", 1000), 1), "Invalid bootstrap_draws")
    require(integer(config.get("seed", 0)), "Invalid random seed")
    require(config.get("budget_scope", "aggregate") in {"aggregate", "per_rank"}, "Invalid budget scope")


def validate_sidecar(data, source_root):
    require(data.get("schema_version") == 1 and type(data.get("synthetic")) is bool, "Unsupported/missing sidecar schema or provenance")
    require(isinstance(data.get("snapshots"), list) and isinstance(data.get("events"), list) and data["events"], "Snapshots/events required")
    snapshots = {}
    versions = defaultdict(list)
    for snapshot in data["snapshots"]:
        validate_snapshot(snapshot, source_root)
        key = snapshot["snapshot_id"]
        require(key not in snapshots, "Duplicate snapshot_id")
        snapshots[key] = snapshot
        versions[identity(snapshot)].append(snapshot)
    for items in versions.values():
        items.sort(key=lambda s: s["sequence"])
        for a, b in zip(items, items[1:]):
            shared = min(len(a["token_ids"]), len(b["token_ids"]))
            require(a["token_ids"][:shared] == b["token_ids"][:shared], "Token rewrite must change context_version")
    events = data["events"]
    seen, requests, trajectory = set(), set(), {}
    # Transitive grouping closes session/episode/pair aliases. Caller labels
    # cannot split paired branches or repetitions into held-out evaluation.
    parent = list(range(len(events)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owners = {}
    for i, event in enumerate(events):
        fields = ("event_id", "request_id", "run_id", "previous_snapshot_id", "snapshot_id",
                  "event_type", "previous_state", "event_position", "resume_position",
                  "workload_id", "episode_id", "source_trace_id", "pair_id", "variant", "repetition", "trajectory_id", "split")
        require(all(k in event for k in fields), "Incomplete event metadata")
        require(all(text_id(event[k]) for k in fields if k not in {"event_position", "resume_position", "repetition"}), "Invalid event identity")
        require(event["split"] in {"train", "eval"} and event["variant"] in {"event", "control"}, "Invalid split/variant")
        require(event["event_type"] in EVENT_TYPES | {"no_event"}, "Unknown event type")
        require(integer(event["repetition"]) and integer(event["event_position"])
                and integer(event["resume_position"], 1), "Invalid event position/repetition")
        require(event["event_id"] not in seen, "Duplicate event_id")
        seen.add(event["event_id"])
        request = (event["run_id"], event["request_id"])
        require(request not in requests, "One restore event per traced request required")
        requests.add(request)
        require(event["snapshot_id"] in snapshots and event["previous_snapshot_id"] in snapshots, "Unknown snapshot reference")
        old, cur = snapshots[event["previous_snapshot_id"]], snapshots[event["snapshot_id"]]
        require(old["session_id"] == cur["session_id"] and old["sequence"] < cur["sequence"], "Invalid session/sequence predecessor")
        resume = event["resume_position"]
        require(event["event_position"] <= resume < len(cur["token_ids"])
                and cur["generated_tokens"] <= resume, "Invalid suffix prefill boundary")
        if "post_window_snapshot_id" in event:
            post = snapshots.get(event["post_window_snapshot_id"])
            require(post is not None and identity(post) == identity(cur)
                    and post["sequence"] == cur["sequence"] and post["token_ids"] == cur["token_ids"]
                    and post["generated_tokens"] == min(resume+32, len(cur["token_ids"])), "Invalid post-window snapshot")
        stream = event["trajectory_id"]
        if stream in trajectory:
            prior = trajectory[stream]
            require(prior.get("post_window_snapshot_id", prior["snapshot_id"]) == event["previous_snapshot_id"] and prior["split"] == event["split"], "Trajectory predecessor/order mismatch")
        trajectory[stream] = event
        for group in (("session", cur["session_id"]),
                      ("source", event["source_trace_id"]),
                      ("episode", event["workload_id"], event["episode_id"]),
                      ("pair", event["workload_id"], event["pair_id"])):
            if group in owners:
                parent[find(i)] = find(owners[group])
            else:
                owners[group] = i
    splits = defaultdict(set)
    for i, event in enumerate(events):
        splits[find(i)].add(event["split"])
    require(all(len(v) == 1 for v in splits.values()), "Training/evaluation leakage across session, episode, pair or repeat")
    groups = {event["event_id"]: str(find(i)) for i, event in enumerate(events)}
    return snapshots, groups


def read_trace(trace_dir, snapshots, events, *, synthetic=False):
    requests = {(e["run_id"], e["request_id"]): e for e in events}
    by_run = defaultdict(set)
    for run_id, req_id in requests:
        by_run[run_id].add(req_id)
    rows = defaultdict(dict)
    files = sorted(Path(trace_dir).glob("rank*.jsonl"))
    require(files, "No native rank trace files")
    for path in files:
        require(path.stem[4:].isdecimal(), "Invalid rank trace filename")
        for row in load_jsonl(path):
            req_id = resolve_api_request_id(row["request_id"], by_run.get(row["run_id"], set()))
            require(req_id is not None, "Unlinked native trace request/run")
            event = requests[(row["run_id"], req_id)]
            snapshot = snapshots[event["snapshot_id"]]
            profile = load_profile("glm53_tiny" if row["raw_index_unit"] == "kv_token_position" else "deepseek_v4")
            require(snapshot["model_id"] == profile["model_id"] and snapshot["model_revision"] == profile["model_revision"], "Snapshot/trace model identity mismatch")
            validate_trace(row, contract=profile)
            require(row["rank"] == int(path.stem[4:]), "Rank filename mismatch")
            require(row["request_context_len"] == len(snapshot["token_ids"]), "Exact token sequence/trace context mismatch")
            key = (row["rank"], row["layer"], row["raw_index_unit"])
            lane = next((l for l in snapshot["lanes"] if lane_key(l) == key), None)
            require(lane is not None, "Native trace has unknown KV lane")
            position = row["prompt_position"]
            for unit in set(row["raw_selected_ids"]) - {-1}:
                require(support(snapshot, lane, unit)[2] <= position, "Native selected KV not yet generated/causal")
                if support(snapshot, lane, unit)[2] < snapshot["generated_tokens"]:
                    require(unit in lane["generated_ids"], "Sidecar omits generated KV selected by native trace")
            row_key = key + (position,)
            require(row_key not in rows[event["event_id"]], "Duplicate rank/layer/token trace row")
            rows[event["event_id"]][row_key] = row
    for event in events:
        cur = snapshots[event["snapshot_id"]]
        profile = load_profile("glm53_tiny" if cur["lanes"][0]["kv_kind"] == "kv_token_position" else "deepseek_v4")
        if not synthetic:
            require(len({l["layer"] for l in cur["lanes"]}) == profile["expected_layers"], "Incomplete model layer coverage")
        start = event["resume_position"]
        stop = min(start + 32, len(cur["token_ids"]))
        if "query_assignment" in event:
            expected_layers = {l["layer"] for l in cur["lanes"]}
            require(set(event["query_assignment"]) == {str(p) for p in range(start-1, stop)}, "Incomplete CP query assignment")
            for position in range(start-1, stop):
                assigned = event["query_assignment"][str(position)]
                require(isinstance(assigned, dict) and set(assigned) == expected_layers, "Incomplete CP layer assignment")
                for layer, rank in assigned.items():
                    require(integer(rank) and (rank, layer, profile["raw_index_unit"], position) in rows[event["event_id"]], "Missing CP owner row")
            expected = {(rank, layer, profile["raw_index_unit"], int(pos))
                        for pos, mapping in event["query_assignment"].items() for layer, rank in mapping.items()}
            require(set(rows[event["event_id"]]) == expected, "Duplicate/unassigned CP trace row")
        else:
            for lane in cur["lanes"]:
                for position in range(start - 1, stop):
                    require(lane_key(lane) + (position,) in rows[event["event_id"]], "Incomplete native selected set for first 32 suffix prefill tokens (including warmup)")
    return rows


def interleave(lists):
    for index in range(max((len(v) for v in lists), default=0)):
        for values in lists:
            if index < len(values):
                yield values[index]


def ordered_candidates(strategy, event, snapshot, lanes, valid, warm, table, shuffled_table, shuffled_label, frames):
    rankings, sources = [], {}
    for lane in lanes:
        key = lane_key(lane)
        units = valid[key]
        previous = [u for u in dict.fromkeys(warm[key]["raw_selected_ids"]) if u in units]
        neighbors = sorted(previous)
        def distance(unit):
            pos = bisect_left(neighbors, unit)
            nearby = neighbors[max(0, pos-1):pos+1]
            return min((abs(unit-p) for p in nearby), default=0)
        sequential = previous + sorted(units - set(previous), key=lambda u: (distance(u), -u))
        if strategy == "demand_only":
            ranking, source = [], "none"
        elif strategy == "sequential_selected_set":
            ranking, source = sequential, "previous_selected_set"
        elif strategy == "oracle_lookahead":
            # Explicitly isolated privileged baseline; no other planner sees future IDs.
            ranking = list(dict.fromkeys(u for row in frames[key] for u in row["raw_selected_ids"] if u in units))
            source = "future_native_trace"
        else:
            event_label, state, active = event["event_type"], event["previous_state"], table
            if strategy == "region_recency":
                event_label, state = "__all_events__", "__all_states__"
            if strategy == "shuffled_events":
                event_label, active = shuffled_label, shuffled_table
            ranking, source = active.rank(snapshot, lane, units, state, event_label)
            if strategy == "transition_sequential":
                ranking = list(dict.fromkeys(interleave([ranking, sequential])))
        rankings.append([(key, unit) for unit in ranking])
        sources[repr(key)] = source
    return list(interleave(rankings)), sources


def unit_key(snapshot, lane, unit, granularity):
    width = 1 if granularity == "native" else 128
    return identity(snapshot) + lane_key(lane) + (unit // width,)


def replay_event(strategy, granularity, event, previous, snapshot, rows, cache, config,
                 table, shuffled_table, shuffled_label):
    lanes = snapshot["lanes"]
    if "query_assignment" in event:
        active = {key[:3] for key in rows}
        lanes = [lane for lane in lanes if lane_key(lane) in active]
    lookup = {lane_key(l): l for l in lanes}
    valid = {lane_key(l): candidates(previous, snapshot, l) for l in lanes}
    cache.migrate(previous, snapshot, valid)
    start, stop = event["resume_position"], min(event["resume_position"] + 32, len(snapshot["token_ids"]))
    frames = {lane_key(l): [rows[lane_key(l) + (pos,)] for pos in range(start, stop)
                           if lane_key(l) + (pos,) in rows] for l in lanes}
    # CP planning shares the observed prefix selection for a layer. This does
    # not create native rows or pretend another rank executed that query.
    warm_by_layer = {r["layer"]: r for r in rows.values() if r["prompt_position"] == start-1}
    warm = {lane_key(l): rows.get(lane_key(l)+(start-1,), warm_by_layer[l["layer"]]) for l in lanes}
    planning_lanes = [l for l in lanes if frames[lane_key(l)]]
    width = 1 if granularity == "native" else 128
    def size(lane):
        require(lane["kv_kind"] in config["kv_unit_bytes"], "Missing explicit byte size for KV kind")
        return width * config["kv_unit_bytes"][lane["kv_kind"]]
    # All future attentions are validated before mutating this cache.
    for lane in lanes:
        for row in frames[lane_key(lane)] + ([rows[lane_key(lane)+(start-1,)]] if lane_key(lane)+(start-1,) in rows else []):
            pages = {unit_key(snapshot, lane, u, granularity) for u in row["raw_selected_ids"] if u >= 0}
            require(len(pages) * size(lane) <= cache.capacity, "Capacity cannot hold complete single-attention native selected set")
    offloaded = {lane_key(l): set(l["generated_ids"]) for l in lanes}
    generated = {lane_key(l): set(l["generated_ids"]) for l in lanes}
    # Discover first availability of locally generated units without consulting selection.
    generation = defaultdict(list)
    for lane in lanes:
        key = lane_key(lane)
        if lane["kv_kind"] == "kv_token_position":
            for u in range(snapshot["generated_tokens"], stop):
                generation[u].append((key, u))
        else:
            for u in support_units(snapshot, lane):
                ready = support(snapshot, lane, u)[2]
                if snapshot["generated_tokens"] <= ready < stop:
                    generation[ready].append((key, u))
    useful, transfers, current_prefetch = set(), 0, set()
    rank_transfers, rank_prefetched, rank_peaks = defaultdict(int), defaultdict(int), defaultdict(int)
    def observe_cache():
        usage = defaultdict(int)
        if isinstance(cache, RankCaches):
            usage.update({r: c.used for r, c in cache.caches.items()})
        else:
            for page, entry in cache.entries.items():
                usage[page[5]] += entry.size
        for rank, used in usage.items():
            rank_peaks[rank] = max(rank_peaks[rank], used)
        require((all(v <= config["capacity_bytes"] for v in usage.values()) if isinstance(cache, RankCaches)
                 else sum(usage.values()) <= config["capacity_bytes"]), "Cache capacity exceeded")
    def attention(lane, row, measure):
        nonlocal transfers
        key = lane_key(lane)
        demand = set(row["raw_selected_ids"]) - {-1}
        protected = {unit_key(snapshot, lane, u, granularity) for u in demand}
        # Every newly available KV is written locally, whether selected or not.
        # The host store is assumed to retain generated KV; write-back DMA is
        # outside this byte model. Recall after a cache eviction is charged.
        newly_generated = {u for pos, values in generation.items() if pos <= row["prompt_position"]
                           for lk, u in values if lk == key and u not in generated[key]}
        generated[key].update(newly_generated)
        for u in sorted(newly_generated):
            cache.put(unit_key(snapshot, lane, u, granularity), size(lane), {u})
        for u in sorted(demand):
            page = unit_key(snapshot, lane, u, granularity)
            was_ready = cache.contains(page, u)
            was_prefetched = was_ready and u in cache.entries[page].prefetched
            if not was_ready:
                require(u in generated[key], "Demand KV is not available in host or generation stream")
                first = (u // width) * width
                available = {v for v in range(first, first+width) if v in generated[key]}
                cache.put(page, size(lane), available, protected=protected)
                if measure:
                    transfers += size(lane)
                    rank_transfers[lane["rank"]] += size(lane)
            if measure and was_prefetched and (page, u) in current_prefetch:
                useful.add((key, u))
            cache.touch(page)
        # Losslessness invariant at the exact simulated native attention boundary.
        require(all(cache.contains(unit_key(snapshot, lane, u, granularity), u) for u in demand), "Native attention KV not ready")
        observe_cache()
    for lane in lanes:
        if lane_key(lane)+(start-1,) in rows:
            attention(lane, rows[lane_key(lane)+(start-1,)], False)
    initial = {(lane_key(l), u) for l in lanes for u in offloaded[lane_key(l)]
               if cache.contains(unit_key(snapshot, l, u, granularity), u)}
    clock = time.process_time_ns()
    ordered, sources = ordered_candidates(strategy, event, snapshot, planning_lanes, valid, warm, table,
                                          shuffled_table, shuffled_label, frames)
    groups = defaultdict(set)
    for lane in lanes:
        for u in valid[lane_key(lane)]:
            groups[unit_key(snapshot, lane, u, granularity)].add(u)
    predicted, prefetched, evictions_before = set(), 0, cache.prefetch_evictions
    visited = set()
    for lk, u in ordered:
        lane = lookup[lk]
        page = unit_key(snapshot, lane, u, granularity)
        if page in visited:
            continue
        visited.add(page)
        members = groups[page]
        missing = {v for v in members if not cache.contains(page, v)}
        spent = rank_prefetched[lane["rank"]] if config.get("budget_scope") == "per_rank" else prefetched
        if not missing or spent + size(lane) > config["prefetch_budget_bytes"]:
            continue
        cache.put(page, size(lane), members, prefetch=True)
        prefetched += size(lane)
        rank_prefetched[lane["rank"]] += size(lane)
        predicted.update((lk, v) for v in missing)
        current_prefetch.update((page, v) for v in missing)
    planning = time.process_time_ns() - clock
    # Labels are derived after planning; updates happen after every strategy has predicted.
    target = {(lk, u) for lk, frame in frames.items() for row in frame for u in row["raw_selected_ids"] if u in valid[lk]}
    old_demand = {(lk, u) for lk, frame in frames.items() for row in frame for u in row["raw_selected_ids"] if u in offloaded[lk]}
    # An optimistic bound: compulsory initial misses minus the same prefetch budget.
    # It relaxes eviction pressure; the feasible lookahead replay remains separate.
    compulsory_pages = {unit_key(snapshot, lookup[lk], u, granularity)
                        for lk, u in old_demand - initial}
    compulsory_bytes = sum(width * config["kv_unit_bytes"][page[7]] for page in compulsory_pages)
    if config.get("budget_scope") == "per_rank":
        compulsory_by_rank = defaultdict(int)
        for page in compulsory_pages:
            compulsory_by_rank[page[5]] += width * config["kv_unit_bytes"][page[7]]
        lower_bound = sum(max(0, v-config["prefetch_budget_bytes"]) for v in compulsory_by_rank.values())
    else:
        lower_bound = max(0, compulsory_bytes - config["prefetch_budget_bytes"])
    for pos in range(start, stop):
        for lane in lanes:
            if lane_key(lane)+(pos,) in rows:
                attention(lane, rows[lane_key(lane) + (pos,)], True)
    useful_bytes = sum(config["kv_unit_bytes"][lk[2]] for lk, u in useful)
    require(useful_bytes <= prefetched and (all(v <= config["prefetch_budget_bytes"] for v in rank_prefetched.values())
            if config.get("budget_scope") == "per_rank" else prefetched <= config["prefetch_budget_bytes"]), "Prefetch accounting/budget violation")
    return {
        "event_id": event["event_id"], "split": event["split"], "workload_id": event["workload_id"],
        "episode_id": event["episode_id"], "pair_id": event["pair_id"], "variant": event["variant"],
        "repetition": event["repetition"], "event_type": event["event_type"], "strategy": strategy,
        "granularity": granularity, "suffix_tokens": stop-start,
        "candidate_native_units": sum(map(len, valid.values())), "predicted_native_units": len(predicted),
        "prediction_sha256": hashlib.sha256(json.dumps(sorted(predicted)).encode()).hexdigest(),
        "target_native_units": len(target), "true_positive_native_units": len(predicted & target),
        "precision": len(predicted & target)/len(predicted) if predicted else 0.0,
        "recall": len(predicted & target)/len(target) if target else 0.0,
        "prefetched_bytes": prefetched, "useful_prefetch_bytes": useful_bytes,
        "wasted_prefetch_bytes": prefetched-useful_bytes, "synchronous_recall_bytes": transfers,
        "prefetch_evicted_native_units": cache.prefetch_evictions-evictions_before,
        "planning_cpu_ns": planning, "table_sources": sources,
        "oracle_sync_lower_bound_bytes": lower_bound, "attention_readiness_assertions": sum(map(len, frames.values())),
        "budget_scope": config.get("budget_scope", "aggregate"),
        "per_rank": {str(r): {"prefetched_bytes": rank_prefetched[r], "synchronous_recall_bytes": rank_transfers[r],
                    "peak_cache_bytes": rank_peaks[r],
                    "useful_prefetch_bytes": sum(config["kv_unit_bytes"][lk[2]] for lk, u in useful if lk[0] == r),
                    "target_native_units": sum(lk[0] == r for lk, u in target),
                    "predicted_native_units": sum(lk[0] == r for lk, u in predicted),
                    "true_positive_native_units": sum(lk[0] == r for lk, u in predicted & target)}
                    for r in sorted({l["rank"] for l in lanes})},
    }


def percentile(values, p):
    values = sorted(values)
    n = (len(values)-1)*p/100
    lo, hi = math.floor(n), math.ceil(n)
    return values[lo]+(values[hi]-values[lo])*(n-lo)


def describe(values):
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    return {"n": len(values), "mean": statistics.mean(values),
            **{f"p{p}": percentile(values, p) for p in (50, 95, 99)}}


def interval(values, config):
    if len(values) < 2:
        return None
    rng = random.Random(config.get("seed", 0))
    estimates = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(config.get("bootstrap_draws", 1000))]
    return [percentile(estimates, 2.5), percentile(estimates, 97.5)]


def summarize(results, groups, config):
    evaluation = [r for r in results if r["split"] == "eval"]
    summary = {}
    for granularity in ("native", "page128"):
        summary[granularity] = {}
        for strategy in STRATEGIES:
            selected = [r for r in evaluation if r["granularity"] == granularity and r["strategy"] == strategy]
            metrics = {}
            for metric in METRICS:
                independent = defaultdict(list)
                for r in selected:
                    independent[groups[r["event_id"]]].append(r[metric])
                values = [statistics.mean(v) for v in independent.values()]
                metrics[metric] = {"event_distribution": describe([r[metric] for r in selected]),
                                   "independent_group_distribution": describe(values), "group_mean_95pct": interval(values, config)}
            total_predicted = sum(r["predicted_native_units"] for r in selected)
            total_target = sum(r["target_native_units"] for r in selected)
            true_positive = sum(r["true_positive_native_units"] for r in selected)
            metrics["micro_precision"] = true_positive/total_predicted if total_predicted else 0.0
            metrics["micro_recall"] = true_positive/total_target if total_target else 0.0
            summary[granularity][strategy] = metrics
        pairs = defaultdict(dict)
        for r in evaluation:
            if r["granularity"] == granularity:
                pairs[(r["strategy"], r["workload_id"], r["pair_id"], r["repetition"])][r["variant"]] = r
        paired = {}
        for strategy in STRATEGIES:
            by_group = defaultdict(list)
            complete = 0
            for key, variants in pairs.items():
                if key[0] == strategy and set(variants) == {"event", "control"}:
                    complete += 1
                    e, c = variants["event"], variants["control"]
                    by_group[groups[e["event_id"]]].append(e["synchronous_recall_bytes"]-c["synchronous_recall_bytes"])
            values = [statistics.mean(v) for v in by_group.values()]
            paired[strategy] = {"complete_pairs": complete, "event_minus_control_sync_bytes": describe(values),
                                "group_bootstrap_95pct": interval(values, config)}
        summary[granularity]["paired_differences"] = paired
        comparison = {}
        baseline = {r["event_id"]: r for r in evaluation if r["granularity"] == granularity and r["strategy"] == "sequential_selected_set"}
        for strategy in STRATEGIES:
            by_group = defaultdict(list)
            for r in evaluation:
                if r["granularity"] == granularity and r["strategy"] == strategy:
                    by_group[groups[r["event_id"]]].append(r["synchronous_recall_bytes"]-baseline[r["event_id"]]["synchronous_recall_bytes"])
            values = [statistics.mean(v) for v in by_group.values()]
            comparison[strategy] = {"strategy_minus_sequential_sync_bytes": describe(values), "group_bootstrap_95pct": interval(values, config)}
        summary[granularity]["against_sequential"] = comparison
        summary[granularity]["oracle_upper_bound"] = {
            "definition": "Optimistic synchronous-byte lower bound under the same capacity and prefetch budget; eviction constraints relaxed. Not a feasible schedule.",
            "sync_bytes": describe([r["oracle_sync_lower_bound_bytes"] for r in evaluation if r["granularity"] == granularity and r["strategy"] == "demand_only"]),
        }
    return summary


def run_replay(trace_dir, sidecar, config, source_root):
    validate_config(config)
    snapshots, groups = validate_sidecar(sidecar, source_root)
    rows = read_trace(trace_dir, snapshots, sidecar["events"],
                      synthetic=sidecar["synthetic"] and sidecar.get("selection_provenance") != "real_npu_native")
    table = TransitionTable(config.get("max_tables", 256))
    shuffled_table = TransitionTable(config.get("max_tables", 256))
    shuffled = {}
    rng = random.Random(config.get("seed", 0))
    for split in ("train", "eval"):
        events = [e for e in sidecar["events"] if e["split"] == split]
        labels = [e["event_type"] for e in events]
        rng.shuffle(labels)
        shuffled.update((e["event_id"], label) for e, label in zip(events, labels))
    caches, results = {}, []
    for split in ("train", "eval"):
        for event in (e for e in sidecar["events"] if e["split"] == split):
            current, previous = snapshots[event["snapshot_id"]], snapshots[event["previous_snapshot_id"]]
            event_results = []
            for granularity in ("native", "page128"):
                for strategy in STRATEGIES:
                    key = (split, granularity, strategy, event["trajectory_id"])
                    if key not in caches:
                        caches[key] = (RankCaches if config.get("budget_scope") == "per_rank" else LRU)(config["capacity_bytes"])
                    event_results.append(replay_event(strategy, granularity, event, previous, current,
                        rows[event["event_id"]], caches[key], config, table, shuffled_table, shuffled[event["event_id"]]))
            for granularity in ("native", "page128"):
                baseline = next(r["synchronous_recall_bytes"] for r in event_results if r["strategy"] == "demand_only" and r["granularity"] == granularity)
                for r in event_results:
                    if r["granularity"] == granularity:
                        r["cache_pollution_bytes"] = max(0, r["synchronous_recall_bytes"]-baseline)
                        demand = next(v for v in event_results if v["strategy"] == "demand_only" and v["granularity"] == granularity)
                        for rank, metrics in r["per_rank"].items():
                            metrics["cache_pollution_bytes"] = max(0, metrics["synchronous_recall_bytes"]-demand["per_rank"][rank]["synchronous_recall_bytes"])
                            metrics["wasted_prefetch_bytes"] = metrics["prefetched_bytes"]-metrics["useful_prefetch_bytes"]
                            metrics["precision"] = metrics["true_positive_native_units"]/metrics["predicted_native_units"] if metrics["predicted_native_units"] else 0.
                            metrics["recall"] = metrics["true_positive_native_units"]/metrics["target_native_units"] if metrics["target_native_units"] else 0.
            results.extend(event_results)
            if split == "train":
                start = event["resume_position"]
                stop = min(start+32, len(current["token_ids"]))
                for lane in current["lanes"]:
                    lk = lane_key(lane)
                    if not any(lk+(p,) in rows[event["event_id"]] for p in range(start, stop)):
                        continue
                    eligible = candidates(previous, current, lane)
                    target = set().union(*(set(rows[event["event_id"]][lk+(p,)]["raw_selected_ids"])-{-1}
                                          for p in range(start, stop) if lk+(p,) in rows[event["event_id"]]))
                    table.update(current, lane, eligible, target, event["previous_state"], event["event_type"], observation_id=event["event_id"])
                    table.update(current, lane, eligible, target, "__all_states__", "__all_events__", observation_id=event["event_id"])
                    shuffled_table.update(current, lane, eligible, target, event["previous_state"], shuffled[event["event_id"]], observation_id=event["event_id"])
            # Caches have a lifetime per trajectory, not per corpus: release ended streams.
            remaining = [e for e in sidecar["events"] if e["split"] == split and e["trajectory_id"] == event["trajectory_id"]]
            if event is remaining[-1]:
                for key in list(caches):
                    if key[0] == split and key[3] == event["trajectory_id"]:
                        del caches[key]
    return {"schema_version": 1, "synthetic": sidecar["synthetic"], "config": config,
            "training_events": sum(e["split"] == "train" for e in sidecar["events"]),
            "evaluation_events": sum(e["split"] == "eval" for e in sidecar["events"]),
            "lane_counts": sorted({len(s["lanes"]) for s in snapshots.values()}),
            "training_table_contexts": len(table.tables), "training_table_cells": sum(len(t["cells"]) for t in table.tables.values()),
            "evaluation_table_updates": 0, "results": results, "summary": summarize(results, groups, config),
            "limitations": ["Offline byte model; no NPU, real DMA, output-equivalence or stall measurement.",
                            "Native trace timestamps are never interpreted as recall stall.",
                            "Sequential selected-set baseline is not ECHO.",
                            "Capacity is per trajectory; concurrent serving is not modeled.",
                            "Host store retains generated KV; write-back DMA and local compute cost are excluded.",
                            "Source citations verify integrity, not correctness of exporter support intervals.",
                            "Prefetch runs once before each 32-token window; partial final windows report actual length."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "Output directory already exists; use a fresh run ID")
    sidecar, config = json.loads(args.sidecar.read_text()), json.loads(args.config.read_text())
    report = run_replay(args.trace_dir, sidecar, config, args.source_root)
    report["input_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [args.sidecar, args.config, *sorted(args.trace_dir.glob("rank*.jsonl"))]}
    report["implementation_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in (Path(__file__), Path(__file__).with_name("working_set.py"))}
    args.output_dir.mkdir(parents=True)
    (args.output_dir/"events.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in report.pop("results")))
    (args.output_dir/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"output_dir": str(args.output_dir), "training_events": report["training_events"],
                      "evaluation_events": report["evaluation_events"]}))


if __name__ == "__main__":
    main()
