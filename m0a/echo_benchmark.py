#!/usr/bin/env python3
"""Analyze fixed-continuation online ECHO measurements when NPU data exist.

Every input row is one timed request. This analyzer never fabricates timings
from CPU replay or accepts a Python reference pool as a timed backend.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .echo_runtime import Capabilities, Mode


METRICS = ("ttft_ms", "tpot_ms", "throughput_tps", "synchronous_recall_bytes",
           "host_to_device_bytes")
CLASSES = ("agent_tool", "ordinary")
MIN_EPISODES_FOR_CLAIM = 20


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_row(row: dict) -> None:
    for key in ("episode_id", "workload_id", "prompt_token_ids_sha256",
                "forced_continuation_sha256", "native_selected_set_sha256"):
        require(isinstance(row.get(key), str) and bool(row[key]), f"Missing {key}")
    require(row.get("workload_class") in CLASSES, "Unknown workload class")
    require(row.get("mode") in {mode.value for mode in Mode}, "Unknown serving mode")
    for key in ("repeat", "concurrency", "capacity_bytes", "decode_tokens"):
        require(type(row.get(key)) is int and row[key] >= 0, f"Invalid {key}")
    require(row["concurrency"] > 0 and row["capacity_bytes"] > 0 and
            row["decode_tokens"] > 0, "Invalid load or decode length")
    require(row.get("fixed_continuation_verified") is True,
            "Timed comparison requires a verified fixed continuation")
    require(row.get("native_selected_set_exact") is True,
            "Native selected set equivalence is unverified")
    caps = row.get("capabilities")
    require(isinstance(caps, dict), "Missing backend capability evidence")
    known = set(Capabilities.__dataclass_fields__)
    require(set(caps) == known and all(type(v) is bool for v in caps.values()),
            "Incomplete backend capability evidence")
    Capabilities(**caps).validate(Mode(row["mode"]), timed=True)
    for key in METRICS:
        require(type(row.get(key)) in (int, float) and row[key] >= 0,
                f"Invalid timed metric {key}")


def _ci(values: list[float], seed: int, draws: int) -> list[float] | None:
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    samples = sorted(mean(rng.choices(values, k=len(values))) for _ in range(draws))
    return [samples[int(0.025 * (draws - 1))],
            samples[int(0.975 * (draws - 1))]]


def _difference_ci(left: list[float], right: list[float], seed: int,
                   draws: int) -> list[float] | None:
    if len(left) < 2 or len(right) < 2:
        return None
    rng = random.Random(seed)
    samples = sorted(
        mean(rng.choices(left, k=len(left))) - mean(rng.choices(right, k=len(right)))
        for _ in range(draws)
    )
    return [samples[int(0.025 * (draws - 1))],
            samples[int(0.975 * (draws - 1))]]


def analyze(rows: list[dict], *, draws: int = 2000, seed: int = 7) -> dict:
    require(draws >= 100, "Bootstrap needs at least 100 draws")
    keyed: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        validate_row(row)
        key = (row["workload_class"], row["workload_id"], row["episode_id"],
               row["concurrency"], row["capacity_bytes"], row["repeat"])
        require(row["mode"] not in keyed[key], "Duplicate timed mode for a repeat")
        keyed[key][row["mode"]] = row
    require(keyed, "No timed measurements")
    per_episode: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    no_offload_episode: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    selected_drift: dict[tuple, int] = defaultdict(int)
    for key, arms in keyed.items():
        require(Mode.OFFLOAD_NO_PREFETCH.value in arms and
                Mode.OFFLOAD_ECHO_PREFETCH.value in arms,
                "Both offload arms are required for every repeat")
        reference = arms[Mode.OFFLOAD_NO_PREFETCH.value]
        for arm in arms.values():
            require(arm["prompt_token_ids_sha256"] == reference["prompt_token_ids_sha256"] and
                    arm["forced_continuation_sha256"] == reference["forced_continuation_sha256"] and
                    arm["decode_tokens"] == reference["decode_tokens"],
                    "Timed arms used different input or continuation tokens")
        if len({arm["native_selected_set_sha256"] for arm in arms.values()}) > 1:
            selected_drift[key[:5]] += 1
        class_name, workload, episode, concurrency, capacity, _ = key
        group = (class_name, workload, episode, concurrency, capacity)
        echo = arms[Mode.OFFLOAD_ECHO_PREFETCH.value]
        for metric in METRICS:
            per_episode[group][metric].append(echo[metric] - reference[metric])
            if Mode.NO_OFFLOAD.value in arms:
                no_offload_episode[group][metric].append(
                    echo[metric] - arms[Mode.NO_OFFLOAD.value][metric])
    grouped: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (class_name, workload, episode, concurrency, capacity), deltas in per_episode.items():
        for metric, values in deltas.items():
            grouped[(class_name, workload, concurrency, capacity)][metric].append(mean(values))
    by_load: dict[tuple, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    for (class_name, workload, episode, concurrency, capacity), deltas in per_episode.items():
        for metric, values in deltas.items():
            by_load[(concurrency, capacity)][class_name][metric].append(mean(values))
    no_offload_grouped: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    for (class_name, workload, episode, concurrency, capacity), deltas in no_offload_episode.items():
        for metric, values in deltas.items():
            no_offload_grouped[(class_name, workload, concurrency, capacity)][metric].append(
                mean(values))
    cells = []
    for (class_name, workload, concurrency, capacity), metrics in sorted(grouped.items()):
        summary = {}
        for index, metric in enumerate(METRICS):
            values = metrics[metric]
            summary[metric] = {
                "mean_echo_minus_no_prefetch": mean(values),
                "episode_means": values,
                "ci95": _ci(values, seed + index, draws),
            }
        no_offload = no_offload_grouped.get((class_name, workload, concurrency, capacity))
        against_no_offload = ({
            metric: {
                "mean_echo_minus_no_offload": mean(no_offload[metric]),
                "ci95": _ci(no_offload[metric], seed + 300 + index, draws),
            }
            for index, metric in enumerate(METRICS)
        } if no_offload else None)
        cells.append({
            "workload_class": class_name, "workload_id": workload,
            "concurrency": concurrency, "capacity_bytes": capacity,
            "independent_episodes": len(metrics[METRICS[0]]),
            "selected_set_drift_repeats": sum(
                count for (cls, work, _episode, conc, cap), count in selected_drift.items()
                if (cls, work, conc, cap) == (class_name, workload, concurrency, capacity)),
            "deltas": summary,
            "against_no_offload": against_no_offload,
        })
    interactions = []
    for (concurrency, capacity), classes in sorted(by_load.items()):
        if set(CLASSES) != set(classes):
            continue
        effects = {}
        for index, metric in enumerate(METRICS):
            agent = classes["agent_tool"][metric]
            ordinary = classes["ordinary"][metric]
            effects[metric] = {
                "agent_mean_delta": mean(agent),
                "ordinary_mean_delta": mean(ordinary),
                "agent_ci95": _ci(agent, seed + 100 + index, draws),
                "agent_minus_ordinary_ci95": _difference_ci(
                    agent, ordinary, seed + 200 + index, draws),
            }
        latency = effects["ttft_ms"]
        throughput = effects["throughput_tps"]
        intervals = (latency["agent_ci95"], latency["agent_minus_ordinary_ci95"],
                     throughput["agent_ci95"], throughput["agent_minus_ordinary_ci95"])
        agent_cells = [cell for cell in cells
                       if cell["workload_class"] == "agent_tool" and
                       cell["concurrency"] == concurrency and
                       cell["capacity_bytes"] == capacity]
        domain_consistent = (
            len(agent_cells) >= 2 and all(
                cell["independent_episodes"] >= MIN_EPISODES_FOR_CLAIM and
                cell["deltas"]["ttft_ms"]["ci95"] is not None and
                cell["deltas"]["ttft_ms"]["ci95"][0] > 0 and
                cell["deltas"]["throughput_tps"]["ci95"] is not None and
                cell["deltas"]["throughput_tps"]["ci95"][1] < 0
                for cell in agent_cells))
        supported = (all(interval is not None for interval in intervals) and
                     domain_consistent and
                     len(classes["agent_tool"]["ttft_ms"]) >= MIN_EPISODES_FOR_CLAIM and
                     len(classes["ordinary"]["ttft_ms"]) >= MIN_EPISODES_FOR_CLAIM and
                     intervals[0][0] > 0 and intervals[1][0] > 0 and
                     intervals[2][1] < 0 and intervals[3][1] < 0 and
                     not any(count for (_cls, _work, _episode, conc, cap), count
                             in selected_drift.items() if (conc, cap) == (concurrency, capacity)))
        interactions.append({
            "concurrency": concurrency, "capacity_bytes": capacity,
            "agent_episodes": len(classes["agent_tool"]["ttft_ms"]),
            "ordinary_episodes": len(classes["ordinary"]["ttft_ms"]),
            "agent_domains_consistent": domain_consistent,
            "minimum_independent_episodes_for_claim": MIN_EPISODES_FOR_CLAIM,
            "selected_set_drift_repeats": sum(
                count for (_cls, _work, _episode, conc, cap), count in selected_drift.items()
                if (conc, cap) == (concurrency, capacity)),
            "effects": effects,
            "agent_specific_latency_and_throughput_regression_supported": supported,
        })
    return {
        "schema_version": 1,
        "evidence_level": "analysis of submitted timed rows; hardware provenance requires independent audit",
        "attestation_scope": "Input fields are checked for consistency, not independently verified on NPU",
        "comparison": "ECHO prefetch minus same offload backend without prefetch",
        "input_rows": len(rows),
        "cells": cells,
        "interactions": interactions,
        "interpretation": "Report direction, magnitude and uncertainty per cell; "
                          "no universal ECHO or agent-specific failure follows from one cell.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Output already exists")
    raw = args.measurements.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    result = analyze(rows)
    result["input_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
