"""Deterministic engineering fixtures; no model or benchmark task is executed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fixture():
    rows = []
    for scenario in ("persistent", "switching"):
        for phase in ("prefill", "decode"):
            for query in range(24):
                stage = query // 8
                hot = stage * 4 if scenario == "switching" else 0
                batch = query // 2 if phase == "prefill" else query
                rows.append({"schema_version": 1, "source_kind": "synthetic_fixture",
                             "task_id": scenario, "run_id": "fixture", "request_id": scenario,
                             "context_version": "append-only-v1", "rank": 0, "layer": "layer.0",
                             "phase": phase, "query": query, "batch": batch,
                             "prediction_step": 2 * batch, "completion_step": 2 * batch + 1,
                             "stage": f"event-{stage}", "event_type": "synthetic_task_switch",
                             "block_count": 16, "eligible_blocks": list(range(16)),
                             "selected_blocks": [hot, hot + 1]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        for row in fixture():
            output.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
