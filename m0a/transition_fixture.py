#!/usr/bin/env python3
"""Engineering fixture with exact synthetic token IDs; not a representative workload."""
import argparse
import copy
import json
from pathlib import Path

try:
    from .model_profiles import load_profile
except ImportError:
    from model_profiles import load_profile


def fixture(root, train_episodes=8, eval_episodes=3):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root/"traces").mkdir(exist_ok=True)
    profile = load_profile("glm53_tiny")
    sidecar = {"schema_version": 1, "synthetic": True, "snapshots": [], "events": []}
    rows = []
    for split, count in (("train", train_episodes), ("eval", eval_episodes)):
        for episode in range(count):
            for label in ("tool_result", "task_switch"):
                for variant in ("event", "control"):
                    for repetition in range(2):
                        stem = f"{split}-{episode}-{label}-{variant}-{repetition}"
                        sid = f"session-{stem}"
                        snapshot = {
                            "snapshot_id": stem+"-old", "session_id": sid, "context_version": "v1", "sequence": 1,
                            "model_id": profile["model_id"], "model_revision": profile["model_revision"],
                            "tokenizer_sha256": "0"*64, "token_ids": list(range(512)), "generated_tokens": 512,
                            "regions": [{"start": 0, "end": 128, "type": "system", "created_at": 0},
                                        {"start": 128, "end": 256, "type": "memory", "created_at": 0},
                                        {"start": 256, "end": 384, "type": "tool", "created_at": 1},
                                        {"start": 384, "end": 512, "type": "task", "created_at": 1}],
                            "lanes": [{"rank": 0, "layer": "model.layers.0", "kv_kind": "kv_token_position", "generated_ids": list(range(512))}],
                        }
                        current = copy.deepcopy(snapshot)
                        current.update(snapshot_id=stem+"-new", context_version="v2", sequence=2, token_ids=list(range(544)))
                        current["regions"][-1]["end"] = 544
                        sidecar["snapshots"].extend([snapshot, current])
                        event = {"event_id": stem, "request_id": stem, "run_id": "transition-fixture", "previous_snapshot_id": snapshot["snapshot_id"],
                                 "snapshot_id": current["snapshot_id"], "event_type": label if variant == "event" else "no_event",
                                 "previous_state": "tool_wait" if label == "tool_result" else "task_active",
                                 "event_position": 512, "resume_position": 512, "workload_id": "engineering_fixture",
                                 "episode_id": f"{split}-{episode}", "source_trace_id": f"source-{split}-{episode}-{label}", "pair_id": f"{split}-{episode}-{label}",
                                 "variant": variant, "repetition": repetition, "trajectory_id": stem, "split": split}
                        sidecar["events"].append(event)
                        target = 260 if label == "tool_result" and variant == "event" else 132
                        for position in range(511, 544):
                            raw = ([0, 1, 2, 3] if position == 511 else [target, target+1, target+2, target+3, position])
                            raw += [-1]*(2048-len(raw))
                            rows.append({"schema_version": 2, "run_id": event["run_id"], "request_id": stem+"-abc12345", "rank": 0,
                                         "layer": "model.layers.0", "prompt_position": position, "context_len": position+1,
                                         "request_context_len": 544, "raw_index_unit": "kv_token_position", "invalid_sentinel": -1,
                                         "compression_ratio": 1, "compressed_block_size": 128, "selected_width": 2048,
                                         "raw_selected_ids": raw, "timestamp_ns": position+1, "model_profile": "glm53_tiny",
                                         "model_id": profile["model_id"], "model_revision": profile["model_revision"], "attention_backend": "sfa",
                                         "native_operator": "torch_npu.npu_lightning_indexer", "selection_source": "computed", "logical_block_size": 128,
                                         "logical_block_ids": [u//128 if u >= 0 else -1 for u in raw]})
    config = {"schema_version": 1, "capacity_bytes": 512, "prefetch_budget_bytes": 256,
              "kv_unit_bytes": {"kv_token_position": 2}, "max_tables": 256, "bootstrap_draws": 1000, "seed": 7}
    (root/"sidecar.json").write_text(json.dumps(sidecar, indent=2)+"\n")
    (root/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    (root/"traces/rank0.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    return sidecar, config, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("Fixture output already exists; use a fresh directory")
    sidecar, config, rows = fixture(args.output_dir)
    print(json.dumps({"events": len(sidecar["events"]), "native_trace_rows": len(rows), "synthetic": True}))


if __name__ == "__main__":
    main()
