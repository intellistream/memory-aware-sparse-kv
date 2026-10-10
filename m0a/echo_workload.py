#!/usr/bin/env python3
"""Seal fixed-token agent workload schedules for later NPU timing.

The schedule is input evidence, not an execution result. A runtime must force
the listed continuation token IDs and attest that it did so before any timed
comparison is accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path


MODES = ("no_offload", "offload_no_prefetch", "offload_echo_prefetch")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha_ids(ids: list[int]) -> str:
    # Match the archived τ³ token-ID hashing contract exactly.
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def seal_agent_cases(pairs_path: Path, responses_path: Path,
                     continuation_length: int) -> list[dict]:
    require(continuation_length > 0, "Continuation length must be positive")
    document = json.loads(pairs_path.read_text())
    tokenizer_sha256 = document.get("tokenizer_json_sha256")
    require(isinstance(tokenizer_sha256, str) and
            re.fullmatch(r"[0-9a-f]{64}", tokenizer_sha256) is not None,
            "Sealed tokenizer SHA-256 is missing")
    pairs = document["pairs"]
    require(len(pairs) == 48, "Expected 48 sealed τ³ pairs")
    responses = [json.loads(line) for line in responses_path.read_text().splitlines() if line]
    selected = {}
    for row in responses:
        key = (row["pair_id"], row["variant"])
        if row["repetition"] == 0:
            require(key not in selected, "Duplicate source continuation")
            selected[key] = row
    cases = []
    for pair in pairs:
        for variant in ("event", "control"):
            key = (pair["pair_id"], variant)
            require(key in selected, "Missing source continuation")
            source = selected[key]
            prompt_ids = pair[variant]["prompt_token_ids"]
            continuation = source["signature"]["token_ids"]
            require(source["prompt_token_ids"] == prompt_ids and
                    source["prompt_token_ids_sha256"] == sha_ids(prompt_ids),
                    "Archived source prompt differs from sealed pair")
            require(len(continuation) >= continuation_length,
                    "Archived continuation is too short")
            forced = continuation[:continuation_length]
            cases.append({
                "case_id": pair["pair_id"] + ":" + variant,
                "pair_id": pair["pair_id"], "variant": variant,
                "workload_id": pair["workload_id"],
                "episode_id": pair["episode_id"],
                "workload_class": "agent_tool",
                "tokenizer_json_sha256": tokenizer_sha256,
                "prompt_token_ids": prompt_ids,
                "prompt_token_ids_sha256": sha_ids(prompt_ids),
                "forced_continuation_token_ids": forced,
                "forced_continuation_sha256": sha_ids(forced),
                "source_request_id": source["request_id"],
            })
    require(len(cases) == 96 and len({case["case_id"] for case in cases}) == 96,
            "Incomplete or duplicate case coverage")
    return cases


def schedule(cases: list[dict], *, concurrencies: tuple[int, ...],
             repeats: int, seed: int = 7) -> list[dict]:
    require(cases and repeats > 0 and concurrencies and
            all(type(value) is int and value > 0 for value in concurrencies),
            "Invalid workload schedule")
    require(len({case["case_id"] for case in cases}) == len(cases), "Duplicate case")
    rng = random.Random(seed)
    result = []
    for concurrency in concurrencies:
        for repeat in range(repeats):
            case_order = list(cases)
            rng.shuffle(case_order)
            mode_order = list(MODES)
            rng.shuffle(mode_order)
            for mode in mode_order:
                for ordinal, case in enumerate(case_order):
                    result.append({
                        **{key: case[key] for key in (
                            "case_id", "pair_id", "variant", "workload_id",
                            "episode_id", "workload_class",
                            "prompt_token_ids_sha256",
                            "forced_continuation_sha256")},
                        "mode": mode,
                        "concurrency": concurrency,
                        "repeat": repeat,
                        "arrival_ordinal": ordinal,
                        "requires_fixed_continuation_attestation": True,
                        "timing_validated": False,
                    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--continuation-length", type=int, default=8)
    parser.add_argument("--concurrencies", default="1,4,8,16,32")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    require(not args.output.exists(), "Output already exists")
    cases_path = args.output.with_suffix(".cases.jsonl")
    require(not cases_path.exists(), "Cases output already exists")
    cases = seal_agent_cases(args.pairs, args.responses, args.continuation_length)
    values = tuple(int(value) for value in args.concurrencies.split(","))
    rows = schedule(cases, concurrencies=values, repeats=args.repeats)
    cases_path.write_text("".join(json.dumps(case, separators=(",", ":")) + "\n"
                                  for case in cases))
    args.output.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n"
                                   for row in rows))
    manifest = {
        "schema_version": 1,
        "pairs_sha256": hashlib.sha256(args.pairs.read_bytes()).hexdigest(),
        "responses_sha256": hashlib.sha256(args.responses.read_bytes()).hexdigest(),
        "schedule_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "cases": len(cases),
        "rows": len(rows),
        "continuation_length": args.continuation_length,
        "tokenizer_json_sha256": cases[0]["tokenizer_json_sha256"],
        "concurrencies": list(values),
        "repeats": args.repeats,
        "fixed_continuation_executed": False,
        "npu_validated": False,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
