"""Fixed single-concurrency 432-request order for the GLM online experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .echo_workload import sha_ids
from .glm_online_stats import MODES


def make_schedule(cases: dict) -> list[dict]:
    items = cases.get("cases")
    if not isinstance(items, list) or len(items) != 48 or \
            len({item["case_id"] for item in items}) != 48:
        raise ValueError("Exactly 48 distinct sealed GLM event cases required")
    for item in items:
        continuation = item.get("forced_continuation_token_ids")
        if not isinstance(continuation, list) or len(continuation) != 32 or \
                sha_ids(continuation) != item.get("forced_continuation_sha256"):
            raise ValueError("Every case needs a sealed 32-token continuation")
    result = []
    for repeat in range(3):
        for item in items:
            for offset in range(3):
                result.append({"ordinal": len(result), "case_id": item["case_id"],
                               "repeat": repeat, "mode": MODES[(repeat + offset) % 3]})
    if len(result) != 432:
        raise AssertionError("Incomplete 432-request schedule")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.cases.read_bytes()
    schedule = make_schedule(json.loads(source))
    payload = {"schema_version": 1, "cases_sha256": hashlib.sha256(source).hexdigest(),
               "warmup_excluded": True, "single_concurrency": True,
               "requests": schedule}
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(json.dumps({"requests": len(schedule), "output_sha256":
                      hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
