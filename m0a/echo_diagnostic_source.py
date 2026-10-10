#!/usr/bin/env python3
"""Prepare a diagnostic DSA overlay that records native top-k scores.

This deliberately does not claim complete indexer scores. It is excluded from
timed ECHO measurements because copying scores to the CPU synchronizes NPU work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def enable_topk_score_capture(source: str, *, kernel_source: str) -> str:
    if "M0A_SPARSE_VALUES_WRITTEN_V1" not in kernel_source:
        raise ValueError("Installed indexer kernel does not prove sparse score output")
    start = "topk_idxs, _ = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer("
    end = "return topk_idxs"
    if source.count(start) != 1:
        raise ValueError("Expected exactly one native indexer call")
    before, tail = source.split(start, 1)
    if end not in tail:
        raise ValueError("Native indexer return point is missing")
    body, after = tail.split(end, 1)
    if body.count("return_value=False,") != 1:
        raise ValueError("Expected the pinned score-return setting")
    if body.count("record_prefill_selected(") != 1:
        raise ValueError("Selected-set trace hook is absent")
    body = body.replace("return_value=False,", "return_value=True,", 1)
    matches = list(re.finditer(r"(?m)^([ \t]*)rank=self\.tp_rank,\n([ \t]*)\)", body))
    if len(matches) != 1:
        raise ValueError("Pinned selected-set call shape changed")
    match = matches[0]
    body = (body[:match.start()] + match.group(1) + "rank=self.tp_rank,\n" +
            match.group(1) + "topk_values=topk_values,\n" +
            match.group(2) + ")" + body[match.end():])
    result = (before +
              "topk_idxs, topk_values = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer(" +
              body + end + after)
    compile(result, "dsa_cp.py", "exec")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dsa", type=Path, required=True)
    parser.add_argument("--kernel-source", type=Path, required=True)
    parser.add_argument("--output-dsa", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dsa.exists():
        raise ValueError("Output already exists")
    source = args.input_dsa.read_text()
    kernel_source = args.kernel_source.read_text()
    changed = enable_topk_score_capture(source, kernel_source=kernel_source)
    args.output_dsa.parent.mkdir(parents=True, exist_ok=True)
    args.output_dsa.write_text(changed)
    manifest = {
        "input_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "kernel_sha256": hashlib.sha256(kernel_source.encode()).hexdigest(),
        "output_sha256": hashlib.sha256(changed.encode()).hexdigest(),
        "score_coverage": "topk_only",
        "timed_performance_eligible": False,
        "requires_npu_validation": True,
    }
    args.output_dsa.with_suffix(".score-probe.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
