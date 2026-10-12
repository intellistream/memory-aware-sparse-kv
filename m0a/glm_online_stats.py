"""Pre-registered analysis of the 48-case, three-mode GLM online experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

MODES = ("no_offload", "offload_no_prefetch", "offload_echo_prefetch")
METRICS = ("prefill_window_ms", "tpot_ms")
REPEATS = 3
CI_COVERAGE = 0.9875


def require(value, message):
    if not value:
        raise ValueError(message)


def _interval(values_by_stratum: dict[str, list[float]], *, seed: int, draws: int) -> list[float]:
    rng = random.Random(seed)
    sampled = []
    for _ in range(draws):
        values = []
        for stratum, members in sorted(values_by_stratum.items()):
            require(members, f"Empty context stratum: {stratum}")
            values.extend(rng.choices(members, k=len(members)))
        sampled.append(mean(values))
    sampled.sort()
    tail = (1.0 - CI_COVERAGE) / 2.0
    return [sampled[int(tail * (draws - 1))], sampled[int((1 - tail) * (draws - 1))]]


def analyze(cases: dict, rows: list[dict], *, draws: int = 10000, seed: int = 20261012) -> dict:
    require(draws >= 1000, "At least 1,000 bootstrap draws required")
    items = cases.get("cases")
    require(isinstance(items, list) and len(items) == 48, "Exactly 48 sealed event cases required")
    by_id = {case["case_id"]: case for case in items}
    require(len(by_id) == 48, "Duplicate sealed case")
    require({domain: sum(case["domain"] == domain for case in items)
             for domain in ("retail", "banking")} == {"retail": 24, "banking": 24},
            "Expected 24 independent episodes per domain")
    keyed = {}
    for row in rows:
        case = by_id.get(row.get("case_id"))
        require(case is not None, "Unknown timed case")
        mode, repeat = row.get("mode"), row.get("repeat")
        require(mode in MODES and type(repeat) is int and 0 <= repeat < REPEATS,
                "Invalid mode or repeat")
        key = (case["case_id"], repeat, mode)
        require(key not in keyed, "Duplicate timed request")
        keyed[key] = row
        require(row.get("prompt_token_ids_sha256") == case["prompt_token_ids_sha256"] and
                row.get("prompt_tokens") == case["prompt_tokens"], "Input token drift")
        require(row.get("forced_continuation_token_ids") == case.get("forced_continuation_token_ids") and
                row.get("fixed_continuation_verified") is True and
                len(row["forced_continuation_token_ids"]) == 32,
                "Fixed continuation was not executed exactly")
        require(all(row.get(flag) is True for flag in
                    ("native_topk_exact", "kv_payload_exact", "attention_equivalent")),
                "NPU score, KV, or attention correctness gate failed")
        require(row.get("prototype_hbm_shadow") is True,
                "Prototype HBM shadow must be disclosed")
        require(isinstance(row.get("native_selected_set_sha256"), str) and
                len(row["native_selected_set_sha256"]) == 64, "Native selected-set hash missing")
        for metric in (*METRICS, "ttft_ms", "throughput_tps", "prefetch_hit_bytes",
                       "prefetch_waste_bytes", "synchronous_recall_bytes"):
            value = row.get(metric)
            require(type(value) in (int, float) and value >= 0, f"Invalid {metric}")
    require(len(keyed) == 48 * REPEATS * len(MODES), "Incomplete 432-request matrix")
    require(len({(case["domain"], case["episode_id"]) for case in items}) == 48,
            "Episodes are not independent")
    drift = []
    for case in items:
        for repeat in range(REPEATS):
            arms = [keyed[(case["case_id"], repeat, mode)] for mode in MODES]
            hashes = {arm["native_selected_set_sha256"] for arm in arms}
            if len(hashes) != 1:
                drift.append({"case_id": case["case_id"], "repeat": repeat,
                              "hashes": {mode: keyed[(case["case_id"], repeat, mode)]
                                         ["native_selected_set_sha256"] for mode in MODES}})
    stable = {case["case_id"] for case in items
              if all(not any(row["case_id"] == case["case_id"] and row["repeat"] == repeat
                             for row in drift) for repeat in range(REPEATS))}
    results = []
    for domain in ("retail", "banking"):
        domain_cases = sorted((case for case in items if case["domain"] == domain),
                              key=lambda case: (case["prompt_tokens"], case["case_id"]))
        # Three equal-count context-length strata are fixed from the sealed inputs.
        strata = {case["case_id"]: str(i // 8) for i, case in enumerate(domain_cases)}
        controlled = [case for case in domain_cases if case["case_id"] in stable]
        for label, selected in (("all_exploratory", domain_cases), ("stable_controlled", controlled)):
            eligible = label == "stable_controlled" and len(selected) >= 20
            for metric_index, metric in enumerate(METRICS):
                by_stratum = defaultdict(list)
                for case in selected:
                    differences = [keyed[(case["case_id"], repeat, "offload_echo_prefetch")][metric]
                                   - keyed[(case["case_id"], repeat, "offload_no_prefetch")][metric]
                                   for repeat in range(REPEATS)]
                    by_stratum[strata[case["case_id"]]].append(mean(differences))
                interval = (_interval(by_stratum, seed=seed + metric_index +
                                      (0 if domain == "retail" else 100) +
                                      (0 if label == "all_exploratory" else 1000), draws=draws)
                            if len(selected) >= 2 else None)
                estimate = mean(value for values in by_stratum.values() for value in values) if selected else None
                results.append({"domain": domain, "subset": label, "metric": metric,
                                "episodes": len(selected), "echo_minus_no_prefetch": estimate,
                                "ci98_75": interval, "controlled_comparison_eligible": eligible,
                                "directional_slowdown_supported": bool(
                                    eligible and interval is not None and interval[0] > 0)})
    return {"schema_version": 1, "expected_requests": 432, "received_requests": len(rows),
            "prototype_hbm_shadow": True, "confidence_level": CI_COVERAGE,
            "comparison": "echo_prefetch_minus_no_prefetch",
            "drift": drift, "stable_episodes": len(stable), "results": results,
            "conclusion_scope": "Simplified GLM tiny single-NPU prototype only; no DeepSeek or paper-system inference"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Output already exists")
    cases = json.loads(args.cases.read_text())
    raw = args.measurements.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    result = analyze(cases, rows)
    result["cases_sha256"] = hashlib.sha256(args.cases.read_bytes()).hexdigest()
    result["measurements_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
