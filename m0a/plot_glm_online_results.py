"""Render the sealed GLM online comparison as a standalone SVG.

Only the preregistered latency metrics carry confidence intervals. Request
throughput is shown as a descriptive measurement of the same requests.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


MODES = ("offload_no_prefetch", "offload_echo_prefetch")
PANEL_METRICS = (
    ("prefill_window_ms", "32-query prefill window", "s", 6000),
    ("tpot_ms", "Time per output token", "ms", 230),
    ("throughput_tps", "Mean request throughput", "token/s", 4),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_checksums(path: Path, files: dict[str, Path]) -> None:
    expected = {}
    for line in path.read_text().splitlines():
        sha, name = line.split(maxsplit=1)
        expected[Path(name).name] = sha
    for name, file in files.items():
        if expected.get(name) != digest(file):
            raise ValueError(f"Completed-run checksum mismatch: {name}")


def summarize(manifest: dict, analysis: dict, rows: list[dict]) -> dict:
    if not manifest.get("formal") or manifest.get("completed_measurements") != 432:
        raise ValueError("Expected a completed formal run")
    if analysis.get("received_requests") != 432 or len(rows) != 432:
        raise ValueError("Incomplete formal measurements")
    if not manifest.get("prototype_hbm_shadow") or not analysis.get("prototype_hbm_shadow"):
        raise ValueError("Missing prototype HBM-shadow disclosure")
    if analysis.get("confidence_level") != 0.9875:
        raise ValueError("Unexpected confidence level")
    counts = Counter((r["domain"], r["mode"]) for r in rows)
    if counts != {(domain, mode): 72 for domain in ("retail", "banking")
                  for mode in ("no_offload", *MODES)}:
        raise ValueError("Missing domain or mode measurements")
    if len({(r["case_id"], r["repeat"], r["mode"]) for r in rows}) != 432:
        raise ValueError("Duplicate formal request")
    if any(not all(r.get(flag) is True for flag in
                   ("fixed_continuation_verified", "native_topk_exact",
                    "kv_payload_exact", "attention_equivalent")) for r in rows):
        raise ValueError("Correctness gate failed")

    drifted = {item["case_id"] for item in analysis["drift"]}
    result_lookup = {(r["domain"], r["subset"], r["metric"]): r
                     for r in analysis["results"]}
    result = {"panels": {}, "drift_records": len(analysis["drift"]),
              "drift_cases": len(drifted), "measurement_sha256":
              manifest["measurements_sha256"]}
    for domain, subset in (("banking", "stable_controlled"),
                           ("retail", "all_exploratory")):
        all_ids = {r["case_id"] for r in rows if r["domain"] == domain}
        selected = all_ids - drifted if subset == "stable_controlled" else all_ids
        expected_episodes = 21 if domain == "banking" else 24
        if len(selected) != expected_episodes:
            raise ValueError(f"Unexpected {domain} episode count")
        panel = {"episodes": len(selected), "controlled": domain == "banking"}
        for metric, _label, _unit, _scale in PANEL_METRICS:
            by_episode = defaultdict(list)
            for row in rows:
                if row["case_id"] in selected and row["mode"] in MODES:
                    by_episode[(row["case_id"], row["mode"])].append(row[metric])
            if any(len(by_episode[(case, mode)]) != 3 for case in selected for mode in MODES):
                raise ValueError("Episode is missing a repeat")
            baseline = mean(mean(by_episode[(case, MODES[0])]) for case in selected)
            echo = mean(mean(by_episode[(case, MODES[1])]) for case in selected)
            item = {"baseline": baseline, "echo": echo,
                    "relative_percent": (echo / baseline - 1) * 100}
            if metric != "throughput_tps":
                registered = result_lookup[(domain, subset, metric)]
                if registered["episodes"] != len(selected) or \
                        abs(echo - baseline - registered["echo_minus_no_prefetch"]) > 1e-8:
                    raise ValueError("Chart values disagree with preregistered analysis")
                if registered["controlled_comparison_eligible"] != (domain == "banking"):
                    raise ValueError("Incorrect controlled-comparison label")
                item["difference"] = registered["echo_minus_no_prefetch"]
                item["ci98_75"] = registered["ci98_75"]
            panel[metric] = item
        result["panels"][domain] = panel
    if result["drift_records"] != 13 or result["drift_cases"] != 10:
        raise ValueError("Selected-set drift differs from the sealed run")
    return result


def render(summary: dict) -> str:
    width, height = 1500, 890
    blue, orange, ink, muted = "#346891", "#d46b35", "#172b3d", "#516476"
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}" role="img" '
             f'aria-labelledby="title description">',
             '<title id="title">GLM agent prompt prefetch comparison</title>',
             '<desc id="description">Paired offload modes across banking and retail agent prompts. '
             'Only the banking subset meets the controlled comparison gate.</desc>',
             '<rect width="1500" height="890" fill="#ffffff"/>']

    def add_text(x, y, value, size=18, color=ink, weight="400", anchor="start"):
        parts.append(f'<text x="{x}" y="{y}" font-family="DejaVu Sans, Arial, sans-serif" '
                     f'font-size="{size}" font-weight="{weight}" fill="{color}" '
                     f'text-anchor="{anchor}">{html.escape(str(value))}</text>')

    add_text(58, 68, "GLM agent prompts: cost of ECHO-style prefetch", 30, ink, "700")
    add_text(58, 102, "Same KV movement in both offload modes · 48 event episodes × 3 repeats · single 910B2", 17, muted)
    parts.append(f'<rect x="1020" y="63" width="17" height="17" rx="3" fill="{blue}"/>')
    add_text(1045, 77, "No prefetch", 15)
    parts.append(f'<rect x="1205" y="63" width="17" height="17" rx="3" fill="{orange}"/>')
    add_text(1230, 77, "ECHO prefetch", 15)

    x_positions = (58, 537, 1016)
    cell_width = 426
    for x, (_metric, label, _unit, _scale) in zip(x_positions, PANEL_METRICS):
        add_text(x, 158, label, 19, ink, "700")

    for domain, y, heading, color in (
        ("banking", 195, "BANKING · CONTROLLED · 21 episodes", "#eaf3f9"),
        ("retail", 478, "RETAIL · EXPLORATORY · 24 episodes", "#f2f4f6"),
    ):
        parts.append(f'<rect x="44" y="{y}" width="1412" height="253" rx="14" fill="{color}"/>')
        add_text(58, y + 37, heading, 18, ink, "700")
        for x, (metric, _label, unit, scale) in zip(x_positions, PANEL_METRICS):
            item = summary["panels"][domain][metric]
            bar_start, max_bar = x + 126, 222
            for row_offset, mode, value, fill in (
                (74, "No prefetch", item["baseline"], blue),
                (111, "ECHO", item["echo"], orange),
            ):
                bar_y = y + row_offset
                add_text(x, bar_y + 17, mode, 15, muted)
                bar_width = max_bar * value / scale
                parts.append(f'<rect x="{bar_start}" y="{bar_y}" width="{bar_width:.2f}" '
                             f'height="22" rx="4" fill="{fill}"/>')
                display = (f"{value / 1000:.2f} s" if metric == "prefill_window_ms" else
                           f"{value:.1f} {unit}" if metric == "tpot_ms" else
                           f"{value:.2f} {unit}")
                add_text(bar_start + bar_width + 8, bar_y + 17, display, 14, ink, "600")
            change = item["relative_percent"]
            direction = "longer" if metric != "throughput_tps" else "lower"
            add_text(x, y + 164, f"{change:+.1f}% {direction} with prefetch", 17,
                     orange, "700")
            if metric != "throughput_tps":
                low, high = item["ci98_75"]
                if metric == "prefill_window_ms":
                    interval = f"Paired Δ {item['difference'] / 1000:+.2f} s · 98.75% CI [{low / 1000:.2f}, {high / 1000:.2f}] s"
                else:
                    interval = f"Paired Δ {item['difference']:+.1f} ms · 98.75% CI [{low:.1f}, {high:.1f}] ms"
                add_text(x, y + 197, interval, 13, muted)
            else:
                add_text(x, y + 197, "Descriptive metric; no preregistered CI", 13, muted)
    add_text(58, 786, "Banking meets the ≥20 stable-episode gate; retail does not (17 stable episodes).", 17, ink, "600")
    add_text(58, 819, "Prefill window = adapter work for 32 post-event queries, not full prefill latency.", 15, muted)
    add_text(58, 843, "Prototype retains an NPU KV shadow and does not overlap prefetch with compute; no HBM-saving claim.", 15, muted)
    add_text(58, 869, f"Source measurements SHA-256: {summary['measurement_sha256']}", 12, muted)
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--summary-json", type=Path)
    parser.add_argument("--output-svg", type=Path, required=True)
    args = parser.parse_args()
    if args.output_svg.exists():
        raise FileExistsError(args.output_svg)
    if args.run_dir:
        paths = {name: args.run_dir / name for name in
                 ("manifest.json", "analysis.json", "measurements.jsonl")}
        verify_checksums(args.run_dir / "completed.sha256", paths)
        manifest = json.loads(paths["manifest.json"].read_text())
        analysis = json.loads(paths["analysis.json"].read_text())
        if digest(paths["measurements.jsonl"]) != manifest["measurements_sha256"] or \
                digest(paths["measurements.jsonl"]) != analysis["measurements_sha256"]:
            raise ValueError("Measurement hash mismatch")
        rows = [json.loads(line) for line in paths["measurements.jsonl"].read_text().splitlines()]
        summary = summarize(manifest, analysis, rows)
    else:
        summary = json.loads(args.summary_json.read_text())
        if summary.get("measurement_sha256") != \
                "5b5319d3fffc30fb8bdbbc4c8a77a1384075945144568c2e9f5e965c1505fe3e":
            raise ValueError("Unrecognized sealed-run summary")
        for domain, expected in (("banking", 21), ("retail", 24)):
            panel = summary["panels"][domain]
            if panel["episodes"] != expected or panel["controlled"] != (domain == "banking"):
                raise ValueError("Incorrect domain label or sample count")
            for metric, _label, _unit, _scale in PANEL_METRICS:
                item = panel[metric]
                if abs((item["echo"] / item["baseline"] - 1) * 100 -
                       item["relative_percent"]) > 0.01:
                    raise ValueError("Incorrect relative change")
                if metric != "throughput_tps" and abs(item["echo"] - item["baseline"] -
                                                      item["difference"]) > 0.001:
                    raise ValueError("Incorrect paired difference")
    args.output_svg.parent.mkdir(parents=True, exist_ok=True)
    args.output_svg.write_text(render(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
