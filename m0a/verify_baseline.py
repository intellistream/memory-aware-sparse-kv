"""Recompute the sealed Stage 0 serving baseline from local raw artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    from .contracts import SCHEMA_VERSION, sha256_file
except ImportError:
    from contracts import SCHEMA_VERSION, sha256_file


INPUT_LENGTHS = (8192, 16384, 32768, 65536)
CONCURRENCIES = (1, 2, 4, 8)


def _number(row, key, cast=float):
    try:
        return cast(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f'Invalid baseline field {key}: {row.get(key)!r}') from error


def _signature(row):
    return {key: row.get(key) for key in
            ('content', 'reasoning', 'token_ids', 'finish_reason')}


def verify(root: Path):
    matrix = root / 'results/baseline/matrix'
    summary_path = matrix / 'summary.csv'
    with summary_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    expected = {
        f'serve_in{length}_out64_n20_c{concurrency}'
        for length in INPUT_LENGTHS for concurrency in CONCURRENCIES
    }
    actual = {row['run_id'] for row in rows}
    if len(rows) != 16 or actual != expected:
        raise ValueError(f'Baseline run matrix mismatch: missing={expected-actual}, extra={actual-expected}')
    total_completed = total_failed = 0
    normalized = []
    for row in rows:
        run_id = row['run_id']
        raw_path = matrix / 'raw' / f'{run_id}.json'
        log_path = matrix / 'logs' / f'{run_id}.log'
        if not raw_path.is_file() or not log_path.is_file():
            raise ValueError(f'Missing raw result or log for {run_id}')
        raw = json.loads(raw_path.read_text())
        for csv_key, raw_key in (
            ('completed', 'completed'), ('failed', 'failed'),
            ('concurrency', 'max_concurrency'),
            ('request_throughput_req_s', 'request_throughput'),
            ('output_throughput_tok_s', 'output_throughput'),
            ('p99_ttft_ms', 'p99_ttft_ms'), ('p99_tpot_ms', 'p99_tpot_ms'),
        ):
            if float(row[csv_key]) != float(raw[raw_key]):
                raise ValueError(f'{run_id} differs between CSV and raw JSON: {csv_key}')
        input_tokens = _number(row, 'input_tokens', int)
        if raw['total_input_tokens'] != input_tokens * raw['num_prompts']:
            raise ValueError(f'{run_id} total input token mismatch')
        completed = _number(row, 'completed', int)
        failed = _number(row, 'failed', int)
        if completed != 20 or failed != 0:
            raise ValueError(f'{run_id} did not complete 20/20 requests')
        total_completed += completed
        total_failed += failed
        normalized.append({
            'run_id': run_id,
            'input_tokens': input_tokens,
            'concurrency': _number(row, 'concurrency', int),
            'completed': completed,
            'failed': failed,
            'request_throughput_req_s': _number(row, 'request_throughput_req_s'),
            'output_throughput_tok_s': _number(row, 'output_throughput_tok_s'),
            'p50_ttft_ms': _number(row, 'median_ttft_ms'),
            'p99_ttft_ms': _number(row, 'p99_ttft_ms'),
            'p50_tpot_ms': _number(row, 'median_tpot_ms'),
            'p99_tpot_ms': _number(row, 'p99_tpot_ms'),
            'raw_sha256': sha256_file(raw_path),
            'log_sha256': sha256_file(log_path),
        })
    sanity_path = root / 'results/sanity_repeat.jsonl'
    sanity = [json.loads(line) for line in sanity_path.read_text().splitlines() if line]
    if len(sanity) != 3 or any(row.get('http_status') != 200 for row in sanity):
        raise ValueError('Stage 0 sanity records are incomplete')
    if any(_signature(row) != _signature(sanity[0]) for row in sanity[1:]):
        raise ValueError('Stage 0 32-token sanity outputs differ')
    if any(row.get('usage', {}).get('completion_tokens') != 32 for row in sanity):
        raise ValueError('Stage 0 sanity scope is not exactly 32 completion tokens')
    return {
        'schema_version': SCHEMA_VERSION,
        'scope': 'Stage 0 serving performance; no sparse selected-set or KV recall inference',
        'matrix': {
            'run_count': len(normalized),
            'completed': total_completed,
            'failed': total_failed,
            'input_lengths': list(INPUT_LENGTHS),
            'concurrencies': list(CONCURRENCIES),
            'runs': sorted(normalized, key=lambda x: (x['input_tokens'], x['concurrency'])),
        },
        'sanity': {
            'repetitions': 3,
            'completion_tokens': 32,
            'identical': True,
            'records_sha256': sha256_file(sanity_path),
            'historical_64_token_divergence_retained': (
                root / 'results/sanity_repeat_failed_20260917.jsonl'
            ).is_file(),
        },
    }


def render_svg(result, output: Path):
    """Dependency-free overview of throughput and p99 TTFT."""
    width, height = 960, 560
    left, right, top, bottom = 75, 25, 55, 60
    panel_w = (width - left - right - 70) / 2
    panel_h = height - top - bottom
    colors = {1: '#006b83', 2: '#4b91a8', 4: '#e07020', 8: '#8e4c97'}
    runs = result['matrix']['runs']

    def panel(x0, metric, title, ymax):
        parts = [
            f'<rect x="{x0}" y="{top}" width="{panel_w}" height="{panel_h}" fill="#fafcfd" stroke="#78909c"/>',
            f'<text x="{x0 + panel_w/2}" y="28" text-anchor="middle" font-size="18" font-weight="bold">{title}</text>',
        ]
        for tick in range(6):
            value = ymax * tick / 5
            y = top + panel_h - panel_h * tick / 5
            parts.append(f'<line x1="{x0}" y1="{y}" x2="{x0+panel_w}" y2="{y}" stroke="#dce5e8"/>')
            parts.append(f'<text x="{x0-8}" y="{y+4}" text-anchor="end" font-size="11">{value:.0f}</text>')
        for concurrency in CONCURRENCIES:
            selected = [row for row in runs if row['concurrency'] == concurrency]
            points = []
            for index, row in enumerate(selected):
                x = x0 + panel_w * index / (len(selected)-1)
                y = top + panel_h - panel_h * row[metric] / ymax
                points.append((x, y))
            value = ' '.join(f'{x:.1f},{y:.1f}' for x, y in points)
            parts.append(f'<polyline points="{value}" fill="none" stroke="{colors[concurrency]}" stroke-width="3"/>')
            parts.extend(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors[concurrency]}"/>' for x,y in points)
        for index, length in enumerate(INPUT_LENGTHS):
            x = x0 + panel_w * index / (len(INPUT_LENGTHS)-1)
            parts.append(f'<text x="{x}" y="{top+panel_h+22}" text-anchor="middle" font-size="11">{length//1024}K</text>')
        return parts

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>']
    svg += panel(left, 'output_throughput_tok_s', 'Output throughput (tok/s)', 60)
    svg += panel(left + panel_w + 70, 'p99_ttft_ms', 'p99 TTFT (ms)', 34000)
    legend_y = height - 15
    for index, concurrency in enumerate(CONCURRENCIES):
        x = 255 + index * 120
        svg.append(f'<line x1="{x}" y1="{legend_y}" x2="{x+24}" y2="{legend_y}" stroke="{colors[concurrency]}" stroke-width="3"/>')
        svg.append(f'<text x="{x+30}" y="{legend_y+4}" font-size="12">C={concurrency}</text>')
    svg.append('</svg>')
    output.write_text('\n'.join(svg) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, default=Path('m0a/baseline_offline_summary.json'))
    parser.add_argument('--svg', type=Path, default=Path('m0a/baseline_overview.svg'))
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    for path in (args.output, args.svg):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f'Refusing to overwrite {path}')
    result = verify(args.root)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    render_svg(result, args.svg)
    print(f"Verified {result['matrix']['run_count']} runs: "
          f"{result['matrix']['completed']} completed, {result['matrix']['failed']} failed")


if __name__ == '__main__':
    main()
