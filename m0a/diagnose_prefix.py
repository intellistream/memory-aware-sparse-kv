"""Diagnose fail-closed event/control prefix selected-set mismatches."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

try:
    from .analyze import load_trace
    from .contracts import load_jsonl
    from .model_profiles import load_profile
except ImportError:
    from analyze import load_trace
    from contracts import load_jsonl
    from model_profiles import load_profile


def set_hash(values: set[int]) -> str:
    payload = ','.join(map(str, sorted(values))).encode()
    return hashlib.sha256(payload).hexdigest()


def diagnose(trace, responses):
    grouped = defaultdict(dict)
    for row in responses:
        grouped[(row['pair_id'], row['repetition'])][row['variant']] = row
    result = []
    for (pair_id, repetition), variants in sorted(grouped.items()):
        if set(variants) != {'event', 'control'}:
            continue
        event, control = variants['event'], variants['control']
        boundary = event['boundary_position']
        event_before = {
            (layer, rank): blocks
            for (request, layer, position), (rank, blocks) in trace.items()
            if request == event['request_id'] and position == boundary - 1
        }
        control_before = {
            (layer, rank): blocks
            for (request, layer, position), (rank, blocks) in trace.items()
            if request == control['request_id'] and position == boundary - 1
        }
        differences = []
        for key in sorted(set(event_before) | set(control_before)):
            left, right = event_before.get(key, set()), control_before.get(key, set())
            if left != right:
                layer, rank = key
                differences.append({
                    'layer': layer,
                    'rank': rank,
                    'event_size': len(left),
                    'control_size': len(right),
                    'intersection': len(left & right),
                    'symmetric_difference': len(left ^ right),
                    'event_only': sorted(left - right)[:16],
                    'control_only': sorted(right - left)[:16],
                    'event_sha256': set_hash(left),
                    'control_sha256': set_hash(right),
                })
        result.append({
            'pair_id': pair_id,
            'repetition': repetition,
            'boundary_position': boundary,
            'layer_count': len(event_before),
            'mismatched_layers': len(differences),
            'differences': differences,
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/workspace/memecho'))
    parser.add_argument('--model-profile', required=True)
    parser.add_argument('--trace-dir', type=Path, required=True)
    parser.add_argument('--responses', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    profile = load_profile(
        args.model_profile, args.root / 'm0a/model_profiles.json'
    )
    responses = list(load_jsonl(args.responses))
    trace, _ = load_trace(args.trace_dir, responses, contract=profile)
    result = {
        'schema_version': 1,
        'purpose': 'diagnostic only; not an experimental effect estimate',
        'model_profile': args.model_profile,
        'comparisons': diagnose(trace, responses),
    }
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    failed = [row for row in result['comparisons'] if row['mismatched_layers']]
    print(json.dumps({
        'comparisons': len(result['comparisons']),
        'mismatched_comparisons': len(failed),
        'mismatched_layers': sum(row['mismatched_layers'] for row in failed),
    }, sort_keys=True))


if __name__ == '__main__':
    main()
