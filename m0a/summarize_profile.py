"""Aggregate the eight-rank Ascend Stage 0 profile without NPU access."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    from .contracts import SCHEMA_VERSION
except ImportError:
    try:
        from contracts import SCHEMA_VERSION
    except ImportError:  # Supports `ssh host python3 -` for read-only analysis.
        SCHEMA_VERSION = 1


TARGET_PATTERNS = (
    'VllmQuantLightningIndexer',
    'npu_vllm_quant_lightning_indexer',
    'vllm::dsa_forward',
)


def number(value):
    if value is None or value in {'', 'N/A'}:
        return 0.0
    return float(str(value).strip())


def rank_from_path(path: Path) -> int:
    match = re.search(r'_rank(\d+)_', str(path))
    if not match:
        raise ValueError(f'Cannot identify rank from {path}')
    return int(match.group(1))


def aggregate(profile_root: Path):
    outputs = sorted(profile_root.glob('**/ASCEND_PROFILER_OUTPUT'))
    if len(outputs) != 8:
        raise ValueError(f'Expected eight parsed rank directories, found {len(outputs)}')
    rank_rows = []
    ops = defaultdict(lambda: {'count': 0, 'total_time_us': 0.0, 'ranks': set()})
    targets = defaultdict(lambda: {
        'operator_count': 0, 'device_self_time_us': 0.0,
        'device_total_time_us': 0.0, 'ranks': set(),
    })
    for directory in outputs:
        rank = rank_from_path(directory)
        if not (directory / 'analyse.done').is_file():
            raise ValueError(f'Rank {rank} has no analyse.done')
        with (directory / 'step_trace_time.csv').open(newline='') as stream:
            steps = list(csv.DictReader(stream))
        if not steps:
            raise ValueError(f'Rank {rank} has no step trace')
        step = steps[-1]
        rank_rows.append({
            'rank': rank,
            'computing_us': number(step.get('Computing')),
            'communication_us': number(step.get('Communication')),
            'communication_not_overlapped_us': number(step.get('Communication(Not Overlapped)')),
            'overlapped_us': number(step.get('Overlapped')),
            'free_us': number(step.get('Free')),
            'stage_us': number(step.get('Stage')),
            'preparing_us': number(step.get('Preparing')),
        })
        with (directory / 'op_statistic.csv').open(newline='') as stream:
            for row in csv.DictReader(stream):
                name = row['OP Type']
                ops[name]['count'] += int(row['Count'])
                ops[name]['total_time_us'] += number(row['Total Time(us)'])
                ops[name]['ranks'].add(rank)
        with (directory / 'operator_details.csv').open(newline='') as stream:
            for row in csv.DictReader(stream):
                name = row['Name']
                if not any(pattern.lower() in name.lower() for pattern in TARGET_PATTERNS):
                    continue
                target = targets[name]
                target['operator_count'] += 1
                target['device_self_time_us'] += number(row.get('Device Self Duration(us)'))
                target['device_total_time_us'] += number(row.get('Device Total Duration(us)'))
                target['ranks'].add(rank)
    if {row['rank'] for row in rank_rows} != set(range(8)):
        raise ValueError('Parsed profile ranks are not exactly 0..7')

    def finish(mapping, limit=None):
        rows = []
        for name, values in mapping.items():
            row = {'name': name, **values, 'ranks': sorted(values['ranks'])}
            rows.append(row)
        rows.sort(key=lambda row: row.get('total_time_us', row.get('device_self_time_us', 0)),
                  reverse=True)
        return rows[:limit] if limit else rows

    computing = [row['computing_us'] for row in rank_rows]
    communication = [row['communication_us'] for row in rank_rows]
    return {
        'schema_version': SCHEMA_VERSION,
        'scope': ('One 1,029-token prompt plus 32-token completion; operator reference only, '
                  'no selected-set or KV recall inference'),
        'rank_count': 8,
        'rank_summary': sorted(rank_rows, key=lambda row: row['rank']),
        'rank_imbalance': {
            'computing_max_over_min': max(computing) / min(computing),
            'communication_max_over_min': max(communication) / min(communication),
        },
        'target_operators': finish(targets),
        'top_op_types': finish(ops, limit=30),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('profile_root', type=Path)
    parser.add_argument('--output', default='-')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    result = aggregate(args.profile_root)
    rendered = json.dumps(result, indent=2) + '\n'
    if args.output == '-':
        sys.stdout.write(rendered)
    else:
        output = Path(args.output)
        if output.exists() and not args.overwrite:
            raise FileExistsError(f'Refusing to overwrite {output}')
        output.write_text(rendered)
        print(f'Aggregated {result["rank_count"]} ranks into {output}')


if __name__ == '__main__':
    main()
