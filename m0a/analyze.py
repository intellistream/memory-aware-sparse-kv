"""Validate M0-A prefill traces and summarize paired locality changes."""

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

try:
    from .contracts import (EXPECTED_DSA_LAYERS, SCHEMA_VERSION, load_jsonl,
                            resolve_api_request_id, trace_blocks,
                            validate_response, validate_trace)
    from .model_profiles import DEFAULT_PROFILE, load_profile
except ImportError:  # Direct execution from /workspace/memecho/m0a.
    from contracts import (EXPECTED_DSA_LAYERS, SCHEMA_VERSION, load_jsonl,
                           resolve_api_request_id, trace_blocks,
                           validate_response, validate_trace)
    from model_profiles import DEFAULT_PROFILE, load_profile


def percentile(values, percent):
    if not values:
        raise ValueError('No samples')
    values = sorted(values)
    position = (len(values) - 1) * percent / 100
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def distribution(values):
    return {'n': len(values), 'p50': percentile(values, 50),
            'p95': percentile(values, 95), 'p99': percentile(values, 99)}


def locality(before, after):
    union = before | after
    jaccard = len(before & after) / len(union) if union else 1.0
    new_rate = len(after - before) / len(after) if after else 0.0
    return {'jaccard': jaccard, 'new_block_rate': new_rate}


def bootstrap_interval(values, *, draws=2000, seed=0):
    rng = random.Random(seed)
    estimates = [statistics.mean(rng.choices(values, k=len(values)))
                 for _ in range(draws)]
    return [percentile(estimates, 2.5), percentile(estimates, 97.5)]


def load_trace(trace_dir, responses, contract=None):
    wanted = {row['request_id']: {row['boundary_position'] - 1,
                                  row['boundary_position']}
              for row in responses}
    run_ids = {row['request_id']: row['run_id'] for row in responses}
    trace = {}
    inventory = defaultdict(lambda: {
        'rows': 0, 'ranks': set(), 'layers': set(), 'engine_request_ids': set(),
        'layer_sources': {}, 'native_operators': set(),
    })
    files = sorted(trace_dir.glob('rank*.jsonl'))
    if not files:
        raise ValueError(f'No rank trace files in {trace_dir}')
    for path in files:
        rank_from_name = int(path.stem.removeprefix('rank'))
        for record in load_jsonl(path):
            validate_trace(record, contract=contract)
            engine_request_id = record['request_id']
            request_id = resolve_api_request_id(engine_request_id, set(wanted))
            if request_id is None:
                raise ValueError(f'Unlinked engine request ID: {engine_request_id}')
            if record['run_id'] != run_ids[request_id]:
                raise ValueError(f'Trace run ID mismatch: {request_id}')
            inventory[request_id]['rows'] += 1
            inventory[request_id]['ranks'].add(record['rank'])
            inventory[request_id]['layers'].add(record['layer'])
            inventory[request_id]['engine_request_ids'].add(engine_request_id)
            source = record.get('selection_source', 'computed')
            previous = inventory[request_id]['layer_sources'].setdefault(
                record['layer'], source
            )
            if previous != source:
                raise ValueError(
                    f'Selection source changes within layer: {record["layer"]}'
                )
            inventory[request_id]['native_operators'].add(
                record.get('native_operator', 'npu_vllm_quant_lightning_indexer')
            )
            position = record['prompt_position']
            if position not in wanted[request_id]:
                continue
            if record['rank'] != rank_from_name:
                raise ValueError(f'Rank filename mismatch: {path}')
            blocks = trace_blocks(record)
            valid = {value for value in blocks if value >= 0}
            key = (request_id, record['layer'], position)
            if key in trace:
                raise ValueError(f'Duplicate request/layer/position: {key}')
            trace[key] = (record['rank'], valid)
    return trace, inventory


def summarize(trace, responses):
    by_pair = defaultdict(dict)
    all_request_metrics = {}
    for row in responses:
        request_id = row['request_id']
        boundary = row['boundary_position']
        before = {(layer, rank): blocks for (req, layer, pos),
                  (rank, blocks) in trace.items()
                  if req == request_id and pos == boundary - 1}
        after = {(layer, rank): blocks for (req, layer, pos),
                 (rank, blocks) in trace.items()
                 if req == request_id and pos == boundary}
        if not before or before.keys() != after.keys():
            raise ValueError(f'Incomplete request/layer/token linkage: {request_id}')
        metrics = {key: locality(before[key], after[key]) for key in before}
        all_request_metrics[request_id] = {'before': before, 'metrics': metrics}
        variants = by_pair[(row['pair_id'], row['repetition'])]
        if row['variant'] in variants:
            raise ValueError(
                f'Duplicate variant in {row["pair_id"]}, repetition {row["repetition"]}'
            )
        variants[row['variant']] = row

    result = {}
    for (pair_id, repetition), variants in by_pair.items():
        if set(variants) != {'event', 'control'}:
            raise ValueError(f'Missing variant in {pair_id}, repetition {repetition}')
        event = variants['event']
        control = variants['control']
        event_data = all_request_metrics[event['request_id']]
        control_data = all_request_metrics[control['request_id']]
        if event_data['metrics'].keys() != control_data['metrics'].keys():
            raise ValueError(f'Event/control layer-rank mismatch: {pair_id}')
        keys = event_data['metrics']
        prefix_matches = sum(event_data['before'][key] == control_data['before'][key]
                             for key in keys)
        if prefix_matches != len(keys):
            raise ValueError(
                f'Event/control prefix selected sets differ: {pair_id}, repetition {repetition}'
            )
        entry = result.setdefault(pair_id, {
            'context_target': event['context_target'],
            'event_type': event['event_type'],
            'workload_id': event.get('workload_id', 'synthetic_smoke'),
            'episode_id': event.get('episode_id', pair_id),
            'trajectory_pairs': [],
            'event_jaccard': [], 'control_jaccard': [],
            'event_new_block_rate': [], 'control_new_block_rate': [],
            'by_layer': {},
        })
        differences = {'jaccard_drop': [], 'new_block_rate_increase': []}
        for key in keys:
            layer, rank = key
            em, cm = event_data['metrics'][key], control_data['metrics'][key]
            entry['event_jaccard'].append(em['jaccard'])
            entry['control_jaccard'].append(cm['jaccard'])
            entry['event_new_block_rate'].append(em['new_block_rate'])
            entry['control_new_block_rate'].append(cm['new_block_rate'])
            differences['jaccard_drop'].append(cm['jaccard'] - em['jaccard'])
            differences['new_block_rate_increase'].append(
                em['new_block_rate'] - cm['new_block_rate'])
            layer_data = entry['by_layer'].setdefault(layer, {
                'ranks': set(), 'event_jaccard': [], 'control_jaccard': [],
                'event_new_block_rate': [], 'control_new_block_rate': [],
                'jaccard_drop': [], 'new_block_rate_increase': [],
            })
            layer_data['ranks'].add(rank)
            layer_data['event_jaccard'].append(em['jaccard'])
            layer_data['control_jaccard'].append(cm['jaccard'])
            layer_data['event_new_block_rate'].append(em['new_block_rate'])
            layer_data['control_new_block_rate'].append(cm['new_block_rate'])
            layer_data['jaccard_drop'].append(cm['jaccard'] - em['jaccard'])
            layer_data['new_block_rate_increase'].append(
                em['new_block_rate'] - cm['new_block_rate'])
        entry['trajectory_pairs'].append({
            'repetition': repetition,
            'event_request_id': event['request_id'],
            'control_request_id': control['request_id'],
            'layer_rank_count': len(keys),
            'prefix_selected_set_matches': prefix_matches,
            'mean_jaccard_drop': statistics.mean(differences['jaccard_drop']),
            'mean_new_block_rate_increase': statistics.mean(
                differences['new_block_rate_increase']),
        })
    for entry in result.values():
        entry['prefix_selected_sets_match'] = all(
            item['prefix_selected_set_matches'] == item['layer_rank_count']
            for item in entry['trajectory_pairs']
        )
        for name in ('event_jaccard', 'control_jaccard',
                     'event_new_block_rate', 'control_new_block_rate'):
            entry[name] = distribution(entry[name])
        trajectories = entry['trajectory_pairs']
        for key in ('mean_jaccard_drop', 'mean_new_block_rate_increase'):
            values = [item[key] for item in trajectories]
            entry[key] = {'mean': statistics.mean(values),
                          'repeat_bootstrap_95pct': bootstrap_interval(values)}
        for layer_data in entry['by_layer'].values():
            layer_data['ranks'] = sorted(layer_data['ranks'])
            for name in ('event_jaccard', 'control_jaccard',
                         'event_new_block_rate', 'control_new_block_rate'):
                layer_data[name] = distribution(layer_data[name])
            for name in ('jaccard_drop', 'new_block_rate_increase'):
                values = layer_data[name]
                layer_data[name] = {
                    'mean': statistics.mean(values),
                    'repeat_bootstrap_95pct': bootstrap_interval(values),
                }
    return result


def summarize_groups(pairs):
    """Aggregate once per independent episode; repetitions are not samples."""
    grouped = defaultdict(lambda: {
        'jaccard_drop': [], 'new_block_rate_increase': [], 'pair_ids': []
    })
    for pair_id, entry in pairs.items():
        key = (entry['workload_id'], entry['event_type'], entry['context_target'])
        grouped[key]['jaccard_drop'].append(entry['mean_jaccard_drop']['mean'])
        grouped[key]['new_block_rate_increase'].append(
            entry['mean_new_block_rate_increase']['mean'])
        grouped[key]['pair_ids'].append(pair_id)
    result = {}
    for (workload, event_type, context), values in grouped.items():
        name = f'{workload}/{event_type}/{context}'
        result[name] = {
            'workload_id': workload,
            'event_type': event_type,
            'context_target': context,
            'independent_episode_count': len(values['pair_ids']),
            'pair_ids': sorted(values['pair_ids']),
        }
        for metric in ('jaccard_drop', 'new_block_rate_increase'):
            samples = values[metric]
            result[name][metric] = {
                'mean': statistics.mean(samples),
                'episode_bootstrap_95pct': bootstrap_interval(samples),
            }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace-dir', type=Path, required=True)
    parser.add_argument('--responses', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('/workspace/memecho'))
    parser.add_argument('--model-profile', default=DEFAULT_PROFILE)
    parser.add_argument('--expected-layers', type=int)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    profile = load_profile(
        args.model_profile, args.root / 'm0a/model_profiles.json'
    )
    expected_layers = args.expected_layers or profile['expected_layers']
    responses = list(load_jsonl(args.responses))
    if not responses:
        raise ValueError('No paired responses')
    if len({row['request_id'] for row in responses}) != len(responses):
        raise ValueError('Duplicate request ID in responses')
    for row in responses:
        validate_response(row)
        if row.get('model_profile', args.model_profile) != args.model_profile:
            raise ValueError('Response model profile mismatch')
    trace, inventory = load_trace(args.trace_dir, responses, contract=profile)
    for row in responses:
        count = len(inventory[row['request_id']]['layers'])
        if count != expected_layers:
            raise ValueError(f'Expected {expected_layers} selected-set layers for '
                             f'{row["request_id"]}, found {count}')
        sources = inventory[row['request_id']]['layer_sources'].values()
        computed = sum(source == 'computed' for source in sources)
        reused = sum(source == 'reused' for source in sources)
        if (computed, reused) != (profile['computed_layers'], profile['reused_layers']):
            raise ValueError(
                f'Selection source layer counts differ for {row["request_id"]}: '
                f'{computed}/{reused}'
            )
    pair_results = summarize(trace, responses)
    result = {
        'schema_version': SCHEMA_VERSION,
        'model_profile': args.model_profile,
        'model_id': profile['model_id'],
        'model_revision': profile['model_revision'],
        'attention_backend': profile['attention_backend'],
        'scope': 'prefill selected-set boundary, one API request per trajectory',
        'statistical_unit': 'independent episode; repetitions assess runtime stability',
        'response_count': len(responses),
        'request_inventory': {
            key: {'rows': value['rows'], 'ranks': sorted(value['ranks']),
                  'layers': len(value['layers']),
                  'computed_layers': sum(
                      source == 'computed'
                      for source in value['layer_sources'].values()
                  ),
                  'reused_layers': sum(
                      source == 'reused'
                      for source in value['layer_sources'].values()
                  ),
                  'native_operators': sorted(value['native_operators']),
                  'engine_request_ids': sorted(value['engine_request_ids'])}
            for key, value in inventory.items()
        },
        'pairs': pair_results,
        'groups': summarize_groups(pair_results),
    }
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Validated {len(responses)} requests and {len(result["pairs"])} pair types')


if __name__ == '__main__':
    main()
