"""Generate deterministic CPU-only trace fixtures for the M0-A analyzer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .contracts import (COMPRESSED_BLOCK_SIZE, COMPRESSION_RATIO, EXPECTED_DSA_LAYERS,
                            RAW_INDEX_UNIT, SCHEMA_VERSION, SELECTED_WIDTH,
                            validate_pair_set, validate_response, validate_trace)
    from .model_profiles import DEFAULT_PROFILE, load_profile
except ImportError:
    from contracts import (COMPRESSED_BLOCK_SIZE, COMPRESSION_RATIO, EXPECTED_DSA_LAYERS,
                           RAW_INDEX_UNIT, SCHEMA_VERSION, SELECTED_WIDTH,
                           validate_pair_set, validate_response, validate_trace)
    from model_profiles import DEFAULT_PROFILE, load_profile


def make_trace_record(*, run_id, request_id, rank, layer, position, raw_values,
                      request_context_len, profile=None,
                      selection_source='computed'):
    width = profile['selected_width'] if profile else SELECTED_WIDTH
    block_size = profile['logical_block_size'] if profile else COMPRESSED_BLOCK_SIZE
    compression_ratio = profile['compression_ratio'] if profile else COMPRESSION_RATIO
    raw = list(raw_values) + [-1] * (width - len(raw_values))
    record = {
        'schema_version': profile['trace_schema_version'] if profile else SCHEMA_VERSION,
        'run_id': run_id,
        'request_id': request_id,
        'rank': rank,
        'layer': layer,
        'prompt_position': position,
        'context_len': position + 1,
        'request_context_len': request_context_len,
        'raw_index_unit': profile['raw_index_unit'] if profile else RAW_INDEX_UNIT,
        'invalid_sentinel': -1,
        'compression_ratio': compression_ratio,
        'compressed_block_size': block_size,
        'selected_width': width,
        'raw_selected_ids': raw,
        'timestamp_ns': 1,
    }
    blocks = [value // block_size if value >= 0 else -1 for value in raw]
    if profile and profile['trace_schema_version'] == 2:
        record.update({
            'model_profile': profile['name'],
            'model_id': profile['model_id'],
            'model_revision': profile['model_revision'],
            'attention_backend': profile['attention_backend'],
            'native_operator': (
                profile['native_operators'][0]
                if selection_source == 'computed'
                else profile['native_operators'][-1]
            ),
            'selection_source': selection_source,
            'logical_block_size': block_size,
            'logical_block_ids': blocks,
        })
    else:
        record['logical_compressed_block_ids'] = blocks
    validate_trace(record, contract=profile)
    return record


def generate(pair_set, *, run_id='synthetic-test', repetitions=3,
             layers=None, profile=None):
    validate_pair_set(pair_set)
    layers = layers or (profile['expected_layers'] if profile else EXPECTED_DSA_LAYERS)
    rank_count = len(profile['devices']) if profile else 8
    responses, by_rank = [], {rank: [] for rank in range(rank_count)}
    sequence = 0
    for pair in pair_set['pairs']:
        for repetition in range(repetitions):
            for variant in ('event', 'control'):
                item = pair[variant]
                request_id = f'synthetic-{sequence:05d}'
                sequence += 1
                response = {
                    'schema_version': SCHEMA_VERSION,
                    'run_id': run_id,
                    'pair_id': pair['pair_id'],
                    'workload_id': pair['workload_id'],
                    'episode_id': pair['episode_id'],
                    'context_target': pair['context_target'],
                    'event_type': pair['event_type'],
                    'variant': variant,
                    'repetition': repetition,
                    'boundary_position': item['boundary_position'],
                    'request_id': request_id,
                    'started_ns': sequence * 10,
                    'finished_ns': sequence * 10 + 1,
                    'prompt_tokens': item['prompt_tokens_expected'],
                    'prompt_token_ids_sha256': item['prompt_token_ids_sha256_expected'],
                    'completion_tokens': 32,
                    'signature': {'content': 'synthetic', 'reasoning': None,
                                  'token_ids': [1], 'finish_reason': 'length'},
                }
                if profile:
                    response.update({
                        'model_profile': profile['name'],
                        'model_id': profile['model_id'],
                        'model_revision': profile['model_revision'],
                    })
                validate_response(response)
                responses.append(response)
                boundary = item['boundary_position']
                rank = (boundary // 128) % rank_count
                for layer_index in range(layers):
                    layer = f'model.layers.{layer_index}'
                    source = (
                        'computed'
                        if not profile or layer_index < profile['computed_layers']
                        else 'reused'
                    )
                    before = make_trace_record(
                        run_id=run_id, request_id=request_id, rank=rank,
                        layer=layer, position=boundary - 1, raw_values=[0, 1],
                        request_context_len=item['prompt_tokens_expected'],
                        profile=profile, selection_source=source,
                    )
                    after_ids = [128, 129] if variant == 'event' else [0, 1]
                    after = make_trace_record(
                        run_id=run_id, request_id=request_id, rank=rank,
                        layer=layer, position=boundary, raw_values=after_ids,
                        request_context_len=item['prompt_tokens_expected'],
                        profile=profile, selection_source=source,
                    )
                    by_rank[rank].extend((before, after))
    return responses, by_rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--trace-dir', type=Path, required=True)
    parser.add_argument('--responses', type=Path, required=True)
    parser.add_argument('--run-id', default='synthetic-test')
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--layers', type=int)
    parser.add_argument('--root', type=Path, default=Path('/workspace/memecho'))
    parser.add_argument('--model-profile', default=DEFAULT_PROFILE)
    args = parser.parse_args()
    if args.trace_dir.exists() or args.responses.exists():
        raise FileExistsError('Refusing to overwrite synthetic trace artifacts')
    pair_set = json.loads(args.pairs.read_text())
    profile = load_profile(
        args.model_profile, args.root / 'm0a/model_profiles.json'
    )
    responses, by_rank = generate(
        pair_set, run_id=args.run_id, repetitions=args.repetitions,
        layers=args.layers, profile=profile,
    )
    args.trace_dir.mkdir(parents=True)
    for rank, records in by_rank.items():
        if records:
            (args.trace_dir / f'rank{rank}.jsonl').write_text(
                ''.join(json.dumps(record) + '\n' for record in records)
            )
    args.responses.parent.mkdir(parents=True, exist_ok=True)
    args.responses.write_text(
        ''.join(json.dumps(response) + '\n' for response in responses)
    )
    print(f'Generated {len(responses)} responses and '
          f'{sum(map(len, by_rank.values()))} trace rows')


if __name__ == '__main__':
    main()
