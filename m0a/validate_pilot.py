"""Fail closed on pilot coverage, output identity, and native semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .contracts import (EXPECTED_DSA_LAYERS, SCHEMA_VERSION, load_jsonl,
                            resolve_api_request_id, validate_trace)
    from .model_profiles import DEFAULT_PROFILE, load_profile
except ImportError:
    from contracts import (EXPECTED_DSA_LAYERS, SCHEMA_VERSION, load_jsonl,
                           resolve_api_request_id, validate_trace)
    from model_profiles import DEFAULT_PROFILE, load_profile


DEFAULT_ROOT = Path('/workspace/memecho')


def validate(response_path: Path, trace_dir: Path, *, run_id: str,
             expected_layers: int | None = None, profile=None):
    if profile is None:
        profile = {
            'name': 'deepseek_v4',
            'model_id': 'Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp',
            'model_revision': 'a037dcf99003c6a46cb12b30d83ffb5519e5fa3f',
            'attention_backend': 'dsa_cp',
            'native_operators': ['npu_vllm_quant_lightning_indexer'],
            'raw_index_unit': 'compressed_kv_token_position',
            'compression_ratio': 4,
            'logical_block_size': 128,
            'selected_width': 512,
            'expected_layers': EXPECTED_DSA_LAYERS,
            'computed_layers': EXPECTED_DSA_LAYERS,
            'reused_layers': 0,
            'pilot_kind': 'stage0_replay',
        }
    expected_layers = expected_layers or profile['expected_layers']
    response = json.loads(response_path.read_text())
    if response.get('schema_version') != SCHEMA_VERSION:
        raise RuntimeError('Pilot response schema mismatch')
    if response.get('run_id') != run_id:
        raise RuntimeError('Pilot response run ID mismatch')
    if response.get('model_profile', profile['name']) != profile['name']:
        raise RuntimeError('Pilot response model profile mismatch')
    if profile['pilot_kind'] == 'stage0_replay':
        output_match = response.get('matches_stage0_32_token_baseline') is True
        if response.get('profile_prompt_tokens') != 1029:
            raise RuntimeError('Pilot profile prompt tokenization changed')
        expected_positions = set(range(0, 13)) | set(range(1018, 1029))
    else:
        output_match = response.get('matches_model_baseline') is True
        expected_positions = set(response.get('profile_expected_positions', []))
        # Model pilots configure 0:12 to exercise the recorder on the short
        # output-equivalence request. The same global ranges also apply to the
        # long profile request. Older pilot responses recorded only its event
        # boundary, so include the orchestrator's fixed short range here.
        expected_positions.update(response.get(
            'short_trace_positions', range(0, 13)
        ))
        if not expected_positions:
            raise RuntimeError('Model pilot has no expected profile positions')
    if not output_match:
        raise RuntimeError('Trace-enabled output differs from trace-off baseline')

    request_id = response['profile_request_id']
    api_request_ids = {response['request_id'], request_id}
    records = []
    for path in trace_dir.glob('rank*.jsonl'):
        rank_from_name = int(path.stem.removeprefix('rank'))
        for record in load_jsonl(path):
            validate_trace(record, contract=profile)
            if record['run_id'] != run_id:
                raise RuntimeError('Pilot trace run ID mismatch')
            api_request_id = resolve_api_request_id(
                record['request_id'], api_request_ids
            )
            if api_request_id is None:
                raise RuntimeError(
                    f'Unlinked pilot engine request ID: {record["request_id"]}'
                )
            if record['rank'] != rank_from_name:
                raise RuntimeError(f'Rank filename mismatch: {path}')
            if api_request_id == request_id:
                records.append(record)
    if not records:
        raise RuntimeError('No native profile selected-set rows captured')

    by_position = {}
    ranks_by_position = {}
    engine_request_ids = set()
    layer_sources = {}
    native_operators = set()
    for row in records:
        engine_request_ids.add(row['request_id'])
        by_position.setdefault(row['prompt_position'], set()).add(row['layer'])
        ranks_by_position.setdefault(row['prompt_position'], set()).add(row['rank'])
        source = row.get('selection_source', 'computed')
        previous = layer_sources.setdefault(row['layer'], source)
        if previous != source:
            raise RuntimeError(f'Layer selection source changed: {row["layer"]}')
        native_operators.add(
            row.get('native_operator', 'npu_vllm_quant_lightning_indexer')
        )
    covered = {position: len(layers) for position, layers in by_position.items()}
    if set(covered) != expected_positions:
        raise RuntimeError(
            f'Pilot position coverage mismatch: expected {sorted(expected_positions)}, '
            f'found {sorted(covered)}'
        )
    if any(layer_count != expected_layers for layer_count in covered.values()):
        raise RuntimeError(
            f'Expected {expected_layers} selected-set layers at every position: '
            f'{covered}'
        )
    computed = sum(source == 'computed' for source in layer_sources.values())
    reused = sum(source == 'reused' for source in layer_sources.values())
    if (computed, reused) != (profile['computed_layers'], profile['reused_layers']):
        raise RuntimeError(
            f'Pilot selection-source coverage mismatch: {computed}/{reused}'
        )
    if any(len(ranks) != 1 for ranks in ranks_by_position.values()):
        raise RuntimeError(f'Pilot position appears on multiple ranks: {ranks_by_position}')
    if len(engine_request_ids) != 1:
        raise RuntimeError(f'Expected one engine request ID: {engine_request_ids}')
    result = {
        'schema_version': SCHEMA_VERSION,
        'pilot_output_matches_baseline': True,
        'run_id': run_id,
        'model_profile': profile['name'],
        'model_id': profile['model_id'],
        'model_revision': profile['model_revision'],
        'profile_request_id': request_id,
        'engine_request_ids': sorted(engine_request_ids),
        'trace_rows': len(records),
        'computed_layers': computed,
        'reused_layers': reused,
        'native_operators': sorted(native_operators),
        'layers_by_position': covered,
        'ranks_by_position': {
            position: sorted(ranks) for position, ranks in ranks_by_position.items()
        },
    }
    if profile['pilot_kind'] == 'stage0_replay':
        result['pilot_output_matches_stage0'] = True
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--response', type=Path)
    parser.add_argument('--trace-dir', type=Path)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--model-profile', default=DEFAULT_PROFILE)
    parser.add_argument('--expected-layers', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    response = args.response or args.root / 'm0a/pilot_response.json'
    trace_dir = args.trace_dir or args.root / 'm0a/traces/pilot'
    profile = load_profile(
        args.model_profile, args.root / 'm0a/model_profiles.json'
    )
    result = validate(
        response, trace_dir, run_id=args.run_id,
        expected_layers=args.expected_layers, profile=profile,
    )
    rendered = json.dumps(result, sort_keys=True)
    if args.output:
        if args.output.exists():
            raise FileExistsError(f'Refusing to overwrite {args.output}')
        args.output.write_text(rendered + '\n')
    print(rendered)


if __name__ == '__main__':
    main()
