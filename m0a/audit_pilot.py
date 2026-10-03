"""Create a reproducible post-run audit of M0-A pilot artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .contracts import (EXPECTED_DSA_LAYERS, SCHEMA_VERSION, load_jsonl,
                            resolve_api_request_id, sha256_file, validate_trace)
    from .validate_pilot import validate as validate_pilot
except ImportError:
    from contracts import (EXPECTED_DSA_LAYERS, SCHEMA_VERSION, load_jsonl,
                           resolve_api_request_id, sha256_file, validate_trace)
    from validate_pilot import validate as validate_pilot


def signature(record):
    return {key: record.get(key) for key in
            ('content', 'reasoning', 'token_ids', 'finish_reason')}


def audit(root: Path, run_dir: Path):
    manifest = json.loads((run_dir / 'run_manifest.json').read_text())
    response = json.loads((run_dir / 'pilot_response.json').read_text())
    run_id = manifest['run_id']
    validation = validate_pilot(
        run_dir / 'pilot_response.json', run_dir / 'traces', run_id=run_id
    )
    baseline = [json.loads(line) for line in
                (root / 'results/sanity_repeat.jsonl').read_text().splitlines()]
    output_matches = response['signature'] == signature(baseline[0])

    api_ids = {response['request_id'], response['profile_request_id']}
    rows_by_rank = Counter()
    rows_by_api = Counter()
    engines_by_api = defaultdict(set)
    positions_by_api = defaultdict(set)
    layers_by_api = defaultdict(set)
    invalid_by_api = Counter()
    valid_widths = []
    raw_min = None
    raw_max = None
    total_rows = 0
    for path in sorted((run_dir / 'traces').glob('rank*.jsonl')):
        rank_from_name = int(path.stem.removeprefix('rank'))
        for record in load_jsonl(path):
            validate_trace(record)
            if record['run_id'] != run_id or record['rank'] != rank_from_name:
                raise ValueError(f'Run/rank linkage mismatch in {path}')
            api_id = resolve_api_request_id(record['request_id'], api_ids)
            if api_id is None:
                raise ValueError(f'Unlinked engine request ID: {record["request_id"]}')
            total_rows += 1
            rows_by_rank[rank_from_name] += 1
            rows_by_api[api_id] += 1
            engines_by_api[api_id].add(record['request_id'])
            positions_by_api[api_id].add(record['prompt_position'])
            layers_by_api[api_id].add(record['layer'])
            valid = [value for value in record['raw_selected_ids'] if value >= 0]
            invalid_by_api[api_id] += len(record['raw_selected_ids']) - len(valid)
            valid_widths.append(len(valid))
            if valid:
                raw_min = min(valid) if raw_min is None else min(raw_min, min(valid))
                raw_max = max(valid) if raw_max is None else max(raw_max, max(valid))

    log = (run_dir / 'server.log').read_text()
    file_hashes = {}
    for path in sorted(run_dir.rglob('*')):
        if path.is_file() and path.name not in {'artifact_audit.json', 'SHA256SUMS'}:
            file_hashes[str(path.relative_to(run_dir))] = sha256_file(path)
    false_negative = manifest['status'] == 'failed' and validation[
        'pilot_output_matches_stage0'
    ]
    return {
        'schema_version': SCHEMA_VERSION,
        'run_id': run_id,
        'orchestration_status': manifest['status'],
        'posthoc_validation': 'passed',
        'orchestration_false_negative': false_negative,
        'false_negative_reason': (
            'original validator required exact API/engine request ID equality and '
            'assumed 20 DSA layers; runtime uses a suffixed engine ID and 21 layers'
            if false_negative else None
        ),
        'response': {
            'stage0_signature_match_recomputed': output_matches,
            'prompt_tokens': response['usage']['prompt_tokens'],
            'completion_tokens': response['usage']['completion_tokens'],
            'profile_prompt_tokens': response['profile_prompt_tokens'],
            'api_request_ids': sorted(api_ids),
        },
        'trace': {
            'total_rows': total_rows,
            'profile_rows': validation['trace_rows'],
            'rows_by_rank': dict(sorted(rows_by_rank.items())),
            'rows_by_api_request': dict(rows_by_api),
            'engine_request_ids_by_api': {
                key: sorted(value) for key, value in engines_by_api.items()
            },
            'positions_by_api_request': {
                key: sorted(value) for key, value in positions_by_api.items()
            },
            'dsa_layers_by_api_request': {
                key: len(value) for key, value in layers_by_api.items()
            },
            'expected_dsa_layers': EXPECTED_DSA_LAYERS,
            'invalid_padding_entries_by_api': dict(invalid_by_api),
            'valid_ids_per_row': {'min': min(valid_widths), 'max': max(valid_widths)},
            'valid_raw_id_range': [raw_min, raw_max],
            'native_contract_validated_for_every_row': True,
        },
        'server': {
            'application_started': 'Application startup complete.' in log,
            'http_200_chat_requests': log.count('POST /v1/chat/completions HTTP/1.1" 200 OK'),
            'error_lines': sum('ERROR' in line for line in log.splitlines()),
            'critical_lines': sum('CRITICAL' in line for line in log.splitlines()),
        },
        'validation': validation,
        'artifact_sha256': file_hashes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/workspace/memecho'))
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    output = args.output or args.run_dir / 'artifact_audit.json'
    if output.exists() and not args.overwrite:
        raise FileExistsError(f'Refusing to overwrite {output}')
    result = audit(args.root, args.run_dir)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({
        'run_id': result['run_id'],
        'posthoc_validation': result['posthoc_validation'],
        'total_rows': result['trace']['total_rows'],
        'output_match': result['response']['stage0_signature_match_recomputed'],
    }, sort_keys=True))


if __name__ == '__main__':
    main()
