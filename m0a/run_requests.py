"""Run model baselines, trace pilots, or sequential matched requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

try:
    from .contracts import SCHEMA_VERSION, validate_pair_set, validate_response
    from .model_profiles import (DEFAULT_PROFILE, load_profile,
                                 resolve_profile_path, validate_pair_policy)
except ImportError:
    from contracts import SCHEMA_VERSION, validate_pair_set, validate_response
    from model_profiles import (DEFAULT_PROFILE, load_profile,
                                resolve_profile_path, validate_pair_policy)


DEFAULT_ROOT = Path('/workspace/memecho')
DEFAULT_URL = 'http://127.0.0.1:8900/v1/chat/completions'
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request(payload, *, url=DEFAULT_URL):
    body = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        url, data=body, headers={'Content-Type': 'application/json'}, method='POST'
    )
    with OPENER.open(req, timeout=900) as response:
        if response.status != 200:
            raise RuntimeError(f'HTTP {response.status}')
        return json.load(response)


def signature(result):
    choice = result['choices'][0]
    message = choice.get('message', {})
    return {
        'content': message.get('content'),
        'reasoning': message.get('reasoning'),
        'token_ids': choice.get('token_ids'),
        'finish_reason': choice.get('finish_reason'),
    }


def prompt_hash(result):
    ids = result.get('prompt_token_ids') or []
    return len(ids), hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def payload_for(profile, prompt):
    return {
        'model': profile['served_model_name'],
        'messages': prompt if isinstance(prompt, list) else [{'role': 'user', 'content': prompt}],
        'temperature': 0,
        'seed': 0,
        'max_tokens': 32,
        'return_token_ids': True,
    }


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f'Refusing to overwrite {path}')
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_pairs(path: Path, profile):
    pair_set = json.loads(path.read_text())
    validate_pair_set(pair_set)
    pinned = pair_set.get('model_profile')
    if pinned is None and profile['name'] != 'deepseek_v4':
        raise ValueError('Non-DeepSeek pair set is missing model_profile')
    if pinned is not None and pinned != profile['name']:
        raise ValueError(f'Pair set profile {pinned} != requested {profile["name"]}')
    validate_pair_policy(pair_set, profile)
    return pair_set


def run_baseline(output, *, profile, run_id, url, repetitions=3,
                 overwrite=False):
    _prepare_output(output, overwrite)
    records = []
    for repetition in range(repetitions):
        started_ns = time.time_ns()
        result = request(payload_for(profile, 'Who are you?'), url=url)
        records.append({
            'repetition': repetition,
            'request_id': result['id'],
            'started_ns': started_ns,
            'finished_ns': time.time_ns(),
            'usage': result.get('usage'),
            'signature': signature(result),
        })
    signatures = [record['signature'] for record in records]
    data = {
        'schema_version': SCHEMA_VERSION,
        'kind': 'model_baseline',
        'run_id': run_id,
        'model_profile': profile['name'],
        'model_id': profile['model_id'],
        'model_revision': profile['model_revision'],
        'trace_enabled': False,
        'repetitions': repetitions,
        'identical': all(item == signatures[0] for item in signatures[1:]),
        'records': records,
    }
    output.write_text(json.dumps(data, ensure_ascii=False) + '\n')
    if not data['identical']:
        raise RuntimeError('Trace-off model baseline is not repeatable')
    print(json.dumps({key: value for key, value in data.items()
                      if key != 'records'}))


def _run_stage0_pilot(output, *, root, profile, run_id, url, overwrite):
    _prepare_output(output, overwrite)
    payload = json.loads((root / 'results/sanity_request.json').read_text())
    baseline = [json.loads(line) for line in
                (root / 'results/sanity_repeat.jsonl').read_text().splitlines()]
    result = request(payload, url=url)
    expected = {key: baseline[0][key] for key in signature(result)}
    record = {
        'schema_version': SCHEMA_VERSION,
        'run_id': run_id,
        'model_profile': profile['name'],
        'model_id': profile['model_id'],
        'model_revision': profile['model_revision'],
        'request_id': result['id'],
        'usage': result.get('usage'),
        'signature': signature(result),
        'matches_stage0_32_token_baseline': signature(result) == expected,
    }
    profile_payload = json.loads(
        (root / 'profiles/baseline_torch/profile_request.json').read_text()
    )
    profile_result = request(profile_payload, url=url)
    record['profile_request_id'] = profile_result['id']
    record['profile_prompt_tokens'] = profile_result.get('usage', {}).get('prompt_tokens')
    record['profile_signature'] = signature(profile_result)
    output.write_text(json.dumps(record, ensure_ascii=False) + '\n')
    print(json.dumps({key: value for key, value in record.items()
                      if key not in {'signature', 'profile_signature'}}))
    if not record['matches_stage0_32_token_baseline']:
        raise RuntimeError('Trace-enabled output differs from Stage 0 acceptance')


def _run_model_pilot(output, *, profile, pairs_path, baseline_path, run_id,
                     url, overwrite):
    _prepare_output(output, overwrite)
    baseline = json.loads(baseline_path.read_text())
    if baseline.get('model_profile') != profile['name'] or not baseline.get('identical'):
        raise RuntimeError('Pilot baseline is absent, unstable, or belongs to another model')
    short = request(payload_for(profile, 'Who are you?'), url=url)
    expected = baseline['records'][0]['signature']

    pair_set = _load_pairs(pairs_path, profile)
    pair = pair_set['pairs'][0]
    item = pair['event']
    profile_result = request(payload_for(profile, item['prompt']), url=url)
    actual_count = profile_result.get('usage', {}).get('prompt_tokens')
    id_count, actual_hash = prompt_hash(profile_result)
    if actual_count != item['prompt_tokens_expected']:
        raise RuntimeError(
            f'Pilot token count changed: {actual_count} != '
            f'{item["prompt_tokens_expected"]}'
        )
    if id_count != actual_count or actual_hash != item['prompt_token_ids_sha256_expected']:
        raise RuntimeError('Pilot prompt token IDs differ from pinned pair')
    boundary = item['boundary_position']
    short_trace_positions = list(range(0, 13))
    expected_positions = sorted(set(short_trace_positions) | set(
        range(max(0, boundary - 4), min(actual_count - 1, boundary + 8) + 1)
    ))
    record = {
        'schema_version': SCHEMA_VERSION,
        'run_id': run_id,
        'model_profile': profile['name'],
        'model_id': profile['model_id'],
        'model_revision': profile['model_revision'],
        'request_id': short['id'],
        'usage': short.get('usage'),
        'signature': signature(short),
        'matches_model_baseline': signature(short) == expected,
        'profile_request_id': profile_result['id'],
        'profile_pair_id': pair['pair_id'],
        'profile_boundary_position': boundary,
        'short_trace_positions': short_trace_positions,
        'profile_expected_positions': expected_positions,
        'profile_prompt_tokens': actual_count,
        'profile_prompt_token_ids_sha256': actual_hash,
        'profile_signature': signature(profile_result),
    }
    output.write_text(json.dumps(record, ensure_ascii=False) + '\n')
    print(json.dumps({key: value for key, value in record.items()
                      if key not in {'signature', 'profile_signature'}}))
    if not record['matches_model_baseline']:
        raise RuntimeError('Trace-enabled output differs from trace-off model baseline')


def run_pilot(output, *, root, profile, pairs_path, baseline_path, run_id,
              url, overwrite=False):
    if profile['pilot_kind'] == 'stage0_replay':
        _run_stage0_pilot(
            output, root=root, profile=profile, run_id=run_id, url=url,
            overwrite=overwrite,
        )
    else:
        _run_model_pilot(
            output, profile=profile, pairs_path=pairs_path,
            baseline_path=baseline_path, run_id=run_id, url=url,
            overwrite=overwrite,
        )


def run_pairs(output, *, profile, pairs_path, repetitions, run_id, url,
              overwrite=False):
    _prepare_output(output, overwrite)
    pair_set = _load_pairs(pairs_path, profile)
    with output.open('w', encoding='utf-8') as stream:
        for pair in pair_set['pairs']:
            for repetition in range(repetitions):
                order = ('event', 'control') if repetition % 2 == 0 else ('control', 'event')
                for variant in order:
                    item = pair[variant]
                    started_ns = time.time_ns()
                    response = request(payload_for(profile, item['prompt']), url=url)
                    actual_count = response.get('usage', {}).get('prompt_tokens')
                    id_count, actual_hash = prompt_hash(response)
                    if actual_count != item['prompt_tokens_expected']:
                        raise RuntimeError(
                            f'Tokenization changed for {pair["pair_id"]}: '
                            f'{actual_count} vs {item["prompt_tokens_expected"]}'
                        )
                    if id_count != actual_count or (
                        actual_hash != item['prompt_token_ids_sha256_expected']
                    ):
                        raise RuntimeError(
                            f'Prompt token IDs differ from pinned pair: {pair["pair_id"]}'
                        )
                    row = {
                        'schema_version': SCHEMA_VERSION,
                        'run_id': run_id,
                        'model_profile': profile['name'],
                        'model_id': profile['model_id'],
                        'model_revision': profile['model_revision'],
                        'pair_id': pair['pair_id'],
                        'workload_id': pair['workload_id'],
                        'episode_id': pair['episode_id'],
                        'context_target': pair['context_target'],
                        'event_type': pair['event_type'],
                        'variant': variant,
                        'repetition': repetition,
                        'boundary_position': item['boundary_position'],
                        'request_id': response['id'],
                        'started_ns': started_ns,
                        'finished_ns': time.time_ns(),
                        'prompt_tokens': actual_count,
                        'prompt_token_ids_sha256': actual_hash,
                        'completion_tokens': response.get('usage', {}).get('completion_tokens'),
                        'signature': signature(response),
                    }
                    validate_response(row)
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                    stream.flush()
                    print(row['pair_id'], variant, repetition, row['request_id'],
                          row['prompt_tokens'], flush=True)


def default_run_id(mode: str) -> str:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    return f'm0a_{mode}_{stamp}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('baseline', 'pilot', 'pairs'))
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--model-profile', default=DEFAULT_PROFILE)
    parser.add_argument('--pairs', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--run-id')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--url', default=DEFAULT_URL)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    profile = load_profile(
        args.model_profile, args.root / 'm0a/model_profiles.json'
    )
    run_id = args.run_id or default_run_id(args.mode)
    output_name = {
        'baseline': 'baseline_response.json',
        'pilot': 'pilot_response.json',
        'pairs': 'pair_responses.jsonl',
    }[args.mode]
    output = args.output or args.root / 'm0a/runs' / run_id / output_name
    pairs = args.pairs or resolve_profile_path(args.root, profile, 'default_pairs')
    baseline = args.baseline or resolve_profile_path(args.root, profile, 'baseline_file')
    if args.mode == 'baseline':
        run_baseline(
            output, profile=profile, run_id=run_id, url=args.url,
            repetitions=args.repetitions, overwrite=args.overwrite,
        )
    elif args.mode == 'pilot':
        run_pilot(
            output, root=args.root, profile=profile, pairs_path=pairs,
            baseline_path=baseline, run_id=run_id, url=args.url,
            overwrite=args.overwrite,
        )
    else:
        run_pairs(
            output, profile=profile, pairs_path=pairs,
            repetitions=args.repetitions, run_id=run_id, url=args.url,
            overwrite=args.overwrite,
        )


if __name__ == '__main__':
    main()
