"""Seal and verify a reused τ³ pair set before starting the Pod worker."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import sha256_file, validate_pair_set
from .deepseek_validation import utc, write_json
from .run_requests import request
from .tau3_reuse import regenerate, reuse_sealed_pairs, verify_snapshot
from .tau3_workload import DOMAINS, EVENTS, TARGETS
from .working_set import require

FILES = ('pairs.json', 'event-audit.json', 'context-feasibility.json',
         'source-lineage.json', 'episodes.json', 'collection-summary.json',
         'rejected-episodes.json', 'tau3-provenance.json')


def check_pairs(directory: Path, tokenizer_sha256: str, *, live_tokens: bool = True) -> dict:
    pairs = json.loads((directory / 'pairs.json').read_text())
    require(pairs['tokenizer_json_sha256'] == tokenizer_sha256, 'Precheck tokenizer hash mismatch')
    validate_pair_set(pairs)
    rows = pairs['pairs']
    require(len(rows) == 48 and len({p['pair_id'] for p in rows}) == 48,
            'Precheck requires 48 distinct pairs')
    checks = 0
    for domain in DOMAINS:
        selected = [p for p in rows if p['workload_id'] == 'tau3_v1.0.1_' + domain]
        require(len(selected) == 24, 'Precheck domain pair count mismatch: ' + domain)
        used = {task for p in selected for task in p['history_task_ids']}
        require(len(used) >= 48, 'Precheck has fewer than 48 distinct domain tasks: ' + domain)
        for target in TARGETS:
            group = [p for p in selected if p['context_target'] == target]
            require(len(group) == 12 and {p['event_type'] for p in group} == set(EVENTS),
                    'Precheck event coverage incomplete')
            for kind in EVENTS:
                chains = [p for p in group if p['event_type'] == kind]
                require(len(chains) == 2 and not set(chains[0]['history_task_ids']) &
                        set(chains[1]['history_task_ids']), 'Precheck task chains overlap')
        first = {task for p in selected if p['context_target'] == TARGETS[0]
                 for task in p['history_task_ids']}
        second = {task for p in selected if p['context_target'] == TARGETS[1]
                  for task in p['history_task_ids']}
        require(not first & second, 'Precheck 8K/32K tasks overlap')
    for pair in rows:
        for variant in ('event', 'control'):
            item = pair[variant]
            ids = item['prompt_token_ids']
            require(isinstance(ids, list) and all(type(value) is int for value in ids),
                    'Precheck missing token IDs')
            from .import_workloads import sha_ids
            require(len(ids) == item['prompt_tokens_expected'] and
                    sha_ids(ids) == item['prompt_token_ids_sha256_expected'],
                    'Precheck sealed token IDs are inconsistent')
            if live_tokens:
                actual = request({'model': 'dsv4', 'messages': item['messages'],
                                  'add_generation_prompt': True},
                                 url='http://127.0.0.1:8900/tokenize')
                require(actual['tokens'] == ids,
                        'Precheck live tokenizer differs: ' + pair['pair_id'] + '/' + variant)
            checks += 1
    require(checks == 96, 'Precheck did not verify all 96 prompts')
    return {'pairs': len(rows), 'tokenized_prompts': checks,
            'domain_distinct_tasks': {domain: len({task for p in rows
                if p['workload_id'] == 'tau3_v1.0.1_' + domain
                for task in p['history_task_ids']}) for domain in DOMAINS}}


def verify_seal(directory: Path, tokenizer_sha256: str, *, live_tokens: bool = True) -> dict:
    directory = Path(directory)
    seal = json.loads((directory / 'tau3-precheck.json').read_text())
    require(seal['status'] == 'passed' and seal['tokenizer_json_sha256'] == tokenizer_sha256,
            'Precheck seal or tokenizer differs')
    require(set(seal['files']) == set(FILES), 'Precheck seal file list differs')
    for name in FILES:
        require(sha256_file(directory / name) == seal['files'][name],
                'Precheck sealed file changed: ' + name)
    source = verify_snapshot(directory / 'source-inputs')
    require(seal['source_input_manifest_sha256'] ==
            sha256_file(directory / 'source-inputs/tau3-source-inputs-manifest.json') and
            source['source_run_id'] == seal['source_run_id'] and
            source['source_final_manifest_sha256'] == seal['source_final_manifest_sha256'],
            'Precheck source snapshot changed')
    if seal.get('sealed_pair_reuse'):
        require(source['old_pairs_sha256'] == sha256_file(directory / 'pairs.json') and
                source['old_pairs_sha256'] == sha256_file(directory / 'source-inputs/old-pairs.json'),
                'Archived sealed pairs changed')
    summary = check_pairs(directory, tokenizer_sha256, live_tokens=live_tokens)
    require(summary == seal['summary'], 'Precheck pair summary changed')
    return seal


def create(directory: Path, tau_root: Path, tokenizer_sha256: str, *, reuse_sealed: bool = False) -> dict:
    directory = Path(directory)
    step = 'source_snapshot'
    try:
        verify_snapshot(directory / 'source-inputs')
        step = 'pair_generation'
        if reuse_sealed:
            reuse_sealed_pairs(directory / 'source-inputs', directory, tau_root, tokenizer_sha256)
        else:
            regenerate(directory / 'source-inputs', directory, tau_root, tokenizer_sha256)
        step = 'tokenizer_and_coverage'
        summary = check_pairs(directory, tokenizer_sha256)
        step = 'seal'
        source = json.loads((directory / 'source-inputs/tau3-source-inputs-manifest.json').read_text())
        seal = {'status': 'passed', 'at': utc(), 'tokenizer_json_sha256': tokenizer_sha256,
                'source_run_id': source['source_run_id'],
                'source_final_manifest_sha256': source['source_final_manifest_sha256'],
                'sealed_pair_reuse': reuse_sealed,
                'archived_pairs_sha256': source['old_pairs_sha256'],
                'source_input_manifest_sha256': sha256_file(
                    directory / 'source-inputs/tau3-source-inputs-manifest.json'),
                'files': {name: sha256_file(directory / name) for name in FILES},
                'summary': summary}
        write_json(directory / 'tau3-precheck.json', seal)
        verify_seal(directory, tokenizer_sha256, live_tokens=False)
        return seal
    except BaseException as error:
        write_json(directory / 'tau3-precheck-failure.json',
                   {'status': 'failed', 'at': utc(), 'step': step, 'error': str(error)})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tau-root', type=Path, required=True)
    parser.add_argument('--tokenizer-sha256', required=True)
    parser.add_argument('--reuse-sealed-pairs', action='store_true')
    args = parser.parse_args()
    print(json.dumps(create(args.output, args.tau_root, args.tokenizer_sha256,
                            reuse_sealed=args.reuse_sealed_pairs)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
