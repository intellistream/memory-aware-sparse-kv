"""Create an explicitly equal-length synthetic corpus for chunk-shape control."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path

try:
    from .contracts import validate_pair_set
    from .import_workloads import common_prefix_length, make_chat_tokenizer, sha_ids
    from .model_profiles import (load_profile, resolve_profile_path,
                                 validate_pair_policy)
except ImportError:
    from contracts import validate_pair_set
    from import_workloads import common_prefix_length, make_chat_tokenizer, sha_ids
    from model_profiles import (load_profile, resolve_profile_path,
                                validate_pair_policy)


FILLERS = ("x", " x", ".", " .", "\n", " ok", " neutral")


def equalize(source, *, chat_ids, profile):
    validate_pair_set(source)
    result = copy.deepcopy(source)
    changes = []
    for pair in result['pairs']:
        original_boundary = pair['event']['boundary_position']
        ids = {variant: chat_ids(pair[variant]['prompt'])
               for variant in ('event', 'control')}
        for variant in ('event', 'control'):
            item = pair[variant]
            if (len(ids[variant]) != item['prompt_tokens_expected']
                    or sha_ids(ids[variant])
                    != item['prompt_token_ids_sha256_expected']):
                raise ValueError(
                    f'Pinned tokenizer contract changed for '
                    f'{pair["pair_id"]} {variant}'
                )
        if len(ids['event']) != len(ids['control']):
            shorter = min(ids, key=lambda name: len(ids[name]))
            target = max(map(len, ids.values()))
            original_prompt = pair[shorter]['prompt']
            chosen = None
            for count in (1, 2):
                for parts in itertools.product(FILLERS, repeat=count):
                    suffix = ''.join(parts)
                    candidate = chat_ids(original_prompt + suffix)
                    if len(candidate) == target:
                        chosen = (suffix, candidate)
                        break
                if chosen:
                    break
            if chosen is None:
                raise ValueError(f'Cannot equalize {pair["pair_id"]}')
            suffix, candidate_ids = chosen
            pair[shorter]['prompt'] = original_prompt + suffix
            ids[shorter] = candidate_ids
            changes.append({
                'pair_id': pair['pair_id'],
                'variant': shorter,
                'suffix': suffix,
                'tokens_added': target - pair[shorter]['prompt_tokens_expected'],
            })
        boundary = common_prefix_length(ids['event'], ids['control'])
        if boundary != original_boundary:
            raise ValueError(
                f'Equalization changed boundary for {pair["pair_id"]}: '
                f'{original_boundary} -> {boundary}'
            )
        for variant in ('event', 'control'):
            pair[variant]['prompt_tokens_expected'] = len(ids[variant])
            pair[variant]['prompt_token_ids_sha256_expected'] = sha_ids(ids[variant])
    result['source_pair_sha256'] = hashlib.sha256(
        json.dumps(source, sort_keys=True).encode()
    ).hexdigest()
    result['length_equalization'] = {
        'scope': 'synthetic engineering prevalidation only',
        'reason': 'hold chunk shape constant for exact prefix selected-set checks',
        'changes': changes,
    }
    validate_pair_set(result)
    validate_pair_policy(result, profile)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/workspace/memecho'))
    parser.add_argument('--model-profile', required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    profile = load_profile(
        args.model_profile, args.root / 'm0a/model_profiles.json'
    )
    model_dir = resolve_profile_path(args.root, profile, 'model_path')
    chat_ids = make_chat_tokenizer(
        profile_name=args.model_profile,
        tokenizer_path=model_dir / 'tokenizer.json', model_dir=model_dir,
    )
    source = json.loads(args.input.read_text())
    result = equalize(source, chat_ids=chat_ids, profile=profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False) + '\n')
    print(json.dumps({
        'pairs': len(result['pairs']),
        'changed': len(result['length_equalization']['changes']),
        'changes': result['length_equalization']['changes'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
