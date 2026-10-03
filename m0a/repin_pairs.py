"""Re-tokenize an existing text-level pair set for another model profile."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from .contracts import SCHEMA_VERSION, validate_pair_set
    from .import_workloads import common_prefix_length, make_chat_tokenizer, sha_ids
    from .model_profiles import load_profile, resolve_profile_path
except ImportError:
    from contracts import SCHEMA_VERSION, validate_pair_set
    from import_workloads import common_prefix_length, make_chat_tokenizer, sha_ids
    from model_profiles import load_profile, resolve_profile_path


def repin(source, *, chat_ids, profile, tokenizer_path,
          context_tolerance=1024):
    validate_pair_set(source)
    pairs = []
    for original in source['pairs']:
        event_prompt = original['event']['prompt']
        control_prompt = original['control']['prompt']
        event_ids = chat_ids(event_prompt)
        control_ids = chat_ids(control_prompt)
        boundary = common_prefix_length(event_ids, control_ids)
        if boundary < 3 or boundary >= min(len(event_ids), len(control_ids)):
            raise ValueError(f'No usable boundary in {original["pair_id"]}')
        if abs(len(event_ids) - len(control_ids)) > 2:
            raise ValueError(f'Pair token difference exceeds two: {original["pair_id"]}')
        target = original['context_target']
        if any(abs(len(ids) - target) > context_tolerance
               for ids in (event_ids, control_ids)):
            raise ValueError(
                f'Context target mismatch in {original["pair_id"]}: '
                f'{len(event_ids)}/{len(control_ids)} vs {target}'
            )
        pair = {
            key: original[key] for key in (
                'pair_id', 'workload_id', 'episode_id', 'source_trace_id',
                'context_target', 'event_type',
            ) if key in original
        }
        pair['event'] = {
            'prompt': event_prompt,
            'boundary_position': boundary,
            'prompt_tokens_expected': len(event_ids),
            'prompt_token_ids_sha256_expected': sha_ids(event_ids),
        }
        pair['control'] = {
            'prompt': control_prompt,
            'boundary_position': boundary,
            'prompt_tokens_expected': len(control_ids),
            'prompt_token_ids_sha256_expected': sha_ids(control_ids),
        }
        pairs.append(pair)
    result = {
        'schema_version': SCHEMA_VERSION,
        'model_profile': profile['name'],
        'model_id': profile['model_id'],
        'model_revision': profile['model_revision'],
        'tokenizer_json_sha256': hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        'source': source.get('source', 'repinned pair set'),
        'source_pair_sha256': hashlib.sha256(
            json.dumps(source, sort_keys=True).encode()
        ).hexdigest(),
        'pair_count': len(pairs),
        'pairs': pairs,
    }
    validate_pair_set(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/workspace/memecho'))
    parser.add_argument('--model-profile', required=True)
    parser.add_argument('--source-pairs', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--context-tolerance', type=int, default=1024)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    profile = load_profile(
        args.model_profile, args.root / 'm0a/model_profiles.json'
    )
    model_dir = args.model_dir or resolve_profile_path(
        args.root, profile, 'model_path'
    )
    tokenizer_path = model_dir / 'tokenizer.json'
    chat_ids = make_chat_tokenizer(
        profile_name=args.model_profile, tokenizer_path=tokenizer_path,
        model_dir=model_dir,
    )
    source = json.loads(args.source_pairs.read_text())
    result = repin(
        source, chat_ids=chat_ids, profile=profile,
        tokenizer_path=tokenizer_path,
        context_tolerance=args.context_tolerance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False) + '\n')
    print(f'Repinned {len(result["pairs"])} pairs for {args.model_profile}')


if __name__ == '__main__':
    main()
