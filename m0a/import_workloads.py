"""Convert normalized, redacted agent episodes into pinned M0-A matched pairs.

Input JSONL fields:
  workload_id, episode_id, event_type, context_target,
  prefix, event_text, control_text, postfix, source_trace_id

The input is deliberately runtime-neutral. A runtime adapter only needs to emit
these fields after redaction and control construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from .contracts import SCHEMA_VERSION, load_jsonl, validate_pair_set
    from .model_profiles import DEFAULT_PROFILE, load_profile, resolve_profile_path
except ImportError:
    from contracts import SCHEMA_VERSION, load_jsonl, validate_pair_set
    from model_profiles import DEFAULT_PROFILE, load_profile, resolve_profile_path


def sha_ids(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def common_prefix_length(left, right):
    index = 0
    while index < min(len(left), len(right)) and left[index] == right[index]:
        index += 1
    return index


def make_chat_tokenizer(*, profile_name: str, tokenizer_path: Path,
                        model_dir: Path | None):
    if profile_name == 'deepseek_v4':
        try:
            from tokenizers import Tokenizer
        except ImportError as error:
            raise RuntimeError('The tokenizers package is required') from error
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

        def chat_ids(prompt):
            body = tokenizer.encode(prompt, add_special_tokens=False).ids
            return [0, 128803, *body, 128804, 128822]

        return chat_ids

    if model_dir is None:
        raise ValueError('A model directory is required for chat-template tokenization')
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError('The transformers package is required') from error
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False
        )
    except Exception as auto_error:
        tokenizer_config_path = model_dir / 'tokenizer_config.json'
        chat_template_path = model_dir / 'chat_template.jinja'
        if not tokenizer_config_path.is_file() or not chat_template_path.is_file():
            raise RuntimeError('Cannot construct the model tokenizer') from auto_error
        tokenizer_config = json.loads(tokenizer_config_path.read_text())
        if tokenizer_config.get('tokenizer_class') != 'TokenizersBackend':
            raise RuntimeError('Cannot construct the model tokenizer') from auto_error
        try:
            from transformers import TokenizersBackend
            tokenizer = TokenizersBackend(
                tokenizer_file=str(tokenizer_path),
                chat_template=chat_template_path.read_text(),
                eos_token=tokenizer_config.get('eos_token'),
                pad_token=tokenizer_config.get('pad_token'),
                padding_side=tokenizer_config.get('padding_side', 'left'),
                model_max_length=tokenizer_config.get(
                    'model_max_length', 1048576
                ),
                clean_up_tokenization_spaces=tokenizer_config.get(
                    'clean_up_tokenization_spaces', False
                ),
            )
        except Exception as backend_error:
            raise RuntimeError('Cannot construct TokenizersBackend') from backend_error

    def chat_ids(prompt):
        encoded = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            add_generation_prompt=True,
            tokenize=True,
        )
        ids = encoded['input_ids'] if hasattr(encoded, 'keys') else encoded
        return list(ids)

    return chat_ids


def convert(input_path: Path, tokenizer_path: Path, *, context_tolerance=128,
            profile_name=DEFAULT_PROFILE, model_dir: Path | None = None,
            model_id: str | None = None, model_revision: str | None = None):
    chat_ids = make_chat_tokenizer(
        profile_name=profile_name, tokenizer_path=tokenizer_path,
        model_dir=model_dir,
    )

    pairs = []
    for row in load_jsonl(input_path):
        required = (
            'workload_id', 'episode_id', 'event_type', 'context_target',
            'prefix', 'event_text', 'control_text', 'postfix', 'source_trace_id',
        )
        missing = [field for field in required if field not in row]
        if missing:
            raise ValueError(f'Workload row missing fields: {missing}')
        event_prompt = row['prefix'] + row['event_text'] + row['postfix']
        control_prompt = row['prefix'] + row['control_text'] + row['postfix']
        event_ids, control_ids = chat_ids(event_prompt), chat_ids(control_prompt)
        boundary = common_prefix_length(event_ids, control_ids)
        if boundary < 3 or boundary >= min(len(event_ids), len(control_ids)):
            raise ValueError(f'No usable single boundary in episode {row["episode_id"]}')
        if abs(len(event_ids) - len(control_ids)) > 2:
            raise ValueError(f'Event/control token mismatch in {row["episode_id"]}')
        if any(abs(len(ids) - row['context_target']) > context_tolerance
               for ids in (event_ids, control_ids)):
            raise ValueError(f'Context target mismatch in {row["episode_id"]}')
        pair_id = f'{row["workload_id"]}_{row["episode_id"]}_{row["context_target"]}'
        pairs.append({
            'pair_id': pair_id,
            'workload_id': row['workload_id'],
            'episode_id': row['episode_id'],
            'source_trace_id': row['source_trace_id'],
            'context_target': row['context_target'],
            'event_type': row['event_type'],
            'event': {
                'prompt': event_prompt,
                'boundary_position': boundary,
                'prompt_tokens_expected': len(event_ids),
                'prompt_token_ids_sha256_expected': sha_ids(event_ids),
            },
            'control': {
                'prompt': control_prompt,
                'boundary_position': boundary,
                'prompt_tokens_expected': len(control_ids),
                'prompt_token_ids_sha256_expected': sha_ids(control_ids),
            },
        })
    result = {
        'schema_version': SCHEMA_VERSION,
        'model_profile': profile_name,
        'model_id': model_id,
        'model_revision': model_revision,
        'tokenizer_json_sha256': hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        'source': str(input_path),
        'pair_count': len(pairs),
        'pairs': pairs,
    }
    validate_pair_set(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/workspace/memecho'))
    parser.add_argument('--model-profile', default=DEFAULT_PROFILE)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--context-tolerance', type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    profile = load_profile(
        args.model_profile, args.root / 'm0a/model_profiles.json'
    )
    model_dir = args.model_dir or resolve_profile_path(
        args.root, profile, 'model_path'
    )
    tokenizer = args.tokenizer or model_dir / 'tokenizer.json'
    result = convert(
        args.input, tokenizer, context_tolerance=args.context_tolerance,
        profile_name=args.model_profile, model_dir=model_dir,
        model_id=profile['model_id'], model_revision=profile['model_revision'],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False) + '\n')
    print(f'Imported {result["pair_count"]} matched pairs')


if __name__ == '__main__':
    main()
