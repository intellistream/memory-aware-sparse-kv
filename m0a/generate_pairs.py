"""Construct same-prefix single-boundary prefill pairs with pinned tokenizer."""

import json
import hashlib
from pathlib import Path

from tokenizers import Tokenizer

from contracts import SCHEMA_VERSION

ROOT = Path('/workspace/memecho')
TOKENIZER = Tokenizer.from_file(str(ROOT / 'models/DeepSeek-V4-Flash-w8a8-mtp/tokenizer.json'))
TOKENIZER_SHA256 = hashlib.sha256(
    (ROOT / 'models/DeepSeek-V4-Flash-w8a8-mtp/tokenizer.json').read_bytes()
).hexdigest()
OUTPUT = ROOT / 'm0a/pairs.json'
EVENTS = {
    'tool_call': '[tool_call] assistant invokes search with query: project schedule and owner.',
    'tool_result': '[tool_result] search returns: the project schedule moved to Tuesday.',
    'memory_write': '[memory_write] Store the user preference: concise weekly summaries.',
    'memory_consolidation': '[memory_consolidation] Merge repeated preferences into one stable summary.',
    'memory_supersession': '[memory_supersession] Replace the old Monday schedule with Tuesday.',
    'task_switch': '[task_switch] Stop the schedule review and begin an expense review.',
}
BASE = (
    'Conversation history for one continuing assistant task. '
    'Read the archive and then follow the latest line.\n'
)
NOTE = 'Archive note: the catalog records a routine observation about maps and dates.\n'
POST = '\nAssistant: Continue the active task and state the next relevant fact.\n'


def ids(value):
    return TOKENIZER.encode(value, add_special_tokens=False).ids


def prompt_hash(value):
    token_ids = [0, 128803, *ids(value), 128804, 128822]
    return hashlib.sha256(json.dumps(token_ids).encode()).hexdigest()


def boundary(prefix, prompt):
    a, b = ids(prefix), ids(prompt)
    index = 0
    while index < min(len(a), len(b)) and a[index] == b[index]:
        index += 1
    return index + 2  # BOS and user-role marker in the pinned chat template.


def neutral_line(target_count, prefix):
    base = '[routine_note] A normal archive entry confirms the same ongoing review.'
    candidates = [base]
    for _ in range(80):
        candidates.append(candidates[-1] + ' detail')
    return min(candidates, key=lambda line: abs(len(ids(prefix + line + POST)) - target_count))


def main():
    pairs = []
    note_size = len(ids(NOTE))
    for context_target in (8192, 32768):
        repeat_count = max(1, (context_target - 150) // note_size)
        prefix = BASE + NOTE * repeat_count + 'Current exchange:\n'
        while len(ids(prefix + max(EVENTS.values(), key=len) + POST)) + 4 < context_target - 40:
            prefix += NOTE
        while len(ids(prefix + max(EVENTS.values(), key=len) + POST)) + 4 > context_target:
            prefix = prefix[:-len(NOTE)]
        for event_name, event_line in EVENTS.items():
            event_prompt = prefix + event_line + POST
            event_size = len(ids(event_prompt))
            control_line = neutral_line(event_size, prefix)
            control_prompt = prefix + control_line + POST
            pair = {
                'pair_id': f'{context_target}_{event_name}',
                'workload_id': 'synthetic_smoke',
                'episode_id': f'{context_target}_{event_name}',
                'context_target': context_target,
                'event_type': event_name,
                'event': {
                    'prompt': event_prompt,
                    'boundary_position': boundary(prefix, event_prompt),
                    'prompt_tokens_expected': event_size + 4,
                    'prompt_token_ids_sha256_expected': prompt_hash(event_prompt),
                },
                'control': {
                    'prompt': control_prompt,
                    'boundary_position': boundary(prefix, control_prompt),
                    'prompt_tokens_expected': len(ids(control_prompt)) + 4,
                    'prompt_token_ids_sha256_expected': prompt_hash(control_prompt),
                },
            }
            assert abs(pair['event']['prompt_tokens_expected'] - pair['control']['prompt_tokens_expected']) <= 2
            pairs.append(pair)
    OUTPUT.write_text(json.dumps({'schema_version': SCHEMA_VERSION,
                                  'tokenizer_json_sha256': TOKENIZER_SHA256,
                                  'pairs': pairs}) + '\n')
    for pair in pairs:
        print(pair['pair_id'], pair['event']['boundary_position'],
              pair['event']['prompt_tokens_expected'],
              pair['control']['boundary_position'],
              pair['control']['prompt_tokens_expected'])


if __name__ == '__main__':
    main()
