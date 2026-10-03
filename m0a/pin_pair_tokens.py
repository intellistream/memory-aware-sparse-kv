"""Pin exact single-user chat-template token IDs for existing pairs."""

import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer

from contracts import validate_pair_set

root = Path('/workspace/memecho')
path = root / 'm0a/pairs.json'
data = json.loads(path.read_text())
validate_pair_set(data)
tokenizer_path = root / 'models/DeepSeek-V4-Flash-w8a8-mtp/tokenizer.json'
if hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() != data['tokenizer_json_sha256']:
    raise RuntimeError('Tokenizer hash mismatch')
tokenizer = Tokenizer.from_file(str(tokenizer_path))
for pair in data['pairs']:
    for variant in ('event', 'control'):
        item = pair[variant]
        token_ids = [0, 128803, *tokenizer.encode(item['prompt'],
                     add_special_tokens=False).ids, 128804, 128822]
        if len(token_ids) != item['prompt_tokens_expected']:
            raise RuntimeError(f'Prompt count mismatch: {pair["pair_id"]}')
        item['prompt_token_ids_sha256_expected'] = hashlib.sha256(
            json.dumps(token_ids).encode()
        ).hexdigest()
path.write_text(json.dumps(data) + '\n')
print(len(data['pairs']), 'pairs pinned')
