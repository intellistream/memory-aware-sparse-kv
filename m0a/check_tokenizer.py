from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(
    "/workspace/memecho/models/DeepSeek-V4-Flash-w8a8-mtp/tokenizer.json"
)
for text in ("Who are you?", "Observation: the record is stable.\n"):
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    print(repr(text), len(ids), ids)
