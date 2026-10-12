"""Retokenize the archived τ³ tool-event prompts for the pinned GLM model.

This file seals *inputs*. It contains no generated continuations or timings.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from .echo_workload import sha_ids

REVISION = "20aae340d157bd2215b9d81165be87e94686dfdf"
EXPECTED_FILES = {
    "config.json": (2365, "09d6d0b4b1131cfc79e280bd71a38f2700b433d868d34d02e92c5e56900d2cdd"),
    "tokenizer.json": (20217719, "44556920e4507367e1615610a11aabf215ed123be7ec45ff217891b1b5a6a21a"),
    "model.safetensors": (1243574976, "51bba6b768ddcdb97b7595d0b9a3c18cb82c4f8efa77e2f7924c4d1d46cb3aed"),
}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model(model_dir: Path) -> None:
    for name, (size, digest) in EXPECTED_FILES.items():
        path = model_dir / name
        if not path.is_file() or path.stat().st_size != size or sha_file(path) != digest:
            raise ValueError(f"Pinned GLM model file mismatch: {name}")


def common_prefix(a: list[int], b: list[int]) -> int:
    return next((i for i, (left, right) in enumerate(zip(a, b)) if left != right), min(len(a), len(b)))


def glm_messages(messages: list[dict]) -> list[dict]:
    """Convert OpenAI tool-call JSON argument strings for GLM's chat template."""
    converted = copy.deepcopy(messages)
    for message in converted:
        for call in message.get("tool_calls") or []:
            function = call.get("function", call)
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                parsed = json.loads(arguments)
                if not isinstance(parsed, dict):
                    raise ValueError("Tool-call arguments must decode to an object")
                function["arguments"] = parsed
    return converted


def seal(pairs_path: Path, model_dir: Path, *, tokenizer=None) -> dict:
    source_hash = sha_file(pairs_path)
    source = json.loads(pairs_path.read_text())
    pairs = source.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 48:
        raise ValueError("Expected 48 archived τ³ pairs")
    if tokenizer is None:
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(model_dir / "tokenizer.json"),
            chat_template=(model_dir / "chat_template.jinja").read_text(),
        )
    cases = []
    identities = set()
    neutral_message = {"role": "user", "content": "Please continue the current task using the context above. " * 4}
    for pair in pairs:
        if pair.get("event_type") not in {"tool_call", "tool_result", "memory_write",
                                          "memory_consolidation", "memory_supersession", "task_switch"}:
            raise ValueError("Unexpected event type in sealed workload")
        event = pair["event"]
        control = pair["control"]
        def encode(messages):
            result = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            if hasattr(result, "get"):
                result = result["input_ids"]
            if hasattr(result, "tolist"):
                result = result.tolist()
            return result

        original_event = encode(glm_messages(event["messages"]))
        original_control = encode(glm_messages(control["messages"]))
        original_boundary = common_prefix(original_event, original_control)
        event_ids = encode(glm_messages(event["messages"]) + [neutral_message])
        control_ids = encode(glm_messages(control["messages"]) + [neutral_message])
        if not isinstance(event_ids, list) or not all(type(x) is int for x in event_ids):
            raise ValueError("GLM tokenizer did not return integer token IDs")
        boundary = common_prefix(event_ids, control_ids)
        if event_ids[:original_boundary] != original_event[:original_boundary] or \
                boundary != original_boundary:
            raise ValueError(f"Neutral suffix changed the event boundary: {pair['pair_id']}")
        domain = "retail" if "retail" in pair["workload_id"] else "banking"
        episode = (domain, pair["episode_id"])
        if episode in identities:
            raise ValueError("Episode is reused")
        identities.add(episode)
        if not 0 < boundary < len(event_ids) - 32:
            raise ValueError(f"Event leaves fewer than 32 prefill tokens: {pair['pair_id']} "
                             f"boundary={boundary} event={len(event_ids)} control={len(control_ids)}")
        if len(event_ids) > 65536:
            raise ValueError("GLM prompt exceeds model context")
        cases.append({
            "case_id": pair["pair_id"] + ":event",
            "domain": domain,
            "episode_id": pair["episode_id"],
            "source_trace_id": pair.get("source_trace_id"),
            "event_type": pair["event_type"],
            "event_position": boundary,
            "input_extension": neutral_message,
            "original_prompt_token_ids_sha256": sha_ids(original_event),
            "prompt_token_ids": event_ids,
            "prompt_token_ids_sha256": sha_ids(event_ids),
            "prompt_tokens": len(event_ids),
        })
    if Counter(case["domain"] for case in cases) != {"retail": 24, "banking": 24}:
        raise ValueError("Expected 24 independent episodes in each domain")
    return {
        "schema_version": 1,
        "model_revision": REVISION,
        "source_pairs_sha256": source_hash,
        "tokenizer_json_sha256": EXPECTED_FILES["tokenizer.json"][1],
        "prototype_hbm_shadow": True,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    verify_model(args.model)
    document = seal(args.pairs, args.model)
    args.output.write_text(json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"cases": len(document["cases"]), "output_sha256": sha_file(args.output),
                      "min_tokens": min(x["prompt_tokens"] for x in document["cases"]),
                      "max_tokens": max(x["prompt_tokens"] for x in document["cases"])}))


if __name__ == "__main__":
    main()
