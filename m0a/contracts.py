"""Versioned, dependency-free contracts for M0-A experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
TRACE_SCHEMA_VERSION = 2
EVENT_TYPES = {
    "tool_call",
    "tool_result",
    "memory_write",
    "memory_consolidation",
    "memory_supersession",
    "task_switch",
}
VARIANTS = {"event", "control"}
RAW_INDEX_UNIT = "compressed_kv_token_position"
INVALID_SENTINEL = -1
COMPRESSED_BLOCK_SIZE = 128
COMPRESSION_RATIO = 4
SELECTED_WIDTH = 512
EXPECTED_DSA_LAYERS = 21
TRACE_INDEX_UNITS = {"compressed_kv_token_position", "kv_token_position"}
SELECTION_SOURCES = {"computed", "reused"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_api_request_id(engine_request_id: str,
                           api_request_ids: set[str]) -> str | None:
    """Map vLLM's suffixed engine request ID to its public API request ID."""
    if engine_request_id in api_request_ids:
        return engine_request_id
    matches = [
        request_id for request_id in api_request_ids
        if engine_request_id.startswith(request_id + '-')
        and re.fullmatch(r'[0-9A-Za-z]+', engine_request_id[len(request_id) + 1:])
    ]
    if len(matches) > 1:
        raise ValueError(f'Ambiguous engine request ID: {engine_request_id}')
    return matches[0] if matches else None


def _required(record: dict[str, Any], fields: tuple[str, ...], kind: str) -> None:
    missing = [field for field in fields if field not in record]
    if missing:
        raise ValueError(f"{kind} missing fields: {', '.join(missing)}")


def _schema(record: dict[str, Any], kind: str) -> None:
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported {kind} schema_version: {record.get('schema_version')!r}"
        )


def validate_pair_set(data: dict[str, Any]) -> None:
    _schema(data, "pair set")
    _required(data, ("tokenizer_json_sha256", "pairs"), "pair set")
    if not isinstance(data["pairs"], list) or not data["pairs"]:
        raise ValueError("Pair set must contain at least one pair")
    seen = set()
    for pair in data["pairs"]:
        _required(
            pair,
            ("pair_id", "workload_id", "episode_id", "context_target",
             "event_type", "event", "control"),
            "pair",
        )
        if pair["pair_id"] in seen:
            raise ValueError(f"Duplicate pair_id: {pair['pair_id']}")
        seen.add(pair["pair_id"])
        if pair["event_type"] not in EVENT_TYPES:
            raise ValueError(f"Unknown event type: {pair['event_type']}")
        for variant in VARIANTS:
            item = pair[variant]
            _required(
                item,
                ("prompt", "boundary_position", "prompt_tokens_expected",
                 "prompt_token_ids_sha256_expected"),
                f"{pair['pair_id']} {variant}",
            )
            if item["boundary_position"] < 1:
                raise ValueError(f"Invalid boundary in {pair['pair_id']} {variant}")
        event_tokens = pair["event"]["prompt_tokens_expected"]
        control_tokens = pair["control"]["prompt_tokens_expected"]
        if abs(event_tokens - control_tokens) > 2:
            raise ValueError(f"Pair token difference exceeds two: {pair['pair_id']}")
        if pair["event"]["boundary_position"] != pair["control"]["boundary_position"]:
            raise ValueError(f"Event/control boundary mismatch: {pair['pair_id']}")


def validate_response(record: dict[str, Any]) -> None:
    _schema(record, "response")
    _required(
        record,
        ("run_id", "pair_id", "workload_id", "episode_id", "context_target",
         "event_type", "variant", "repetition", "boundary_position",
         "request_id", "prompt_tokens", "prompt_token_ids_sha256",
         "started_ns", "finished_ns", "signature"),
        "response",
    )
    if record["event_type"] not in EVENT_TYPES:
        raise ValueError(f"Unknown response event type: {record['event_type']}")
    if record["variant"] not in VARIANTS:
        raise ValueError(f"Unknown response variant: {record['variant']}")
    if record["finished_ns"] < record["started_ns"]:
        raise ValueError("Response finish timestamp precedes start")


def trace_blocks(record: dict[str, Any]) -> list[int]:
    """Return model-neutral logical block IDs from either trace schema."""
    if record.get("schema_version") == 1:
        return record["logical_compressed_block_ids"]
    return record["logical_block_ids"]


def validate_trace(record: dict[str, Any],
                   contract: dict[str, Any] | None = None) -> None:
    version = record.get("schema_version")
    if version not in {SCHEMA_VERSION, TRACE_SCHEMA_VERSION}:
        raise ValueError(f"Unsupported trace schema_version: {version!r}")
    _required(
        record,
        ("run_id", "request_id", "rank", "layer", "prompt_position",
         "context_len", "request_context_len", "raw_index_unit",
         "invalid_sentinel", "compression_ratio", "compressed_block_size",
         "selected_width", "raw_selected_ids",
         "timestamp_ns"),
        "trace",
    )
    if version == 1:
        _required(record, ("logical_compressed_block_ids",), "trace v1")
    else:
        _required(
            record,
            ("model_profile", "model_id", "model_revision",
             "attention_backend", "native_operator", "selection_source",
             "logical_block_size", "logical_block_ids"),
            "trace v2",
        )
    if not record["run_id"] or not record["request_id"] or not record["layer"]:
        raise ValueError("Trace identity fields must be non-empty")
    if record["rank"] < 0 or record["prompt_position"] < 0:
        raise ValueError("Trace rank and prompt position must be non-negative")
    if not isinstance(record["timestamp_ns"], int) or record["timestamp_ns"] <= 0:
        raise ValueError("Trace timestamp must be a positive integer")
    if record["context_len"] != record["prompt_position"] + 1:
        raise ValueError("Context/position mismatch")
    if record["request_context_len"] < record["context_len"]:
        raise ValueError("Request context is shorter than traced position")
    if record["raw_index_unit"] not in TRACE_INDEX_UNITS:
        raise ValueError("Unexpected raw index unit")
    if record["invalid_sentinel"] != INVALID_SENTINEL:
        raise ValueError("Unexpected invalid sentinel")
    if not isinstance(record["compression_ratio"], int) or record["compression_ratio"] < 1:
        raise ValueError("Unexpected compression ratio")
    if not isinstance(record["compressed_block_size"], int) or record["compressed_block_size"] < 1:
        raise ValueError("Unexpected logical block size")
    raw = record["raw_selected_ids"]
    blocks = trace_blocks(record)
    width = record["selected_width"]
    if not isinstance(width, int) or width < 1:
        raise ValueError("Unexpected selected-set width")
    if version == 1 and width != SELECTED_WIDTH:
        raise ValueError("Unexpected legacy selected-set width")
    if contract is not None and width != contract["selected_width"]:
        raise ValueError(
            "Trace selected_width differs from model profile: "
            f"{width!r} != {contract['selected_width']!r}"
        )
    if len(raw) != width or len(blocks) != width:
        raise ValueError("Incomplete native selected ID list")
    if any(not isinstance(value, int) or value < INVALID_SENTINEL for value in raw):
        raise ValueError("Unexpected native selected ID")
    compressed_context = (
        record["context_len"] + record["compression_ratio"] - 1
    ) // record["compression_ratio"]
    if any(value >= compressed_context for value in raw if value >= 0):
        raise ValueError("Native index exceeds causal compressed context")
    expected = [
        value // record["compressed_block_size"] if value >= 0 else INVALID_SENTINEL
        for value in raw
    ]
    if blocks != expected:
        raise ValueError("Block mapping does not match raw output")
    if version == 1:
        if record["raw_index_unit"] != RAW_INDEX_UNIT:
            raise ValueError("Unexpected legacy raw index unit")
        if record["compression_ratio"] != COMPRESSION_RATIO:
            raise ValueError("Unexpected legacy compression ratio")
        if record["compressed_block_size"] != COMPRESSED_BLOCK_SIZE:
            raise ValueError("Unexpected legacy compressed block size")
    else:
        if record["selection_source"] not in SELECTION_SOURCES:
            raise ValueError("Unexpected selection source")
        if record["logical_block_size"] != record["compressed_block_size"]:
            raise ValueError("Trace logical block size fields disagree")
        if not all(isinstance(record[field], str) and record[field]
                   for field in ("model_profile", "model_id", "model_revision",
                                 "attention_backend", "native_operator")):
            raise ValueError("Trace v2 model identity fields must be non-empty")
    if contract is not None:
        expected_contract = {
            "raw_index_unit": contract["raw_index_unit"],
            "compression_ratio": contract["compression_ratio"],
            "compressed_block_size": contract["logical_block_size"],
            "selected_width": contract["selected_width"],
        }
        if version != contract["trace_schema_version"]:
            raise ValueError("Trace schema differs from model profile")
        if version == TRACE_SCHEMA_VERSION:
            expected_contract.update({
                "model_profile": contract["name"],
                "model_id": contract["model_id"],
                "model_revision": contract["model_revision"],
                "attention_backend": contract["attention_backend"],
            })
        for field, expected_value in expected_contract.items():
            if record.get(field) != expected_value:
                raise ValueError(
                    f"Trace {field} differs from model profile: "
                    f"{record.get(field)!r} != {expected_value!r}"
                )
        if (version == TRACE_SCHEMA_VERSION
                and record["native_operator"] not in contract["native_operators"]):
            raise ValueError("Trace native operator differs from model profile")


def validate_run_manifest(record: dict[str, Any]) -> None:
    _schema(record, "run manifest")
    _required(
        record,
        ("run_id", "mode", "status", "created_at", "root", "trace_dir",
         "responses", "commands", "artifact_hashes"),
        "run manifest",
    )
    if record["mode"] not in {"baseline", "pilot", "pairs"}:
        raise ValueError(f"Unknown run mode: {record['mode']}")
    if record["status"] not in {"planned", "running", "passed", "failed"}:
        raise ValueError(f"Unknown run status: {record['status']}")


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{number}: {error}") from error
