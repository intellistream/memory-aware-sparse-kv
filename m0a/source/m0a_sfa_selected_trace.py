"""Read-only selected-set capture for the GLM SFA prefill path."""

import json
import os
import time

from vllm_ascend import envs


def parse_positions(value: str) -> tuple[tuple[int, int], ...]:
    ranges = []
    for item in value.split(","):
        if not item:
            continue
        start, separator, end = item.partition(":")
        if not separator:
            end = start
        first, last = int(start), int(end)
        if first < 0 or last < first:
            raise ValueError(f"Invalid M0-A position range: {item}")
        ranges.append((first, last))
    return tuple(ranges)


def _selected_rows(positions, local_start: int, row_count: int, ranges):
    rows = []
    for row in range(row_count):
        global_row = local_start + row
        if global_row >= len(positions):
            break
        position = int(positions[global_row])
        if any(first <= position <= last for first, last in ranges):
            rows.append((row, position))
    return rows


def _required_env(name: str) -> str:
    value = getattr(envs, name)
    if not value:
        raise RuntimeError(f"M0-A trace needs non-empty {name}")
    return value


def record_sfa_prefill_selected(
    topk,
    metadata,
    *,
    layer_name: str,
    rank: int,
    selection_source: str,
) -> None:
    """Copy configured SFA rows and return without changing ``topk``."""
    trace_dir = envs.VLLM_ASCEND_M0A_TRACE_DIR
    if not trace_dir:
        return
    if selection_source not in {"computed", "reused"}:
        raise RuntimeError(f"Unexpected selection source: {selection_source}")
    if not metadata.trace_request_ids:
        return
    if len(metadata.trace_request_ids) != 1:
        raise RuntimeError("M0-A selected-set tracing requires concurrency=1")
    positions = metadata.trace_positions_cpu
    seq_lens = metadata.trace_seq_lens_cpu
    if positions is None or seq_lens is None:
        raise RuntimeError("M0-A SFA trace is missing CPU request metadata")
    ranges = parse_positions(envs.VLLM_ASCEND_M0A_TRACE_POSITIONS)
    if not ranges:
        raise RuntimeError("M0-A trace needs prompt-position ranges")
    if topk.ndim < 2:
        raise RuntimeError(f"Unexpected native selected-set shape: {tuple(topk.shape)}")

    local_start = 0
    if metadata.dsa_cp_context is not None:
        local_start = metadata.dsa_cp_context.local_start
    rows = _selected_rows(positions, local_start, topk.shape[0], ranges)
    if not rows:
        return

    run_id = _required_env("VLLM_ASCEND_M0A_RUN_ID")
    model_profile = _required_env("VLLM_ASCEND_M0A_MODEL_PROFILE")
    model_id = _required_env("VLLM_ASCEND_M0A_MODEL_ID")
    model_revision = _required_env("VLLM_ASCEND_M0A_MODEL_REVISION")
    request_id = metadata.trace_request_ids[0]
    request_context_len = int(seq_lens[0])
    block_size = int(metadata.block_size)
    native_operator = (
        "torch_npu.npu_lightning_indexer"
        if selection_source == "computed"
        else "index_cache_reuse"
    )

    records = []
    groups = []
    for row, position in rows:
        if groups and row == groups[-1][-1][0] + 1:
            groups[-1].append((row, position))
        else:
            groups.append([(row, position)])
    for group in groups:
        first_row, last_row = group[0][0], group[-1][0]
        copied = topk[first_row : last_row + 1].detach().cpu()
        for row, position in group:
            raw_ids = [int(value) for value in copied[row - first_row].reshape(-1).tolist()]
            logical_blocks = [
                value // block_size if value >= 0 else -1 for value in raw_ids
            ]
            records.append(
                {
                    "schema_version": 2,
                    "run_id": run_id,
                    "request_id": request_id,
                    "model_profile": model_profile,
                    "model_id": model_id,
                    "model_revision": model_revision,
                    "attention_backend": "sfa",
                    "native_operator": native_operator,
                    "selection_source": selection_source,
                    "rank": rank,
                    "layer": layer_name,
                    "prompt_position": position,
                    "context_len": position + 1,
                    "request_context_len": request_context_len,
                    "raw_index_unit": "kv_token_position",
                    "invalid_sentinel": -1,
                    "compression_ratio": 1,
                    "compressed_block_size": block_size,
                    "logical_block_size": block_size,
                    "selected_width": len(raw_ids),
                    "raw_selected_ids": raw_ids,
                    "logical_block_ids": logical_blocks,
                    "timestamp_ns": time.time_ns(),
                }
            )
    os.makedirs(trace_dir, exist_ok=True)
    path = os.path.join(trace_dir, f"rank{rank}.jsonl")
    rendered = "".join(
        json.dumps(record, separators=(",", ":")) + "\n" for record in records
    )
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(rendered)
