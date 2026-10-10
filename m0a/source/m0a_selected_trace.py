"""Research-only, read-only selected-set capture for DSA CP prefill."""

import json
import os
import time

from vllm_ascend import envs


def parse_positions(value: str) -> tuple[tuple[int, int], ...]:
    """Parse inclusive prompt-position ranges."""
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


def _selected_rows(positions, local_start: int, row_count: int, ranges, request_context_len: int):
    rows = []
    for row in range(row_count):
        global_row = local_start + row
        if global_row >= len(positions):
            break
        position = int(positions[global_row])
        # CP rounds a chunk up to its rank count. These padded query rows can
        # still appear in the indexer output but are not request tokens.
        if position < request_context_len and any(first <= position <= last for first, last in ranges):
            rows.append((row, position))
    return rows


def record_prefill_selected(topk, metadata, *, layer_name: str, rank: int,
                            topk_values=None) -> None:
    """Copy only configured output rows; leave the native tensor untouched."""
    trace_dir = envs.VLLM_ASCEND_M0A_TRACE_DIR
    if not trace_dir:
        return
    # Graph warmup has no API request identity and is outside the experiment.
    if not metadata.trace_request_ids:
        return
    if len(metadata.trace_request_ids) != 1:
        raise RuntimeError("M0-A selected-set tracing requires concurrency=1")
    positions = metadata.trace_positions_cpu
    seq_lens = metadata.trace_seq_lens_cpu
    if positions is None or seq_lens is None:
        raise RuntimeError("M0-A trace is missing CPU request metadata")
    ranges = parse_positions(envs.VLLM_ASCEND_M0A_TRACE_POSITIONS)
    if not ranges:
        raise RuntimeError("M0-A trace needs prompt-position ranges")
    if topk.ndim < 2:
        raise RuntimeError(f"Unexpected native selected-set shape: {tuple(topk.shape)}")
    if topk_values is not None and tuple(topk_values.shape) != tuple(topk.shape):
        raise RuntimeError("Native selected scores do not match selected ID shape")

    prompt_lens = getattr(metadata, 'trace_prompt_lens_cpu', None)
    request_context_len = int(prompt_lens[0]) if prompt_lens else int(seq_lens[0])
    if request_context_len <= 0:
        raise RuntimeError("M0-A trace has an invalid request length")
    local_start = metadata.cp_metadata.local_start
    rows = _selected_rows(positions, local_start, topk.shape[0], ranges, request_context_len)
    if not rows:
        return

    request_id = metadata.trace_request_ids[0]
    block_size = metadata.block_size
    run_id = envs.VLLM_ASCEND_M0A_RUN_ID
    if not run_id:
        raise RuntimeError("M0-A trace needs a non-empty run ID")
    records = []
    # Adjacent rows share one device-to-host transfer. Disjoint windows remain
    # separate, so a 32K request never copies the gap between 8K and 32K.
    groups = []
    for row, position in rows:
        if groups and row == groups[-1][-1][0] + 1:
            groups[-1].append((row, position))
        else:
            groups.append([(row, position)])
    for group in groups:
        first_row, last_row = group[0][0], group[-1][0]
        copied = topk[first_row : last_row + 1].detach().cpu()
        copied_values = (topk_values[first_row : last_row + 1].detach().cpu()
                         if topk_values is not None else None)
        for row, position in group:
            raw_ids = [int(value) for value in copied[row - first_row].reshape(-1).tolist()]
            record = {
                    "schema_version": 1,
                    "run_id": run_id,
                    "request_id": request_id,
                    "rank": rank,
                    "cp_world_size": 8,
                    "cp_local_start": int(metadata.cp_metadata.local_start),
                    "cp_local_end": int(metadata.cp_metadata.local_end),
                    "chunk_start_position": int(positions[0]),
                    "chunk_token_count": len(positions),
                    "chunk_context_len": min(int(seq_lens[0]), request_context_len),
                    "query_global_index": local_start + row,
                    "layer": layer_name,
                    "prompt_position": position,
                    "context_len": position + 1,
                    "request_context_len": request_context_len,
                    "raw_index_unit": "compressed_kv_token_position",
                    "invalid_sentinel": -1,
                    "compression_ratio": 4,
                    "compressed_block_size": block_size,
                    "selected_width": len(raw_ids),
                    "raw_selected_ids": raw_ids,
                    "logical_compressed_block_ids": [
                        value // block_size if value >= 0 else -1 for value in raw_ids
                    ],
                    "timestamp_ns": time.time_ns(),
                }
            if copied_values is not None:
                record["score_coverage"] = "topk_only"
                record["native_selected_scores"] = [
                    float(value) for value in
                    copied_values[row - first_row].reshape(-1).tolist()
                ]
            records.append(record)
    os.makedirs(trace_dir, exist_ok=True)
    path = os.path.join(trace_dir, f"rank{rank}.jsonl")
    data = "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records)
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(data)
