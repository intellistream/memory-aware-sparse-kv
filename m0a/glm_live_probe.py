"""Opt-in in-process NPU diagnostic at the native GLM indexer call site."""
from __future__ import annotations

import json
from pathlib import Path

_CASE = None
_SEEN = set()


def set_case(case: dict):
    global _CASE
    _CASE = case


def install(path: Path, *, verify_scores: bool) -> None:
    from vllm_ascend.device.device_op import DeviceOperator
    from .glm_live_score import verify_rows

    native = DeviceOperator.indexer_select_post_process

    def wrapped(sfa_impl, q_li, q_li_scale, q_li_shape_ori, weights, kv_cache,
                metadata, actual_seq_lengths_query, actual_seq_lengths_key,
                enable_sparse_li_c8, use_torch_npu_lightning_indexer):
        topk = native(sfa_impl, q_li, q_li_scale, q_li_shape_ori, weights,
                      kv_cache, metadata, actual_seq_lengths_query,
                      actual_seq_lengths_key, enable_sparse_li_c8,
                      use_torch_npu_lightning_indexer)
        if _CASE is None:
            return topk
        key_lengths = actual_seq_lengths_key.detach().cpu().reshape(-1).tolist()
        query_lengths = actual_seq_lengths_query.detach().cpu().reshape(-1).tolist()
        context_len = int(key_lengths[-1])
        query_start = context_len - int(q_li.shape[0])
        layer = str(sfa_impl.layer_name)
        target_start = _CASE["event_position"]
        target_end = target_start + 32
        rows = [position - query_start for position in range(max(target_start, query_start),
                 min(target_end, query_start + q_li.shape[0]))]
        if rows:
            identity = (_CASE["case_id"], layer, query_start)
            if identity not in _SEEN:
                record = {"case_id": _CASE["case_id"], "layer": layer,
                          "query_start": query_start, "query_rows": int(q_li.shape[0]),
                          "context_len": context_len, "target_rows": rows,
                          "query_shape": list(q_li.shape), "weights_shape": list(weights.shape),
                          "topk_shape": list(topk.shape),
                          "indexer_cache_shape": list(kv_cache[2].shape),
                          "block_table_shape": list(metadata.block_table.shape),
                          "actual_seq_lengths_query": query_lengths,
                          "actual_seq_lengths_key": key_lengths,
                          "enable_sparse_li_c8": bool(enable_sparse_li_c8),
                          "use_torch_npu_lightning_indexer": bool(use_torch_npu_lightning_indexer)}
                if verify_scores:
                    checked = verify_rows(q_li, weights, kv_cache[2],
                                          metadata.block_table, topk,
                                          query_start=query_start, rows=rows)
                    record["full_score_rows_verified"] = len(checked)
                with path.open("a") as output:
                    output.write(json.dumps(record) + "\n")
                _SEEN.add(identity)
        return topk

    DeviceOperator.indexer_select_post_process = staticmethod(wrapped)
