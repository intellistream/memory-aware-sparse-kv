"""Full BF16 GLM indexer scores, checked against native top-k per query row."""
from __future__ import annotations

import math


def full_indexer_scores(query, keys, weights):
    """Evaluate sum_h weight[h] * ReLU(dot(query[h], key[n])) for every n."""
    import torch
    if query.ndim != 2 or keys.ndim != 2 or weights.ndim != 1 or \
            query.shape[0] != weights.shape[0] or query.shape[1] != keys.shape[1]:
        raise ValueError("GLM indexer tensor dimensions disagree")
    if not query.isfinite().all() or not keys.isfinite().all() or not weights.isfinite().all():
        raise ValueError("Nonfinite indexer input")
    return (torch.relu(query.float() @ keys.float().T) * weights.float()[:, None]).sum(0)


def gather_logical_keys(key_cache, block_table, context_len: int):
    import torch
    if key_cache.ndim != 4 or block_table.ndim != 2 or block_table.shape[0] != 1:
        raise ValueError("Expected paged GLM indexer cache and one request")
    block_size = key_cache.shape[1]
    needed = math.ceil(context_len / block_size)
    blocks = block_table[0, :needed].to(torch.long)
    if needed == 0 or (blocks < 0).any() or (blocks >= key_cache.shape[0]).any():
        raise ValueError("Invalid GLM request block table")
    gathered = torch.index_select(key_cache, 0, blocks).reshape(-1, key_cache.shape[-1])
    return gathered[:context_len]


def verify_rows(query, weights, key_cache, block_table, native_topk,
                *, query_start: int, rows: list[int], tolerance: float = 1e-4):
    """Return complete eligible score tensors only if every selected set agrees."""
    import torch
    if native_topk.shape[0] != query.shape[0] or query.shape[0] != weights.shape[0]:
        raise ValueError("Query and native selection rows disagree")
    if not rows:
        return {}
    context_len = query_start + max(rows) + 1
    keys = gather_logical_keys(key_cache, block_table, context_len)
    verified = {}
    for row in rows:
        if not 0 <= row < query.shape[0]:
            raise ValueError("Query row outside chunk")
        position = query_start + row
        scores = full_indexer_scores(query[row], keys[:position + 1], weights[row])
        selected = native_topk[row].reshape(-1)
        selected = selected[selected >= 0].to(torch.long)
        if selected.numel() != min(native_topk.shape[-1], position + 1) or \
                selected.unique().numel() != selected.numel() or \
                (selected > position).any():
            raise ValueError("Invalid native selected IDs")
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask[selected] = False
        if mask.any() and scores[mask].max() > scores[selected].min() + tolerance:
            raise ValueError(f"Complete GLM scores disagree with native top-k at position {position}")
        verified[position] = scores
    return verified
