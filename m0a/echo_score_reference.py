"""Slow full-score DSA diagnostic from explicitly dequantized indexer inputs.

This is a reference for validating a future native score exporter. Its input
layout is explicit; it does not silently reinterpret the deployed paged cache.
"""
from __future__ import annotations


def weighted_relu_scores(query, keys, weights):
    """Return all DSA index scores.

    query: [indexer_heads, head_dim], keys: [native_kv_units, head_dim],
    weights: [indexer_heads]. Inputs must already be dequantized and carry
    the same rotary transformation as the native indexer.
    """
    import torch

    if query.ndim != 2 or keys.ndim != 2 or weights.ndim != 1:
        raise ValueError("Expected query [H,D], keys [N,D], weights [H]")
    if query.shape[0] != weights.shape[0] or query.shape[1] != keys.shape[1]:
        raise ValueError("Indexer head or dimension mismatch")
    if keys.shape[0] < 1:
        raise ValueError("No native keys")
    q = query.to(torch.float32)
    k = keys.to(torch.float32)
    w = weights.to(torch.float32)
    if not torch.isfinite(q).all() or not torch.isfinite(k).all() or not torch.isfinite(w).all():
        raise ValueError("Non-finite indexer input")
    return (torch.relu(q @ k.T) * w[:, None]).sum(dim=0)


def verify_native_topk(scores, native_selected_ids, k: int, *, tolerance: float = 1e-4):
    """Fail if the diagnostic full scores disagree with native selected IDs."""
    import torch

    if scores.ndim != 1 or k <= 0 or not native_selected_ids:
        raise ValueError("Invalid full scores or native selection")
    ids = list(native_selected_ids)
    if len(ids) != len(set(ids)) or len(ids) != min(k, scores.numel()) or any(
        type(native_id) is not int or native_id < 0 or native_id >= scores.numel()
        for native_id in ids
    ):
        raise ValueError("Invalid native selected ID")
    if not torch.isfinite(scores).all():
        raise ValueError("Non-finite full score")
    selected = set(ids)
    if len(selected) < scores.numel():
        selected_floor = min(float(scores[native_id]) for native_id in selected)
        other_ceiling = max(float(scores[native_id]) for native_id in range(scores.numel())
                            if native_id not in selected)
        if other_ceiling > selected_floor + tolerance:
            raise ValueError("Diagnostic scores disagree with native top-k")
    return True
