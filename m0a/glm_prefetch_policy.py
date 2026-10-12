"""Vectorized ECHO-style 256-bin prefill and EMA decode candidate selection."""
from __future__ import annotations

import numpy as np

from .glm_scratch import PREFETCH_TOKENS


def prefill_candidates(scores, predicted_kth: float | None, k: int) -> list[int]:
    if predicted_kth is None:
        return []
    delta = (np.asarray(scores, dtype=np.float32) - np.float32(predicted_kth))
    bits = delta.view(np.uint32)
    sortable = np.where((bits & np.uint32(0x80000000)) != 0,
                        ~bits, bits ^ np.uint32(0x80000000))
    bins = (sortable >> np.uint32(24)).astype(np.uint8)
    counts = np.bincount(bins, minlength=256)
    accepted = np.zeros(256, dtype=bool)
    size = 0
    for bin_id in range(255, -1, -1):
        count = int(counts[bin_id])
        if count and size + count > k:
            break
        if count:
            accepted[bin_id] = True
            size += count
    # Preserve the score-stream order. Re-ranking with the exact current
    # scores would turn a bounded prefetch into an oracle top-k subset and
    # mechanically hide false positives from the 256-bin approximation.
    return np.flatnonzero(accepted[bins])[:PREFETCH_TOKENS].tolist()


def decode_candidates(scores, threshold: float | None) -> list[int]:
    if threshold is None:
        return []
    scores = np.asarray(scores)
    return np.flatnonzero(scores > threshold)[:PREFETCH_TOKENS].tolist()


def update_ema(previous: float | None, native_kth: float) -> float:
    return native_kth if previous is None else 0.5 * previous + 0.5 * native_kth
