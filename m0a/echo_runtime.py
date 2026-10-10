"""Fail-closed ECHO runtime contract and a correctness-only tensor KV pool.

The reference pool can run on CPU or NPU tensors, but it uses Python metadata
and per-unit copies. It is never eligible for ECHO latency/throughput claims.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class Mode(str, Enum):
    NO_OFFLOAD = "no_offload"
    OFFLOAD_NO_PREFETCH = "offload_no_prefetch"
    OFFLOAD_ECHO_PREFETCH = "offload_echo_prefetch"


@dataclass(frozen=True)
class Capabilities:
    native_full_scores: bool = False
    exact_guaranteed_recall: bool = False
    asynchronous_transfer: bool = False
    fused_indexer_prefetch: bool = False
    graph_safe_cache: bool = False
    fixed_continuation: bool = False

    def validate(self, mode: Mode, *, timed: bool) -> None:
        if mode != Mode.NO_OFFLOAD and not self.exact_guaranteed_recall:
            raise RuntimeError("Offload requires exact guaranteed recall")
        if mode == Mode.OFFLOAD_ECHO_PREFETCH and not self.native_full_scores:
            raise RuntimeError("ECHO prefetch requires complete native indexer scores")
        if timed and mode == Mode.OFFLOAD_ECHO_PREFETCH and not (
            self.asynchronous_transfer and self.fused_indexer_prefetch and
            self.graph_safe_cache and self.fixed_continuation
        ):
            raise RuntimeError("Timed ECHO comparison requires fused NPU path and fixed continuation")
        if timed and mode == Mode.OFFLOAD_NO_PREFETCH and not (
            self.asynchronous_transfer and self.graph_safe_cache and self.fixed_continuation
        ):
            raise RuntimeError("Timed offload comparison requires graph-safe NPU path")


class ReferenceKVPool:
    """Per-layer KV slots with exact demand recall and score-priority eviction."""

    timed_performance_eligible = False

    def __init__(self, host_kv, device_slots):
        if host_kv.device.type != "cpu":
            raise ValueError("Reference host KV must reside on CPU")
        if device_slots.ndim != host_kv.ndim or device_slots.shape[1:] != host_kv.shape[1:]:
            raise ValueError("Host and device KV unit shapes differ")
        if device_slots.shape[0] < 1:
            raise ValueError("Device KV capacity must be positive")
        self.host_kv = host_kv
        self.device_slots = device_slots
        self.capacity = int(device_slots.shape[0])
        self.native_to_slot: dict[int, int] = {}
        self.slot_to_native: dict[int, int] = {}
        self.priority: dict[int, tuple[float, int]] = {}
        self.clock = 0
        self.prefetched_units = 0
        self.synchronous_units = 0

    def _validate_ids(self, ids: Iterable[int]) -> list[int]:
        result = list(ids)
        if len(result) != len(set(result)):
            raise ValueError("Duplicate native KV ID")
        if any(type(native_id) is not int or native_id < 0 or
               native_id >= self.host_kv.shape[0] for native_id in result):
            raise ValueError("Native KV ID outside host store")
        return result

    def _slot_for_new(self, protected: set[int]) -> int:
        for slot in range(self.capacity):
            if slot not in self.slot_to_native:
                return slot
        victims = [(self.priority[native_id], native_id, slot)
                   for slot, native_id in self.slot_to_native.items()
                   if native_id not in protected]
        if not victims:
            raise ValueError("Selected native KV set exceeds device pool")
        _, old_id, slot = min(victims)
        del self.native_to_slot[old_id]
        del self.slot_to_native[slot]
        del self.priority[old_id]
        return slot

    def _load(self, native_id: int, priority: float, protected: set[int]) -> bool:
        if native_id in self.native_to_slot:
            return False
        slot = self._slot_for_new(protected)
        # This reference intentionally synchronizes through ordinary PyTorch
        # semantics. A production path must fuse/index/transfer on NPU.
        self.device_slots[slot].copy_(self.host_kv[native_id])
        self.native_to_slot[native_id] = slot
        self.slot_to_native[slot] = native_id
        self.clock += 1
        self.priority[native_id] = (float(priority), self.clock)
        return True

    def prefetch(self, candidates: Iterable[int], scores: dict[int, float],
                 budget_units: int) -> list[int]:
        if budget_units < 0:
            raise ValueError("Negative prefetch budget")
        ids = self._validate_ids(candidates)
        copied: list[int] = []
        for native_id in ids:
            if len(copied) >= budget_units:
                break
            if native_id not in scores:
                raise ValueError("Predicted native KV lacks its index score")
            if self._load(native_id, scores[native_id], set()):
                copied.append(native_id)
        self.prefetched_units += len(copied)
        return copied

    def guaranteed_recall(self, selected: Iterable[int],
                          scores: dict[int, float]) -> list[int]:
        ids = self._validate_ids(selected)
        if len(ids) > self.capacity:
            raise ValueError("Selected native KV set exceeds device pool")
        if any(native_id not in scores for native_id in ids):
            raise ValueError("Selected native KV lacks its index score")
        protected = set(ids)
        recalled: list[int] = []
        for native_id in ids:
            if self._load(native_id, scores[native_id], protected):
                recalled.append(native_id)
        for native_id in ids:
            self.clock += 1
            self.priority[native_id] = (float(scores[native_id]), self.clock)
        self.synchronous_units += len(recalled)
        self.assert_ready(ids)
        return recalled

    def assert_ready(self, selected: Iterable[int]) -> list[int]:
        ids = self._validate_ids(selected)
        slots: list[int] = []
        for native_id in ids:
            slot = self.native_to_slot.get(native_id)
            if slot is None:
                raise AssertionError("Native selected KV is absent after recall")
            if not self.device_slots[slot].cpu().equal(self.host_kv[native_id]):
                raise AssertionError("Recalled KV payload differs from host store")
            slots.append(slot)
        return slots
