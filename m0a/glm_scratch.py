"""Exact 16-token scratch mapping for the GLM shadow-KV prototype."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

PAGE_SIZE = 16
CAPACITY_TOKENS = 3072
PREFETCH_TOKENS = 512


@dataclass
class Transfer:
    prefetched: tuple[int, ...]
    recalled: tuple[int, ...]
    remapped: tuple[int, ...]
    useful_prefetch: tuple[int, ...]
    wasted_prefetch: tuple[int, ...]


class ScratchKV:
    """Pack arbitrary native tokens into dense pages and preserve top-k order."""

    def __init__(self, host_key, host_rope, device_key, device_rope):
        if host_key.device.type != "cpu" or host_rope.device.type != "cpu":
            raise ValueError("KV source must be on CPU")
        if device_key.device.type == "npu" and (not host_key.is_pinned() or not host_rope.is_pinned()):
            raise ValueError("NPU scratch requires pinned CPU KV source")
        if device_key.shape[:2] != (CAPACITY_TOKENS // PAGE_SIZE, PAGE_SIZE):
            raise ValueError("Expected 192 scratch pages of 16 tokens")
        if device_rope.shape[:2] != device_key.shape[:2]:
            raise ValueError("Key and RoPE scratch pages differ")
        if host_key.shape[0] != host_rope.shape[0] or host_key.shape[1:] != device_key.shape[2:] or \
                host_rope.shape[1:] != device_rope.shape[2:]:
            raise ValueError("Host and scratch KV payload shape mismatch")
        self.host_key, self.host_rope = host_key, host_rope
        self.device_key, self.device_rope = device_key, device_rope
        self.native_to_slot: dict[int, int] = {}
        self.slot_to_native: dict[int, int] = {}
        self.priority: dict[int, tuple[float, int]] = {}
        self.free_slots = list(reversed(range(CAPACITY_TOKENS)))
        self.age = 0

    def _copy_batch(self, pending: list[tuple[int, int]]) -> None:
        if not pending:
            return
        import torch
        native_ids = torch.tensor([native_id for native_id, _slot in pending], dtype=torch.long)
        slots = torch.tensor([slot for _native_id, slot in pending], dtype=torch.long,
                             device=self.device_key.device)
        key = self.host_key.index_select(0, native_ids).to(self.device_key.device, non_blocking=True)
        rope = self.host_rope.index_select(0, native_ids).to(self.device_rope.device, non_blocking=True)
        self.device_key.reshape(CAPACITY_TOKENS, *self.device_key.shape[2:]).index_copy_(0, slots, key)
        self.device_rope.reshape(CAPACITY_TOKENS, *self.device_rope.shape[2:]).index_copy_(0, slots, rope)

    def _victims(self, protected: set[int]) -> list[int]:
        # A phase can add at most 2048 demand units or 512 prefetch units to a
        # 3072-unit pool. Existing unprotected units therefore suffice; sort
        # once instead of scanning all resident slots for each inserted token.
        return [native_id for _priority, native_id in sorted(
            ((self.priority[native_id], native_id) for native_id in self.native_to_slot
             if native_id not in protected), reverse=True)]

    def _insert(self, native_id: int, score: float, victims: list[int],
                pending: list[tuple[int, int]]) -> bool:
        if native_id in self.native_to_slot:
            return False
        if self.free_slots:
            slot = self.free_slots.pop()
        else:
            if not victims:
                raise ValueError("Selected KV exceeds scratch capacity")
            old = victims.pop()
            slot = self.native_to_slot[old]
            del self.native_to_slot[old], self.slot_to_native[slot], self.priority[old]
        pending.append((native_id, slot))
        self.native_to_slot[native_id] = slot
        self.slot_to_native[slot] = native_id
        self.age += 1
        self.priority[native_id] = (float(score), self.age)
        return True

    def prepare(self, native_selected: list[int], full_scores: dict[int, float],
                candidates: list[int], *, prefetch: bool) -> Transfer:
        def has_score(native_id):
            return native_id in full_scores if isinstance(full_scores, Mapping) else native_id < len(full_scores)

        valid = [native_id for native_id in native_selected if native_id != -1]
        if len(valid) != len(set(valid)) or len(valid) > CAPACITY_TOKENS:
            raise ValueError("Invalid native selected set")
        if any(type(native_id) is not int or native_id < 0 or
               native_id >= self.host_key.shape[0] or not has_score(native_id)
               for native_id in valid):
            raise ValueError("Native selected ID lacks a valid full score or KV")
        if any(not has_score(native_id) for native_id in candidates):
            raise ValueError("Prefetch candidate lacks full score")
        predicted = []
        pending = []
        if prefetch:
            victims = self._victims(set(candidates[:PREFETCH_TOKENS]))
            for native_id in candidates:
                if len(predicted) == PREFETCH_TOKENS:
                    break
                if self._insert(native_id, full_scores[native_id], victims, pending):
                    predicted.append(native_id)
            self._copy_batch(pending)
        present_after_prefetch = set(self.native_to_slot)
        protected = set(valid)
        recalled = []
        pending = []
        victims = self._victims(protected)
        for native_id in valid:
            if self._insert(native_id, full_scores[native_id], victims, pending):
                recalled.append(native_id)
        self._copy_batch(pending)
        if any(native_id not in self.native_to_slot for native_id in valid):
            raise AssertionError("Guaranteed recall missed a native selected ID")
        remapped = tuple(self.native_to_slot[native_id] if native_id >= 0 else -1
                         for native_id in native_selected)
        if len({value for value in remapped if value >= 0}) != len(valid):
            raise AssertionError("Scratch remap is not injective")
        useful = tuple(native_id for native_id in predicted
                       if native_id in protected and native_id in present_after_prefetch)
        wasted = tuple(native_id for native_id in predicted if native_id not in useful)
        return Transfer(tuple(predicted), tuple(recalled), remapped, useful, wasted)

    def assert_payload_exact(self, native_selected: list[int]) -> None:
        import torch
        ids = [native_id for native_id in native_selected if native_id >= 0]
        if not ids:
            return
        slots = [self.native_to_slot[native_id] for native_id in ids]
        ids_tensor = torch.tensor(ids, dtype=torch.long)
        slots_tensor = torch.tensor(slots, dtype=torch.long, device=self.device_key.device)
        expected_key = self.host_key.index_select(0, ids_tensor).to(self.device_key.device)
        expected_rope = self.host_rope.index_select(0, ids_tensor).to(self.device_rope.device)
        actual_key = self.device_key.reshape(CAPACITY_TOKENS, *self.device_key.shape[2:]).index_select(0, slots_tensor)
        actual_rope = self.device_rope.reshape(CAPACITY_TOKENS, *self.device_rope.shape[2:]).index_select(0, slots_tensor)
        if not torch.equal(expected_key, actual_key) or not torch.equal(expected_rope, actual_rope):
            raise AssertionError("Scratch KV payload differs from pinned CPU source")

    def block_table(self):
        import torch
        return torch.arange(CAPACITY_TOKENS // PAGE_SIZE, dtype=torch.int32,
                            device=self.device_key.device).view(1, -1)
