"""CPU-only working-set identities, dependency validation, bounded statistics and LRU."""
from __future__ import annotations

import hashlib
import json
from collections import ChainMap, Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

REGION_TYPES = {"system", "user", "assistant", "tool", "memory", "task", "other", "mixed"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def text_id(value):
    return isinstance(value, str) and 0 < len(value) <= 256


def identity(snapshot):
    return (snapshot["session_id"], snapshot["model_id"], snapshot["model_revision"],
            snapshot["tokenizer_sha256"], snapshot["context_version"])


def lane_key(lane):
    return (lane["rank"], lane["layer"], lane["kv_kind"])


def prefix_length(a, b):
    for i, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return i
    return min(len(a), len(b))


def support(snapshot, lane, unit):
    if lane["kv_kind"] == "kv_token_position":
        return (unit, unit + 1, unit)
    if lane.get("support_contract") == "c4_overlap_v1":
        require(integer(unit) and 4*(unit+1) <= len(snapshot["token_ids"]), "Missing C4 support interval")
        return (max(0, 4*unit-4), 4*(unit+1), 4*(unit+1)-1)
    value = lane["support_intervals"].get(str(unit))
    require(value is not None, f"Missing compressor support interval for native KV {unit}")
    return tuple(value)


def support_units(snapshot, lane):
    return range(len(snapshot["token_ids"])//4) if lane.get("support_contract") == "c4_overlap_v1" else map(int, lane["support_intervals"])


def validate_snapshot(snapshot, source_root):
    fields = ("snapshot_id", "session_id", "context_version", "sequence", "model_id",
              "model_revision", "tokenizer_sha256", "token_ids", "generated_tokens",
              "regions", "lanes")
    require(all(k in snapshot for k in fields), "Incomplete snapshot identity/token metadata")
    require(all(text_id(snapshot[k]) for k in fields[:3] + fields[4:7]), "Invalid snapshot identity")
    digest = snapshot["tokenizer_sha256"]
    require(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "Invalid tokenizer SHA256")
    tokens = snapshot["token_ids"]
    require(isinstance(tokens, list) and tokens and all(integer(t) for t in tokens), "Exact token IDs required")
    if "token_ids_sha256" in snapshot:
        require(hashlib.sha256(json.dumps(tokens).encode()).hexdigest() == snapshot["token_ids_sha256"],
                "Snapshot token IDs SHA256 mismatch")
    require(integer(snapshot["sequence"]) and integer(snapshot["generated_tokens"])
            and snapshot["generated_tokens"] <= len(tokens), "Invalid generated token count")
    require(isinstance(snapshot["regions"], list) and snapshot["regions"], "Regions required")
    end = 0
    for region in snapshot["regions"]:
        require(set(("start", "end", "type", "created_at")) <= region.keys(), "Incomplete region")
        require(integer(region["start"]) and integer(region["end"], 1)
                and region["start"] == end and region["start"] < region["end"] <= len(tokens), "Regions must partition exact token sequence")
        require(region["type"] in REGION_TYPES - {"mixed"}, "Unknown region type")
        require(integer(region["created_at"]) and region["created_at"] <= snapshot["sequence"], "Invalid region age")
        end = region["end"]
    require(end == len(tokens), "Regions must cover entire token sequence")
    require(isinstance(snapshot["lanes"], list) and snapshot["lanes"], "KV lanes required")
    seen = set()
    for lane in snapshot["lanes"]:
        require(set(("rank", "layer", "kv_kind", "generated_ids")) <= lane.keys(), "Incomplete KV lane")
        require(integer(lane["rank"]) and text_id(lane["layer"]), "Invalid rank/layer")
        require(lane["kv_kind"] in {"kv_token_position", "compressed_kv_token_position"}, "Unknown KV kind")
        key = lane_key(lane)
        require(key not in seen, "Duplicate KV lane")
        seen.add(key)
        ids = lane["generated_ids"]
        require(isinstance(ids, list) and all(integer(i) for i in ids) and len(ids) == len(set(ids)), "Invalid generated KV IDs")
        if lane["kv_kind"] == "compressed_kv_token_position":
            formula = lane.get("support_contract") == "c4_overlap_v1"
            require(formula or isinstance(lane.get("support_intervals"), dict), "Compressed KV region learning requires support intervals")
            if formula:
                require(snapshot["model_id"] == "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp", "C4 contract model mismatch")
            evidence = lane.get("support_evidence", {})
            require(set(("path", "sha256", "start_line", "end_line", "explanation")) <= evidence.keys(),
                    "Compressed KV requires source-backed compressor support evidence")
            root = Path(source_root).resolve()
            path = (root / evidence["path"]).resolve()
            require(path.is_relative_to(root) and path.is_file(), "Compressor source must be inside source root")
            data = path.read_bytes()
            require(hashlib.sha256(data).hexdigest() == evidence["sha256"], "Compressor source SHA256 mismatch")
            require(integer(evidence["start_line"], 1) and integer(evidence["end_line"], 1)
                    and evidence["start_line"] <= evidence["end_line"] <= len(data.splitlines())
                    and text_id(evidence["explanation"]), "Invalid compressor source citation")
            for unit, interval in lane.get("support_intervals", {}).items():
                require(unit.isdecimal() and str(int(unit)) == unit and isinstance(interval, list)
                        and len(interval) == 3 and all(integer(v) for v in interval), "Invalid compressor support interval")
                start, stop, ready = interval
                require(start < stop <= len(tokens) and stop - 1 <= ready < len(tokens), "Invalid compressor dependency/ready position")
        if lane["kv_kind"] == "kv_token_position":
            require(set(ids) == set(range(snapshot["generated_tokens"])), "Generated ordinary KV inventory must be complete")
        else:
            available = {u for u in support_units(snapshot, lane) if support(snapshot, lane, u)[2] < snapshot["generated_tokens"]}
            require(set(ids) == available, "Generated compressor KV inventory must match support availability")
        for unit in ids:
            start, stop, ready = support(snapshot, lane, unit)
            require(stop <= snapshot["generated_tokens"] and ready < snapshot["generated_tokens"],
                    "Generated KV depends on ungenerated tokens")


def candidates(previous, current, lane):
    if identity(previous)[:4] != identity(current)[:4]:
        return set()
    old_lane = next((item for item in previous["lanes"] if lane_key(item) == lane_key(lane)), None)
    if old_lane is None:
        return set()
    prefix = prefix_length(previous["token_ids"], current["token_ids"])
    result = set()
    for unit in set(old_lane["generated_ids"]) & set(lane["generated_ids"]):
        old = support(previous, old_lane, unit)
        new = support(current, lane, unit)
        if old == new and new[1] <= prefix and new[2] < current["generated_tokens"]:
            result.add(unit)
    return result


def feature(snapshot, lane, unit):
    start, stop, _ = support(snapshot, lane, unit)
    regions = [r for r in snapshot["regions"] if r["start"] < stop and r["end"] > start]
    require(regions, "KV support has no region")
    if len(regions) == 1:
        r = regions[0]
        kind = r["type"]
        age = snapshot["sequence"] - r["created_at"]
        relative = min(15, ((start + stop - 1 - 2 * r["start"]) * 16) // (2 * (r["end"] - r["start"])))
    else:
        kind = "mixed"
        age = min(snapshot["sequence"] - r["created_at"] for r in regions)
        relative = min(15, (start * 16) // len(snapshot["token_ids"]))
    bucket = 0 if age == 0 else 1 if age == 1 else 2 if age < 4 else 3 if age < 8 else 4
    return (kind, bucket, relative)


class TransitionTable:
    """LRU-bounded contexts; each has at most 8 region x 5 age x 16 position cells.

    Counts are token hits/opportunities, with Beta(alpha, alpha) smoothing.
    Request-local native IDs never enter a learned table.
    """
    def __init__(self, max_tables=256, alpha=1.0, minimum_observations=8):
        require(integer(max_tables, 1) and alpha > 0 and integer(minimum_observations, 1), "Invalid table limits")
        self.max_tables = max_tables
        self.alpha = alpha
        self.minimum_observations = minimum_observations
        self.tables = OrderedDict()

    def key(self, snapshot, lane, previous_state, event):
        return (snapshot["model_id"], snapshot["model_revision"], snapshot["tokenizer_sha256"],
                lane["layer"], lane["kv_kind"], previous_state, event)

    def rank(self, snapshot, lane, units, previous_state, event):
        combined = self.key(snapshot, lane, previous_state, event)
        fallback = self.key(snapshot, lane, None, event)
        entry = self.tables.get(combined)
        if entry is not None and entry["observations"] >= self.minimum_observations:
            source = "previous_state+event"
        else:
            entry = self.tables.get(fallback)
            source = "event" if entry else "cold"
        def score(unit):
            hits, access = (entry["cells"].get(feature(snapshot, lane, unit), (0, 0)) if entry else (0, 0))
            return (hits + self.alpha) / (access + 2 * self.alpha)
        return sorted(units, key=lambda u: (-score(u), -support(snapshot, lane, u)[1], u)), source

    def update(self, snapshot, lane, units, target, previous_state, event, *, observation_id=None):
        accessible = Counter(feature(snapshot, lane, u) for u in units)
        hits = Counter(feature(snapshot, lane, u) for u in units & target)
        for state in (None, previous_state):
            key = self.key(snapshot, lane, state, event)
            entry = self.tables.pop(key, {"observations": 0, "cells": {}})
            # A layer observed on several ranks still contributes one historical
            # event observation. Token opportunities/hits remain per rank.
            if observation_id is None or entry.get("last_observation") != observation_id:
                entry["observations"] += 1
            if observation_id is not None:
                entry["last_observation"] = observation_id
            for cell, count in accessible.items():
                h, a = entry["cells"].get(cell, (0, 0))
                entry["cells"][cell] = (h + hits[cell], a + count)
            self.tables[key] = entry
            while len(self.tables) > self.max_tables:
                self.tables.popitem(last=False)


@dataclass
class Entry:
    size: int
    members: set = field(default_factory=set)
    prefetched: set = field(default_factory=set)


class LRU:
    """Byte-capacity cache. Partial pages retain a validity mask of native KV IDs."""
    def __init__(self, capacity):
        require(integer(capacity, 1), "Explicit positive cache capacity required")
        self.capacity = capacity
        self.entries = OrderedDict()
        self.used = 0
        self.prefetch_evictions = 0

    def contains(self, key, unit):
        return key in self.entries and unit in self.entries[key].members

    def put(self, key, size, members, *, protected=frozenset(), prefetch=False):
        require(size <= self.capacity, "Cache cannot hold one transfer unit")
        evicted = []
        if key not in self.entries:
            while self.used + size > self.capacity:
                victim = next((k for k in self.entries if k not in protected), None)
                require(victim is not None, "Cache cannot hold complete attention demand")
                entry = self.entries.pop(victim)
                self.used -= entry.size
                evicted.append((victim, entry))
                if prefetch:
                    self.prefetch_evictions += len(entry.members)
            self.entries[key] = Entry(size)
            self.used += size
        entry = self.entries[key]
        require(entry.size == size, "Cache unit size changed")
        new_members = set(members) - entry.members
        entry.members.update(members)
        if prefetch:
            entry.prefetched.update(new_members)
        self.entries.move_to_end(key)
        return evicted

    def touch(self, key):
        self.entries.move_to_end(key)

    def migrate(self, previous, current, valid_by_lane):
        """Rekey the unchanged prefix; invalidate rewritten suffix/dependencies.

        Also prunes earlier revisions of this session, even if the current
        predecessor was not previously replayed (branches remain independent).
        """
        current_id, old_id = identity(current), identity(previous)
        for key in list(self.entries):
            if key[0] != current["session_id"]:
                continue
            entry = self.entries.pop(key)
            self.used -= entry.size
            keep = entry.members & valid_by_lane.get(key[5:8], set()) if key[:5] == old_id else set()
            if keep:
                new_key = current_id + key[5:]
                entry.members = keep
                entry.prefetched.intersection_update(keep)
                existing = self.entries.get(new_key)
                if existing:
                    existing.members.update(keep)
                    existing.prefetched.update(entry.prefetched)
                else:
                    self.entries[new_key] = entry
                    self.used += entry.size


class RankCaches:
    """Independent rank capacities; cache identities retain the real rank."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.caches = {}

    def for_rank(self, rank):
        if rank not in self.caches:
            self.caches[rank] = LRU(self.capacity)
        return self.caches[rank]

    @property
    def entries(self):
        return ChainMap(*(cache.entries for cache in self.caches.values()))

    @property
    def used(self):
        return sum(c.used for c in self.caches.values())

    @property
    def prefetch_evictions(self):
        return sum(c.prefetch_evictions for c in self.caches.values())

    def contains(self, key, unit):
        return self.for_rank(key[5]).contains(key, unit)

    def put(self, key, size, members, **kwargs):
        return self.for_rank(key[5]).put(key, size, members, **kwargs)

    def touch(self, key):
        self.for_rank(key[5]).touch(key)

    def migrate(self, previous, current, valid_by_lane):
        for cache in self.caches.values():
            cache.migrate(previous, current, valid_by_lane)
