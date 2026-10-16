"""Causal, CPU access-history baselines for issue #6, task 6.

One instance belongs to one immutable KV namespace, rank, layer and phase.
Only observe() accepts selected blocks; predict() has no target-selection input.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


def require(ok, message):
    if not ok:
        raise ValueError(message)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def block_ids(values):
    result = tuple(values)
    require(all(integer(v) for v in result), "Invalid logical block ID")
    require(len(result) == len(set(result)), "Duplicate logical block ID")
    return result


@dataclass(frozen=True)
class PredictionContext:
    query: int
    prediction_step: int
    stage: str  # Unique event occurrence, not a reusable event-type label.
    eligible_blocks: tuple[int, ...]  # Generated and transferable before prediction.


@dataclass(frozen=True)
class Candidate:
    block: int
    accesses: int
    last_completed_query: int | None
    region_accesses: int
    region_last_completed_query: int | None


class AccessHistoryPredictor:
    """Frequency, recency or fixed-address-region priority; deterministic ties.

    `global` means this KV namespace's completed history across stages, never
    another task/request/model/context version. `stage` cold-starts at every
    boundary. Construct separate instances for prefill and decode.
    """

    def __init__(self, *, scope="global", signal="frequency", region_blocks=4):
        require(scope in {"global", "stage"}, "Unknown history scope")
        require(signal in {"frequency", "recency", "region", "last_selection", "none"},
                "Unknown history signal")
        require(integer(region_blocks, 1), "Invalid region width")
        self.scope, self.signal, self.region_blocks = scope, signal, region_blocks
        self.counts, self.region_counts = Counter(), Counter()
        self.last, self.region_last = {}, {}
        self.previous = ()
        self.stage = None
        self.closed_stages = set()
        self.pending = {}
        self.last_prediction = self.last_completion = -1
        self.last_query = self.last_observed_query = -1

    def predict(self, context: PredictionContext, *, budget: int) -> list[Candidate]:
        require(integer(budget), "Invalid candidate budget")
        require(integer(context.query) and context.query > self.last_query,
                "Query IDs must increase; duplicate/reordered prediction")
        require(integer(context.prediction_step)
                and context.prediction_step >= self.last_prediction
                and context.prediction_step > self.last_completion,
                "Prediction cannot see uncompleted/future query selections")
        require(not self.pending or context.prediction_step == self.last_prediction,
                "Complete the pending visibility batch before the next prediction step")
        require(isinstance(context.stage, str) and bool(context.stage), "Missing stage")
        eligible = block_ids(context.eligible_blocks)
        if context.stage != self.stage:
            require(not self.pending, "Cannot change stage with incomplete queries")
            require(context.stage not in self.closed_stages, "Stage occurrence cannot be reused")
            if self.stage is not None:
                self.closed_stages.add(self.stage)
            if self.scope == "stage":
                self.counts.clear()
                self.region_counts.clear()
                self.last.clear()
                self.region_last.clear()
                self.previous = ()
            self.stage = context.stage
        self.pending[context.query] = context.prediction_step
        self.last_prediction, self.last_query = context.prediction_step, context.query
        if self.signal == "none" or not budget:
            return []
        previous = set(self.previous)
        candidates = []
        for block in eligible:
            region = block // self.region_blocks
            known = (region in self.region_counts if self.signal == "region" else
                     block in previous if self.signal == "last_selection" else block in self.counts)
            if known:
                candidates.append(Candidate(block, self.counts[block], self.last.get(block),
                                            self.region_counts[region], self.region_last.get(region)))

        def priority(c):
            last = c.last_completed_query if c.last_completed_query is not None else -1
            if self.signal == "region":
                return (-c.region_accesses, -c.region_last_completed_query,
                        -c.accesses, -last, c.block)
            if self.signal == "recency":
                return (-last, -c.accesses, c.block)
            if self.signal == "last_selection":
                return (c.block,)
            return (-c.accesses, -last, c.block)

        return sorted(candidates, key=priority)[:budget]

    def observe(self, *, query: int, completion_step: int, selected_blocks) -> None:
        """Publish a completed query after its prediction; no future selections."""
        require(integer(query) and query in self.pending and query == min(self.pending)
                and query > self.last_observed_query,
                "Missing prediction, duplicate or reordered completion")
        require(integer(completion_step) and completion_step > self.pending[query]
                and completion_step >= self.last_completion
                and completion_step > self.last_prediction,
                "Completion must follow all pending batch predictions")
        selected = block_ids(selected_blocks)
        self.counts.update(selected)
        self.last.update((block, query) for block in selected)
        regions = {block // self.region_blocks for block in selected}
        self.region_counts.update(regions)  # Count queries, not repeated IDs in a region.
        self.region_last.update((region, query) for region in regions)
        self.previous = selected
        self.last_completion, self.last_observed_query = completion_step, query
        del self.pending[query]
