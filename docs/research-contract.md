# Seven-question research contract

## 1. What is the research question?

Can explicit agent-memory state transitions predict the exact native
sparse-attention KV working set early enough to improve host-to-device recall
without changing the model-selected sparse KV set?

## 2. Why is it important?

Native sparse attention reduces attention work but its complete KV state still
grows with context length. Offloading relieves device-capacity pressure, yet a
late KV recall remains on the decoding critical path. Agent workloads add tool
calls, memory writes, consolidation, supersession, and task switches that can
break purely sequential query locality.

## 3. What is missing from prior work?

ECHO demonstrates graph-friendly cache management and lossless intra-query and
inter-query prefetching based on index-score predictability and sequential
query processing. The unresolved question is whether explicit memory-state
transitions provide additional predictive structure under agent workloads,
especially when adjacent queries are not semantically or operationally local.

## 4. What is the key idea?

Keep the model's native sparse-attention indexer authoritative. Use memory
events only to schedule exact KV residency and prefetch: predict likely future
selected blocks, prefetch candidates, and validate them against the eventual
indexer output. Wrong predictions may waste bandwidth but must never alter the
selected KV set or output.

## 5. Why might it work?

Agent runtimes expose causal events before subsequent model invocations:
tool-result insertion, memory write, episode transition, consolidation, and
supersession. These events may delimit a smaller and more stable KV working set
than adjacent-query score extrapolation alone.

## 6. How will it be evaluated?

Use matched real agent traces and a native sparse-attention model/runtime.
Compare no offload, ECHO-style sequential prefetch, and memory-conditioned
lossless prefetch. Report prediction precision/recall, synchronous recalled
bytes, wasted prefetched bytes, recall-stall time, TTFT/TPOT, p50/p95/p99
latency, throughput, device KV residency, graph recompilation, and
host-device-transfer counters. Validate exact selected-block and output
equivalence.

## 7. What would constitute a contribution or a stop?

Proceed only if at least two representative agent workloads show a recurring
memory-transition-induced locality break and the memory signal reduces
synchronous recall or stall by at least 20% over the strongest sequential
prefetch baseline without changing outputs. Stop or reframe if the phenomenon
is rare, prediction overhead dominates, or gains disappear under matched
concurrency and device-memory budgets.


## 2026-09-09 clarification: causal lead time and stronger controls

This amendment preserves the original native-sparse/lossless question and the
existing two-workload, 20% necessity gate. It does not authorize implementation
or hardware work, and does not change the student owner.

The architecture chain is: memory event available before the next indexer ->
additional selected-block predictability -> early transfer -> less exposed
recall stall, after prediction and wasted-prefetch cost. Record event availability,
prediction, DMA submission/completion and consumption timestamps. Events inferred
from future queries are forbidden. Equal selected-block overlap without enough
lead time is not a useful predictor. Adjacent-layer similarity is neither assumed
nor required; stratify models that actually share selections and those that do not.

Strong controls now include native full-resident execution, demand-only host
offload, ECHO's supported lossless prefetch, and HiSparse's supported exact
hierarchical cache (including layer prefetch only where supported). Compare on
the same checkpoint/indexer, graph mode, host/device byte budget, transfer
bandwidth and concurrency. Unsupported ports are reported as unavailable, not
weak substitutes. Add event-shuffled and event-removed controls, with identical
prediction/transfer budgets, and an explicitly nondeployable future-selection
oracle to measure headroom. No quality-approximate dense-KV eviction is a
matched lossless control.

Every consumed block must have the eventual native indexer's exact identity,
version and bytes; misses block for authoritative fetch. Cancellation and slot
reuse cannot expose a previous request's state. Wrong predictions may waste
bandwidth and increase tail latency. Report both wasted traffic and p99; do not
trade worse tail or fewer completed requests for apparent recall improvement.

References verified against primary sources:
- ECHO: https://www.usenix.org/conference/osdi26/presentation/liu-guangda
- HiSparse: https://arxiv.org/abs/2608.07009

The distinct hypothesis is incremental value of early memory events over these
controls. Neither ECHO nor HiSparse is assumed to lack all relevant locality
signals; their supported implementations must be pinned before a comparison.
