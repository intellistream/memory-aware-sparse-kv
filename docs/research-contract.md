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

