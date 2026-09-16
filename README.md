# Memory-Aware Sparse KV

This repository studies memory-aware, lossless KV-cache offloading and
prefetching for native sparse-attention LLM serving.

The starting point is ECHO-style serving: full KV state is host-authoritative,
while the accelerator retains and recalls only the KV blocks selected by the
model's native sparse-attention indexer. This project does **not** approximate
dense attention or replace the model-selected sparse KV set. It asks whether
agent memory-state transitions can predict that exact set early enough to
reduce synchronous recall stalls.

## Research boundary

- Owner: Hongyi Ruan (`@Lan-Fa`)
- Baseline: sequential/index-score prefetch for native sparse attention
- Treatment: memory-state-conditioned residency and lossless prefetch
- Correctness: identical model-selected KV set and output under matched runs
- Hardware claims require a complete native execution evidence chain

## Repository structure

- `paper/`: paper source and references
- `src/`: mechanism implementation
- `scripts/`: reproducible experiment entry points
- `results/`: immutable result manifests and raw-result pointers
- `docs/`: research contract and design records

The first milestone is a necessity study. Implementation expansion is gated on
showing that tool calls, memory writes, consolidation, or task switches create
material recall stalls that ECHO-style locality alone cannot hide.


## Strong-baseline and lead-time gate

The [research contract](docs/research-contract.md) now explicitly includes ECHO
and HiSparse, event-shuffled controls, timestamped causal lead time, and exact
block-version checks. Native sparse selection is unchanged; no layer-similarity
assumption or quality-approximate eviction is introduced. This is advisor-authored
research planning; implementation and experiments remain with `@Lan-Fa`.
