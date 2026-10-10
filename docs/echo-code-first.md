# ECHO agent-workload code phase

Status: 2026-10-10. The isolated server checkout is
`/root/memecho-echo-code` on branch `feat/echo-ascend-code-first`, based on
`4d3909ed4556c71a28cdf768a7f1c8093fd052d6`. The recovery checkout
`/root/memory-aware-sparse-kv` and historical runs are not changed. The
server currently has no `/dev/davinci*` devices.

This phase prepares a bounded test of ECHO-style mechanisms on the available
DeepSeek-V4/Ascend stack. It does not implement the author's NVIDIA system or
validate an online ECHO performance effect. The [paper](https://www.usenix.org/system/files/osdi26-liu-guangda.pdf)
uses score-threshold prefetch during decode and approximate top-k prefetch
between query blocks *inside one prefill*. Reuse of the previous tool call's
selected set is a separate baseline, not ECHO.

## Implemented and CPU-verifiable

- The native selected-set trace hook skips CP-padded query positions beyond
  the actual API prompt and records the unpadded chunk end. A regression test
  covers an 8,191-token prompt in an 8,192-position CP chunk.
- Existing regression fixtures now include the required launch metadata and
  match the current three-retry error text. The actual NPU capsule test skips
  when no device is mounted; it remains pending for the hardware phase.
- `m0a/echo_replay.py` accepts only **complete native score vectors**. It
  implements the decode EMA threshold (alpha 0.5), a paper-derived 256-bin
  prefill subset, bounded score-priority KV residency, and guaranteed recall
  of every native selected ID. Output is a CPU transfer-count model only.
  Native IDs and selected sets are never replaced by the prediction.
  CPU replay is never marked eligible for an empirical performance conclusion,
  even when its score source declares itself native.
- `m0a/echo_score_reference.py` computes all weighted-ReLU scores from
  explicitly dequantized and rotary-transformed query/key tensors, then checks
  the result against native top-k. It is a slow diagnostic reference; the
  deployed paged-cache layout and native numerical equivalence still need NPU
  checks.
- `m0a/echo_runtime.py` provides an exact tensor KV pool for correctness
  tests and explicit capability gates for no offload, offload without
  prefetch, and offload with ECHO prefetch. The Python pool is **not eligible
  for timed performance claims**.
- `m0a/echo_workload.py` seals all 96 archived τ³ event/control prompts and
  the first eight generated token IDs from each first trace-on response into
  separate cases and randomized mode schedules. A future runtime must attest
  that it forced these continuation IDs; a schedule alone is not execution.
- `m0a/echo_benchmark.py` rejects unforced continuations, mismatched prompt
  or continuation hashes, missing native selection equivalence, and
  unqualified backends. It reports offload prefetch differences and, when
  available, no-offload comparisons, with independent-episode confidence
  intervals and an ordinary-workload interaction. Selected-set drift is
  retained and prevents a causal regression label for that load. A directional
  claim also needs at least 20 independent episodes in each workload class
  at the tested load and consistent directions in both agent domains; no
  percentage effect threshold is imposed.

The score JSONL contract for `echo_replay` is one row per native query:
`schema_version=1`, `score_coverage="full"`, `run_id`, `request_id`,
`score_source_kind` (`native_indexer_full` or `synthetic_fixture`),
`score_source_sha256`, `native_topk_verified=true`,
`rank`, `layer`, `phase`, strictly increasing `sequence` and `position`,
`context_len`, `k`, all eligible `scores=[[native_id, score], ...]` in
native-ID order, and the authoritative `selected_ids`. Prefill rows also
need `prefill_chunk` and `query_block`. The replay rejects a score vector
that disagrees with the native top-k set. It does not accept the archived
selected-set-only trace as a full score trace.

CPU checks on the server:

```bash
cd /root/memecho-echo-code
python3 -m unittest m0a.test_echo_replay m0a.test_echo_runtime \
  m0a.test_echo_workload m0a.test_echo_benchmark \
  m0a.test_echo_diagnostic_source m0a.test_echo_score_reference \
  m0a.test_selected_trace -v
```

Final code-stage validation on 2026-10-10: the full CPU suite ran 159 tests
and passed with two NPU-dependent skips; after the final source-contract
change, all 24 targeted ECHO and trace tests passed. `git diff --check`
passed. These results do not exercise the NPU kernel or timed serving path.

To prepare a schedule from the retained server capture:

```bash
cd /root/memecho-echo-code
python3 -m m0a.echo_workload \
  --pairs /root/memory-aware-sparse-kv/m0a/runs/deepseek_20261006T051335Z_9ae72511/pairs.json \
  --responses /root/memory-aware-sparse-kv/m0a/runs/deepseek_20261006T051335Z_9ae72511/trace_on-responses.jsonl \
  --output /root/echo-agent-schedule-NEW.jsonl
```

The command also writes the matching `.cases.jsonl` and a SHA-256
manifest. It does not call the model.

The retained server preparation is `/root/echo-agent-schedule-v2.jsonl`,
with 96 cases and 4,320 schedule rows. Its schedule SHA-256 is
`89ec57d06aa78bf0a9109183d2d97f569541045b13f08e1ab3044abaca5cd1b5`;
its case-file SHA-256 is
`329a0af71b6afd70995a9c7207576b5aa8db60f83768b909584b9c75b7fb1ade`.
The manifest explicitly records `fixed_continuation_executed=false` and
`npu_validated=false`.

## Hardware gate

The installed 910B2 indexer has a Python `return_value` parameter, but
the inspected kernel entry does not connect its `sparseValues` output to
the compute path. The opt-in diagnostic patch therefore refuses to enable
score capture unless a modified kernel carries explicit score-output
evidence. Even working top-k scores would omit the nonselected candidates
needed to measure ECHO's false prefetches. A native **full-score** diagnostic
and a fused, graph-safe host-to-device prefetch/guaranteed-recall path remain
unimplemented. The current tensor pool cannot substitute for them in timing.

When eight NPUs are available, implement and verify those kernel paths,
validate exact native selected sets and KV payloads on real requests, then
force the sealed continuation tokens and run all three modes under matched
capacity and arrival schedules. Keep output drift and selected-set drift
recorded; do not infer an ECHO failure from CPU byte counts or from an
unoptimized Python/host copy path. The experiment should report the measured
direction, magnitude and uncertainty; if the regression is absent or
inconclusive, report that result.
