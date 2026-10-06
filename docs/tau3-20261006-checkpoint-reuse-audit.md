# τ³ checkpoint repair and input reuse audit

Source run: `deepseek_20261005T134845Z_3eb88906`. Its worker failed while waiting
for the `01-inputs` local acknowledgment. The original status remains failed.
The later final synchronization verified 1,875 files, including the raw input
snapshot, against `final-checksums.json` (SHA-256
`273c88eda13ee1d2991c8e5f3cd15bab1c68f1ea2646696d8d2e149773328082`).
The source `episodes.json` contains 203 eligible episodes: 114 retail and 89
banking knowledge. The snapshot selected for reuse contains 216 files and
267,709,106 uncompressed bytes: all 211 task logs, episodes, collection
summary, rejected episode log, τ³ provenance, and the old pairs as a separate
comparison artifact. Every file is checked against the source final manifest
before packaging and again after Pod extraction. The new launch records the
archive SHA-256, source manifest SHA-256, each file SHA-256, and old/new pair
SHA-256 values.

The old `pairs.json` has SHA-256
`5e77755d331ce8d6412b377f9bfef1f98cbc11ba46f05a7812678a4de551bb1b`.
Its 8K and 32K histories overlap on 40 retail tasks and 43 banking knowledge
tasks. It is preserved in `source-inputs/old-pairs.json` and never supplied as
the new pair set. For each domain, the repaired builder sorts task IDs, assigns
the first 28 tasks to 8K and the remainder to 32K, then splits each target
into two disjoint chains. It checks all 48 pairs, event cells, at least 48
different task histories per domain, chain isolation, and zero 8K/32K overlap
immediately after construction. The worker then confirms all new prompt token
IDs against the served model's tokenizer.

The τ³ checkpoint ACK wait is 3,600 seconds. Its worker total remains 43,200
seconds. The guardian transfers in batches up to 32 MiB or 64 files, skips
locally verified files, and records discovery, transfer, verification, and ACK
progress. A checkpoint discovered after worker termination is downloaded and
verified without writing a late ACK; it cannot change the failed experiment.

The failed ACK's exact bottleneck is undetermined. These changes record enough
per-step timing to distinguish discovery, transfer, verification, and ACK
writing on the next run. The new run remains unqualified until its full trace,
CPU replay, final hashes, service restoration, and result publication pass.
