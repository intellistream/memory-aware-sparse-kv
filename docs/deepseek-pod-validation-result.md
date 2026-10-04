# DeepSeek Pod validation result

- Capture run: `deepseek_20261003T154057Z_8f2f3a23`; capture status: `failed` (`Invalid CP chunk range` validator rule).
- CPU continuation: `engineering_validated`; strict output acceptance: `not_qualified`.
- Capture commit: `6c0d54df012ba5cb08fa24dcbd570930790f31b0`; validator commit: `e5b79041f685d08bd4b7711372e2f6bfc2a6f1e5`.
- Valid responses: 48 trace-off and 48 trace-on. Native rows: 49,896; window rows: 33,264 (48 × 33 × 21).
- Output differences: `{'trace_off': 15, 'trace_on': 40}`. Selected-set comparison: `exploratory`; repeated/prefix drift: 16632/504.
- Replay reports: `replay-{per_rank,aggregate}-{64,128}mib/report.json` under the postprocess directory.
- Supervised service: `memecho-deepseek                 RUNNING   pid 631336, uptime 9:50:10`; model `dsv4` at the read-only mount.
- Remote capture: `/root/memory-aware-sparse-kv/m0a/runs/deepseek_20261003T154057Z_8f2f3a23`; remote report: `/root/memory-aware-sparse-kv/m0a/runs/deepseek_20261003T154057Z_8f2f3a23/postprocess-cp-padding-v1/report.json`.
- Local postprocess: `/home/ruan/memecho/memory-aware-sparse-kv/m0a/runs/deepseek_20261003T154057Z_8f2f3a23/postprocess-cp-padding-v1`; verified files: 47; postprocess manifest SHA-256: `ec272742734b7ccc19855800bc41ab0238f6a1aa442fdb07f868b782ad26d2e6`.

This is offline selected-set CPU replay, not online KV offload or a performance gain measurement. The model metadata hashes and 70 indexed shards were checked; no readable model commit marker or saved original per-shard hashes were available. The original failed capture remains archived separately.
