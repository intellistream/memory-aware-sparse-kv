# 2026-10-04 τ³ run audit

Run `deepseek_20261004T032925Z_f0e69394` failed before pair construction:
the retail domain yielded 48 usable tool episodes, but banking_knowledge yielded
19 of the required 48. The adapter had sent customer simulator instructions as
an agent-facing user turn. No trace-off/on or CPU replay conclusion is available.

The original final status remains unchanged. Its `service_health=unverified`
was caused by an extra `HBM used (MB)` line on the final check's JSON stdout.
An independent read-only check found `/health` healthy, model `dsv4` serving
`/models/DeepSeek-V4-Flash-W8A8`, the original PID 631336 with start tick
624429940 and unchanged supervisor config SHA-256
`7a32b02e9aa69a65ffcc303ebac1e455495bda99500b9977ef02ca870347289b`.
All eight NPUs were healthy and no experiment-owned processes remained.

The final manifest SHA-256 was
`1cb43d77a0a724bfb1c4f9c5591aff9495d7d80fa375b040342fae83cdd203cf`;
all 1801 listed local files matched their sizes and hashes. The published
result commit was `b528f3d7857114650c5ab21f39b673081767c16f`.
