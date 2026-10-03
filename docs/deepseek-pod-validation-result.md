# DeepSeek Pod validation result

Capture run `deepseek_20261003T154057Z_8f2f3a23` completed 48 trace-off and 48 trace-on requests. The original capture report records a failed CP chunk-range validator rule, while the supervised DeepSeek service was restored healthy. The corrected rule passes all 49,896 native rows and the 33,264-row request window on the archived capture.

The CPU continuation is running at `/root/memory-aware-sparse-kv/m0a/runs/deepseek_20261003T154057Z_8f2f3a23/postprocess-cp-padding-v1`. The detached local scheduler updates this document with the four replay results, hashes, and final service check after completion. Local progress is in `m0a/runs/deepseek_20261003T154057Z_8f2f3a23/automation-status.json`.

Strict output acceptance remains `not_qualified`; selected-set drift makes replay comparisons exploratory. This is offline replay, not an online offload or performance-gain measurement.
