#!/usr/bin/env bash
set -euo pipefail

run_id="${1:?pass a unique run id}"
if [[ ! "$run_id" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "Run id must contain only letters, numbers, underscores or hyphens" >&2
  exit 2
fi

cd /root/memecho-echo-code
export PYTHONPATH="/root/memecho-echo-code:${PYTHONPATH:-}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0

output_dir="/root/echo-glm-formal-${run_id}"
if [[ -e "$output_dir" ]]; then
  echo "Output directory already exists: $output_dir" >&2
  exit 2
fi

# The Python loop also checks elapsed time between requests; timeout bounds a
# request stuck inside an NPU kernel. This command acts only on this run.
timeout --signal=TERM --kill-after=30s 12h \
  python3 -m m0a.glm_online_run \
    --cases /root/echo-glm-calibration.json \
    --model /root/echo-glm-model \
    --output-dir "$output_dir"

python3 -m m0a.glm_online_stats \
  --cases /root/echo-glm-calibration.json \
  --measurements "$output_dir/measurements.jsonl" \
  --output "$output_dir/analysis.json"

sha256sum "$output_dir/manifest.json" "$output_dir/measurements.jsonl" \
  "$output_dir/analysis.json" > "$output_dir/completed.sha256"
echo "Formal GLM experiment complete: $output_dir"
