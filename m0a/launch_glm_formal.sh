#!/usr/bin/env bash
set -euo pipefail

run_id="${1:?pass a unique run id}"
if [[ ! "$run_id" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "Run id must contain only letters, numbers, underscores or hyphens" >&2
  exit 2
fi

code_dir="${GLM_CODE_DIR:-/root/memory-aware-sparse-kv}"
cases="${GLM_CASES:-/root/experiments/glm-20261012/inputs/echo-glm-calibration.json}"
model="${GLM_MODEL:-/root/models/echo-glm-model}"
run_root="${GLM_RUN_ROOT:-/root/experiments/glm-20261012/runs}"
cd "$code_dir"
export PYTHONPATH="$code_dir:${PYTHONPATH:-}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0

mkdir -p "$run_root"
output_dir="$run_root/$run_id"
if [[ -e "$output_dir" ]]; then
  echo "Output directory already exists: $output_dir" >&2
  exit 2
fi

# The Python loop also checks elapsed time between requests; timeout bounds a
# request stuck inside an NPU kernel. This command acts only on this run.
schedule_arg=()
if [[ -n "${GLM_SEALED_SCHEDULE:-}" ]]; then
  schedule_arg=(--sealed-schedule "$GLM_SEALED_SCHEDULE")
fi
timeout --signal=TERM --kill-after=30s 12h \
  python3 -m m0a.glm_online_run \
    --cases "$cases" \
    --model "$model" \
    --output-dir "$output_dir" "${schedule_arg[@]}"

python3 -m m0a.glm_online_stats \
  --cases "$cases" \
  --measurements "$output_dir/measurements.jsonl" \
  --output "$output_dir/analysis.json"

sha256sum "$output_dir/manifest.json" "$output_dir/measurements.jsonl" \
  "$output_dir/analysis.json" > "$output_dir/completed.sha256"
echo "Formal GLM experiment complete: $output_dir"
