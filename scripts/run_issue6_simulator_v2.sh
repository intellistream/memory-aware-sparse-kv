#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: PY=/path/to/python $0 <CPU-component-result-directory> <new-output-directory>" >&2
  exit 2
fi
SIM_SOURCE=$(realpath "$1")
SIM_OUTPUT=$(realpath -m "$2")
SIM_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-python}
MODEL_DIR=${MODEL_DIR:-/models/DeepSeek-V4-Flash-W8A8}
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
if [[ -e "$SIM_OUTPUT" ]]; then
  echo "Output must be a new directory: $SIM_OUTPUT" >&2
  exit 2
fi
mkdir -p "$(dirname "$SIM_OUTPUT")"
cd "$SIM_ROOT"
"$PY" -m unittest m0a.test_issue6_async_simulator m0a.test_issue6_simulator m0a.test_deepseek_v4_cpu -v 2>&1 | tee "${SIM_OUTPUT}.tests.log"
"$PY" -u -m m0a.run_issue6_async_study --source "$SIM_SOURCE" --output "$SIM_OUTPUT" \
  --model-dir "$MODEL_DIR" --verify-payloads 2>&1 | tee "${SIM_OUTPUT}.run.log"
"$PY" -m m0a.summarize_issue6_async "$SIM_OUTPUT" 2>&1 | tee "${SIM_OUTPUT}.audit.log"
