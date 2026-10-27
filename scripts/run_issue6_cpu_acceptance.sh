#!/usr/bin/env bash
# Run from this checkout. MODEL_DIR is read-only; every output uses a new directory.
set -euo pipefail
if [[ $# != 4 ]]; then
  echo "Usage: $0 PYTHON TAU_SOURCE PRIOR_CPU_RUN NEW_OUTPUT" >&2
  exit 2
fi
PYTHON="$1"
TAU_SOURCE="$2"
PRIOR_CPU_RUN="$3"
OUTPUT="$4"
MODEL_DIR="${MODEL_DIR:-/models/DeepSeek-V4-Flash-W8A8}"
cd "$(dirname "$0")/.."
mkdir "$OUTPUT"
OUTPUT="$(cd "$OUTPUT" && pwd)"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="$PWD/m0a:$PWD${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -m m0a.issue6_cpu_readiness --model-dir "$MODEL_DIR" --output "$OUTPUT/readiness.json" > "$OUTPUT/readiness.log" 2>&1
TEST_MODULES=()
for TEST_FILE in m0a/test_*.py; do
  TEST_MODULE="${TEST_FILE%.py}"
  TEST_MODULES+=("${TEST_MODULE//\//.}")
done
"$PYTHON" -m unittest "${TEST_MODULES[@]}" -v > "$OUTPUT/tests.log" 2>&1
if [[ -n "${TAU3_PYTHON:-}" ]]; then
  TAU3_SOURCE="$TAU_SOURCE" PYTHONPATH="$PWD:$PWD/m0a:$TAU_SOURCE/src" LITELLM_LOCAL_MODEL_COST_MAP=True \
    "$TAU3_PYTHON" -m unittest \
    m0a.test_chat_generation.ChatGenerationTests.test_tau_user_truncation_is_logged_before_message_construction \
    m0a.test_tau3_workload.UserHistoryTests.test_pinned_tau_user_tool_replies_keep_ids_through_role_flip \
    -v > "$OUTPUT/tau3-supplemental-tests.log" 2>&1
fi
"$PYTHON" -m m0a.issue6_generalization_contracts --tau-source "$TAU_SOURCE" --output "$OUTPUT/benchmark-split-audit.json" > "$OUTPUT/benchmark-split-audit.log" 2>&1
"$PYTHON" -m m0a.run_issue6_cpu_long_context --model-dir "$MODEL_DIR" --output "$OUTPUT/long-context" > "$OUTPUT/long-context.log" 2>&1
"$PYTHON" -m m0a.run_issue6_async_study --source "$PRIOR_CPU_RUN" --output "$OUTPUT/async-study" --model-dir "$MODEL_DIR" --verify-payloads > "$OUTPUT/async-study.log" 2>&1
"$PYTHON" -m m0a.summarize_issue6_async "$OUTPUT/async-study" > "$OUTPUT/async-audit.log" 2>&1
echo "CPU checks completed: $OUTPUT; full issue6 and eight-NPU validation remain pending."
