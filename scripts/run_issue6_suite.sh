#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 3 ]]; then
  echo 'Usage: run_issue6_suite.sh PHYSICAL_NPU RUNTIME_DEVICE NEW_RUN_ID' >&2
  exit 2
fi
PHYSICAL=$1
RUNTIME=$2
RUN_ID=$3
[[ $PHYSICAL =~ ^[0-9]+$ && $RUNTIME =~ ^[0-9]+$ && $RUN_ID =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$ ]]
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_ROOT=/root/experiments/memkv-env-89df7fad
RUN="$CODE/m0a/runs/issue6-$RUN_ID"
[[ ! -e "$RUN" && -e /dev/davinci$PHYSICAL ]]
mkdir -p "$RUN"
cd "$CODE"
export ASCEND_RT_VISIBLE_DEVICES=$RUNTIME
export ISSUE6_PHYSICAL_NPU=$PHYSICAL ISSUE6_RUNTIME_DEVICE=$RUNTIME
export ISSUE6_SERVER_RECEIPT="$RUN/server-receipt.json"
export VLLM_ENABLE_V1_MULTIPROCESSING=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
native_env() {
  source "$ENV_ROOT/activate.sh"
  export PYTHONPATH="$CODE:$CODE/m0a:${PYTHONPATH:-}"
}
tau_env() {
  source "$ENV_ROOT/activate-tau3.sh"
  export PYTHONPATH="$CODE:$CODE/m0a:${PYTHONPATH:-}"
}
SERVER_PID=''
cleanup() {
  if [[ -n "$SERVER_PID" && -f "$ISSUE6_SERVER_RECEIPT" ]]; then
    (native_env; python -m m0a.issue6_serve --stop "$ISSUE6_SERVER_RECEIPT")
    wait "$SERVER_PID" || true
    SERVER_PID=''
  fi
}
trap cleanup EXIT
python3 - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 8866))
PY
(
  native_env
  exec python -m m0a.issue6_serve serve /root/models/echo-glm-model \
    --host 127.0.0.1 --port 8866 --served-model-name glm-tiny \
    --tensor-parallel-size 1 --dtype bfloat16 --max-model-len 65536 \
    --max-num-batched-tokens 8192 --max-num-seqs 1 --block-size 128 \
    --no-enable-prefix-caching --enforce-eager --gpu-memory-utilization 0.25 \
    --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 --no-async-scheduling
) > "$RUN/server.log" 2>&1 &
SERVER_PID=$!
READY=0
for attempt in $(seq 1 90); do
  kill -0 "$SERVER_PID"
  if python3 - <<'PY'
import urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    assert opener.open('http://127.0.0.1:8866/health', timeout=2).status == 200
except Exception:
    raise SystemExit(1)
PY
  then
    READY=1
    break
  fi
  sleep 2
done
[[ $READY == 1 ]]
(
  tau_env
  python -m m0a.issue6_workload --tau-source "$TAU3_SOURCE" \
    --output "$RUN/agent" --max-tokens 512 --reasoning-effort low
  python -m m0a.issue6_agent_probe --tau-source "$TAU3_SOURCE" \
    --materials "$RUN/agent/prompt-material.jsonl" --output "$RUN/agent-probe"
) > "$RUN/agent.log" 2>&1
cleanup
(
  native_env
  python -m m0a.issue6_capture --materials "$RUN/agent/prompt-material.jsonl" \
    --physical-npu "$PHYSICAL" --runtime-device "$RUNTIME" --repeats 2 \
    --output "$RUN/native-$RUN_ID"
  python -m m0a.issue6_capture --audit "$RUN/native-$RUN_ID" > "$RUN/independent-audit.json"
  python -m m0a.issue6_provenance --native-manifest "$RUN/native-$RUN_ID/manifest.json" \
    --output "$RUN/runtime-provenance.json"
  python -m venv --system-site-packages "$RUN/analysis-env"
  "$RUN/analysis-env/bin/python" -m pip install --index-url https://pypi.org/simple --only-binary=:all: --timeout 120 --retries 3 \
    -r m0a/issue6-analysis-requirements.txt > "$RUN/analysis-install.log" 2>&1
  "$RUN/analysis-env/bin/python" -m pip freeze > "$RUN/analysis-freeze.txt"
  "$RUN/analysis-env/bin/python" -m m0a.issue6_evaluate --trace "$RUN/native-$RUN_ID/trace.jsonl.gz" \
    --native-audit "$RUN/independent-audit.json" \
    --capacity 4096 --capacity 8192 --budget 128 --budget 512 --output "$RUN/evaluation"
  python -m unittest m0a.test_issue6_suite m0a.test_glm_native_trace -v
) > "$RUN/native-and-evaluation.log" 2>&1
echo "$RUN"
