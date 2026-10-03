#!/usr/bin/env bash
set -euo pipefail

ROOT=${MEMECHO_ROOT:-/workspace/memecho}
export DOCKER_HOST="unix://$ROOT/runtime/docker.sock"
IMAGE=${M0A_IMAGE:-memecho/m0a-glm53-trace:v3}
MODEL_DIR=${M0A_MODEL_DIR:-$ROOT/models/GLM-5.3-0.6B-A0.4B}
NPU_ID=${M0A_NPU_ID:-0}
TRACE_DIR=${VLLM_ASCEND_M0A_TRACE_DIR:-}
TRACE_POSITIONS=${VLLM_ASCEND_M0A_TRACE_POSITIONS:-}
TRACE_RUN_ID=${VLLM_ASCEND_M0A_RUN_ID:-}
NAME=${M0A_CONTAINER_NAME:-memecho-m0a-glm53-trace}
OWNER_RUN_ID=${M0A_OWNER_RUN_ID:-}
[[ "$NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]

test -s "$MODEL_DIR/config.json"
test -s "$MODEL_DIR/tokenizer.json"
test -s "$MODEL_DIR/model.safetensors"
test -s "$MODEL_DIR/.m0a_revision"
test -s "$MODEL_DIR/.m0a_model_manifest.json"
test "$(cat "$MODEL_DIR/.m0a_revision")" = "20aae340d157bd2215b9d81165be87e94686dfdf"
python3 "$ROOT/m0a/preflight.py" --devices "$NPU_ID"

if [[ -n "$TRACE_DIR" ]]; then
  test -n "$TRACE_POSITIONS"
  test -n "$TRACE_RUN_ID"
  mkdir -p "$TRACE_DIR"
fi

if docker inspect "$NAME" >/dev/null 2>&1; then
  if [[ ${RECREATE:-0} != 1 ]]; then
    echo "$NAME already exists; set RECREATE=1 only after confirming it is safe to replace" >&2
    exit 1
  fi
  docker rm -f "$NAME"
fi

jemalloc=$(docker run --rm "$IMAGE" \
  bash -lc "ldconfig -p | awk '/libjemalloc.so.2/{print \$NF; exit}'")
test -n "$jemalloc"

docker run -d \
  --name "$NAME" \
  --label "memecho.validation.run=$OWNER_RUN_ID" \
  --restart no \
  --privileged \
  --network host \
  --shm-size=64g \
  --device "/dev/davinci$NPU_ID" \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/sbin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /etc/hccn.conf:/etc/hccn.conf:ro \
  -v "$ROOT:$ROOT" \
  -e ASCEND_RT_VISIBLE_DEVICES="$NPU_ID" \
  -e OMP_PROC_BIND=false \
  -e OMP_NUM_THREADS=10 \
  -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  -e LD_PRELOAD="$jemalloc" \
  -e VLLM_ASCEND_M0A_TRACE_DIR="$TRACE_DIR" \
  -e VLLM_ASCEND_M0A_TRACE_POSITIONS="$TRACE_POSITIONS" \
  -e VLLM_ASCEND_M0A_RUN_ID="$TRACE_RUN_ID" \
  -e VLLM_ASCEND_M0A_MODEL_PROFILE=glm53_tiny \
  -e VLLM_ASCEND_M0A_MODEL_ID=inference-optimization/GLM-5.3-0.6B-A0.4B \
  -e VLLM_ASCEND_M0A_MODEL_REVISION=20aae340d157bd2215b9d81165be87e94686dfdf \
  "$IMAGE" \
  vllm serve "$MODEL_DIR" \
    --host 127.0.0.1 \
    --port 8900 \
    --served-model-name glm53-tiny \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --dtype bfloat16 \
    --kv-cache-dtype auto \
    --no-enable-prefix-caching \
    --max-model-len 65536 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 1 \
    --gpu-memory-utilization 0.50 \
    --block-size 128 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true}'

echo "Started $NAME on physical NPU $NPU_ID"
