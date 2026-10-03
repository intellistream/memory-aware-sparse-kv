#!/usr/bin/env bash
set -euo pipefail

ROOT=${MEMECHO_ROOT:-/workspace/memecho}
export DOCKER_HOST="unix://$ROOT/runtime/docker.sock"
IMAGE=${M0A_IMAGE:-memecho/m0a-trace:stage0-v3}
MODEL_DIR="$ROOT/models/DeepSeek-V4-Flash-w8a8-mtp"
TRACE_DIR=${VLLM_ASCEND_M0A_TRACE_DIR:-$ROOT/m0a/traces/pilot}
TRACE_POSITIONS=${VLLM_ASCEND_M0A_TRACE_POSITIONS:-0:12,1018:1032}
TRACE_RUN_ID=${VLLM_ASCEND_M0A_RUN_ID:-m0a_pilot_$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$TRACE_DIR"
NAME=memecho-m0a-trace

test -s "$MODEL_DIR/config.json"
test -s "$MODEL_DIR/tokenizer.json"
test -s "$MODEL_DIR/quant_model_weights.safetensors.index.json"
python3 "$ROOT/m0a/preflight.py"

if docker inspect "$NAME" >/dev/null 2>&1; then
  if [[ ${RECREATE:-0} != 1 ]]; then
    echo "$NAME already exists; set RECREATE=1 only after confirming it is safe to replace" >&2
    exit 1
  fi
  docker rm -f "$NAME"
fi

npu_args=()
for i in {0..7}; do npu_args+=(--device "/dev/davinci$i"); done

jemalloc=$(docker run --rm "$IMAGE" \
  bash -lc "ldconfig -p | awk '/libjemalloc.so.2/{print \$NF; exit}'")
test -n "$jemalloc"

docker run -d \
  --name "$NAME" \
  --restart no \
  --privileged \
  --network host \
  --shm-size=512g \
  "${npu_args[@]}" \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/sbin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /etc/hccn.conf:/etc/hccn.conf:ro \
  -v "$ROOT:$ROOT" \
  -e OMP_PROC_BIND=false \
  -e OMP_NUM_THREADS=10 \
  -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  -e LD_PRELOAD="$jemalloc" \
  -e HCCL_BUFFSIZE=1024 \
  -e TASK_QUEUE_ENABLE=1 \
  -e HCCL_OP_EXPANSION_MODE=AIV \
  -e VLLM_ASCEND_ENABLE_FLASHCOMM1=1 \
  -e VLLM_ASCEND_M0A_TRACE_DIR="$TRACE_DIR" \
  -e VLLM_ASCEND_M0A_TRACE_POSITIONS="$TRACE_POSITIONS" \
  -e VLLM_ASCEND_M0A_RUN_ID="$TRACE_RUN_ID" \
  "$IMAGE" \
  vllm serve "$MODEL_DIR" \
    --host 127.0.0.1 \
    --port 8900 \
    --served-model-name dsv4 \
    --tensor-parallel-size 8 \
    --data-parallel-size 1 \
    --enable-expert-parallel \
    --quantization ascend \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --reasoning-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --no-enable-prefix-caching \
    --max-model-len 133120 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 32 \
    --gpu-memory-utilization 0.90 \
    --block-size 128 \
    --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
    --speculative-config '{"num_speculative_tokens":1,"method":"mtp","enforce_eager":true}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"enable_dsa_cp":true,"enable_flashcomm1":true,"multistream_overlap_shared_expert":true}'

echo "Started $NAME. Follow logs with: docker logs -f $NAME"
