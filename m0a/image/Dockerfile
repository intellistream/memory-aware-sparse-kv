FROM quay.io/ascend/vllm-ascend@sha256:b2d13e24b295171d8f63678506fad5542f142e81d57e407a0b8c98c16ba0c4f7
COPY envs.py /vllm-workspace/vllm-ascend/vllm_ascend/envs.py
COPY model_runner_v1.py /vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py
COPY dsa_cp.py /vllm-workspace/vllm-ascend/vllm_ascend/attention/context_parallel/dsa_cp.py
COPY m0a_selected_trace.py /vllm-workspace/vllm-ascend/vllm_ascend/attention/context_parallel/m0a_selected_trace.py
