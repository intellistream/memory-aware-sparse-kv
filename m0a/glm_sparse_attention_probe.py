"""Single-operator NPU check for 16-token scratch remapping.

This is a correctness gate. Random tensors are deliberately never reported as
agent-workload latency evidence.
"""
from __future__ import annotations

import argparse
import json
import math

from .glm_scratch import CAPACITY_TOKENS, PAGE_SIZE, ScratchKV


def run(device: str = "npu:0") -> dict:
    import torch
    import torch_npu  # noqa: F401
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    torch.manual_seed(171)
    torch.npu.set_device(device)
    native_tokens, native_page = 4096, 128
    native_key = torch.randn(native_tokens // native_page, native_page, 1, 512,
                             dtype=torch.bfloat16, device=device)
    native_rope = torch.randn(native_tokens // native_page, native_page, 1, 64,
                              dtype=torch.bfloat16, device=device)
    host_key = native_key.reshape(native_tokens, 1, 512).cpu().pin_memory()
    host_rope = native_rope.reshape(native_tokens, 1, 64).cpu().pin_memory()
    q = torch.randn(1, 8, 512, dtype=torch.bfloat16, device=device)
    q_rope = torch.randn(1, 8, 64, dtype=torch.bfloat16, device=device)
    selected = list(range(0, native_tokens, 2))
    scores = {i: float(native_tokens - i) for i in range(native_tokens)}
    native_indices = torch.tensor(selected, dtype=torch.int32, device=device).view(1, 1, -1)

    def attention(key, rope, indices, block_table, kv_len):
        result = torch.ops._C_ascend.npu_sparse_flash_attention(
            query=q, key=key, value=key, sparse_indices=indices,
            scale_value=1 / math.sqrt(576), sparse_block_size=1,
            block_table=block_table,
            actual_seq_lengths_query=torch.tensor([1], dtype=torch.int32, device=device),
            actual_seq_lengths_kv=torch.tensor([kv_len], dtype=torch.int32, device=device),
            query_rope=q_rope, key_rope=rope, layout_query="TND", layout_kv="PA_BSND",
            sparse_mode=3, attention_mode=2, return_softmax_lse=False)
        return result[0] if isinstance(result, tuple) else result

    native_table = torch.arange(native_tokens // native_page, dtype=torch.int32,
                                device=device).view(1, -1)
    original = attention(native_key, native_rope, native_indices, native_table, native_tokens)
    results = {}
    for name, candidates in (("no_prefetch", []),
                             ("prefetch", list(range(0, 1024, 2)))):
        scratch_key = torch.empty(CAPACITY_TOKENS // PAGE_SIZE, PAGE_SIZE, 1, 512,
                                  dtype=torch.bfloat16, device=device)
        scratch_rope = torch.empty(CAPACITY_TOKENS // PAGE_SIZE, PAGE_SIZE, 1, 64,
                                   dtype=torch.bfloat16, device=device)
        scratch = ScratchKV(host_key, host_rope, scratch_key, scratch_rope)
        transfer = scratch.prepare(selected, scores, candidates, prefetch=bool(candidates))
        scratch.assert_payload_exact(selected)
        remapped = torch.tensor(transfer.remapped, dtype=torch.int32, device=device).view(1, 1, -1)
        result = attention(scratch_key, scratch_rope, remapped,
                           scratch.block_table(), CAPACITY_TOKENS)
        torch.npu.synchronize()
        error = (result.float() - original.float()).abs()
        maximum = error.max().item()
        if not torch.allclose(result.float(), original.float(), atol=0.03, rtol=0.03):
            raise AssertionError(f"Sparse attention changed after {name} remap: {maximum}")
        results[name] = {"max_absolute_error": maximum,
                         "prefetched_tokens": len(transfer.prefetched),
                         "synchronous_recall_tokens": len(transfer.recalled)}
    return {"device": device, "native_block_size": native_page,
            "scratch_page_size": PAGE_SIZE, "scratch_capacity_tokens": CAPACITY_TOKENS,
            "selected_tokens": len(selected), "attention_equivalent": True,
            "synthetic_operator_probe": True, "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()
    print(json.dumps(run(args.device), indent=2))


if __name__ == "__main__":
    main()
