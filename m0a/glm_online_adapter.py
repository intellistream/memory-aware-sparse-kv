"""Single-request BF16 GLM SFA shadow-KV adapter for an online prototype.

The original NPU KV cache remains resident for the indexer and an exact
attention reference. The offload arms really call sparse attention on their
own 16-token scratch pages; they do not demonstrate any HBM saving.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import time
from pathlib import Path

from .glm_live_score import verify_rows
from .glm_prefetch_policy import decode_candidates, prefill_candidates, update_ema
from .glm_scratch import CAPACITY_TOKENS, PAGE_SIZE, ScratchKV

KV_BYTES = 1152


class _LayerState:
    def __init__(self, kv_cache, max_context: int):
        import torch
        key, rope = kv_cache[:2]
        self.host_key = torch.empty((max_context, *key.shape[2:]), dtype=key.dtype,
                                    device="cpu", pin_memory=True)
        self.host_rope = torch.empty((max_context, *rope.shape[2:]), dtype=rope.dtype,
                                     device="cpu", pin_memory=True)
        scratch_key = torch.empty((CAPACITY_TOKENS // PAGE_SIZE, PAGE_SIZE, *key.shape[2:]),
                                  dtype=key.dtype, device=key.device)
        scratch_rope = torch.empty((CAPACITY_TOKENS // PAGE_SIZE, PAGE_SIZE, *rope.shape[2:]),
                                   dtype=rope.dtype, device=rope.device)
        self.scratch = ScratchKV(self.host_key, self.host_rope, scratch_key, scratch_rope)
        self.copied = 0
        self.prefill_threshold = None
        self.decode_ema = None

    def ensure_host(self, kv_cache, block_table, context_len: int):
        import torch
        if context_len > self.host_key.shape[0] or context_len < self.copied:
            raise ValueError("KV source length drift or overflow")
        if context_len == self.copied:
            return
        page_size = kv_cache[0].shape[1]
        block_count = math.ceil(context_len / page_size)
        blocks = block_table[0, :block_count].detach().cpu().tolist()
        for page in range(self.copied // page_size, block_count):
            physical = int(blocks[page])
            if physical < 0 or physical >= kv_cache[0].shape[0]:
                raise ValueError("Invalid source KV page")
            begin = max(self.copied, page * page_size)
            end = min(context_len, (page + 1) * page_size)
            first, last = begin - page * page_size, end - page * page_size
            self.host_key[begin:end].copy_(kv_cache[0][physical, first:last], non_blocking=True)
            self.host_rope[begin:end].copy_(kv_cache[1][physical, first:last], non_blocking=True)
        torch.npu.synchronize()
        self.copied = context_len


class OnlineAdapter:
    def __init__(self, trace_dir: Path):
        self.trace_dir = trace_dir
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.case = None
        self.mode = None
        self.request_label = None
        self.layers = {}
        self.scores = {}
        self.native_selected = {}
        self.selected_records = []
        self.prefill_ns = 0
        self.decode_ns = 0
        self.score_rows_verified = 0
        self.attention_rows_verified = 0
        self.kv_rows_verified = 0
        self.max_attention_error = 0.0
        self.prefetch_hit_bytes = 0
        self.prefetch_waste_bytes = 0
        self.prefetched_bytes = 0
        self.synchronous_recall_bytes = 0

    def set_request(self, case: dict, mode: str, label: str):
        if mode not in {"no_offload", "offload_no_prefetch", "offload_echo_prefetch"}:
            raise ValueError("Unknown GLM prototype mode")
        self.__init__(self.trace_dir)
        self.case, self.mode, self.request_label = case, mode, label

    def _targets(self, query_start: int, query_rows: int):
        if self.case is None:
            return []
        prompt = self.case["prompt_tokens"]
        prefill = range(self.case["event_position"], self.case["event_position"] + 32)
        # 32 generated tokens have 31 decode attention passes in vLLM: the
        # first output token comes from the final prefill logit.
        decode = range(prompt, prompt + 31)
        return [(position - query_start, position, "prefill") for position in prefill
                if query_start <= position < query_start + query_rows] + [
                (position - query_start, position, "decode") for position in decode
                if query_start <= position < query_start + query_rows]

    @staticmethod
    def _position(metadata, q_rows, lengths_key):
        if metadata.block_table.shape[0] != 1:
            raise RuntimeError("GLM prototype requires one request")
        context_len = int(lengths_key.detach().cpu().reshape(-1)[-1])
        return context_len - q_rows, context_len

    def install(self):
        from vllm_ascend.device.device_op import DeviceOperator
        native_indexer = DeviceOperator.indexer_select_post_process
        native_attention = DeviceOperator.execute_sparse_flash_attention_process
        adapter = self

        def indexer(sfa_impl, q_li, q_li_scale, q_li_shape_ori, weights, kv_cache,
                    metadata, seq_q, seq_key, enable_sparse_li_c8,
                    use_torch_npu_lightning_indexer):
            topk = native_indexer(sfa_impl, q_li, q_li_scale, q_li_shape_ori,
                                  weights, kv_cache, metadata, seq_q, seq_key,
                                  enable_sparse_li_c8, use_torch_npu_lightning_indexer)
            if adapter.case is None or adapter.mode == "no_offload":
                return topk
            if enable_sparse_li_c8 or q_li_scale is not None:
                raise RuntimeError("Full-score reference supports only BF16 GLM indexer")
            query_start, _ = adapter._position(metadata, q_li.shape[0], seq_key)
            targets = adapter._targets(query_start, q_li.shape[0])
            if targets:
                t0 = time.perf_counter_ns()
                rows = [row for row, _position, _phase in targets]
                scores = verify_rows(q_li, weights, kv_cache[2], metadata.block_table,
                                     topk, query_start=query_start, rows=rows)
                layer = str(sfa_impl.layer_name)
                for row, position, _phase in targets:
                    adapter.scores[(layer, position)] = scores[position]
                    adapter.native_selected[(layer, position)] = topk[row].detach().cpu().reshape(-1).tolist()
                adapter.score_rows_verified += len(targets)
                elapsed = time.perf_counter_ns() - t0
                if all(phase == "prefill" for _, _, phase in targets):
                    adapter.prefill_ns += elapsed
                elif all(phase == "decode" for _, _, phase in targets):
                    adapter.decode_ns += elapsed
                else:
                    raise RuntimeError("Mixed prefill/decode indexer batch")
            return topk

        def attention(sfa_impl, ql_nope, q_pe, kv_cache, topk, metadata,
                      seq_q, seq_key, **kwargs):
            native_out = native_attention(sfa_impl, ql_nope, q_pe, kv_cache,
                                          topk, metadata, seq_q, seq_key, **kwargs)
            if adapter.case is None:
                return native_out
            if not hasattr(native_out, "shape"):
                raise RuntimeError("Unexpected tuple sparse-attention output")
            query_start, context_len = adapter._position(metadata, ql_nope.shape[0], seq_key)
            targets = adapter._targets(query_start, ql_nope.shape[0])
            if not targets:
                return native_out
            layer = str(sfa_impl.layer_name)
            for row, position, phase in targets:
                native_ids = topk[row].detach().cpu().reshape(-1).tolist()
                adapter.selected_records.append({"layer": layer, "position": position,
                                                 "phase": phase, "selected_ids": native_ids})
                if adapter.mode == "no_offload":
                    continue
                t0 = time.perf_counter_ns()
                source_layer = layer
                match = re.search(r"layers\.(\d+)\.", layer)
                if match and int(match[1]) >= 3:
                    source_layer = layer.replace(f"layers.{match[1]}.", "layers.2.")
                    if adapter.native_selected.get((source_layer, position)) != native_ids:
                        raise AssertionError("Shared layer changed the native selected set")
                score_tensor = adapter.scores.get((source_layer, position))
                if score_tensor is None:
                    raise RuntimeError("Full native candidate scores missing before attention")
                scores = score_tensor.detach().cpu().numpy()
                valid = [int(value) for value in native_ids if value >= 0]
                state = adapter.layers.get(layer)
                if state is None:
                    state = _LayerState(kv_cache, adapter.case["prompt_tokens"] + 32)
                    adapter.layers[layer] = state
                state.ensure_host(kv_cache, metadata.block_table, context_len)
                kth = min(float(scores[native_id]) for native_id in valid)
                if phase == "prefill":
                    candidates = prefill_candidates(scores, state.prefill_threshold,
                                                    len(valid)) if adapter.mode == "offload_echo_prefetch" else []
                    state.prefill_threshold = kth
                else:
                    candidates = decode_candidates(scores, state.decode_ema) \
                        if adapter.mode == "offload_echo_prefetch" else []
                    state.decode_ema = update_ema(state.decode_ema, kth)
                transfer = state.scratch.prepare(native_ids, scores, candidates,
                                                 prefetch=adapter.mode == "offload_echo_prefetch")
                state.scratch.assert_payload_exact(native_ids)
                adapter.kv_rows_verified += 1
                import torch
                scratch_ids = torch.tensor(transfer.remapped, dtype=topk.dtype,
                                           device=topk.device).view_as(topk[row:row + 1])
                small_q = torch.tensor([1], dtype=torch.int32, device=topk.device)
                small_k = torch.tensor([CAPACITY_TOKENS], dtype=torch.int32,
                                       device=topk.device)
                scratch_out = native_attention(
                    sfa_impl, ql_nope[row:row + 1], q_pe[row:row + 1],
                    (state.scratch.device_key, state.scratch.device_rope),
                    scratch_ids, metadata, small_q, small_k,
                    block_table=state.scratch.block_table(), sparse_mode=3)
                error = (scratch_out.float() - native_out[row:row + 1].float()).abs().max().item()
                adapter.max_attention_error = max(adapter.max_attention_error, error)
                if not torch.allclose(scratch_out.float(), native_out[row:row + 1].float(),
                                      atol=0.03, rtol=0.03):
                    raise AssertionError(f"GLM scratch attention differs at {layer} pos {position}: {error}")
                native_out[row:row + 1].copy_(scratch_out)
                adapter.attention_rows_verified += 1
                adapter.prefetched_bytes += len(transfer.prefetched) * KV_BYTES
                adapter.prefetch_hit_bytes += len(transfer.useful_prefetch) * KV_BYTES
                adapter.prefetch_waste_bytes += len(transfer.wasted_prefetch) * KV_BYTES
                adapter.synchronous_recall_bytes += len(transfer.recalled) * KV_BYTES
                elapsed = time.perf_counter_ns() - t0
                if phase == "prefill":
                    adapter.prefill_ns += elapsed
                else:
                    adapter.decode_ns += elapsed
            return native_out

        DeviceOperator.indexer_select_post_process = staticmethod(indexer)
        DeviceOperator.execute_sparse_flash_attention_process = classmethod(
            lambda _cls, *args, **kwargs: attention(*args, **kwargs))

    def finish_request(self) -> dict:
        if self.case is None or self.request_label is None:
            raise RuntimeError("No GLM prototype request")
        records = sorted(self.selected_records, key=lambda row: (row["layer"], row["position"]))
        expected = 6 * (32 + 31)
        if len(records) != expected or len({(row["layer"], row["position"]) for row in records}) != expected:
            raise RuntimeError(f"Incomplete native selected-set coverage: {len(records)}/{expected}")
        digest = hashlib.sha256()
        trace_path = self.trace_dir / (self.request_label + ".selected.jsonl.gz")
        if trace_path.exists():
            raise FileExistsError(trace_path)
        with gzip.open(trace_path, "wt") as output:
            for record in records:
                rendered = json.dumps(record, separators=(",", ":"))
                digest.update((rendered + "\n").encode())
                output.write(rendered + "\n")
        if self.mode != "no_offload" and (
            self.score_rows_verified != 3 * (32 + 31) or
            self.attention_rows_verified != expected or self.kv_rows_verified != expected
        ):
            raise RuntimeError("Incomplete full-score or scratch correctness coverage")
        return {"native_selected_set_sha256": digest.hexdigest(),
                "native_selected_trace": str(trace_path),
                "native_topk_exact": True, "kv_payload_exact": True,
                "attention_equivalent": True,
                "score_rows_verified": self.score_rows_verified,
                "attention_rows_verified": self.attention_rows_verified,
                "max_attention_error": self.max_attention_error,
                "prefill_window_ms": self.prefill_ns / 1e6,
                "decode_window_ms": self.decode_ns / 1e6,
                "prefetch_hit_bytes": self.prefetch_hit_bytes,
                "prefetch_waste_bytes": self.prefetch_waste_bytes,
                "prefetched_bytes": self.prefetched_bytes,
                "synchronous_recall_bytes": self.synchronous_recall_bytes,
                "prototype_hbm_shadow": True}
