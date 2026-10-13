"""Single-process, single-NPU GLM shadow-KV online experiment runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

from .echo_workload import sha_ids
from .glm_agent_cases import EXPECTED_FILES, verify_model
from .glm_online_adapter import OnlineAdapter
from .glm_schedule import verified_schedule

LIMIT_SECONDS = 12 * 60 * 60
CODE_FILES = ("glm_online_adapter.py", "glm_scratch.py", "glm_live_score.py",
              "glm_prefetch_policy.py", "glm_fixed_logits.py", "glm_online_run.py",
              "glm_agent_cases.py", "glm_schedule.py", "glm_online_stats.py",
              "echo_workload.py", "launch_glm_formal.sh")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(cases_path: Path, model_dir: Path, output_dir: Path, *,
        sealed_schedule_path: Path | None = None, max_requests: int = 432):
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not 1 <= max_requests <= 432:
        raise ValueError("Invalid request limit")
    verify_model(model_dir)
    repo = Path(__file__).resolve().parents[1]
    resolved_output = output_dir.resolve()
    if resolved_output.is_relative_to(repo) and not resolved_output.is_relative_to(repo / "m0a/runs"):
        raise ValueError("Run output inside this checkout must be under ignored m0a/runs/")
    source = cases_path.read_bytes()
    cases = json.loads(source)
    order = verified_schedule(source, sealed_schedule_path)
    code_dir = Path(__file__).parent
    by_id = {case["case_id"]: case for case in cases["cases"]}
    if len(by_id) != 48:
        raise ValueError("Sealed input case drift")
    output_dir.mkdir(parents=True)
    code_hashes = {name: digest(code_dir / name) for name in CODE_FILES}
    manifest = {"schema_version": 1, "formal": max_requests == 432,
                "physical_npu": 6, "process_device": "npu:0", "tp": 1,
                "dtype": "bfloat16", "concurrency": 1, "eager": True,
                "block_size": 128, "prefix_caching": False,
                "scratch_tokens_per_layer": 3072, "scratch_page_tokens": 16,
                "prefetch_budget_tokens": 512, "prototype_hbm_shadow": True,
                "model_files": {name: {"size": size, "sha256": sha}
                                for name, (size, sha) in EXPECTED_FILES.items()},
                "cases_sha256": hashlib.sha256(source).hexdigest(),
                "sealed_schedule_sha256": digest(sealed_schedule_path) if sealed_schedule_path else None,
                "code_sha256": code_hashes, "max_seconds": LIMIT_SECONDS,
                "requested_measurements": max_requests}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_dir / "schedule.json").write_text(json.dumps(order, separators=(",", ":")) + "\n")
    started = time.monotonic()

    import transformers.configuration_utils as config_utils
    if "deepseek_sparse_attention" not in config_utils.ALLOWED_LAYER_TYPES:
        config_utils.ALLOWED_LAYER_TYPES += ("deepseek_sparse_attention",)
    from vllm import LLM, SamplingParams
    llm = LLM(model=str(model_dir), tensor_parallel_size=1, dtype="bfloat16",
              max_model_len=65536, max_num_batched_tokens=8192, max_num_seqs=1,
              block_size=128, enable_prefix_caching=False, enforce_eager=True,
              gpu_memory_utilization=0.5, disable_log_stats=False,
              logits_processors=["m0a.glm_fixed_logits:FixedContinuation"])
    adapter = OnlineAdapter(output_dir / "selected")
    adapter.install()

    def request(case, mode, label):
        adapter.set_request(case, mode, label)
        sampling = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True,
                                  seed=7, extra_args={"fixed_token_ids":
                                                      case["forced_continuation_token_ids"]})
        started_request = time.monotonic()
        outputs = llm.generate({"prompt_token_ids": case["prompt_token_ids"]},
                               sampling, use_tqdm=False)
        elapsed = time.monotonic() - started_request
        if len(outputs) != 1 or outputs[0].prompt_token_ids != case["prompt_token_ids"]:
            raise AssertionError("Live input token IDs changed")
        generated = list(outputs[0].outputs[0].token_ids)
        if generated != case["forced_continuation_token_ids"] or len(generated) != 32:
            raise AssertionError("Fixed continuation differs between modes")
        evidence = adapter.finish_request()
        metrics = outputs[0].metrics
        if metrics is None or metrics.first_token_ts <= 0 or metrics.last_token_ts <= 0:
            raise RuntimeError(f"vLLM request timing metrics missing: {metrics!r}")
        ttft = metrics.first_token_latency * 1000
        tpot = (metrics.last_token_ts - metrics.first_token_ts) * 1000 / 31
        if ttft <= 0 or tpot <= 0:
            raise RuntimeError("vLLM request timing metrics invalid")
        return {**evidence, "prompt_token_ids_sha256": case["prompt_token_ids_sha256"],
                "prompt_tokens": case["prompt_tokens"],
                "forced_continuation_token_ids": generated,
                "forced_continuation_sha256": sha_ids(generated),
                "fixed_continuation_verified": True,
                "ttft_ms": ttft, "tpot_ms": tpot,
                "throughput_tps": 32 / elapsed,
                "decode_tokens": 32, "decode_attention_steps": 31,
                "elapsed_ms": elapsed * 1000}

    try:
        warmup_case = cases["cases"][0]
        for index, mode in enumerate(("no_offload", "offload_no_prefetch", "offload_echo_prefetch")):
            if time.monotonic() - started > LIMIT_SECONDS:
                raise TimeoutError("12-hour worker limit reached in warmup")
            request(warmup_case, mode, f"warmup_{index}_{mode}")
            print(json.dumps({"warmup": mode, "done": True}), flush=True)
        with (output_dir / "measurements.jsonl").open("w") as output:
            for item in order[:max_requests]:
                if time.monotonic() - started > LIMIT_SECONDS:
                    raise TimeoutError("12-hour worker limit reached")
                case = by_id[item["case_id"]]
                if sha_ids(case["prompt_token_ids"]) != case["prompt_token_ids_sha256"]:
                    raise AssertionError("Sealed input hash changed")
                label = f"request_{item['ordinal']:03d}_{item['mode']}"
                result = request(case, item["mode"], label)
                row = {"case_id": item["case_id"], "domain": case["domain"],
                       "episode_id": case["episode_id"],
                       "event_type": case["event_type"],
                       "repeat": item["repeat"], "mode": item["mode"],
                       "ordinal": item["ordinal"], **result}
                output.write(json.dumps(row, separators=(",", ":")) + "\n")
                output.flush()
                os.fsync(output.fileno())
                print(json.dumps({"request": item["ordinal"] + 1,
                                  "of": max_requests, "case_id": item["case_id"],
                                  "mode": item["mode"], "elapsed_ms": result["elapsed_ms"]}),
                      flush=True)
    finally:
        del llm
    manifest["completed_measurements"] = max_requests
    manifest["measurements_sha256"] = digest(output_dir / "measurements.jsonl")
    manifest["elapsed_seconds"] = time.monotonic() - started
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sealed-schedule", type=Path,
                        help="Optional external schedule to verify against the generated order")
    parser.add_argument("--max-requests", type=int, default=432)
    args = parser.parse_args()
    print(json.dumps(run(args.cases, args.model, args.output_dir,
                         sealed_schedule_path=args.sealed_schedule,
                         max_requests=args.max_requests), indent=2))


if __name__ == "__main__":
    main()
