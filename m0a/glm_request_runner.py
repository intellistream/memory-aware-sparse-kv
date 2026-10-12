"""Single-NPU GLM continuation calibration and fixed-token correctness check."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path

from .echo_workload import sha_ids
from .glm_agent_cases import verify_model


def run(cases_path: Path, model_dir: Path, output: Path, *, fixed: bool,
        limit: int = 48, score_probe: Path | None = None,
        verify_scores: bool = False) -> dict:
    if output.exists():
        raise FileExistsError(output)
    verify_model(model_dir)
    source = json.loads(cases_path.read_text())
    cases = source["cases"]
    if len(cases) != 48 or not 1 <= limit <= 48:
        raise ValueError("Expected 48 sealed cases")
    if fixed and any(len(case.get("forced_continuation_token_ids", [])) != 32 for case in cases[:limit]):
        raise ValueError("Missing calibrated 32-token continuation")

    # Fixed model uses a sparse attention layer type omitted from the generic
    # Transformers allowlist. Keep this change process-local; model files stay sealed.
    import transformers.configuration_utils as config_utils
    if "deepseek_sparse_attention" not in config_utils.ALLOWED_LAYER_TYPES:
        config_utils.ALLOWED_LAYER_TYPES += ("deepseek_sparse_attention",)
    from vllm import LLM, SamplingParams
    if score_probe is not None:
        if score_probe.exists():
            raise FileExistsError(score_probe)

    llm = LLM(model=str(model_dir), tensor_parallel_size=1, dtype="bfloat16",
              max_model_len=65536, max_num_batched_tokens=8192, max_num_seqs=1,
              block_size=128, enable_prefix_caching=False, enforce_eager=True,
              gpu_memory_utilization=0.5,
              logits_processors=["m0a.glm_fixed_logits:FixedContinuation"])
    if score_probe is not None:
        from .glm_live_probe import install, set_case
        install(score_probe, verify_scores=verify_scores)
    result = copy.deepcopy(source)
    result["fixed_continuation_executed"] = fixed
    result["calibration_executed"] = not fixed
    result["request_results"] = []
    try:
        for index, case in enumerate(cases[:limit]):
            if score_probe is not None:
                set_case(case)
            ids = case["prompt_token_ids"]
            if sha_ids(ids) != case["prompt_token_ids_sha256"]:
                raise ValueError("Sealed input token hash mismatch")
            sampling = SamplingParams(
                temperature=0, max_tokens=32, ignore_eos=True, seed=7,
                extra_args=({"fixed_token_ids": case["forced_continuation_token_ids"]}
                            if fixed else {}))
            start = time.monotonic()
            output_list = llm.generate({"prompt_token_ids": ids}, sampling, use_tqdm=False)
            elapsed = time.monotonic() - start
            if len(output_list) != 1 or output_list[0].prompt_token_ids != ids:
                raise RuntimeError("vLLM changed prompt token IDs")
            generated = list(output_list[0].outputs[0].token_ids)
            if len(generated) != 32:
                raise RuntimeError("GLM did not produce exactly 32 tokens")
            if fixed and generated != case["forced_continuation_token_ids"]:
                raise RuntimeError("Fixed logits processor did not force exact tokens")
            if not fixed:
                result["cases"][index]["forced_continuation_token_ids"] = generated
                result["cases"][index]["forced_continuation_sha256"] = sha_ids(generated)
            result["request_results"].append({"case_id": case["case_id"],
                    "prompt_token_ids_sha256": case["prompt_token_ids_sha256"],
                    "output_token_ids": generated, "output_token_ids_sha256": sha_ids(generated),
                    "elapsed_seconds_diagnostic_only": elapsed})
            print(json.dumps({"case": index + 1, "total": limit, "elapsed_seconds": round(elapsed, 3)}),
                  flush=True)
    finally:
        del llm
    result["processed_cases"] = limit
    result["input_sha256"] = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    temp.replace(output)
    return {"processed_cases": limit, "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "fixed_continuation_executed": fixed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fixed", action="store_true")
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument("--score-probe", type=Path)
    parser.add_argument("--verify-scores", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.cases, args.model, args.output, fixed=args.fixed,
                         limit=args.limit, score_probe=args.score_probe,
                         verify_scores=args.verify_scores)))


if __name__ == "__main__":
    main()
