"""Single-NPU trace audit. Fixed replay and unconstrained greedy checks stay separate."""
from __future__ import annotations

import argparse
import gzip
import inspect
import json
import os
import subprocess
from pathlib import Path

from .echo_workload import sha_ids
from .glm_agent_cases import verify_model, sha_file, REVISION
from .glm_native_trace import NativeCapture, validate_request, audit_repeats, decode_history
from .history_prefetch import require


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_trace(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")


def run(args):
    require(args.repeats >= 2, "At least two repeats required")
    require(Path(f"/dev/davinci{args.physical_npu}").exists(), "Physical NPU is not mounted")
    require(os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == str(args.runtime_device), "Runtime visibility mismatch")
    verify_model(args.model)
    source = json.loads(args.cases.read_text())
    by_id = {c["case_id"]: c for c in source["cases"]}
    require(len(set(args.case_id)) == len(args.case_id) and all(c in by_id for c in args.case_id), "Invalid case list")
    cases = [by_id[c] for c in args.case_id]
    for case in cases:
        require(sha_ids(case["prompt_token_ids"]) == case["prompt_token_ids_sha256"]
                and len(case["prompt_token_ids"]) == case["prompt_tokens"], "Prompt hash/length drift")
        require(len(case["forced_continuation_token_ids"]) == 32 and
                sha_ids(case["forced_continuation_token_ids"]) == case["forced_continuation_sha256"], "Continuation drift")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "status": "starting", "run_id": args.output_dir.name,
        "model_dir": str(args.model), "model_revision": REVISION, "cases_sha256": sha_file(args.cases),
        "case_ids": args.case_id, "repeats": args.repeats, "physical_npu": args.physical_npu,
        "runtime_device_before_visibility": args.runtime_device, "ascend_rt_visible_devices": os.environ["ASCEND_RT_VISIBLE_DEVICES"],
        "process_device": 0, "autonomous_agent": False, "performance_eligible": False,
        "candidate_rule": "generated KV before current batch; excludes current-batch KV",
        "capture_scope": "event predecessor + 32 prefill queries + final prefill + 31 decode queries",
        "limitations": ["archived DeepSeek-agent prompts replayed by GLM tiny", "fixed/periodic continuation remains",
                        "sampled prefill is not complete query history", "no async offload or HBM savings measured"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "code_sha256": {p.name: sha_file(p) for p in Path(__file__).parent.glob("*native_trace*.py")}}
    write_json(args.output_dir / "manifest.json", manifest)
    capture, llm = NativeCapture(), None
    installed = False
    requests, checks, history = [], [], []
    try:
        import torch
        import torch_npu
        import transformers.configuration_utils as config_utils
        require(torch.npu.device_count() == 1, "Exactly one visible runtime NPU required")
        torch.npu.set_device(0)
        manifest["device_name"] = torch.npu.get_device_name(0)
        if "deepseek_sparse_attention" not in config_utils.ALLOWED_LAYER_TYPES:
            config_utils.ALLOWED_LAYER_TYPES += ("deepseek_sparse_attention",)
        from vllm import LLM, SamplingParams
        manifest["versions"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__}
        llm = LLM(model=str(args.model), tensor_parallel_size=1, dtype="bfloat16",
            max_model_len=65536, max_num_batched_tokens=8192, max_num_seqs=1,
            block_size=128, enable_prefix_caching=False, enforce_eager=True, gpu_memory_utilization=0.5,
            logits_processors=["m0a.glm_fixed_logits:FixedContinuation"])
        # Import after normal LLM initialization; importing device_op directly
        # before Ascend ops registration causes a circular import in this runtime.
        from vllm_ascend.device.device_op import DeviceOperator
        hook_path = Path(inspect.getfile(DeviceOperator))
        manifest["runtime_operator_source"] = {"path": str(hook_path), "sha256": sha_file(hook_path)}
        write_json(args.output_dir / "manifest.json", manifest)

        def generate(case, fixed):
            params = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True, seed=7,
                extra_args={"fixed_token_ids": case["forced_continuation_token_ids"]} if fixed else {})
            result = llm.generate({"prompt_token_ids": case["prompt_token_ids"]}, params, use_tqdm=False)
            require(len(result) == 1 and result[0].prompt_token_ids == case["prompt_token_ids"], "Live prompt changed")
            tokens = list(result[0].outputs[0].token_ids)
            require(len(tokens) == 32, "Missing output tokens")
            if fixed:
                require(tokens == case["forced_continuation_token_ids"], "Fixed schedule changed")
            return tokens

        baseline = {}
        for case in cases:
            baseline[case["case_id"]] = [generate(case, False) for _ in range(args.repeats)]
        capture.install()
        installed = True
        for index, case in enumerate(cases):
            capture.begin(case)
            greedy = generate(case, False)
            greedy_rows = validate_request(capture.rows, case)
            greedy_path = args.output_dir / f"case-{index:02d}-greedy.jsonl.gz"
            write_trace(greedy_path, greedy_rows)
            native_outputs = baseline[case["case_id"]]
            checks.append({"case_id": case["case_id"], "trace_off_tokens": native_outputs[0],
                           "trace_off_repeats": native_outputs,
                           "native_greedy_repeat_stable": all(t == native_outputs[0] for t in native_outputs),
                           "trace_on_tokens": greedy, "trace": greedy_path.name,
                           "greedy_output_equal": greedy == native_outputs[0]})
            for repeat in range(args.repeats):
                capture.begin(case)
                tokens = generate(case, True)
                rows = validate_request(capture.rows, case)
                path = args.output_dir / f"case-{index:02d}-repeat-{repeat}.jsonl.gz"
                write_trace(path, rows)
                requests.append({"case_id": case["case_id"], "repeat": repeat, "trace": path.name,
                                 "sha256": sha_file(path), "rows": rows, "output_tokens": tokens})
                if repeat == 0:
                    history.extend(decode_history(rows, case, run_id=args.output_dir.name, repeat=repeat))
                print(json.dumps({"case_id": case["case_id"], "repeat": repeat, "rows": len(rows)}), flush=True)
        audit = audit_repeats(requests, repeats=args.repeats)
        audit["greedy_checks"] = checks
        audit["all_greedy_outputs_equal"] = all(c["greedy_output_equal"] for c in checks)
        audit["all_native_greedy_repeats_stable"] = all(c["native_greedy_repeat_stable"] for c in checks)
        audit["requests"] = [{k: v for k, v in r.items() if k != "rows"} for r in requests]
        audit["scope"] = manifest["capture_scope"]
        write_json(args.output_dir / "audit.json", audit)
        with (args.output_dir / "history-decode.jsonl").open("w", encoding="utf-8") as output:
            for row in history:
                output.write(json.dumps(row, separators=(",", ":")) + "\n")
        manifest["status"] = "passed" if (audit["all_selected_sets_stable"] and audit["all_greedy_outputs_equal"]
                                          and audit["all_native_greedy_repeats_stable"]) else "unstable"
        manifest["artifacts"] = {p.name: sha_file(p) for p in args.output_dir.iterdir() if p.name != "manifest.json"}
    except Exception as error:
        manifest["status"], manifest["error"] = "failed", f"{type(error).__name__}: {error}"
        raise
    finally:
        if installed:
            capture.restore()
        write_json(args.output_dir / "manifest.json", manifest)
        del llm
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/root/models/echo-glm-model"))
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--physical-npu", type=int, required=True)
    parser.add_argument("--runtime-device", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
