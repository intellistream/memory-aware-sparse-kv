"""CPU-only contract tests for the single-NPU GLM engineering lane."""

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

from m0a.analyze import load_trace, summarize
from m0a.contracts import validate_trace
from m0a.model_profiles import load_profile, validate_pair_policy
from m0a.orchestrate import build_plan
from m0a.synthetic_trace import generate
from m0a.test_selected_trace import FakeTopK


ROOT = Path(__file__).resolve().parents[1]


def pair_set():
    item = {
        "prompt": "synthetic",
        "boundary_position": 200,
        "prompt_tokens_expected": 300,
        "prompt_token_ids_sha256_expected": "0" * 64,
    }
    return {
        "schema_version": 1,
        "model_profile": "glm53_tiny",
        "model_id": "inference-optimization/GLM-5.3-0.6B-A0.4B",
        "model_revision": "20aae340d157bd2215b9d81165be87e94686dfdf",
        "tokenizer_json_sha256": "1" * 64,
        "pairs": [{
            "pair_id": "glm-smoke",
            "workload_id": "synthetic",
            "episode_id": "one",
            "context_target": 300,
            "event_type": "tool_result",
            "event": dict(item),
            "control": dict(item),
        }],
    }


class GLM53OfflineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = load_profile("glm53_tiny", ROOT / "m0a/model_profiles.json")

    def test_model_profile_and_schema2_fixture_analyze(self):
        responses, by_rank = generate(
            pair_set(), run_id="glm-fixture", repetitions=1,
            profile=self.profile,
        )
        self.assertEqual(set(by_rank), {0})
        first = by_rank[0][0]
        validate_trace(first, contract=self.profile)
        self.assertEqual(first["raw_index_unit"], "kv_token_position")
        self.assertEqual(first["selected_width"], 2048)

        with tempfile.TemporaryDirectory() as directory:
            trace_dir = Path(directory)
            (trace_dir / "rank0.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in by_rank[0])
            )
            trace, inventory = load_trace(
                trace_dir, responses, contract=self.profile
            )
            result = summarize(trace, responses)
        self.assertEqual(len(result), 1)
        for value in inventory.values():
            self.assertEqual(len(value["layers"]), 6)
            self.assertEqual(
                list(value["layer_sources"].values()).count("computed"), 3
            )
            self.assertEqual(
                list(value["layer_sources"].values()).count("reused"), 3
            )

    def test_glm_orchestration_plans_all_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "m0a/glm_image").mkdir(parents=True)
            (root / "m0a/model_profiles.json").write_text(
                (ROOT / "m0a/model_profiles.json").read_text()
            )
            for name in ("manifest.yaml", "glm_image/m0a_sfa_selected_trace.py",
                         "glm_image/sfa_v1.py"):
                path = root / "m0a" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n")
            pairs = root / "m0a/workloads/glm53_tiny.synthetic.pairs.json"
            pairs.parent.mkdir(parents=True)
            pairs.write_text(json.dumps(pair_set()))
            baseline = root / "m0a/baselines/glm53_tiny.json"
            for mode in ("baseline", "pilot", "pairs"):
                plan = build_plan(
                    root, mode=mode, run_id=f"glm-{mode}",
                    pairs_path=pairs if mode != "baseline" else None,
                    baseline_path=baseline, repetitions=1,
                    profile_name="glm53_tiny", device=3,
                )
                self.assertEqual(plan["devices"], [3])
                self.assertEqual(plan["model_profile"], "glm53_tiny")
                self.assertEqual(plan["container"], "memecho-m0a-glm53-trace")

    def test_glm_profile_rejects_unequal_pair_lengths(self):
        data = pair_set()
        data['pairs'][0]['control']['prompt_tokens_expected'] -= 1
        with self.assertRaisesRegex(ValueError, 'requires equal pair lengths'):
            validate_pair_policy(data, self.profile)


class GLM53RecorderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        envs = types.SimpleNamespace(
            VLLM_ASCEND_M0A_TRACE_DIR="",
            VLLM_ASCEND_M0A_TRACE_POSITIONS="",
            VLLM_ASCEND_M0A_RUN_ID="glm-run",
            VLLM_ASCEND_M0A_MODEL_PROFILE="glm53_tiny",
            VLLM_ASCEND_M0A_MODEL_ID="inference-optimization/GLM-5.3-0.6B-A0.4B",
            VLLM_ASCEND_M0A_MODEL_REVISION="20aae340d157bd2215b9d81165be87e94686dfdf",
        )
        package = types.ModuleType("vllm_ascend")
        package.envs = envs
        sys.modules["vllm_ascend"] = package
        path = ROOT / "m0a/source/m0a_sfa_selected_trace.py"
        spec = importlib.util.spec_from_file_location("m0a_sfa_trace", path)
        cls.trace = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.trace)
        cls.envs = envs

    def test_disabled_is_noop_and_enabled_preserves_native_rows(self):
        self.trace.record_sfa_prefill_selected(
            object(), object(), layer_name="model.layers.0", rank=0,
            selection_source="computed",
        )
        with tempfile.TemporaryDirectory() as directory:
            self.envs.VLLM_ASCEND_M0A_TRACE_DIR = directory
            self.envs.VLLM_ASCEND_M0A_TRACE_POSITIONS = "300:301"
            pad = [-1] * 2045
            topk = FakeTopK([
                [0, 127, -1, *pad],
                [128, 255, 300, *pad],
                [129, 256, 301, *pad],
            ])
            metadata = types.SimpleNamespace(
                trace_request_ids=("request-1",),
                trace_positions_cpu=[299, 300, 301],
                trace_seq_lens_cpu=[302], dsa_cp_context=None, block_size=128,
            )
            self.trace.record_sfa_prefill_selected(
                topk, metadata, layer_name="model.layers.0", rank=0,
                selection_source="computed",
            )
            rows = [json.loads(line) for line in
                    (Path(directory) / "rank0.jsonl").read_text().splitlines()]
        self.assertEqual([row["prompt_position"] for row in rows], [300, 301])
        self.assertEqual(rows[0]["logical_block_ids"][:3], [1, 1, 2])
        self.assertEqual(rows[0]["selected_width"], 2048)
        self.assertEqual(topk.rows[1][:3], [128, 255, 300])


if __name__ == "__main__":
    unittest.main()
