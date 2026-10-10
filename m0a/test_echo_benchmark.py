"""Synthetic contracts for later fixed-token NPU benchmark analysis."""
import unittest

from m0a.echo_benchmark import analyze
from m0a.echo_runtime import Capabilities, Mode


def measurement(episode, workload_class, mode, *, ttft, throughput, workload_id=None):
    return {
        "episode_id": episode,
        "workload_id": workload_id or (
            "retail" if workload_class == "agent_tool" else "ordinary"),
        "workload_class": workload_class,
        "mode": mode.value,
        "repeat": 0,
        "concurrency": 4,
        "capacity_bytes": 64 * 1024**2,
        "decode_tokens": 32,
        "prompt_token_ids_sha256": episode + "-prompt",
        "forced_continuation_sha256": episode + "-continuation",
        "native_selected_set_sha256": episode + "-native",
        "fixed_continuation_verified": True,
        "native_selected_set_exact": True,
        "capabilities": {key: True for key in Capabilities.__dataclass_fields__},
        "ttft_ms": ttft,
        "tpot_ms": 20.0,
        "throughput_tps": throughput,
        "synchronous_recall_bytes": 1024,
        "host_to_device_bytes": 2048,
    }


class BenchmarkTests(unittest.TestCase):
    def fixture(self):
        rows = []
        for workload_class in ("agent_tool", "ordinary"):
            for index in range(40 if workload_class == "agent_tool" else 20):
                episode = workload_class + f"-{index}"
                workload_id = ("retail" if index < 20 else "banking") if (
                    workload_class == "agent_tool") else "ordinary"
                rows.append(measurement(episode, workload_class,
                                        Mode.OFFLOAD_NO_PREFETCH,
                                        ttft=100.0, throughput=50.0,
                                        workload_id=workload_id))
                rows.append(measurement(episode, workload_class,
                                        Mode.OFFLOAD_ECHO_PREFETCH,
                                        ttft=110.0 if workload_class == "agent_tool" else 100.0,
                                        throughput=45.0 if workload_class == "agent_tool" else 50.0,
                                        workload_id=workload_id))
        return rows

    def test_agent_specific_direction_is_only_qualified_with_controls(self):
        report = analyze(self.fixture(), draws=100, seed=4)
        self.assertEqual(report["input_rows"], 120)
        self.assertTrue(report["interactions"][0]["agent_domains_consistent"])
        self.assertTrue(report["interactions"][0][
            "agent_specific_latency_and_throughput_regression_supported"])
        self.assertEqual(report["interactions"][0]["effects"]["ttft_ms"]["agent_mean_delta"], 10.0)

    def test_no_offload_arm_is_reported_when_feasible(self):
        rows = self.fixture()
        for episode in ("agent_tool-0", "agent_tool-1"):
            rows.append(measurement(episode, "agent_tool", Mode.NO_OFFLOAD,
                                    ttft=90.0, throughput=55.0))
        report = analyze(rows, draws=100)
        agent = next(cell for cell in report["cells"] if cell["workload_id"] == "retail")
        self.assertEqual(agent["against_no_offload"]["ttft_ms"][
            "mean_echo_minus_no_offload"], 20.0)

    def test_selected_set_drift_is_reported_without_a_causal_claim(self):
        rows = self.fixture()
        rows[1]["native_selected_set_sha256"] = "drifted"
        report = analyze(rows, draws=100)
        self.assertEqual(report["interactions"][0]["selected_set_drift_repeats"], 1)
        self.assertFalse(report["interactions"][0][
            "agent_specific_latency_and_throughput_regression_supported"])

    def test_drift_or_reference_backend_cannot_be_called_causal_timing(self):
        rows = self.fixture()
        rows[1]["forced_continuation_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "different input"):
            analyze(rows, draws=100)
        rows = self.fixture()
        rows[1]["fixed_continuation_verified"] = False
        with self.assertRaisesRegex(ValueError, "fixed continuation"):
            analyze(rows, draws=100)
        rows = self.fixture()
        rows[1]["capabilities"]["fused_indexer_prefetch"] = False
        with self.assertRaisesRegex(RuntimeError, "fused NPU"):
            analyze(rows, draws=100)


if __name__ == "__main__":
    unittest.main()
