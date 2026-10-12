import unittest

from m0a.glm_online_stats import MODES, analyze


def fixture():
    cases = {"cases": []}
    rows = []
    for domain in ("retail", "banking"):
        for episode in range(24):
            case_id = f"{domain}:{episode}"
            case = {"case_id": case_id, "domain": domain, "episode_id": str(episode),
                    "prompt_tokens": 9000 + 100 * episode,
                    "prompt_token_ids_sha256": "a" * 64,
                    "forced_continuation_token_ids": list(range(32))}
            cases["cases"].append(case)
            for repeat in range(3):
                for mode in MODES:
                    delta = 1 if mode == "offload_echo_prefetch" else 0
                    rows.append({"case_id": case_id, "repeat": repeat, "mode": mode,
                                 "prompt_tokens": case["prompt_tokens"],
                                 "prompt_token_ids_sha256": "a" * 64,
                                 "forced_continuation_token_ids": list(range(32)),
                                 "fixed_continuation_verified": True,
                                 "native_topk_exact": True, "kv_payload_exact": True,
                                 "attention_equivalent": True, "prototype_hbm_shadow": True,
                                 "native_selected_set_sha256": "b" * 64,
                                 "prefill_window_ms": 10 + delta, "tpot_ms": 2 + delta,
                                 "ttft_ms": 100, "throughput_tps": 10,
                                 "prefetch_hit_bytes": 0, "prefetch_waste_bytes": 0,
                                 "synchronous_recall_bytes": 0})
    return cases, rows


class OnlineStatsTest(unittest.TestCase):
    def test_four_registered_directions_and_drift_gate(self):
        cases, rows = fixture()
        report = analyze(cases, rows, draws=1000)
        controlled = [result for result in report["results"]
                      if result["subset"] == "stable_controlled"]
        self.assertEqual(len(controlled), 4)
        self.assertTrue(all(result["directional_slowdown_supported"] for result in controlled))
        for row in rows:
            if row["case_id"] in {f"retail:{i}" for i in range(5)} and \
                    row["mode"] == "offload_echo_prefetch" and row["repeat"] == 0:
                row["native_selected_set_sha256"] = "c" * 64
        report = analyze(cases, rows, draws=1000)
        retail = [result for result in report["results"] if result["domain"] == "retail" and
                  result["subset"] == "stable_controlled"]
        self.assertEqual([result["episodes"] for result in retail], [19, 19])
        self.assertFalse(any(result["directional_slowdown_supported"] for result in retail))
        self.assertEqual(len(report["drift"]), 5)

    def test_wrong_forced_tokens_stop_analysis(self):
        cases, rows = fixture()
        rows[0]["forced_continuation_token_ids"][0] = 99
        with self.assertRaisesRegex(ValueError, "Fixed continuation"):
            analyze(cases, rows, draws=1000)


if __name__ == "__main__":
    unittest.main()
