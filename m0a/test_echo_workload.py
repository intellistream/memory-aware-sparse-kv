"""Fixed-token schedule checks without an NPU or live agent service."""
import json
import tempfile
import unittest
from pathlib import Path

from m0a.echo_workload import seal_agent_cases, schedule, sha_ids


class EchoWorkloadTests(unittest.TestCase):
    def test_seals_all_pairs_and_reuses_order_across_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = []
            responses = []
            for index in range(48):
                pair = {"pair_id": f"pair-{index}", "workload_id": "retail",
                        "episode_id": f"episode-{index}"}
                for variant in ("event", "control"):
                    prompt = [1, index + 2, 3 if variant == "event" else 4]
                    pair[variant] = {"prompt_token_ids": prompt}
                    responses.append({
                        "pair_id": pair["pair_id"], "variant": variant,
                        "repetition": 0, "prompt_token_ids": prompt,
                        "prompt_token_ids_sha256": sha_ids(prompt),
                        "signature": {"token_ids": [5, 6, 7]},
                        "request_id": f"{index}-{variant}",
                    })
                pairs.append(pair)
            (root / "pairs.json").write_text(json.dumps({
                "pairs": pairs, "tokenizer_json_sha256": "a" * 64}))
            (root / "responses.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in responses))
            cases = seal_agent_cases(root / "pairs.json", root / "responses.jsonl", 2)
            self.assertEqual(len(cases), 96)
            plan = schedule(cases, concurrencies=(1, 4), repeats=1)
            self.assertEqual(len(plan), 96 * 3 * 2)
            first = [row["case_id"] for row in plan if row["concurrency"] == 1 and
                     row["mode"] == "no_offload"]
            second = [row["case_id"] for row in plan if row["concurrency"] == 1 and
                      row["mode"] == "offload_echo_prefetch"]
            self.assertEqual(first, second)
            self.assertNotIn("prompt_token_ids", plan[0])
            responses[0]["prompt_token_ids"][0] = 9
            (root / "responses.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in responses))
            with self.assertRaisesRegex(ValueError, "differs"):
                seal_agent_cases(root / "pairs.json", root / "responses.jsonl", 2)


if __name__ == "__main__":
    unittest.main()
