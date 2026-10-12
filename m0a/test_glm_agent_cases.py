import json
import unittest
from collections import Counter
from pathlib import Path

from m0a.echo_workload import sha_ids
from m0a.glm_agent_cases import glm_messages


class GLMAgentCasesTest(unittest.TestCase):
    def test_sealed_48_inputs_and_source_hash(self):
        path = Path(__file__).parent / "artifacts/glm_tau3_event_cases.json"
        source = json.loads(path.read_text())
        self.assertEqual(source["source_pairs_sha256"],
                         "3ec0dd5e14a7192aebe294c406670db5237890aa9e6bc116ee61cb56de8afdb7")
        cases = source["cases"]
        self.assertEqual(len(cases), 48)
        self.assertEqual(Counter(case["domain"] for case in cases),
                         {"retail": 24, "banking": 24})
        self.assertEqual(len({(case["domain"], case["episode_id"]) for case in cases}), 48)
        for case in cases:
            self.assertEqual(sha_ids(case["prompt_token_ids"]),
                             case["prompt_token_ids_sha256"])
            self.assertEqual(len(case["prompt_token_ids"]), case["prompt_tokens"])
            self.assertLessEqual(case["prompt_tokens"], 65536)
            self.assertGreaterEqual(case["prompt_tokens"] - case["event_position"], 32)

    def test_tool_arguments_normalized_without_mutating_source(self):
        messages = [{"role": "assistant", "tool_calls": [{"function": {
            "name": "lookup", "arguments": '{"id": 5}'}}]}]
        converted = glm_messages(messages)
        self.assertEqual(converted[0]["tool_calls"][0]["function"]["arguments"], {"id": 5})
        self.assertEqual(messages[0]["tool_calls"][0]["function"]["arguments"], '{"id": 5}')


if __name__ == "__main__":
    unittest.main()
