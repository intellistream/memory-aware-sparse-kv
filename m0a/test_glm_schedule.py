import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from m0a.echo_workload import sha_ids
from m0a.glm_schedule import make_schedule, verified_schedule


class GLMScheduleTest(unittest.TestCase):
    def test_constructed_continuations_and_rotated_order(self):
        cases = {"cases": [{"case_id": f"case-{i}",
                            "forced_continuation_token_ids": list(range(i, i + 32)),
                            "forced_continuation_sha256": sha_ids(list(range(i, i + 32)))}
                           for i in range(48)]}
        schedule = make_schedule(cases)
        self.assertEqual(len(schedule), 432)
        self.assertEqual([schedule[i]["case_id"] for i in (0, 144, 288)],
                         [cases["cases"][0]["case_id"]] * 3)
        self.assertEqual([schedule[i]["mode"] for i in (0, 144, 288)],
                         ["no_offload", "offload_no_prefetch", "offload_echo_prefetch"])
        source = json.dumps(cases).encode()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sealed.json"
            path.write_text(json.dumps({"cases_sha256": hashlib.sha256(source).hexdigest(),
                                        "requests": schedule}))
            self.assertEqual(verified_schedule(source, path), schedule)
            path.write_text(json.dumps({"cases_sha256": "wrong", "requests": schedule}))
            with self.assertRaisesRegex(ValueError, "schedule drift"):
                verified_schedule(source, path)


if __name__ == "__main__":
    unittest.main()
