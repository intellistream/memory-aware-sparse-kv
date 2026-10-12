import json
import unittest
from pathlib import Path

from m0a.glm_schedule import make_schedule


class GLMScheduleTest(unittest.TestCase):
    def test_archived_continuations_and_rotated_order(self):
        artifact = Path(__file__).parent / "artifacts/glm_tau3_calibration.json"
        cases = json.loads(artifact.read_text())
        schedule = make_schedule(cases)
        self.assertEqual(len(schedule), 432)
        self.assertEqual([schedule[i]["case_id"] for i in (0, 144, 288)],
                         [cases["cases"][0]["case_id"]] * 3)
        self.assertEqual([schedule[i]["mode"] for i in (0, 144, 288)],
                         ["no_offload", "offload_no_prefetch", "offload_echo_prefetch"])


if __name__ == "__main__":
    unittest.main()
