import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from m0a.contracts import sha256_file
from scripts.finalize_deepseek_validation import finalize_run


class FinalizeDeepSeekTest(unittest.TestCase):
    def test_writes_run_local_terminal_evidence_without_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "launch.json").write_text(json.dumps({"run_id": "test-run"}))
            (run / "report.json").write_text(json.dumps({
                "status": "engineering_validated",
                "engineering_evidence_validated": True,
                "restoration": {"restored": True},
                "strict_output_acceptance": "passed",
                "event_locality_conclusion": "qualified",
                "paired_effect_interpretation": "qualified_for_offline_comparison",
            }))
            (run / "final-local-verification.json").write_text("{}")
            (run / "final-checksums.json").write_text(json.dumps({"files": [
                {"path": "report.json", "sha256": sha256_file(run / "report.json")}
            ]}))
            with patch("scripts.finalize_deepseek_validation.status_remote", return_value={
                "stage": "finished", "worker_alive": False, "watchdog_alive": False
            }), patch("scripts.finalize_deepseek_validation.service_check", return_value={
                "healthy": True, "owned_processes_remaining": []
            }):
                self.assertEqual(finalize_run("host", "/remote/run", run, guardian_exit_code=0), 0)
            status = json.loads((run / "final-status.json").read_text())
            self.assertTrue(status["engineering_acceptance"])
            self.assertTrue(status["complete_acceptance"])
            self.assertEqual(status["publication"], "not_requested")
            self.assertEqual(status["push"], "not_requested")
            self.assertTrue((run / "final-report.md").is_file())


if __name__ == "__main__":
    unittest.main()
