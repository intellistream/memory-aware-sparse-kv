"""Archive linkage and fail-closed decode import checks using generated data."""
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from m0a.history_glm_import import import_decode


def archive(root, *, bad_position=False):
    rows = []
    for layer in range(6):
        for phase, positions in (("prefill", range(2968, 3000)), ("decode", range(3000, 3031))):
            for position in positions:
                rows.append({"layer": f"layer.{layer}", "position": position,
                             "phase": phase, "selected_ids": list(range(2048))})
    if bad_position:
        rows[-1]["position"] += 1
    raw = "".join(json.dumps(r) + "\n" for r in rows).encode()
    compressed = gzip.compress(raw)
    (root / "selected").mkdir()
    (root / "selected" / "trace.gz").write_bytes(compressed)
    measurement = {"case_id": "case", "mode": "no_offload", "repeat": 0,
                   "native_selected_trace": "trace.gz", "native_selected_set_sha256": hashlib.sha256(raw).hexdigest(),
                   "prompt_tokens": 3000, "domain": "test", "episode_id": "episode",
                   "prompt_token_ids_sha256": "a" * 64, "event_type": "tool_result"}
    (root / "measurements.jsonl").write_text(json.dumps(measurement) + "\n")
    (root / "independent-audit.json").write_text(json.dumps({
        "passed": True, "run_id": "test", "selected_trace_sha256": {"trace.gz": hashlib.sha256(compressed).hexdigest()}}))


class HistoryImportTests(unittest.TestCase):
    def test_complete_decode_and_conservative_block_eligibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive(root)
            rows, manifest = import_decode(root, ["case"], block_tokens=128)
            self.assertEqual(len(rows), 186)
            self.assertEqual(rows[0]["selected_blocks"], list(range(16)))
            self.assertEqual(rows[0]["eligible_blocks"], list(range(23)))
            self.assertEqual(rows[0]["block_count"], 24)
            self.assertTrue(all(row["phase"] == "decode" for row in rows))
            self.assertTrue(manifest["limitations"])

    def test_modified_archive_and_missing_case_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive(root)
            with self.assertRaisesRegex(ValueError, "Missing/duplicate"):
                import_decode(root, ["absent"], block_tokens=1)
            with (root / "selected" / "trace.gz").open("ab") as output:
                output.write(b"changed")
            with self.assertRaisesRegex(ValueError, "differs from audit"):
                import_decode(root, ["case"], block_tokens=1)

    def test_missing_decode_position_is_rejected_even_with_matching_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive(root, bad_position=True)
            with self.assertRaisesRegex(ValueError, "decode positions"):
                import_decode(root, ["case"], block_tokens=1)


if __name__ == "__main__":
    unittest.main()
