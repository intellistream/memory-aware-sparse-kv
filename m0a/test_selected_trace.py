"""Small CPU-only checks for the research capture boundary."""

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


class FakeRow:
    def __init__(self, values):
        self.values = values

    def reshape(self, *_):
        return self

    def tolist(self):
        return self.values


class FakeTopK:
    def __init__(self, rows):
        self.rows = rows
        self.ndim = 2
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, item):
        if isinstance(item, slice):
            return FakeTopK(self.rows[item])
        return FakeRow(self.rows[item])

    def detach(self):
        return self

    def cpu(self):
        return self


class TraceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        envs = types.SimpleNamespace(
            VLLM_ASCEND_M0A_TRACE_DIR="",
            VLLM_ASCEND_M0A_TRACE_POSITIONS="",
            VLLM_ASCEND_M0A_RUN_ID="test-run",
        )
        package = types.ModuleType("vllm_ascend")
        package.envs = envs
        sys.modules["vllm_ascend"] = package
        path = Path(__file__).parent / "source" / "m0a_selected_trace.py"
        spec = importlib.util.spec_from_file_location("m0a_selected_trace", path)
        cls.trace = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.trace)
        cls.envs = envs

    def test_ranges_and_invalid_input(self):
        self.assertEqual(self.trace.parse_positions("2:3,8"), ((2, 3), (8, 8)))
        with self.assertRaises(ValueError):
            self.trace.parse_positions("5:4")

    def test_disabled_trace_does_not_touch_output_or_metadata(self):
        self.envs.VLLM_ASCEND_M0A_TRACE_DIR = ""
        self.trace.record_prefill_selected(
            object(), object(), layer_name="model.layers.3", rank=0
        )

    def test_raw_indices_and_block_mapping_preserve_output(self):
        with tempfile.TemporaryDirectory() as directory:
            self.envs.VLLM_ASCEND_M0A_TRACE_DIR = directory
            self.envs.VLLM_ASCEND_M0A_TRACE_POSITIONS = "10:11"
            pad = [-1] * 509
            topk = FakeTopK([[0, 127, -1, *pad],
                             [128, 255, 256, *pad],
                             [400, -1, -1, *pad]])
            metadata = types.SimpleNamespace(
                trace_request_ids=("request-1",),
                trace_positions_cpu=[9, 10, 11],
                trace_seq_lens_cpu=[12],
                cp_metadata=types.SimpleNamespace(local_start=0, local_end=3),
                block_size=128,
            )
            self.trace.record_prefill_selected(
                topk, metadata, layer_name="model.layers.3", rank=2
            )
            records = [json.loads(line) for line in (Path(directory) / "rank2.jsonl").read_text().splitlines()]
            self.assertEqual([record["prompt_position"] for record in records], [10, 11])
            self.assertEqual(records[0]["raw_selected_ids"][:3], [128, 255, 256])
            self.assertEqual(records[0]["logical_compressed_block_ids"][:3], [1, 1, 2])
            self.assertEqual(records[1]["logical_compressed_block_ids"][:3], [3, -1, -1])
            self.assertEqual(records[0]["compression_ratio"], 4)
            self.assertEqual(records[0]["selected_width"], 512)
            self.assertEqual(topk.rows[1][:3], [128, 255, 256])

    def test_enabled_trace_requires_run_id(self):
        with tempfile.TemporaryDirectory() as directory:
            self.envs.VLLM_ASCEND_M0A_TRACE_DIR = directory
            self.envs.VLLM_ASCEND_M0A_TRACE_POSITIONS = "10"
            self.envs.VLLM_ASCEND_M0A_RUN_ID = ""
            topk = FakeTopK([[0] + [-1] * 511])
            metadata = types.SimpleNamespace(
                trace_request_ids=("request-1",), trace_positions_cpu=[10],
                trace_seq_lens_cpu=[11],
                cp_metadata=types.SimpleNamespace(local_start=0), block_size=128,
            )
            with self.assertRaisesRegex(RuntimeError, "run ID"):
                self.trace.record_prefill_selected(
                    topk, metadata, layer_name="model.layers.3", rank=0
                )
            self.envs.VLLM_ASCEND_M0A_RUN_ID = "test-run"


if __name__ == "__main__":
    unittest.main()
