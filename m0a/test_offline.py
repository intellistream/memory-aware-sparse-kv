import csv
import json
import tempfile
import unittest
from pathlib import Path

from m0a.contracts import resolve_api_request_id, validate_trace
from m0a.orchestrate import build_plan
from m0a.summarize_profile import aggregate
from m0a.synthetic_trace import make_trace_record
from m0a.verify_baseline import verify


ROOT = Path(__file__).resolve().parents[1]


class OfflineTest(unittest.TestCase):
    def test_sealed_baseline_recomputes(self):
        result = verify(ROOT)
        self.assertEqual(result['matrix']['run_count'], 16)
        self.assertEqual(result['matrix']['completed'], 320)
        self.assertEqual(result['matrix']['failed'], 0)
        self.assertTrue(result['sanity']['identical'])

    def test_trace_contract_rejects_range_and_width(self):
        record = make_trace_record(
            run_id='run', request_id='request', rank=0,
            layer='model.layers.3', position=1023, raw_values=[128],
            request_context_len=1100,
        )
        broken = dict(record, raw_selected_ids=list(record['raw_selected_ids']))
        broken['raw_selected_ids'][0] = 9999
        with self.assertRaisesRegex(ValueError, 'causal compressed context'):
            validate_trace(broken)
        broken = dict(record, selected_width=511)
        with self.assertRaisesRegex(ValueError, 'width'):
            validate_trace(broken)

    def test_engine_request_id_maps_to_public_api_id(self):
        api = 'chatcmpl-85f05cf8ceb8ddea'
        self.assertEqual(
            resolve_api_request_id(api + '-a69f2001', {api}), api
        )
        self.assertIsNone(resolve_api_request_id('chatcmpl-other-deadbeef', {api}))

    def test_orchestration_dry_plan_is_complete_and_side_effect_free(self):
        run_id = 'dry-run-test'
        run_dir = ROOT / 'm0a/runs' / run_id
        self.assertFalse(run_dir.exists())
        plan = build_plan(
            ROOT, mode='pairs', run_id=run_id,
            pairs_path=ROOT / 'm0a/pairs.json', repetitions=3,
        )
        self.assertEqual(plan['status'], 'planned')
        self.assertEqual(plan['trace_positions'], '8126:8138,32696:32708')
        self.assertFalse(run_dir.exists())
        self.assertEqual(len(plan['commands']), 5)

    def test_profile_aggregation_requires_and_combines_eight_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for rank in range(8):
                out = root / f'dp0_rank{rank}_test' / 'ASCEND_PROFILER_OUTPUT'
                out.mkdir(parents=True)
                (out / 'analyse.done').write_text('done\n')
                with (out / 'step_trace_time.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=[
                        'Computing', 'Communication', 'Communication(Not Overlapped)',
                        'Overlapped', 'Free', 'Stage', 'Preparing'])
                    writer.writeheader()
                    writer.writerow({
                        'Computing': 100 + rank, 'Communication': 200 + rank,
                        'Communication(Not Overlapped)': 180, 'Overlapped': 20,
                        'Free': 300, 'Stage': 600, 'Preparing': 1,
                    })
                with (out / 'op_statistic.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=[
                        'OP Type', 'Count', 'Total Time(us)'])
                    writer.writeheader()
                    writer.writerow({'OP Type': 'VllmQuantLightningIndexer',
                                     'Count': 10, 'Total Time(us)': 20})
                with (out / 'operator_details.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=[
                        'Name', 'Device Self Duration(us)', 'Device Total Duration(us)'])
                    writer.writeheader()
                    writer.writerow({'Name': 'vllm::dsa_forward',
                                     'Device Self Duration(us)': 3,
                                     'Device Total Duration(us)': 4})
            result = aggregate(root)
            self.assertEqual(result['rank_count'], 8)
            self.assertEqual(result['top_op_types'][0]['count'], 80)
            self.assertEqual(result['target_operators'][0]['operator_count'], 8)


if __name__ == '__main__':
    unittest.main()
