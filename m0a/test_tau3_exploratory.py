"""Sealed input and drift-tolerant τ³ engineering gates."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from m0a.contracts import sha256_file
from m0a.deepseek_pod_validation import PodWorker
from m0a.deepseek_validation import write_json
from m0a.import_workloads import sha_ids
from m0a.model_profiles import load_profile
from m0a.tau3_reuse import reuse_sealed_pairs
from scripts.finalize_deepseek_validation import acceptance_gates


RUN_ID = 'deepseek_20261006T000000Z_1234abcd'


def pair(index: int) -> dict:
    item = {'prompt': 'hello', 'prompt_token_ids': [1, 2],
            'prompt_tokens_expected': 2,
            'prompt_token_ids_sha256_expected': sha_ids([1, 2]),
            'boundary_position': 1}
    return {'pair_id': f'pair-{index}', 'event': item, 'control': item,
            'context_target': 8192}


class ExploratoryTests(unittest.TestCase):
    def worker(self, root: Path, exploratory: bool = True) -> PodWorker:
        directory = root / 'm0a/runs' / RUN_ID
        directory.mkdir(parents=True)
        write_json(directory / 'launch.json', {
            'runtime': 'pod', 'validation_mode': 'trace-replay',
            'model_dir': '/models/DeepSeek-V4-Flash-W8A8',
            'workload': 'tau3_v1.0.1', 'reuse_tau3_run': 'archived',
            'tau_root': str(root / 'tau'),
            'exploratory_drift': exploratory})
        worker = PodWorker(root, RUN_ID)
        worker.original = {'Config': {'Cmd': ['vllm', 'serve', 'model']}}
        return worker

    def test_sealed_reuse_preserves_pair_bytes_and_rejects_tokenizer(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source, target = root / 'source', root / 'target'
            source.mkdir()
            data = b'{"tokenizer_json_sha256":"' + b'a' * 64 + b'","pairs":[]}\n'
            (source / 'old-pairs.json').write_bytes(data)
            for filename in ('episodes.json', 'collection-summary.json',
                             'rejected-episodes.json', 'event-audit.json',
                             'context-feasibility.json'):
                (source / filename).write_text('{}')
            (source / 'tau3-provenance.json').write_text(json.dumps({
                'commit': 'pinned', 'data_sha256': 'data'}))
            (source / 'tau3-source-inputs-manifest.json').write_text('{}')
            manifest = {'source_run_id': 'archived', 'old_pairs_sha256': sha256_file(source / 'old-pairs.json'),
                        'source_final_manifest_sha256': 'f' * 64}
            with patch('m0a.tau3_reuse.verify_snapshot', return_value=manifest), \
                 patch('m0a.tau3_workload.verify_tau', return_value={
                     'commit': 'pinned', 'data_sha256': 'data'}):
                with self.assertRaisesRegex(ValueError, 'tokenizer hash mismatch'):
                    reuse_sealed_pairs(source, target, root, 'b' * 64)
                reuse_sealed_pairs(source, target, root, 'a' * 64)
            self.assertEqual((target / 'pairs.json').read_bytes(), data)
            self.assertEqual(json.loads((target / 'source-lineage.json').read_text())['regenerated'], [])

    def test_exploration_records_all_192_requests_per_phase_and_drift(self):
        with tempfile.TemporaryDirectory() as name:
            worker = self.worker(Path(name))
            pairs = {'pairs': [pair(i) for i in range(48)]}
            calls = 0
            def respond(_payload):
                nonlocal calls
                calls += 1
                token = 3 if calls % 2 else 4
                return {'id': f'request-{calls}', 'prompt_token_ids': [1, 2],
                        'usage': {'prompt_tokens': 2, 'completion_tokens': 1},
                        'choices': [{'token_ids': [token], 'finish_reason': 'stop',
                                     'message': {'content': str(token)}}]}
            with patch.object(worker, 'start_candidate') as started, \
                 patch.object(worker, 'cleanup'), patch.object(worker, 'wait_release'), \
                 patch('m0a.deepseek_validation.request', side_effect=respond):
                baseline = worker.stable_workload_baseline(pairs, load_profile('deepseek_v4'))
                trace = worker.requests('trace_on', pairs, load_profile('deepseek_v4'), baseline,
                                        soft_output_differences=True)
            self.assertEqual(len(baseline), 192)
            self.assertEqual(len(trace), 192)
            self.assertEqual(calls, 384)
            self.assertEqual(started.call_args.args[0]['command'], worker.original['Config']['Cmd'])
            self.assertEqual(json.loads((worker.directory / 'selected-config.json').read_text())['diagnostic_requests'], 0)
            self.assertTrue(json.loads((worker.directory / 'trace_off-output-differences.json').read_text()))
            self.assertTrue(json.loads((worker.directory / 'trace_on-output-differences.json').read_text()))
            self.assertFalse(json.loads((worker.directory / 'trace_on-failures.json').read_text()))

    def test_hard_input_error_stops_exploration(self):
        with tempfile.TemporaryDirectory() as name:
            worker = self.worker(Path(name))
            response = {'id': 'bad', 'prompt_token_ids': [9, 9],
                        'usage': {'prompt_tokens': 2, 'completion_tokens': 1},
                        'choices': [{'token_ids': [3], 'finish_reason': 'stop',
                                     'message': {'content': '3'}}]}
            with patch.object(worker, 'start_candidate'), patch.object(worker, 'cleanup'), \
                 patch.object(worker, 'wait_release'), \
                 patch('m0a.deepseek_validation.request', return_value=response):
                with self.assertRaisesRegex(ValueError, 'request_or_token_validation'):
                    worker.stable_workload_baseline({'pairs': [pair(0)]}, load_profile('deepseek_v4'))
            self.assertEqual(len((worker.directory / 'trace_off-api.jsonl').read_text().splitlines()), 1)

    def test_report_separates_engineering_from_strict_acceptance(self):
        with tempfile.TemporaryDirectory() as name:
            worker = self.worker(Path(name))
            write_json(worker.directory / 'pairs.json', {'pairs': [pair(0)]})
            write_json(worker.directory / 'restoration.json', {'restored': True})
            write_json(worker.directory / 'tau3-precheck.json', {'summary': {'pairs': 48}})
            write_json(worker.directory / 'trace-validation.json', {
                'repeat_selected_ids_identical': False,
                'pair_prefix_selected_ids_identical': True,
                'paired_effect_interpretation': 'exploratory'})
            write_json(worker.directory / 'trace_off-output-differences.json', [{'kind': 'repeat'}])
            worker.report('engineering_validated')
            report = json.loads((worker.directory / 'report.json').read_text())
            self.assertTrue(report['engineering_evidence_validated'])
            self.assertFalse(report['engineering_acceptance'])
            self.assertFalse(report['complete_acceptance'])
            self.assertEqual(report['strict_output_acceptance'], 'failed')
            self.assertEqual(report['event_locality_conclusion'], 'exploratory')
            state = {'hash_sync': 'verified', 'service_health': 'verified',
                     'output_acceptance': report['strict_output_acceptance']}
            self.assertEqual(acceptance_gates(0, state, {**report, 'missing_artifacts': []}),
                             (True, False))
            self.assertEqual(acceptance_gates(0, {**state, 'output_acceptance': 'passed'},
                {**report, 'missing_artifacts': [], 'strict_output_acceptance': 'passed'}),
                (True, False))


if __name__ == '__main__':
    unittest.main()
