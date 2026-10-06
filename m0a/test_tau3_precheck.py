from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from m0a.import_workloads import sha_ids
from m0a.tau3_precheck import FILES, create, verify_seal
from m0a.tau3_workload import DOMAINS, EVENTS, TARGETS
from m0a.deepseek_pod_validation import PodWorker
from scripts.launch_deepseek_validation import checked_bootstrap


def fixture(directory: Path) -> None:
    pairs = []
    for domain in DOMAINS:
        for target in TARGETS:
            for kind in EVENTS:
                for chain in (0, 1):
                    index = len(pairs)
                    item = {'prompt': '[]', 'messages': [{'role': 'user', 'content': str(index)}],
                            'boundary_position': 1, 'prompt_tokens_expected': 2,
                            'prompt_token_ids_sha256_expected': sha_ids([1, 2]),
                            'prompt_token_ids': [1, 2]}
                    pairs.append({'pair_id': f'{domain}-{target}-{kind}-{chain}',
                                  'workload_id': 'tau3_v1.0.1_' + domain,
                                  'episode_id': str(index), 'context_target': target,
                                  'event_type': kind,
                                  'history_task_ids': [f'{domain}-{target}-{chain}-{kind}-a',
                                                       f'{domain}-{target}-{chain}-{kind}-b'],
                                  'event': item, 'control': item})
    (directory / 'pairs.json').write_text(json.dumps({
        'schema_version': 1, 'tokenizer_json_sha256': 'a' * 64, 'pairs': pairs}))
    for name in FILES:
        if name != 'pairs.json':
            (directory / name).write_text('{}')


class PrecheckTests(unittest.TestCase):
    def test_sealed_handoff_rechecks_every_prompt_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            inputs = directory / 'source-inputs'
            inputs.mkdir()
            (inputs / 'tau3-source-inputs-manifest.json').write_text(
                json.dumps({'source_run_id': 'old-run',
                            'source_final_manifest_sha256': 'b' * 64,
                            'old_pairs_sha256': 'c' * 64}))
            fixture(directory)
            with patch('m0a.tau3_precheck.regenerate', return_value=None), \
                 patch('m0a.tau3_precheck.verify_snapshot', return_value={
                     'source_run_id': 'old-run', 'source_final_manifest_sha256': 'b' * 64}), \
                 patch('m0a.tau3_precheck.request', return_value={'tokens': [1, 2]}) as tokenize:
                seal = create(directory, directory, 'a' * 64)
                self.assertEqual(seal['summary']['pairs'], 48)
                self.assertEqual(seal['summary']['tokenized_prompts'], 96)
                self.assertEqual(tokenize.call_count, 96)
                self.assertEqual(verify_seal(directory, 'a' * 64)['status'], 'passed')
                self.assertEqual(tokenize.call_count, 192)
                (directory / 'event-audit.json').write_text('tampered')
                with self.assertRaisesRegex(ValueError, 'sealed file changed'):
                    verify_seal(directory, 'a' * 64, live_tokens=False)

    def test_pair_generation_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            with patch('m0a.tau3_precheck.verify_snapshot', return_value={}), \
                 patch('m0a.tau3_precheck.regenerate', side_effect=ValueError('no feasible context')):
                with self.assertRaisesRegex(ValueError, 'no feasible context'):
                    create(directory, directory, 'a' * 64)
            failure = json.loads((directory / 'tau3-precheck-failure.json').read_text())
            self.assertEqual(failure['step'], 'pair_generation')
            self.assertFalse((directory / 'tau3-precheck.json').exists())

    def test_worker_uses_sealed_pairs_without_regeneration(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            fixture(directory)
            (directory / 'identity.json').write_text(json.dumps({
                'metadata': {'tokenizer.json': {'sha256': 'a' * 64}}}))
            worker = object.__new__(PodWorker)
            worker.directory = directory
            worker.workload = 'tau3_v1.0.1'
            worker.reuse_source_run = 'old-run'
            worker.phase_deadline = None
            worker.phase_label = None
            with patch.object(worker, 'update'), patch.object(worker, 'command') as command, \
                 patch('m0a.tau3_precheck.verify_seal', return_value={'source_run_id': 'old-run'}):
                result = worker.prepare()
            self.assertEqual(len(result['pairs']), 48)
            command.assert_not_called()

    def test_failed_remote_precheck_blocks_worker_start(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            (directory / 'launch.json').write_text(json.dumps({'reuse_tau3_run': 'old-run'}))
            calls = []
            def remote(_host, code, **_kwargs):
                calls.append(code)
                if len(calls) == 1:
                    raise RuntimeError('precheck failed')
                return b'{"tau3-precheck-failure.json":{"step":"tokenizer_and_coverage"}}'
            with patch('scripts.launch_deepseek_validation.remote', side_effect=remote):
                with self.assertRaisesRegex(RuntimeError, 'precheck failed'):
                    checked_bootstrap('host', 'bootstrap', directory, '/remote/run')
            self.assertEqual(len(calls), 2)
            failure = json.loads((directory / 'launch-failure.json').read_text())
            self.assertEqual(failure['remote_evidence']['tau3-precheck-failure.json']['step'],
                             'tokenizer_and_coverage')


if __name__ == '__main__':
    unittest.main()
