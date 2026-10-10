"""Adversarial checks for exact tokens, native coverage, process isolation and sync barriers."""
import copy
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from model_profiles import load_profile
from import_workloads import sha_ids
from single_npu_validation import (Worker, export_sidecar, extend_pairs, validate_api_result,
                                   validate_layout, validate_native_trace, window_ranges, write_json)
from transition_replay import read_trace, run_replay, validate_sidecar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.launch_single_npu_validation import (alive, detached_process, extract_checked,
    retry_sync, safe_relative, synchronize_checkpoint, qualified_final_status)


def chat(text):
    return [0] + list(text.encode()) + [999]


def source_pairs():
    profile = load_profile('glm53_tiny')
    source = {'schema_version': 1, 'tokenizer_json_sha256': '0' * 64, 'model_profile': 'glm53_tiny', 'pairs': []}
    for context in (8192, 32768):
        for i in range(6):
            pair = {'pair_id': f'{context}-{i}', 'workload_id': 'synthetic', 'episode_id': f'{context}-{i}',
                    'context_target': context, 'event_type': 'tool_call'}
            for v, text in [('event', 'abcE'), ('control', 'abcC')]:
                ids = chat(text)
                pair[v] = {'prompt': text, 'boundary_position': 4, 'prompt_tokens_expected': len(ids),
                           'prompt_token_ids_sha256_expected': sha_ids(ids)}
            source['pairs'].append(pair)
    return source, profile


def response_fixture(pairs, profile, raw):
    responses = []
    profile = dict(profile, selected_width=4)
    for pair in pairs['pairs']:
        for repetition in range(2):
            for variant in ('event', 'control'):
                stem = f'{pair["pair_id"]}-{variant}-{repetition}'
                item = pair[variant]
                responses.append({'request_id': stem, 'run_id': 'test-run', 'pair_id': pair['pair_id'],
                    'variant': variant, 'repetition': repetition, 'prompt_token_ids': item['prompt_token_ids'],
                    'prompt_tokens': len(item['prompt_token_ids']), 'boundary_position': 4})
    records = []
    for response in responses:
        for position in range(3, 36):
            for layer in range(6):
                source = 'computed' if layer < 3 else 'reused'
                records.append({'schema_version': 2, 'request_id': response['request_id'] + '-abc', 'run_id': 'test-run',
                    'rank': 0, 'layer': f'model.layers.{layer}.self_attn.attn', 'prompt_position': position,
                    'context_len': position + 1, 'request_context_len': response['prompt_tokens'],
                    'raw_index_unit': 'kv_token_position', 'invalid_sentinel': -1, 'compression_ratio': 1,
                    'compressed_block_size': 128, 'logical_block_size': 128, 'selected_width': 4,
                    'raw_selected_ids': [0, 1, position, -1], 'logical_block_ids': [0, 0, 0, -1], 'timestamp_ns': 1,
                    'model_profile': 'glm53_tiny', 'model_id': profile['model_id'], 'model_revision': profile['model_revision'],
                    'attention_backend': 'sfa', 'native_operator': 'torch_npu.npu_lightning_indexer' if layer < 3 else 'index_cache_reuse',
                    'selection_source': source})
    raw.mkdir()
    (raw / 'rank0.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    return responses, records, profile


class SingleNPUValidationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.profile = source_pairs()
        self.pairs = extend_pairs(self.source, chat, self.profile)

    def test_suffix_completes_window_keeps_prefix_hash_and_source(self):
        before = copy.deepcopy(self.source)
        for original, pair in zip(self.source['pairs'], self.pairs['pairs']):
            self.assertEqual(pair['event']['prompt'].removeprefix(original['event']['prompt']), pair['neutral_suffix'])
            self.assertEqual(pair['control']['prompt'].removeprefix(original['control']['prompt']), pair['neutral_suffix'])
            for v in ('event', 'control'):
                item = pair[v]
                self.assertGreaterEqual(len(item['prompt_token_ids']) - item['boundary_position'], 32)
                self.assertEqual(item['prompt_token_ids_sha256_expected'], sha_ids(item['prompt_token_ids']))
                self.assertEqual(item['prompt_token_ids'][:4], chat(original[v]['prompt'])[:4])
            self.assertEqual(pair['event']['prompt_tokens_expected'], pair['control']['prompt_tokens_expected'])
        self.assertEqual(self.source, before)

    def test_old_token_hash_and_unequal_inputs_fail(self):
        self.source['pairs'][0]['event']['prompt_token_ids_sha256_expected'] = 'x' * 64
        with self.assertRaisesRegex(ValueError, 'hash'):
            extend_pairs(self.source, chat, self.profile)
        self.source, _ = source_pairs()
        self.source['pairs'][0]['event']['prompt_tokens_expected'] += 1
        with self.assertRaises(ValueError):
            extend_pairs(self.source, chat, self.profile)

    def test_identical_suffix_cannot_hide_unequal_tokenizer(self):
        def bad(text):
            ids = chat(text)
            return ids + ([1] if len(text) > 4 and text.startswith('abcE') else [])
        with self.assertRaisesRegex(ValueError, 'equal lengths'):
            extend_pairs(self.source, bad, self.profile)

    def test_api_exact_ids_output_equivalence_and_generated_count(self):
        item = self.pairs['pairs'][0]['event']
        result = {'prompt_token_ids': item['prompt_token_ids'], 'usage': {'prompt_tokens': len(item['prompt_token_ids']), 'completion_tokens': 2},
                  'choices': [{'message': {'content': 'text'}, 'token_ids': [8, 9], 'finish_reason': 'length'}]}
        sig = validate_api_result(result, item)
        validate_api_result(result, item, sig, sig)
        with self.assertRaisesRegex(ValueError, 'Trace-on/off'):
            validate_api_result(result, item, expected_signature=dict(sig, content='other'))
        with self.assertRaisesRegex(ValueError, 'Repeated output'):
            validate_api_result(result, item, previous_signature=dict(sig, token_ids=[8, 10]))
        bad = copy.deepcopy(result)
        bad['prompt_token_ids'][0] = 123
        with self.assertRaisesRegex(ValueError, 'prompt token'):
            validate_api_result(bad, item)
        result['usage']['completion_tokens'] = 3
        with self.assertRaisesRegex(ValueError, 'Generated'):
            validate_api_result(result, item)

    def test_full_native_coverage_and_semantic_sidecar(self):
        responses, _, profile = response_fixture(self.pairs, self.profile, self.root / 'raw')
        evidence = validate_native_trace(self.root / 'raw', self.root / 'traces', responses, window_ranges(self.pairs), profile)
        self.assertEqual(evidence['window_rows'], 48 * 33 * 6)
        sidecar = export_sidecar(responses, self.pairs, self.profile)
        snapshots, groups = validate_sidecar(sidecar, self.root)
        self.assertEqual(len(groups), 48)
        for snapshot in snapshots.values():
            self.assertEqual(len(snapshot['lanes']), 6)
            self.assertEqual({r['type'] for r in snapshot['regions']}, {'other'})
            self.assertTrue(all(l['generated_ids'] == list(range(snapshot['generated_tokens'])) for l in snapshot['lanes']))
        self.assertTrue(sidecar['synthetic'])
        self.assertEqual(sidecar['selection_provenance'], 'real_npu_native')
        self.assertFalse(sidecar['online_kv_restore'])
        # Width in this small fixture is deliberately four; production requires 2048.
        with patch('transition_replay.load_profile', return_value=profile):
            read_trace(self.root / 'traces', snapshots, sidecar['events'], synthetic=False)

    def test_missing_layer_source_causal_and_repeat_checks(self):
        responses, records, profile = response_fixture(self.pairs, self.profile, self.root / 'raw')
        path = self.root / 'raw/rank0.jsonl'
        for label in ('missing', 'source', 'causal', 'repeat'):
            rows = copy.deepcopy(records)
            if label == 'missing':
                rows.pop()
            elif label == 'source':
                rows[0]['selection_source'] = 'reused'
            elif label == 'causal':
                rows[0]['raw_selected_ids'][0] = 10000
                rows[0]['logical_block_ids'][0] = 10000 // 128
            else:
                index = 2 * 33 * 6 + 6
                rows[index]['raw_selected_ids'][0] = 2
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            with self.subTest(label=label), self.assertRaises(ValueError):
                validate_native_trace(self.root / 'raw', self.root / label, responses, window_ranges(self.pairs), profile)

    def test_grouped_original_pairs_episodes_sources_and_repeats(self):
        responses, _, _ = response_fixture(self.pairs, self.profile, self.root / 'raw')
        data = export_sidecar(responses, self.pairs, self.profile)
        validate_sidecar(data, self.root)
        for field in ('pair_id', 'episode_id', 'source_trace_id'):
            bad = copy.deepcopy(data)
            bad['events'][24][field] = bad['events'][0][field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'leakage'):
                validate_sidecar(bad, self.root)
        data['events'][1]['split'] = 'eval'
        with self.assertRaisesRegex(ValueError, 'leakage'):
            validate_sidecar(data, self.root)

    def test_bf16_layout_and_source_required(self):
        config = {'dtype': 'bfloat16', 'kv_lora_rank': 512, 'qk_rope_head_dim': 64,
                  'num_hidden_layers': 6, 'indexer_types': ['full'] * 3 + ['shared'] * 3}
        mla = 'self.head_size = kv_lora_rank + qk_rope_head_dim\nself.num_kv_heads = 1\ndtype=kv_cache_dtype'
        sfa = 'return (num_blocks, block_size, num_kv_heads, head_size)'
        self.assertEqual(validate_layout(config, mla, sfa)['native_kv_payload_bytes_per_layer'], 1152)
        for field, value in [('dtype', 'float16'), ('kv_lora_rank', 511), ('quantization_config', {'x': 1})]:
            with self.assertRaisesRegex(ValueError, 'layout mismatch'):
                validate_layout(dict(config, **{field: value}), mla, sfa)
        with self.assertRaisesRegex(ValueError, 'source'):
            validate_layout(config, '', sfa)

    def test_sidecar_token_hash_and_real_npu_layers_cannot_be_bypassed(self):
        responses, _, _ = response_fixture(self.pairs, self.profile, self.root / 'raw')
        data = export_sidecar(responses, self.pairs, self.profile)
        bad = copy.deepcopy(data)
        bad['snapshots'][0]['token_ids'][0] += 1
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            validate_sidecar(bad, self.root)
        for snapshot in data['snapshots']:
            snapshot['lanes'] = snapshot['lanes'][:1]
        config = {'schema_version': 1, 'capacity_bytes': 64*1024**2, 'prefetch_budget_bytes': 8*1024**2,
                  'kv_unit_bytes': {'kv_token_position': 1152}}
        # Remove every other layer in the trace as well, so trace linkage cannot mask the layer check.
        rows = [json.loads(line) for line in (self.root / 'raw/rank0.jsonl').read_text().splitlines()]
        for row in rows:
            row['raw_selected_ids'] += [-1]*2044
            row['logical_block_ids'] += [-1]*2044
            row['selected_width'] = 2048
        (self.root / 'raw/rank0.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows if r['layer']=='model.layers.0.self_attn.attn'))
        with self.assertRaisesRegex(ValueError, 'layer coverage'):
            run_replay(self.root / 'raw', data, config, self.root)

    def test_http_failure_response_is_preserved(self):
        worker = Worker(self.root, 'single_npu_20260929T000000Z_12345678')
        worker.directory.mkdir(parents=True)
        error = urllib.error.HTTPError('http://localhost', 500, 'failed', {}, io.BytesIO(b'{"error":"native failure"}'))
        with patch('single_npu_validation.request', side_effect=error), self.assertRaises(urllib.error.HTTPError):
            worker.requests('trace_off', self.pairs, self.profile)
        failure = json.loads((worker.directory / 'trace_off-api.jsonl').read_text())
        self.assertEqual(failure['http_status'], 500)
        self.assertIn('native failure', failure['response_body'])

    def test_background_process_owns_session_and_logs(self):
        log = self.root / 'child.log'
        process = detached_process([sys.executable, '-c', 'import os,time; print(os.getpid(),os.getsid(0),flush=True); time.sleep(0.2)'], log)
        self.assertTrue(alive(process.pid))
        self.assertEqual(process.wait(timeout=3), 0)
        pid, session = map(int, log.read_text().split())
        self.assertEqual(pid, session)
        self.assertFalse(alive(pid))

    def test_three_sync_failures_stop_retries(self):
        calls = []
        def broken():
            calls.append(1)
            raise IOError('unreachable')
        with self.assertRaisesRegex(RuntimeError, '3 attempts'):
            retry_sync(broken, delay=0)
        self.assertEqual(len(calls), 3)

    def test_checksums_reject_tampering_and_unsafe_archives(self):
        def archive(name, data):
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode='w') as out:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                out.addfile(info, io.BytesIO(data))
            stream.seek(0)
            return stream
        records = [{'path': 'response.json', 'size': 3, 'sha256': hashlib.sha256(b'abc').hexdigest()}]
        extract_checked(archive('response.json', b'abc'), records, self.root)
        self.assertEqual((self.root / 'response.json').read_bytes(), b'abc')
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            extract_checked(archive('response.json', b'bad'), records, self.root)
        with self.assertRaisesRegex(ValueError, 'Unexpected'):
            extract_checked(archive('../escape', b'abc'), records, self.root)
        for name in ('../escape', '/absolute'):
            with self.assertRaises(ValueError):
                safe_relative(name)

    def test_barrier_accepts_exact_ack_and_fails_on_sync_error(self):
        worker = Worker(self.root, 'single_npu_20260929T000000Z_12345678')
        worker.directory.mkdir(parents=True)
        artifact = worker.directory / 'file.json'
        artifact.write_text('data')
        def acknowledge():
            manifest = worker.directory / 'checkpoints/00-test.json'
            for _ in range(100):
                if manifest.exists():
                    break
                time.sleep(0.01)
            data = json.loads(manifest.read_text())
            write_json(worker.directory / 'acks/00-test.json', {'run_id': worker.run_id,
                'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(), 'files': data['files']})
        thread = threading.Thread(target=acknowledge)
        thread.start()
        worker.checkpoint('test', [artifact])
        thread.join()
        self.assertEqual(worker.state['last_synced_checkpoint'], '00-test')
        write_json(worker.directory / 'sync-failed.json', {'error': 'unreachable'})
        with self.assertRaisesRegex(ValueError, 'three attempts'):
            worker.checkpoint('next', [artifact])

    def test_sync_only_acknowledges_after_verification(self):
        directory = self.root / 'test-run'
        directory.mkdir()
        manifest = json.dumps({'run_id': 'test-run', 'files': [{'path': 'data', 'size': 1, 'sha256': '0'*64}]}).encode()
        with patch('scripts.launch_single_npu_validation.remote', return_value=manifest) as ssh, \
                patch('scripts.launch_single_npu_validation.download_records', side_effect=ValueError('hash mismatch')):
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                synchronize_checkpoint('hust', '/remote', directory, '00-test', hashlib.sha256(manifest).hexdigest())
            self.assertEqual(ssh.call_count, 1)
            self.assertFalse((directory / 'acks/00-test.json').exists())

    def test_cleanup_only_owns_labelled_containers(self):
        worker = Worker(self.root, 'single_npu_20260929T000000Z_12345678')
        worker.directory.mkdir(parents=True)
        worker.owned.add('task-container')
        other = [{'Config': {'Labels': {'memecho.validation.run': 'other'}}}]
        with patch('single_npu_validation.subprocess.run', return_value=subprocess.CompletedProcess([], 0, json.dumps(other), '')), \
                patch.object(worker, 'command') as command:
            with self.assertRaisesRegex(ValueError, 'cleanup failed'):
                worker.cleanup()
            command.assert_not_called()
        owned = [{'Config': {'Labels': {'memecho.validation.run': worker.run_id}}}]
        results = [subprocess.CompletedProcess([], 0, json.dumps(owned), ''),
                   subprocess.CompletedProcess([], 1, '', 'Error: No such object: task-container')]
        with patch('single_npu_validation.subprocess.run', side_effect=results), patch.object(worker, 'command') as command:
            worker.cleanup()
            command.assert_called_once_with(['docker', 'rm', '-f', 'task-container'], timeout=60)
        with patch('single_npu_validation.subprocess.run', return_value=subprocess.CompletedProcess([], 1, '', 'daemon unreachable')):
            with self.assertRaisesRegex(ValueError, 'cleanup failed'):
                worker.cleanup()

    def test_dry_run_does_not_create_directory_or_connect(self):
        result = subprocess.run([sys.executable, 'scripts/launch_single_npu_validation.py', '--device', '0'],
                                cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, check=True)
        plan = json.loads(result.stdout)
        self.assertFalse(plan['executed'])
        self.assertFalse(Path(plan['local_directory']).exists())
        self.assertEqual(plan['device'], 0)

    def test_engineering_status_requires_restoration_archive_and_full_capture(self):
        with tempfile.TemporaryDirectory() as name:
            directory=Path(name)/'deepseek_20260929T000000Z_12345678'
            archive=directory/'local-archive'
            archive.mkdir(parents=True)
            bundle=archive/'final-artifacts.bundle'
            bundle.write_bytes(b'bundle')
            digest=hashlib.sha256(bundle.read_bytes()).hexdigest()
            write_json(archive/'final-code-provenance.json',{'git_commit':'a'*40,'bundle_sha256':digest})
            write_json(directory/'final-local-verification.json',{'run_id':directory.name,'git_commit':'a'*40})
            restoration={'restored':True,'healthy':True,'config_unchanged':True,'eight_npu_healthy':True}
            write_json(directory/'restoration.json',restoration)
            report={'status':'engineering_validated','validation_mode':'trace-replay',
                    'strict_output_acceptance':'not_qualified','missing_artifacts':[],
                    'completed_requests_per_phase':{'trace_off':48,'trace_on':48},'restoration':restoration}
            write_json(directory/'report.json',report)
            write_json(directory/'launch.json',{'exploratory_drift':False})
            paths={'report.json','restoration.json','trace-validation.json','sidecar.json',
                   'trace_off-output-differences.json','trace_on-output-differences.json',
                   'implementation.bundle','code-provenance.json','local-archive/final-artifacts.bundle',
                   'local-archive/final-code-provenance.json'}
            paths.update(f'replay-{scope}-{capacity}mib/report.json'
                         for scope in ('per_rank','aggregate') for capacity in (64,128))
            write_json(directory/'final-checksums.json',{'files':[{'path':path} for path in paths]})
            status={'status':'engineering_validated','original_service_restored':True}
            self.assertTrue(qualified_final_status(directory,status))
            status['original_service_restored']=False
            with self.assertRaisesRegex(ValueError,'restoration'):
                qualified_final_status(directory,status)
            status['original_service_restored']=True
            report['completed_requests_per_phase']['trace_on']=47
            write_json(directory/'report.json',report)
            with self.assertRaisesRegex(ValueError,'Incomplete requests'):
                qualified_final_status(directory,status)


if __name__ == '__main__':
    unittest.main()
