from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tarfile
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from m0a.tau3_reuse import build_archive, source_records, unpack_archive, verify_snapshot
from scripts.launch_single_npu_validation import download_records, retry_sync, synchronize_checkpoint


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class SourceReuseTests(unittest.TestCase):
    def test_full_source_inventory_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root / 'deepseek_source'
            source.mkdir()
            episodes = [{'domain': domain, 'task_id': f'{index:03d}'}
                        for domain in ('retail', 'banking_knowledge') for index in range(48)]
            content = {'episodes.json': json.dumps(episodes).encode(),
                       'collection-summary.json': b'{}', 'rejected-episodes.json': b'[]',
                       'tau3-provenance.json': b'{}', 'pairs.json': b'{"old":true}'}
            content.update({f'episode-{e["domain"]}-{e["task_id"]}.jsonl': b'original log\n'
                            for e in episodes})
            for filename, data in content.items():
                (source / filename).write_bytes(data)
            inventory = [{'path': filename, 'size': len(data), 'sha256': digest(data)}
                         for filename, data in content.items()]
            manifest = source / 'final-checksums.json'
            manifest.write_text(json.dumps({'files': inventory}))
            (source / 'final-status.json').write_text(json.dumps({
                'run_id': source.name, 'hash_sync': 'verified',
                'manifest_sha256': digest(manifest.read_bytes())}))
            target = root / 'new'
            target.mkdir()
            info = build_archive(source, target)
            unpack_archive(target / info['archive'], target / 'tau3-source-inputs-manifest.json',
                           target / 'source-inputs', info)
            self.assertEqual(len(verify_snapshot(target / 'source-inputs')['files']), len(content))
            self.assertTrue((target / 'source-inputs/old-pairs.json').exists())
            (source / 'episode-retail-000.jsonl').write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'Source file hash mismatch'):
                source_records(source)
            (target / 'source-inputs/episodes.json').write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'Source snapshot hash mismatch'):
                verify_snapshot(target / 'source-inputs')
            with (target / info['archive']).open('ab') as stream:
                stream.write(b'tampered')
            with self.assertRaisesRegex(ValueError, 'Source archive hash mismatch'):
                unpack_archive(target / info['archive'], target / 'tau3-source-inputs-manifest.json',
                               target / 'second-extraction', info)


class SynchronizationTests(unittest.TestCase):
    def test_small_batches_retry_disconnect_and_hash_skip(self):
        with tempfile.TemporaryDirectory() as name:
            local = Path(name)
            rows = [{'path': f'f{i}', 'size': 20 * 1024**2, 'sha256': digest(bytes([i]) * (20 * 1024**2))}
                    for i in range(3)]
            (local / 'f0').write_bytes(bytes([0]) * rows[0]['size'])
            attempts, batches = [], []

            def transfer(command, *, input, stdout, timeout, check):
                names = input.decode().strip('\0').split('\0')
                attempts.append(names)
                if len(attempts) == 1:
                    raise subprocess.CalledProcessError(255, command)
                batches.append(names)
                with tarfile.open(fileobj=stdout, mode='w') as archive:
                    for filename in names:
                        row = next(r for r in rows if r['path'] == filename)
                        data = bytes([int(filename[-1])]) * row['size']
                        item = tarfile.TarInfo(filename)
                        item.size = len(data)
                        archive.addfile(item, io.BytesIO(data))
                return subprocess.CompletedProcess(command, 0)

            events = []
            with patch('scripts.launch_single_npu_validation.subprocess.run', side_effect=transfer):
                retry_sync(lambda: download_records('host', '/remote', rows, local,
                                                     event=lambda stage, **details: events.append(stage)),
                           attempts=2, delay=0)
            self.assertEqual(batches, [['f1'], ['f2']])
            self.assertIn('hash_skip', events)
            self.assertIn('batch_verified', events)

    def test_late_checkpoint_keeps_failure_without_ack(self):
        with tempfile.TemporaryDirectory(prefix='deepseek_') as name:
            local = Path(name)
            run_id = local.name
            payload = b'evidence'
            record = {'path': 'evidence.txt', 'size': len(payload), 'sha256': digest(payload)}
            manifest = json.dumps({'run_id': run_id, 'files': [record]}).encode()
            calls = []

            def remote(_host, code, **_kwargs):
                calls.append(code)
                return manifest

            def download(_host, _server, records, directory, **_kwargs):
                (Path(directory) / records[0]['path']).write_bytes(payload)

            with patch('scripts.launch_single_npu_validation.remote', side_effect=remote), \
                 patch('scripts.launch_single_npu_validation.download_records', side_effect=download):
                synchronize_checkpoint('host', '/remote', local, '01-inputs', digest(manifest),
                                       write_ack=False)
            self.assertFalse((local / 'acks/01-inputs.json').exists())
            self.assertTrue((local / 'checkpoints/01-inputs.json').exists())
            self.assertEqual(len(calls), 1)

    def test_delayed_ack_is_written_only_after_verified_transfer(self):
        with tempfile.TemporaryDirectory(prefix='deepseek_') as name:
            local = Path(name)
            payload = b'verified'
            record = {'path': 'input.txt', 'size': len(payload), 'sha256': digest(payload)}
            manifest = json.dumps({'run_id': local.name, 'files': [record]}).encode()
            observed = []

            def remote(_host, code, **_kwargs):
                if 'read_bytes()' in code:
                    return manifest
                self.assertEqual((local / 'input.txt').read_bytes(), payload)
                time.sleep(0.01)
                observed.append('remote_ack')
                return b''

            def download(_host, _server, records, directory, **_kwargs):
                (Path(directory) / records[0]['path']).write_bytes(payload)
                observed.append('download')

            with patch('scripts.launch_single_npu_validation.remote', side_effect=remote), \
                 patch('scripts.launch_single_npu_validation.download_records', side_effect=download):
                synchronize_checkpoint('host', '/remote', local, '01-inputs', digest(manifest))
            self.assertEqual(observed, ['download', 'remote_ack'])
            self.assertTrue((local / 'acks/01-inputs.json').exists())


if __name__ == '__main__':
    unittest.main()
