"""Pod adapter checks that do not require an NPU or a listening socket."""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from m0a.deepseek_pod_validation import PodWorker
from m0a.pod_runtime import (EXPECTED_VERSIONS, SupervisorService, model_inventory, owned_pids,
                             preflight, process_identity, service_command, stop_owned)
from scripts.launch_single_npu_validation import download_records


RUN_ID = 'deepseek_20261003T000000Z_1234abcd'


class PodRuntimeTests(unittest.TestCase):
    def test_original_command_and_supervisor_config(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            model = Path('/models/DeepSeek-V4-Flash-W8A8')
            service = SupervisorService(root, model)
            path = service.install()
            config = path.read_text()
            command = service_command(model)
            self.assertEqual(command[command.index('--tensor-parallel-size') + 1], '8')
            self.assertEqual(command[command.index('--quantization') + 1], 'ascend')
            self.assertEqual(json.loads(command[command.index('--speculative-config') + 1])['method'], 'mtp')
            self.assertTrue(json.loads(command[command.index('--additional-config') + 1])['enable_dsa_cp'])
            self.assertIn('autorestart=true', config)
            self.assertIn('[rpcinterface:supervisor]', config)
            self.assertIn('stopasgroup=true', config)
            self.assertEqual(json.loads((path.parent / 'service-command.json').read_text())['command'], command)

    def test_model_inventory_requires_all_indexed_shards(self):
        with tempfile.TemporaryDirectory() as name:
            model = Path(name)
            for file in ('config.json', 'tokenizer.json'):
                (model / file).write_text('{}')
            shards = [f'part-{i:02d}.safetensors' for i in range(70)]
            (model / 'quant_model_weights.safetensors.index.json').write_text(json.dumps(
                {'weight_map': {f'weight.{i}': shard for i, shard in enumerate(shards)}}))
            for shard in shards:
                (model / shard).write_bytes(b'weight')
            with patch.object(Path, 'as_posix', return_value='/models/fake'), \
                 patch('m0a.pod_runtime.SAVED_METADATA_SHA256', {}):
                self.assertEqual(len(model_inventory(model)['weights']), 70)
                (model / shards[-1]).unlink()
                with self.assertRaisesRegex(ValueError, 'Missing weight shard'):
                    model_inventory(model)

    def test_preflight_port_and_devices_fail_closed(self):
        model = Path('/models/fake')
        with patch('m0a.pod_runtime.model_inventory', return_value={'model_dir': str(model), 'read_only_mount': True}), \
             patch('m0a.pod_runtime.port_available', return_value=False), \
             patch('m0a.pod_runtime.os.statvfs') as stat, \
             patch('m0a.pod_runtime.importlib.metadata.version', side_effect=EXPECTED_VERSIONS.__getitem__), \
             patch('m0a.pod_runtime.check', return_value={i: 0 for i in range(8)}), \
             patch('m0a.pod_runtime.ctypes.util.find_library', return_value='libjemalloc.so.2'), \
             patch.dict(sys.modules, torch_npu=type('NPU', (), {'npu': type('Device', (), {'device_count': staticmethod(lambda: 8)})()})()):
            stat.return_value.f_frsize = 4096
            stat.return_value.f_bavail = 16 * 1024**3 // 4096
            with self.assertRaisesRegex(ValueError, 'Port 8900'):
                preflight(model, expect_port_free=True)
            result = preflight(model, expect_port_free=False)
            self.assertEqual(result['hbm_used_mb'], {i: 0 for i in range(8)})

    def test_only_marked_processes_are_stopped(self):
        env = dict(os.environ, MEMECHO_VALIDATION_RUN=RUN_ID)
        owned = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], env=env)
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        try:
            time.sleep(.1)
            identity = process_identity(owned.pid)
            self.assertIsNotNone(identity)
            self.assertIn(owned.pid, owned_pids(RUN_ID))
            self.assertNotIn(unrelated.pid, owned_pids(RUN_ID))
            stopped = stop_owned(RUN_ID, identity, timeout=2)
            self.assertIn(owned.pid, stopped)
            self.assertIsNone(unrelated.poll())
        finally:
            for child in (owned, unrelated):
                if child.poll() is None:
                    child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)

    def test_recovery_restarts_only_the_supervised_service(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            directory = root / 'm0a/runs' / RUN_ID
            directory.mkdir(parents=True)
            model = Path('/models/DeepSeek-V4-Flash-W8A8')
            (directory / 'launch.json').write_text(json.dumps(
                {'runtime': 'pod', 'validation_mode': 'trace-replay', 'model_dir': str(model)}))
            worker = PodWorker(root, RUN_ID)
            worker.original = {'Config': {'Cmd': service_command(model)}}
            worker.restore_required = True
            calls = []
            worker.service.control = calls.append
            worker.service.conf.parent.mkdir(parents=True)
            worker.service.conf.write_text('fixed')
            with patch.object(worker, 'cleanup'), patch.object(worker, 'wait_release'), \
                 patch.object(worker, 'service_health', return_value=[('dsv4', str(model))]), \
                 patch.object(worker, 'inspect', return_value={'ProcessIdentity': {'pid': 1}}):
                worker.restore()
            self.assertEqual(calls, ['start'])
            self.assertTrue(json.loads((directory / 'restoration.json').read_text())['restored'])

    def test_large_artifact_manifest_is_streamed_to_tar(self):
        records = [{'path': f'trace-python/file-{i:05d}.py', 'size': 1, 'sha256': '0' * 64}
                   for i in range(12000)]
        with patch('scripts.launch_single_npu_validation.subprocess.run') as run, \
             patch('scripts.launch_single_npu_validation.extract_checked'):
            download_records('hust', '/remote/run', records, '/local/run')
        args, kwargs = run.call_args
        self.assertLess(len(args[0][-1]), 200)
        self.assertIn('--dereference', args[0][-1])
        self.assertIn('--hard-dereference', args[0][-1])
        self.assertIn(b'trace-python/file-11999.py\0', kwargs['input'])
        self.assertEqual(kwargs['input'].count(b'\0'), len(records))


if __name__ == '__main__':
    unittest.main()
