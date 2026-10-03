"""Process and supervisor ownership rules for the eight-NPU Pod runtime."""
from __future__ import annotations

import fcntl
import ctypes.util
import hashlib
import importlib.metadata
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from .preflight import check
from .working_set import require

SERVICE = 'memecho-deepseek'
PORT = 8900
ENV_KEYS = ('OMP_PROC_BIND', 'OMP_NUM_THREADS', 'PYTORCH_NPU_ALLOC_CONF',
            'LD_PRELOAD', 'HCCL_BUFFSIZE', 'TASK_QUEUE_ENABLE',
            'HCCL_OP_EXPANSION_MODE', 'VLLM_ASCEND_ENABLE_FLASHCOMM1')
ENV_VALUES = {'OMP_PROC_BIND': 'false', 'OMP_NUM_THREADS': '10',
              'PYTORCH_NPU_ALLOC_CONF': 'expandable_segments:True',
              'HCCL_BUFFSIZE': '1024', 'TASK_QUEUE_ENABLE': '1',
              'HCCL_OP_EXPANSION_MODE': 'AIV', 'VLLM_ASCEND_ENABLE_FLASHCOMM1': '1'}
EXPECTED_VERSIONS = {'vllm': '0.23.0+empty', 'vllm-ascend': '0.23.0',
                     'torch': '2.10.0+cpu', 'torch-npu': '2.10.0.post4'}
SAVED_METADATA_SHA256 = {
    'config.json': 'c3137398b85ef26b9592debadc424349f0b949b7ebe2f20603dda12efab4d533',
    'tokenizer.json': '8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf',
    'tokenizer_config.json': '6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547',
    'generation_config.json': '5fccff80f55a4d455bbe516bdd552edf3e9623df95e99fbf2a3c3389fdf91af0',
    'configuration.json': 'a78221ae93dd8977b26fcd1d106fa5ad7e7f1fd3a7d93bfd9d6fba5ed00ec4d2',
    'quant_model_description.json': '612d87d0a9a7545fb19f8cbaff75af3cc467d16ada72d2b6d84d22149a511cf0',
    'quant_model_weights.safetensors.index.json': '932abcc237e82bbb52fc044ae52dd7aa9d8af259cd40100ef978e479231164aa',
}


def service_command(model_dir: Path) -> list[str]:
    """The preserved TP=8, W8A8, MTP and DSA CP deployment arguments."""
    return ['vllm', 'serve', str(model_dir), '--host', '127.0.0.1', '--port', str(PORT),
            '--served-model-name', 'dsv4', '--tensor-parallel-size', '8',
            '--data-parallel-size', '1', '--enable-expert-parallel', '--quantization', 'ascend',
            '--tokenizer-mode', 'deepseek_v4', '--tool-call-parser', 'deepseek_v4',
            '--reasoning-parser', 'deepseek_v4', '--enable-auto-tool-choice',
            '--no-enable-prefix-caching', '--max-model-len', '133120',
            '--max-num-batched-tokens', '8192', '--max-num-seqs', '32',
            '--gpu-memory-utilization', '0.90', '--block-size', '128',
            '--model-loader-extra-config', '{"enable_multithread_load":true,"num_threads":128}',
            '--speculative-config', '{"num_speculative_tokens":1,"method":"mtp","enforce_eager":true}',
            '--compilation-config', '{"cudagraph_mode":"FULL_DECODE_ONLY"}',
            '--additional-config', '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"enable_dsa_cp":true,"enable_flashcomm1":true,"multistream_overlap_shared_expert":true}']


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def model_inventory(model_dir: Path) -> dict:
    require(model_dir.is_dir() and model_dir.as_posix().startswith('/models/'),
            'Model must be a directory under read-only /models')
    required = tuple(SAVED_METADATA_SHA256)
    metadata = {}
    for name in required:
        path = model_dir / name
        require(path.is_file() and path.stat().st_size > 0, 'Missing model metadata: ' + name)
    index = json.loads((model_dir / 'quant_model_weights.safetensors.index.json').read_text())
    shards = sorted(set(index['weight_map'].values()))
    require(len(shards) == 70 and all(re.fullmatch(r'[^/]+\.safetensors', name) for name in shards),
            'Expected exactly 70 indexed weight shards')
    for name in shards:
        path = model_dir / name
        require(path.is_file() and path.stat().st_size > 0, 'Missing weight shard: ' + name)
    for path in sorted(model_dir.iterdir()):
        if path.is_file() and path.suffix != '.safetensors':
            metadata[path.name] = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
    require(all(metadata[name]['sha256'] == digest for name, digest in SAVED_METADATA_SHA256.items()),
            'Model metadata differs from the preserved 2026-09-30 copy')
    return {'model_dir': str(model_dir), 'metadata': metadata,
            'weights': {name: (model_dir / name).stat().st_size for name in shards},
            'preserved_metadata_sha256': SAVED_METADATA_SHA256,
            'read_only_mount': bool(os.statvfs(model_dir).f_flag & os.ST_RDONLY),
            'verification_scope': 'Preserved metadata SHA-256 and indexed shard presence/size; no original per-shard hashes'}


def process_identity(pid: int) -> dict | None:
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        if fields[0] == 'Z':
            return None
        return {'pid': pid, 'start_ticks': int(fields[19]),
                'pgrp': int(fields[2]),
                'cmdline': Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip()}
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def same_process(saved: dict) -> bool:
    current = process_identity(saved['pid'])
    return bool(current and current['start_ticks'] == saved['start_ticks']
                and current['pgrp'] == saved['pgrp'] and current['cmdline'] == saved['cmdline'])


def owned_pids(run_id: str) -> list[int]:
    """Find only processes carrying the exact experiment marker in their environment."""
    marker = ('MEMECHO_VALIDATION_RUN=' + run_id).encode()
    result = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if marker in (entry / 'environ').read_bytes().split(b'\0') and process_identity(int(entry.name)):
                result.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return result


def stop_owned(run_id: str, saved: dict | None, *, timeout: float = 30) -> list[int]:
    require(re.fullmatch(r'deepseek_\d{8}T\d{6}Z_[0-9a-f]{8}', run_id), 'Invalid run ID')
    targets = set(owned_pids(run_id))
    if saved and same_process(saved):
        targets.add(saved['pid'])
    # A process that has lost its marker must never be killed by a stale PID file.
    targets &= set(owned_pids(run_id))
    for pid in targets:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and any(process_identity(pid) for pid in targets):
        time.sleep(.2)
    remaining = set(owned_pids(run_id))
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    require(not owned_pids(run_id), 'Owned experiment processes remain')
    return sorted(targets | remaining)


def port_available(port: int = PORT) -> bool:
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(('127.0.0.1', port))
        except OSError:
            return False
    with socket.socket() as connection:
        connection.settimeout(.2)
        return connection.connect_ex(('127.0.0.1', port)) != 0


def preflight(model_dir: Path, *, expect_port_free: bool, check_hardware: bool = True) -> dict:
    inventory = model_inventory(model_dir)
    require(inventory.get('read_only_mount') is True, 'Model directory is not mounted read-only')
    require(port_available() == expect_port_free, 'Port 8900 ownership does not match expected service state')
    shm = os.statvfs('/dev/shm')
    require(shm.f_frsize * shm.f_bavail >= 16 * 1024**3, 'Less than 16 GiB shared memory available')
    versions = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
    require(versions == EXPECTED_VERSIONS, 'Pod vLLM/Ascend/PyTorch versions differ from the fixed runtime')
    inventory.update(port_free=expect_port_free, shm_available_bytes=shm.f_frsize * shm.f_bavail,
                     versions=versions, command=service_command(model_dir))
    if check_hardware:
        inventory['hbm_used_mb'] = check(tuple(range(8)), 65536 if not expect_port_free else 6000)
        import torch_npu
        require(torch_npu.npu.device_count() == 8, 'PyTorch does not see eight NPUs')
        require(os.environ.get('LD_PRELOAD') or ctypes.util.find_library('jemalloc'),
                'The preserved launch requires libjemalloc.so.2')
    return inventory


def health(model_dir: Path, *, timeout: float = 900) -> list[tuple[str, str]]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            with opener.open(f'http://127.0.0.1:{PORT}/health', timeout=10) as response:
                require(response.status == 200, 'Health endpoint failed')
            with opener.open(f'http://127.0.0.1:{PORT}/v1/models', timeout=10) as response:
                models = [(x['id'], x['root']) for x in json.load(response)['data']]
            require(models == [('dsv4', str(model_dir))], 'Served model identity mismatch')
            return models
        except (OSError, ValueError, KeyError) as error:
            last = error
            time.sleep(5)
    raise TimeoutError('Pod service health timeout: ' + str(last))


class SupervisorService:
    def __init__(self, root: Path, model_dir: Path):
        self.root, self.model_dir = Path(root), Path(model_dir)
        self.runtime = self.root / 'runtime' / 'deepseek-pod'
        self.conf = self.runtime / 'supervisord.conf'

    def environment(self) -> dict[str, str]:
        result = dict(ENV_VALUES)
        preload = os.environ.get('LD_PRELOAD') or ctypes.util.find_library('jemalloc')
        if preload:
            result['LD_PRELOAD'] = preload
        return result

    def install(self) -> Path:
        self.runtime.mkdir(parents=True, exist_ok=True)
        cmd = service_command(self.model_dir)
        require(all('\n' not in item and '\r' not in item for item in cmd), 'Unsafe supervisor command')
        env = ','.join(f'{key}="{value}"' for key, value in self.environment().items())
        config = (f'[unix_http_server]\nfile={self.runtime}/supervisor.sock\nchmod=0700\n'
                  f'[supervisord]\nlogfile={self.runtime}/supervisord.log\npidfile={self.runtime}/supervisord.pid\n'
                  'nodaemon=false\n[supervisorctl]\n'
                  f'serverurl=unix://{self.runtime}/supervisor.sock\n'
                  f'[program:{SERVICE}]\ncommand={shlex.join(cmd)}\ndirectory={self.root}\n'
                  f'environment={env}\nautostart=false\nautorestart=true\n'
                  'startsecs=10\nstopasgroup=true\nkillasgroup=true\nstopwaitsecs=60\n'
                  f'stdout_logfile={self.runtime}/service.log\nredirect_stderr=true\n')
        self.conf.write_text(config)
        self.conf.chmod(0o600)
        (self.runtime / 'service-command.json').write_text(json.dumps({
            'command': cmd, 'environment': self.environment(), 'model_dir': str(self.model_dir),
            'config_sha256': sha256(self.conf)}, indent=2) + '\n')
        return self.conf

    def control(self, action: str) -> str:
        require(action in ('status', 'start', 'stop'), 'Unsupported supervisor action')
        self.install()
        socket_path = self.runtime / 'supervisor.sock'
        if not socket_path.exists():
            require(action == 'start', 'Supervisor is not running')
            subprocess.run(['supervisord', '-c', str(self.conf)], check=True, timeout=30)
        result = subprocess.run(['supervisorctl', '-c', str(self.conf), action, SERVICE],
                                capture_output=True, text=True, timeout=90)
        require(result.returncode == 0, 'supervisorctl failed: ' + result.stdout + result.stderr)
        return result.stdout.strip()

    def identity(self) -> dict:
        status = self.control('status')
        match = re.search(r'\bRUNNING\s+pid\s+(\d+)', status)
        require(match is not None, 'Service is not RUNNING: ' + status)
        identity = process_identity(int(match[1]))
        require(identity is not None and 'vllm' in identity['cmdline'], 'Supervisor process identity mismatch')
        return identity


class ExclusiveLock:
    def __init__(self, path: Path):
        self.path = path
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('a+')
        fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return self

    def __exit__(self, *_):
        fcntl.flock(self.stream, fcntl.LOCK_UN)
        self.stream.close()
