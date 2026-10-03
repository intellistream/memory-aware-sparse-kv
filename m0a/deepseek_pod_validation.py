"""Pod execution of the existing DeepSeek trace and CPU replay contract."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from .contracts import sha256_file
from .deepseek_validation import (Worker, SOURCE_PATHS, TaskDeadline, TOTAL_SECONDS,
                                  extend_pairs, validate_layout, utc, write_json)
from .import_workloads import make_chat_tokenizer
from .model_profiles import load_profile
from .pod_runtime import (ExclusiveLock, SupervisorService, health, model_inventory,
                          owned_pids, port_available, preflight, process_identity,
                          same_process, service_command, sha256, stop_owned)
from .preflight import check
from .run_requests import request
from .working_set import require


class PodWorker(Worker):
    def __init__(self, root: Path, run_id: str):
        super().__init__(root, run_id)
        launch = json.loads((self.directory / 'launch.json').read_text())
        require(launch.get('runtime') == 'pod' and launch.get('validation_mode') == 'trace-replay',
                'Pod worker requires explicit trace-replay runtime')
        require(not self.repair and not self.diagnostic_only, 'Pod execution accepts trace-replay only')
        self.model_dir = Path(launch['model_dir'])
        self.service = SupervisorService(self.root, self.model_dir)
        self.env = dict(os.environ)
        self.experiment: dict | None = None
        self.trace_source: Path | None = None
        self.restore_required = False

    def checkpoint(self, stage, paths):
        extra = []
        if stage == 'bootstrap':
            extra = [self.directory / name for name in
                     ('pr-branch.bundle', 'branch-provenance.json', 'service-bootstrap.json')]
        elif stage == 'inputs':
            trace = self.directory / 'trace-python/vllm_ascend'
            extra = [self.directory / 'trace-source.json',
                     trace / 'envs.py', trace / 'attention/context_parallel/dsa_cp.py',
                     trace / 'attention/context_parallel/m0a_selected_trace.py',
                     trace / 'worker/model_runner_v1.py']
        return super().checkpoint(stage, list(paths) + extra)

    def command(self, args, *, timeout=900, output=None, env=None):
        # The shared replay driver requests historical Docker logs twice.
        if args[:2] == ['docker', 'logs']:
            source = (self.directory / 'trace_on-service.log' if 'trace_on' in self.state.get('stage', '')
                      else self.directory / 'trace-off-service/service.log')
            data = source.read_text() if source.exists() else ''
            if output:
                Path(output).write_text(data)
                return ''
            return data
        require(not args or args[0] != 'docker', 'Docker execution is disabled in Pod runtime')
        return super().command(args, timeout=timeout, output=output, env=env)

    def inspect(self, name):
        require(name == 'pod-service', 'Only the supervised Pod service may be inspected')
        identity = self.service.identity()
        return {'Id': 'pod-service', 'State': {'Running': True, 'Pid': identity['pid']},
                'Config': {'Cmd': service_command(self.model_dir),
                           'Env': [f'{k}={v}' for k, v in self.service.environment().items()]},
                'HostConfig': {'Runtime': 'pod'}, 'ProcessIdentity': identity}

    def service_health(self, name, expected_models=None):
        require(name in ('pod-service', self.container), 'Unknown service identity')
        if name == 'pod-service':
            self.service.identity()
        else:
            require(self.experiment and same_process(self.experiment), 'Experiment service exited')
        models = health(self.model_dir)
        require(models == (expected_models or [('dsv4', str(self.model_dir))]), 'Model list changed')
        check(tuple(range(8)), 65536)
        return models

    def _copy_and_patch_trace(self, package: Path) -> dict:
        target = self.directory / 'trace-python'
        target.mkdir()
        copied = target / 'vllm_ascend'
        shutil.copytree(package, copied, symlinks=True)
        patch = self.code_root / 'm0a/vllm-ascend-trace.patch'
        subprocess.run(['git', 'apply', '--check', str(patch)], cwd=target, check=True, timeout=30)
        subprocess.run(['git', 'apply', str(patch)], cwd=target, check=True, timeout=30)
        dsa = copied / 'attention/context_parallel/dsa_cp.py'
        runner = copied / 'worker/model_runner_v1.py'
        text = dsa.read_text()
        needle = 'req_metadata.trace_request_ids = kwargs.get("m0a_request_ids", ())'
        require(text.count(needle) == 1, 'Trace request metadata injection changed')
        dsa.write_text(text.replace(needle, needle + '\n        req_metadata.trace_prompt_lens_cpu = kwargs.get("m0a_prompt_lens", ())'))
        text = runner.read_text()
        needle = 'm0a_request_ids=('
        require(text.count(needle) == 1, 'Runner request metadata injection changed')
        runner.write_text(text.replace(needle,
            'm0a_prompt_lens=(self.input_batch.num_prompt_tokens_cpu_tensor[:num_reqs].tolist()\n'
            '                        if not for_cudagraph_capture else ()),\n                    ' + needle))
        for path in (dsa, runner, copied / 'attention/context_parallel/m0a_selected_trace.py'):
            compile(path.read_text(), str(path), 'exec')
        env = dict(os.environ, PYTHONPATH=str(target) + os.pathsep + os.environ.get('PYTHONPATH', ''))
        imported = subprocess.check_output([sys.executable, '-c',
            'import vllm_ascend, vllm_ascend.attention.context_parallel.m0a_selected_trace as t; '
            'print(vllm_ascend.__file__); print(t.__file__)'], env=env, text=True, timeout=60).splitlines()
        require(all(str(target) in value for value in imported), 'Trace Python import escaped isolated copy')
        files = [copied / 'attention/context_parallel/dsa_cp.py', copied / 'envs.py',
                 copied / 'worker/model_runner_v1.py', copied / 'attention/context_parallel/m0a_selected_trace.py']
        return {'source_package': str(package), 'trace_package': str(copied),
                'patch_sha256': sha256(patch), 'imported': imported,
                'patched_files': {str(p.relative_to(target)): sha256(p) for p in files}}

    def preflight(self):
        self.update(stage='preflight')
        require(self.service.identity(), 'Supervised service is not running')
        inventory = preflight(self.model_dir, expect_port_free=False)
        require(inventory['metadata']['tokenizer.json']['sha256'] ==
                json.loads((self.code_root / 'm0a/pairs.json').read_text())['tokenizer_json_sha256'],
                'Model tokenizer differs from saved input fixture')
        self.original = self.inspect('pod-service')
        write_json(self.directory / 'original-service.json', self.original)
        write_json(self.directory / 'original-models.json', self.service_health('pod-service'))
        write_json(self.directory / 'identity.json', inventory)
        spec = importlib.util.find_spec('vllm_ascend')
        require(spec and spec.submodule_search_locations, 'vLLM Ascend source import unavailable')
        package = Path(next(iter(spec.submodule_search_locations)))
        sources = {}
        source_dir = self.directory / 'sources'
        source_dir.mkdir()
        for name, old in SOURCE_PATHS.items():
            relative = old.split('/vllm_ascend/', 1)[1]
            origin = package / relative
            require(origin.is_file(), 'Missing installed source: ' + relative)
            destination = source_dir / name
            shutil.copy2(origin, destination)
            sources[name] = destination
        layout = validate_layout(json.loads((self.model_dir / 'config.json').read_text()), sources, self.directory)
        write_json(self.directory / 'layout.json', layout)
        trace = self._copy_and_patch_trace(package)
        write_json(self.directory / 'trace-source.json', trace)
        self.trace_source = Path(trace['trace_package']).parent
        self.command(['npu-smi', 'info'], output=self.directory / 'npu-before.txt', timeout=30)
        write_json(self.directory / 'preflight.json', {'at': utc(), 'runtime': 'pod', 'port': 8900,
                   'devices': list(range(8)), 'supervisor_config_sha256': sha256(self.service.conf),
                   'original_process': self.original['ProcessIdentity']})
        write_json(self.directory / 'worker-identity.json', process_identity(os.getpid()))
        with (self.directory / 'watchdog.log').open('ab') as output:
            child = subprocess.Popen([sys.executable, '-m', 'm0a.deepseek_pod_validation', '--watchdog',
                                      '--root', str(self.root), '--run-id', self.run_id],
                                     env=dict(os.environ, PYTHONPATH=str(self.code_root)),
                                     stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                     start_new_session=True, close_fds=True)
        write_json(self.directory / 'watchdog.json', {'pid': child.pid, 'identity': process_identity(child.pid)})
        self.update(watchdog_pid=child.pid)
        return layout

    def prepare(self):
        self.update(stage='prepare_inputs')
        profile = load_profile('deepseek_v4')
        source = self.code_root / 'm0a/pairs.json'
        data = json.loads(source.read_text())
        require(sha256_file(self.model_dir / 'tokenizer.json') == data['tokenizer_json_sha256'],
                'Pinned tokenizer hash mismatch')
        chat = make_chat_tokenizer(profile_name='deepseek_v4',
                                   tokenizer_path=self.model_dir / 'tokenizer.json', model_dir=self.model_dir)
        result = extend_pairs(data, chat, profile)
        result['source_file_sha256'] = sha256_file(source)
        write_json(self.directory / 'pairs.json', result)
        for pair in result['pairs']:
            for variant in ('event', 'control'):
                payload = {'model': 'dsv4', 'messages': [{'role': 'user', 'content': pair[variant]['prompt']}],
                           'add_generation_prompt': True}
                actual = request(payload, url='http://127.0.0.1:8900/tokenize')
                require(actual['tokens'] == pair[variant]['prompt_token_ids'], 'Actual tokenizer differs')
        (self.directory / 'prepare.log').write_text('Prepared and checked all prompt token IDs\n')
        return result

    def wait_release(self):
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if not owned_pids(self.run_id) and port_available():
                try:
                    usage = check(tuple(range(8)), 6000)
                    write_json(self.directory / 'release-check.json',
                               {'at': utc(), 'port_free': True, 'hbm_used_mb': usage, 'device_processes': []})
                    return
                except RuntimeError:
                    pass
            time.sleep(5)
        raise TimeoutError('Pod experiment did not release port and eight NPUs')

    def pause_original(self):
        require(same_process(self.original['ProcessIdentity']), 'Original process changed before pause')
        self.restore_required = True
        self.update(original_service_restored=False)
        write_json(self.directory / 'restore-required.json',
                   {'at': utc(), 'identity': self.original['ProcessIdentity']})
        self.service.control('stop')
        self.wait_release()

    def restore(self):
        self.update(stage='restore_original')
        self.cleanup()
        if self.restore_required:
            self.wait_release()
            # A new PID is expected; the saved supervisor command and config remain fixed.
            self.service.control('start')
        elif port_available():
            # Recovery may observe a stopped service before restore-required was written.
            self.wait_release()
            self.service.control('start')
        models = self.service_health('pod-service')
        require(service_command(self.model_dir) == self.original['Config']['Cmd'], 'Service command changed')
        saved_preflight = self.directory / 'preflight.json'
        if saved_preflight.exists():
            require(sha256(self.service.conf) == json.loads(saved_preflight.read_text())['supervisor_config_sha256'],
                    'Supervisor configuration changed')
        current = self.inspect('pod-service')
        self.restore_required = False
        write_json(self.directory / 'restoration.json', {'at': utc(), 'restored': True,
                   'container_id': 'pod-service', 'process_identity': current['ProcessIdentity'],
                   'healthy': True, 'model_identity': models, 'eight_npu_healthy': True,
                   'config_unchanged': True})
        self.update(original_service_restored=True)

    def cleanup(self):
        killed = stop_owned(self.run_id, self.experiment)
        self.experiment = None
        write_json(self.directory / 'resource-release.json',
                   {'at': utc(), 'runtime': 'pod', 'owned_pids_stopped': killed})

    def _launch(self, command: list[str], path: Path, *, trace=False, ranges=()):
        self.wait_release()
        env = dict(os.environ, **self.service.environment(), MEMECHO_VALIDATION_RUN=self.run_id)
        env.pop('VLLM_ASCEND_M0A_TRACE_DIR', None)
        if trace:
            require(self.trace_source is not None, 'Trace source missing')
            env['PYTHONPATH'] = str(self.trace_source) + os.pathsep + env.get('PYTHONPATH', '')
            env.update(VLLM_ASCEND_M0A_TRACE_DIR=str(self.directory / 'raw-traces'),
                       VLLM_ASCEND_M0A_RUN_ID=self.run_id,
                       VLLM_ASCEND_M0A_TRACE_POSITIONS=','.join(f'{a}:{b}' for a,b in ranges))
        output = path.open('ab')
        try:
            child = subprocess.Popen(command, env=env, cwd=self.root, stdin=subprocess.DEVNULL,
                                     stdout=output, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        finally:
            output.close()
        self.experiment = process_identity(child.pid)
        require(self.experiment, 'Experiment exited immediately')
        write_json(self.directory / 'experiment-process.json', self.experiment)
        return self.experiment

    def start_candidate(self, candidate, path, *, operator_diagnostics=False):
        require(not operator_diagnostics and candidate['id'] == 'original_fresh', 'Only original Pod configuration is supported')
        self.update(stage='trace_off_startup', candidate=candidate['id'])
        write_json(path / 'launch.json', {'runtime': 'pod', 'command': candidate['command'],
                   'configuration': candidate, 'trace_enabled': False})
        self._launch(candidate['command'], path / 'service.log')
        self.service_health(self.container)

    def start_trace(self, ranges):
        self.update(stage='trace_on_startup')
        trace_dir = self.directory / 'raw-traces'
        trace_dir.mkdir()
        write_json(self.directory / 'trace-launch.json',
                   {'runtime': 'pod', 'command': self.selected_config['command'],
                    'configuration': self.selected_config, 'original_command': self.original['Config']['Cmd'],
                    'trace_source_sha256': sha256(self.directory / 'trace-source.json'),
                    'positions': ranges})
        self._launch(self.selected_config['command'], self.directory / 'trace_on-service.log', trace=True, ranges=ranges)
        self.service_health(self.container)

    def trace_replay_baseline(self, pairs, profile):
        candidate = {'id': 'original_fresh', 'command': list(self.original['Config']['Cmd']), 'changes': []}
        self.selected_config = candidate
        write_json(self.directory / 'selected-config.json',
                   {'id': candidate['id'], 'command': candidate['command'], 'changes': [],
                    'runtime': 'pod', 'trace_source_sha256': sha256(self.directory / 'trace-source.json'),
                    'original_command': candidate['command'], 'validation_mode': self.validation_mode,
                    'diagnostic_requests': 0, 'baseline_requests': 48})
        path = self.directory / 'trace-off-service'
        path.mkdir()
        try:
            self.start_candidate(candidate, path)
            return self.requests('trace_off', pairs, profile, soft_output_differences=True)
        finally:
            self.cleanup()
            self.wait_release()

    def contract(self, *, initialize_new_blocks=False):
        require(not initialize_new_blocks, 'Repair contract unsupported in Pod trace-replay')
        self.update(stage='npu_compressor_contract')
        output = self.directory / 'compressor-contract.json'
        command = [sys.executable, str(self.code_root / 'm0a/deepseek_compressor_contract.py'),
                   '--output', str(output), '--artifact-directory', str(self.directory),
                   '--model-config', str(self.model_dir / 'config.json'),
                   '--source-dir', str(self.directory / 'sources'), '--block-size', '128']
        env = dict(os.environ, MEMECHO_VALIDATION_RUN=self.run_id)
        self.command(command, timeout=900, output=self.directory / 'compressor-contract.log', env=env)
        result = json.loads(output.read_text())
        require(result.get('passed') is True and result.get('support_contract') == 'c4_overlap_v1',
                'NPU compressor contract failed')
        self.wait_release()
        return result

    def report(self, status, error=None):
        super().report(status, error)
        path = self.directory / 'report.json'
        report = json.loads(path.read_text())
        report['runtime'] = 'pod'
        report['model_dir'] = str(self.model_dir)
        report['model_revision_verification'] = (
            'Preserved metadata hashes and indexed shard presence/size matched; no readable commit marker '
            'or original per-shard hashes were available.')
        report['limitations'].append('The model commit and every weight shard byte were not independently proven.')
        write_json(path, report)
        with (self.directory / 'report.md').open('a') as stream:
            stream.write('\n模型核验：保存的元数据哈希与索引分片数量/大小一致；无可读 commit 标记或原始逐分片哈希。\n')

    def run(self):
        with ExclusiveLock(self.root / 'runtime/deepseek-pod/validation.lock'):
            return super().run()


def watchdog(root: Path, run_id: str) -> int:
    worker = PodWorker(root, run_id)
    saved = json.loads((worker.directory / 'worker-identity.json').read_text())
    deadline = time.monotonic() + TOTAL_SECONDS + 1800
    while time.monotonic() < deadline:
        state = json.loads((worker.directory / 'status.json').read_text())
        if state['stage'] == 'finished' and state.get('original_service_restored'):
            return 0
        if not same_process(saved):
            break
        time.sleep(15)
    if same_process(saved):
        os.kill(saved['pid'], signal.SIGTERM)
        time.sleep(15)
        if same_process(saved):
            os.kill(saved['pid'], signal.SIGKILL)
    state = json.loads((worker.directory / 'status.json').read_text())
    worker.original = json.loads((worker.directory / 'original-service.json').read_text())
    worker.restore_required = (worker.directory / 'restore-required.json').exists()
    worker.experiment = json.loads((worker.directory / 'experiment-process.json').read_text()) if (
        worker.directory / 'experiment-process.json').exists() else None
    error = 'Worker exited unexpectedly; independent Pod recovery'
    try:
        worker.restore()
    except BaseException as failure:
        error += '; restoration failed: ' + str(failure)
        write_json(worker.directory / 'restoration.json', {'restored': False, 'error': str(failure), 'at': utc()})
    worker.completed = state.get('completed_stages', [])
    worker.report('failed', error)
    restored = json.loads((worker.directory / 'restoration.json').read_text()).get('restored', False)
    worker.update(**dict(state, status='failed', stage='finished', error=error,
                         finished_at=utc(), original_service_restored=restored))
    write_json(worker.directory / 'watchdog-recovery.json', {'at': utc(), 'error': error})
    return 1


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--watchdog', action='store_true')
    args = parser.parse_args()
    return watchdog(args.root, args.run_id) if args.watchdog else PodWorker(args.root, args.run_id).run()


if __name__ == '__main__':
    raise SystemExit(main())
