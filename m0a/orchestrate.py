"""Fail-closed, model-aware M0-A orchestration with a dry-run default."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

try:
    from .contracts import (SCHEMA_VERSION, sha256_file, validate_pair_set,
                            validate_run_manifest)
    from .model_profiles import (DEFAULT_PROFILE, load_profile,
                                 resolve_profile_path, validate_pair_policy)
except ImportError:
    from contracts import (SCHEMA_VERSION, sha256_file, validate_pair_set,
                           validate_run_manifest)
    from model_profiles import (DEFAULT_PROFILE, load_profile,
                                resolve_profile_path, validate_pair_policy)


DEFAULT_ROOT = Path('/workspace/memecho')
BASELINE_CONTAINER = 'memecho-vllm-ascend'
KNOWN_TRACE_CONTAINERS = ('memecho-m0a-trace', 'memecho-m0a-glm53-trace')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def merge_ranges(ranges):
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def pair_positions(pair_set, *, first_pair_only=False, include_short=False):
    ranges = [(0, 12)] if include_short else []
    pairs = pair_set['pairs'][:1] if first_pair_only else pair_set['pairs']
    for pair in pairs:
        for variant in ('event', 'control'):
            boundary = pair[variant]['boundary_position']
            ranges.append((max(0, boundary - 4), boundary + 8))
    return ','.join(f'{start}:{end}' for start, end in merge_ranges(ranges))


def _load_pair_set(pairs_path: Path, profile_name: str):
    pair_set = json.loads(pairs_path.read_text())
    validate_pair_set(pair_set)
    pinned_profile = pair_set.get('model_profile')
    if pinned_profile is None and profile_name != 'deepseek_v4':
        raise ValueError('Non-DeepSeek pair sets must pin model_profile')
    if pinned_profile is not None and pinned_profile != profile_name:
        raise ValueError(
            f'Pair set is pinned to {pinned_profile}, requested {profile_name}'
        )
    return pair_set


def _profile_contract(profile):
    return {
        key: profile[key] for key in (
            'name', 'model_id', 'model_revision', 'attention_backend',
            'native_operators', 'raw_index_unit', 'compression_ratio',
            'logical_block_size', 'selected_width', 'expected_layers',
            'computed_layers', 'reused_layers', 'trace_schema_version',
            'requires_equal_pair_tokens',
        )
    }


def build_plan(root: Path, *, mode: str, run_id: str, pairs_path: Path | None,
               repetitions: int, profile_name: str = DEFAULT_PROFILE,
               baseline_path: Path | None = None, device: int | None = None):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', run_id):
        raise ValueError('run_id may contain only letters, digits, dot, underscore, dash')
    if mode not in {'baseline', 'pilot', 'pairs'}:
        raise ValueError(f'Unknown mode: {mode}')
    if repetitions < 1:
        raise ValueError('repetitions must be positive')
    profile = load_profile(profile_name, root / 'm0a/model_profiles.json')
    if device is not None:
        if profile['tensor_parallel_size'] != 1:
            raise ValueError('--device override is valid only for TP=1 profiles')
        if device < 0 or device > 7:
            raise ValueError('--device must be between 0 and 7')
        profile['devices'] = [device]

    pair_set = None
    if mode in {'pilot', 'pairs'}:
        if pairs_path is None:
            pairs_path = resolve_profile_path(root, profile, 'default_pairs')
        pair_set = _load_pair_set(pairs_path, profile_name)
        validate_pair_policy(pair_set, profile)
    if mode == 'pairs':
        trace_positions = pair_positions(pair_set)
    elif mode == 'pilot' and profile['pilot_kind'] == 'model_baseline':
        trace_positions = pair_positions(
            pair_set, first_pair_only=True, include_short=True
        )
    elif mode == 'pilot':
        trace_positions = '0:12,1018:1032'
    else:
        trace_positions = ''

    if baseline_path is None:
        baseline_path = resolve_profile_path(root, profile, 'baseline_file')
    run_dir = root / 'm0a/runs' / run_id
    trace_dir = run_dir / 'traces'
    response_names = {
        'baseline': 'baseline_response.json',
        'pilot': 'pilot_response.json',
        'pairs': 'pair_responses.jsonl',
    }
    responses = run_dir / response_names[mode]
    request_command = [
        'python3', str(root / 'm0a/run_requests.py'), mode,
        '--root', str(root), '--run-id', run_id, '--output', str(responses),
        '--model-profile', profile_name,
    ]
    if mode == 'baseline':
        request_command += ['--repetitions', '3']
        final_command = [
            'python3', str(root / 'm0a/validate_model_baseline.py'),
            '--response', str(responses), '--model-profile', profile_name,
        ]
    elif mode == 'pairs':
        request_command += [
            '--pairs', str(pairs_path), '--repetitions', str(repetitions)
        ]
        final_command = [
            'python3', str(root / 'm0a/analyze.py'),
            '--trace-dir', str(trace_dir), '--responses', str(responses),
            '--output', str(run_dir / 'analysis.json'),
            '--model-profile', profile_name,
        ]
    else:
        request_command += ['--pairs', str(pairs_path)]
        if profile['pilot_kind'] == 'model_baseline':
            request_command += ['--baseline', str(baseline_path)]
        final_command = [
            'python3', str(root / 'm0a/validate_pilot.py'),
            '--root', str(root), '--response', str(responses),
            '--trace-dir', str(trace_dir), '--run-id', run_id,
            '--output', str(run_dir / 'pilot_validation.json'),
            '--model-profile', profile_name,
        ]

    hashes = {}
    candidates = {
        'm0a_manifest': root / 'm0a/manifest.yaml',
        'model_profiles': root / 'm0a/model_profiles.json',
        'pairs': pairs_path,
    }
    if profile_name == 'deepseek_v4':
        candidates['trace_patch'] = root / 'm0a/vllm-ascend-trace.patch'
    else:
        candidates['trace_source'] = root / 'm0a/glm_image/m0a_sfa_selected_trace.py'
        candidates['sfa_source'] = root / 'm0a/glm_image/sfa_v1.py'
    for name, path in candidates.items():
        if path is not None and path.is_file():
            hashes[name] = sha256_file(path)

    devices = ','.join(map(str, profile['devices']))
    launcher = resolve_profile_path(root, profile, 'launcher')
    commands = [
        ['python3', str(root / 'm0a/preflight.py'), '--devices', devices],
        ['bash', str(launcher)],
        request_command,
        final_command,
        ['docker', 'stop', profile['container']],
    ]
    manifest = {
        'schema_version': SCHEMA_VERSION,
        'run_id': run_id,
        'mode': mode,
        'status': 'planned',
        'created_at': utc_now(),
        'root': str(root),
        'run_dir': str(run_dir),
        'trace_dir': str(trace_dir) if mode != 'baseline' else None,
        'trace_positions': trace_positions,
        'responses': str(responses),
        'repetitions': repetitions if mode == 'pairs' else (3 if mode == 'baseline' else None),
        'model_profile': profile_name,
        'model_contract': _profile_contract(profile),
        'image': profile['image'],
        'container': profile['container'],
        'devices': profile['devices'],
        'baseline': str(baseline_path) if mode == 'pilot' else None,
        'baseline_target': str(baseline_path) if mode == 'baseline' else None,
        'commands': commands,
        'artifact_hashes': hashes,
    }
    validate_run_manifest(manifest)
    return manifest


def write_manifest(path: Path, manifest):
    validate_run_manifest(manifest)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    temporary.replace(path)


def docker_command(root, *args, capture_output=False):
    env = os.environ.copy()
    env['DOCKER_HOST'] = f'unix://{root}/runtime/docker.sock'
    return subprocess.run(
        ['docker', *args], env=env, check=False, text=True,
        capture_output=capture_output,
    )


def ensure_not_running(root, name):
    result = docker_command(root, 'inspect', '-f', '{{.State.Running}}', name,
                            capture_output=True)
    if result.returncode == 0 and result.stdout.strip() == 'true':
        raise RuntimeError(f'Container is already running: {name}')


def snapshot(command, output):
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    output.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f'Snapshot command failed: {command}')


def wait_healthy(root, container, url, timeout_seconds):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout_seconds
    last_error = None
    while time.monotonic() < deadline:
        try:
            with opener.open(url, timeout=5) as response:
                if response.status == 200:
                    return
        except Exception as error:
            last_error = error
        state = docker_command(root, 'inspect', '-f', '{{.State.Running}}',
                               container, capture_output=True)
        if state.returncode == 0 and state.stdout.strip() != 'true':
            raise RuntimeError(f'{container} exited before becoming healthy')
        time.sleep(5)
    raise TimeoutError(
        f'Service did not become healthy in {timeout_seconds}s: {last_error}'
    )


def execute(plan, *, health_timeout):
    root = Path(plan['root'])
    run_dir = Path(plan['run_dir'])
    if run_dir.exists():
        raise FileExistsError(f'Refusing to reuse run directory {run_dir}')
    for name in (BASELINE_CONTAINER, *KNOWN_TRACE_CONTAINERS):
        ensure_not_running(root, name)
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / 'run_manifest.json'
    write_manifest(manifest_path, plan)
    manifest = dict(plan)
    manifest.update(status='running', started_at=utc_now())
    write_manifest(manifest_path, manifest)
    server_started = False
    container = plan['container']
    try:
        snapshot(['npu-smi', 'info'], run_dir / 'npu_before.txt')
        subprocess.run(plan['commands'][0], check=True, cwd=root)
        env = os.environ.copy()
        trace_enabled = plan['mode'] != 'baseline'
        env.update({
            'RECREATE': '1',
            'MEMECHO_ROOT': str(root),
            'M0A_NPU_ID': str(plan['devices'][0]),
            'VLLM_ASCEND_M0A_TRACE_DIR': plan['trace_dir'] if trace_enabled else '',
            'VLLM_ASCEND_M0A_TRACE_POSITIONS': plan['trace_positions'] if trace_enabled else '',
            'VLLM_ASCEND_M0A_RUN_ID': plan['run_id'] if trace_enabled else '',
        })
        started = subprocess.run(plan['commands'][1], cwd=root, env=env,
                                 text=True, capture_output=True, check=True)
        (run_dir / 'start.log').write_text(started.stdout + started.stderr)
        server_started = True
        wait_healthy(root, container, 'http://127.0.0.1:8900/health',
                     health_timeout)
        subprocess.run(plan['commands'][2], cwd=root, check=True)
        subprocess.run(plan['commands'][3], cwd=root, check=True)
        if plan['mode'] == 'baseline':
            baseline_target = Path(plan['baseline_target'])
            if baseline_target.exists():
                raise FileExistsError(
                    f'Refusing to overwrite model baseline {baseline_target}'
                )
            baseline_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(plan['responses'], baseline_target)
        manifest.update(status='passed')
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if server_started:
            logs = docker_command(root, 'logs', container, capture_output=True)
            (run_dir / 'server.log').write_text(logs.stdout + logs.stderr)
            docker_command(root, 'stop', container, capture_output=True)
        try:
            snapshot(['npu-smi', 'info'], run_dir / 'npu_after.txt')
        except Exception as error:
            manifest['npu_after_error'] = str(error)
        manifest['finished_at'] = utc_now()
        write_manifest(manifest_path, manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('baseline', 'pilot', 'pairs'))
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--model-profile', default=DEFAULT_PROFILE)
    parser.add_argument('--pairs', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--device', type=int)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--health-timeout', type=int, default=900)
    parser.add_argument('--execute', action='store_true',
                        help='Actually start the container; omission is a dry run')
    args = parser.parse_args()
    plan = build_plan(
        args.root, mode=args.mode, run_id=args.run_id,
        pairs_path=args.pairs, repetitions=args.repetitions,
        profile_name=args.model_profile, baseline_path=args.baseline,
        device=args.device,
    )
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    execute(plan, health_timeout=args.health_timeout)


if __name__ == '__main__':
    main()
