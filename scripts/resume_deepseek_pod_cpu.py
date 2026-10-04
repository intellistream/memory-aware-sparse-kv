#!/usr/bin/env python3
"""Finish CPU validation of an archived Pod capture after a validator correction.

The captured run stays immutable and retains its original failed status. This
creates a separately hashed postprocess result, with both code revisions named.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from m0a.contracts import sha256_file
from m0a.deepseek_pod_validation import PodWorker
from m0a.deepseek_validation import WINDOW, export_sidecar, validate_native_trace, write_json
from m0a.model_profiles import load_profile
from m0a.pod_runtime import ExclusiveLock
from m0a.transition_replay import read_trace, validate_sidecar
from m0a.working_set import require


INPUTS = (
    'launch.json', 'pairs.json', 'layout.json', 'selected-config.json',
    'trace_off-api.jsonl', 'trace_on-api.jsonl',
    'trace_off-responses.jsonl', 'trace_on-responses.jsonl',
    'trace_off-failures.json', 'trace_on-failures.json',
    'trace_off-output-differences.json', 'trace_on-output-differences.json',
    'compressor-contract.json', 'restoration.json',
)


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    worker = PodWorker(args.root, args.run_id)
    capture = worker.directory
    result = capture / 'postprocess-cp-padding-v1'
    with ExclusiveLock(args.root / 'runtime/deepseek-pod/validation.lock'):
        require(not result.exists(), 'Postprocess directory already exists')
        status = json.loads((capture / 'status.json').read_text())
        require(status['status'] == 'failed' and status.get('error') == 'Invalid CP chunk range',
                'Only the archived CP padding validator failure may be resumed')
        restoration = json.loads((capture / 'restoration.json').read_text())
        require(restoration.get('restored') and restoration.get('config_unchanged'),
                'Original service was not verified as restored')
        worker.service_health('pod-service')
        manifest_path = capture / 'local-archive/capture-checksums.json'
        manifest = {item['path']: item for item in json.loads(manifest_path.read_text())['files']}
        source_names = [str(p.relative_to(capture)) for p in (capture / 'sources').glob('*') if p.is_file()]
        raw_names = [str(p.relative_to(capture)) for p in (capture / 'raw-traces').glob('rank*.jsonl')]
        require(raw_names and source_names, 'Archived native trace or source evidence missing')
        names = [*INPUTS, *source_names, *raw_names]
        for name in names:
            path = capture / name
            require(name in manifest and path.is_file() and path.stat().st_size == manifest[name]['size'] and
                    sha256_file(path) == manifest[name]['sha256'], 'Capture checksum mismatch: ' + name)
        result.mkdir()
        try:
            for name in INPUTS:
                shutil.copy2(capture / name, result / name)
            shutil.copytree(capture / 'sources', result / 'sources')
            revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
            write_json(result / 'postprocess-provenance.json', {
                'capture_run_id': args.run_id,
                'capture_status': status,
                'capture_manifest_sha256': sha256_file(manifest_path),
                'capture_git_commit': json.loads((capture / 'branch-provenance.json').read_text())['head'],
                'validator_git_commit': revision,
                'verified_capture_inputs': {name: manifest[name]['sha256'] for name in names},
                'reason': 'DSA CP pads each chunk to eight ranks; the original validator compared padded length to actual tokens.',
            })
            pairs = json.loads((result / 'pairs.json').read_text())
            responses = records(result / 'trace_on-responses.jsonl')
            profile = load_profile('deepseek_v4')
            ranges = sorted(set((pair[variant]['boundary_position'] - 1,
                                 pair[variant]['boundary_position'] + WINDOW - 1)
                                for pair in pairs['pairs'] for variant in ('event', 'control')))
            evidence = validate_native_trace(capture / 'raw-traces', result / 'traces', responses, ranges,
                                             profile, allow_selected_drift=True)
            write_json(result / 'trace-validation.json', evidence)
            layout = json.loads((result / 'layout.json').read_text())
            support = next(item for item in layout['source_evidence']
                           if item['path'].endswith('compressor_kernel.h'))
            sidecar = export_sidecar(responses, pairs, profile, evidence, support)
            snapshots, _ = validate_sidecar(sidecar, result)
            read_trace(result / 'traces', snapshots, sidecar['events'], synthetic=False)
            write_json(result / 'sidecar.json', sidecar)
            del snapshots, sidecar
            completed = ['preflight', 'inputs', 'npu_compressor_contract', 'trace_off',
                         'trace_on', 'restore_original', 'native_trace_and_sidecar']
            def replay_one(item):
                    scope, capacity = item
                    label = f'{scope}-{capacity}mib'
                    config = {'schema_version': 1, 'capacity_bytes': capacity * 1024**2,
                              'prefetch_budget_bytes': 8 * 1024**2, 'budget_scope': scope,
                              'kv_unit_bytes': {'compressed_kv_token_position': 1024},
                              'max_tables': 256, 'bootstrap_draws': 1000, 'seed': 0}
                    path = result / f'cache-{label}.json'
                    write_json(path, config)
                    command = [sys.executable, str(ROOT / 'm0a/transition_replay.py'),
                               '--trace-dir', str(result / 'traces'), '--sidecar', str(result / 'sidecar.json'),
                               '--config', str(path), '--source-root', str(result),
                               '--output-dir', str(result / f'replay-{label}')]
                    with (result / f'replay-{label}.log').open('w') as log:
                        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                    report_path = result / f'replay-{label}/report.json'
                    report = json.loads(report_path.read_text())
                    report['paired_effect_interpretation'] = evidence['paired_effect_interpretation']
                    if evidence['paired_effect_interpretation'] == 'exploratory':
                        report['limitations'].append('Selected sets drift across repeats or paired prefixes; paired effects are exploratory.')
                    write_json(report_path, report)
                    print(label, report['training_events'], report['evaluation_events'], flush=True)
                    return label

            budgets = [(scope, capacity) for scope in ('per_rank', 'aggregate')
                       for capacity in (64, 128)]
            with ThreadPoolExecutor(max_workers=4) as pool:
                completed.extend('replay_' + label for label in pool.map(replay_one, budgets))
            worker.directory = result
            worker.selected_config = json.loads((result / 'selected-config.json').read_text())
            worker.completed = completed
            worker.validate_engineering_evidence()
            worker.service_health('pod-service')
            worker.report('engineering_validated')
            report_path = result / 'report.json'
            report = json.loads(report_path.read_text())
            report['capture_run_status'] = 'failed'
            report['capture_error'] = status['error']
            report['postprocess_provenance'] = 'postprocess-provenance.json'
            write_json(report_path, report)
            write_json(result / 'status.json', {'run_id': args.run_id, 'status': 'engineering_validated',
                                                'original_service_restored': True,
                                                'capture_run_status': 'failed',
                                                'validator_git_commit': revision})
            files = sorted(path for path in result.rglob('*') if path.is_file())
            write_json(result / 'sha256-manifest.json', {'files': [
                {'path': str(path.relative_to(result)), 'size': path.stat().st_size,
                 'sha256': sha256_file(path)} for path in files]})
            print(json.dumps({'status': 'engineering_validated', 'result': str(result),
                              'files': len(files), 'validator_git_commit': revision}), flush=True)
            return 0
        except BaseException:
            (result / 'failure.log').write_text(traceback.format_exc())
            raise


if __name__ == '__main__':
    raise SystemExit(main())
