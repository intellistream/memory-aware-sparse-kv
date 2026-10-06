#!/usr/bin/env python3
"""Publish a small terminal report and index after the detached sync guardian."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from m0a.contracts import sha256_file
from m0a.deepseek_validation import write_json
from scripts.launch_single_npu_validation import remote, status_remote


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def service_check(host: str, remote_directory: str, launch: dict) -> dict:
    code = '''import contextlib,io,json,pathlib,sys
p=pathlib.Path(DIRECTORY)
sys.path.insert(0,str(p/'implementation'))
from m0a.pod_runtime import SupervisorService,health,owned_pids,service_command,sha256
from m0a.preflight import check
root=pathlib.Path(ROOT); model=pathlib.Path(MODEL)
service=SupervisorService(root,model)
identity=service.identity()
models=health(model,timeout=60)
saved=json.loads((p/'original-service.json').read_text()) if (p/'original-service.json').exists() else None
preflight=json.loads((p/'preflight.json').read_text()) if (p/'preflight.json').exists() else None
bootstrap=json.loads((p/'service-bootstrap.json').read_text())
assert saved is None or service_command(model)==saved['Config']['Cmd']
assert sha256(service.conf)==(preflight or bootstrap)['supervisor_config_sha256']
with contextlib.redirect_stdout(io.StringIO()):
 hbm=check(tuple(range(8)),65536)
print(json.dumps({'healthy':True,'process_identity':identity,'models':models,
                  'config_unchanged':True,'eight_npu_healthy':True,
                  'hbm_used_mb':hbm,
                  'owned_processes_remaining':owned_pids(p.name)}))
'''.replace('DIRECTORY',repr(remote_directory)).replace('ROOT',repr(launch['remote_directory'].split('/m0a/runs/')[0])).replace('MODEL',repr(launch['model_dir']))
    return json.loads(remote(host, code, timeout=90))


def acceptance_gates(guardian_exit_code: int, state: dict, report: dict) -> tuple[bool, bool]:
    engineering = bool(
        guardian_exit_code == 0 and state['hash_sync'] == 'verified'
        and state['service_health'] == 'verified'
        and report.get('status') == 'engineering_validated'
        and report.get('engineering_evidence_validated') is True
        and not report.get('missing_artifacts')
        and report.get('restoration', {}).get('restored') is True)
    complete = bool(engineering and state['output_acceptance'] == 'passed'
        and report.get('event_locality_conclusion') not in (None, 'not_qualified', 'exploratory')
        and report.get('paired_effect_interpretation') == 'qualified_for_offline_comparison')
    return engineering, complete


def finalize_run(host: str, remote_directory: str, directory: Path,
                 *, guardian_exit_code: int) -> int:
    directory = Path(directory)
    launch = json.loads((directory / 'launch.json').read_text())
    run_id = launch['run_id']
    final = directory / 'final-status.json'
    state = {'run_id': run_id, 'finished_at': utc(),
             'experiment_conclusion': 'not_qualified', 'output_acceptance': 'not_qualified',
             'hash_sync': 'failed', 'service_health': 'unverified', 'push': 'failed',
             'engineering_acceptance': False, 'complete_acceptance': False,
             'guardian_exit_code': guardian_exit_code,
             'errors': []}
    report = {}
    try:
        # The guardian can fail while the independent watchdog is still
        # restoring the original service. Do not publish a terminal service
        # verdict until that recovery has had its own budget.
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            try:
                remote_status = status_remote(host, remote_directory)
                if (remote_status.get('stage') == 'finished' and
                        not remote_status.get('worker_alive') and
                        not remote_status.get('watchdog_alive')):
                    break
            except Exception as error:
                state['last_status_error'] = str(error)
            time.sleep(15)
        else:
            state['errors'].append('Remote worker/watchdog recovery exceeded 1800 seconds')
        report_path = directory / 'report.json'
        if report_path.exists():
            report = json.loads(report_path.read_text())
            state['experiment_conclusion'] = report.get('event_locality_conclusion', report.get('status', 'not_qualified'))
            state['output_acceptance'] = report.get('strict_output_acceptance', 'not_qualified')
        verification = directory / 'final-local-verification.json'
        manifest_path = directory / 'final-checksums.json'
        if verification.exists() and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            mismatches = [row['path'] for row in manifest['files']
                          if not (directory / row['path']).is_file()
                          or sha256_file(directory / row['path']) != row['sha256']]
            if not mismatches:
                state['hash_sync'] = 'verified'
                state['manifest_sha256'] = sha256_file(manifest_path)
                state['verified_files'] = len(manifest['files'])
            else:
                state['errors'].append('Hash mismatches: ' + ', '.join(mismatches[:10]))
        service = service_check(host, remote_directory, launch)
        state['service_health'] = 'verified' if service['healthy'] and not service['owned_processes_remaining'] else 'failed'
        state['service'] = service
    except Exception as error:
        state['errors'].append('Final verification: ' + str(error))
    engineering_preconditions, strict_preconditions = acceptance_gates(
        guardian_exit_code, state, report)
    state['push'] = 'pending'
    index_dir = ROOT / 'results/tau3'
    index_dir.mkdir(parents=True, exist_ok=True)
    index = index_dir / (run_id + '.json')
    summary = ROOT / 'docs/tau3-latest-result.md'
    def publish_files():
        write_json(final, state)
        write_json(index, {**state, 'local_run_directory': str(directory),
                           'remote_run_directory': remote_directory,
                           'report_sha256': sha256_file(directory / 'report.json')
                               if (directory / 'report.json').exists() else None})
        summary.write_text('# τ³ workload validation\n\n'
                           f'Run: `{run_id}`  \n'
                           f'Complete acceptance: `{state["complete_acceptance"]}`  \n'
                           f'Engineering acceptance: `{state["engineering_acceptance"]}`  \n'
                           f'Experiment: `{state["experiment_conclusion"]}`  \n'
                           f'Output: `{state["output_acceptance"]}`  \n'
                           f'Hash synchronization: `{state["hash_sync"]}`  \n'
                           f'Service health: `{state["service_health"]}`  \n'
                           f'Push: `{state["push"]}`\n\n'
                           'The local and Pod raw artifacts are linked by the manifest SHA-256 in the index. '
                           'Offline replay is not an online ECHO performance measurement.\n')
    publish_files()
    try:
        branch_info = json.loads((directory / 'branch-provenance.json').read_text())
        branch = branch_info['branch']
        current = subprocess.check_output(['git', '-C', str(ROOT), 'branch', '--show-current'], text=True).strip()
        if current != branch:
            raise RuntimeError('Active branch changed during unattended run')
        rel_index, rel_summary = str(index.relative_to(ROOT)), str(summary.relative_to(ROOT))
        subprocess.run(['git', '-C', str(ROOT), 'add', '--', rel_index, rel_summary], check=True)
        subprocess.run(['git', '-C', str(ROOT), 'commit', '--only', '-m',
                        f'Archive τ³ workload result {run_id}', '--', rel_index, rel_summary], check=True)
        subprocess.run(['git', '-C', str(ROOT), 'push', 'origin', f'HEAD:refs/heads/{branch}'],
                       check=True, timeout=180)
        state['push'] = 'verified'
        state['initial_push_commit'] = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                                                                text=True).strip()
        state['engineering_acceptance'] = engineering_preconditions
        state['complete_acceptance'] = strict_preconditions
        publish_files()
        subprocess.run(['git', '-C', str(ROOT), 'add', '--', rel_index, rel_summary], check=True)
        subprocess.run(['git', '-C', str(ROOT), 'commit', '--only', '-m',
                        f'Verify τ³ workload result publication {run_id}', '--', rel_index, rel_summary], check=True)
        subprocess.run(['git', '-C', str(ROOT), 'push', 'origin', f'HEAD:refs/heads/{branch}'],
                       check=True, timeout=180)
    except Exception as error:
        state['push'] = 'failed'
        state['engineering_acceptance'] = False
        state['complete_acceptance'] = False
        state['errors'].append('Commit/push: ' + str(error))
    publish_files()
    return 0 if state['engineering_acceptance'] else 1


if __name__ == '__main__':
    raise SystemExit('Use through launch_deepseek_validation.py --guardian')
