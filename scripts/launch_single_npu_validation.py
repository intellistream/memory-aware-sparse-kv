#!/usr/bin/env python3
"""Dry-run by default; deploy a detached worker and checksum synchronization guardian."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shlex
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from m0a.single_npu_validation import IMAGE, IMAGE_ID, TOTAL_SECONDS, utc, write_json
from m0a.contracts import sha256_file
from m0a.working_set import require

SSH_OPTIONS = ['-o', 'ClearAllForwardings=yes', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
               '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3']


def ssh_command(host, command):
    return ['ssh', *SSH_OPTIONS, host, command]


def remote(host, code, *, timeout=60):
    return subprocess.check_output(ssh_command(host, 'python3 -'), input=code.encode(), timeout=timeout)


def safe_relative(path):
    value = PurePosixPath(path)
    require(isinstance(path, str) and path and not value.is_absolute() and '..' not in value.parts,
            'Unsafe manifest path')
    return path


def detached_process(args, log):
    with Path(log).open('ab') as output:
        return subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                start_new_session=True, close_fds=True)


def alive(pid):
    try:
        os.kill(pid, 0)
        stat = Path(f'/proc/{pid}/stat')
        return not stat.exists() or stat.read_text().split()[2] != 'Z'
    except (OSError, ValueError):
        return False


def deployment_files():
    paths = [p for p in (ROOT / 'm0a').glob('*.py')]
    paths += [ROOT / 'm0a/model_profiles.json', ROOT / 'm0a/serve_glm53_tiny_trace.sh',
              ROOT / 'm0a/artifacts/glm53_tiny_model_manifest.json',
              ROOT / 'm0a/workloads/glm53_tiny.synthetic.equalized.pairs.json',
              ROOT / 'docs/single-npu-validation.md',
              ROOT / 'scripts/launch_single_npu_validation.py']
    return sorted(paths)


def archive_code(directory, paths, *, bundle_name='implementation.bundle', provenance_name='code-provenance.json',
                 message='Implement single NPU GLM validation and synchronization'):
    """Commit an isolated exact code snapshot when the workspace Git is unavailable."""
    with tempfile.TemporaryDirectory(prefix='memecho-validation-git-') as name:
        archive = Path(name)
        for path in paths:
            target = archive / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
        def git(*args):
            return subprocess.check_output(['git', '-C', str(archive), *args], stderr=subprocess.STDOUT, text=True).strip()
        git('init', '-b', 'single-npu-validation')
        git('add', '.')
        git('-c', 'user.name=Memecho Validation', '-c', 'user.email=validation@localhost',
            'commit', '-m', message)
        commit = git('rev-parse', 'HEAD')
        git('bundle', 'create', str(directory / bundle_name), '--all')
        git('bundle', 'verify', str(directory / bundle_name))
    write_json(directory / provenance_name, {'git_commit': commit, 'scope': 'isolated local exact artifact snapshot',
        'workspace_git_available': (ROOT / '.git/HEAD').exists(), 'github_url': None,
        'bundle_sha256': sha256_file(directory / bundle_name)})
    return commit


def transfer_files(host, remote_directory, files, *, destination_prefix=''):
    """Upload an explicit code list. No repository deletion or broad synchronization."""
    with tempfile.TemporaryFile() as stream:
        with tarfile.open(fileobj=stream, mode='w') as archive:
            for source, relative in files:
                archive.add(source, arcname=destination_prefix + safe_relative(relative), recursive=False)
        stream.seek(0)
        command = 'tar -xf - -C ' + shlex.quote(remote_directory)
        subprocess.run(ssh_command(host, command), stdin=stream, timeout=180, check=True)


def extract_checked(stream, records, target):
    expected = {safe_relative(r['path']): r for r in records}
    require(len(expected) == len(records), 'Duplicate manifest paths')
    received = set()
    with tarfile.open(fileobj=stream, mode='r|') as archive:
        for member in archive:
            require(member.isfile() and member.name in expected and member.name not in received, 'Unexpected sync archive member')
            record = expected[member.name]
            require(member.size == record['size'], 'Sync file size mismatch')
            destination = Path(target) / member.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + '.sync-tmp')
            digest = hashlib.sha256()
            with archive.extractfile(member) as source, temporary.open('wb') as output:
                while block := source.read(1024 * 1024):
                    output.write(block)
                    digest.update(block)
            require(digest.hexdigest() == record['sha256'], 'Sync SHA-256 mismatch: ' + member.name)
            temporary.replace(destination)
            received.add(member.name)
    require(received == set(expected), 'Missing sync archive files')


def download_records(host, server_directory, records, local_directory):
    if not records:
        return
    paths = [safe_relative(r['path']) for r in records]
    command = 'tar -cf - -C ' + shlex.quote(server_directory) + ' -- ' + ' '.join(map(shlex.quote, paths))
    with tempfile.TemporaryFile() as stream:
        subprocess.run(ssh_command(host, command), stdout=stream, timeout=900, check=True)
        stream.seek(0)
        extract_checked(stream, records, local_directory)


def synchronize_checkpoint(host, server_directory, local_directory, name, digest):
    safe_relative(name)
    require('/' not in name, 'Invalid checkpoint name')
    path = str(PurePosixPath(server_directory) / 'checkpoints' / (name + '.json'))
    data = remote(host, 'from pathlib import Path\nimport sys\nsys.stdout.buffer.write(Path(' + repr(path) + ').read_bytes())\n')
    require(hashlib.sha256(data).hexdigest() == digest, 'Manifest SHA-256 mismatch')
    manifest = json.loads(data)
    require(manifest['run_id'] == Path(local_directory).name, 'Wrong synchronization run')
    records = manifest['files']
    download_records(host, server_directory, records, local_directory)
    manifest_path = Path(local_directory) / 'checkpoints' / (name + '.json')
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(data)
    ack = {'run_id': manifest['run_id'], 'manifest_sha256': digest, 'files': records, 'verified_at': utc()}
    ack_path = str(PurePosixPath(server_directory) / 'acks' / (name + '.json'))
    remote(host, 'import pathlib,json\np=pathlib.Path(' + repr(ack_path) + ')\np.parent.mkdir(parents=True,exist_ok=True)\n'
                't=p.with_name(p.name+".tmp")\nt.write_text(' + repr(json.dumps(ack)) + ')\nt.replace(p)\n')
    write_json(Path(local_directory) / 'acks' / (name + '.json'), ack)
    with (Path(local_directory) / 'sync-ledger.jsonl').open('a') as out:
        out.write(json.dumps({'checkpoint': name, **ack}) + '\n')
    return ack


def retry_sync(action, *, attempts=3, delay=5):
    last = None
    for attempt in range(attempts):
        try:
            return action()
        except Exception as error:
            last = error
            print(f'Sync attempt {attempt + 1}/{attempts}: {error}', flush=True)
            if attempt + 1 < attempts:
                time.sleep(delay)
    raise RuntimeError('Synchronization failed after three attempts') from last


def status_remote(host, directory):
    code = ('import pathlib,json,os\np=pathlib.Path(' + repr(directory) + ')\n'
            's=json.loads((p/"status.json").read_text())\n'
            'try:\n os.kill(s["worker_pid"],0)\n s["worker_alive"]=pathlib.Path("/proc/"+str(s["worker_pid"])+"/stat").read_text().split()[2]!="Z"\n'
            'except (ProcessLookupError,FileNotFoundError):\n s["worker_alive"]=False\n'
            's["watchdog_alive"]=False\n'
            'if s.get("watchdog_pid"):\n'
            ' try: s["watchdog_alive"]=pathlib.Path("/proc/"+str(s["watchdog_pid"])+"/stat").read_text().split()[2]!="Z"\n'
            ' except FileNotFoundError: pass\n'
            'print(json.dumps(s))\n')
    return json.loads(remote(host, code))


def final_inventory(host, directory):
    """Freeze all final artifacts after worker exit, including logs and ACK records."""
    code = ('import pathlib,json,hashlib\np=pathlib.Path(' + repr(directory) + ')\nout=[]\n'
            'for f in sorted(p.rglob("*")):\n'
            ' if f.is_file() and not f.name.endswith(".tmp") and f.name!="final-checksums.json":\n'
            '  h=hashlib.sha256()\n'
            '  with f.open("rb") as s:\n'
            '   for b in iter(lambda:s.read(1048576),b""): h.update(b)\n'
            '  out.append({"path":str(f.relative_to(p)),"size":f.stat().st_size,"sha256":h.hexdigest()})\n'
            '(p/"final-checksums.json").write_text(json.dumps({"files":out},indent=2)+"\\n")\nprint(json.dumps(out))\n')
    return json.loads(remote(host, code, timeout=180))


def qualified_final_status(directory, current):
    """Accept trace replay only with the complete restored and hashed evidence set."""
    if current['status']=='passed':
        return True
    if current['status']!='engineering_validated':
        return False
    directory=Path(directory)
    report=json.loads((directory/'report.json').read_text())
    restoration=json.loads((directory/'restoration.json').read_text())
    verification=json.loads((directory/'final-local-verification.json').read_text())
    provenance=json.loads((directory/'local-archive/final-code-provenance.json').read_text())
    records=json.loads((directory/'final-checksums.json').read_text())['files']
    paths={record['path'] for record in records}
    required={'report.json','restoration.json','trace-validation.json','sidecar.json',
              'trace_off-output-differences.json','trace_on-output-differences.json',
              'implementation.bundle','code-provenance.json',
              'local-archive/final-artifacts.bundle','local-archive/final-code-provenance.json'}
    required.update(f'replay-{scope}-{capacity}mib/report.json'
                    for scope in ('per_rank','aggregate') for capacity in (64,128))
    require(required <= paths, 'Missing final trace-replay evidence')
    require(report['status']=='engineering_validated' and report['validation_mode']=='trace-replay' and
            report['strict_output_acceptance']=='not_qualified' and not report['missing_artifacts'],
            'Engineering report does not qualify')
    require(report['completed_requests_per_phase']=={'trace_off':48,'trace_on':48} and
            report['restoration'].get('restored') is True, 'Incomplete requests or restoration')
    require(restoration.get('restored') is True and restoration.get('healthy') is True and
            restoration.get('config_unchanged') is True and restoration.get('eight_npu_healthy') is True and
            current.get('original_service_restored') is True, 'Original service restoration not verified')
    require(verification['run_id']==directory.name and verification['git_commit']==provenance['git_commit'] and
            sha256_file(directory/'local-archive/final-artifacts.bundle')==provenance['bundle_sha256'],
            'Final archive provenance mismatch')
    return True


def guardian(host, remote_directory, directory):
    directory = Path(directory)
    state = {'pid': os.getpid(), 'run_id': directory.name, 'status': 'running', 'started_at': utc()}
    completed = set()
    lock, stop = threading.Lock(), threading.Event()
    deadline = time.monotonic() + TOTAL_SECONDS + 1800
    def update(**values):
        with lock:
            state.update(values, heartbeat=utc())
            write_json(directory / 'sync-status.json', state)
    def heartbeat():
        while not stop.wait(15):
            update()
    update()
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        while time.monotonic() < deadline:
            current = retry_sync(lambda: status_remote(host, remote_directory))
            write_json(directory / 'status.json', current)
            update(server_stage=current['stage'], server_status=current['status'])
            pending = current.get('pending_checkpoint')
            if pending and pending not in completed and current['stage'] != 'finished':
                ack = retry_sync(lambda: synchronize_checkpoint(host, remote_directory, directory, pending,
                                                                current['pending_manifest_sha256']))
                completed.add(pending)
                update(last_verified_checkpoint=pending, last_manifest_sha256=ack['manifest_sha256'], verified_files=len(ack['files']))
            if not current['worker_alive']:
                if current['stage'] != 'finished':
                    if current.get('watchdog_alive'):
                        # DeepSeek's independent watchdog owns unexpected-exit recovery.
                        time.sleep(15)
                        continue
                    raise RuntimeError('Server worker exited without terminal status')
                if current.get('watchdog_alive'):
                    time.sleep(15)
                    continue
                records = retry_sync(lambda: final_inventory(host, remote_directory))
                # Check already downloaded immutable files; download only missing/changed final artifacts.
                missing = [r for r in records if not (directory / r['path']).is_file() or sha256_file(directory / r['path']) != r['sha256']]
                retry_sync(lambda: download_records(host, remote_directory, missing, directory))
                write_json(directory / 'final-checksums.json', {'files': records})
                for record in records:
                    require(sha256_file(directory / record['path']) == record['sha256'], 'Final artifact hash mismatch')
                archive = directory / 'local-archive'
                archive.mkdir(exist_ok=True)
                (archive / 'capture-checksums.json').write_bytes((directory / 'final-checksums.json').read_bytes())
                paths = [directory / r['path'] for r in records
                         if r['path'].startswith('implementation/') and '__pycache__' not in r['path']]
                paths += [directory / r['path'] for r in records if (r['path'].startswith('replay-') and r['path'].endswith('/report.json')) or r['path'].startswith('diagnostics/') or r['path'] in
                    {'report.json', 'report.md', 'status.json', 'layout.json', 'trace-validation.json',
                     'replay-64mib/report.json', 'replay-128mib/report.json', 'deployment.json', 'resource-release.json',
                     'restoration.json', 'compressor-contract.json', 'identity.json', 'diagnostics.json', 'selected-config.json',
                     'trace_off-failures.json', 'trace_on-failures.json', 'trace_off-output-differences.json',
                     'trace_on-output-differences.json', 'trace-launch.json'}]
                paths += [directory / r['path'] for r in records if r['path'].startswith(('repair-source/', 'operator-source/', 'operator-diagnostics/'))
                          or r['path'] in {'experimental-provenance.json', 'operator-repairs.json', 'trace-deterministic-ranks.json',
                                            'compressor-contract-before.json', 'compressor-contract-repaired.json', 'npu-contract-tests.log'}]
                paths += [archive / 'capture-checksums.json']
                final_commit = archive_code(archive, paths, bundle_name='final-artifacts.bundle', provenance_name='final-code-provenance.json',
                                           message='Archive final single NPU engineering validation evidence')
                retry_sync(lambda: transfer_files(host, remote_directory, [(archive / name, 'local-archive/' + name) for name in
                     ('capture-checksums.json', 'final-artifacts.bundle', 'final-code-provenance.json')]))
                records = retry_sync(lambda: final_inventory(host, remote_directory))
                for record in records:
                    require(sha256_file(directory / record['path']) == record['sha256'], 'Final archive hash mismatch')
                write_json(directory / 'final-checksums.json', {'files': records})
                write_json(directory / 'final-local-verification.json', {'run_id': directory.name, 'at': utc(),
                           'verified_files': len(records), 'git_commit': final_commit})
                qualified=qualified_final_status(directory,current)
                update(status=current['status'], finished_at=utc(), final_verified_files=len(records), final_git_commit=final_commit)
                return 0 if qualified else 1
            time.sleep(15)
        raise TimeoutError('Local guardian total deadline exceeded')
    except BaseException as error:
        update(status='failed', error=str(error), finished_at=utc())
        # Tell the worker to stop at the barrier; SIGTERM also interrupts a request/replay in progress.
        failure = {'at': utc(), 'error': str(error), 'attempts': 3}
        code = ('import pathlib,json,os,signal\np=pathlib.Path(' + repr(remote_directory) + ')\n'
                '(p/"sync-failed.json").write_text(' + repr(json.dumps(failure)) + ')\n'
                's=json.loads((p/"status.json").read_text())\n'
                'if s["stage"]!="finished":\n'
                ' try: os.kill(s["worker_pid"],signal.SIGTERM)\n'
                ' except ProcessLookupError: pass\n')
        try:
            remote(host, code)
        except Exception as notify_error:
            print('Could not notify worker; its sync/total deadline will clean up:', notify_error, flush=True)
        print(str(error), flush=True)
        return 1
    finally:
        stop.set()
        thread.join(timeout=1)


def launch(args):
    require(args.device == 0, 'Only NPU 0 is authorized')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_id = f'single_npu_{stamp}_{uuid.uuid4().hex[:8]}'
    remote_directory = str(PurePosixPath(args.remote_root) / 'm0a/runs' / run_id)
    directory = ROOT / 'm0a/runs' / run_id
    plan = {'run_id': run_id, 'device': 0, 'host': args.host, 'remote_directory': remote_directory,
        'local_directory': str(directory), 'image': IMAGE, 'image_id': IMAGE_ID, 'requests_per_phase': 48,
        'repetitions': 2, 'window_tokens': 32, 'capacity_mib': [64, 128], 'prefetch_budget_mib': 8,
        'total_timeout_seconds': TOTAL_SECONDS, 'stage_sync_required': True,
        'container_name': 'memecho-' + run_id, 'executed': args.execute}
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'launch.json', plan)
    files = deployment_files()
    commit = archive_code(directory, files)
    deployment = {'files': [{'path': 'implementation/' + str(p.relative_to(ROOT)), 'sha256': sha256_file(p),
                             'size': p.stat().st_size} for p in files], 'git_commit': commit}
    write_json(directory / 'deployment.json', deployment)
    remote(args.host, 'from pathlib import Path\nPath(' + repr(remote_directory) + ').mkdir(parents=True,exist_ok=False)\n')
    transfer_files(args.host, remote_directory, [(p, str(p.relative_to(ROOT))) for p in files], destination_prefix='implementation/')
    transfer_files(args.host, remote_directory, [(directory / n, n) for n in
                    ('launch.json', 'code-provenance.json', 'implementation.bundle', 'deployment.json')])
    worker_script = str(PurePosixPath(remote_directory) / 'implementation/m0a/single_npu_validation.py')
    code = ('import pathlib,subprocess,json,hashlib\np=pathlib.Path(' + repr(remote_directory) + ')\n'
            'for r in ' + repr(deployment['files']) + ':\n'
            ' f=p/r["path"]\n assert f.stat().st_size==r["size"] and hashlib.sha256(f.read_bytes()).hexdigest()==r["sha256"]\n'
            'with (p/"worker.log").open("ab") as log:\n'
            ' child=subprocess.Popen(' + repr(['python3', '-u', worker_script, '--root', args.remote_root, '--run-id', run_id, '--device', '0']) + ',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)\n'
            '(p/"worker.pid").write_text(str(child.pid)+"\\n")\nprint(json.dumps({"worker_pid":child.pid}))\n')
    started = json.loads(remote(args.host, code))
    sync = detached_process([sys.executable, '-u', str(Path(__file__).resolve()), '--guardian', '--host', args.host,
                             '--remote-directory', remote_directory, '--local-directory', str(directory)], directory / 'sync.log')
    (directory / 'sync.pid').write_text(str(sync.pid) + '\n')
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            require(alive(sync.pid), 'Local guardian exited before startup verification')
            if (directory / 'sync-status.json').exists() and (directory / 'status.json').exists():
                guardian_status = json.loads((directory / 'sync-status.json').read_text())
                server = json.loads((directory / 'status.json').read_text())
                require(server['status'] == 'running', 'Server failed during startup: ' + str(server.get('error')))
                if guardian_status.get('last_verified_checkpoint') == '00-bootstrap':
                    fresh = status_remote(args.host, remote_directory)
                    require(fresh['worker_alive'] and fresh['worker_pid'] == started['worker_pid'] and fresh.get('heartbeat'),
                            'Server PID/heartbeat unavailable')
                    require((datetime.now(timezone.utc) - datetime.fromisoformat(fresh['heartbeat'])).total_seconds() < 45,
                            'Server heartbeat is stale')
                    evidence = dict(plan, **started, sync_pid=sync.pid, local_git_commit=commit,
                        verified_checkpoint='00-bootstrap', verified_files=guardian_status['verified_files'],
                        manifest_sha256=guardian_status['last_manifest_sha256'], heartbeat=fresh['heartbeat'],
                        worker_log=remote_directory + '/worker.log', local_sync_log=str(directory / 'sync.log'),
                        status_path=str(directory / 'status.json'), sync_status_path=str(directory / 'sync-status.json'))
                    write_json(directory / 'startup-evidence.json', evidence)
                    print(json.dumps(evidence, ensure_ascii=False, indent=2))
                    return 0
            time.sleep(1)
        raise TimeoutError('Detached startup verification timed out')
    except BaseException:
        # Launcher failure must not leave an orphan experiment running.
        remote(args.host, 'import os,signal\ntry: os.kill(' + str(started['worker_pid']) + ',signal.SIGTERM)\nexcept ProcessLookupError: pass\n')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=int, choices=[0], default=0)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--host', default='hust')
    parser.add_argument('--remote-root', default='/workspace/memecho')
    parser.add_argument('--guardian', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--remote-directory', help=argparse.SUPPRESS)
    parser.add_argument('--local-directory', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.guardian:
        require(args.remote_directory and args.local_directory, 'Guardian paths required')
        return guardian(args.host, args.remote_directory, args.local_directory)
    return launch(args)


if __name__ == '__main__':
    raise SystemExit(main())
