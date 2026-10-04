#!/usr/bin/env python3
"""Monitor an archived Pod CPU continuation, mirror hashes, and publish evidence.

Run detached from the local checkout after resume_deepseek_pod_cpu.py starts
on the Pod. This process never changes the serving process or model mount.
"""
from __future__ import annotations

import argparse
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
from m0a.working_set import require
from scripts.launch_single_npu_validation import download_records, remote, retry_sync


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def remote_state(host: str, deployment: str, directory: str) -> dict:
    code = ("import pathlib,json\n"
            f"r=pathlib.Path({deployment!r}); d=pathlib.Path({directory!r})\n"
            "p=int((r/'cpu-postprocess.pid').read_text())\n"
            "q=pathlib.Path(f'/proc/{p}/stat')\n"
            "alive=q.exists() and q.read_text().split()[2]!='Z'\n"
            "print(json.dumps({'alive':alive,'manifest':(d/'sha256-manifest.json').exists(),"
            "'failure':(d/'failure.log').read_text()[-1200:] if (d/'failure.log').exists() else None,"
            "'sidecar':(d/'sidecar.json').exists(),"
            "'replays':sorted(x.parent.name for x in d.glob('replay-*/report.json'))}))\n")
    return json.loads(remote(host, code, timeout=60))


def final_service(host: str, deployment: str, root: str) -> dict:
    code = ("import sys,json,pathlib,subprocess\n"
            f"sys.path.insert(0,{deployment!r})\n"
            "from m0a.pod_runtime import SupervisorService,health\n"
            f"r=pathlib.Path({root!r}); m=pathlib.Path('/models/DeepSeek-V4-Flash-W8A8')\n"
            "s=SupervisorService(r,m); identity=s.identity(); models=health(m)\n"
            "status=subprocess.check_output(['supervisorctl','-c',str(s.conf),'status','memecho-deepseek'],text=True).strip()\n"
            "print(json.dumps({'identity':identity,'models':models,'supervisor':status}))\n")
    result = json.loads(remote(host, code, timeout=60))
    require(result['models'] == [['dsv4', '/models/DeepSeek-V4-Flash-W8A8']] and
            'RUNNING' in result['supervisor'], 'Supervised service is unhealthy')
    return result


def result_document(run_id: str, capture: Path, post: Path, report: dict, provenance: dict,
                    service: dict, manifest_hash: str, verification: dict) -> str:
    trace = json.loads((post / 'trace-validation.json').read_text())
    return (f'# DeepSeek Pod validation result\n\n'
            f'- Capture run: `{run_id}`; capture status: `failed` (`Invalid CP chunk range` validator rule).\n'
            f'- CPU continuation: `engineering_validated`; strict output acceptance: '
            f'`{report["strict_output_acceptance"]}`.\n'
            f'- Capture commit: `{provenance["capture_git_commit"]}`; validator commit: '
            f'`{provenance["validator_git_commit"]}`.\n'
            f'- Valid responses: 48 trace-off and 48 trace-on. Native rows: {trace["raw_rows"]:,}; '
            f'window rows: {trace["window_rows"]:,} (48 × 33 × 21).\n'
            f'- Output differences: `{report["output_difference_counts"]}`. Selected-set comparison: '
            f'`{trace["paired_effect_interpretation"]}`; repeated/prefix drift: '
            f'{trace["repeat_selected_id_differences"]}/{trace["pair_prefix_selected_id_differences"]}.\n'
            f'- Replay reports: `replay-{{per_rank,aggregate}}-{{64,128}}mib/report.json` '
            f'under the postprocess directory.\n'
            f'- Supervised service: `{service["supervisor"]}`; model `dsv4` at the read-only mount.\n'
            f'- Remote capture: `{capture}`; remote report: `{capture / post.name / "report.json"}`.\n'
            f'- Local postprocess: `{post}`; verified files: {verification["verified_files"]}; '
            f'postprocess manifest SHA-256: `{manifest_hash}`.\n\n'
            'This is offline selected-set CPU replay, not online KV offload or a performance gain measurement. '
            'The model metadata hashes and 70 indexed shards were checked; no readable model commit marker or '
            'saved original per-shard hashes were available. The original failed capture remains archived separately.\n')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--host', default='hust')
    parser.add_argument('--remote-root', default='/root/memory-aware-sparse-kv')
    parser.add_argument('--deployment', required=True)
    parser.add_argument('--timeout-hours', type=float, default=8)
    args = parser.parse_args()
    capture = ROOT / 'm0a/runs' / args.run_id
    post = capture / 'postprocess-cp-padding-v1'
    remote_capture = Path(args.remote_root) / 'm0a/runs' / args.run_id
    remote_post = remote_capture / post.name
    state_file = capture / 'automation-status.json'
    deadline = time.monotonic() + args.timeout_hours * 3600
    write_json(state_file, {'run_id': args.run_id, 'status': 'running', 'started_at': utc(),
                            'remote_postprocess': str(remote_post)})
    try:
        while time.monotonic() < deadline:
            try:
                state = remote_state(args.host, args.deployment, str(remote_post))
                write_json(state_file, {'run_id': args.run_id, 'status': 'running', 'updated_at': utc(),
                                        'remote_state': state, 'remote_postprocess': str(remote_post)})
                if state['manifest']:
                    break
                require(state['alive'], 'Remote CPU continuation exited: ' + str(state['failure']))
            except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                print('Transient SSH status error:', error, flush=True)
            time.sleep(45)
        else:
            raise TimeoutError('CPU continuation exceeded the monitoring deadline')
        manifest_bytes = remote(args.host, f"import pathlib,sys\nsys.stdout.buffer.write(pathlib.Path({str(remote_post / 'sha256-manifest.json')!r}).read_bytes())\n", timeout=60)
        manifest = json.loads(manifest_bytes)
        records = manifest['files']
        require(records and len({item['path'] for item in records}) == len(records), 'Invalid postprocess manifest')
        post.mkdir(exist_ok=True)
        retry_sync(lambda: download_records(args.host, str(remote_post), records, str(post)), attempts=3)
        (post / 'sha256-manifest.json').write_bytes(manifest_bytes)
        require(all((post / item['path']).stat().st_size == item['size'] and
                    sha256_file(post / item['path']) == item['sha256'] for item in records),
                'Local postprocess hash verification failed')
        report = json.loads((post / 'report.json').read_text())
        provenance = json.loads((post / 'postprocess-provenance.json').read_text())
        require(report['status'] == 'engineering_validated' and
                report['strict_output_acceptance'] == 'not_qualified' and
                not report['missing_artifacts'], 'Engineering report did not pass')
        service = final_service(args.host, args.deployment, args.remote_root)
        write_json(capture / 'final-service-verification.json', service)
        verification = {'run_id': args.run_id, 'verified_files': len(records),
                        'manifest_sha256': sha256_file(post / 'sha256-manifest.json'),
                        'verified_at': utc(), 'service_healthy': True}
        write_json(capture / 'postprocess-local-verification.json', verification)
        doc = ROOT / 'docs/deepseek-pod-validation-result.md'
        doc.write_text(result_document(args.run_id, remote_capture, post, report, provenance,
                                       service, verification['manifest_sha256'], verification))
        branch = subprocess.check_output(['git', '-C', str(ROOT), 'branch', '--show-current'], text=True).strip()
        require(branch, 'Named branch required for publishing')
        subprocess.run(['git', '-C', str(ROOT), 'add', str(doc.relative_to(ROOT))], check=True)
        subprocess.run(['git', '-C', str(ROOT), 'commit', '--only', '-m',
                        f'Archive DeepSeek Pod engineering result {args.run_id}',
                        str(doc.relative_to(ROOT))], check=True)
        subprocess.run(['git', '-C', str(ROOT), 'push', 'origin',
                        f'HEAD:refs/heads/{branch}'], check=True, timeout=180)
        write_json(state_file, {'run_id': args.run_id, 'status': 'complete', 'finished_at': utc(),
                                'verified_files': len(records), 'service_healthy': True,
                                'report': str(post / 'report.json'), 'branch': branch})
        return 0
    except BaseException as error:
        write_json(state_file, {'run_id': args.run_id, 'status': 'failed', 'finished_at': utc(),
                                'error': str(error), 'remote_postprocess': str(remote_post)})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
