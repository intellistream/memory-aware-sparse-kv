#!/usr/bin/env python3
"""Print the DeepSeek plan by default; --execute launches a detached validated run."""
from __future__ import annotations
import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from m0a.deepseek_validation import IMAGE, IMAGE_ID, TOTAL_SECONDS, utc, write_json
from m0a.contracts import sha256_file
from m0a.working_set import require
from scripts.launch_single_npu_validation import (remote, transfer_files, archive_code, detached_process,
                                                 alive, guardian, status_remote, retry_sync)


def deployment_files():
    files=list((ROOT/'m0a').glob('*.py'))
    files += [ROOT/'m0a/model_profiles.json',ROOT/'m0a/pairs.json',ROOT/'m0a/source/m0a_selected_trace.py',
              ROOT/'scripts/launch_single_npu_validation.py',Path(__file__).resolve(),ROOT/'docs/deepseek-validation.md']
    return sorted(files)


def previous_run_barrier(args):
    """Finish an older owned run's recovery and synchronization before new work."""
    code = '''import json,pathlib,os,signal
root=pathlib.Path(ROOT)
rows=[]
for p in sorted((root/'m0a/runs').glob('deepseek_*/status.json')):
 s=json.loads(p.read_text())
 try:
  cmd=pathlib.Path(f'/proc/{s["worker_pid"]}/cmdline').read_bytes()
  running=str(p.parent/'implementation/m0a/deepseek_validation.py').encode() in cmd
 except FileNotFoundError: running=False
 if running and s['stage'] not in ('finished','restore_original'):
  source=(p.parent/'implementation/m0a/deepseek_validation.py').read_text()
  assert 'class TaskDeadline(BaseException)' in source, 'Older worker has no non-retryable exit'
  os.kill(s['worker_pid'],signal.SIGALRM)
 rows.append({'directory':str(p.parent),'run_id':s['run_id'],'running':running,'stage':s['stage']})
print(json.dumps(rows))
'''.replace('ROOT',repr(args.remote_root))
    rows = json.loads(retry_sync(lambda:remote(args.host,code)))
    deadline = time.monotonic()+1800
    for row in rows:
        if not row['running']:
            continue
        while time.monotonic()<deadline:
            status = retry_sync(lambda:status_remote(args.host,row['directory']))
            if status['stage']=='finished' and not status['worker_alive'] and not status.get('watchdog_alive'):
                require(status.get('original_service_restored'), 'Prior worker did not restore the original service')
                break
            time.sleep(15)
        else:
            raise TimeoutError('Prior run recovery exceeded its independent budget')
    if rows:
        prior = rows[-1]
        local = ROOT/'m0a/runs'/prior['run_id']
        local.mkdir(parents=True,exist_ok=True)
        # Let an existing guardian finish; two writers would race on final bundles.
        sync_path = local/'sync-status.json'
        sync_deadline = time.monotonic()+900
        while sync_path.exists():
            state = json.loads(sync_path.read_text())
            if state.get('status') != 'running' or not alive(state['pid']):
                break
            require(time.monotonic()<sync_deadline,'Prior guardian final synchronization timed out')
            time.sleep(15)
        final = local/'final-local-verification.json'
        if not final.exists():
            # A failed experiment can still have a complete, verified final archive.
            guardian(args.host,prior['directory'],local)
        require((local/'final-local-verification.json').exists(), 'Prior final synchronization incomplete')
        require(json.loads(final.read_text())['run_id']==prior['run_id'],'Prior final verification belongs to another run')
        for record in json.loads((local/'final-checksums.json').read_text())['files']:
            require(sha256_file(local/record['path'])==record['sha256'],'Prior final artifact SHA-256 mismatch')
        audit = '''import json,pathlib,subprocess,os,urllib.request
root=pathlib.Path(ROOT);p=pathlib.Path(PRIOR)
saved=json.loads((p/'original-service.json').read_text())
env=dict(os.environ,DOCKER_HOST='unix://'+str(root/'runtime/docker.sock'))
current=json.loads(subprocess.check_output(['docker','inspect',saved['Id']],env=env))[0]
assert current['State']['Running'] and current['Config']==saved['Config'] and current['HostConfig']==saved['HostConfig']
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
assert opener.open('http://127.0.0.1:8900/health',timeout=10).status==200
models=json.load(opener.open('http://127.0.0.1:8900/v1/models',timeout=10))
assert [(m['id'],m['root']) for m in models['data']]==[tuple(m) for m in json.loads((p/'original-models.json').read_text())]
print(json.dumps({'container_id':saved['Id'],'healthy':True,'config_unchanged':True,'prior_run':p.name}))
'''.replace('ROOT',repr(args.remote_root)).replace('PRIOR',repr(prior['directory']))
        return json.loads(retry_sync(lambda:remote(args.host,audit)))
    return {'prior_run':None}


def launch(args):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_id=f'deepseek_{stamp}_{uuid.uuid4().hex[:8]}'
    remote_directory=str(PurePosixPath(args.remote_root)/'m0a/runs'/run_id)
    directory=ROOT/'m0a/runs'/run_id
    plan={'run_id':run_id,'devices':list(range(8)),'host':args.host,'remote_directory':remote_directory,
          'local_directory':str(directory),'image':IMAGE,'image_id':IMAGE_ID,'requests_per_phase':48,'repetitions':2,
          'window_tokens':32,'capacity_mib':[64,128],'budget_scopes':['per_rank','aggregate'],'prefetch_budget_mib':8,
          'total_timeout_seconds':TOTAL_SECONDS,'stage_sync_required':True,'restore_original_service':True,
          'container_name':'memecho-'+run_id,'executed':args.execute}
    plan.update(diagnostic_only=getattr(args,'diagnostic_only',False),diagnostic_requests_per_candidate=20,
                repair=getattr(args,'repair',False),operator_replays=20,deterministic_rank_verification=bool(getattr(args,'repair',False)),
                candidate_order=['original_fresh','eager','eager_single_stream','eager_single_stream_no_mtp',
                                 'eager_single_stream_no_mtp_sync','eager_single_stream_no_mtp_sync_seq1'],
                trace_off_and_on_share_selected_configuration=True,production_configuration_changed=False)
    if plan['repair']:
        plan['candidate_order'] += ['eager_single_stream_no_mtp_sync_seq1_hccl','eager_single_stream_no_mtp_sync_seq1_hccl_npu']
    if not args.execute:
        print(json.dumps(plan,ensure_ascii=False,indent=2))
        return 0
    recovery = previous_run_barrier(args)
    directory.mkdir(parents=True,exist_ok=False)
    write_json(directory/'prior-run-recovery.json',recovery)
    write_json(directory/'launch.json',plan)
    files=deployment_files()
    commit=archive_code(directory,files,message='Implement DeepSeek operator diagnosis, deterministic candidates and recovery')
    deployment={'files':[{'path':'implementation/'+str(p.relative_to(ROOT)),'sha256':sha256_file(p),'size':p.stat().st_size} for p in files], 'git_commit':commit}
    write_json(directory/'deployment.json',deployment)
    remote(args.host,'from pathlib import Path\nPath('+repr(remote_directory)+').mkdir(parents=True,exist_ok=False)\n')
    transfer_files(args.host,remote_directory,[(p,str(p.relative_to(ROOT))) for p in files],destination_prefix='implementation/')
    transfer_files(args.host,remote_directory,[(directory/n,n) for n in ('launch.json','code-provenance.json','implementation.bundle','deployment.json')])
    worker_script=str(PurePosixPath(remote_directory)/'implementation/m0a/deepseek_validation.py')
    code=('import pathlib,subprocess,json,hashlib\np=pathlib.Path('+repr(remote_directory)+')\n'
          'for r in '+repr(deployment['files'])+':\n'
          ' f=p/r["path"]\n assert f.stat().st_size==r["size"] and hashlib.sha256(f.read_bytes()).hexdigest()==r["sha256"]\n'
          'with (p/"worker.log").open("ab") as log:\n'
          ' child=subprocess.Popen('+repr(['python3','-u',worker_script,'--root',args.remote_root,'--run-id',run_id])+',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)\n'
          '(p/"worker.pid").write_text(str(child.pid)+"\\n")\nprint(json.dumps({"worker_pid":child.pid}))\n')
    started=json.loads(remote(args.host,code))
    try:
        sync=detached_process([sys.executable,'-u',str(Path(__file__).resolve()),'--guardian','--host',args.host,
                              '--remote-directory',remote_directory,'--local-directory',str(directory)],directory/'sync.log')
        (directory/'sync.pid').write_text(str(sync.pid)+'\n')
        deadline=time.monotonic()+180
        while time.monotonic()<deadline:
            require(alive(sync.pid),'Local guardian exited before startup verification')
            if (directory/'sync-status.json').exists():
                state=json.loads((directory/'sync-status.json').read_text())
                if state.get('last_verified_checkpoint')=='00-bootstrap':
                    fresh=status_remote(args.host,remote_directory)
                    require(fresh['worker_alive'] and fresh['worker_pid']==started['worker_pid'] and fresh['status']=='running', 'Server failed at startup')
                    require((datetime.now(timezone.utc)-datetime.fromisoformat(fresh['heartbeat'])).total_seconds()<45,'Stale server heartbeat')
                    evidence=dict(plan,**started,sync_pid=sync.pid,local_git_commit=commit,
                                  verified_checkpoint='00-bootstrap',verified_files=state['verified_files'],
                                  manifest_sha256=state['last_manifest_sha256'],heartbeat=fresh['heartbeat'],
                                  worker_log=remote_directory+'/worker.log',local_sync_log=str(directory/'sync.log'),
                                  status_path=str(directory/'status.json'),sync_status_path=str(directory/'sync-status.json'))
                    write_json(directory/'startup-evidence.json',evidence)
                    print(json.dumps(evidence,ensure_ascii=False,indent=2))
                    return 0
            time.sleep(1)
        raise TimeoutError('Detached startup verification timed out')
    except BaseException:
        remote(args.host,'import os,signal\ntry: os.kill('+str(started['worker_pid'])+',signal.SIGTERM)\nexcept ProcessLookupError: pass\n')
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--diagnostic-only',action='store_true',help='Stop after stability diagnostics and restore the original service')
    parser.add_argument('--repair',action='store_true',help='Test deterministic candidates, then diagnose and replay operators if stability fails')
    parser.add_argument('--host',default='hust')
    parser.add_argument('--remote-root',default='/workspace/memecho')
    parser.add_argument('--guardian',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--remote-directory',help=argparse.SUPPRESS)
    parser.add_argument('--local-directory',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.guardian:
        require(args.remote_directory and args.local_directory,'Guardian paths required')
        return guardian(args.host,args.remote_directory,args.local_directory)
    return launch(args)


if __name__=='__main__': raise SystemExit(main())
