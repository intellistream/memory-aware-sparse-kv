#!/usr/bin/env python3
"""Launch a supervised eight-NPU Pod trace-replay run with checksum synchronization."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
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
              ROOT/'scripts/launch_single_npu_validation.py',Path(__file__).resolve(),ROOT/'docs/deepseek-validation.md',
              ROOT/'m0a/vllm-ascend-trace.patch',ROOT/'scripts/serve_dsv4_pod.py',
              ROOT/'docs/deepseek-pod-runbook.md']
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
          'container_name':'memecho-'+run_id,'executed':args.execute,
          'validation_mode':args.validation_mode,'runtime':args.runtime,'model_dir':str(args.model_dir) if args.model_dir else None,
          'execution_available':args.runtime=='pod' and args.validation_mode=='trace-replay'}
    plan.update(diagnostic_only=getattr(args,'diagnostic_only',False),diagnostic_requests_per_candidate=20,
                repair=getattr(args,'repair',False),operator_replays=20,deterministic_rank_verification=bool(getattr(args,'repair',False)),
                candidate_order=['original_fresh','eager','eager_single_stream','eager_single_stream_no_mtp',
                                 'eager_single_stream_no_mtp_sync','eager_single_stream_no_mtp_sync_seq1'],
                trace_off_and_on_share_selected_configuration=True,production_configuration_changed=False)
    if plan['repair']:
        plan['candidate_order'] += ['eager_single_stream_no_mtp_sync_seq1_hccl','eager_single_stream_no_mtp_sync_seq1_hccl_npu']
    if args.validation_mode=='trace-replay':
        plan.update(candidate_order=['original_fresh'],diagnostic_requests_per_candidate=0,
                    strict_output_acceptance='not_qualified')
    if not args.execute:
        print(json.dumps(plan,ensure_ascii=False,indent=2))
        return 0
    require(args.runtime=='pod', 'Live execution is disabled for the old Docker runtime; pass --runtime pod')
    require(args.validation_mode=='trace-replay' and not args.repair and not args.diagnostic_only,
            'Pod execution requires trace-replay without repair/diagnostic mode')
    require(args.model_dir and args.model_dir.is_absolute() and str(args.model_dir).startswith('/models/'),
            'Pass the absolute read-only --model-dir under /models')
    return launch_pod(args, plan, directory, remote_directory)


def launch_pod(args, plan, directory, remote_directory):
    """Transfer a checked source snapshot, start the service, then detach the worker."""
    directory.mkdir(parents=True,exist_ok=False)
    write_json(directory/'launch.json',plan)
    files=deployment_files()
    commit=archive_code(directory,files,message='Archive Pod DeepSeek trace-replay implementation')
    deployment={'files':[{'path':'implementation/'+str(p.relative_to(ROOT)),'sha256':sha256_file(p),'size':p.stat().st_size}
                         for p in files], 'git_commit':commit}
    write_json(directory/'deployment.json',deployment)
    # Bundle the actual PR branch separately from the exact worker snapshot.
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    require(branch=='feat/deepseek-trace-replay-offline', 'Expected the existing draft PR branch')
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),
            'Commit all Pod adapter changes before live execution')
    bundle=directory/'pr-branch.bundle'
    subprocess.run(['git','bundle','create',str(bundle),branch],cwd=ROOT,check=True)
    subprocess.run(['git','bundle','verify',str(bundle)],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    write_json(directory/'branch-provenance.json',{'branch':branch,'head':head,'bundle_sha256':sha256_file(bundle)})
    remote(args.host,'from pathlib import Path\nPath('+repr(remote_directory)+').mkdir(parents=True,exist_ok=False)\n')
    transfer_files(args.host,remote_directory,[(p,str(p.relative_to(ROOT))) for p in files],destination_prefix='implementation/')
    transfer_files(args.host,remote_directory,[(directory/n,n) for n in
        ('launch.json','code-provenance.json','implementation.bundle','deployment.json','pr-branch.bundle','branch-provenance.json')])
    bootstrap='''import hashlib,json,os,pathlib,subprocess,sys,urllib.request
p=pathlib.Path(DIRECTORY)
for row in FILES:
 f=p/row['path']
 assert f.stat().st_size==row['size'] and hashlib.sha256(f.read_bytes()).hexdigest()==row['sha256']
provenance=json.loads((p/'branch-provenance.json').read_text())
assert hashlib.sha256((p/'pr-branch.bundle').read_bytes()).hexdigest()==provenance['bundle_sha256']
checkout=pathlib.Path('/root')/('memecho-deploy-'+p.name)
assert not checkout.exists()
subprocess.run(['git','clone','-b',provenance['branch'],str(p/'pr-branch.bundle'),str(checkout)],check=True)
assert subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()==provenance['head']
sys.path.insert(0,str(p/'implementation'))
from m0a.pod_runtime import SupervisorService,preflight,health,port_available,same_process,sha256,service_command
from m0a.run_requests import request
model=pathlib.Path(MODEL)
root=pathlib.Path(ROOT)
service=SupervisorService(root,model)
for old in sorted((root/'m0a/runs').glob('deepseek_*/status.json')):
 state=json.loads(old.read_text())
 identity=old.parent/'worker-identity.json'
 if identity.exists() and same_process(json.loads(identity.read_text())) and state['stage']!='finished':
  raise RuntimeError('Prior Pod validation is still active: '+str(old.parent))
if port_available():
 preflight(model,expect_port_free=True)
 service.install()
 service.control('start')
else:
 preflight(model,expect_port_free=False)
 saved=json.loads((service.runtime/'service-command.json').read_text())
 assert saved['command']==service_command(model) and saved['model_dir']==str(model)
 assert service.identity()
models=health(model)
result=request({'model':'dsv4','messages':[{'role':'user','content':'Reply with OK.'}],
                'temperature':0,'max_tokens':8})
assert result['choices'] and result['usage']['completion_tokens']>0
evidence={'checkout':str(checkout),'git_commit':provenance['head'],'models':models,
          'process_identity':service.identity(),'supervisor_config_sha256':sha256(service.conf),
          'short_request_id':result['id']}
(p/'service-bootstrap.json').write_text(json.dumps(evidence,indent=2)+'\\n')
print(json.dumps(evidence))
'''.replace('DIRECTORY',repr(remote_directory)).replace('FILES',repr(deployment['files'])).replace('MODEL',repr(str(args.model_dir))).replace('ROOT',repr(args.remote_root))
    bootstrap_result=json.loads(remote(args.host,bootstrap,timeout=1200))
    write_json(directory/'service-bootstrap.json',bootstrap_result)
    start='''import json,os,pathlib,subprocess,sys
p=pathlib.Path(DIRECTORY)
env=dict(os.environ,PYTHONPATH=str(p/'implementation'))
with (p/'worker.log').open('ab') as log:
 child=subprocess.Popen([sys.executable,'-u','-m','m0a.deepseek_pod_validation',
                         '--root',ROOT,'--run-id',p.name],env=env,
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                        start_new_session=True,close_fds=True)
(p/'worker.pid').write_text(str(child.pid)+'\\n')
print(json.dumps({'worker_pid':child.pid}))
'''.replace('DIRECTORY',repr(remote_directory)).replace('ROOT',repr(args.remote_root))
    started=json.loads(remote(args.host,start))
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
                require(fresh['worker_alive'] and fresh['worker_pid']==started['worker_pid'], 'Server worker failed at startup')
                evidence=dict(plan,**started,sync_pid=sync.pid,local_git_commit=commit,
                              verified_checkpoint='00-bootstrap',service_bootstrap=bootstrap_result,
                              worker_log=remote_directory+'/worker.log',local_sync_log=str(directory/'sync.log'))
                write_json(directory/'startup-evidence.json',evidence)
                print(json.dumps(evidence,ensure_ascii=False,indent=2))
                return 0
        time.sleep(1)
    raise TimeoutError('Detached Pod startup verification timed out; watchdog and guardian own recovery')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--runtime',choices=('pod','docker'),default='docker')
    parser.add_argument('--model-dir',type=Path)
    parser.add_argument('--validation-mode',choices=('strict','trace-replay'),default='strict')
    parser.add_argument('--diagnostic-only',action='store_true',help='Stop after stability diagnostics and restore the original service')
    parser.add_argument('--repair',action='store_true',help='Test deterministic candidates, then diagnose and replay operators if stability fails')
    parser.add_argument('--host',default='hust')
    parser.add_argument('--remote-root',default='/root/memory-aware-sparse-kv')
    parser.add_argument('--guardian',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--remote-directory',help=argparse.SUPPRESS)
    parser.add_argument('--local-directory',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.validation_mode=='trace-replay' and (args.repair or args.diagnostic_only):
        parser.error('--validation-mode trace-replay cannot be combined with --repair or --diagnostic-only')
    if args.runtime=='pod' and args.model_dir is None:
        parser.error('--runtime pod requires --model-dir')
    if args.guardian:
        require(args.remote_directory and args.local_directory,'Guardian paths required')
        return guardian(args.host,args.remote_directory,args.local_directory)
    return launch(args)


if __name__=='__main__': raise SystemExit(main())
