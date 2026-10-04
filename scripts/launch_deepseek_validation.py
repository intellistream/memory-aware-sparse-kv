#!/usr/bin/env python3
"""Launch a supervised eight-NPU Pod trace-replay run with checksum synchronization."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
import shlex
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
from m0a.tau3_bundle import build as build_tau3_bundles
from m0a.working_set import require
from scripts.launch_single_npu_validation import (remote, transfer_files, archive_code, detached_process,
                                                 alive, guardian, status_remote, retry_sync)
from scripts.finalize_deepseek_validation import finalize_run


def deployment_files():
    files=list((ROOT/'m0a').glob('*.py'))
    files += [ROOT/'m0a/model_profiles.json',ROOT/'m0a/pairs.json',ROOT/'m0a/source/m0a_selected_trace.py',
              ROOT/'scripts/launch_single_npu_validation.py',Path(__file__).resolve(),ROOT/'docs/deepseek-validation.md',
              ROOT/'m0a/vllm-ascend-trace.patch',ROOT/'scripts/serve_dsv4_pod.py',
              ROOT/'docs/deepseek-pod-runbook.md', ROOT/'m0a/tau3-requirements.txt',
              ROOT/'scripts/finalize_deepseek_validation.py']
    return sorted(files)


def previous_run_barrier(args):
    """Reject overlap or unrestored service before creating a new run."""
    code = '''import json,pathlib,sys
root=pathlib.Path(ROOT)
active=[]
for status in sorted((root/'m0a/runs').glob('deepseek_*/status.json')):
 try: state=json.loads(status.read_text())
 except (OSError,ValueError): continue
 if state.get('stage')=='finished':
  if state.get('original_service_restored') is False:
   active.append({'run':status.parent.name,'reason':'service restoration unverified'})
  continue
 identity=status.parent/'worker-identity.json'
 if identity.exists():
  sys.path.insert(0,str(status.parent/'implementation'))
  from m0a.pod_runtime import same_process
  if same_process(json.loads(identity.read_text())):
   active.append({'run':status.parent.name,'reason':'worker active'})
print(json.dumps(active))
'''.replace('ROOT',repr(args.remote_root))
    active = json.loads(remote(args.host, code))
    require(not active, 'Prior Pod validation needs recovery: ' + json.dumps(active))
    return {'prior_runs_checked': True}


def launch(args):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_id=f'deepseek_{stamp}_{uuid.uuid4().hex[:8]}'
    remote_directory=str(PurePosixPath(args.remote_root)/'m0a/runs'/run_id)
    directory=ROOT/'m0a/runs'/run_id
    plan={'run_id':run_id,'devices':list(range(8)),'host':args.host,'remote_directory':remote_directory,
          'local_directory':str(directory),'image':IMAGE if args.runtime=='docker' else None,
          'image_id':IMAGE_ID if args.runtime=='docker' else None,
          'requests_per_phase':192 if args.workload=='tau3_v1.0.1' else 48,'repetitions':2,
          'window_tokens':32,'capacity_mib':[64,128],'budget_scopes':['per_rank','aggregate'],'prefetch_budget_mib':8,
          'total_timeout_seconds':86400 if args.workload=='tau3_v1.0.1' else TOTAL_SECONDS,
          'stage_sync_required':True,'restore_original_service':True,
          'container_name':'memecho-'+run_id if args.runtime=='docker' else None,
          'supervisor_service':'memecho-deepseek' if args.runtime=='pod' else None,'executed':args.execute,
          'validation_mode':args.validation_mode,'runtime':args.runtime,'model_dir':str(args.model_dir) if args.model_dir else None,
          'execution_available':args.runtime=='pod' and args.validation_mode=='trace-replay',
          'workload':args.workload,
          'tau_root':str(PurePosixPath(args.remote_root)/'runtime/tau2-v1.0.1')
              if args.workload=='tau3_v1.0.1' else None}
    plan.update(diagnostic_only=getattr(args,'diagnostic_only',False),diagnostic_requests_per_candidate=20,
                repair=getattr(args,'repair',False),operator_replays=20,deterministic_rank_verification=bool(getattr(args,'repair',False)),
                candidate_order=['original_fresh','eager','eager_single_stream','eager_single_stream_no_mtp',
                                 'eager_single_stream_no_mtp_sync','eager_single_stream_no_mtp_sync_seq1'],
                trace_off_and_on_share_selected_configuration=True,production_configuration_changed=False)
    if plan['repair']:
        plan['candidate_order'] += ['eager_single_stream_no_mtp_sync_seq1_hccl','eager_single_stream_no_mtp_sync_seq1_hccl_npu']
    if args.validation_mode=='trace-replay':
        plan.update(candidate_order=['original_fresh'] if args.workload=='synthetic' else plan['candidate_order'],
                    diagnostic_requests_per_candidate=0 if args.workload=='synthetic' else 20,
                    strict_output_acceptance='not_qualified')
    if not args.execute:
        print(json.dumps(plan,ensure_ascii=False,indent=2))
        return 0
    require(args.runtime=='pod', 'Live execution is disabled for the old Docker runtime; pass --runtime pod')
    require(args.validation_mode=='trace-replay' and not args.repair and not args.diagnostic_only,
            'Pod execution requires trace-replay without repair/diagnostic mode')
    require(args.model_dir and args.model_dir.is_absolute() and str(args.model_dir).startswith('/models/'),
            'Pass the absolute read-only --model-dir under /models')
    lock_path = ROOT / 'm0a/runs/.launch.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another validation launch owns the local startup lock') from error
        return launch_pod(args, plan, directory, remote_directory)


def launch_pod(args, plan, directory, remote_directory):
    """Transfer a checked source snapshot, start the service, then detach the worker."""
    previous_run_barrier(args)
    directory.mkdir(parents=True,exist_ok=False)
    write_json(directory/'launch.json',plan)
    bundles = (build_tau3_bundles(directory, ROOT/'m0a/tau3-requirements.txt',
                                 tau_source=Path('/tmp/memecho-tau2-bench'),
                                 wheelhouse=Path('/tmp/memecho-tau3-wheelhouse-aarch64'),
                                 uv_cache=Path('/tmp/memecho-uv-aarch64'))
               if args.workload=='tau3_v1.0.1' else None)
    if bundles:
        write_json(directory/'tau3-bundles.json',bundles)
    files=deployment_files()
    commit=archive_code(directory,files,message='Archive Pod DeepSeek trace-replay implementation')
    deployment={'files':[{'path':'implementation/'+str(p.relative_to(ROOT)),'sha256':sha256_file(p),'size':p.stat().st_size}
                         for p in files], 'git_commit':commit, 'tau3_bundles': bundles}
    write_json(directory/'deployment.json',deployment)
    # Bundle the active source branch separately from the exact worker snapshot.
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    require(bool(branch), 'Run from a named branch for final evidence push')
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),
            'Commit all Pod adapter changes before live execution')
    bundle=directory/'source-branch.bundle'
    subprocess.run(['git','bundle','create',str(bundle),branch],cwd=ROOT,check=True)
    subprocess.run(['git','bundle','verify',str(bundle)],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    write_json(directory/'branch-provenance.json',{'branch':branch,'head':head,'bundle_sha256':sha256_file(bundle)})
    remote(args.host,'from pathlib import Path\nPath('+repr(remote_directory)+').mkdir(parents=True,exist_ok=False)\n')
    transfer_files(args.host,remote_directory,[(p,str(p.relative_to(ROOT))) for p in files],destination_prefix='implementation/')
    transfer_files(args.host,remote_directory,[(directory/n,n) for n in
        ('launch.json','code-provenance.json','implementation.bundle','deployment.json','source-branch.bundle','branch-provenance.json')])
    if bundles:
        transfer_files(args.host,remote_directory,
                       [(directory/value['archive'],value['archive']) for value in bundles.values()] +
                       [(directory/'tau3-bundles.json','tau3-bundles.json')])
    bootstrap='''import hashlib,json,os,pathlib,subprocess,sys,tarfile,urllib.request
p=pathlib.Path(DIRECTORY)
for row in FILES:
 f=p/row['path']
 assert f.stat().st_size==row['size'] and hashlib.sha256(f.read_bytes()).hexdigest()==row['sha256']
provenance=json.loads((p/'branch-provenance.json').read_text())
assert hashlib.sha256((p/'source-branch.bundle').read_bytes()).hexdigest()==provenance['bundle_sha256']
checkout=pathlib.Path('/root')/('memecho-deploy-'+p.name)
assert not checkout.exists()
subprocess.run(['git','clone','-b',provenance['branch'],str(p/'source-branch.bundle'),str(checkout)],check=True)
assert subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()==provenance['head']
sys.path.insert(0,str(p/'implementation'))
from m0a.pod_runtime import SupervisorService,preflight,health,port_available,same_process,sha256,service_command
from m0a.run_requests import request
model=pathlib.Path(MODEL)
root=pathlib.Path(ROOT)
service=SupervisorService(root,model)
launch=json.loads((p/'launch.json').read_text())
tool_smoke=None
if launch['workload']=='tau3_v1.0.1':
 tau=pathlib.Path(launch['tau_root'])
 expected='fc0055dc4e0a316c3f83133267fbd6faaa770992'
 bundles=json.loads((p/'tau3-bundles.json').read_text())
 assert bundles==json.loads((p/'deployment.json').read_text())['tau3_bundles']
 assert os.uname().machine==bundles['uv']['machine']
 def unpack(info,destination):
  archive_path=p/info['archive']
  assert archive_path.stat().st_size==info['size'] and sha256(archive_path)==info['sha256']
  destination.mkdir(parents=True,exist_ok=True)
  with tarfile.open(archive_path,'r:gz') as archive:
   members=archive.getmembers()
   manifest_member=next(m for m in members if m.name==info['manifest'])
   manifest=json.load(archive.extractfile(manifest_member))
   rows={row['path']:row for row in manifest['files']}
   assert len(rows)==info['files'] and {m.name for m in members}==set(rows)|{info['manifest']}
   for member in members:
    if member.name==info['manifest']: continue
    name=pathlib.PurePosixPath(member.name)
    assert member.isfile() and not name.is_absolute() and '..' not in name.parts
    record=rows[member.name]
    assert member.size==record['size']
    target=destination/member.name
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name(target.name+'.part')
    digest=hashlib.sha256()
    with archive.extractfile(member) as source,temporary.open('wb') as output:
     for block in iter(lambda:source.read(1048576),b''):
      output.write(block);digest.update(block)
    assert digest.hexdigest()==record['sha256']
    temporary.replace(target)
   (destination/info['manifest']).write_text(json.dumps(manifest,sort_keys=True,indent=2))
  # Prior runs may leave Python's generated bytecode in this dedicated runtime.
  # Every bundled source file is checked above; only bytecode caches are ignored.
  actual={str(f.relative_to(destination)) for f in destination.rglob('*')
          if f.is_file() and not (f.suffix=='.pyc' and '__pycache__' in f.parts)}
  assert actual==set(rows)|{info['manifest']}
  return manifest
 source_manifest=unpack(bundles['source'],tau)
 assert source_manifest['commit']==expected
 wheelhouse=root/'runtime/tau3-wheelhouse-aarch64'
 wheel_manifest=unpack(bundles['wheels'],wheelhouse)
 uv_info=bundles['uv']
 uv=p/uv_info['archive']
 assert uv.stat().st_size==uv_info['size'] and sha256(uv)==uv_info['sha256']
 uv.chmod(0o755)
 uv_version=subprocess.check_output([str(uv),'--version'],text=True).strip()
 assert uv_version.startswith('uv '+uv_info['version']+' ') and uv_info['machine'] in uv_version
 uv_env=dict(os.environ,UV_CACHE_DIR=str(root/'runtime/tau3-uv-cache'),
             UV_PYTHON_DOWNLOADS='never')
 venv=root/'runtime/tau3-v1.0.1-venv'
 if not (venv/'bin/python').exists():
  subprocess.run([str(uv),'venv','--python',sys.executable,str(venv)],
                 check=True,env=uv_env)
 req=p/'implementation/m0a/tau3-requirements.txt'
 digest=hashlib.sha256(req.read_bytes()).hexdigest()
 assert wheel_manifest['requirements_sha256']==digest
 marker=venv/'installed-requirements.sha256'
 if not marker.exists() or marker.read_text().strip()!=digest:
  subprocess.run([str(uv),'pip','install','--offline','--no-index',
                  '--find-links',str(wheelhouse),'--python',str(venv/'bin/python'),
                  '-r',str(req)],check=True,env=uv_env,timeout=1200)
  marker.write_text(digest)
 tau_env={k:v for k,v in os.environ.items() if k.lower() not in
          ('http_proxy','https_proxy','all_proxy')}
 subprocess.run([str(venv/'bin/python'),'-c','import tau2,rank_bm25'],
                check=True,env=dict(tau_env,PYTHONPATH=str(tau/'src')))
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
if launch['workload']=='tau3_v1.0.1':
 smoke=''' + repr('''import json
from tau2.domains.retail.environment import get_environment
from m0a.run_requests import request
env=get_environment()
tool=next(t for t in env.get_tools() if t.name=='find_user_id_by_name_zip')
response=request({'model':'dsv4','messages':[{'role':'user','content':'Find user Yusuf Rossi in ZIP 19122.'}],
                  'tools':[tool.openai_schema],
                  'tool_choice':{'type':'function','function':{'name':tool.name}},
                  'max_tokens':128,'temperature':0})
calls=response['choices'][0]['message'].get('tool_calls') or []
assert calls and calls[0]['function']['name']==tool.name
args=json.loads(calls[0]['function']['arguments'])
value=env.make_tool_call(tool.name,**args)
assert value is not None
print(json.dumps({'tool':tool.name,'args':args,'result':str(value)[:200],
                  'request_id':response['id']}))''') + '''
 output=subprocess.check_output([str(venv/'bin/python'),'-c',smoke],
  env=dict(tau_env,PYTHONPATH=str(p/'implementation')+os.pathsep+str(tau/'src')),timeout=300,text=True)
 tool_smoke=json.loads(output.splitlines()[-1])
evidence={'checkout':str(checkout),'git_commit':provenance['head'],'models':models,
          'process_identity':service.identity(),'supervisor_config_sha256':sha256(service.conf),
          'short_request_id':result['id'],'tool_smoke':tool_smoke}
(p/'service-bootstrap.json').write_text(json.dumps(evidence,indent=2)+'\\n')
print(json.dumps(evidence))
'''
    bootstrap=(bootstrap.replace('DIRECTORY',repr(remote_directory))
                        .replace('FILES',repr(deployment['files']))
                        .replace('MODEL',repr(str(args.model_dir)))
                        .replace('ROOT',repr(args.remote_root)))
    bootstrap_output=remote(args.host,bootstrap,timeout=3600).decode().splitlines()
    require(bootstrap_output, 'Remote Pod bootstrap returned no evidence')
    bootstrap_result=json.loads(bootstrap_output[-1])
    write_json(directory/'service-bootstrap.json',bootstrap_result)
    start='''import json,os,pathlib,subprocess,sys
p=pathlib.Path(DIRECTORY)
env=dict(os.environ,PYTHONPATH=str(p/'implementation')+os.pathsep+os.environ.get('PYTHONPATH',''))
sys.path.insert(0,str(p/'implementation'))
from m0a.pod_runtime import process_identity
with (p/'worker.log').open('ab') as log:
 child=subprocess.Popen([sys.executable,'-u','-m','m0a.deepseek_pod_validation',
                         '--root',ROOT,'--run-id',p.name],env=env,
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                        start_new_session=True,close_fds=True)
(p/'worker.pid').write_text(str(child.pid)+'\\n')
identity=process_identity(child.pid)
assert identity and identity['pid']==child.pid
(p/'launcher-worker-identity.json').write_text(json.dumps(identity,indent=2)+'\\n')
print(json.dumps({'worker_pid':child.pid,'worker_identity':identity}))
'''.replace('DIRECTORY',repr(remote_directory)).replace('ROOT',repr(args.remote_root))
    started=json.loads(remote(args.host,start))
    sync=detached_process([sys.executable,'-u',str(Path(__file__).resolve()),'--guardian','--host',args.host,
                          '--remote-directory',remote_directory,'--local-directory',str(directory)],directory/'sync.log')
    (directory/'sync.pid').write_text(str(sync.pid)+'\n')
    deadline=time.monotonic()+900
    while time.monotonic()<deadline:
        require(alive(sync.pid),'Local guardian exited before startup verification')
        if (directory/'sync-status.json').exists():
            state=json.loads((directory/'sync-status.json').read_text())
            ack_path=directory/'acks/00-bootstrap.json'
            if ack_path.exists():
                ack=json.loads(ack_path.read_text())
                manifest=directory/'checkpoints/00-bootstrap.json'
                require(manifest.exists() and sha256_file(manifest)==ack['manifest_sha256'],
                        'Bootstrap checkpoint manifest hash mismatch')
                require(all(sha256_file(directory/r['path'])==r['sha256'] for r in ack['files']),
                        'Bootstrap checkpoint artifact hash mismatch')
                fresh=status_remote(args.host,remote_directory)
                require(fresh['worker_alive'] and fresh['worker_pid']==started['worker_pid'], 'Server worker failed at startup')
                identity_code='''import json,pathlib,sys
sys.path.insert(0,IMPL)
from m0a.pod_runtime import same_process
p=pathlib.Path(DIRECTORY)
worker=json.loads((p/'launcher-worker-identity.json').read_text())
watchdog=json.loads((p/'watchdog.json').read_text()) if (p/'watchdog.json').exists() else None
print(json.dumps({'worker':same_process(worker),
                  'watchdog':bool(watchdog and same_process(watchdog['identity']))}))
'''.replace('IMPL',repr(remote_directory+'/implementation')).replace('DIRECTORY',repr(remote_directory))
                identities=json.loads(remote(args.host,identity_code))
                if not identities['watchdog']:
                    time.sleep(1)
                    continue
                require(identities['worker'], 'Worker PID was reused or command changed')
                evidence=dict(plan,**started,sync_pid=sync.pid,local_git_commit=commit,
                              verified_checkpoint='00-bootstrap',service_bootstrap=bootstrap_result,
                              worker_log=remote_directory+'/worker.log',local_sync_log=str(directory/'sync.log'),
                              watchdog_identity_verified=True,
                              checkpoint_manifest_sha256=ack['manifest_sha256'],
                              status_path=str(directory/'status.json'),
                              sync_status_path=str(directory/'sync-status.json'),
                              final_status_path=str(directory/'final-status.json'),
                              recovery_command=' '.join(shlex.quote(x) for x in
                                  [sys.executable,str(Path(__file__).resolve()),'--guardian',
                                   '--host',args.host,'--remote-directory',remote_directory,
                                   '--local-directory',str(directory)]))
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
    parser.add_argument('--workload',choices=('tau3_v1.0.1','synthetic'),default='synthetic')
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
        result=guardian(args.host,args.remote_directory,args.local_directory)
        return finalize_run(args.host,args.remote_directory,args.local_directory,
                            guardian_exit_code=result)
    return launch(args)


if __name__=='__main__': raise SystemExit(main())
