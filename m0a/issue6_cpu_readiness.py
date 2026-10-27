"""Read-only resource evidence for the CPU-only issue6 acceptance run."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_inventory(root):
    root=Path(root).resolve()
    config_path=root/'config.json'
    config=json.loads(config_path.read_text())
    index=next((root/name for name in ('quant_model_weights.safetensors.index.json','model.safetensors.index.json')
                if (root/name).is_file()),None)
    if index is None:
        raise ValueError('No checkpoint weight index')
    data=json.loads(index.read_text())
    shards=sorted(set(data['weight_map'].values()))
    if not shards:
        raise ValueError('Empty weight index')
    for name in shards:
        if Path(name).is_absolute() or '..' in Path(name).parts or not (root/name).resolve().is_relative_to(root):
            raise ValueError('Unsafe indexed shard path')
    missing=[name for name in shards if not (root/name).is_file()]
    return dict(path=str(root),config_sha256=sha(config_path),index_path=str(index),index_sha256=sha(index),
        indexed_shards=len(shards),missing_shards=missing,
        indexed_weight_bytes=data.get('metadata',{}).get('total_size'),
        disk_bytes_present=sum((root/name).stat().st_size for name in shards if (root/name).is_file()),
        architecture=config.get('model_type'),index_topk=config.get('index_topk'),
        checkpoint_structure_verified=not missing and config.get('model_type')=='deepseek_v4',
        full_weight_contents_hashed=False,model_execution_verified=False)


def collect(model_dir):
    devices=sorted(str(p) for p in Path('/dev').glob('davinci*') if re.fullmatch(r'davinci\d+',p.name))
    try:
        proc=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=8)
        smi=dict(returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr)
    except (OSError,subprocess.TimeoutExpired) as exc:
        smi=dict(error=str(exc))
    resources={}
    for name in ('memory.max','cpu.max'):
        path=Path('/sys/fs/cgroup')/name
        resources[name]=path.read_text().strip() if path.exists() else None
    versions={}
    for package in ('torch','numpy','simpy','safetensors','matplotlib'):
        try:
            versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package]=None
    checkpoint=model_inventory(model_dir)
    blockers=[]
    if len(devices)!=8:
        blockers.append('eight_NPUs_not_mounted')
    if smi.get('returncode')!=0:
        blockers.append('NPU_health_not_verified')
    if not checkpoint['checkpoint_structure_verified']:
        blockers.append('checkpoint_structure_incomplete')
    return dict(schema_version=1,recorded_at=datetime.now(timezone.utc).isoformat(),
        scope='CPU_acceptance_inventory_not_runtime_qualification',devices=devices,npu_smi=smi,
        cgroup=resources,versions=versions,checkpoint=checkpoint,
        online_acceptance_blockers=blockers,full_issue6_completed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,default=Path('/models/DeepSeek-V4-Flash-W8A8'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    report=collect(args.model_dir)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
