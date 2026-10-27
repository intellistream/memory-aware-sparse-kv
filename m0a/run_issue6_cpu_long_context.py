"""Long-context real-checkpoint C4 component checks; never full-model inference."""
import os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch
from .deepseek_v4_cpu import Checkpoint, AttentionComponent
from .run_deepseek_v4_cpu import run_case


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path,default=Path('/models/DeepSeek-V4-Flash-W8A8'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=8)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(args.threads)
    sources={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    checkpoint=Checkpoint(args.model_dir)
    if checkpoint.config['model_type']!='deepseek_v4' or checkpoint.config['index_topk']!=512:
        raise ValueError('Expected DeepSeek V4 native top-512 configuration')
    component=AttentionComponent(checkpoint,2,'dequant_fp32')
    reports=[]
    # Equal seed and query count control two major experimental factors.
    # Both lengths include stable and shifted inputs to avoid confounding length with shift.
    for tokens in (16384,32768):
        for profile in ('stable','shift'):
            case=dict(name=f'c4-{tokens}-{profile}-fp32',layer=2,prompt_tokens=tokens,
                      queries=16,seed=31,profile=profile,precision='dequant_fp32')
            print('START',case['name'],flush=True)
            reports.append(run_case(checkpoint,component,case,args.output))
            print('DONE',case['name'],reports[-1]['cpu_wall_seconds'],flush=True)
    if sources!={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}:
        raise RuntimeError('Source changed during experiment')
    manifest=dict(schema_version=1,recorded_at=datetime.now(timezone.utc).isoformat(),
        source_commit='65f5af11565123b5ea44d3e6ba8796fa69a73c2d',
        modifications='issue6 CPU acceptance patch, see code_sha256',
        model_dir=str(args.model_dir),model_config_sha256=sha(args.model_dir/'config.json'),
        weight_index_sha256=sha(args.model_dir/'quant_model_weights.safetensors.index.json'),
        loaded_tensor_receipts=checkpoint.receipt,
        loaded_distinct_tensor_bytes=sum(r['bytes'] for r in checkpoint.receipt.values()),
        code_sha256=sources,torch=torch.__version__,threads=args.threads,
        NPU_used=False,full_model_inference=False,task16_acceptance=False,
        scope='C4 layer 2 real-weight CPU attention on controlled hidden states',
        limitation='No natural-language task, long-conversation interference, full-model outputs or measured TPOT',
        cases=len(reports),numerical_query_checks=sum(r['queries_checked'] for r in reports),
        cache_output_checks=sum(r['queries_checked']*len(r['cache_checks']) for r in reports),
        results=reports,artifact_sha256={str(p.relative_to(args.output)):sha(p)
            for p in args.output.rglob('*') if p.is_file()})
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:manifest[k] for k in ('cases','numerical_query_checks','cache_output_checks')}),flush=True)


if __name__=='__main__':
    main()
