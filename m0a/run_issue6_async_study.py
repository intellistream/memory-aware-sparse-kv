"""Reproduce asynchronous rank replay of saved real-weight CPU component traces."""
import os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD','0')
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .issue6_async_simulator import AsyncConfig, CostProfile, memory_plan, replay
from .issue6_protocol import validate
from .issue6_simulator import predict


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_steps(rows, choices, ranks, strategy):
    steps=[]
    for row,candidates in zip(rows,choices):
        group=[]
        for rank in range(ranks):
            clone=deepcopy(row)
            clone['rank']=rank
            group.append(dict(row=clone,candidates=candidates,strategy=strategy,
                              signal_kind='oracle' if strategy=='oracle_diagnostic' else 'history'))
        steps.append(group)
    return steps


def compact(result):
    times=[r['end_us']-r['begin_us'] for r in result['collectives']]
    total=result['decode_schedule_us']
    hidden=0
    for t in result['transfers']:
        if t['reason']=='prefetch':
            matching=[r for r in result['records'] if r['rank']==t['rank'] and r['step']==t['step']]
            if matching:
                r=matching[0]
                hidden+=max(0,min(t['end_us'],r['native_ready_us'])-max(t['start_us'],r['begin_us']))
    return dict(decode_schedule_us=total, mean_group_service_us=float(np.mean(times)),
        p95_group_service_us=float(np.percentile(times,95)),
        mean_rank_wait_us=float(np.mean([r['wait_us'] for r in result['records']])),
        mean_barrier_wait_us=float(np.mean([r['barrier_wait_us'] for r in result['records']])),
        h2d_bytes=result['total_h2d_bytes'],d2h_bytes=result['total_d2h_bytes'],
        unused_prefetch_bytes=result['unused_prefetch_bytes'],prefetch_overlap_rank_sum_us=hidden,
        peak_slots=max(result['peak_slots_by_namespace'].values()),
        numerical_attention_checks=0,metrics_are_measured_TPOT=False)


def verify_real_payloads(source, output, model_dir):
    import torch
    from .deepseek_v4_cpu import Checkpoint, AttentionComponent, controlled_hidden
    torch.set_num_threads(8)
    report=json.loads((source/'c4-4096-stable-fp32'/'summary.json').read_text())
    checkpoint=Checkpoint(model_dir)
    component=AttentionComponent(checkpoint,2,'dequant_fp32')
    hidden=controlled_hidden(report['seed'],report['prompt_tokens'],report['queries'],dim=4096)
    state=component.prepare(hidden,report['prompt_tokens'])
    # Assert actual checkpoint content, not just its path, matches the archived run.
    old=json.loads((source/'manifest.json').read_text())
    assert all(old['loaded_tensor_receipts'][k]==v for k,v in checkpoint.receipt.items())
    rows=[json.loads(line) for line in gzip.open(source/'c4-4096-stable-fp32'/'trace.jsonl.gz','rt')][:8]
    baseline=np.load(source/'c4-4096-stable-fp32'/'numerics.npz',allow_pickle=False)['output'][:8]
    reports=[]
    for strategy in ('none','global_frequency'):
        errors=[]
        def verify(row,payloads):
            q=row['query']
            selected,_=component.select(state,q)
            assert set(selected.tolist())==set(row['selected_blocks'])
            lookup=dict(zip(row['selected_blocks'],payloads))
            gathered=torch.stack([lookup[k] for k in selected.tolist()])
            out,_=component.attend(state,q,selected,supplied=gathered)
            torch.testing.assert_close(out,torch.from_numpy(baseline[q]),atol=0,rtol=0)
            errors.append(float((out-torch.from_numpy(baseline[q])).abs().max()))
        result=replay(make_steps(rows,predict(rows,strategy,512,None,None),1,strategy),
            AsyncConfig(chunk_units=64),payload_provider=lambda ns,k:state['compressed'][k],attention_check=verify)
        reports.append(dict(strategy=strategy,queries=len(errors),max_output_abs_error=max(errors),
            output_equal_to_archived_real_weight_CPU=True,total_h2d_bytes=result['total_h2d_bytes']))
        with gzip.open(output/f'payload-{strategy}.json.gz','wt') as stream:
            json.dump(result,stream)
    return dict(scope='actual_BF16_payloads_through_async_cache_then_actual_attention',
                inputs='controlled_hidden_not_full_model',reports=reports,
                weight_receipts_verified=len(checkpoint.receipt))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,default=Path('/models/DeepSeek-V4-Flash-W8A8'))
    parser.add_argument('--verify-payloads',action='store_true')
    parser.add_argument('--profile',type=Path)
    parser.add_argument('--device-label',default='unmeasured-logical-ranks')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    source_manifest=json.loads((args.source/'manifest.json').read_text())
    assert source_manifest['cases']==5 and not source_manifest['full_model_inference']
    base=AsyncConfig(chunk_units=64,device=args.device_label)
    configurations={
        'rank1-lead0':replace(base,native_us=0),
        'rank1-lead50':base,
        'rank1-cap1024':replace(base,capacity=1024),
        'rank8-shared':replace(base,ranks=8),
        'rank8-per-rank':replace(base,ranks=8,host_link='per_rank'),
        'rank8-straggler':replace(base,ranks=8,rank_slowdown=(1,1,1,1,1,1,1,2)),
    }
    profile=CostProfile(args.profile,model=base.model,device=base.device)
    if profile.data is not None:
        configurations={k:v for k,v in configurations.items() if v.ranks==profile.data['parallel_ranks']}
        assert configurations, 'No scenario matches profile parallelism'
    strategies=('none','last_selection','global_frequency','stage_frequency','oracle_diagnostic')
    summaries,receipts=[],{}
    for case in ('c4-4096-stable-fp32','c4-8192-shift-fp32'):
        source=args.source/case/'trace.jsonl.gz'
        receipts[case]=sha(source)
        rows=[json.loads(line) for line in gzip.open(source,'rt')]
        validate(rows)
        assert all(r['source_scope']=='real_checkpoint_CPU_attention_controlled_hidden' for r in rows)
        for name,cfg in configurations.items():
            baseline=None
            for strategy in strategies:
                choices=predict(rows,strategy,cfg.budget,None,None)
                result=replay(make_steps(rows,choices,cfg.ranks,strategy),cfg,profile,capture_transitions=False)
                filename=f'{case}--{name}--{strategy}.json.gz'
                with gzip.open(args.output/filename,'wt') as stream:
                    json.dump(result,stream,separators=(',',':'))
                summary=dict(case=case,scenario=name,strategy=strategy,**compact(result),artifact=filename)
                if strategy=='none':
                    baseline=summary['decode_schedule_us']
                summary['service_reduction_percent']=100*(1-summary['decode_schedule_us']/baseline)
                summaries.append(summary)
            print('DONE',case,name,flush=True)
    weight_index=args.model_dir/'quant_model_weights.safetensors.index.json'
    weights=json.loads(weight_index.read_text())['metadata']['total_size']
    plans=[]
    for tokens in (4096,8192,32768,131072):
        for concurrent in (1,16):
            plans.append(dict(tokens=tokens,concurrent_requests=concurrent,**memory_plan(source_manifest['model_config'],
                tokens=tokens,ranks=8,capacity=512,concurrent_requests=concurrent,weights_total_bytes=weights,
                device_bytes=64*2**30,workspace_bytes=8*2**30,safety_bytes=4*2**30)))
    payload=verify_real_payloads(args.source,args.output,args.model_dir) if args.verify_payloads else None
    manifest=dict(schema_version=1,source_manifest_sha256=sha(args.source/'manifest.json'),source_trace_sha256=receipts,
        weight_index_sha256=sha(weight_index),rank_expansion='identical_selected_C4_KV_replica_fixture_not_native_rank_capture',
        full_model_executed=False,native_NPU_validated=False,cost_profile_sha256=profile.sha256,
        configurations={k:asdict(v) for k,v in configurations.items()},
        source_files_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        scenario_count=len(summaries),summaries=summaries,memory_plans=plans,payload_validation=payload,
        artifact_sha256={p.name:sha(p) for p in args.output.glob('*.gz')})
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(scenarios=len(summaries),payload_validation=payload)),flush=True)


if __name__=='__main__':
    main()
