"""Execute actual-checkpoint attention components, numerical checks and KV replay."""
import os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
import argparse
import gc
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .deepseek_v4_cpu import Checkpoint, AttentionComponent, TensorLRU, controlled_hidden, digest
from .issue6_predictors import Predictor
from .issue6_protocol import context, Observation, validate
from .issue6_simulator import Timing, simulate


def c4_row(case, q, selected):
    p = case['prompt_tokens']+q
    return dict(schema_version=2, unit='compressed_kv_token_position', unit_bytes=1024,
        source_kind='synthetic_fixture', source_scope='real_checkpoint_CPU_attention_controlled_hidden',
        model_behavior_scope='attention_component_only_not_full_model',
        attention_scope='deepseek_v4_C4_indexed_KV_only', autonomous_agent=False,
        compression_ratio=4, token_position=p, position=(p+1)//4-1, eligible_end=p//4,
        locally_generated_ids=list(range(p//4, (p+1)//4)), scores=None,
        task_id=case['name'], split='test', variant=case['profile'], run_id='dsv4-cpu-component',
        request_id=case['name'], context_version=case['hidden_sha256'], rank=0, layer=case['layer'],
        phase='decode', query=q, prediction_step=2*q, completion_step=2*q+1,
        stage='latent-after' if case['profile']=='shift' and q>=case['queries']//2 else 'latent-before',
        event_type='artificial_hidden_state_condition', query_text='Controlled latent input; no semantic claim',
        fragments=[], selected_blocks=selected)


@torch.inference_mode()
def run_case(checkpoint, component, case, output):
    start = time.perf_counter()
    path = output/case['name']
    path.mkdir()
    hidden = controlled_hidden(case['seed'], case['prompt_tokens'], case['queries'],
                               dim=checkpoint.config['hidden_size'], shift=case['profile']=='shift')
    case = dict(case, hidden_sha256=digest(hidden))
    state = component.prepare(hidden, case['prompt_tokens'])
    rows, selected_sets, scores_saved, outputs, details = [], [], [], [], []
    for q in range(case['queries']):
        selected, scores = component.select(state, q)
        selected_sets.append(selected.tolist())
        if scores is not None:
            scores_saved.append(scores.numpy())
        out, checks = component.attend(state, q, selected, compare_dense=True)
        outputs.append(out)
        available = (case['prompt_tokens']+q+1)//component.ratio if component.ratio else 0
        details.append(dict(query=q, position=case['prompt_tokens']+q, available_compressed=available,
                            selected_compressed=len(selected), **checks))
        if component.ratio == 4:
            rows.append(c4_row(case, q, sorted(selected.tolist())))
    baseline = torch.stack(outputs)
    np.savez_compressed(path/'numerics.npz', output=baseline.numpy(),
                        **{f'index_scores_q{i}': s for i,s in enumerate(scores_saved)})
    # Variable-length IDs are plain JSON; no pickled arrays in the archive.
    (path/'selected-ids.json').write_text(json.dumps(selected_sets))
    cache_checks, timing_results = [], []
    if rows:
        validate(rows)
        with gzip.open(path/'trace.jsonl.gz', 'wt') as stream:
            for row in rows:
                stream.write(json.dumps(row)+'\n')
        for strategy in ('none', 'last_selection', 'global_frequency', 'stage_frequency'):
            predictor = Predictor(strategy, semantic=None, event=None, size=len(state['compressed'])+1)
            cache = TensorLRU(512, state['compressed'])
            choices, errors, case_outputs = [], [], []
            for q, row in enumerate(rows):
                candidates, _ = predictor.predict(context(row), 512)
                choices.append(candidates)
                for block in candidates:
                    cache.insert(block)
                for block in row['locally_generated_ids']:
                    cache.insert(block, kind='generated')
                selected = selected_sets[q]  # preserve score order for identical reduction order
                resident_kv = cache.recall(selected)
                result, _ = component.attend(state, q, torch.tensor(selected), supplied=resident_kv)
                torch.testing.assert_close(result, baseline[q], atol=0, rtol=0)
                errors.append(float((result-baseline[q]).abs().max()))
                case_outputs.append(result)
                predictor.observe(Observation(q, 2*q+1, tuple(row['selected_blocks'])))
            cache_checks.append(dict(strategy=strategy, max_output_abs_error=max(errors),
                output_sha256=digest(torch.stack(case_outputs)), capacity_units=512, peak_resident_units=cache.peak,
                prefetch_units=cache.prefetch, sync_recall_units=cache.demand, locally_generated_units=cache.generated,
                unit_bytes=1024, peak_logical_cache_bytes=cache.peak*1024,
                scope='CPU_payload_recall_not_physical_device_memory'))
            for lead in (0,50):
                records, transfers, _ = simulate([rows], [choices], capacity=512, budget=512,
                    unit_bytes=1024, timing=Timing(lead_us=lead), strategy=strategy)
                # Independently implemented payload cache and old accounting agree.
                assert sum(r['prefetch_units'] for r in records) == cache.prefetch
                assert sum(r['sync_recall_units'] for r in records) == cache.demand
                timing_results.append(dict(strategy=strategy, lead_us=lead,
                    mean_query_service_us=sum(r['service_us'] for r in records)/len(records),
                    mean_wait_us=sum(r['wait_us'] for r in records)/len(records),
                    total_transfer_bytes=sum(t['bytes'] for t in transfers),
                    timing_is_assumed=True, metric_scope='one_attention_layer_not_TPOT'))
    overlaps = [len(set(a)&set(b))/len(set(a)|set(b)) for a,b in zip(selected_sets,selected_sets[1:]) if set(a)|set(b)]
    report = dict(case, compress_ratio=component.ratio, precision=component.precision,
        hidden_source='controlled_prototypes_plus_noise_not_model_activations',
        real_checkpoint_weights=True, whole_model_executed=False, native_NPU_kernel_validated=False,
        cache_storage='BF16', math_accumulation='FP32', quantized_weight_handling='INT8_times_per_output_scale',
        activation_quantization='symmetric_INT8_reference' if component.precision=='dynamic_int8_reference' else 'disabled',
        indexer_quantization='symmetric_INT8_reference' if component.precision=='dynamic_int8_reference' else 'disabled',
        config_dimensions={k:checkpoint.config[k] for k in ('hidden_size','head_dim','num_attention_heads','index_n_heads',
            'index_head_dim','index_topk','q_lora_rank','o_groups','o_lora_rank','sliding_window')},
        checks=state['checks'], queries_checked=len(details),
        dense_mask_vs_sparse_gather_max_abs_error=max(r['dense_mask_absmax'] for r in details),
        mean_adjacent_selection_jaccard=sum(overlaps)/len(overlaps) if overlaps else None,
        output_sha256=digest(baseline), output_all_finite=bool(torch.isfinite(baseline).all()),
        cpu_wall_seconds=time.perf_counter()-start, cache_checks=cache_checks, timing=timing_results, query_details=details)
    if not report['output_all_finite']:
        raise ValueError('Nonfinite attention output')
    (path/'summary.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, default=Path('/models/DeepSeek-V4-Flash-W8A8'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=8)
    args=p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    checkpoint=Checkpoint(args.model_dir)
    assert checkpoint.config['model_type']=='deepseek_v4'
    assert checkpoint.config['index_topk']==512 and checkpoint.config['hidden_size']==4096
    specifications=[
        dict(name='c4-4096-stable-fp32', layer=2, prompt_tokens=4096, queries=32, seed=7, profile='stable', precision='dequant_fp32'),
        dict(name='c4-8192-shift-fp32', layer=2, prompt_tokens=8192, queries=32, seed=19, profile='shift', precision='dequant_fp32'),
        dict(name='c4-4096-stable-int8', layer=2, prompt_tokens=4096, queries=32, seed=7, profile='stable', precision='dynamic_int8_reference'),
        dict(name='c128-4096-fp32', layer=3, prompt_tokens=4096, queries=16, seed=7, profile='stable', precision='dequant_fp32'),
        dict(name='swa-4096-fp32', layer=0, prompt_tokens=4096, queries=16, seed=7, profile='stable', precision='dequant_fp32'),
    ]
    reports=[]
    component=None
    for case in specifications:
        if component is None or component.layer != case['layer']:
            del component
            gc.collect()
            component=AttentionComponent(checkpoint, case['layer'], case['precision'])
        component.precision=case['precision']
        print('START',case['name'],flush=True)
        reports.append(run_case(checkpoint,component,case,args.output))
        print('DONE',case['name'],reports[-1]['cpu_wall_seconds'],flush=True)
    a=json.loads((args.output/specifications[0]['name']/'selected-ids.json').read_text())
    b=json.loads((args.output/specifications[2]['name']/'selected-ids.json').read_text())
    overlap=[len(set(x)&set(y))/len(set(x)|set(y)) for x,y in zip(a,b)]
    files=('deepseek_v4_cpu.py','run_deepseek_v4_cpu.py','test_deepseek_v4_cpu.py','issue6_simulator.py',
           'issue6_evaluate.py','issue6_protocol.py','issue6_predictors.py','history_prefetch.py')
    manifest=dict(model_dir=str(args.model_dir), model_config=checkpoint.config,
        model_config_sha256=hashlib.sha256((args.model_dir/'config.json').read_bytes()).hexdigest(),
        weight_index_sha256=hashlib.sha256((args.model_dir/'quant_model_weights.safetensors.index.json').read_bytes()).hexdigest(),
        loaded_tensor_receipts=checkpoint.receipt, loaded_distinct_tensor_bytes=sum(r['bytes'] for r in checkpoint.receipt.values()),
        code_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in files},
        torch=torch.__version__, threads=args.threads, NPU_used=False, full_model_inference=False,
        cases=len(reports), numerical_query_checks=sum(r['queries_checked'] for r in reports),
        precision_comparison=dict(scope='same_controlled_hidden_and_checkpoint_fp32_vs_emulated_dynamic_int8',
                                  mean_selected_Jaccard=sum(overlap)/len(overlap), per_query=overlap),
        results=reports)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(cases=len(reports),queries=manifest['numerical_query_checks'],
        loaded_MiB=manifest['loaded_distinct_tensor_bytes']/2**20,
        quantization_selection_Jaccard=manifest['precision_comparison']['mean_selected_Jaccard'])),flush=True)


if __name__=='__main__':
    main()
