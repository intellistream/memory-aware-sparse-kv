"""Capture verified full-score GLM scenario replays; never label these autonomous Agent traces."""
from __future__ import annotations
import argparse
import gzip
import json
import os
import subprocess
from pathlib import Path
from .glm_agent_cases import verify_model, sha_file, REVISION
from .glm_native_trace import NativeCapture, validate_request, audit_repeats, selected_ids
from .issue6_protocol import validate
from .issue6_workload import write
from .run_glm_native_trace import write_trace
from .echo_workload import sha_ids
from .history_prefetch import require


def export(rows, case, run_id):
    meta = case['material']
    output = []
    for row in rows:
        if row['phase'] != 'decode':
            continue
        query = row['position'] - case['prompt_tokens']
        output.append({'schema_version': 1, 'unit': 'native_kv_token', 'source_kind': 'native_trace',
            'source_scope': meta['source_scope'], 'autonomous_agent': False,
            'task_id': meta['task_id'], 'run_id': run_id, 'request_id': case['case_id'],
            'context_version': case['prompt_token_ids_sha256'], 'rank': 0, 'layer': row['layer'],
            'phase': 'decode', 'query': query, 'prediction_step': query * 2, 'completion_step': query * 2 + 1,
            'position': row['position'], 'stage': 'request:' + case['case_id'], 'event_type': meta['event_type'],
            'eligible_end': row['position'],
            'selected_blocks': sorted(selected_ids(row['raw_selected_ids'], row['position'])),
            'scores': row['full_scores'],
            'score_visibility': 'full_reference_after_native_indexer; logical replay only, no measured prefetch lead',
            'query_text': meta['query_text'], 'fragments': case['fragments'],
            'variant': meta['variant'], 'split': meta['split']})
    return output


def audit(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    require(manifest['status'] in {'captured', 'unstable'}, 'Incomplete native run')
    for name, expected in manifest['artifacts'].items():
        require(Path(name).name == name and sha_file(root / name) == expected, 'Artifact changed: ' + name)
    cases = json.loads((root / 'cases.json').read_text())
    require(len(cases) == len({c['case_id'] for c in cases}), 'Duplicate cases')
    reports, validation_rows, decode_rows = [], [], 0
    export_stream = gzip.open(root / 'trace.jsonl.gz', 'rt')
    for index, case in enumerate(cases):
        requests = []
        require(sha_ids(case['prompt_token_ids']) == case['prompt_token_ids_sha256'], 'Prompt token hash drift')
        for repeat in range(manifest['repeats']):
            path = root / f'case-{index:03d}-repeat-{repeat}.jsonl.gz'
            require(path.name in manifest['artifacts'], 'Unsealed trace')
            with gzip.open(path, 'rt') as stream:
                raw = [json.loads(line) for line in stream]
            enriched = validate_request(raw, case)
            require(raw == enriched and all('full_scores' in row for row in raw), 'Incomplete raw/derived score trace')
            requests.append({'case_id': case['case_id'], 'repeat': repeat,
                             'rows': [{k: v for k, v in row.items() if k != 'full_scores'} for row in raw]})
            if repeat == 0:
                for expected in export(raw, case, manifest['run_id']):
                    line = export_stream.readline()
                    require(line and json.loads(line) == expected, 'Export differs from native raw trace')
                    validation_rows.append({k: v for k, v in expected.items() if k != 'scores'})
                    decode_rows += 1
        reports.extend(audit_repeats(requests, repeats=manifest['repeats'])['cases'])
    require(not export_stream.readline(), 'Unexpected exported rows')
    export_stream.close()
    result = {'all_selected_sets_stable': all(r['set_stable'] for r in reports),
              'cases': sorted(reports, key=lambda r: r['case_id'])}
    validate(validation_rows)
    require(json.loads(json.dumps(result)) == json.loads((root / 'stability.json').read_text()), 'Stability report differs')
    require((manifest['status'] == 'captured') == result['all_selected_sets_stable'], 'Reported stability status differs')
    return {'independent_audit_passed': True, 'fixed_selected_sets_stable': result['all_selected_sets_stable'],
            'case_count': len(cases), 'decode_rows': decode_rows, 'fixed_requests': len(cases) * manifest['repeats'],
            'manifest_sha256': sha_file(root / 'manifest.json'),
            'source_scope': 'scenario_disclosed_controlled_replay', 'autonomous_agent': False}


def run(args):
    require(args.repeats >= 2, 'At least two repeated requests required')
    require(Path(f'/dev/davinci{args.physical_npu}').exists() and
            os.environ.get('ASCEND_RT_VISIBLE_DEVICES') == str(args.runtime_device), 'Device mapping mismatch')
    verify_model(args.model)
    args.output.mkdir(parents=True, exist_ok=False)
    materials = [json.loads(line) for line in args.materials.read_text().splitlines()]
    require(materials and len({(r['task_id'], r['variant']) for r in materials}) == len(materials), 'Duplicate/empty materials')
    manifest = {'schema_version': 1, 'status': 'starting', 'run_id': args.output.name,
        'source_scope': 'scenario_disclosed_controlled_replay', 'autonomous_agent': False,
        'materials_sha256': sha_file(args.materials), 'model': str(args.model), 'model_revision': REVISION,
        'repeats': args.repeats, 'physical_npu': args.physical_npu,
        'runtime_device_before_visibility': args.runtime_device, 'process_device': 0,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'code_sha256': {p.name: sha_file(p) for p in Path(__file__).parent.glob('issue6_*.py')},
        'native_capture_sha256': sha_file(Path(__file__).with_name('glm_native_trace.py')),
        'limitations': ['fixed continuation after 32-token calibration', 'sampled prefill, complete 31-query decode window',
                        'disclosed benchmark scenario; no autonomous task completion claim',
                        'full score reconstruction synchronizes NPU; no online timing eligibility']}
    write(args.output / 'manifest.json', manifest)
    capture, installed, llm = NativeCapture(capture_full_scores=True), False, None
    cases, reports, validation_rows = [], [], []
    export_stream = gzip.open(args.output / 'trace.jsonl.gz', 'wt', encoding='utf-8')
    try:
        import torch
        import torch_npu
        import transformers.configuration_utils as config
        require(torch.npu.device_count() == 1, 'Expected one visible NPU')
        torch.npu.set_device(0)
        if 'deepseek_sparse_attention' not in config.ALLOWED_LAYER_TYPES:
            config.ALLOWED_LAYER_TYPES += ('deepseek_sparse_attention',)
        from vllm import LLM, SamplingParams
        llm = LLM(model=str(args.model), tensor_parallel_size=1, dtype='bfloat16', max_model_len=65536,
            max_num_batched_tokens=8192, max_num_seqs=1, block_size=128, enable_prefix_caching=False,
            enforce_eager=True, gpu_memory_utilization=0.45, async_scheduling=False,
            logits_processors=['m0a.glm_fixed_logits:FixedContinuation'])
        tokenizer = llm.get_tokenizer()
        capture.install()
        installed = True
        for index, material in enumerate(materials):
            requests = []
            ids = tokenizer.apply_chat_template(material['messages'], tools=material['tools'],
                tokenize=True, return_dict=False, add_generation_prompt=True, reasoning_effort='low')
            require(type(ids) is list and all(type(token) is int for token in ids) and 65 < len(ids) <= 65504,
                    f'Unsupported prompt token layout/length: type={type(ids).__name__}, length={len(ids)}')
            case = {'case_id': material['task_id'] + ':' + material['variant'], 'event_position': len(ids) - 64,
                    'prompt_tokens': len(ids), 'prompt_token_ids': ids, 'prompt_token_ids_sha256': sha_ids(ids),
                    'material': material, 'capture_window': 'request_tail_probe; not an actual tool event',
                    'fragments': [{'start': start, 'end': min(start + 128, len(ids)),
                                   'text': tokenizer.decode(ids[start:start + 128])}
                                  for start in range(0, len(ids), 128)]}
            capture.begin(None)
            calibration = llm.generate({'prompt_token_ids': ids}, SamplingParams(temperature=0, max_tokens=32,
                ignore_eos=True, seed=7), use_tqdm=False)[0]
            tokens = list(calibration.outputs[0].token_ids)
            require(len(tokens) == 32 and calibration.prompt_token_ids == ids, 'Calibration changed prompt/output length')
            case['forced_continuation_token_ids'] = tokens
            case['forced_continuation_sha256'] = sha_ids(tokens)
            cases.append(case)
            write(args.output / 'cases.json', cases)
            for repeat in range(args.repeats):
                capture.begin(case)
                result = llm.generate({'prompt_token_ids': ids}, SamplingParams(temperature=0, max_tokens=32,
                    ignore_eos=True, seed=7, extra_args={'fixed_token_ids': tokens}), use_tqdm=False)[0]
                require(result.prompt_token_ids == ids and list(result.outputs[0].token_ids) == tokens,
                        'Replay prompt or continuation changed')
                raw = validate_request(capture.rows, case)
                path = args.output / f'case-{index:03d}-repeat-{repeat}.jsonl.gz'
                write_trace(path, raw)
                # Stability only needs selected IDs and batch shape; release large score arrays.
                requests.append({'case_id': case['case_id'], 'repeat': repeat,
                                 'rows': [{k: v for k, v in row.items() if k != 'full_scores'} for row in raw]})
                if repeat == 0:
                    for record in export(raw, case, args.output.name):
                        export_stream.write(json.dumps(record, separators=(',', ':')) + '\n')
                        validation_rows.append({k: v for k, v in record.items() if k != 'scores'})
                print(json.dumps({'case': index + 1, 'total': len(materials), 'repeat': repeat,
                                  'prompt_tokens': len(ids), 'rows': len(raw)}), flush=True)
            reports.extend(audit_repeats(requests, repeats=args.repeats)['cases'])
        export_stream.close()
        validate(validation_rows)
        stability = {'all_selected_sets_stable': all(r['set_stable'] for r in reports),
                     'cases': sorted(reports, key=lambda r: r['case_id'])}
        write(args.output / 'stability.json', stability)
        manifest['status'] = 'captured' if stability['all_selected_sets_stable'] else 'unstable'
        manifest['case_count'] = len(cases)
        manifest['artifacts'] = {p.name: sha_file(p) for p in args.output.iterdir() if p.name != 'manifest.json'}
    except Exception as error:
        manifest['status'], manifest['error'] = 'failed', f'{type(error).__name__}: {error}'
        raise
    finally:
        export_stream.close()
        if installed:
            capture.restore()
        write(args.output / 'manifest.json', manifest)
        del llm
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--materials', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--model', type=Path, default=Path('/root/models/echo-glm-model'))
    parser.add_argument('--physical-npu', type=int)
    parser.add_argument('--runtime-device', type=int)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    if args.audit:
        result = audit(args.audit)
    else:
        require(args.materials is not None and args.output is not None, 'Materials and new output directory required')
        result = run(args)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
