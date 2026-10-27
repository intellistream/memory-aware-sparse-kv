"""Audit saved CPU component results and plot real-weight indexer selections."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run
    manifest = json.loads((run/'manifest.json').read_text())
    assert not manifest['NPU_used'] and not manifest['full_model_inference']
    for name, expected in manifest['code_sha256'].items():
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == expected, name
    summaries, masks, audited_scores, audited_cache = [], {}, 0, 0
    for report in manifest['results']:
        name = report['name']
        path = run/name
        assert json.loads((path/'summary.json').read_text()) == report
        selected = json.loads((path/'selected-ids.json').read_text())
        data = np.load(path/'numerics.npz', allow_pickle=False)
        assert data['output'].shape == (report['queries'], 4096)
        assert np.isfinite(data['output']).all()
        assert hashlib.sha256(data['output'].tobytes()).hexdigest() == report['output_sha256']
        assert len(selected) == report['queries'] == len(report['query_details'])
        for q, ids in enumerate(selected):
            detail = report['query_details'][q]
            assert len(ids) == len(set(ids)) == detail['selected_compressed']
            assert all(0 <= x < detail['available_compressed'] for x in ids)
            if report['compress_ratio'] == 4:
                scores = data[f'index_scores_q{q}']
                assert len(scores) == detail['available_compressed'] and np.isfinite(scores).all()
                assert np.argsort(-scores, kind='stable')[:512].tolist() == ids
                audited_scores += 1
            else:
                assert ids == list(range(detail['available_compressed']))
        if report['compress_ratio'] == 4:
            rows = [json.loads(line) for line in gzip.open(path/'trace.jsonl.gz', 'rt')]
            assert len(rows) == len(selected)
            for row, ids in zip(rows, selected):
                assert row['selected_blocks'] == sorted(ids)
                assert row['source_kind'] == 'synthetic_fixture'
                assert row['model_behavior_scope'] == 'attention_component_only_not_full_model'
            mask = np.zeros((len(selected), report['query_details'][-1]['available_compressed']))
            for q, ids in enumerate(selected):
                mask[q, ids] = 1
            masks[name] = mask
        for cache in report['cache_checks']:
            assert cache['max_output_abs_error'] == 0
            assert cache['output_sha256'] == report['output_sha256']
            assert cache['peak_resident_units'] <= cache['capacity_units'] == 512
            assert cache['peak_logical_cache_bytes'] == cache['peak_resident_units']*1024
            audited_cache += report['queries']
        baseline = {r['lead_us']: r['mean_query_service_us'] for r in report['timing'] if r['strategy'] == 'none'}
        times = [dict(r, service_reduction_percent=100*(1-r['mean_query_service_us']/baseline[r['lead_us']]))
                 for r in report['timing']]
        shift_jaccard = None
        if report['profile'] == 'shift':
            a, b = map(set, selected[report['queries']//2-1:report['queries']//2+1])
            shift_jaccard = len(a & b)/len(a | b)
        summaries.append(dict(name=name, dense_mask_absmax=report['dense_mask_vs_sparse_gather_max_abs_error'],
            compressor_checks=report['checks'], mean_adjacent_jaccard=report['mean_adjacent_selection_jaccard'],
            shift_boundary_jaccard=shift_jaccard, first_query=report['query_details'][0],
            cache_checks=report['cache_checks'], timing_assumptions_only=times))
    analysis = dict(audit_passed=True, audited_source_files=len(manifest['code_sha256']),
        numerical_queries=sum(r['queries'] for r in manifest['results']), audited_index_score_vectors=audited_scores,
        cache_output_comparisons=audited_cache, loaded_tensor_count=len(manifest['loaded_tensor_receipts']),
        loaded_distinct_tensor_MiB=manifest['loaded_distinct_tensor_bytes']/2**20,
        precision_comparison=manifest['precision_comparison'], results=summaries)
    (run/'analysis.json').write_text(json.dumps(analysis, indent=2))
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), constrained_layout=True)
    for ax, name in zip(axes, masks):
        ax.imshow(masks[name], aspect='auto', interpolation='nearest', cmap='Blues', vmin=0, vmax=1,
                  extent=(-.5, masks[name].shape[1]-.5, masks[name].shape[0]-.5, -.5))
        ax.set(title=name, xlabel='Compressed C4 unit ID (blue = selected by indexer)', ylabel='Decode query')
        if 'shift' in name:
            ax.axhline(15.5, color='#d95f02', lw=1.5, label='Controlled latent changes')
            ax.legend(loc='upper right', fontsize=8)
    fig.suptitle('DeepSeek-V4 real attention weights / controlled hidden states / CPU reference\n'
                 'Actual top-512 selections; not full-model generated activations', fontsize=13)
    fig.savefig(run/'c4-selected-units.png', dpi=170)
    plt.close(fig)
    print(json.dumps({k: v for k,v in analysis.items() if k not in {'results','precision_comparison'}}, indent=2))
    for item in summaries:
        print(json.dumps({k:v for k,v in item.items() if k not in {'cache_checks','timing_assumptions_only','first_query'}}, indent=2))
    print('Precision Jaccard:', manifest['precision_comparison']['mean_selected_Jaccard'])


if __name__ == '__main__':
    main()
