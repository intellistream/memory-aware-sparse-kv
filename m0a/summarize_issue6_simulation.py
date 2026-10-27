"""Produce plots and a compact report from a completed simulation study."""
import csv
import hashlib
import json
from pathlib import Path
import sys

from .issue6_evaluate import interval


def main():
    root = Path(sys.argv[1])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    scenarios = ('stable', 'shift', 'random', 'misleading')
    strategies = ('last_selection', 'global_frequency', 'stage_frequency', 'semantic_lsa',
                  'event_history', 'adaptive', 'oracle_diagnostic')
    expected = [f'{s}-seed{seed}' for s in scenarios for seed in (7, 19, 31)] + [
        'sensitivity', 'contention-2', 'contention-4', 'length-2048', 'length-16384', 'audited-GLM-replay']
    summaries = {n: json.loads((root/n/'summary.json').read_text()) for n in expected}
    detail = {}
    for name in expected:
        with (root/name/'per-task.csv').open() as f:
            detail[name] = list(csv.DictReader(f))

    def pooled(names, capacity=512, budget=512, timing_id=0):
        rows = [r for n in names for r in detail[n] if int(r['capacity']) == capacity and
                int(r['budget']) == budget and int(r['timing_id']) == timing_id]
        baseline = {r['task_id']: r for r in rows if r['strategy'] == 'none'}
        result = {}
        for strategy in sorted({r['strategy'] for r in rows}):
            selected = [r for r in rows if r['strategy'] == strategy]
            metrics = {m: interval([float(r[m]) for r in selected]) for m in
                       ('service_us', 'wait_us', 'coverage', 'sync_bytes_per_query',
                        'total_bytes_per_query', 'wasted_bytes_per_query', 'boundary_wait_us', 'steady_wait_us')}
            metrics['reduction'] = interval([(float(baseline[r['task_id']]['service_us'])-float(r['service_us'])) /
                                            float(baseline[r['task_id']]['service_us']) for r in selected])
            result[strategy] = metrics
        return result

    aggregate = {s: pooled([f'{s}-seed{seed}' for seed in (7, 19, 31)]) for s in scenarios}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    for ax, scenario in zip(axes.flat, scenarios):
        metrics = [aggregate[scenario][s]['reduction'] for s in strategies]
        means = [m['mean']*100 for m in metrics]
        errors = [[(m['mean']-m['low'])*100 for m in metrics], [(m['high']-m['mean'])*100 for m in metrics]]
        ax.barh(strategies, means, xerr=errors, color=['#4477aa']*6+['#999999'], capsize=3)
        ax.axvline(0, color='black', lw=.8)
        ax.invert_yaxis()
        ax.set_title(scenario)
        ax.set_xlabel('Simulated per-lane service reduction (%)')
    fig.suptitle('Synthetic C4 stress tests | C=512, P=512 | 12 synthetic test tasks\n'
                 'Assumed: 16 GB/s, lead 50 us, predictor 5 us; 95% task bootstrap. Not NPU measurements.')
    fig.savefig(root/'synthetic-comparison.png', dpi=160)
    plt.close(fig)

    sensitivity = summaries['sensitivity']['summary']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    values = []
    for strategy in ('last_selection', 'adaptive', 'oracle_diagnostic'):
        matrix = np.array([[next(r['metrics']['paired_service_reduction']['mean']*100 for r in sensitivity
            if r['strategy'] == strategy and r['timing']['bandwidth_GBs'] == 16 and
            r['timing']['lead_us'] == lead and r['timing']['prediction_us'] == cost)
            for lead in (0, 10, 50, 200)] for cost in (0, 5, 50)])
        values.append(matrix)
    bound = max(abs(m).max() for m in values)
    for ax, strategy, matrix in zip(axes, ('last_selection', 'adaptive', 'oracle_diagnostic'), values):
        im = ax.imshow(matrix, cmap='RdBu', vmin=-bound, vmax=bound, aspect='auto')
        ax.set_xticks(range(4), [0, 10, 50, 200])
        ax.set_yticks(range(3), [0, 5, 50])
        ax.set_xlabel('Assumed lead time (us)')
        ax.set_ylabel('Assumed predictor cost (us)')
        ax.set_title(strategy)
        for i in range(3):
            for j in range(4):
                ax.text(j, i, f'{matrix[i,j]:.1f}%', ha='center', va='center',
                        color='white' if abs(matrix[i,j]) > bound*.55 else 'black')
    fig.colorbar(im, ax=axes, label='Simulated service reduction (%)', shrink=.8)
    fig.suptitle('Timing sensitivity on synthetic shifted hotspots | 16 GB/s | C=P=512\nUncalibrated assumptions; per-lane query service, not TPOT')
    fig.savefig(root/'timing-sensitivity.png', dpi=160)
    plt.close(fig)

    native = [pooled(['audited-GLM-replay'], capacity=4096, timing_id=i) for i in (0, 1)]
    fig, ax = plt.subplots(figsize=(12, 5), layout='constrained')
    names = [s for s in native[0] if s != 'none']
    x = np.arange(len(names))
    for i, lead in enumerate((0, 50)):
        ax.bar(x + (i-.5)*.38, [native[i][s]['reduction']['mean']*100 for s in names],
               width=.38, label=f'Assumed lead {lead} us')
    ax.set_xticks(x, names, rotation=25, ha='right')
    ax.axhline(0, color='black', lw=.8)
    ax.set_ylabel('Simulated service reduction (%)')
    ax.legend()
    ax.set_title('Audited controlled GLM trace replay | C=4096, P=512, unit=1152 bytes\n'
                 'True selection IDs; hypothetical timing. Not DeepSeek or autonomous Agent evidence.')
    fig.savefig(root/'native-replay.png', dpi=160)
    plt.close(fig)

    all_configs = sum(len(s['summary']) for s in summaries.values())
    result = dict(configuration_strategy_rows=all_configs, run_count=len(expected),
        synthetic_C512_P512=aggregate, native_C4096_P512=native,
        timing_assumptions=summaries['shift-seed7']['summary'][0]['timing'],
        caveat='All timing is simulated; no DeepSeek forward pass, NPU DMA, TPOT, TTFT or numerical correctness validation.')
    (root/'study-summary.json').write_text(json.dumps(result, indent=2))
    lines = ['# Issue 6 CPU simulation results', '',
             f'{len(expected)} runs; {all_configs} configuration/strategy rows. All runs completed.', '',
             '**All latency numbers are simulated, conditional on uncalibrated inputs.**', '',
             'Synthetic table: C=512, P=512, 1024-byte C4 unit, B=16 GB/s, lead=50 us, predictor=5 us,',
             'launch=5 us per 128 units, attention=20 us. Three seeds, 12 synthetic test-task clusters per scenario.', '',
             'Positive reduction means faster than no-prefetch. Oracle is diagnostic, not an implementable predictor.', '',
             '| Scenario | Predictor | Coverage | Service us/query | Reduction vs none, 95% CI | Transfer KiB/query | Waste KiB/query |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for scenario in scenarios:
        for strategy in ('none',) + strategies:
            m = aggregate[scenario][strategy]
            gain = m['reduction']
            lines.append(f"| {scenario} | {strategy} | {m['coverage']['mean']:.1%} | {m['service_us']['mean']:.2f} | "
                         f"{gain['mean']:.1%} [{gain['low']:.1%}, {gain['high']:.1%}] | "
                         f"{m['total_bytes_per_query']['mean']/1024:.1f} | {m['wasted_bytes_per_query']['mean']/1024:.1f} |")
    lines += ['', '## Audited GLM replay', '',
              'Original 6696 rows retained as the source; evaluation uses only its four held-out tasks.',
              'Controlled continuation, not an autonomous Agent or DeepSeek run. Unit remains 1152 bytes.', '',
              '| Lead us | Predictor | Service us/query | Reduction vs none | Total KiB/query |',
              '| ---: | --- | ---: | ---: | ---: |']
    for lead, metrics in zip((0, 50), native):
        for strategy, m in metrics.items():
            lines.append(f"| {lead} | {strategy} | {m['service_us']['mean']:.2f} | {m['reduction']['mean']:.1%} | {m['total_bytes_per_query']['mean']/1024:.1f} |")
    lines += ['', '## Scope', '',
              'FIFO non-preemptible speculative batches drain before demand recall. Destination reservations count against capacity.',
              'Shared-link replicas are independent streams; no TP/layer scheduling model or CPU-worker contention.',
              'Task bootstrap does not cover simulator/hardware-model error. Synthetic semantics are manufactured region cues.',
              'Real Agent generalization (issue tasks 11–17), physical asynchronous offload and numerical correctness (18–20) remain unvalidated.', '',
              'See the companion protocol document for tool selection, analytical equations, reproduction and issue acceptance mapping.', '',
              '![Synthetic comparisons](synthetic-comparison.png)', '',
              '![Timing sensitivity](timing-sensitivity.png)', '',
              '![Audited GLM replay](native-replay.png)', '']
    (root/'RESULTS.md').write_text('\n'.join(lines))
    # Hash all result files after report/plots are complete; raw inputs have their own receipts.
    artifacts = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'artifacts-sha256.json'}
    (root/'artifacts-sha256.json').write_text(json.dumps(artifacts, indent=2))
    print(json.dumps(dict(runs=len(expected), configuration_strategy_rows=all_configs,
                         output=str(root), artifact_count=len(artifacts))))


if __name__ == '__main__':
    main()
