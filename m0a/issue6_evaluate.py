"""Equal-capacity offline cache accounting, task-cluster intervals and plots."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import random
import time
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from .history_prefetch import require
from .issue6_protocol import context, Observation, validate
from .issue6_predictors import EventModel, Predictor, SemanticLSA, STRATEGIES


class Cache:
    """Common LRU with selected KV protected during synchronous guaranteed recall."""
    def __init__(self, capacity):
        require(type(capacity) is int and capacity > 0, 'Invalid cache capacity')
        self.capacity, self.resident, self.pending = capacity, OrderedDict(), {}

    def _insert(self, block, protected):
        if block in self.resident:
            return False
        if len(self.resident) == self.capacity:
            victim = next((b for b in self.resident if b not in protected), None)
            require(victim is not None, 'Selected working set exceeds capacity')
            self.resident.pop(victim)
            origin = self.pending.pop(victim, None)
            if origin is not None:
                origin['wasted_prefetch_units'] += 1
        self.resident[block] = None
        return True

    def prefetch(self, candidates, record):
        for block in candidates:
            if self._insert(block, set()):
                record['prefetch_units'] += 1
                self.pending[block] = record

    def demand(self, selected, record):
        selected = set(selected)
        require(len(selected) <= self.capacity, 'Selected working set exceeds capacity')
        for block in sorted(selected):
            if self._insert(block, selected):
                record['sync_recall_units'] += 1
            origin = self.pending.pop(block, None)
            if origin is not None:
                origin['useful_prefetch_units'] += 1
            self.resident.move_to_end(block)
        require(selected <= self.resident.keys() and len(self.resident) <= self.capacity,
                'Guaranteed recall/residency invariant failed')

    def finish(self):
        for origin in self.pending.values():
            origin['wasted_prefetch_units'] += 1
        self.pending.clear()


def interval(values):
    require(values, 'No independent task measurements')
    mean = sum(values) / len(values)
    if len(values) < 2:
        return {'mean': mean, 'low': None, 'high': None, 'independent_tasks': len(values)}
    rng = random.Random(7)
    samples = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(1000))
    return {'mean': mean, 'low': samples[24], 'high': samples[974], 'independent_tasks': len(values)}


def summarize(records, unit_bytes):
    groups = defaultdict(list)
    for row in records:
        for variant in ('all_variants', row['variant']):
            groups[(row['strategy'], row['capacity'], row['budget'], variant)].append(row)
    result, per_task = [], []
    for (strategy, capacity, budget, variant), items in sorted(groups.items()):
        tasks = defaultdict(list)
        for row in items:
            tasks[row['task_id']].append(row)
        measurements = []
        for task_id, task_rows in sorted(tasks.items()):
            selected = sum(r['selected_count'] for r in task_rows)
            candidates = sum(r['candidate_count'] for r in task_rows)
            hits = sum(r['candidate_hits'] for r in task_rows)
            prefetch = sum(r['prefetch_units'] for r in task_rows)
            useful = sum(r['useful_prefetch_units'] for r in task_rows)
            waste = sum(r['wasted_prefetch_units'] for r in task_rows)
            sync = sum(r['sync_recall_units'] for r in task_rows)
            require(prefetch == useful + waste, 'Unaccounted speculative transfer')
            data = {'task_id': task_id, 'coverage': hits / selected,
                    'candidate_precision': hits / candidates if candidates else 0,
                    'sync_recall_MiB': sync * unit_bytes / 2**20,
                    'wasted_prefetch_MiB': waste * unit_bytes / 2**20,
                    'total_transfer_MiB': (prefetch + sync) * unit_bytes / 2**20,
                    'prediction_us_per_query': sum(r['prediction_ns'] for r in task_rows) / len(task_rows) / 1000}
            measurements.append(data)
            per_task.append(dict(data, strategy=strategy, capacity=capacity, budget=budget, variant=variant,
                                 queries=len(task_rows)))
        result.append({'strategy': strategy, 'capacity': capacity, 'budget': budget, 'variant': variant,
                       'metrics': {key: interval([r[key] for r in measurements]) for key in measurements[0] if key != 'task_id'}})
    return result, per_task


def locality(lanes):
    groups = defaultdict(lambda: {'adjacent': [], 'stage_boundary': [], 'positions': []})
    for lane in lanes.values():
        if lane[0]['split'] != 'test':
            continue
        group = groups[(lane[0]['task_id'], lane[0]['variant'])]
        previous, stage = None, None
        for row in lane:
            selected = set(row['selected_blocks'])
            group['positions'].append(row['position'])
            if previous is not None:
                jaccard = len(previous & selected) / len(previous | selected)
                group['adjacent'].append(jaccard)
                if row['stage'] != stage:
                    group['stage_boundary'].append(jaccard)
            previous, stage = selected, row['stage']
    return [{'task_id': task, 'variant': variant,
             'mean_adjacent_query_jaccard': sum(g['adjacent']) / len(g['adjacent']) if g['adjacent'] else None,
             'cross_stage_boundary_count': len(g['stage_boundary']),
             'mean_cross_stage_jaccard': sum(g['stage_boundary']) / len(g['stage_boundary']) if g['stage_boundary'] else None,
             'context_position_range': [min(g['positions']), max(g['positions'])]}
            for (task, variant), g in sorted(groups.items())]


def evaluate(rows, *, capacities=(4096, 8192), budgets=(128, 512), unit_bytes=1152,
             strategies=STRATEGIES):
    started = time.perf_counter_ns()
    lanes = validate(rows)
    training = [r for r in rows if r['split'] == 'train']
    test = [r for r in rows if r['split'] == 'test']
    require(training and test, 'Training and held-out test tasks required')
    require(all(type(c) is int and c > 0 for c in capacities) and
            all(type(b) is int and 0 <= b <= min(capacities) for b in budgets), 'Invalid capacity/budget')
    event = EventModel(training)
    semantic = SemanticLSA([f['text'] for row in training for f in row['fragments']])
    # Prepare identical input-only semantic features once for every strategy.
    # Context keys prevent later fragments/text from leaking to an earlier query.
    for row in test:
        semantic.rank(context(row))
    records = []
    for capacity in capacities:
        for budget in budgets:
            for strategy in strategies:
                require(strategy in STRATEGIES, 'Unknown strategy')
                for namespace, lane in sorted(lanes.items()):
                    if lane[0]['split'] != 'test':
                        continue
                    require(lane[0]['task_id'] not in event.tasks, 'Training/test leakage')
                    cache = Cache(capacity)
                    predictor = None if strategy == 'oracle_diagnostic' else Predictor(strategy,
                        semantic=semantic, event=event, size=max(r['position'] for r in lane) + 1)
                    for row in lane:
                        ctx = context(row)
                        tick = time.perf_counter_ns()
                        if predictor is None:
                            eligible_selected = set(row['selected_blocks']).intersection(ctx.eligible)
                            missing = eligible_selected - cache.resident.keys()
                            candidates = (sorted(missing) + sorted(eligible_selected - missing))[:budget]
                            allocation = {}
                        else:
                            candidates, allocation = predictor.predict(ctx, budget)
                        prediction_ns = time.perf_counter_ns() - tick
                        record = {'task_id': row['task_id'], 'variant': row['variant'],
                            'request_id': row['request_id'], 'layer': row['layer'], 'query': ctx.query,
                            'run_id': row['run_id'], 'context_version': row['context_version'], 'rank': row['rank'],
                            'phase': ctx.phase, 'candidates_in_priority_order': candidates,
                            'strategy': strategy, 'capacity': capacity, 'budget': budget,
                            'stage': ctx.stage, 'event_type': ctx.event_type,
                            'candidate_count': len(candidates), 'selected_count': len(row['selected_blocks']),
                            'candidate_hits': len(set(candidates).intersection(row['selected_blocks'])),
                            'prefetch_units': 0, 'sync_recall_units': 0, 'useful_prefetch_units': 0,
                            'wasted_prefetch_units': 0, 'prediction_ns': prediction_ns,
                            'allocation': allocation}
                        cache.prefetch(candidates, record)
                        cache.demand(row['selected_blocks'], record)
                        if predictor is not None:
                            predictor.observe(Observation(ctx.query, row['completion_step'], tuple(row['selected_blocks'])))
                        records.append(record)
                    cache.finish()
    summary, per_task = summarize(records, unit_bytes)
    return {'schema_version': 1, 'source_kind': rows[0]['source_kind'],
            'source_scopes': sorted({r.get('source_scope', 'synthetic_fixture') for r in rows}),
            'autonomous_agent_validated': False,
            'evaluation': 'offline_equal_capacity_LRU_simulation', 'cache_initialization': 'empty_per_request_layer_decode_window',
            'unit': 'native_kv_token', 'unit_bytes': unit_bytes, 'phase': 'decode',
            'capacities': capacities, 'budgets': budgets, 'strategies': strategies,
            'training_tasks': event.tasks, 'validation_tasks': sorted({r['task_id'] for r in rows if r['split'] == 'validation'}),
            'test_tasks': sorted({r['task_id'] for r in test}),
            'locality': locality(lanes),
            'confidence_interval': 'seed-7 1000-resample percentile bootstrap over independent task clusters; all variants clustered',
            'overheads_ns': {'semantic_fit': semantic.fit_ns, 'semantic_index_build': semantic.index_build_ns,
                             'semantic_encoding': semantic.encoding_ns, 'semantic_retrieval': semantic.retrieval_ns,
                             'evaluation_wall': time.perf_counter_ns() - started},
            'overhead_scope': 'semantic_index_build includes encoding; feature costs reported separately, not summed twice; prediction timing excludes shared feature preparation',
            'code_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                            for name in ('issue6_evaluate.py', 'issue6_predictors.py', 'issue6_protocol.py')},
            'python': sys.version, 'numpy': __import__('numpy').__version__,
            'fitted_models': {'training_tasks': event.tasks,
                'event': [{'key': list(key), 'region_counts': counts.tolist()} for key, counts in sorted(event.counts.items())],
                'event_unconditioned': [{'key': list(key), 'region_counts': counts.tolist()} for key, counts in sorted(event.global_counts.items())],
                'lsa': {'vocabulary': semantic.vocabulary, 'idf': semantic.idf.tolist(),
                        'projection': semantic.projection.tolist()}},
            'echo_scope': 'decode EMA threshold on verified full current scores; no score-stream timing or prefill claim',
            'online_performance_validated': False, 'transfer_bytes_are_simulated': True,
            'summary': summary}, per_task, records


def plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    capacity, budget = summary['capacities'][0], summary['budgets'][-1]
    rows = [r for r in summary['summary'] if r['variant'] == 'all_variants' and
            r['capacity'] == capacity and r['budget'] == budget]
    fig, axes = plt.subplots(3, 2, figsize=(18, 17), layout='constrained')
    for axis, metric in zip(axes.flat, ('candidate_precision', 'coverage', 'sync_recall_MiB',
                                     'wasted_prefetch_MiB', 'total_transfer_MiB', 'prediction_us_per_query')):
        values = [r['metrics'][metric]['mean'] for r in rows]
        errors = [[max(0, v - r['metrics'][metric]['low']) if r['metrics'][metric]['low'] is not None else 0
                   for r, v in zip(rows, values)],
                  [max(0, r['metrics'][metric]['high'] - v) if r['metrics'][metric]['high'] is not None else 0
                   for r, v in zip(rows, values)]]
        axis.bar(range(len(rows)), values, yerr=errors, capsize=2)
        axis.set_xticks(range(len(rows)), [r['strategy'] for r in rows], rotation=70, ha='right', fontsize=8)
        axis.set_title(metric)
    fig.suptitle(f'Offline controlled replay: C={capacity} tokens/layer, P={budget}; task-cluster 95% CI\nNot an autonomous Agent benchmark or online performance result')
    fig.savefig(output / 'comparison.png', dpi=150)
    fig.savefig(output / 'comparison.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--capacity', type=int, action='append')
    parser.add_argument('--budget', type=int, action='append')
    parser.add_argument('--unit-bytes', type=int, default=1152)
    parser.add_argument('--native-audit', type=Path)
    parser.add_argument('--include-unstable', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    opener = gzip.open if args.trace.suffix == '.gz' else open
    with opener(args.trace, 'rt', encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream]
    if rows and rows[0]['source_kind'] == 'native_trace':
        require(args.native_audit is not None, 'Native evaluation requires independent archive audit')
        audit = json.loads(args.native_audit.read_text())
        manifest_path = args.trace.parent / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        require(audit['independent_audit_passed'] and
                audit['manifest_sha256'] == hashlib.sha256(manifest_path.read_bytes()).hexdigest() and
                manifest['artifacts'][args.trace.name] == hashlib.sha256(args.trace.read_bytes()).hexdigest() and
                audit['decode_rows'] == len(rows), 'Native input differs from audited archive')
        require(audit['fixed_selected_sets_stable'] or args.include_unstable,
                'Unstable native data: explicitly opt in without dropping samples')
    summary, per_task, records = evaluate(rows, capacities=args.capacity or (4096, 8192),
        budgets=args.budget or (128, 512), unit_bytes=args.unit_bytes)
    summary['input_sha256'] = hashlib.sha256(args.trace.read_bytes()).hexdigest()
    summary['native_audit'] = json.loads(args.native_audit.read_text()) if args.native_audit else None
    model_path = args.output / 'fitted-models.json'
    model_path.write_text(json.dumps(summary.pop('fitted_models'), indent=2) + '\n')
    summary['fitted_models_sha256'] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (args.output / 'per-task.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_task[0]))
        writer.writeheader()
        writer.writerows(per_task)
    with (args.output / 'predictions.jsonl').open('w') as stream:
        for record in records:
            stream.write(json.dumps(record) + '\n')
    plot(summary, args.output)
    print(json.dumps({'test_tasks': summary['test_tasks'], 'predictions': len(records), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
