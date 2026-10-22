"""CPU discrete-event KV prefetch simulator. Times are assumptions, never NPU measurements.

One lane is one request/rank/layer. Concurrent lanes share one FIFO DMA link;
they are independent streams, NOT a tensor-parallel or whole-model schedule.
Each query drains its speculative batch before demand recall (conservative policy).
Destination slots are reserved at submission and count against capacity.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import random
from collections import defaultdict

import simpy

from .history_prefetch import require
from .issue6_evaluate import Cache, interval
from .issue6_protocol import context, Observation, validate
from .issue6_predictors import EventModel, Predictor, SemanticLSA, STRATEGIES


@dataclass(frozen=True)
class Timing:
    bandwidth_GBs: float = 16.0  # decimal GB/s, hence 1000 bytes/us
    launch_us: float = 5.0
    lead_us: float = 50.0  # native selection ready after query start
    prediction_us: float = 5.0  # all predictor-side feature/index costs amortized here
    signal_delay_us: float = 0.0
    attention_us: float = 20.0
    batch_units: int = 128  # launch cost per chunk, not per KV element

    def check(self):
        require(math.isfinite(self.bandwidth_GBs) and self.bandwidth_GBs > 0,
                'Bandwidth must be positive and finite')
        require(all(math.isfinite(v) and v >= 0 for k, v in asdict(self).items()
                    if k not in {'bandwidth_GBs', 'batch_units'}), 'Invalid timing')
        require(type(self.batch_units) is int and self.batch_units > 0, 'Invalid batch size')

    def transfer_us(self, count, unit_bytes):
        return (math.ceil(count / self.batch_units) * self.launch_us +
                count * unit_bytes / (self.bandwidth_GBs * 1000)) if count else 0.0


def simulate(lanes, candidates, *, capacity, budget, unit_bytes, timing=Timing(), strategy='none'):
    """Candidate sequences must have been generated causally outside this function.

An empty-cache start is used for each lane. CPU predictors are independent workers;
only the DMA is shared. Attention cannot read a reserved slot until DMA completes.
"""
    timing.check()
    require(type(capacity) is int and capacity > 0 and 0 <= budget <= capacity, 'Invalid capacity/budget')
    require(type(unit_bytes) is int and unit_bytes > 0, 'Invalid unit size')
    env = simpy.Environment()
    link = simpy.Resource(env, capacity=1)
    records, transfers, final = [], [], []

    def dma(count, lane_id, query, kind):
        if not count:
            return
        queued = env.now
        with link.request() as ticket:
            yield ticket
            start = env.now
            yield env.timeout(timing.transfer_us(count, unit_bytes))
            transfers.append(dict(lane=lane_id, query=query, kind=kind, queued_us=queued,
                                  start_us=start, end_us=env.now, units=count,
                                  bytes=count * unit_bytes))

    def worker(lane_id, lane, choices):
        cache = Cache(capacity)
        previous_stage, stage_age = None, 0
        for row, selected_candidates in zip(lane, choices):
            stage_age = stage_age + 1 if previous_stage == row['stage'] else 0
            previous_stage = row['stage']
            selected = row['selected_blocks']
            require(len(set(selected)) <= capacity, 'Selected working set exceeds capacity')
            eligible = context(row).eligible
            require(len(selected_candidates) <= budget and
                    len(selected_candidates) == len(set(selected_candidates)) and
                    set(selected_candidates) <= set(eligible), 'Invalid candidate universe/budget')
            begin = env.now
            rec = dict(task_id=row['task_id'], variant=row['variant'], lane=lane_id,
                       query=row['query'], stage=row['stage'], stage_age=stage_age, event_type=row['event_type'],
                       strategy=strategy, candidate_count=len(selected_candidates),
                       candidate_hits=len(set(selected_candidates).intersection(selected)),
                       selected_count=len(selected), prefetch_units=0, sync_recall_units=0,
                       useful_prefetch_units=0, wasted_prefetch_units=0, local_generated_units=0,
                       begin_us=begin)

            def prefetch():
                if strategy != 'none' and budget:
                    # ECHO needs the current score vector; its earliest availability is
                    # conservatively the native selection-ready time in this model.
                    delay = max(timing.signal_delay_us, timing.lead_us if strategy == 'echo_style' else 0)
                    yield env.timeout(delay + timing.prediction_us)
                rec['prediction_ready_us'] = env.now
                cache.prefetch(selected_candidates, rec)  # reserves destination slots
                rec['prefetch_reserved_units'] = len(cache.resident)
                yield env.process(dma(rec['prefetch_units'], lane_id, row['query'], 'prefetch'))
                rec['prefetch_done_us'] = env.now

            future = env.process(prefetch())
            yield env.timeout(timing.lead_us)
            rec['native_ready_us'] = env.now
            yield future  # conservative drain: includes useless/non-preemptible DMA
            local = row.get('locally_generated_ids', [])
            if row['schema_version'] == 2:
                cache.generate(local, rec)
            cache.demand(selected, rec, locally_generated=row['position'] if row['schema_version'] == 1 else None)
            yield env.process(dma(rec['sync_recall_units'], lane_id, row['query'], 'demand'))
            rec['attention_start_us'] = env.now
            rec['wait_us'] = env.now - rec['native_ready_us']
            rec['resident_units'] = len(cache.resident)
            require(set(selected) <= cache.resident.keys() and len(cache.resident) <= capacity,
                    'Attention readiness/capacity invariant failed')
            yield env.timeout(timing.attention_us)
            rec['end_us'] = env.now
            rec['service_us'] = env.now - begin
            records.append(rec)
        cache.finish()
        final.append(dict(lane=lane_id, completion_us=env.now))

    require(len(lanes) == len(candidates) and lanes, 'Missing lanes')
    for i, (lane, choices) in enumerate(zip(lanes, candidates)):
        require(lane and len(lane) == len(choices), 'Missing candidate queries')
        env.process(worker(i, lane, choices))
    env.run()
    require(all(r['prefetch_units'] == r['useful_prefetch_units'] + r['wasted_prefetch_units']
                for r in records), 'Transfer conservation failed')
    require(all(a['end_us'] <= b['start_us'] + 1e-9 for a, b in zip(transfers, transfers[1:])),
            'Serialized DMA link overlaps itself')
    return records, transfers, final


def predict(lane, strategy, budget, semantic, event):
    if strategy == 'oracle_diagnostic':
        # Perfect current labels, but no future eviction knowledge or cache adaptation.
        # This diagnoses selection predictability, not a global latency upper bound.
        return [sorted(set(r['selected_blocks']).intersection(context(r).eligible))[:budget] for r in lane]
    p = Predictor(strategy, semantic=semantic, event=event, size=max(r['position'] for r in lane) + 1)
    result = []
    for row in lane:
        result.append(p.predict(context(row), budget)[0])
        p.observe(Observation(row['query'], row['completion_step'], tuple(row['selected_blocks'])))
    return result


def synthetic(*, seed=7, tasks=12, queries=32, units=4096, topk=512, scenario='stable'):
    """C4-shaped stress traces; no model inference, semantics or genuine Agent events.

    Stable/shift: 80% selected from a hot quarter-window; random: independent draws.
    Event/semantic labels intentionally encode the region in stable/shift and are
    adversarial in misleading. These test mechanisms, never semantic generalization.
    """
    require(tasks >= 3 and queries > 0 and units >= 4 * topk and topk > 0, 'Invalid generator size')
    require(scenario in {'stable', 'shift', 'random', 'misleading'}, 'Unknown scenario')
    rows = []
    labels = ['bank account statement', 'retail order return', 'tool inventory lookup', 'memory customer record']
    for task in range(tasks):
        rng = random.Random(seed * 100003 + task)
        split = ('train', 'validation', 'test')[task % 3]
        initial = rng.randrange(4)
        for q in range(queries):
            p = units * 4 + q
            available = (p + 1) // 4
            stage = q // max(1, queries // 4)
            region = (initial + stage) % 4 if scenario in {'shift', 'misleading'} else initial
            hot = list(range(region * units // 4, (region + 1) * units // 4))
            if scenario == 'random':
                selected = rng.sample(range(available), topk)
            else:
                selected = rng.sample(hot, int(topk * .8))
                remainder = sorted(set(range(available)) - set(selected))
                selected += rng.sample(remainder, topk - len(selected))
            cue = (region + 1) % 4 if scenario == 'misleading' else region
            rows.append(dict(schema_version=2, unit='compressed_kv_token_position', unit_bytes=1024,
                compression_ratio=4, token_position=p, position=available - 1, eligible_end=p // 4,
                locally_generated_ids=list(range(p // 4, available)), scores=None,
                attention_scope='deepseek_v4_C4_indexed_KV_only', source_kind='synthetic_fixture',
                source_scope='synthetic_C4_stress_not_model_inference', task_id=f'seed{seed}-task{task}',
                split=split, variant=scenario, run_id=f'synthetic-{seed}', request_id=f'req{task}',
                context_version=f'synthetic-{seed}-{task}', rank=0, layer=2, phase='decode',
                query=q, prediction_step=2*q, completion_step=2*q+1,
                stage=f'stage{stage}' if scenario != 'stable' else 'stage0', event_type=f'artificial_region{cue}',
                query_text=labels[cue], fragments=[dict(start=i*units//4, end=(i+1)*units//4,
                    text=label) for i, label in enumerate(labels)], selected_blocks=sorted(selected)))
    validate(rows)
    return rows


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_trace(path, audit_path=None):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]
    validate(rows)
    receipt = dict(trace_sha256=sha(path), source_kind=rows[0]['source_kind'])
    if rows[0]['source_kind'] == 'native_trace':
        require(audit_path is not None, 'Native trace requires audit receipt')
        audit = json.loads(audit_path.read_text())
        manifest_path = path.parent / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        require(audit['independent_audit_passed'] and audit['fixed_selected_sets_stable'] and
                audit['manifest_sha256'] == sha(manifest_path) and
                manifest['artifacts'][path.name] == sha(path) and audit['decode_rows'] == len(rows),
                'Trace differs from stable audited archive')
        receipt.update(native_audit=audit, manifest_sha256=sha(manifest_path),
                       model=manifest.get('model'), model_revision=manifest.get('model_revision'))
    return rows, receipt


def run_grid(rows, output, *, capacities, budgets, timings, strategies, concurrency=1):
    lanes = validate(rows)
    training = [r for r in rows if r['split'] == 'train']
    require(training and any(r['split'] == 'test' for r in rows), 'Train/test required')
    require(type(concurrency) is int and concurrency > 0, 'Invalid concurrency')
    require(strategies and all(s in STRATEGIES for s in strategies), 'Unknown strategies')
    require('echo_style' not in strategies or all(r.get('scores') is not None for r in rows),
            'ECHO requires actual full scores; selected-only traces cannot supply them')
    event = EventModel(training)
    semantic = SemanticLSA([f['text'] for r in training for f in r['fragments']])
    test_lanes = [v for _, v in sorted(lanes.items()) if v[0]['split'] == 'test']
    unit_bytes = rows[0].get('unit_bytes', 1152)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'fitted-models.json').write_text(json.dumps(dict(training_tasks=event.tasks,
        vocabulary=semantic.vocabulary, idf=semantic.idf.tolist(), projection=semantic.projection.tolist(),
        event_counts=[dict(key=list(k), counts=v.tolist()) for k, v in sorted(event.counts.items())]), indent=2))
    summaries, detail = [], []
    for budget in budgets:
        choices = {(s, i): predict(lane, s, budget, semantic, event)
                   for s in strategies for i, lane in enumerate(test_lanes)}
        for capacity in capacities:
            for ti, timing in enumerate(timings):
                for strategy in strategies:
                    grouped = defaultdict(list)
                    # Each source lane is replayed with N identical independent replicas.
                    # Replicas are a contention stress condition, never extra task samples.
                    for i, lane in enumerate(test_lanes):
                        records, transfers, final = simulate([lane] * concurrency,
                            [choices[strategy, i]] * concurrency, capacity=capacity, budget=budget,
                            unit_bytes=unit_bytes, timing=timing, strategy=strategy)
                        grouped[lane[0]['task_id']].extend(records)
                        if i == 0 and capacity == capacities[0] and budget == budgets[0] and ti == 0:
                            with (output / f'events-{strategy}.json').open('w') as f:
                                json.dump(dict(records=records, transfers=transfers, lanes=final), f)
                    per_task = []
                    for task, records in sorted(grouped.items()):
                        n = len(records)
                        total = lambda key: sum(r[key] for r in records)
                        item = dict(task_id=task, strategy=strategy, capacity=capacity, budget=budget,
                            timing_id=ti, concurrency=concurrency, **asdict(timing), queries=n,
                            service_us=total('service_us') / n, wait_us=total('wait_us') / n,
                            coverage=total('candidate_hits') / total('selected_count'),
                            precision=total('candidate_hits') / max(1, total('candidate_count')),
                            prefetch_bytes_per_query=total('prefetch_units') * unit_bytes / n,
                            sync_bytes_per_query=total('sync_recall_units') * unit_bytes / n,
                            total_bytes_per_query=(total('prefetch_units')+total('sync_recall_units')) * unit_bytes / n,
                            wasted_bytes_per_query=total('wasted_prefetch_units') * unit_bytes / n,
                            boundary_wait_us=sum(r['wait_us'] for r in records if r['stage_age'] < 4) /
                                max(1, sum(r['stage_age'] < 4 for r in records)),
                            steady_wait_us=sum(r['wait_us'] for r in records if r['stage_age'] >= 4) /
                                max(1, sum(r['stage_age'] >= 4 for r in records)),
                            peak_reserved_bytes=max(max(r['resident_units'], r['prefetch_reserved_units']) for r in records)*unit_bytes)
                        per_task.append(item)
                        detail.append(item)
                    metrics = ('service_us', 'wait_us', 'coverage', 'precision', 'prefetch_bytes_per_query',
                               'sync_bytes_per_query', 'total_bytes_per_query', 'wasted_bytes_per_query', 'peak_reserved_bytes',
                               'boundary_wait_us', 'steady_wait_us')
                    summaries.append(dict(strategy=strategy, capacity=capacity, budget=budget, timing_id=ti,
                        concurrency=concurrency, timing=asdict(timing),
                        metrics={k: interval([r[k] for r in per_task]) for k in metrics}))
    baseline = {(r['task_id'], r['capacity'], r['budget'], r['timing_id']): r for r in detail if r['strategy'] == 'none'}
    for s in summaries:
        values = []
        for r in detail:
            if (r['strategy'], r['capacity'], r['budget'], r['timing_id']) == (s['strategy'], s['capacity'], s['budget'], s['timing_id']):
                b = baseline.get((r['task_id'], r['capacity'], r['budget'], r['timing_id']))
                if b:
                    values.append((b['service_us'] - r['service_us']) / b['service_us'])
        if values:
            s['metrics']['paired_service_reduction'] = interval(values)
    with (output / 'per-task.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(detail[0]))
        writer.writeheader()
        writer.writerows(detail)
    report = dict(evaluation='SimPy_discrete_event_KV_prefetch', source_kind=rows[0]['source_kind'],
        source_scopes=sorted({r.get('source_scope', 'unspecified') for r in rows}),
        unit=rows[0]['unit'], unit_bytes=unit_bytes, training_tasks=event.tasks,
        test_tasks=sorted({r[0]['task_id'] for r in test_lanes}), independent_tasks=len({r[0]['task_id'] for r in test_lanes}),
        timing_parameters='assumed_sensitivity_parameters_not_hardware_measurements',
        performance_scope='per_request_rank_layer_decode_query_not_TPOT_or_TTFT',
        dma_policy='FIFO_nonpreemptible_batch_drain_before_demand',
        cache_policy='empty_start_LRU_destination_reservation_counts_against_capacity',
        concurrency_scope='identical_independent_lane_replicas_share_DMA_only_not_TP_ranks',
        oracle_scope='current_selection_only_not_optimal_eviction_or_global_latency_bound',
        uncertainty='seed7_1000_task_cluster_bootstrap_conditional_on_fixed_timing_and_trace',
        autonomous_agent_validated=False, online_performance_validated=False,
        numerical_attention_validated=False, simulated_KV_memory_only=True,
        python=platform.python_version(), simpy=simpy.__version__,
        code_sha256={n: sha(Path(__file__).with_name(n)) for n in
                     ('issue6_simulator.py', 'issue6_evaluate.py', 'issue6_predictors.py', 'issue6_protocol.py', 'history_prefetch.py')},
        fitted_models_sha256=sha(output / 'fitted-models.json'), summary=summaries)
    (output / 'summary.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path)
    parser.add_argument('--native-audit', type=Path)
    parser.add_argument('--scenario', choices=['stable', 'shift', 'random', 'misleading'], default='stable')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--units', type=int, default=4096)
    parser.add_argument('--queries', type=int, default=32)
    parser.add_argument('--tasks', type=int, default=12)
    parser.add_argument('--capacity', action='append', type=int)
    parser.add_argument('--budget', action='append', type=int)
    parser.add_argument('--strategy', choices=STRATEGIES, action='append')
    parser.add_argument('--bandwidth-GBs', action='append', type=float)
    parser.add_argument('--lead-us', action='append', type=float)
    parser.add_argument('--prediction-us', action='append', type=float)
    parser.add_argument('--launch-us', type=float, default=5)
    parser.add_argument('--attention-us', type=float, default=20)
    parser.add_argument('--signal-delay-us', type=float, default=0)
    parser.add_argument('--batch-units', type=int, default=128)
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.trace:
        rows, receipt = load_trace(args.trace, args.native_audit)
    else:
        rows = synthetic(seed=args.seed, units=args.units, queries=args.queries, tasks=args.tasks, scenario=args.scenario)
        receipt = dict(source_kind='synthetic_fixture', generator=dict(seed=args.seed, units=args.units,
            queries=args.queries, tasks=args.tasks, scenario=args.scenario, topk=512))
    timings = [Timing(bandwidth_GBs=b, lead_us=l, prediction_us=p, launch_us=args.launch_us,
                      attention_us=args.attention_us, signal_delay_us=args.signal_delay_us, batch_units=args.batch_units)
               for b in (args.bandwidth_GBs or [16]) for l in (args.lead_us or [50]) for p in (args.prediction_us or [5])]
    report = run_grid(rows, args.output, capacities=args.capacity or [512, 1024], budgets=args.budget or [128, 512],
        timings=timings, strategies=args.strategy or ['none', 'last_selection', 'global_frequency', 'stage_frequency',
        'semantic_lsa', 'event_history', 'adaptive', 'adaptive_no_semantic', 'adaptive_no_event', 'oracle_diagnostic'],
        concurrency=args.concurrency)
    if not args.trace:
        with gzip.open(args.output / 'synthetic-trace.jsonl.gz', 'wt', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row) + '\n')
        receipt['trace_sha256'] = sha(args.output / 'synthetic-trace.jsonl.gz')
    (args.output / 'input-receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(dict(output=str(args.output), configurations=len(report['summary']), test_tasks=report['independent_tasks'])))


if __name__ == '__main__':
    main()
