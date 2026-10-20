"""Information boundaries, leakage rejection, pollution accounting and ablations."""
import copy
import unittest
import json
import os
import subprocess
import sys
import tempfile
import hashlib
import random
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from m0a.issue6_evaluate import Cache, evaluate, interval
from m0a.issue6_predictors import EventModel, SemanticLSA, Predictor, STRATEGIES, allocate
from m0a.issue6_protocol import context, Observation, validate
from m0a.issue6_workload import split_tasks, validate_dataset, digest


def fixture():
    rows = []
    for domain in ('banking', 'retail'):
        for task in range(6):
            for variant in ('original', 'natural_conversation'):
                for query in range(8):
                    selected = list(range(4)) if query < 4 else list(range(20, 24))
                    position = 32 + query
                    scores = [2.0 if b in selected else 0.1 for b in range(position + 1)]
                    rows.append({'schema_version': 1, 'unit': 'native_kv_token',
                        'source_kind': 'synthetic_fixture', 'task_id': f'{domain}:{task}',
                        'run_id': 'test', 'request_id': f'{task}:{variant}', 'context_version': f'{domain}:{task}:{variant}',
                        'rank': 0, 'layer': 'layer0', 'phase': 'decode', 'query': query,
                        'prediction_step': query * 2, 'completion_step': query * 2 + 1,
                        'position': position, 'eligible_blocks': list(range(position)),
                        'selected_blocks': selected, 'scores': scores,
                        'stage': 'A' if query < 4 else 'B', 'event_type': 'tool_result' if query < 4 else 'task_switch',
                        'query_text': 'bank account money' if query < 4 else 'garden flowers trees',
                        'fragments': [{'start': 0, 'end': 16, 'text': 'bank account money card retail order'},
                                      {'start': 16, 'end': 32, 'text': 'garden flowers trees plants leaves'}],
                        'variant': variant, 'split': ('train', 'validation', 'test')[task % 3]})
    return rows


def models(rows):
    train = [r for r in rows if r['split'] == 'train']
    return EventModel(train), SemanticLSA([f['text'] for r in train for f in r['fragments']])


def metric():
    return dict(prefetch_units=0, useful_prefetch_units=0, wasted_prefetch_units=0, sync_recall_units=0)


class Issue6SuiteTests(unittest.TestCase):
    def test_grouped_variants_and_task_hashes(self):
        task = {'evaluation_criteria': {'actions': [{'name': 'read'}]}, 'user_scenario': {'fact': 'original'}}
        row = {'task_id': 'a', 'group_id': 'a', 'variant': 'original', 'split': 'train',
               'task': task, 'source_task_sha256': digest(task), 'evaluation_criteria_sha256': digest(task['evaluation_criteria']),
               'distractor': ''}
        validate_dataset([row])
        altered = copy.deepcopy(row)
        altered['task']['user_scenario']['fact'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'facts changed'):
            validate_dataset([altered])
        duplicated = dict(row, variant='natural_conversation', split='test')
        with self.assertRaisesRegex(ValueError, 'leakage'):
            validate_dataset([row, duplicated])
        splits = split_tasks([SimpleNamespace(id=str(i)) for i in range(6)])
        self.assertEqual([s for _, s in splits], ['train', 'validation', 'test'] * 2)

    def test_trace_rejects_missing_query_and_unavailable_kv(self):
        rows = fixture()
        validate(rows)
        for change in ('missing', 'candidate', 'split', 'scores'):
            altered = copy.deepcopy(rows)
            if change == 'missing':
                altered.pop(1)
            elif change == 'candidate':
                altered[0]['eligible_blocks'].append(32)
            elif change == 'split':
                altered[8]['split'] = 'test'
            else:
                altered[0]['scores'][10] = 999
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate(altered)

    def test_target_and_future_mutations_cannot_change_current_prediction(self):
        rows = fixture()
        event, semantic = models(rows)
        for strategy in STRATEGIES[:-1]:
            first = Predictor(strategy, semantic=semantic, event=event, size=64)
            second = Predictor(strategy, semantic=semantic, event=event, size=64)
            altered = copy.deepcopy(rows[0])
            altered['selected_blocks'] = [30, 31]  # Observation is absent from Context.
            self.assertEqual(first.predict(context(rows[0]), 4), second.predict(context(altered), 4))
            with self.assertRaisesRegex(ValueError, 'Complete previous'):
                first.predict(context(rows[1]), 4)
        with self.assertRaisesRegex(ValueError, 'training tasks'):
            EventModel([r for r in rows if r['split'] == 'test'])

    def test_stage_history_resets_and_global_history_survives(self):
        rows = fixture()
        event, semantic = models(rows)
        predictions = {}
        for strategy in ('global_frequency', 'stage_frequency'):
            predictor = Predictor(strategy, semantic=semantic, event=event, size=64)
            for row in rows[:5]:
                result, _ = predictor.predict(context(row), 4)
                if row['query'] == 4:
                    predictions[strategy] = result
                predictor.observe(Observation(row['query'], row['completion_step'], tuple(row['selected_blocks'])))
        self.assertTrue(predictions['global_frequency'])
        self.assertFalse(predictions['stage_frequency'])

    def test_vectorized_histories_match_the_merged_task6_contract(self):
        from m0a.history_prefetch import AccessHistoryPredictor, PredictionContext
        rows = fixture()
        event, semantic = models(rows)
        base = context(rows[0])
        selections = [(1, 129), (1, 130), (257,), (257,), (129,)]
        for strategy, scope, signal in (('global_frequency', 'global', 'frequency'),
                ('stage_frequency', 'stage', 'frequency'), ('global_recency', 'global', 'recency'),
                ('global_region', 'global', 'region')):
            reference = AccessHistoryPredictor(scope=scope, signal=signal, region_blocks=128)
            candidate = Predictor(strategy, semantic=semantic, event=event, size=512)
            for query, selected in enumerate(selections):
                ctx = replace(base, query=query, prediction_step=2 * query, position=300 + query,
                              eligible=range(300 + query), stage='A' if query < 4 else 'B', scores=None)
                expected = reference.predict(PredictionContext(query, 2 * query, ctx.stage, tuple(ctx.eligible)), budget=7)
                self.assertEqual(candidate.predict(ctx, 7)[0], [item.block for item in expected], strategy)
                reference.observe(query=query, completion_step=2 * query + 1, selected_blocks=selected)
                candidate.observe(Observation(query, 2 * query + 1, selected))

    def test_lsa_maps_request_fragments_and_current_scores_are_required_by_echo(self):
        rows = fixture()
        event, semantic = models(rows)
        ctx = context(rows[0])
        result = semantic.rank(ctx)
        self.assertTrue(result and set(result[:4]) <= set(range(16)))
        changed = replace(ctx, fragments=(), namespace=ctx.namespace[:3] + ('new-context',) + ctx.namespace[4:])
        self.assertEqual(semantic.rank(changed), [])
        predictor = Predictor('echo_style', semantic=semantic, event=event, size=64)
        with self.assertRaisesRegex(ValueError, 'full current'):
            predictor.predict(replace(ctx, scores=None), 4)

    def test_adaptive_allocations_use_only_completed_feedback(self):
        rows = fixture()
        event, semantic = models(rows)
        for budget in (0, 1, 3, 7):
            self.assertEqual(sum(allocate([0.2, 0.3, 0.4, 0.1], budget)), budget)
        predictor = Predictor('adaptive', semantic=semantic, event=event, size=64)
        result, allocation = predictor.predict(context(rows[0]), 7)
        self.assertEqual(sum(allocation.values()), 7)
        self.assertEqual(len(result), len(set(result)))
        before = dict(predictor.feedback)
        predictor.observe(Observation(0, 1, tuple(rows[0]['selected_blocks'])))
        self.assertNotEqual(before, predictor.feedback)
        ablated = Predictor('adaptive_no_semantic', semantic=semantic, event=event, size=64)
        _, shares = ablated.predict(context(rows[0]), 7)
        self.assertNotIn('semantic', shares)

    def test_cache_charges_pollution_and_never_evades_capacity(self):
        cache = Cache(2)
        initial = metric()
        cache.demand([0, 1], initial)
        bad = metric()
        cache.prefetch([2], bad)  # Evicts native block 0 before demand is revealed.
        cache.demand([0], bad)
        cache.finish()
        self.assertEqual(initial['sync_recall_units'], 2)
        self.assertEqual(bad['prefetch_units'], 1)
        self.assertEqual(bad['sync_recall_units'], 1)
        self.assertEqual(bad['wasted_prefetch_units'], 1)
        self.assertLessEqual(len(cache.resident), 2)
        with self.assertRaisesRegex(ValueError, 'exceeds capacity'):
            cache.demand([0, 1, 2], metric())
        local = metric()
        Cache(2).demand([4], local, locally_generated=4)
        self.assertEqual(local['sync_recall_units'], 0)
        self.assertEqual(local['local_generated_units'], 1)

    def test_bulk_eviction_matches_tokenwise_lru_and_transfer_accounting(self):
        class Reference(Cache):
            def demand(self, selected, record, *, locally_generated=None):
                selected = set(selected)
                for block in sorted(selected):
                    if self._insert(block, selected):
                        if block == locally_generated:
                            record['local_generated_units'] = record.get('local_generated_units', 0) + 1
                        else:
                            record['sync_recall_units'] += 1
                    origin = self.pending.pop(block, None)
                    if origin is not None:
                        origin['useful_prefetch_units'] += 1
                    self.resident.move_to_end(block)
        rng = random.Random(19)
        for capacity in (8, 16, 32):
            candidate, reference = Cache(capacity), Reference(capacity)
            candidate_records, reference_records = [], []
            for query in range(100):
                position = 64 + query
                selected = rng.sample(range(position), 6)
                if query % 3 == 0:
                    selected.append(position)
                prefetched = rng.sample(range(position), 5)
                left, right = metric(), metric()
                candidate_records.append(left)
                reference_records.append(right)
                for cache, record in ((candidate, left), (reference, right)):
                    cache.prefetch(prefetched, record)
                    cache.demand(selected, record, locally_generated=position)
                self.assertEqual(list(candidate.resident), list(reference.resident))
                self.assertEqual(candidate_records, reference_records)
            candidate.finish()
            reference.finish()
            self.assertEqual(candidate_records, reference_records)

    def test_transfer_conservation_and_task_cluster_intervals(self):
        summary, tasks, records = evaluate(fixture(), capacities=(8,), budgets=(2,), unit_bytes=4)
        self.assertEqual(len(summary['test_tasks']), 4)
        self.assertTrue(all(r['cross_stage_boundary_count'] > 0 for r in summary['locality']))
        for record in summary['summary']:
            self.assertEqual(record['metrics']['coverage']['independent_tasks'], 4)
        self.assertTrue(all(r['candidate_count'] <= 2 for r in records))
        self.assertTrue(all(r['candidate_hits'] == 2 for r in records if r['strategy'] == 'oracle_diagnostic'))
        self.assertEqual(sum(r['prefetch_units'] for r in records),
                         sum(r['useful_prefetch_units'] + r['wasted_prefetch_units'] for r in records))
        self.assertIsNone(interval([1])['low'])
        self.assertFalse(summary['online_performance_validated'])
        self.assertTrue(summary['transfer_bytes_are_simulated'])

    @unittest.skipUnless(Path('/proc/self/environ').exists(), 'Linux process ownership check')
    def test_server_cleanup_stops_only_its_run_marker(self):
        from m0a.issue6_serve import stop_server
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / 'receipt.json'
            owned = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                env=dict(os.environ, ISSUE6_SERVER_RECEIPT=str(receipt.resolve())))
            unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
            try:
                receipt.write_text(json.dumps({'pid': owned.pid}))
                stop_server(receipt)
                owned.wait(timeout=5)
                self.assertIsNone(unrelated.poll())
            finally:
                for child in (owned, unrelated):
                    if child.poll() is None:
                        # The assertions above verify selective graceful cleanup.
                        # Always reap our own fixture children even if imported
                        # runtime libraries changed inherited SIGTERM handling.
                        child.kill()
                    child.wait(timeout=5)

    def test_native_cli_requires_matching_audit_and_explicit_instability_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = fixture()
            for row in rows:
                row['source_kind'] = 'native_trace'
            trace = root / 'trace.jsonl'
            trace.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            command = [sys.executable, '-m', 'm0a.issue6_evaluate', '--trace', str(trace)]
            missing = subprocess.run(command + ['--output', str(root / 'missing')], capture_output=True, text=True)
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn('requires independent archive audit', missing.stderr)
            manifest = {'artifacts': {trace.name: hashlib.sha256(trace.read_bytes()).hexdigest()}}
            manifest_path = root / 'manifest.json'
            manifest_path.write_text(json.dumps(manifest))
            audit = {'independent_audit_passed': True, 'manifest_sha256': 'wrong',
                     'decode_rows': len(rows), 'fixed_selected_sets_stable': False}
            audit_path = root / 'audit.json'
            audit_path.write_text(json.dumps(audit))
            rejected = subprocess.run(command + ['--output', str(root / 'mismatch'), '--native-audit', str(audit_path)],
                                      capture_output=True, text=True)
            self.assertIn('differs from audited archive', rejected.stderr)
            audit['manifest_sha256'] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            audit_path.write_text(json.dumps(audit))
            unstable = subprocess.run(command + ['--output', str(root / 'unstable'), '--native-audit', str(audit_path)],
                                      capture_output=True, text=True)
            self.assertIn('Unstable native data', unstable.stderr)


if __name__ == '__main__':
    unittest.main()
