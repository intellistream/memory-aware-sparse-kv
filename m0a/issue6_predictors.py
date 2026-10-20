"""Causal semantic-LSA, learned event-region, history and feedback budget predictors."""
from __future__ import annotations
import re
import time
from collections import Counter
import numpy as np
from .history_prefetch import require

STRATEGIES = ('none', 'last_selection', 'echo_style', 'global_frequency', 'stage_frequency',
              'global_recency', 'global_region', 'semantic_lsa', 'event_history', 'event_only', 'unconditioned_history',
              'sequential', 'adaptive', 'adaptive_no_semantic', 'adaptive_no_event', 'adaptive_fixed',
              'oracle_diagnostic')


def words(text):
    return re.findall(r'[a-z]+|[\u4e00-\u9fff]', text.lower())


class SemanticLSA:
    """Distributional semantic projection fitted only on training-task text.

    This is a small CPU LSA baseline, not a pretrained neural text encoder.
    Each request index maps exact token spans, never IDs from another context.
    """
    def __init__(self, texts):
        started = time.perf_counter_ns()
        documents = sorted(set(texts))[:512]
        require(documents, 'No training fragment text')
        counts = Counter(w for doc in documents for w in set(words(doc)))
        self.vocabulary = {w: i for i, (w, _) in enumerate(sorted(counts.items(), key=lambda p: (-p[1], p[0]))[:1024])}
        require(self.vocabulary, 'No semantic vocabulary')
        self.idf = np.array([np.log((1 + len(documents)) / (1 + counts[w])) + 1 for w in self.vocabulary])
        matrix = np.vstack([self.vector(text) for text in documents])
        _, _, vt = np.linalg.svd(matrix, full_matrices=False)
        self.projection = vt[:min(16, len(vt))].T
        self.cache = {}
        self.fit_ns = time.perf_counter_ns() - started
        self.index_build_ns = self.encoding_ns = self.retrieval_ns = 0

    def vector(self, text):
        result = np.zeros(len(self.vocabulary))
        counts = Counter(words(text))
        for word, count in counts.items():
            if word in self.vocabulary:
                result[self.vocabulary[word]] = (1 + np.log(count)) * self.idf[self.vocabulary[word]]
        norm = np.linalg.norm(result)
        return result / norm if norm else result

    def encode(self, text):
        started = time.perf_counter_ns()
        result = self.vector(text) @ self.projection
        norm = np.linalg.norm(result)
        self.encoding_ns += time.perf_counter_ns() - started
        return result / norm if norm else result

    def rank(self, ctx):
        # Shared across layers; fragment identity includes exact context version.
        key = (ctx.namespace[3], ctx.query_text, ctx.fragments)
        if key not in self.cache:
            started = time.perf_counter_ns()
            matrix = np.vstack([self.encode(f.text) for f in ctx.fragments]) if ctx.fragments else np.empty((0, self.projection.shape[1]))
            query = self.encode(ctx.query_text)
            self.index_build_ns += time.perf_counter_ns() - started
            started = time.perf_counter_ns()
            similarities = matrix @ query
            order = np.argsort(-similarities, kind='stable')
            ranked, seen = [], set()
            for index in order:
                if similarities[index] <= 0:
                    continue
                fragment = ctx.fragments[index]
                for block in range(fragment.start, fragment.end):
                    if block not in seen:
                        seen.add(block)
                        ranked.append(block)
            self.cache[key] = ranked
            self.retrieval_ns += time.perf_counter_ns() - started
        return self.cache[key]


class EventModel:
    def __init__(self, rows):
        require(rows and all(r['split'] == 'train' for r in rows), 'Event fitting accepts training tasks only')
        self.tasks = sorted({r['task_id'] for r in rows})
        self.counts, self.global_counts = {}, {}
        for row in rows:
            key = (row['layer'], row['phase'], row['event_type'], row['query'] < 4)
            global_key = (row['layer'], row['phase'])
            counts = self.counts.setdefault(key, np.ones(16))
            baseline = self.global_counts.setdefault(global_key, np.ones(16))
            regions = np.minimum(15, np.array(row['selected_blocks']) * 16 // (row['position'] + 1))
            values = np.bincount(regions, minlength=16)
            counts += values
            baseline += values

    def prior(self, ctx, *, conditioned=True):
        key = (ctx.namespace[5], ctx.phase, ctx.event_type, ctx.query < 4)
        values = self.counts.get(key) if conditioned else None
        if values is None:
            values = self.global_counts.get((ctx.namespace[5], ctx.phase), np.ones(16))
        return values / values.sum()


def allocate(weights, budget):
    require(budget >= 0 and weights and all(w > 0 for w in weights), 'Invalid adaptive allocation')
    shares = np.asarray(weights) / sum(weights) * budget
    result = np.floor(shares).astype(int)
    order = np.argsort(-(shares - result), kind='stable')
    for index in order[:budget - sum(result)]:
        result[index] += 1
    return result.tolist()


class Predictor:
    def __init__(self, strategy, *, semantic, event, size):
        require(strategy in STRATEGIES and strategy != 'oracle_diagnostic', 'Oracle is evaluator-only')
        self.strategy, self.semantic, self.event = strategy, semantic, event
        self.count = np.zeros(size, dtype=np.int64)
        self.last = np.full(size, -1, dtype=np.int64)
        self.stage_count = np.zeros(size, dtype=np.int64)
        self.region = np.zeros((size + 127) // 128, dtype=np.int64)
        self.region_last = np.full((size + 127) // 128, -1, dtype=np.int64)
        self.previous = ()
        self.threshold, self.stage, self.pending = None, None, None
        self.feedback = {arm: 0.5 for arm in ('history', 'semantic', 'event', 'sequential')}
        self.last_completion = -1
        self.last_query, self.namespace = -1, None

    def predict(self, ctx, budget):
        require(self.pending is None and ctx.prediction_step > self.last_completion, 'Complete previous query before predicting')
        require(ctx.query == self.last_query + 1 and (self.namespace is None or self.namespace == ctx.namespace),
                'Missing query or mixed KV namespace')
        self.namespace = ctx.namespace
        require(0 <= budget and ctx.phase == 'decode', 'Invalid budget/phase')
        if ctx.stage != self.stage:
            self.stage_count.fill(0)
            self.stage = ctx.stage
            self.feedback = dict.fromkeys(self.feedback, 0.5)
        ids = np.asarray(ctx.eligible, dtype=np.int64)
        def rank(kind):
            if kind == 'history':
                known = ids[self.count[ids] > 0]
                return known[np.lexsort((known, -self.last[known], -self.count[known]))].tolist()
            if kind == 'semantic':
                return self.semantic.rank(ctx)
            if kind == 'sequential':
                return ids[::-1].tolist()
            prior = self.event.prior(ctx, conditioned=kind != 'unconditioned')
            regions = np.minimum(15, ids * 16 // (ctx.position + 1))
            signal = prior[regions]
            if kind in {'event', 'unconditioned'}:
                signal = signal + self.count[ids] / max(1, self.count[ids].max(initial=0)) / 16
            return ids[np.lexsort((ids, -signal))].tolist()
        strategy, arms, allocation = self.strategy, {}, {}
        if strategy == 'none' or not budget:
            result = []
        elif strategy == 'last_selection':
            eligible = set(ctx.eligible)
            result = [b for b in self.previous if b in eligible][:budget]
        elif strategy == 'echo_style':
            require(ctx.scores is not None, 'ECHO-style requires verified full current native scores')
            result = [] if self.threshold is None else [int(i) for i in ids if ctx.scores[i] > self.threshold][:budget]
        elif strategy == 'global_frequency':
            result = rank('history')[:budget]
        elif strategy == 'stage_frequency':
            known = ids[self.stage_count[ids] > 0]
            result = known[np.lexsort((known, -self.last[known], -self.stage_count[known]))][:budget].tolist()
        elif strategy == 'global_recency':
            known = ids[self.count[ids] > 0]
            result = known[np.lexsort((known, -self.count[known], -self.last[known]))][:budget].tolist()
        elif strategy == 'global_region':
            known = ids[self.region[ids // 128] > 0]
            result = known[np.lexsort((known, -self.last[known], -self.count[known],
                                      -self.region_last[known // 128], -self.region[known // 128]))][:budget].tolist()
        elif strategy in {'semantic_lsa', 'event_history', 'event_only', 'unconditioned_history', 'sequential'}:
            result = rank({'semantic_lsa': 'semantic', 'event_history': 'event',
                           'event_only': 'event_only', 'unconditioned_history': 'unconditioned',
                           'sequential': 'sequential'}[strategy])[:budget]
        else:
            names = [a for a in self.feedback if strategy != 'adaptive_no_' + a]
            arms = {a: rank(a)[:budget] for a in names}
            # Confidence and historical precision use only completed observations.
            weights = [1 if strategy == 'adaptive_fixed' else max(0.05, self.feedback[a]) for a in names]
            counts = allocate(weights, budget)
            allocation = dict(zip(names, counts))
            result, seen = [], set()
            for name in names:
                for block in arms[name][:allocation[name]]:
                    if block not in seen:
                        result.append(block)
                        seen.add(block)
            for name in names:  # Fill overlap/shortfall without exceeding the shared budget.
                for block in arms[name]:
                    if len(result) == budget:
                        break
                    if block not in seen:
                        result.append(block)
                        seen.add(block)
        require(len(result) <= budget and len(result) == len(set(result)) and set(result) <= set(ctx.eligible),
                'Candidate budget/universe violation')
        self.pending = (ctx, arms)
        self.last_query = ctx.query
        return result, allocation

    def observe(self, observation):
        require(self.pending is not None, 'Prediction required')
        ctx, arms = self.pending
        require(observation.query == ctx.query and observation.completion_step > ctx.prediction_step,
                'Future/reordered completion')
        selected = np.array(observation.selected, dtype=np.int64)
        require(len(selected) == len(set(selected.tolist())) and np.all((0 <= selected) & (selected <= ctx.position)),
                'Invalid observation IDs')
        self.count[selected] += 1
        self.stage_count[selected] += 1
        self.last[selected] = ctx.query
        self.region[np.unique(selected // 128)] += 1
        self.region_last[np.unique(selected // 128)] = ctx.query
        self.previous = tuple(sorted(observation.selected))
        if ctx.scores is not None:
            kth = min(ctx.scores[i] for i in observation.selected)
            self.threshold = kth if self.threshold is None else (self.threshold + kth) / 2
        selected_set = set(observation.selected)
        for arm, candidates in arms.items():
            if candidates:
                precision = len(selected_set.intersection(candidates)) / len(candidates)
                self.feedback[arm] = (self.feedback[arm] + precision) / 2
        self.last_completion = observation.completion_step
        self.pending = None
