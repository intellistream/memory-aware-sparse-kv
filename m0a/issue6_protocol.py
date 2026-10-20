"""Protocol v1: immutable prediction inputs, completed observations, logical KV units."""
from dataclasses import dataclass
from .history_prefetch import require


@dataclass(frozen=True)
class Fragment:
    start: int
    end: int  # Half-open native token coordinates in this exact context version.
    text: str


@dataclass(frozen=True)
class Context:
    namespace: tuple
    query: int
    prediction_step: int
    position: int
    phase: str
    stage: str
    event_type: str
    eligible: tuple[int, ...] | range
    query_text: str
    fragments: tuple[Fragment, ...]
    scores: tuple[float, ...] | None = None


@dataclass(frozen=True)
class Observation:
    query: int
    completion_step: int
    selected: tuple[int, ...]


def context(row):
    return Context(namespace=tuple(row[k] for k in ('task_id', 'run_id', 'request_id',
                    'context_version', 'rank', 'layer', 'phase')),
        query=row['query'], prediction_step=row['prediction_step'], position=row['position'],
        phase=row['phase'], stage=row['stage'], event_type=row['event_type'],
        eligible=range(row['eligible_end']) if 'eligible_end' in row else tuple(row['eligible_blocks']), query_text=row['query_text'],
        fragments=tuple(Fragment(**f) for f in row['fragments']),
        scores=tuple(row['scores']) if row.get('scores') is not None else None)


def validate(rows):
    import math
    lanes, groups = {}, {}
    for row in rows:
        require(row['schema_version'] == 1 and row['unit'] == 'native_kv_token', 'Unsupported trace protocol/unit')
        require(row['source_kind'] in {'native_trace', 'synthetic_fixture'}, 'Unknown source')
        require(row['split'] in {'train', 'validation', 'test'}, 'Invalid split')
        require(groups.setdefault(row['task_id'], row['split']) == row['split'], 'Task split leakage')
        ctx = context(row)
        require(ctx.phase == 'decode', 'Cache accounting requires complete contiguous decode; sampled prefill is diagnostic only')
        lane = lanes.setdefault(ctx.namespace, [])
        require(ctx.query == len(lane), 'Missing/duplicate/reordered decode query')
        require(row['completion_step'] > ctx.prediction_step and (not lane or
                ctx.prediction_step > lane[-1]['completion_step']), 'Future observation visibility')
        require(ctx.position >= 0 and (row.get('eligible_end') == ctx.position if 'eligible_end' in row else
                row['eligible_blocks'] == list(range(ctx.position))),
                'Candidate universe must be pre-query generated KV')
        selected = row['selected_blocks']
        require(selected and len(selected) == len(set(selected)) and all(type(v) is int and 0 <= v <= ctx.position
                for v in selected), 'Invalid native selected set')
        require(all(0 <= f.start < f.end <= ctx.position for f in ctx.fragments), 'Fragment outside completed KV')
        if ctx.scores is not None:
            require(len(ctx.scores) == ctx.position + 1 and all(math.isfinite(s) for s in ctx.scores),
                    'Incomplete/nonfinite full native scores')
            chosen = set(selected)
            floor = min(ctx.scores[i] for i in chosen)
            require(all(s <= floor + 1e-4 for i, s in enumerate(ctx.scores) if i not in chosen),
                    'Full native score vector disagrees with top-k')
        lane.append(row)
    require(lanes and len({r['source_kind'] for r in rows}) == 1, 'Empty/mixed provenance')
    for lane in lanes.values():
        require(all(r['position'] == lane[0]['position'] + i for i, r in enumerate(lane)), 'Incomplete decode positions')
    return lanes
