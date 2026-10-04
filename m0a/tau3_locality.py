"""Paired, budget-matched locality audit of sequential selected-set replay."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .deepseek_validation import write_json
from .working_set import require


def analyze(directory: Path) -> dict:
    pairs = json.loads((directory / 'pairs.json').read_text())['pairs']
    metadata = {p['pair_id']: p for p in pairs}
    require(len(metadata) == 48, 'Incomplete τ³ pair set')
    budgets = {}
    for scope in ('per_rank', 'aggregate'):
        for capacity in (64, 128):
            label = f'{scope}-{capacity}mib'
            path = directory / f'replay-{label}/events.jsonl'
            with path.open() as source:
                rows = [json.loads(line) for line in source]
            selected = [r for r in rows if r['strategy'] == 'sequential_selected_set'
                        and r['granularity'] == 'native']
            keyed = {(r['pair_id'], r['repetition'], r['variant']): r for r in selected}
            require(len(keyed) == len(pairs) * 4, 'Missing or duplicate locality replay row')
            groups = defaultdict(list)
            for pair in pairs:
                for repetition in (0, 1):
                    event = keyed[pair['pair_id'], repetition, 'event']
                    control = keyed[pair['pair_id'], repetition, 'control']
                    group = (pair['workload_id'], pair['event_type'], pair['context_target'])
                    groups[group].append({'pair_id': pair['pair_id'], 'repetition': repetition,
                                          'delta_sync_bytes': event['synchronous_recall_bytes'] - control['synchronous_recall_bytes'],
                                          'delta_recall': event['recall'] - control['recall'],
                                          'event_sync_bytes': event['synchronous_recall_bytes'],
                                          'control_sync_bytes': control['synchronous_recall_bytes']})
            require(len(groups) == 24, 'Missing domain/event/context locality cell')
            cells = []
            for (domain, kind, context), values in sorted(groups.items()):
                by_pair = defaultdict(list)
                for row in values:
                    by_pair[row['pair_id']].append(row['delta_sync_bytes'])
                require(len(by_pair) == 2 and all(len(v) == 2 for v in by_pair.values()),
                        'Incomplete independent chains or repetitions')
                means = [sum(v)/len(v) for v in by_pair.values()]
                cells.append({'domain': domain, 'event_type': kind, 'context_target': context,
                              'independent_pairs': 2, 'repetitions_per_pair': 2,
                              'pair_mean_delta_sync_bytes': means,
                              'mean_delta_sync_bytes': sum(means)/2,
                              'direction': 'weaker_locality' if all(v > 0 for v in means)
                                  else 'stronger_locality' if all(v < 0 for v in means) else 'inconclusive',
                              'observations': values})
            budgets[label] = cells
    # A budget-wide directional claim requires agreement by both independent
    # task chains in every tested budget. A mixed cell stays inconclusive.
    conclusions = {}
    for cell in budgets['aggregate-64mib']:
        key = f"{cell['domain']}:{cell['event_type']}:{cell['context_target']}"
        directions = [next(c['direction'] for c in budgets[label] if
                           (c['domain'], c['event_type'], c['context_target']) ==
                           (cell['domain'], cell['event_type'], cell['context_target']))
                      for label in budgets]
        conclusions[key] = directions[0] if len(set(directions)) == 1 else 'inconclusive'
    result = {'scope': 'offline C4 attention selected-set CPU replay',
              'metric': 'event minus control synchronous recall bytes; positive means weaker sequential locality',
              'budgets': budgets, 'cell_conclusions': conclusions,
              'overall': 'mixed_or_inconclusive' if len(set(conclusions.values())) != 1
                  else next(iter(conclusions.values())),
              'echo_mechanism_conclusion': 'not_qualified_without_indexer_scores_and_timed_replay'}
    write_json(directory / 'locality.json', result)
    return result
