from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from m0a.tau3_locality import analyze
from m0a.tau3_memory import Memory, digest
from m0a.tau3_workload import DOMAINS, EVENTS, TARGETS


class MemoryAuditTests(unittest.TestCase):
    def test_versions_and_hash_chain(self):
        memory = Memory('task-a')
        events = [memory.write('a', 'first'), memory.write('b', 'second'),
                  memory.consolidate('both', ['a', 'b'], 'summary'),
                  memory.supersede('a', 'corrected'), memory.switch_task('task-b')]
        self.assertEqual([event['event_type'] for event in events],
                         ['memory_write', 'memory_write', 'memory_consolidation',
                          'memory_supersession', 'task_switch'])
        self.assertEqual(memory.entries['a']['version'], 2)
        self.assertEqual(events[3]['detail']['replaced_version'], 1)
        for event in events:
            self.assertEqual(event['before_sha256'], digest(event['before']))
            self.assertEqual(event['after_sha256'], digest(event['after']))
            self.assertNotEqual(event['before_sha256'], event['after_sha256'])
        with self.assertRaises(ValueError):
            memory.write('a', 'silently overwrite')


class LocalityAuditTests(unittest.TestCase):
    def test_full_coverage_and_direction_across_budgets(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            pairs = []
            for domain in DOMAINS:
                for kind in EVENTS:
                    for context in TARGETS:
                        for chain in range(2):
                            pairs.append({'pair_id': f'{domain}-{kind}-{context}-{chain}',
                                          'workload_id': domain, 'event_type': kind,
                                          'context_target': context})
            (directory / 'pairs.json').write_text(json.dumps({'pairs': pairs}))
            for scope in ('per_rank', 'aggregate'):
                for capacity in (64, 128):
                    path = directory / f'replay-{scope}-{capacity}mib'
                    path.mkdir()
                    with (path / 'events.jsonl').open('w') as output:
                        for pair in pairs:
                            for repetition in (0, 1):
                                for variant in ('event', 'control'):
                                    delta = 10 if (pair['event_type'] == 'tool_result'
                                                   and pair['context_target'] == 32768
                                                   and variant == 'event') else 0
                                    output.write(json.dumps({**pair, 'repetition': repetition,
                                        'variant': variant, 'strategy': 'sequential_selected_set',
                                        'granularity': 'native', 'synchronous_recall_bytes': 100 + delta,
                                        'recall': .5}) + '\n')
            result = analyze(directory)
            self.assertEqual(len(result['cell_conclusions']), 24)
            self.assertEqual(result['cell_conclusions']['retail:tool_result:32768'], 'weaker_locality')
            self.assertEqual(result['cell_conclusions']['banking_knowledge:tool_result:32768'], 'weaker_locality')
            self.assertEqual(result['overall'], 'mixed_or_inconclusive')
            self.assertEqual(result['echo_mechanism_conclusion'],
                             'not_qualified_without_indexer_scores_and_timed_replay')


if __name__ == '__main__':
    unittest.main()
