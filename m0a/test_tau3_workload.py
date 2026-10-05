from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from m0a.tau3_locality import analyze
from m0a.tau3_memory import Memory, digest
from m0a.tau3_workload import (DOMAINS, EVENTS, TARGETS, complete_fragments,
                              record_user_incoming, select_episodes)


class UserHistoryTests(unittest.TestCase):
    def test_pinned_tau_user_tool_replies_keep_ids_through_role_flip(self):
        try:
            from tau2.data_model.message import (AssistantMessage, MultiToolMessage,
                                                 SystemMessage, ToolCall, ToolMessage, UserMessage)
            from tau2.user.user_simulator_base import UserState
            from tau2.utils.llm_utils import to_litellm_messages
        except ImportError:
            self.skipTest('Pinned τ³ dependencies are available in the Pod environment')

        call = ToolCall(id='user-call-034', name='submit_transfer',
                        arguments={}, requestor='user')
        state = UserState(system_messages=[SystemMessage(role='system', content='scenario')],
                          messages=[UserMessage(role='user', tool_calls=[call])])
        single = ToolMessage(id=call.id, role='tool', content='Transfer request submitted',
                             requestor='user')
        record_user_incoming(single, state)
        self.assertIs(state.messages[-1], single)
        self.assertEqual(state.messages[-1].requestor, 'user')
        second = ToolMessage(id='user-call-035', role='tool', content='Second result',
                             requestor='user')
        record_user_incoming(MultiToolMessage(role='tool', tool_messages=[second]), state)
        record_user_incoming(AssistantMessage(role='assistant', content='Continue'), state)
        record_user_incoming(AssistantMessage(role='assistant', content=None), state)
        flipped = to_litellm_messages(state.system_messages + state.flip_roles())
        self.assertEqual([message['tool_call_id'] for message in flipped
                          if message['role'] == 'tool'], ['user-call-034', 'user-call-035'])
        self.assertEqual(flipped[1]['role'], 'assistant')
        self.assertEqual(flipped[1]['tool_calls'][0]['id'], 'user-call-034')
        self.assertEqual(flipped[-1], {'role': 'user', 'content': 'Continue'})


class EpisodeSelectionTests(unittest.TestCase):
    def test_complete_fragments_wait_for_all_tool_results(self):
        messages = [{'role': 'system', 'content': 'policy'},
                    {'role': 'assistant', 'content': 'Hello'},
                    {'role': 'user', 'content': 'Find a card'},
                    {'role': 'assistant', 'content': None, 'tool_calls': [
                        {'id': 'a'}, {'id': 'b'}]},
                    {'role': 'tool', 'tool_call_id': 'a', 'content': 'first'},
                    {'role': 'tool', 'tool_call_id': 'b', 'content': 'second'}]
        fragments = complete_fragments(messages)
        self.assertEqual([len(fragment) for fragment in fragments], [1, 5])

    def test_selection_preserves_independent_short_and_long_pools(self):
        candidates = {domain: [{'domain': domain, 'task_id': f'{i:03d}'} for i in range(50)]
                      for domain in DOMAINS}
        lengths = {f'{domain}:{i:03d}': 5000 if i < 12 else 9000
                   for domain in DOMAINS for i in range(50)}
        selected = select_episodes(candidates, lengths)
        for domain in DOMAINS:
            chosen = [e['task_id'] for e in selected if e in candidates[domain]]
            self.assertEqual(len(chosen), 48)
            self.assertEqual(len(set(chosen)), 48)
            self.assertEqual(chosen[:12], [f'{i:03d}' for i in range(12)])


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


class LauncherTemplateTests(unittest.TestCase):
    def test_remote_bootstrap_has_no_unresolved_placeholders(self):
        source = Path(__file__).resolve().parents[1] / 'scripts/launch_deepseek_validation.py'
        tree = ast.parse(source.read_text())
        launcher = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'launch_pod')
        namespace = {'remote_directory': '/remote/m0a/runs/deepseek_test',
                     'deployment': {'files': []},
                     'args': SimpleNamespace(model_dir=Path('/models/model'), remote_root='/remote')}
        assignments = [node for node in launcher.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == 'bootstrap'
                               for target in node.targets)]
        for assignment in assignments:
            exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(source), 'exec'), namespace)
        bootstrap = namespace['bootstrap']
        compile(bootstrap, '<remote bootstrap>', 'exec')
        for placeholder in ('DIRECTORY', 'FILES', 'MODEL', 'ROOT'):
            self.assertNotIn(placeholder, bootstrap)


if __name__ == '__main__':
    unittest.main()
