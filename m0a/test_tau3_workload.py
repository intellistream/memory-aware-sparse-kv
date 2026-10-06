from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from m0a.tau3_locality import analyze
from m0a.tau3_memory import Memory, digest
from m0a.tau3_workload import (DOMAINS, EVENTS, TARGETS, complete_fragments,
                              choose_history, make_pairs, record_user_incoming,
                              select_episodes)


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

    def test_selection_keeps_all_distinct_history_sources(self):
        candidates = {domain: [{'domain': domain, 'task_id': f'{i:03d}'} for i in range(50)]
                      for domain in DOMAINS}
        selected = select_episodes(candidates, {})
        for domain in DOMAINS:
            chosen = [e['task_id'] for e in selected if e in candidates[domain]]
            self.assertEqual(chosen, [f'{i:03d}' for i in range(50)])

    def test_grouped_history_uses_each_task_at_most_once(self):
        options = [('a', [(1400, [{'role': 'user', 'content': 'a-short'}]),
                           (4000, [{'role': 'user', 'content': 'a-long'}])]),
                   ('b', [(3000, [{'role': 'user', 'content': 'b'}])])]
        selected = choose_history(1500, 8192, options)
        self.assertIsNotNone(selected)
        messages, task_ids, actual = selected
        self.assertEqual(task_ids, ['a', 'b'])
        self.assertEqual(len(task_ids), len(set(task_ids)))
        self.assertLessEqual(abs(actual - 8192), 1024)
        self.assertEqual([m['content'] for m in messages], ['a-long', 'b'])
        self.assertIsNone(choose_history(1655, 8192,
                          [(str(i), [(200, [{'role': 'user', 'content': str(i)}])])
                           for i in range(5)]))

    def test_stop_marker_is_not_history_fragment(self):
        messages = [{'role': 'system', 'content': 'policy'},
                    {'role': 'assistant', 'content': 'hello'},
                    {'role': 'assistant', 'content': '###STOP###'}]
        self.assertEqual(complete_fragments(messages), [messages[1:2]])

    def test_pair_builder_covers_all_cells_with_distinct_long_histories(self):
        episodes = []
        for domain in DOMAINS:
            for index in range(60):
                task_id = f'{index:03d}'
                call_id = f'{domain}-{task_id}'
                messages = [{'role': 'system', 'content': 'policy', '_tokens': 1000},
                            {'role': 'user', 'content': 'ask', '_tokens': 50},
                            {'role': 'assistant', 'content': None, '_tokens': 300,
                             'tool_calls': [{'id': call_id, 'type': 'function',
                                             'function': {'name': 'KB_search', 'arguments': '{}'}}]},
                            {'role': 'tool', 'tool_call_id': call_id,
                             'content': 'result', '_tokens': 2750}]
                event = {'eligible': True, 'call_id': call_id, 'result': 'result',
                         'tool_name': 'KB_search', 'started_ns': 1, 'finished_ns': 2,
                         'assistant_index': 2}
                episodes.append({'domain': domain, 'task_id': task_id,
                                 'messages': messages, 'events': [event]})

        def tokens(messages):
            return list(range(sum(message.get('_tokens', 10) for message in messages)))

        def match(prefix, event, control):
            size = len(tokens(prefix + event))
            return list(range(size)), [0, -1, *range(2, size)], control

        with tempfile.TemporaryDirectory() as name, \
             patch('m0a.tau3_workload.api_tokens', side_effect=tokens), \
             patch('m0a.tau3_workload.match_control', side_effect=match):
            pairs = make_pairs(episodes, Path(name), 'tokenizer-hash')['pairs']
            self.assertEqual(len(pairs), 48)
            for domain in DOMAINS:
                source = [pair for pair in pairs if pair['workload_id'].endswith(domain)]
                self.assertEqual(len({pair['episode_id'] for pair in source}), 24)
                self.assertGreaterEqual(len({task_id for pair in source
                                             for task_id in pair['history_task_ids']}), 48)
                self.assertTrue(all(len(pair['history_task_ids']) ==
                                    len(set(pair['history_task_ids'])) for pair in source))
                training = {task for pair in source if pair['context_target'] == TARGETS[0]
                            for task in pair['history_task_ids']}
                evaluation = {task for pair in source if pair['context_target'] == TARGETS[1]
                              for task in pair['history_task_ids']}
                self.assertFalse(training & evaluation)


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
