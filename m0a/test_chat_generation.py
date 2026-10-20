import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from m0a.chat_generation import ChatGenerationError, generate_chat


def reply(*, finish='stop', content=None, calls=None, reasoning=None, tokens=1):
    return {'choices': [{'finish_reason': finish, 'message': {
        'content': content, 'tool_calls': calls, 'reasoning': reasoning}}],
        'usage': {'completion_tokens': tokens}}


class ChatGenerationTests(unittest.TestCase):
    def test_truncation_retries_same_conversation_until_real_body(self):
        payload = {'max_tokens': 512, 'messages': [{'role': 'user', 'content': 'request'}],
                   'tools': [{'type': 'function', 'function': {'name': 'lookup'}}]}
        original = copy.deepcopy(payload)
        requests, audit = [], []
        responses = iter([reply(finish='length', reasoning='private', tokens=512),
                          reply(finish='length', content='unfinished', tokens=2048),
                          reply(content='complete')])
        def request(current):
            requests.append(copy.deepcopy(current))
            return next(responses)
        result = generate_chat(payload, request_fn=request, max_tokens_ceiling=8192,
                               audit=lambda p, r, a: audit.append((p, r, a)))
        self.assertEqual([r['max_tokens'] for r in requests], [512, 2048, 8192])
        self.assertTrue(all(r['messages'] == original['messages'] and r['tools'] == original['tools']
                            for r in requests))
        self.assertEqual(payload, original)
        self.assertEqual(result['choices'][0]['message']['content'], 'complete')
        self.assertEqual(len(audit), 3)
        self.assertEqual([a[2]['attempt'] for a in audit], [1, 2, 3])

    def test_cap_exhaustion_audits_partial_text_but_does_not_accept_it(self):
        audit = []
        with self.assertRaises(ChatGenerationError) as raised:
            generate_chat({'max_tokens': 512}, max_tokens_ceiling=1000,
                request_fn=lambda _: reply(finish='length', content='unfinished'),
                audit=lambda p, r, a: audit.append(a))
        self.assertEqual([a['max_tokens'] for a in audit], [512, 1000])
        self.assertEqual(raised.exception.details['reason'], 'generation_budget_exhausted')

    def test_empty_stop_is_explicit_failure_not_reasoning_as_body(self):
        audit = []
        with self.assertRaises(ChatGenerationError) as raised:
            generate_chat({'max_tokens': 512}, max_tokens_ceiling=8192,
                request_fn=lambda _: reply(reasoning='private'),
                audit=lambda p, r, a: audit.append(a))
        self.assertEqual(raised.exception.details['reason'], 'empty_model_message')
        self.assertEqual(len(audit), 1)
        self.assertNotIn('private', str(raised.exception))

    def test_real_tool_call_is_accepted_without_invented_body(self):
        calls = [{'id': 'c1', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{}'}}]
        response = reply(finish='tool_calls', calls=calls)
        self.assertIs(generate_chat({'max_tokens': 512}, request_fn=lambda _: response), response)
        self.assertIsNone(response['choices'][0]['message']['content'])

    def test_invalid_budget_never_calls_server(self):
        for initial, ceiling in [(0, 8192), (512, 128), (True, 8192), (512, 8192.0)]:
            with self.subTest(initial=initial, ceiling=ceiling), self.assertRaises(ValueError):
                generate_chat({'max_tokens': initial}, max_tokens_ceiling=ceiling,
                              request_fn=lambda _: self.fail('invalid budget sent'))

    def test_tau_user_truncation_is_logged_before_message_construction(self):
        source = os.environ.get('TAU3_SOURCE')
        if not source:
            self.skipTest('Pinned tau3 environment required')
        try:
            import tau2  # noqa: F401
        except ImportError:
            self.skipTest('Pinned tau3 dependencies required')
        from m0a.tau3_workload import collect_episode, verify_tau
        task = verify_tau(Path(source))['tasks']['retail'][0]
        def response(_payload, **_kwargs):
            result = reply(finish='length', reasoning='private')
            result['usage']['prompt_tokens'] = 1
            result['prompt_token_ids'] = [1]
            return result
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'episode.jsonl'
            with patch('m0a.tau3_workload.request', side_effect=response), \
                    self.assertRaises(ChatGenerationError) as raised:
                collect_episode('retail', task, log, model='glm-tiny', max_tokens=512,
                                max_tokens_ceiling=2048, require_success=False)
            self.assertEqual(raised.exception.details['reason'], 'generation_budget_exhausted')
            records = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual([r['request']['max_tokens'] for r in records], [512, 2048])
            self.assertTrue(all(r['actor'] == 'user_simulator' for r in records))
            self.assertTrue(all(r['response']['choices'][0]['message']['content'] is None for r in records))


if __name__ == '__main__':
    unittest.main()
