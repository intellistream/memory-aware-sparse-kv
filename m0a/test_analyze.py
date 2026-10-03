import json
import tempfile
import unittest
from pathlib import Path

from m0a.analyze import load_trace, locality, summarize, summarize_groups
from m0a.synthetic_trace import make_trace_record


def response(request_id, variant, repetition, *, pair_id='episode-a'):
    return {
        'schema_version': 1,
        'pair_id': pair_id,
        'workload_id': 'tool_research',
        'episode_id': pair_id,
        'repetition': repetition,
        'variant': variant,
        'request_id': request_id,
        'run_id': 'test-run',
        'boundary_position': 1024,
        'context_target': 8192,
        'event_type': 'tool_call',
        'started_ns': 1,
        'finished_ns': 2,
        'prompt_tokens': 1100,
        'prompt_token_ids_sha256': 'abc',
        'signature': {},
    }


class AnalyzeTest(unittest.TestCase):
    def test_locality(self):
        self.assertEqual(locality({1, 2}, {2, 3}),
                         {'jaccard': 1 / 3, 'new_block_rate': 1 / 2})

    def build_fixture(self, directory):
        trace_dir = Path(directory)
        rows, responses = [], []
        for repetition in range(3):
            for variant in ('event', 'control'):
                request_id = f'{variant}-{repetition}'
                responses.append(response(request_id, variant, repetition))
                for position, raw in ((1023, [0]),
                                      (1024, [128] if variant == 'event' else [0])):
                    rows.append(make_trace_record(
                        run_id='test-run', request_id=request_id + '-deadbeef', rank=0,
                        layer='model.layers.3', position=position,
                        raw_values=raw, request_context_len=1100,
                    ))
        (trace_dir / 'rank0.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in rows))
        return responses

    def test_complete_pair_and_missing_token(self):
        with tempfile.TemporaryDirectory() as directory:
            responses = self.build_fixture(directory)
            trace, _ = load_trace(Path(directory), responses)
            pair = summarize(trace, responses)['episode-a']
            self.assertEqual(pair['mean_jaccard_drop']['mean'], 1.0)
            self.assertEqual(pair['mean_new_block_rate_increase']['mean'], 1.0)
            self.assertEqual(pair['trajectory_pairs'][0]['prefix_selected_set_matches'], 1)
            self.assertIn('model.layers.3', pair['by_layer'])
            groups = summarize_groups({'episode-a': pair})
            group = groups['tool_research/tool_call/8192']
            self.assertEqual(group['independent_episode_count'], 1)
            missing = dict(trace)
            missing.pop(('event-0', 'model.layers.3', 1024))
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                summarize(missing, responses)

    def test_trace_contract_rejects_bad_mapping_and_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            responses = self.build_fixture(directory)
            path = Path(directory) / 'rank0.jsonl'
            records = [json.loads(line) for line in path.read_text().splitlines()]
            records[0]['logical_compressed_block_ids'][0] = 99
            path.write_text(''.join(json.dumps(row) + '\n' for row in records))
            with self.assertRaisesRegex(ValueError, 'Block mapping'):
                load_trace(Path(directory), responses)

        with tempfile.TemporaryDirectory() as directory:
            responses = self.build_fixture(directory)
            path = Path(directory) / 'rank0.jsonl'
            first = path.read_text().splitlines()[0]
            with path.open('a') as stream:
                stream.write(first + '\n')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                load_trace(Path(directory), responses)

    def test_group_bootstrap_counts_episodes_not_repetitions(self):
        with tempfile.TemporaryDirectory() as directory:
            responses = self.build_fixture(directory)
            trace, _ = load_trace(Path(directory), responses)
            first = summarize(trace, responses)['episode-a']
            second = dict(first)
            second['episode_id'] = 'episode-b'
            groups = summarize_groups({'episode-a': first, 'episode-b': second})
            self.assertEqual(
                groups['tool_research/tool_call/8192']['independent_episode_count'], 2
            )

    def test_prefix_mismatch_and_duplicate_variant_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            responses = self.build_fixture(directory)
            trace, _ = load_trace(Path(directory), responses)
            rank, _ = trace[('control-0', 'model.layers.3', 1023)]
            trace[('control-0', 'model.layers.3', 1023)] = (rank, {9})
            with self.assertRaisesRegex(ValueError, 'prefix selected sets differ'):
                summarize(trace, responses)
        with tempfile.TemporaryDirectory() as directory:
            responses = self.build_fixture(directory)
            trace, _ = load_trace(Path(directory), responses)
            duplicate = dict(responses[0])
            with self.assertRaisesRegex(ValueError, 'Duplicate variant'):
                summarize(trace, [*responses, duplicate])


if __name__ == '__main__':
    unittest.main()
