"""Fixed task splits, read-only distractors, and genuine GLM/GLM benchmark episodes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from .glm_agent_cases import REVISION, sha_file
from .tau3_workload import verify_tau, collect_episode
from .history_prefetch import require


NOTES = {
    'natural_conversation': 'Earlier unrelated conversation: I enjoyed walking through the botanical garden. '
        'The shaded path passed a greenhouse, a lily pond and several tall oak trees. '
        'A gardener explained how the seasonal flowers attract butterflies. We discussed the weather '
        'and the different colors of the leaves before returning to the current conversation.',
    'unrelated_result': 'Unrelated read-only archive excerpt about astronomy: The Moon reflects sunlight. '
        'Its visible phases change during the month as its position relative to Earth and the Sun changes. '
        'A lunar eclipse requires Earth to lie between the Moon and the Sun. This excerpt is background '
        'reading and contains no instructions or facts about the current customer request.'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def split_tasks(tasks, count=6):
    require(count >= 6, 'Need at least two independent tasks per partition/domain')
    chosen = sorted(tasks, key=lambda t: t.id)[:count]
    require(len(chosen) == count and len({t.id for t in chosen}) == count, 'Insufficient/duplicate task IDs')
    return [(task, ('train', 'validation', 'test')[index % 3]) for index, task in enumerate(chosen)]


def run(args):
    require(0 < args.max_tokens <= args.max_tokens_ceiling, 'Invalid generation budget range')
    require(args.episode_timeout > 0, 'Invalid episode timeout')
    args.output.mkdir(parents=True, exist_ok=False)
    verified = verify_tau(args.tau_source)
    tasks = verified.pop('tasks')
    rows, episodes = [], []
    materials = []
    for domain in ('retail', 'banking_knowledge'):
        for task, split in split_tasks(tasks[domain], args.tasks_per_domain):
            original = task.model_dump(mode='json')
            source_hash = digest(original)
            from tau2.domains.retail.environment import get_environment as retail_env
            from tau2.domains.banking_knowledge.environment import get_environment as bank_env
            env = (retail_env() if domain == 'retail' else
                   bank_env(retrieval_variant='bm25', retrieval_kwargs={'top_k': 3}, task=task))
            instructions = original['user_scenario']['instructions']
            text = ('Customer request: ' + instructions['reason_for_call'] + '\nKnown information: ' + instructions['known_info']
                    if isinstance(instructions, dict) else
                    'Controlled replay of the complete benchmark customer scenario (including simulator instructions):\n' + instructions)
            for variant in ('original', *NOTES):
                rows.append({'task_id': domain + ':' + task.id, 'domain': domain, 'split': split,
                    'variant': variant, 'group_id': domain + ':' + task.id,
                    'source_task_sha256': source_hash, 'task': original,
                    'distractor': NOTES.get(variant, ''),
                    'variant_kind': 'read_only_harness_context' if variant != 'original' else 'benchmark_original',
                    'business_facts_unchanged': True, 'policy_and_tools_unchanged': True,
                    'evaluation_criteria_sha256': digest(original['evaluation_criteria'])})
                messages = [{'role': 'system', 'content': env.get_policy()}]
                if variant in NOTES:
                    messages.extend([{'role': 'user', 'content': NOTES[variant]},
                                     {'role': 'assistant', 'content': 'Let us return to the current request.'}])
                messages.append({'role': 'user', 'content': text})
                materials.append({k: v for k, v in rows[-1].items() if k != 'task'} | {
                    'messages': messages, 'tools': [tool.openai_schema for tool in env.get_tools()],
                    'query_text': text, 'event_type': 'task_request',
                    'source_scope': 'scenario_disclosed_controlled_replay',
                    'autonomous_agent': False, 'actual_tool_result': False})
            if not args.dataset_only:
                name = domain + '-' + task.id
                try:
                    episode = collect_episode(domain, task, args.output / (name + '.jsonl'),
                        model='glm-tiny', url=args.url, max_steps=args.max_steps,
                        max_tokens=args.max_tokens, require_success=False,
                        chat_template_kwargs={'reasoning_effort': args.reasoning_effort},
                        max_tokens_ceiling=args.max_tokens_ceiling,
                        episode_timeout=args.episode_timeout)
                    episode['agent_tool_calls'] = sum(e['requestor'] == 'assistant' for e in episode['events'])
                    episode['successful_agent_tool_calls'] = sum(e['requestor'] == 'assistant' and not e['error']
                                                                 for e in episode['events'])
                    episode['status'] = 'tool_loop_observed' if episode['successful_agent_tool_calls'] else 'no_successful_tool_call'
                    write(args.output / (name + '-episode.json'), episode)
                    episodes.append({k: v for k, v in episode.items() if k not in {'messages', 'events'}})
                except Exception as error:
                    failure = {'domain': domain, 'task_id': task.id, 'status': 'failed',
                               'error': f'{type(error).__name__}: {error}'}
                    if hasattr(error, 'details'):
                        failure['generation_failure'] = error.details
                    episodes.append(failure)
                write(args.output / 'episodes.json', episodes)
                print(json.dumps(episodes[-1]), flush=True)
    validate_dataset(rows)
    with (args.output / 'prompt-material.jsonl').open('w', encoding='utf-8') as output:
        for row in materials:
            output.write(json.dumps(row, ensure_ascii=False) + '\n')
    with (args.output / 'dataset.jsonl').open('w', encoding='utf-8') as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + '\n')
    manifest = {'source': verified, 'model': 'GLM-5.3-0.6B-A0.4B', 'revision': REVISION,
                'rows': len(rows), 'independent_tasks': len(rows) // 3, 'episodes': episodes,
                'forced_continuation': False, 'scripted_agent_actions': False,
                'user_simulator': 'same GLM tiny; reliability must be assessed separately',
                'official_benchmark_reward_computed': False,
                'generation_config': {'max_steps': args.max_steps, 'max_tokens': args.max_tokens,
                                      'max_tokens_ceiling': args.max_tokens_ceiling,
                                      'episode_timeout': args.episode_timeout,
                                      'reasoning_effort': args.reasoning_effort},
                'variant_scope': 'read-only distractor context; source task, policy, DB and actions unchanged',
                'limitations': ['bounded capability run, not a benchmark score',
                                'no natural-task-completion claim from max-step termination'],
                'artifacts': {p.name: sha_file(p) for p in args.output.iterdir() if p.is_file()}}
    write(args.output / 'manifest.json', manifest)
    return manifest


def validate_dataset(rows):
    groups, seen = {}, set()
    for row in rows:
        require(row['split'] in {'train', 'validation', 'test'}, 'Invalid split')
        key = row['task_id'], row['variant']
        require(key not in seen, 'Duplicate task variant')
        seen.add(key)
        require(digest(row['task']) == row['source_task_sha256'], 'Task facts changed')
        require(digest(row['task']['evaluation_criteria']) == row['evaluation_criteria_sha256'], 'Task criteria changed')
        expected = (row['split'], row['source_task_sha256'])
        require(groups.setdefault(row['group_id'], expected) == expected, 'Task/variant split leakage')
        require(row['group_id'] == row['task_id'], 'Variants must retain the independent task ID')
        require(row['distractor'] == NOTES.get(row['variant'], '') and row['variant'] in {'original', *NOTES},
                'Unsealed distractor')
    require(groups, 'Empty dataset')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tau-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:8866/v1/chat/completions')
    parser.add_argument('--tasks-per-domain', type=int, default=6)
    parser.add_argument('--max-steps', type=int, default=8)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--max-tokens-ceiling', type=int, default=8192)
    parser.add_argument('--episode-timeout', type=int, default=900)
    parser.add_argument('--dataset-only', action='store_true')
    parser.add_argument('--reasoning-effort', choices=('low', 'high', 'max'), default='low')
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == '__main__':
    main()
