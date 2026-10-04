"""Collect auditable τ³-bench task/tool episodes through the served DeepSeek API.

This is a workload adapter, not a τ³-bench score: user turns are rendered from
the public task scenario and no τ user simulator or official grader is run.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .contracts import EVENT_TYPES, validate_pair_set
from .deepseek_validation import write_json
from .import_workloads import sha_ids
from .run_requests import request
from .tau3_memory import Memory
from .working_set import require

TAU_COMMIT = 'fc0055dc4e0a316c3f83133267fbd6faaa770992'
DOMAINS = ('retail', 'banking_knowledge')
EVENTS = ('tool_call', 'tool_result', 'memory_write', 'memory_consolidation',
          'memory_supersession', 'task_switch')
TARGETS = (8192, 32768)
PAIRS_PER_CELL = 2
TOKEN_TOLERANCE = 1024


def verify_tau(root: Path) -> dict:
    # The selected retrieval path is fully offline and the model endpoint is
    # loopback. τ³ imports LiteLLM, whose eager HTTP client otherwise tries a
    # workspace SOCKS proxy even though this adapter never uses that client.
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
                'http_proxy', 'https_proxy', 'all_proxy'):
        os.environ.pop(key, None)
    root = root.resolve()
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    require(head == TAU_COMMIT, 'τ³-bench commit differs from pinned v1.0.1')
    dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--',
                                     'src', 'data/tau2/domains/retail',
                                     'data/tau2/domains/banking_knowledge'], text=True).strip()
    require(not dirty, 'Pinned τ³ source or domain data was modified')
    source = root / 'src/tau2'
    require(source.is_dir(), 'Missing τ³-bench source')
    if str(root / 'src') not in sys.path:
        sys.path.insert(0, str(root / 'src'))
    from tau2.domains.retail.environment import get_tasks as retail_tasks
    from tau2.domains.banking_knowledge.environment import get_tasks as bank_tasks
    tasks = {'retail': retail_tasks('base'), 'banking_knowledge': bank_tasks()}
    require(all(len(tasks[d]) >= 48 for d in DOMAINS), 'Insufficient public tasks')
    files = [root / 'data/tau2/domains/retail/tasks.json',
             root / 'data/tau2/domains/retail/db.json',
             root / 'data/tau2/domains/banking_knowledge/tasks.json',
             root / 'data/tau2/domains/banking_knowledge/db.json']
    checksums = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return {'commit': head, 'task_counts': {d: len(tasks[d]) for d in DOMAINS},
            'data_sha256': checksums, 'tasks': tasks}


def scenario_text(task, domain: str) -> str:
    scenario = task.user_scenario
    instructions = scenario.instructions
    if domain == 'retail':
        data = instructions.model_dump() if hasattr(instructions, 'model_dump') else instructions
        return '\n'.join(str(data[k]) for k in ('reason_for_call', 'known_info') if data.get(k))
    return str(instructions)


def tool_json(value) -> str:
    if hasattr(value, 'model_dump'):
        value = value.model_dump(mode='json')
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def api_tokens(messages: list[dict]) -> list[int]:
    result = request({'model': 'dsv4', 'messages': messages, 'add_generation_prompt': True},
                     url='http://127.0.0.1:8900/tokenize')
    ids = result.get('tokens')
    require(isinstance(ids, list) and all(type(x) is int for x in ids), 'Tokenizer did not return IDs')
    return ids


def collect_episode(domain: str, task, path: Path, *, max_rounds: int = 8) -> dict:
    from tau2.domains.retail.environment import get_environment as retail_env
    from tau2.domains.banking_knowledge.environment import get_environment as bank_env
    env = retail_env() if domain == 'retail' else bank_env(retrieval_variant='bm25', task=task)
    initial = task.initial_state
    require(not (initial and initial.message_history),
            'Task has initial messages that this adapter cannot preserve')
    env.set_state(initialization_data=initial.initialization_data if initial else None,
                  initialization_actions=initial.initialization_actions if initial else None,
                  message_history=initial.message_history or [] if initial else [])
    tools = [tool.openai_schema for tool in env.get_tools()]
    messages = [{'role': 'system', 'content': env.get_policy()},
                {'role': 'user', 'content': scenario_text(task, domain)}]
    events = []
    with path.open('w') as out:
        for turn in range(max_rounds):
            payload = {'model': 'dsv4', 'messages': messages, 'tools': tools,
                       'tool_choice': 'auto', 'temperature': 0, 'seed': 0,
                       'max_tokens': 512, 'return_token_ids': True}
            started = time.time_ns()
            result = request(payload)
            finished = time.time_ns()
            message = result['choices'][0]['message']
            require(result.get('usage', {}).get('prompt_tokens', 0) > 0,
                    'Missing actual prompt token usage')
            prompt_ids = result.get('prompt_token_ids')
            require(isinstance(prompt_ids, list) and
                    len(prompt_ids) == result['usage']['prompt_tokens'],
                    'Missing or inconsistent actual prompt token IDs')
            out.write(json.dumps({'kind': 'model', 'domain': domain, 'task_id': task.id,
                                  'turn': turn, 'started_ns': started, 'finished_ns': finished,
                                  'request': payload, 'response': result}, ensure_ascii=False) + '\n')
            out.flush()
            assistant = {key: message[key] for key in ('role', 'content', 'tool_calls') if key in message}
            messages.append(assistant)
            calls = message.get('tool_calls') or []
            if not calls:
                break
            assistant_index = len(messages) - 1
            for call in calls:
                name = call['function']['name']
                args = json.loads(call['function']['arguments'])
                require(isinstance(args, dict), 'Tool arguments must be an object')
                started = time.time_ns()
                try:
                    value = env.make_tool_call(name, **args)
                    content, error = tool_json(value), None
                except Exception as exc:
                    content, error = f'{type(exc).__name__}: {exc}', str(exc)
                finished = time.time_ns()
                reply = {'role': 'tool', 'tool_call_id': call['id'], 'content': content}
                messages.append(reply)
                event = {'source': 'benchmark_native', 'kind': 'tool_result', 'domain': domain,
                         'task_id': task.id, 'tool_name': name, 'arguments': args,
                         'result': content, 'error': error, 'call_id': call['id'],
                         'started_ns': started, 'finished_ns': finished,
                         'call_count': len(calls), 'assistant_index': assistant_index}
                events.append(event)
                out.write(json.dumps(event, ensure_ascii=False) + '\n')
                out.flush()
        else:
            raise RuntimeError(f'Model did not finish task {domain}/{task.id} within {max_rounds} turns')
    require(any(e['call_count'] == 1 and e['error'] is None for e in events),
            f'No usable successful single benchmark tool call in {domain}/{task.id}')
    return {'domain': domain, 'task_id': task.id, 'messages': messages, 'events': events,
            'retrieval': 'offline_bm25' if domain == 'banking_knowledge' else None}


def memory_event(kind: str, memory: Memory, task_id: str) -> dict:
    if kind == 'memory_write':
        return memory.write('customer_fact', f'Fact from public task {task_id}', actor='harness')
    if kind == 'memory_consolidation':
        memory.write('fact_a', task_id, actor='harness')
        memory.write('fact_b', 'verified', actor='harness')
        return memory.consolidate('summary', ['fact_a', 'fact_b'],
                                  f'{task_id} verified', actor='harness')
    if kind == 'memory_supersession':
        memory.write('customer_fact', task_id, actor='harness')
        return memory.supersede('customer_fact', f'{task_id} corrected', actor='harness')
    if kind == 'task_switch':
        return memory.switch_task(task_id + ':next', actor='harness')
    raise ValueError(kind)


def paired_tail(episode: dict, kind: str, memory: Memory) -> tuple[list[dict], list[dict], dict]:
    call = next(e for e in episode['events'] if e['call_count'] == 1 and e['error'] is None)
    if kind in {'tool_call', 'tool_result'}:
        actual = {'role': 'tool', 'tool_call_id': call['call_id'], 'content': call['result']}
        source = {'source': 'benchmark_native', 'tool_name': call['tool_name'],
                  'call_id': call['call_id'], 'started_ns': call['started_ns'],
                  'finished_ns': call['finished_ns'],
                  'assistant_index': call['assistant_index']}
        if kind == 'tool_call':
            assistant = episode['messages'][call['assistant_index']]
            return [assistant, actual], [
                {'role': 'assistant', 'content': 'I will review the request.'},
                {'role': 'user', 'content': 'Please continue.'}], source
    else:
        event = memory_event(kind, memory, episode['task_id'])
        actual = {'role': 'tool', 'tool_call_id': 'memory-observe-' + episode['task_id'],
                  'content': json.dumps(event, ensure_ascii=False, sort_keys=True)}
        source = event
    control = dict(actual, content='No new information was recorded.')
    # The shared assistant call gives both branches a valid tool response slot.
    if kind not in {'tool_call', 'tool_result'}:
        shared = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': actual['tool_call_id'],
                  'type': 'function', 'function': {'name': 'memory_observe', 'arguments': '{}'}}]}
        return [shared, actual], [shared, control], source
    return [actual], [control], source


def match_control(prefix: list[dict], event_tail: list[dict], control_tail: list[dict]) -> tuple[list[int], list[int], list[dict]]:
    event_ids = api_tokens(prefix + event_tail)
    target = len(event_ids)
    best = None
    low, high = 1, max(2, target)
    for _ in range(16):
        count = (low + high) // 2
        trial = [dict(m) for m in control_tail]
        trial[-1]['content'] = 'Record unchanged. ' * count
        ids = api_tokens(prefix + trial)
        if best is None or abs(len(ids) - target) < abs(len(best[0]) - target):
            best = ids, trial
        if abs(len(ids) - target) <= 2:
            break
        if len(ids) > target:
            high = count - 1
        else:
            low = count + 1
        if low > high:
            break
    require(best is not None and abs(len(best[0]) - target) <= 2,
            'Unable to match event/control token budgets')
    return event_ids, best[0], best[1]


def make_pairs(episodes: list[dict], output: Path, tokenizer_sha256: str) -> dict:
    require(len(episodes) == 96, 'Expected 48 real episodes per domain')
    by_domain = {d: [e for e in episodes if e['domain'] == d] for d in DOMAINS}
    pairs, audit = [], []
    for domain in DOMAINS:
        corpus = by_domain[domain]
        require(len(corpus) == 48, 'Missing domain episodes')
        lengths = {e['task_id']: len(api_tokens(e['messages'])) for e in corpus}
        for index, episode in enumerate(corpus[:24]):
            kind = EVENTS[(index // 2) % 6]
            target = TARGETS[(index // 12) % 2]
            chain = index % 2
            memory = Memory(episode['task_id'])
            event_tail, control_tail, event_source = paired_tail(episode, kind, memory)
            # Keep the anchor's system policy and its real local conversation
            # immediately before the observed event. Older complete episodes
            # form prior turns; no tool result is substituted in the event arm.
            prefix = episode['messages'][:1]
            if kind == 'tool_call':
                anchor_prefix = episode['messages'][1:event_source['assistant_index']]
            elif kind == 'tool_result':
                anchor_prefix = episode['messages'][1:event_source['assistant_index'] + 1]
            else:
                anchor_prefix = episode['messages'][1:]
            pool = range(0, 12) if target == 8192 else range(12, 48)
            remaining = [e for j, e in enumerate(corpus) if j in pool and j % 2 == chain and
                         e['task_id'] != episode['task_id']]
            history_task_ids = [episode['task_id']]
            current = len(api_tokens(prefix + anchor_prefix + event_tail))
            while current < target - TOKEN_TOLERANCE and remaining:
                gap = target - current
                fitting = [e for e in remaining if lengths[e['task_id']] <= gap + TOKEN_TOLERANCE // 2]
                if not fitting:
                    break
                previous = max(fitting, key=lambda e: lengths[e['task_id']])
                remaining.remove(previous)
                prefix.extend(previous['messages'][1:])
                history_task_ids.append(previous['task_id'])
                current = len(api_tokens(prefix + anchor_prefix + event_tail))
            prefix.extend(anchor_prefix)
            event_ids, control_ids, matched = match_control(prefix, event_tail, control_tail)
            require(abs(len(event_ids)-target) <= TOKEN_TOLERANCE and
                    abs(len(control_ids)-target) <= TOKEN_TOLERANCE,
                    f'Context target unavailable for {domain}/{episode["task_id"]}/{kind}/{target}')
            boundary = next((i for i, (a, b) in enumerate(zip(event_ids, control_ids)) if a != b), -1)
            require(boundary > 0 and boundary < min(len(event_ids), len(control_ids)),
                    'No usable event/control token boundary')
            pair_id = f'{domain}_{kind}_{target}_{chain}'
            pair = {'pair_id': pair_id, 'workload_id': 'tau3_v1.0.1_' + domain,
                    'episode_id': episode['task_id'], 'source_trace_id': domain + ':' + episode['task_id'],
                    'context_target': target, 'event_type': kind, 'event_source': event_source,
                    'history_task_ids': history_task_ids}
            for variant, tail, ids in (('event', event_tail, event_ids), ('control', matched, control_ids)):
                pair[variant] = {'prompt': json.dumps(prefix + tail, ensure_ascii=False),
                                 'messages': prefix + tail, 'boundary_position': boundary,
                                 'prompt_tokens_expected': len(ids),
                                 'prompt_token_ids_sha256_expected': sha_ids(ids),
                                 'prompt_token_ids': ids}
            pairs.append(pair)
            audit.append({'pair_id': pair_id, 'domain': domain, 'task_id': episode['task_id'],
                          'event_type': kind, 'context_target': target, 'chain': chain,
                          'event_source': event_source, 'memory_audit': memory.audit,
                          'history_task_ids': history_task_ids,
                          'event_tokens': len(event_ids), 'control_tokens': len(control_ids),
                          'boundary_position': boundary})
    result = {'schema_version': 1, 'tokenizer_json_sha256': tokenizer_sha256,
              'workload': 'tau3_v1.0.1', 'pairs': pairs}
    validate_pair_set(result)
    write_json(output / 'pairs.json', result)
    write_json(output / 'event-audit.json', audit)
    return result


def collect(tau_root: Path, output: Path, tokenizer_sha256: str) -> dict:
    provenance = verify_tau(tau_root)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'tau3-provenance.json', {k: v for k, v in provenance.items() if k != 'tasks'})
    episodes = []
    rejected = []
    for domain in DOMAINS:
        successes = 0
        for task in sorted(provenance['tasks'][domain], key=lambda t: t.id):
            if successes == 48:
                break
            path = output / f'episode-{domain}-{task.id}.jsonl'
            try:
                episodes.append(collect_episode(domain, task, path))
                successes += 1
            except (ValueError, RuntimeError, KeyError, TypeError) as error:
                rejected.append({'domain': domain, 'task_id': task.id, 'reason': str(error),
                                 'episode_log': str(path)})
        require(successes == 48, f'Only {successes}/48 usable {domain} episodes')
    write_json(output / 'rejected-episodes.json', rejected)
    write_json(output / 'episodes.json', episodes)
    return make_pairs(episodes, output, tokenizer_sha256)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tau-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokenizer-sha256', required=True)
    args = parser.parse_args()
    result = collect(args.tau_root, args.output, args.tokenizer_sha256)
    print(json.dumps({'pairs': len(result['pairs']), 'output': str(args.output)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
