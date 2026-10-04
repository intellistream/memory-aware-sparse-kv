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
    manifest_path = root / 'tau3-source-manifest.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        head = manifest['commit']
        records = manifest['files']
        require(records and len({row['path'] for row in records}) == len(records),
                'Invalid τ³ source manifest')
        for row in records:
            path = root / row['path']
            require(path.is_file() and path.stat().st_size == row['size'] and
                    hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'],
                    'τ³ source checksum mismatch: ' + row['path'])
    else:
        head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--',
                                         'src', 'data/tau2/domains/retail',
                                         'data/tau2/domains/banking_knowledge'], text=True).strip()
        require(not dirty, 'Pinned τ³ source or domain data was modified')
    require(head == TAU_COMMIT, 'τ³-bench commit differs from pinned v1.0.1')
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


def api_tokens(messages: list[dict]) -> list[int]:
    result = request({'model': 'dsv4', 'messages': messages, 'add_generation_prompt': True},
                     url='http://127.0.0.1:8900/tokenize')
    ids = result.get('tokens')
    require(isinstance(ids, list) and all(type(x) is int for x in ids), 'Tokenizer did not return IDs')
    return ids


def collect_episode(domain: str, task, path: Path, *, max_steps: int = 40) -> dict:
    """Run the pinned τ³ half-duplex protocol through the local DeepSeek API.

    Stop at the first complete, eligible agent tool result.  The resulting
    conversation is an authentic task prefix, not an official τ³ score.
    """
    from tau2.agent.llm_agent import LLMAgent
    from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall, ToolMessage, UserMessage
    from tau2.domains.retail.environment import get_environment as retail_env
    from tau2.domains.banking_knowledge.environment import get_environment as bank_env
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.user.user_simulator import UserSimulator
    from tau2.utils.llm_utils import to_litellm_messages
    env = (retail_env() if domain == 'retail' else
           bank_env(retrieval_variant='bm25', retrieval_kwargs={'top_k': 3}, task=task))
    events = []
    call_counts = {}
    with path.open('w') as out:
        def invoke(actor: str, messages, tools):
            payload = {'model': 'dsv4', 'messages': to_litellm_messages(messages),
                       'temperature': 0, 'seed': 0, 'max_tokens': 512,
                       'return_token_ids': True}
            if tools:
                payload.update(tools=[tool.openai_schema for tool in tools], tool_choice='auto')
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
            out.write(json.dumps({'kind': 'model', 'actor': actor, 'domain': domain,
                                  'task_id': task.id, 'started_ns': started, 'finished_ns': finished,
                                  'request': payload, 'response': result}, ensure_ascii=False) + '\n')
            out.flush()
            calls = message.get('tool_calls') or []
            for call in calls:
                call_counts[call['id']] = len(calls)
            return result

        def tau_calls(result, requestor):
            return [ToolCall(id=call['id'], name=call['function']['name'],
                             arguments=json.loads(call['function']['arguments']),
                             requestor=requestor)
                    for call in result['choices'][0]['message'].get('tool_calls') or []] or None

        class AuditedAgent(LLMAgent):
            def _generate_next_message(self, incoming, state):
                if isinstance(incoming, MultiToolMessage):
                    state.messages.extend(incoming.tool_messages)
                else:
                    state.messages.append(incoming)
                result = invoke('agent', state.system_messages + state.messages, self.tools)
                message = result['choices'][0]['message']
                return AssistantMessage(role='assistant', content=message.get('content'),
                                        tool_calls=tau_calls(result, 'assistant'), cost=0.0,
                                        usage=result['usage'], raw_data=result)

        class AuditedUser(UserSimulator):
            def _generate_next_message(self, incoming, state):
                if isinstance(incoming, MultiToolMessage):
                    state.messages.extend(incoming.tool_messages)
                elif incoming.has_content() or incoming.is_tool_call():
                    state.messages.append(incoming)
                result = invoke('user_simulator', state.system_messages + state.flip_roles(), self.tools)
                message = result['choices'][0]['message']
                return UserMessage(role='user', content=message.get('content'),
                                   tool_calls=tau_calls(result, 'user'), cost=0.0,
                                   usage=result['usage'], raw_data=result)

        original_get_response = env.get_response

        def audited_response(call):
            started = time.time_ns()
            response = original_get_response(call)
            finished = time.time_ns()
            eligible = (call.requestor == 'assistant' and not response.error and
                        call_counts.get(call.id) == 1 and
                        (domain != 'banking_knowledge' or call.name == 'KB_search'))
            event = {'source': 'benchmark_native' if call.requestor == 'assistant'
                     else 'benchmark_user_tool', 'kind': 'tool_result', 'domain': domain,
                     'task_id': task.id, 'requestor': call.requestor,
                     'tool_name': call.name, 'arguments': call.arguments,
                     'result': response.content, 'error': response.error,
                     'call_id': call.id, 'started_ns': started, 'finished_ns': finished,
                     'call_count': call_counts.get(call.id), 'eligible': eligible}
            events.append(event)
            out.write(json.dumps(event, ensure_ascii=False) + '\n')
            out.flush()
            return response

        env.get_response = audited_response
        agent = AuditedAgent(tools=env.get_tools(), domain_policy=env.get_policy(), llm='dsv4')
        user_tools = (env.get_user_tools(include=task.user_tools or [])
                      if env.user_tools is not None else None)
        user = AuditedUser(llm='dsv4', instructions=task.user_scenario,
                           tools=user_tools or None)
        orchestrator = Orchestrator(domain=domain, agent=agent, user=user,
                                    environment=env, task=task, max_steps=max_steps,
                                    max_errors=3, seed=0, timeout=240)
        orchestrator.initialize()
        while not orchestrator.done:
            orchestrator.step()
            orchestrator._check_termination()
            if any(event['eligible'] for event in events):
                incoming = orchestrator.message
                if isinstance(incoming, MultiToolMessage):
                    orchestrator.agent_state.messages.extend(incoming.tool_messages)
                elif isinstance(incoming, ToolMessage):
                    orchestrator.agent_state.messages.append(incoming)
                break
        messages = to_litellm_messages(orchestrator.agent_state.system_messages +
                                       orchestrator.agent_state.messages)
    require(any(e['eligible'] for e in events),
            f'No successful single native {"KB_search" if domain == "banking_knowledge" else "tool"} call')
    indexes = {call['id']: index for index, message in enumerate(messages)
               for call in (message.get('tool_calls') or [])}
    for event in events:
        event['assistant_index'] = indexes.get(event['call_id'])
    require(all(e['assistant_index'] is not None for e in events if e['eligible']),
            'Eligible tool call missing from agent conversation')
    return {'domain': domain, 'task_id': task.id, 'messages': messages, 'events': events,
            'retrieval': 'offline_bm25_top_k_3' if domain == 'banking_knowledge' else None,
            'protocol': 'tau3_v1.0.1_user_simulator_orchestrator'}


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
    call = next(e for e in episode['events'] if e['eligible'])
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


def complete_fragments(messages: list[dict]) -> list[list[dict]]:
    """Complete agent-visible prefixes, with no unanswered tool call."""
    pending, fragments = set(), []
    for index, message in enumerate(messages[1:], 1):
        if message['role'] == 'assistant':
            pending.update(call['id'] for call in message.get('tool_calls') or [])
        elif message['role'] == 'tool':
            pending.discard(message['tool_call_id'])
        if not pending and message['role'] in ('assistant', 'tool'):
            fragments.append(messages[1:index + 1])
    return fragments


def select_episodes(candidates: dict[str, list[dict]], lengths: dict[str, int]) -> list[dict]:
    selected = []
    for domain in DOMAINS:
        ordered = sorted(candidates[domain], key=lambda e: (lengths[domain + ':' + e['task_id']], e['task_id']))
        short = [e for e in ordered if lengths[domain + ':' + e['task_id']] <= 8192 - TOKEN_TOLERANCE]
        medium = [e for e in ordered if lengths[domain + ':' + e['task_id']] <= 32768 - TOKEN_TOLERANCE]
        require(len(short) >= 12 and len(medium) >= 48,
                f'Insufficient {domain} episode lengths: {len(short)} fit 8K, {len(medium)} fit 32K')
        first = short[:12]
        used = {e['task_id'] for e in first}
        selected.extend(first + [e for e in medium if e['task_id'] not in used][:36])
    return selected


def make_pairs(episodes: list[dict], output: Path, tokenizer_sha256: str) -> dict:
    require(len(episodes) == 96, 'Expected 48 real episodes per domain')
    by_domain = {d: [e for e in episodes if e['domain'] == d] for d in DOMAINS}
    pairs, audit = [], []
    for domain in DOMAINS:
        corpus = by_domain[domain]
        require(len(corpus) == 48, 'Missing domain episodes')
        fragments = {e['task_id']: complete_fragments(e['messages']) for e in corpus}
        policy_lengths = {e['task_id']: len(api_tokens(e['messages'][:1])) for e in corpus}
        fragment_lengths = {e['task_id']: [(len(api_tokens(e['messages'][:1] + fragment))
                                            - policy_lengths[e['task_id']], fragment)
                                           for fragment in fragments[e['task_id']]] for e in corpus}
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
                fitting = [(length, e, fragment) for e in remaining
                           for length, fragment in fragment_lengths[e['task_id']]
                           if length <= gap + TOKEN_TOLERANCE // 2]
                if not fitting:
                    break
                _, previous, fragment = max(fitting, key=lambda item: item[0])
                remaining.remove(previous)
                prefix.extend(fragment)
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
    candidates = {domain: [] for domain in DOMAINS}
    lengths = {}
    rejected = []
    attempted = {domain: 0 for domain in DOMAINS}
    for domain in DOMAINS:
        for task in sorted(provenance['tasks'][domain], key=lambda t: t.id):
            attempted[domain] += 1
            path = output / f'episode-{domain}-{task.id}.jsonl'
            try:
                episode = collect_episode(domain, task, path)
                candidates[domain].append(episode)
                lengths[domain + ':' + task.id] = len(api_tokens(episode['messages']))
            except (ValueError, RuntimeError, KeyError, TypeError) as error:
                rejected.append({'domain': domain, 'task_id': task.id, 'reason': str(error),
                                 'episode_log': str(path)})
            finally:
                write_json(output / 'rejected-episodes.json', rejected)
                write_json(output / 'collection-summary.json', {'attempted': attempted,
                    'eligible': {d: len(candidates[d]) for d in DOMAINS},
                    'rejected': len(rejected), 'banking_retrieval': 'bm25_top_k_3',
                    'simulator': 'tau3_v1.0.1_on_local_dsv4'})
    episodes = select_episodes(candidates, lengths)
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
