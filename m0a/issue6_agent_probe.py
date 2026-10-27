"""Seeded-customer tool capability probe, isolated from a failing GLM user simulator."""
import argparse
import json
from pathlib import Path
from .issue6_workload import write
from .tau3_workload import verify_tau
from .run_requests import request
from .chat_generation import ChatGenerationError, generate_chat


def run(args):
    if not 0 < args.max_tokens <= args.max_tokens_ceiling:
        raise ValueError('Require 0 < max_tokens <= max_tokens_ceiling')
    args.output.mkdir(parents=True, exist_ok=False)
    source = verify_tau(args.tau_source)
    tasks = {(domain, task.id): task for domain, entries in source['tasks'].items() for task in entries}
    from tau2.domains.retail.environment import get_environment as retail_env
    from tau2.domains.banking_knowledge.environment import get_environment as bank_env
    from tau2.data_model.message import ToolCall
    results = []
    for material in (json.loads(line) for line in args.materials.read_text().splitlines()):
        if material['variant'] != 'original':
            continue
        domain, task_id = material['task_id'].split(':', 1)
        task = tasks[(domain, task_id)]
        env = (retail_env() if domain == 'retail' else
               bank_env(retrieval_variant='bm25', retrieval_kwargs={'top_k': 3}, task=task))
        messages, calls, successful, reads = list(material['messages']), 0, 0, 0
        log = args.output / (domain + '-' + task_id + '.jsonl')
        status = 'max_steps'
        generation_failure = None
        with log.open('w', encoding='utf-8') as stream:
            for step in range(args.max_steps):
                payload = {'model': 'glm-tiny', 'messages': messages, 'tools': material['tools'],
                    'tool_choice': 'auto', 'max_tokens': args.max_tokens, 'temperature': 0, 'seed': 7,
                    'return_token_ids': True, 'chat_template_kwargs': {'reasoning_effort': 'low'}}
                try:
                    def audit(current, response, attempt):
                        stream.write(json.dumps({'step': step, 'request': current,
                            'response': response, 'generation_attempt': attempt}) + '\n')
                        stream.flush()
                    response = generate_chat(payload,
                        request_fn=lambda current: request(current, url=args.url),
                        max_tokens_ceiling=args.max_tokens_ceiling, audit=audit)
                except ChatGenerationError as error:
                    generation_failure = error.details
                    status = error.details['reason']
                    break
                except Exception as error:
                    stream.write(json.dumps({'error': f'{type(error).__name__}: {error}', 'request': payload}) + '\n')
                    status = 'request_failed'
                    break
                if any(m['role'] == 'tool' for m in messages):
                    reads += 1
                message = response['choices'][0]['message']
                tool_calls = message.get('tool_calls') or []
                if not tool_calls:
                    status = 'final_text_without_tool' if message.get('content') else 'no_usable_action'
                    break
                messages.append({'role': 'assistant', 'content': message.get('content'), 'tool_calls': tool_calls})
                for call in tool_calls:
                    calls += 1
                    try:
                        tool = ToolCall(id=call['id'], name=call['function']['name'],
                            arguments=json.loads(call['function']['arguments']), requestor='assistant')
                        result = env.get_response(tool)
                        successful += not result.error
                        body = result.content
                    except Exception as error:
                        body = f'Tool error: {type(error).__name__}: {error}'
                    messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': body})
                    stream.write(json.dumps({'kind': 'tool_result', 'call': call, 'result': body}) + '\n')
                    stream.flush()
        record = {'task_id': material['task_id'], 'status': status, 'agent_tool_calls': calls,
                  'successful_agent_tool_calls': successful, 'post_tool_model_decisions': reads,
                  'probe_kind': 'scenario-disclosed seeded customer; no benchmark reward',
                  'actual_model_decisions': True, 'forced_tool_actions': False,
                  'generation_failure': generation_failure,
                  'generation_config': {'max_tokens': args.max_tokens,
                                        'max_tokens_ceiling': args.max_tokens_ceiling}}
        results.append(record)
        write(args.output / 'summary.json', results)
        print(json.dumps(record), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tau-source', type=Path, required=True)
    parser.add_argument('--materials', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:8866/v1/chat/completions')
    parser.add_argument('--max-steps', type=int, default=4)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--max-tokens-ceiling', type=int, default=8192)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
