"""Bounded retries for truncated chat generation, with every attempt audited."""
from __future__ import annotations

import time


class ChatGenerationError(RuntimeError):
    """The server replied, but no complete benchmark message can be consumed."""
    def __init__(self, reason, attempts):
        self.details = {'reason': reason, 'attempts': attempts}
        last = attempts[-1]
        super().__init__(f"{reason}: finish_reason={last['finish_reason']}, "
                         f"max_tokens={last['max_tokens']}, attempts={len(attempts)}")


def generate_chat(payload, *, request_fn, max_tokens_ceiling=None, audit=None):
    """Retry the unchanged conversation at 4x budget only after truncation.

    Never turn reasoning into visible content or invent tool actions. A partial
    message is not passed to the benchmark, even if it contains some text.
    """
    initial = payload.get('max_tokens')
    ceiling = initial if max_tokens_ceiling is None else max_tokens_ceiling
    if type(initial) is not int or type(ceiling) is not int or not 0 < initial <= ceiling:
        raise ValueError('Require 0 < max_tokens <= max_tokens_ceiling')
    attempts, budget = [], initial
    while True:
        current = dict(payload, max_tokens=budget)
        started = time.time_ns()
        response = request_fn(current)
        finished = time.time_ns()
        choice = response['choices'][0]
        message = choice['message']
        usable = bool((message.get('content') or '').strip() or message.get('tool_calls'))
        attempt = {'attempt': len(attempts) + 1, 'max_tokens': budget,
                   'finish_reason': choice.get('finish_reason'),
                   'completion_tokens': response.get('usage', {}).get('completion_tokens'),
                   'has_content': bool((message.get('content') or '').strip()),
                   'tool_calls': len(message.get('tool_calls') or []),
                   'started_ns': started, 'finished_ns': finished}
        attempts.append(attempt)
        if audit is not None:
            audit(current, response, attempt)
        if choice.get('finish_reason') == 'length':
            if budget < ceiling:
                budget = min(ceiling, budget * 4)
                continue
            raise ChatGenerationError('generation_budget_exhausted', attempts)
        if not usable:
            raise ChatGenerationError('empty_model_message', attempts)
        return response
