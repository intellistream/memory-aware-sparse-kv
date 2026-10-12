"""vLLM v1 logits processor for request-scoped exact continuations."""
from __future__ import annotations

import torch
from vllm.v1.sample.logits_processor.interface import LogitsProcessor, MoveDirectionality


class FixedContinuation(LogitsProcessor):
    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool):
        self.targets: dict[int, tuple[int, ...]] = {}
        self.outputs: dict[int, list[int]] = {}

    @classmethod
    def validate_params(cls, sampling_params):
        targets = (sampling_params.extra_args or {}).get("fixed_token_ids")
        if targets is not None and (not isinstance(targets, list) or len(targets) != 32 or
                                    any(type(value) is not int or value < 0 for value in targets)):
            raise ValueError("Expected exactly 32 fixed continuation token IDs")

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update):
        if batch_update is None:
            return
        for index in batch_update.removed:
            self.targets.pop(index, None)
            self.outputs.pop(index, None)
        for index, params, _prompt, output in batch_update.added:
            values = (params.extra_args or {}).get("fixed_token_ids")
            if values is None:
                self.targets.pop(index, None)
                self.outputs.pop(index, None)
            else:
                self.targets[index] = tuple(values)
                self.outputs[index] = output
        for source, target, direction in batch_update.moved:
            left = (self.targets.get(source), self.outputs.get(source))
            right = (self.targets.get(target), self.outputs.get(target))
            if left[0] is None:
                self.targets.pop(target, None)
                self.outputs.pop(target, None)
            else:
                self.targets[target], self.outputs[target] = left
            if direction == MoveDirectionality.SWAP:
                if right[0] is None:
                    self.targets.pop(source, None)
                    self.outputs.pop(source, None)
                else:
                    self.targets[source], self.outputs[source] = right
            else:
                self.targets.pop(source, None)
                self.outputs.pop(source, None)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        for index, targets in self.targets.items():
            if index >= logits.shape[0]:
                continue
            position = len(self.outputs[index])
            if position >= len(targets):
                raise RuntimeError("Fixed continuation exceeded 32 tokens")
            token = targets[position]
            if token >= logits.shape[1]:
                raise ValueError("Fixed token is outside the model vocabulary")
            logits[index].fill_(float("-inf"))
            logits[index, token] = 0
        return logits
