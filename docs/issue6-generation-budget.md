# GLM tiny generation budgets and empty benchmark messages

The model server and the benchmark message protocol have separate success
conditions. An HTTP 200 response containing only a `reasoning` field is not a
usable user utterance or assistant tool action. Do not copy that field into
`content`, invent a customer reply, force a tool action, or count a partial
generation as a successful benchmark turn.

`issue6_workload` and `issue6_agent_probe` now accept an initial `--max-tokens`
and a `--max-tokens-ceiling`. On `finish_reason=length`, the adapter resubmits
the unchanged conversation with four times the budget, capped at the ceiling.
Every request and response is logged with its own timing, budget, finish reason,
and attempt number. A later attempt is a fresh generation, not a continuation of
the truncated reasoning. The retry does not alter temperature, tools, or seed.

The defaults are 512 -> 2048 -> 8192 tokens. Set both options to the same value
for a single-budget comparison. The workload's `--episode-timeout` defaults to
900 seconds, so the previous 240-second episode timeout does not prematurely
end a longer generation. The HTTP request timeout remains 900 seconds.

At the ceiling, a truncated response raises `ChatGenerationError` with reason
`generation_budget_exhausted`, even if some partial visible text is present.
A non-truncated response with neither visible text nor tool calls raises
`empty_model_message`. This happens before constructing `UserMessage` or
`AssistantMessage`; the workload records the structured failure and continues
to the next independent task. The agent probe also records the structured
failure. This fixes the misleading empty-message adapter crash, but cannot
make the model produce a valid utterance or tool call.

Examples, after starting the existing loopback GLM server in the fixed native
environment and activating the fixed tau3 environment:

```bash
python -m m0a.issue6_workload --tau-source "$TAU3_SOURCE" \
  --tasks-per-domain 6 --max-tokens 2048 --max-tokens-ceiling 8192 \
  --episode-timeout 900 --output m0a/runs/budget-workload-NEW
python -m m0a.issue6_agent_probe --tau-source "$TAU3_SOURCE" \
  --materials m0a/runs/budget-workload-NEW/prompt-material.jsonl \
  --max-tokens 2048 --max-tokens-ceiling 8192 \
  --output m0a/runs/budget-agent-NEW
python -m unittest m0a.test_chat_generation m0a.test_tau3_workload -v
```

The pinned model is a development checkpoint. Its publisher describes random
initialization followed by training on a small toy text dataset, with reduced
layer and expert counts. Preserving the sparse-attention architecture is useful
for operator and trace checks; it does not establish instruction-following or
tool-use ability. See the [publisher's model card](https://huggingface.co/inference-optimization/GLM-5.3-0.6B-A0.4B#creation-process).
Neither more generated tokens nor a nonempty response alone establishes that
the benchmark task was performed correctly.
