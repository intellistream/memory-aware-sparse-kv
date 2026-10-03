# Current-Pod DeepSeek service and trace replay

This runbook applies to the current eight-910B2 Pod. The weight mount is read-only at
`/models/DeepSeek-V4-Flash-W8A8`; all code, supervisor files, logs and run artifacts live under
`/root`. A Pod rebuild does not automatically restart the service.

## Repeatable service startup

After the Git bundle has been checked out under `/root/memecho-deploy-<run-id>`, run:

```bash
cd /root/memecho-deploy-<run-id>
python3 scripts/serve_dsv4_pod.py --execute \
  --root /root/memory-aware-sparse-kv \
  --model-dir /models/DeepSeek-V4-Flash-W8A8
```

The script checks all eight NPUs, `/dev/shm`, the saved model metadata SHA-256 hashes,
70 indexed weight files, installed vLLM/Ascend versions and port 8900. It starts the fixed
TP=8, W8A8, MTP, DSA CP command through `/root/memory-aware-sparse-kv/runtime/deepseek-pod/supervisord.conf`.
It verifies `/health`, `/v1/models` and a short generation request. The saved service command,
environment allowlist, configuration hash and process identity are in the same runtime directory.
`supervisorctl -c /root/memory-aware-sparse-kv/runtime/deepseek-pod/supervisord.conf status memecho-deepseek`
shows the current process.

## Validation launch

From the clean, committed PR branch on the local machine:

```bash
python3 scripts/launch_deepseek_validation.py --execute --runtime pod \
  --model-dir /models/DeepSeek-V4-Flash-W8A8 \
  --validation-mode trace-replay --host hust \
  --remote-root /root/memory-aware-sparse-kv
```

The launcher transfers a SHA-256 checked Git bundle and an exact worker snapshot. It clones
the bundle into a new `/root/memecho-deploy-<run-id>` directory; it does not alter the recovered
branch or untracked recovery files. The worker holds an exclusive lock, saves the supervised
service identity and configuration, then pauses it. Trace-off and trace-on each require 48 valid
requests. The trace source is copied from the installed vLLM Ascend package into the run directory,
patched with `m0a/vllm-ascend-trace.patch`, checked for import origin and hashed before the service
is paused. An independent watchdog stops only processes bearing the exact run marker and restores
the supervised service after failure, timeout or worker death. The local guardian verifies each
checkpoint and final artifact hashes. CPU replay runs after restoration.

The final `report.json` must say `engineering_validated`, with `strict_output_acceptance` set to
`not_qualified`; output differences remain in the phase files. The four replay reports are
`replay-{per_rank,aggregate}-{64,128}mib/report.json`. This is offline selected-set replay,
not an online KV offload or performance measurement. The model mount has no readable commit marker
or saved per-shard hashes, so metadata agreement and complete indexed shard sizes do not prove
every weight byte matches the original download.

If all 48+48 requests and native rows were archived but a CPU validator needs a
correction, keep the capture and its failed status unchanged. From a new checked
out validation commit on the Pod, run:

```bash
python3 scripts/resume_deepseek_pod_cpu.py \
  --root /root/memory-aware-sparse-kv --run-id <capture-run-id>
```

This accepts only the archived `Invalid CP chunk range` failure, checks the
captured inputs against `local-archive/capture-checksums.json`, and writes a
separate `postprocess-cp-padding-v1` directory with the capture and validator
commits, four CPU replays, report and SHA-256 manifest. It checks the supervised
service before and after and never pauses it. Report the capture failure and
postprocess result together; the postprocess is not a new online capture.

To finish an already running CPU continuation without an interactive session,
start `scripts/finalize_deepseek_pod.py` detached on the local PR checkout with
`--run-id`, `--host`, `--remote-root`, and `--deployment` (the independent Pod
checkout path). It watches the remote manifest, downloads and checks every
postprocess file, verifies the supervised service, writes
`docs/deepseek-pod-validation-result.md`, and pushes the result commit to the
draft PR branch. Read `automation-status.json` and `automation.log` in the
local run directory for completion or errors.
