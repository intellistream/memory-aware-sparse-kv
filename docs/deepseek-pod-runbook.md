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

## Public τ³ workload validation

From this clean, committed branch, run the single launch command:

```bash
python3 scripts/launch_deepseek_validation.py --execute --runtime pod \
  --model-dir /models/DeepSeek-V4-Flash-W8A8 \
  --validation-mode trace-replay --workload tau3_v1.0.1 \
  --host hust --remote-root /root/memory-aware-sparse-kv
```

The launcher pins τ³-bench v1.0.1 to commit `fc0055dc4e0a316c3f83133267fbd6faaa770992`,
installs its locked dependencies in an isolated Pod environment, checks the service and
executes a real retail tool call. It then transfers the checked implementation and starts
a worker, an independent recovery watchdog, and a local SHA-256 synchronization guardian.
Exit code 0 means these processes have taken over and the first checkpoint is verified;
it does not mean the experiment has passed. The JSON printed by the launcher contains
`run_id`, `status_path`, `sync_status_path`, `final_status_path`, log paths and a recovery
command for use only if the guardian exits unexpectedly.

The worker collects 48 pairs: two domains × six event classes × 8K/32K contexts × two
independent chains. Each phase attempts 192 requests: two variants and two repetitions
per pair. The retail tools and offline BM25 banking search come from the pinned public
package. The workload adapter renders a user turn from the public task scenario and
chains real model/tool episodes; it does not run the official τ³ user simulator or grader.
The event and control branches use the same token budget within two tokens. Role-bearing
messages, actual tool results, token IDs, timestamps, source hashes and memory state
versions are retained. Incomplete coverage, tool failures, token mismatch, output drift or
selected-set drift fail the acceptance gate.

The worker first tests output stability using up to six fresh serving configurations. A
configuration is selected only after 20 stable diagnostic responses and a full 192-request
trace-off phase. Trace-on uses the same selected command plus the capture hook. The
original supervised service is restored before four same-budget CPU replays. The
`locality.json` report gives paired sequential-prefetch differences for each
domain/event/context and budget. The ECHO mechanism remains unqualified until indexer
scores and timed replay are available; these offline results are not online ECHO
performance measurements.

The local guardian synchronizes every checkpoint and final file by SHA-256. At termination,
`final-status.json` records separate experiment, output, hash, service and push results.
Only a complete result passes all gates. Success and failure both produce a lightweight
Git report/index commit and push; raw logs stay in the local and Pod run directories.

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
start `scripts/finalize_deepseek_pod.py` detached on the local checkout with
`--run-id`, `--host`, `--remote-root`, and `--deployment` (the independent Pod
checkout path). It watches the remote manifest, downloads and checks every
postprocess file, verifies the supervised service, writes
`docs/deepseek-pod-validation-result.md`, and pushes the result commit to the
current branch. Read `automation-status.json` and `automation.log` in the
local run directory for completion or errors.
