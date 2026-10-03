#!/usr/bin/env python3
"""Single-NPU engineering validation, immutable checkpoints and offline replay."""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import signal
import socket
import subprocess
import threading
import time
import traceback
import urllib.error
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    from .contracts import load_jsonl, resolve_api_request_id, sha256_file, validate_pair_set, validate_trace
    from .import_workloads import common_prefix_length, make_chat_tokenizer, sha_ids
    from .model_profiles import load_profile, validate_pair_policy
    from .orchestrate import merge_ranges
    from .preflight import check
    from .run_requests import OPENER, payload_for, request, signature
    from .transition_replay import read_trace, validate_sidecar
    from .working_set import require
except ImportError:
    from contracts import load_jsonl, resolve_api_request_id, sha256_file, validate_pair_set, validate_trace
    from import_workloads import common_prefix_length, make_chat_tokenizer, sha_ids
    from model_profiles import load_profile, validate_pair_policy
    from orchestrate import merge_ranges
    from preflight import check
    from run_requests import OPENER, payload_for, request, signature
    from transition_replay import read_trace, validate_sidecar
    from working_set import require

IMAGE = 'memecho/m0a-glm53-trace:v3'
IMAGE_ID = 'sha256:8506acf12eb6f248758317414d2ed9b01db40f4b04d6abf5e853427aa8b57f50'
WINDOW = 32
HEARTBEAT_SECONDS = 15
TOTAL_SECONDS = 6 * 60 * 60
SYNC_SECONDS = 900


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def extend_pairs(source, chat_ids, profile):
    """Keep original identities/boundaries; append one identical neutral suffix."""
    validate_pair_set(source)
    validate_pair_policy(source, profile)
    require(len(source['pairs']) == 12, 'Expected exactly 12 source pairs')
    result = copy.deepcopy(source)
    for pair in result['pairs']:
        original = {v: chat_ids(pair[v]['prompt']) for v in ('event', 'control')}
        boundary = pair['event']['boundary_position']
        for v, ids in original.items():
            item = pair[v]
            require(len(ids) == item['prompt_tokens_expected'] and
                    sha_ids(ids) == item['prompt_token_ids_sha256_expected'], 'Source token hash/count mismatch')
        require(common_prefix_length(*original.values()) == boundary, 'Source common prefix mismatch')
        selected = None
        for count in range(1, 65):
            for separator in ('\n', ' ', '\n\n', ''):
                suffix = separator + 'Neutral archive observation.\n' * count
                ids = {v: chat_ids(pair[v]['prompt'] + suffix) for v in original}
                if (len(ids['event']) == len(ids['control']) and min(map(len, ids.values())) - boundary >= WINDOW
                        and common_prefix_length(*ids.values()) == boundary):
                    selected = suffix, ids
                    break
            if selected:
                break
        require(selected is not None, 'Cannot preserve equal lengths with common neutral suffix')
        suffix, ids = selected
        require(common_prefix_length(*ids.values()) == boundary, 'Suffix changed event boundary')
        for v in original:
            require(ids[v][:boundary] == original[v][:boundary], 'Suffix changed original prefix')
            pair[v].update(prompt=pair[v]['prompt'] + suffix, prompt_tokens_expected=len(ids[v]),
                           prompt_token_ids_sha256_expected=sha_ids(ids[v]), prompt_token_ids=ids[v])
        pair['neutral_suffix'] = suffix
        # Older synthetic corpora have no source_trace_id: retain the original episode as its source identity.
        pair.setdefault('source_trace_id', pair['workload_id'] + ':' + pair['episode_id'])
    result.update(synthetic=True, validation_window_tokens=WINDOW)
    validate_pair_set(result)
    validate_pair_policy(result, profile)
    return result


def validate_api_result(result, item, previous_signature=None, expected_signature=None):
    ids = result.get('prompt_token_ids')
    usage = result.get('usage', {})
    require(isinstance(ids, list) and ids == item['prompt_token_ids'] and
            len(ids) == usage.get('prompt_tokens') == item['prompt_tokens_expected'] and
            sha_ids(ids) == item['prompt_token_ids_sha256_expected'], 'API prompt token hash/IDs/count mismatch')
    output = signature(result)
    generated = output.get('token_ids')
    require(isinstance(generated, list) and generated and all(type(t) is int and t >= 0 for t in generated),
            'Accurate generated token IDs required')
    require(len(generated) == usage.get('completion_tokens') and len(generated) <= 32, 'Generated token count mismatch')
    require(output.get('finish_reason') in {'stop', 'length'}, 'Unexpected output finish reason')
    if previous_signature is not None:
        require(output == previous_signature, 'Repeated output signature differs')
    if expected_signature is not None:
        require(output == expected_signature, 'Trace-on/off output signature differs')
    return output


def window_ranges(pairs):
    return merge_ranges([(p[v]['boundary_position'] - 1, p[v]['boundary_position'] + WINDOW - 1)
                         for p in pairs['pairs'] for v in ('event', 'control')])


def validate_native_trace(raw_dir, output_dir, responses, ranges, profile):
    """Validate all captured rows; export only each request's complete window."""
    by_request = {r['request_id']: r for r in responses}
    require(len(by_request) == 48, 'Expected 48 distinct trace-on requests')
    coverage, warm, repeat = defaultdict(dict), {}, {}
    output_dir.mkdir()
    files = list(Path(raw_dir).glob('rank*.jsonl'))
    require(len(files) == 1 and files[0].name == 'rank0.jsonl', 'Expected NPU0 rank0 trace only')
    raw_count = kept = 0
    with (output_dir / 'rank0.jsonl').open('w') as out:
        seen = set()
        for row in load_jsonl(files[0]):
            validate_trace(row, contract=profile)
            req = resolve_api_request_id(row['request_id'], set(by_request))
            require(req is not None, 'Unlinked native request')
            response = by_request[req]
            pos = row['prompt_position']
            require(row['rank'] == 0 and row['run_id'] == response['run_id'] and
                    row['request_context_len'] == response['prompt_tokens'], 'Native request/rank/context mismatch')
            require(any(a <= pos <= b for a, b in ranges), 'Unexpected captured position')
            layer = row['layer']
            match = re.fullmatch(r'model\.layers\.([0-5])(?:\.self_attn\.attn)?', layer)
            require(match is not None, 'Unexpected native layer identity')
            source = 'computed' if int(match[1]) < 3 else 'reused'
            require(row['selection_source'] == source and row['native_operator'] ==
                    ('torch_npu.npu_lightning_indexer' if source == 'computed' else 'index_cache_reuse'), 'Computed/reused layer mismatch')
            key = (req, layer, pos)
            require(key not in seen, 'Duplicate native layer/position')
            seen.add(key)
            raw_count += 1
            start = response['boundary_position']
            if not start - 1 <= pos < start + WINDOW:
                continue
            coverage[(req, pos)][layer] = source
            if pos == start - 1:
                group = (response['pair_id'], response['repetition'], layer)
                blocks = set(row['logical_block_ids']) - {-1}
                if group in warm:
                    require(blocks == warm[group], 'Paired prefix selected blocks differ')
                warm[group] = blocks
            group = (response['pair_id'], response['variant'], layer, pos)
            if group in repeat:
                require(row['raw_selected_ids'] == repeat[group], 'Repeated native selected IDs differ')
            repeat[group] = row['raw_selected_ids']
            out.write(json.dumps(row, separators=(',', ':')) + '\n')
            kept += 1
    for response in responses:
        for pos in range(response['boundary_position'] - 1, response['boundary_position'] + WINDOW):
            sources = coverage[(response['request_id'], pos)]
            require(len(sources) == 6 and list(sources.values()).count('computed') == 3 and
                    list(sources.values()).count('reused') == 3, 'Incomplete six-layer 32-token window')
    return {'raw_rows': raw_count, 'window_rows': kept, 'requests': 48, 'positions_per_request': 33,
            'layers': 6, 'computed': 3, 'reused': 3, 'causal_scope_checked': True,
            'repeat_selected_ids_identical': True, 'pair_prefix_blocks_identical': True}


def export_sidecar(responses, pairs, profile):
    lookup = {p['pair_id']: p for p in pairs['pairs']}
    data = {'schema_version': 1, 'synthetic': True, 'selection_provenance': 'real_npu_native',
            'online_kv_restore': False, 'snapshot_provenance': 'logical_prefix_inventory_from_exact_API_tokens',
            'split_evidence': '8K train / 32K engineering evaluation; no real-task generalization evidence',
            'snapshots': [], 'events': []}
    for row in responses:
        pair = lookup[row['pair_id']]
        ids, start = row['prompt_token_ids'], row['boundary_position']
        stem = row['request_id']
        common = {'session_id': 'session:' + pair['pair_id'], 'model_id': profile['model_id'],
                  'model_revision': profile['model_revision'], 'tokenizer_sha256': pairs['tokenizer_json_sha256']}
        for label, tokens, version, sequence, generated in (
                ('prefix', ids[:start], 'prefix', 1, start),
                ('current', ids, row['variant'], 2, start),
                ('post', ids, row['variant'], 2, start + WINDOW)):
            snapshot = dict(common, snapshot_id=stem + ':' + label, context_version=version,
                            sequence=sequence, token_ids=tokens, token_ids_sha256=sha_ids(tokens), generated_tokens=generated,
                            regions=[{'start': 0, 'end': min(start, len(tokens)), 'type': 'other', 'created_at': 1}],
                            lanes=[{'rank': 0, 'layer': f'model.layers.{i}.self_attn.attn', 'kv_kind': 'kv_token_position',
                                    'generated_ids': list(range(generated))} for i in range(6)])
            if len(tokens) > start:
                snapshot['regions'].append({'start': start, 'end': len(tokens), 'type': 'other', 'created_at': 2})
            data['snapshots'].append(snapshot)
        require(pair['context_target'] in {8192, 32768}, 'Unexpected split context')
        data['events'].append({'event_id': stem, 'request_id': stem, 'run_id': row['run_id'],
            'previous_snapshot_id': stem + ':prefix', 'snapshot_id': stem + ':current', 'post_window_snapshot_id': stem + ':post',
            'event_type': pair['event_type'] if row['variant'] == 'event' else 'no_event', 'previous_state': 'other',
            'event_position': start, 'resume_position': start, 'workload_id': pair['workload_id'],
            'episode_id': pair['episode_id'], 'source_trace_id': pair['source_trace_id'], 'pair_id': pair['pair_id'],
            'variant': row['variant'], 'repetition': row['repetition'], 'trajectory_id': stem,
            'split': 'train' if pair['context_target'] == 8192 else 'eval'})
    return data


def validate_layout(config, mla_source, sfa_source):
    require(config.get('dtype', config.get('torch_dtype')) == 'bfloat16' and
            config.get('kv_lora_rank') == 512 and config.get('qk_rope_head_dim') == 64 and
            config.get('num_hidden_layers') == 6 and config.get('indexer_types') == ['full'] * 3 + ['shared'] * 3 and
            not config.get('quantization_config'), 'Fixed BF16 MLA layout mismatch')
    require('self.head_size = kv_lora_rank + qk_rope_head_dim' in mla_source and
            'self.num_kv_heads = 1' in mla_source and
            'dtype=kv_cache_dtype' in mla_source and 'return (num_blocks, block_size, num_kv_heads, head_size)' in sfa_source,
            'MLA cache source contract mismatch')
    return {'dtype': 'bfloat16', 'kv_cache_dtype': 'auto', 'kv_lora_rank': 512, 'qk_rope_head_dim': 64,
            'element_bytes': 2, 'native_kv_payload_bytes_per_layer': 1152, 'formula': '(512 + 64) * 2',
            'scope': 'MLA attention payload only; indexer keys, scales, allocator metadata and DMA overhead excluded'}


class Worker:
    def __init__(self, root, run_id):
        require(re.fullmatch(r'single_npu_\d{8}T\d{6}Z_[0-9a-f]{8}', run_id), 'Invalid run ID')
        self.root, self.run_id = Path(root), run_id
        self.directory = self.root / 'm0a/runs' / run_id
        self.code_root = Path(__file__).resolve().parents[1]
        self.env = dict(os.environ, DOCKER_HOST='unix://' + str(self.root / 'runtime/docker.sock'))
        self.container = 'memecho-' + run_id
        self.owned = set()
        self.state = {'schema_version': 1, 'run_id': run_id, 'device': 0, 'worker_pid': os.getpid(),
                      'started_at': utc(), 'status': 'running', 'stage': 'bootstrap', 'error': None}
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.checkpoint_number = 0

    def update(self, **values):
        with self.lock:
            self.state.update(values, heartbeat=utc())
            write_json(self.directory / 'status.json', self.state)

    def heartbeat(self):
        while not self.stop.wait(HEARTBEAT_SECONDS):
            self.update()

    def command(self, args, *, timeout=900, output=None, env=None):
        print('Command:', args, flush=True)
        if output:
            with Path(output).open('w') as out:
                subprocess.run(args, env=env or self.env, stdout=out, stderr=subprocess.STDOUT, timeout=timeout, check=True)
            return ''
        return subprocess.check_output(args, env=env or self.env, timeout=timeout, text=True)

    def checkpoint(self, stage, paths):
        """No next stage until local hashes are verified and server ACK matches."""
        self.update(stage=stage + '_sync')
        name = f'{self.checkpoint_number:02d}-{stage}'
        self.checkpoint_number += 1
        records = []
        for path in sorted(set(map(Path, paths))):
            require(path.is_file() and path.is_relative_to(self.directory), 'Invalid checkpoint file')
            records.append({'path': str(path.relative_to(self.directory)), 'size': path.stat().st_size, 'sha256': sha256_file(path)})
        manifest = self.directory / 'checkpoints' / (name + '.json')
        write_json(manifest, {'run_id': self.run_id, 'stage': stage, 'created_at': utc(), 'files': records})
        digest = sha256_file(manifest)
        self.update(pending_checkpoint=name, pending_manifest_sha256=digest)
        deadline = time.monotonic() + SYNC_SECONDS
        while time.monotonic() < deadline:
            failure = self.directory / 'sync-failed.json'
            require(not failure.exists(), 'Local synchronization failed after three attempts')
            ack = self.directory / 'acks' / (name + '.json')
            if ack.exists():
                data = json.loads(ack.read_text())
                require(data.get('manifest_sha256') == digest and data.get('files') == records and
                        data.get('run_id') == self.run_id, 'Invalid local synchronization ACK')
                self.update(pending_checkpoint=None, last_synced_checkpoint=name)
                return
            time.sleep(1)
        raise TimeoutError('Local synchronization acknowledgment timed out')

    def cleanup(self):
        results = []
        for name in sorted(self.owned):
            try:
                result = subprocess.run(['docker', 'inspect', name], env=self.env, text=True, capture_output=True, timeout=30)
                if result.returncode:
                    # Absence is successful; an unavailable daemon is not.
                    require('No such' in result.stderr, 'Cannot verify container release: ' + result.stderr)
                    results.append({'container': name, 'released': True, 'already_absent': True})
                    continue
                metadata = json.loads(result.stdout)[0]
                require(metadata['Config'].get('Labels', {}).get('memecho.validation.run') == self.run_id,
                        'Refusing to stop a container without this task ownership label')
                self.command(['docker', 'rm', '-f', name], timeout=60)
                result = subprocess.run(['docker', 'inspect', name], env=self.env, capture_output=True, text=True, timeout=30)
                require(result.returncode != 0 and 'No such' in result.stderr, 'Container release not verified')
                results.append({'container': name, 'released': True})
            except Exception as error:
                results.append({'container': name, 'released': False, 'error': str(error)})
        write_json(self.directory / 'resource-release.json', {'at': utc(), 'containers': results})
        require(all(r['released'] for r in results), 'Owned container cleanup failed')

    def preflight(self):
        self.update(stage='preflight')
        npu = self.command(['npu-smi', 'info'])
        (self.directory / 'npu-before.txt').write_text(npu)
        usage = check((0,), 6000)
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 8900))
        manifest = json.loads((self.code_root / 'm0a/artifacts/glm53_tiny_model_manifest.json').read_text())
        model = self.root / 'models/GLM-5.3-0.6B-A0.4B'
        require((model / '.m0a_revision').read_text().strip() == manifest['revision'], 'Model revision mismatch')
        checks = {}
        for name, expected in manifest['files'].items():
            path = model / name
            require(path.stat().st_size == expected['size'] and sha256_file(path) == expected['sha256'], 'Model/tokenizer identity mismatch: ' + name)
            checks[name] = expected
        identity = self.command(['docker', 'image', 'inspect', IMAGE, '--format', '{{.Id}}']).strip()
        require(identity == IMAGE_ID, 'Fixed v3 image ID mismatch')
        config = json.loads((model / 'config.json').read_text())
        helper = self.container + '-prepare'
        self.owned.add(helper)
        self.command(['docker', 'create', '--name', helper, '--label', 'memecho.validation.run=' + self.run_id, IMAGE, 'true'])
        for dest, source in [('mla.py', '/vllm-workspace/vllm/vllm/model_executor/layers/attention/mla_attention.py'),
                             ('sfa.py', '/vllm-workspace/vllm-ascend/vllm_ascend/attention/sfa_v1.py')]:
            self.command(['docker', 'cp', helper + ':' + source, str(self.directory / dest)])
        layout = validate_layout(config, (self.directory / 'mla.py').read_text(), (self.directory / 'sfa.py').read_text())
        layout['source_evidence'] = []
        for name, needles in [('mla.py', ['self.head_size =', 'self.num_kv_heads = 1', 'dtype=kv_cache_dtype']),
                              ('sfa.py', ['return (num_blocks, block_size, num_kv_heads, head_size)'])]:
            lines = (self.directory / name).read_text().splitlines()
            layout['source_evidence'].append({'path': name, 'sha256': sha256_file(self.directory / name),
                'lines': [i + 1 for i, line in enumerate(lines) if any(n in line for n in needles)]})
        write_json(self.directory / 'layout.json', layout)
        write_json(self.directory / 'preflight.json', {'device': 0, 'hbm_used_mb': usage[0], 'port': 8900,
                   'image': IMAGE, 'image_id': identity, 'model_files': checks, 'at': utc()})
        self.cleanup()

    def prepare(self):
        self.update(stage='prepare_inputs')
        helper = self.container + '-prepare'
        self.command(['docker', 'run', '--rm', '--name', helper, '--label', 'memecho.validation.run=' + self.run_id,
                      '-v', str(self.root) + ':' + str(self.root), IMAGE, 'python3', str(Path(__file__).resolve()),
                      '--prepare', '--root', str(self.root), '--run-id', self.run_id],
                     output=self.directory / 'prepare.log')
        return json.loads((self.directory / 'pairs.json').read_text())

    def start_service(self, stage, ranges):
        self.update(stage=stage + '_startup')
        # Repeat NPU and port preflight before each service start.
        check((0,), 6000)
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 8900))
        env = dict(self.env, MEMECHO_ROOT=str(self.root), M0A_MODEL_DIR=str(self.root / 'models/GLM-5.3-0.6B-A0.4B'),
                   M0A_CONTAINER_NAME=self.container, M0A_OWNER_RUN_ID=self.run_id, M0A_NPU_ID='0', M0A_IMAGE=IMAGE,
                   RECREATE='0', VLLM_ASCEND_M0A_TRACE_DIR='', VLLM_ASCEND_M0A_TRACE_POSITIONS='', VLLM_ASCEND_M0A_RUN_ID='')
        if stage == 'trace_on':
            trace_dir = self.directory / 'raw-traces'
            trace_dir.mkdir()
            env.update(VLLM_ASCEND_M0A_TRACE_DIR=str(trace_dir), VLLM_ASCEND_M0A_TRACE_POSITIONS=','.join(f'{a}:{b}' for a, b in ranges),
                       VLLM_ASCEND_M0A_RUN_ID=self.run_id)
        self.owned.add(self.container)
        self.command(['bash', str(self.code_root / 'm0a/serve_glm53_tiny_trace.sh')], env=env,
                     output=self.directory / (stage + '-launcher.log'))
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            metadata = json.loads(self.command(['docker', 'inspect', self.container]))[0]
            require(metadata['State']['Running'], 'Validation service exited during startup')
            try:
                with OPENER.open('http://127.0.0.1:8900/health', timeout=10) as response:
                    if response.status == 200:
                        return
            except Exception:
                pass
            time.sleep(5)
        raise TimeoutError('Service health timeout (900 seconds)')

    def requests(self, stage, pairs, profile, baseline=None):
        self.update(stage=stage, completed_requests=0)
        records, stable = [], {}
        expected = {(r['pair_id'], r['variant'], r['repetition']): r['signature'] for r in baseline or []}
        with (self.directory / (stage + '-responses.jsonl')).open('w') as out, \
                (self.directory / (stage + '-api.jsonl')).open('w') as raw:
            for pair in pairs['pairs']:
                for repetition in range(2):
                    for variant in (('event', 'control') if repetition == 0 else ('control', 'event')):
                        item = pair[variant]
                        started = time.time_ns()
                        try:
                            result = request(payload_for(profile, item['prompt']))
                        except Exception as error:
                            failure = {'pair_id': pair['pair_id'], 'variant': variant, 'repetition': repetition,
                                       'started_ns': started, 'error': str(error)}
                            if isinstance(error, urllib.error.HTTPError):
                                failure.update(http_status=error.code, response_body=error.read().decode('utf-8', errors='replace'))
                            raw.write(json.dumps(failure, ensure_ascii=False) + '\n')
                            raw.flush()
                            raise
                        # Preserve even a response that fails acceptance.
                        raw.write(json.dumps({'pair_id': pair['pair_id'], 'variant': variant,
                            'repetition': repetition, 'response': result}, ensure_ascii=False) + '\n')
                        raw.flush()
                        key = (pair['pair_id'], variant)
                        sig = validate_api_result(result, item, stable.get(key), expected.get(key + (repetition,)))
                        stable[key] = sig
                        row = {k: pair[k] for k in ('pair_id', 'workload_id', 'episode_id', 'context_target', 'event_type', 'source_trace_id')}
                        row.update(schema_version=1, run_id=self.run_id, variant=variant, repetition=repetition,
                            boundary_position=item['boundary_position'], request_id=result['id'], prompt_token_ids=result['prompt_token_ids'],
                            prompt_token_ids_sha256=sha_ids(result['prompt_token_ids']), prompt_tokens=len(result['prompt_token_ids']),
                            signature=sig, completion_tokens=result['usage']['completion_tokens'], started_ns=started, finished_ns=time.time_ns())
                        out.write(json.dumps(row, ensure_ascii=False) + '\n')
                        out.flush()
                        records.append(row)
                        self.update(completed_requests=len(records))
        require(len(records) == 48, 'Incomplete request phase')
        return records

    def stop_service(self, stage):
        self.command(['docker', 'logs', self.container], output=self.directory / (stage + '-service.log'), timeout=60)
        self.cleanup()
        (self.directory / (stage + '-resource-release.json')).write_bytes(
            (self.directory / 'resource-release.json').read_bytes())
        self.command(['npu-smi', 'info'], output=self.directory / (stage + '-npu-after.txt'), timeout=30)

    def run(self):
        self.update()
        thread = threading.Thread(target=self.heartbeat, daemon=True)
        thread.start()
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(RuntimeError('Worker terminated')))
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Six-hour task limit exceeded')))
        signal.alarm(TOTAL_SECONDS)
        try:
            self.checkpoint('bootstrap', [self.directory / 'launch.json', self.directory / 'code-provenance.json',
                                       self.directory / 'implementation.bundle', self.directory / 'deployment.json'])
            self.preflight()
            pairs = self.prepare()
            ranges = window_ranges(pairs)
            profile = load_profile('glm53_tiny')
            self.checkpoint('inputs', [self.directory / n for n in ('preflight.json', 'layout.json', 'mla.py', 'sfa.py',
                                                                 'npu-before.txt', 'pairs.json', 'prepare.log')])
            self.start_service('trace_off', ranges)
            baseline = self.requests('trace_off', pairs, profile)
            self.stop_service('trace_off')
            self.checkpoint('trace_off', [self.directory / n for n in ('trace_off-responses.jsonl', 'trace_off-api.jsonl',
                'trace_off-launcher.log', 'trace_off-service.log', 'trace_off-npu-after.txt', 'trace_off-resource-release.json')])
            self.start_service('trace_on', ranges)
            responses = self.requests('trace_on', pairs, profile, baseline)
            # Release NPU before any CPU trace processing/replay.
            self.stop_service('trace_on')
            self.update(stage='trace_validation')
            evidence = validate_native_trace(self.directory / 'raw-traces', self.directory / 'traces', responses, ranges, profile)
            write_json(self.directory / 'trace-validation.json', evidence)
            sidecar = export_sidecar(responses, pairs, profile)
            snapshots, _ = validate_sidecar(sidecar, self.directory)
            read_trace(self.directory / 'traces', snapshots, sidecar['events'], synthetic=False)
            write_json(self.directory / 'sidecar.json', sidecar)
            self.checkpoint('npu_capture', [self.directory / n for n in ('trace_on-responses.jsonl', 'trace_on-api.jsonl',
                'trace_on-launcher.log', 'trace_on-service.log', 'trace_on-npu-after.txt', 'trace_on-resource-release.json',
                'raw-traces/rank0.jsonl', 'traces/rank0.jsonl', 'trace-validation.json', 'sidecar.json')])
            for capacity in (64, 128):
                self.update(stage=f'cpu_replay_{capacity}mib')
                config = {'schema_version': 1, 'capacity_bytes': capacity * 1024**2, 'prefetch_budget_bytes': 8 * 1024**2,
                          'kv_unit_bytes': {'kv_token_position': 1152}, 'max_tables': 256, 'bootstrap_draws': 1000, 'seed': 0}
                config_path = self.directory / f'cache-{capacity}mib.json'
                write_json(config_path, config)
                output = self.directory / f'replay-{capacity}mib'
                self.command(['python3', str(self.code_root / 'm0a/transition_replay.py'), '--trace-dir', str(self.directory / 'traces'),
                    '--sidecar', str(self.directory / 'sidecar.json'), '--config', str(config_path), '--source-root', str(self.directory),
                    '--output-dir', str(output)], timeout=TOTAL_SECONDS, output=self.directory / f'replay-{capacity}mib.log')
                self.checkpoint(f'replay_{capacity}mib', [config_path, output / 'events.jsonl', output / 'report.json',
                                                       self.directory / f'replay-{capacity}mib.log'])
            report = {'run_id': self.run_id, 'status': 'passed', 'evidence_level': 'synthetic single-NPU engineering validation',
                      'selection_provenance': 'real_npu_native', 'online_kv_restore': False, 'requests_per_phase': 48,
                      'window_tokens': 32, 'trace_validation': evidence, 'replay_reports': ['replay-64mib/report.json', 'replay-128mib/report.json'],
                      'limitations': ['Synthetic 8K/32K split is not real-task generalization evidence.',
                                      'Logical snapshots do not demonstrate online KV restoration.',
                                      'CPU byte replay does not measure DMA stalls or serving latency.',
                                      'Indexer storage and transfer overhead are excluded from the MLA payload byte model.']}
            write_json(self.directory / 'report.json', report)
            (self.directory / 'report.md').write_text('# 单卡 GLM 工程验证\n\n96 个请求、真实 NPU selected set、完整 32-token 窗口和输出等价校验通过。\n\n'
                '64/128 MiB 容量、每窗口 8 MiB 预算、七种策略和 native/page128 两种粒度的 precision/recall、有效/浪费字节、同步 recall、污染、规划 CPU 时间和配对区间见 replay-*/report.json。\n\n'
                '证据仅限合成单卡工程验证；未验证在线 KV 恢复、DMA stall 或真实任务泛化。\n')
            self.checkpoint('report', [self.directory / 'report.json', self.directory / 'report.md'])
            self.update(status='passed', stage='finished', finished_at=utc())
        except BaseException as error:
            traceback.print_exc()
            self.update(status='failed', stage='cleanup', error=str(error))
            try:
                if self.container in self.owned:
                    with (self.directory / 'failure-service.log').open('w') as log:
                        subprocess.run(['docker', 'logs', self.container], env=self.env, stdout=log,
                                       stderr=subprocess.STDOUT, timeout=30)
            except BaseException:
                traceback.print_exc()
            try:
                self.cleanup()
            except BaseException:
                traceback.print_exc()
            try:
                self.command(['npu-smi', 'info'], output=self.directory / 'npu-failure.txt', timeout=30)
            except BaseException:
                traceback.print_exc()
            self.update(stage='finished', finished_at=utc())
        finally:
            signal.alarm(0)
            self.stop.set()
            thread.join(timeout=1)
        return 0 if self.state['status'] == 'passed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/workspace/memecho'))
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--device', type=int, choices=[0], default=0)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        code = Path(__file__).resolve().parents[1]
        directory = args.root / 'm0a/runs' / args.run_id
        model = args.root / 'models/GLM-5.3-0.6B-A0.4B'
        profile = load_profile('glm53_tiny')
        chat = make_chat_tokenizer(profile_name='glm53_tiny', tokenizer_path=model / 'tokenizer.json', model_dir=model)
        source = code / 'm0a/workloads/glm53_tiny.synthetic.equalized.pairs.json'
        result = extend_pairs(json.loads(source.read_text()), chat, profile)
        result['source_file_sha256'] = sha256_file(source)
        write_json(directory / 'pairs.json', result)
        return 0
    return Worker(args.root, args.run_id).run()


if __name__ == '__main__':
    raise SystemExit(main())
