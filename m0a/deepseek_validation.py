#!/usr/bin/env python3
"""Eight-NPU DeepSeek capture, restoration barriers and lossless CPU replay."""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
from collections import defaultdict
from pathlib import Path

try:
    from .single_npu_validation import Worker as BaseWorker, WINDOW, TOTAL_SECONDS, utc, write_json, extend_pairs as append_pairs, validate_api_result
    from .contracts import load_jsonl, resolve_api_request_id, sha256_file, validate_trace
    from .equalize_synthetic_pairs import equalize
    from .import_workloads import make_chat_tokenizer, sha_ids
    from .model_profiles import load_profile
    from .preflight import check, get_hbm_usage
    from .run_requests import OPENER, request, payload_for
    from .transition_replay import validate_sidecar, read_trace
    from .working_set import require
except ImportError:
    from single_npu_validation import Worker as BaseWorker, WINDOW, TOTAL_SECONDS, utc, write_json, extend_pairs as append_pairs, validate_api_result
    from contracts import load_jsonl, resolve_api_request_id, sha256_file, validate_trace
    from equalize_synthetic_pairs import equalize
    from import_workloads import make_chat_tokenizer, sha_ids
    from model_profiles import load_profile
    from preflight import check, get_hbm_usage
    from run_requests import OPENER, request, payload_for
    from transition_replay import validate_sidecar, read_trace
    from working_set import require

IMAGE = 'memecho/m0a-trace:stage0-v3'
IMAGE_ID = 'sha256:a41da3a931b86b1c24d5acb24503e15d75bfe6ae6480bf082de89abe08220b64'
BASE_IMAGE_ID = 'sha256:cb25ae391cad9549f884b9ba57877e67b5a1da6f9200b870dde15b38df6696ba'
ORIGINAL = 'memecho-vllm-ascend'
LAYERS = tuple(f'model.layers.{i}.attn' for i in range(2, 43, 2))
ASCEND = '/vllm-workspace/vllm-ascend/'
SOURCE_PATHS = {
    'model.py': ASCEND+'vllm_ascend/models/deepseek_v4.py',
    'dsa_cp.py': ASCEND+'vllm_ascend/attention/context_parallel/dsa_cp.py',
    'cache_spec.py': ASCEND+'vllm_ascend/core/kv_cache_interface.py',
    'device_op.py': ASCEND+'vllm_ascend/device/device_op.py',
    'model_runner_v1.py': ASCEND+'vllm_ascend/worker/model_runner_v1.py',
    'compressor_kernel.h': ASCEND+'vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/compressor/arch32/compressor_block_vec_perf.h',
    'slot_mapping.h': ASCEND+'vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/compressor_metadata/compressor_metadata.h',
    'worker.py': ASCEND+'vllm_ascend/worker/worker.py',
    'layer.py': ASCEND+'vllm_ascend/models/layer/attention/layer.py',
    'dsa_v1.py': ASCEND+'vllm_ascend/attention/dsa_v1.py',
}


class TaskDeadline(BaseException):
    """The global deadline must bypass candidate-level rejection handling."""


class TaskTerminated(BaseException):
    """A service termination is never a retryable request/candidate failure."""


def terminate_worker(*_):
    raise TaskTerminated('Worker terminated by SIGTERM')


class DiagnosticComplete(Exception):
    pass


def stability_candidates(original, *, repair=False):
    """Cumulative experimental changes; never mutate the retained service."""
    command = list(original)
    def option(name, value=None):
        while name in command:
            index = command.index(name)
            del command[index:index + (1 if value is None else 2)]
        command.append(name)
        if value is not None:
            command.append(str(value))
    candidates = []
    changes = []
    def save(name):
        candidates.append({'id':name, 'command':list(command), 'changes':list(changes)})
    save('original_fresh')
    compilation = json.loads(command[command.index('--compilation-config')+1])
    compilation['cudagraph_mode'] = 'NONE'
    additional = json.loads(command[command.index('--additional-config')+1])
    additional['ascend_compilation_config']['enable_npugraph_ex'] = False
    option('--enforce-eager')
    option('--compilation-config', json.dumps(compilation))
    option('--additional-config', json.dumps(additional))
    changes.append('graphs_disabled')
    save('eager')
    additional['multistream_overlap_shared_expert'] = False
    option('--additional-config', json.dumps(additional))
    changes.append('shared_expert_multistream_disabled')
    save('eager_single_stream')
    index = command.index('--speculative-config')
    del command[index:index+2]
    changes.append('mtp_disabled')
    save('eager_single_stream_no_mtp')
    if '--async-scheduling' in command:
        command.remove('--async-scheduling')
    option('--no-async-scheduling')
    changes.append('async_scheduling_disabled')
    save('eager_single_stream_no_mtp_sync')
    option('--max-num-seqs', 1)
    changes.append('max_num_seqs_1')
    save('eager_single_stream_no_mtp_sync_seq1')
    if repair:
        changes.append('hccl_deterministic')
        save('eager_single_stream_no_mtp_sync_seq1_hccl')
        candidates[-1]['environment'] = {'HCCL_DETERMINISTIC':'true'}
        changes.append('npu_deterministic_level_1')
        save('eager_single_stream_no_mtp_sync_seq1_hccl_npu')
        candidates[-1]['environment'] = {'HCCL_DETERMINISTIC':'true', 'MEMECHO_DETERMINISTIC_LEVEL':'1'}
    return candidates


def output_difference(expected, actual):
    left, right = expected['token_ids'], actual['token_ids']
    index = next((i for i in range(max(len(left),len(right)))
                  if (left[i] if i<len(left) else None) != (right[i] if i<len(right) else None)), None)
    return {'first_differing_token_index':index,
            'expected_token':left[index] if index is not None and index<len(left) else None,
            'actual_token':right[index] if index is not None and index<len(right) else None,
            'expected_signature':expected, 'actual_signature':actual}


def extend_pairs(source, chat, profile):
    result = append_pairs(equalize(source, chat_ids=chat, profile=profile), chat, profile)
    result['original_source_sha256'] = sha_ids(source)
    for old, new in zip(source['pairs'], result['pairs']):
        new['original_identity'] = {k: old[k] for k in ('pair_id', 'workload_id', 'episode_id', 'context_target')}
        new['original_prompt_hashes'] = {v: old[v]['prompt_token_ids_sha256_expected'] for v in ('event', 'control')}
        require(new['event']['prompt_tokens_expected'] == new['control']['prompt_tokens_expected'], 'Strict equal length required')
    return result


def port_available(port=8900):
    """Ignore TIME_WAIT, but reject an actual listener even with SO_REUSEADDR."""
    with socket.socket() as connection:
        connection.settimeout(.2)
        if connection.connect_ex(('127.0.0.1', port)) == 0:
            return False
    try:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(('127.0.0.1', port))
        return True
    except OSError:
        return False


def layer_number(name):
    match = re.fullmatch(r'model\.layers\.(\d+)\.(?:self_attn\.)?attn', name)
    require(match is not None and int(match[1]) in range(2, 43, 2), 'Unexpected C4 DSA layer')
    return int(match[1])


def validate_cp(row):
    fields = ('cp_world_size', 'cp_local_start', 'cp_local_end', 'chunk_start_position',
              'chunk_token_count', 'chunk_context_len', 'query_global_index')
    require(all(type(row.get(k)) is int for k in fields), 'Missing exact CP/chunk metadata')
    count = row['chunk_token_count']
    width = (count+7)//8
    require(count > 0 and row['cp_world_size'] == 8 and 0 <= row['rank'] < 8, 'Invalid CP world/rank')
    require(row['cp_local_start'] == row['rank']*width and row['cp_local_end'] == (row['rank']+1)*width,
            'CP ownership interval mismatch')
    index = row['query_global_index']
    require(row['cp_local_start'] <= index < min(count, row['cp_local_end']) and
            row['prompt_position'] == row['chunk_start_position']+index, 'Wrong CP query owner/position')
    # DSA CP pads each chunk to a multiple of its eight ranks. The actual
    # context ends at chunk_context_len; the last 1-7 padded positions have no
    # query rows. Compare the padded end to that exact per-chunk length.
    context = row['chunk_context_len']
    require(row['chunk_start_position'] >= 0 and
            row['chunk_start_position'] < context <= row['request_context_len'] and
            row['chunk_start_position']+count == ((context+7)//8)*8 and
            row['prompt_position'] < context,
            'Invalid CP chunk range')


def validate_native_trace(raw_dir, output_dir, responses, ranges, profile, *, allow_selected_drift=False):
    lookup = {r['request_id']: r for r in responses}
    require(len(lookup) == len(responses) and responses, 'Duplicate or missing trace-on request IDs')
    files = sorted(Path(raw_dir).glob('rank*.jsonl'))
    require(files, 'No native traces')
    output_dir = Path(output_dir)
    output_dir.mkdir()
    coverage, repeat, warm, assignments = defaultdict(dict), {}, {}, defaultdict(dict)
    seen, count, kept, layers = set(), 0, 0, {}
    repeat_drift, prefix_drift, drift_examples = 0, 0, []
    for path in files:
        with (output_dir/path.name).open('w') as out:
            for row in load_jsonl(path):
                validate_trace(row, contract=profile)
                validate_cp(row)
                req = resolve_api_request_id(row['request_id'], set(lookup))
                require(req is not None, 'Unlinked native request')
                response = lookup[req]
                number = layer_number(row['layer'])
                require(layers.setdefault(number, row['layer']) == row['layer'], 'Inconsistent layer identity')
                pos = row['prompt_position']
                require(path.name == f'rank{row["rank"]}.jsonl' and row['run_id'] == response['run_id'] and
                        row['request_context_len'] == response['prompt_tokens'], 'Native rank/run/context mismatch')
                require(any(a <= pos <= b for a, b in ranges), 'Unexpected captured position')
                key = (req, row['layer'], pos)
                require(key not in seen, 'Duplicate native layer/position across ranks')
                seen.add(key)
                count += 1
                for unit in set(row['raw_selected_ids'])-{-1}:
                    require(4*(unit+1)-1 <= pos, 'Selected C4 KV outside causal support')
                start = response['boundary_position']
                if not start-1 <= pos < start+WINDOW:
                    continue
                coverage[(req, pos)][number] = row['rank']
                mapping = assignments[req].setdefault(str(pos), {})
                mapping[row['layer']] = row['rank']
                if pos == start-1:
                    group = (response['pair_id'], response['repetition'], number)
                    if group in warm and warm[group] != row['raw_selected_ids']:
                        prefix_drift += 1
                        if len(drift_examples) < 20:
                            drift_examples.append({'kind':'pair_prefix','request_id':req,'layer':row['layer'],'position':pos})
                        require(allow_selected_drift, 'Paired prefix selected IDs differ')
                    warm.setdefault(group, row['raw_selected_ids'])
                group = (response['pair_id'], response['variant'], number, pos)
                if group in repeat and repeat[group] != row['raw_selected_ids']:
                    repeat_drift += 1
                    if len(drift_examples) < 20:
                        drift_examples.append({'kind':'repeat','request_id':req,'layer':row['layer'],'position':pos})
                    require(allow_selected_drift, 'Repeated selected IDs differ')
                repeat.setdefault(group, row['raw_selected_ids'])
                out.write(json.dumps(row, separators=(',', ':'))+'\n')
                kept += 1
    for response in responses:
        for pos in range(response['boundary_position']-1, response['boundary_position']+WINDOW):
            mapping = coverage[(response['request_id'], pos)]
            require(set(mapping) == set(range(2, 43, 2)), 'Incomplete 21-layer CP union window')
            require(len(set(mapping.values())) == 1, 'Layers disagree on CP query owner')
    return {'raw_rows': count, 'window_rows': kept, 'requests': len(responses), 'positions_per_request': 33,
            'layers': 21, 'ranks_with_rows': sorted({v for m in coverage.values() for v in m.values()}),
            'query_assignment': dict(assignments), 'layer_names': [layers[i] for i in range(2, 43, 2)],
            'causal_scope_checked': True, 'repeat_selected_ids_identical': repeat_drift == 0,
            'pair_prefix_selected_ids_identical': prefix_drift == 0,
            'repeat_selected_id_differences': repeat_drift, 'pair_prefix_selected_id_differences': prefix_drift,
            'selected_id_drift_examples': drift_examples,
            'paired_effect_interpretation': 'exploratory' if repeat_drift or prefix_drift else 'qualified_for_offline_comparison'}


def source_evidence(path, needles, directory):
    lines = path.read_text().splitlines()
    selected = [i+1 for i, line in enumerate(lines) if any(n in line for n in needles)]
    require(selected, 'No source evidence for '+path.name)
    return {'path': str(path.relative_to(directory)), 'sha256': sha256_file(path),
            'start_line': min(selected), 'end_line': max(selected),
            'explanation': 'Fixed DeepSeek C4 overlapping compressor, replicated CP KV, or BF16 payload contract.'}


def validate_layout(config, sources, directory):
    require(config['head_dim'] == 512 and config['torch_dtype'] == 'bfloat16' and config['num_key_value_heads'] == 1 and
            config['num_hidden_layers'] == 43 and config['index_topk'] == 512 and
            [i for i, r in enumerate(config['compress_ratios'][:43]) if r == 4] == list(range(2,43,2)), 'DeepSeek model C4 layout mismatch')
    model, dsa, cache = (sources[n].read_text() for n in ('model.py', 'dsa_cp.py', 'cache_spec.py'))
    require('self.overlap = compress_ratio == 4' in model and 'head_size=self.head_dim' in model and
            'dtype=self.dtype' in model and 'num_kv_heads=1' in model, 'Compressed KV spec mismatch')
    require('self.block_size' in cache and 'get_dtype_size(self.dtype)' in cache and
            'hidden_states_cache,' in dsa and 'compressed_kv' in dsa and 'compress_kv_cache' in dsa,
            'CP replicated compressor/cache source mismatch')
    return {'dtype': 'bfloat16', 'native_kv_payload_bytes_per_layer': 1024, 'page128_bytes': 131072,
            'formula': '512 * 2', 'support_contract': 'c4_overlap_v1',
            'support': '[max(0,4*i-4),4*(i+1))', 'ready_position': '4*(i+1)-1',
            'source_evidence': [source_evidence(path, ['overlap', 'head_size=', 'dtype=', 'DataCopy', 'state', 'compress', 'block_size'], directory)
                                for path in sources.values()],
            'scope': 'C4 attention payload only; indexer, SWA, C128, compressor state and communication excluded'}


def export_sidecar(responses, pairs, profile, evidence, support_evidence):
    lookup = {p['pair_id']: p for p in pairs['pairs']}
    data = {'schema_version': 1, 'synthetic': True, 'selection_provenance': 'real_npu_native', 'online_kv_restore': False,
            'snapshot_provenance': 'exact_API_tokens_and_source_verified_replicated_C4_inventory',
            'support_contract': {'name': 'c4_overlap_v1', 'start': 'max(0,4*i-4)', 'stop': '4*(i+1)', 'ready': '4*(i+1)-1'},
            'snapshots': [], 'events': []}
    for response in responses:
        pair = lookup[response['pair_id']]
        ids, start, stem = response['prompt_token_ids'], response['boundary_position'], response['request_id']
        common = {'session_id': 'session:'+pair['pair_id'], 'model_id': profile['model_id'],
                  'model_revision': profile['model_revision'], 'tokenizer_sha256': pairs['tokenizer_json_sha256']}
        for label, tokens, version, sequence, generated in (
                ('prefix', ids[:start], 'prefix', 1, start), ('current', ids, response['variant'], 2, start),
                ('post', ids, response['variant'], 2, start+WINDOW)):
            snapshot = dict(common, snapshot_id=stem+':'+label, context_version=version, sequence=sequence,
                            token_ids=tokens, token_ids_sha256=sha_ids(tokens), generated_tokens=generated,
                            regions=[{'start': 0, 'end': start, 'type': 'other', 'created_at': 1}],
                            lanes=[{'rank': rank, 'layer': layer, 'kv_kind': 'compressed_kv_token_position',
                                    'generated_ids': list(range(generated//4)), 'support_contract': 'c4_overlap_v1',
                                    'support_evidence': support_evidence} for rank in range(8) for layer in evidence['layer_names']])
            if len(tokens) > start:
                snapshot['regions'].append({'start': start, 'end': len(tokens), 'type': 'other', 'created_at': 2})
            data['snapshots'].append(snapshot)
        require(pair['context_target'] in {8192,32768}, 'Unexpected split context')
        data['events'].append({'event_id': stem, 'request_id': stem, 'run_id': response['run_id'],
            'previous_snapshot_id': stem+':prefix', 'snapshot_id': stem+':current', 'post_window_snapshot_id': stem+':post',
            'event_type': pair['event_type'] if response['variant'] == 'event' else 'no_event', 'previous_state': 'other',
            'event_position': start, 'resume_position': start, 'workload_id': pair['workload_id'], 'episode_id': pair['episode_id'],
            'source_trace_id': pair['source_trace_id'], 'pair_id': pair['pair_id'], 'variant': response['variant'],
            'repetition': response['repetition'], 'trajectory_id': stem, 'split': 'train' if pair['context_target'] == 8192 else 'eval',
            'query_assignment': evidence['query_assignment'][stem]})
    return data


def clone_args(metadata, name, run_id, image, *, env_extra=None, mounts=(), command=None):
    cfg, host = metadata['Config'], metadata['HostConfig']
    require(host['NetworkMode'] == 'host' and not host.get('AutoRemove'), 'Unsupported original container network/lifecycle')
    args = ['docker', 'run', '-d', '--name', name, '--label', 'memecho.validation.run='+run_id,
            '--network', host['NetworkMode'], '--ipc', host['IpcMode'], '--shm-size', str(host['ShmSize']), '--restart', 'no']
    if host['Privileged']:
        args += ['--privileged']
    for setting in host.get('SecurityOpt') or []:
        args += ['--security-opt', setting]
    for binding in (host.get('Binds') or [])+list(mounts):
        args += ['-v', binding]
    for device in host.get('Devices') or []:
        args += ['--device', device['PathOnHost']+':'+device['PathInContainer']+':'+device['CgroupPermissions']]
    environment = dict(v.split('=',1) for v in cfg['Env'])
    environment.update(env_extra or {})
    for key, value in environment.items():
        args += ['-e', key+'='+value]
    if cfg.get('WorkingDir'):
        args += ['-w', cfg['WorkingDir']]
    if cfg.get('User'):
        args += ['-u', cfg['User']]
    # The trace image inherits the exact original entrypoint from its pinned base.
    args += [image, *(command if command is not None else cfg['Cmd'])]
    return args


class Worker(BaseWorker):
    def __init__(self, root, run_id):
        require(re.fullmatch(r'deepseek_\d{8}T\d{6}Z_[0-9a-f]{8}', run_id), 'Invalid DeepSeek run ID')
        self.root, self.run_id = Path(root), run_id
        self.directory = self.root/'m0a/runs'/run_id
        self.code_root = Path(__file__).resolve().parents[1]
        if str(self.code_root) not in sys.path:
            sys.path.insert(0,str(self.code_root))
        self.env = dict(os.environ, DOCKER_HOST='unix://'+str(self.root/'runtime/docker.sock'))
        self.container = 'memecho-'+run_id
        self.owned = set()
        self.original = None
        self.restore_required = False
        self.completed = []
        launch_path = self.directory/'launch.json'
        launch = json.loads(launch_path.read_text()) if launch_path.exists() else {}
        self.diagnostic_only = launch.get('diagnostic_only',False)
        self.repair = launch.get('repair',False)
        self.validation_mode = launch.get('validation_mode','strict')
        require(self.validation_mode in ('strict','trace-replay'), 'Unknown validation mode')
        self.selected_config = None
        self.diagnostics = []
        self.failures = []
        self.output_differences = []
        self.operator_repairs = []
        self.state = {'schema_version': 1, 'run_id': run_id, 'devices': list(range(8)), 'worker_pid': os.getpid(),
                      'started_at': utc(), 'status': 'running', 'stage': 'bootstrap', 'error': None}
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.checkpoint_number = 0

    def update(self, **values):
        values.setdefault('completed_stages', list(self.completed))
        super().update(**values)

    def command(self, args, *, timeout=900, output=None, env=None):
        """Interrupt command groups promptly so deadline/signal recovery can run."""
        print('Command:', args, flush=True)
        stream = Path(output).open('w') if output else None
        child = subprocess.Popen(args, env=env or self.env, stdout=stream or subprocess.PIPE,
                                 stderr=subprocess.STDOUT if stream else None, text=True, start_new_session=True)
        try:
            result, _ = child.communicate(timeout=timeout)
            require(child.returncode == 0, f'Command exited {child.returncode}: {args}')
            return result or ''
        except BaseException:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=5)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
            raise
        finally:
            if stream:
                stream.close()

    def inspect(self, name):
        return json.loads(self.command(['docker','inspect',name], timeout=30))[0]

    def service_health(self, name, expected_models=None):
        deadline = time.monotonic()+900
        while time.monotonic() < deadline:
            require(self.inspect(name)['State']['Running'], 'Service container exited')
            try:
                with OPENER.open('http://127.0.0.1:8900/health', timeout=10) as response:
                    require(response.status == 200, 'Service unhealthy')
                with OPENER.open('http://127.0.0.1:8900/v1/models', timeout=10) as response:
                    models = [(m['id'],m['root']) for m in json.load(response)['data']]
                require(models == (expected_models or [('dsv4',str(self.root/load_profile('deepseek_v4')['model_path']))]), 'Model identity mismatch')
                check(tuple(range(8)), 65536)
                return models
            except (OSError, ValueError):
                time.sleep(5)
        raise TimeoutError('Service health/model/eight-NPU timeout (900 seconds)')

    def preflight(self):
        self.update(stage='preflight')
        self.original = self.inspect(ORIGINAL)
        require(self.original['State']['Running'] and self.original['Image'] == BASE_IMAGE_ID, 'Original fixed base service not running')
        cfg = self.original['Config']
        require('--tensor-parallel-size' in cfg['Cmd'] and cfg['Cmd'][cfg['Cmd'].index('--tensor-parallel-size')+1] == '8' and
                '--speculative-config' in cfg['Cmd'] and 'mtp' in cfg['Cmd'][cfg['Cmd'].index('--speculative-config')+1] and
                'enable_dsa_cp' in ' '.join(cfg['Cmd']), 'Original TP/MTP/DSA CP configuration mismatch')
        require(not any(v.startswith('VLLM_ASCEND_M0A_TRACE_DIR=') and v.split('=',1)[1] for v in cfg['Env']), 'Original service has trace enabled')
        write_json(self.directory/'original-service.json', self.original)
        models = self.service_health(self.original['Id'])
        write_json(self.directory/'original-models.json', models)
        require(self.command(['docker','image','inspect',IMAGE,'--format','{{.Id}}']).strip() == IMAGE_ID, 'Pinned trace image mismatch')
        image = json.loads(self.command(['docker','image','inspect',IMAGE]))[0]
        require(image['Config']['Entrypoint'] == cfg['Entrypoint'], 'Trace/original entrypoint mismatch')
        base = json.loads(self.command(['docker','image','inspect',BASE_IMAGE_ID]))[0]
        require(base['RootFS']['Layers'] == image['RootFS']['Layers'][:len(base['RootFS']['Layers'])], 'Trace image base mismatch')
        model = self.root/load_profile('deepseek_v4')['model_path']
        revision_manifest = self.root/'manifests/revision_manifest.yaml'
        require(load_profile('deepseek_v4')['model_revision'] in revision_manifest.read_text(), 'Fixed model revision evidence mismatch')
        model_manifest = (self.root/'manifests/model_manifest.txt').read_text()
        require('revision='+load_profile('deepseek_v4')['model_revision'] in model_manifest and
                'repo='+load_profile('deepseek_v4')['model_id'] in model_manifest, 'Downloaded model identity mismatch')
        self.command(['npu-smi','info'], output=self.directory/'npu-before.txt', timeout=30)
        sources = {}
        helper = self.container+'-prepare'
        self.owned.add(helper)
        self.command(['docker','create','--name',helper,'--label','memecho.validation.run='+self.run_id,IMAGE,'true'])
        for name, source in SOURCE_PATHS.items():
            path = self.directory/'sources'/name
            path.parent.mkdir(exist_ok=True)
            self.command(['docker','cp',helper+':'+source,str(path)])
            sources[name] = path
        self.cleanup()
        # Add only CPU prompt-length capture; seq_lens is a chunk end during
        # chunked prefill and cannot identify the complete API prompt.
        dsa = sources['dsa_cp.py'].read_text()
        needle = 'req_metadata.trace_request_ids = kwargs.get("m0a_request_ids", ())'
        require(dsa.count(needle) == 1, 'Pinned DSA trace injection point changed')
        dsa = dsa.replace(needle, needle+'\n        req_metadata.trace_prompt_lens_cpu = kwargs.get("m0a_prompt_lens", ())')
        runner = sources['model_runner_v1.py'].read_text()
        needle = 'm0a_request_ids=('
        require(runner.count(needle) == 1, 'Pinned runner trace injection point changed')
        runner = runner.replace(needle, 'm0a_prompt_lens=(self.input_batch.num_prompt_tokens_cpu_tensor[:num_reqs].tolist()\n'
                                '                        if not for_cudagraph_capture else ()),\n                    '+needle)
        capture = self.directory/'capture-source'
        capture.mkdir()
        (capture/'dsa_cp.py').write_text(dsa)
        (capture/'model_runner_v1.py').write_text(runner)
        compile(dsa, str(capture/'dsa_cp.py'), 'exec')
        compile(runner, str(capture/'model_runner_v1.py'), 'exec')
        if self.repair:
            from m0a.deepseek_repair import patch_worker, instrument_source
            runtime = self.code_root/'m0a/deepseek_repair.py'
            repair_source = self.directory/'repair-source'
            repair_source.mkdir()
            (repair_source/'worker.py').write_text(patch_worker(sources['worker.py'].read_text(),runtime))
            diagnostic = self.directory/'operator-source'
            diagnostic.mkdir()
            runtime = self.code_root/'m0a/deepseek_operator_diagnostics.py'
            (diagnostic/'dsa_cp.py').write_text(instrument_source(dsa,runtime,'install_dsa'))
            (diagnostic/'model.py').write_text(instrument_source(sources['model.py'].read_text(),runtime,'install_model'))
            files = [*capture.glob('*.py'),*repair_source.glob('*.py'),*diagnostic.glob('*.py'),
                     self.code_root/'m0a/deepseek_repair.py',runtime,self.code_root/'m0a/deepseek_compressor_contract.py']
            write_json(self.directory/'experimental-provenance.json',{
                'image_id':IMAGE_ID,'patches':[], 'files':[{'path':str(p),'sha256':sha256_file(p)} for p in files],
                'acceptance_excludes_operator_instrumentation':True})
        config = json.loads((model/'config.json').read_text())
        layout = validate_layout(config, sources, self.directory)
        write_json(self.directory/'layout.json',layout)
        identity = {'model_revision': load_profile('deepseek_v4')['model_revision'], 'revision_evidence': revision_manifest.read_text(),
                    'download_manifest': model_manifest,
                    'revision_evidence_sha256': sha256_file(revision_manifest), 'trace_image_id': IMAGE_ID, 'original_image_id': self.original['Image'],
                    'model_files': {p.name: {'size':p.stat().st_size,'sha256':sha256_file(p)} for p in model.iterdir() if p.is_file() and p.suffix != '.safetensors'},
                    'weight_files': {p.name:p.stat().st_size for p in model.glob('*.safetensors')},
                    'revision_verification_scope': 'Existing deployment revision manifest, exact tokenizer/config/index hashes and weight sizes; weights not re-downloaded.'}
        write_json(self.directory/'identity.json',identity)
        write_json(self.directory/'preflight.json',{'at':utc(),'devices':list(range(8)),'original_container_id':self.original['Id'],'port':8900})
        # Independent process restores the retained original even after SIGKILL.
        with (self.directory/'watchdog.log').open('ab') as log:
            child = subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--watchdog','--root',str(self.root),'--run-id',self.run_id],
                                     stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        write_json(self.directory/'watchdog.json',{'pid':child.pid})
        self.update(watchdog_pid=child.pid)
        return layout

    def prepare(self):
        self.update(stage='prepare_inputs')
        helper = self.container+'-prepare'
        self.owned.add(helper)
        self.command(['docker','run','--rm','--name',helper,'--label','memecho.validation.run='+self.run_id,
                      '-v',str(self.root)+':'+str(self.root),IMAGE,'python3',str(Path(__file__).resolve()),
                      '--prepare','--root',str(self.root),'--run-id',self.run_id], output=self.directory/'prepare.log')
        self.cleanup()
        pairs = json.loads((self.directory/'pairs.json').read_text())
        # Verify the full input using the actual serving tokenizer before baseline.
        for pair in pairs['pairs']:
            for variant in ('event','control'):
                payload = {'model':'dsv4','messages':[{'role':'user','content':pair[variant]['prompt']}],
                           'add_generation_prompt':True}
                result = request(payload,url='http://127.0.0.1:8900/tokenize')
                require(result['tokens'] == pair[variant]['prompt_token_ids'], 'Actual service tokenizer differs')
        return pairs

    def wait_release(self):
        deadline = time.monotonic()+900
        last = None
        while time.monotonic()<deadline:
            try:
                require(port_available(), 'Port still listening')
                usage = check(tuple(range(8)),6000)
                # npu-smi process rows represent HCCL/worker device ownership.
                text = self.command(['npu-smi','info'],timeout=30)
                process_rows = re.findall(r'\|\s*[0-7]\s+0\s+\|\s*(\d+)\s*\|',text)
                require(not process_rows, 'NPU/HCCL processes remain after stop')
                write_json(self.directory/'release-check.json',{'at':utc(),'port_free':True,'hbm_used_mb':usage,'device_processes':process_rows})
                return
            except (OSError, ValueError, RuntimeError) as error:
                last=error
                time.sleep(5)
        raise TimeoutError('Port/HCCL/eight-NPU release timeout: '+str(last))

    def pause_original(self):
        require(self.inspect(self.original['Id'])['State']['Running'], 'Original changed before pause')
        self.restore_required = True
        self.update(original_service_restored=False)
        write_json(self.directory/'restore-required.json',{'original_id':self.original['Id'],'at':utc()})
        self.command(['docker','stop','--time','60',self.original['Id']],timeout=90)
        self.wait_release()

    def restore(self):
        self.update(stage='restore_original')
        self.cleanup()
        if self.restore_required:
            self.wait_release()
            self.command(['docker','start',self.original['Id']],timeout=90)
        models = self.service_health(self.original['Id'])
        current = self.inspect(self.original['Id'])
        require(current['Config'] == self.original['Config'] and current['HostConfig'] == self.original['HostConfig'], 'Original service configuration changed')
        self.restore_required = False
        write_json(self.directory/'restoration.json',{'at':utc(),'restored':True,'container_id':current['Id'],
                   'healthy':True,'model_identity':models,'eight_npu_healthy':True,'config_unchanged':True})
        self.update(original_service_restored=True)

    def contract(self, *, initialize_new_blocks=False):
        self.update(stage='npu_compressor_contract')
        helper = self.container+'-contract'
        self.owned.add(helper)
        output = self.directory/('compressor-contract-repaired.json' if initialize_new_blocks else 'compressor-contract.json')
        command = ['python3',str(self.code_root/'m0a/deepseek_compressor_contract.py'),
                                                                         '--output',str(output),
                                                                         '--artifact-directory',str(self.directory),
                                                                         '--model-config',str(self.root/load_profile('deepseek_v4')['model_path']/'config.json'),
                                                                         '--source-dir',str(self.directory/'sources'),
                                                                         '--block-size',self.original['Config']['Cmd'][self.original['Config']['Cmd'].index('--block-size')+1]
                                                                         if '--block-size' in self.original['Config']['Cmd'] else '128']
        if initialize_new_blocks:
            command.append('--initialize-new-blocks')
        elif self.repair:
            tests = ['python3','-m','unittest','discover','-s',str(self.code_root/'m0a'),'-p','test_deepseek_repair.py']
            command = ['python3','-c','import subprocess\nsubprocess.run('+repr(command)+',check=True)\n'
                       'with open('+repr(str(self.directory/'npu-contract-tests.log'))+',"w") as log:\n'
                       ' subprocess.run('+repr(tests)+',stdout=log,stderr=subprocess.STDOUT,check=True)\n']
        args = clone_args(self.original,helper,self.run_id,IMAGE,command=command)
        self.command(args)
        exit_code = self.command(['docker','wait',helper],timeout=900).strip()
        self.command(['docker','logs',helper],output=output.with_suffix('.log'),timeout=30)
        require(exit_code == '0','NPU compressor contract failed (see compressor-contract.log)')
        result = json.loads(output.read_text())
        self.cleanup()
        self.wait_release()
        if self.repair and not initialize_new_blocks and result.get('repairable_state_reuse'):
            after = self.contract(initialize_new_blocks=True)
            from m0a.deepseek_repair import verify_state_repair, patch_compressor_initialization
            evidence = verify_state_repair(result,after)
            (self.directory/'compressor-contract-before.json').write_bytes(output.read_bytes())
            output.write_bytes((self.directory/'compressor-contract-repaired.json').read_bytes())
            source = self.directory/'capture-source/dsa_cp.py'
            repaired = self.directory/'repair-source/dsa_cp.py'
            repaired.write_text(patch_compressor_initialization(source.read_text(),self.code_root/'m0a/deepseek_repair.py'))
            evidence.update(source_sha256=sha256_file(repaired),before_sha256=sha256_file(self.directory/'compressor-contract-before.json'),
                            after_sha256=sha256_file(self.directory/'compressor-contract-repaired.json'))
            self.operator_repairs.append(evidence)
            write_json(self.directory/'operator-repairs.json',self.operator_repairs)
            provenance = json.loads((self.directory/'experimental-provenance.json').read_text())
            provenance['patches'] = self.operator_repairs
            provenance['files'].append({'path':str(repaired),'sha256':sha256_file(repaired)})
            write_json(self.directory/'experimental-provenance.json',provenance)
            result = after
        require(result['passed'] is True and result['support_contract'] == 'c4_overlap_v1', 'Unverified compressor support')
        return result

    def experimental_mounts(self, *, operator_diagnostics=False):
        if not self.repair:
            return []
        mounts = [str(self.directory/'repair-source/worker.py')+':'+SOURCE_PATHS['worker.py']+':ro',
                  str(self.directory/'capture-source/model_runner_v1.py')+':'+SOURCE_PATHS['model_runner_v1.py']+':ro']
        dsa = self.directory/('operator-source/dsa_cp.py' if operator_diagnostics else 'capture-source/dsa_cp.py')
        if self.operator_repairs:
            dsa = self.directory/'repair-source/dsa_cp.py'
            if operator_diagnostics:
                from m0a.deepseek_repair import instrument_source
                diagnostic = self.directory/'operator-source/repaired-dsa_cp.py'
                diagnostic.write_text(instrument_source(dsa.read_text(),self.code_root/'m0a/deepseek_operator_diagnostics.py','install_dsa'))
                dsa = diagnostic
        mounts.append(str(dsa)+':'+SOURCE_PATHS['dsa_cp.py']+':ro')
        if operator_diagnostics:
            mounts.append(str(self.directory/'operator-source/model.py')+':'+SOURCE_PATHS['model.py']+':ro')
        return mounts

    def candidate_environment(self,candidate,path,*,operator_diagnostics=False):
        environment = dict(candidate.get('environment',{}), VLLM_ASCEND_M0A_TRACE_DIR='')
        if self.repair:
            environment['MEMECHO_DETERMINISTIC_DIR'] = str(path/'rank-settings')
        if operator_diagnostics:
            environment['MEMECHO_OPERATOR_DIR'] = str(path)
        return environment

    def start_candidate(self,candidate,path,*,operator_diagnostics=False):
        self.update(stage='diagnostic_startup_'+candidate['id'],candidate=candidate['id'])
        self.wait_release()
        self.owned.add(self.container)
        args = clone_args(self.original,self.container,self.run_id,IMAGE,
                          env_extra=self.candidate_environment(candidate,path,operator_diagnostics=operator_diagnostics),
                          mounts=self.experimental_mounts(operator_diagnostics=operator_diagnostics),command=candidate['command'])
        write_json(path/'launch.json',{'args':args,'configuration':candidate,'trace_enabled':False})
        self.command(args,output=path/'launcher.log')
        self.service_health(self.container)
        if self.repair:
            from m0a.deepseek_repair import validate_deterministic_ranks
            write_json(path/'deterministic-ranks.json',validate_deterministic_ranks(path/'rank-settings',candidate))

    def request_sequence(self,stage,sequence,profile,*,directory=None,baseline=None,collect=False,logprobs=False,
                         soft_output_differences=False):
        directory = directory or self.directory
        records, stable, failures, differences = [], {}, [], []
        expected = {(r['pair_id'],r['variant'],r['repetition']):r['signature'] for r in baseline or []}
        config = self.selected_config or self.state.get('candidate')
        self.update(stage=stage,completed_requests=0)
        with (directory/(stage+'-responses.jsonl')).open('w') as out, (directory/(stage+'-api.jsonl')).open('w') as raw:
            for pair,variant,repetition in sequence:
                item = pair[variant]
                payload = payload_for(profile,item.get('messages', item['prompt']))
                if logprobs:
                    payload.update(logprobs=True,top_logprobs=5)
                started = time.time_ns()
                identity = {'pair_id':pair['pair_id'],'variant':variant,'repetition':repetition,
                            'prompt_token_ids_sha256':item['prompt_token_ids_sha256_expected'],
                            'prompt_token_ids':item['prompt_token_ids'],'configuration':config,'started_ns':started}
                evidence = dict(identity,request=payload)
                try:
                    result = request(payload)
                    evidence['response'] = result
                    sig = validate_api_result(result,item)
                    key = (pair['pair_id'],variant)
                    comparisons = [] if logprobs else [('repeat',stable.get(key)),('trace_on_off',expected.get(key+(repetition,)))]
                    for kind,reference in comparisons:
                        if reference is not None and reference != sig:
                            difference = dict(identity,kind=kind,request=payload,response=result,
                                              **output_difference(reference,sig))
                            differences.append(difference)
                            if not soft_output_differences:
                                failures.append(difference)
                    stable.setdefault(key,sig)  # Always compare with the cold first response.
                    row = {k:pair[k] for k in ('pair_id','workload_id','episode_id','context_target','event_type','source_trace_id') if k in pair}
                    row.update(identity,schema_version=1,run_id=self.run_id,boundary_position=item['boundary_position'],
                               request_id=result['id'],prompt_tokens=len(item['prompt_token_ids']),signature=sig,
                               completion_tokens=result['usage']['completion_tokens'],finished_ns=time.time_ns())
                    if logprobs:
                        content = (result['choices'][0].get('logprobs') or {}).get('content') or []
                        row['first_token_logprobs'] = content[0] if content else None
                        values = sorted((t['logprob'] for t in (content[0].get('top_logprobs',[]) if content else [])),reverse=True)
                        row['first_token_top2_gap'] = values[0]-values[1] if len(values)>1 else None
                    out.write(json.dumps(row,ensure_ascii=False)+'\n')
                    out.flush()
                    records.append(row)
                except Exception as error:
                    failure = dict(identity,kind='request_or_token_validation',error=str(error))
                    if isinstance(error,urllib.error.HTTPError):
                        failure.update(http_status=error.code,response_body=error.read().decode('utf-8',errors='replace'))
                    evidence['error'] = failure
                    failures.append(failure)
                except (TaskTerminated,TaskDeadline) as error:
                    evidence['termination'] = str(error)
                    raw.write(json.dumps(evidence,ensure_ascii=False)+'\n')
                    raw.flush()
                    raise
                raw.write(json.dumps(evidence,ensure_ascii=False)+'\n')
                raw.flush()
                self.update(completed_requests=len(records))
                if failures and not collect:
                    write_json(directory/(stage+'-failures.json'),failures)
                    write_json(directory/(stage+'-output-differences.json'),differences)
                    self.failures += failures
                    self.output_differences += differences
                    raise ValueError(f'{stage} rejected: {failures[-1]["kind"]}, pair={pair["pair_id"]}, variant={variant}, repetition={repetition}')
        write_json(directory/(stage+'-failures.json'),failures)
        write_json(directory/(stage+'-output-differences.json'),differences)
        self.failures += failures
        self.output_differences += differences
        return records, failures

    def requests(self,stage,pairs,profile,baseline=None,*,soft_output_differences=False):
        sequence = [(p,v,r) for p in pairs['pairs'] for r in range(2)
                    for v in (('event','control') if r==0 else ('control','event'))]
        records,_ = self.request_sequence(stage,sequence,profile,baseline=baseline,
                                          soft_output_differences=soft_output_differences)
        require(len(records)==4*len(pairs['pairs']),'Incomplete request phase')
        return records

    def choose_configuration(self,pairs,profile):
        root = self.directory/'diagnostics'
        root.mkdir()
        pair = next(p for p in pairs['pairs'] if p['pair_id']=='8192_tool_call')
        sequence = [(pair,v,cycle*2+(1 if index>=2 else 0)) for cycle in range(5)
                    for index,v in enumerate(('event','control','control','event'))]
        candidates = stability_candidates(self.original['Config']['Cmd'],repair=self.repair)
        for candidate in candidates:
            path = root/candidate['id']
            path.mkdir()
            result = {'configuration':candidate,'status':'running','diagnostic_requests_expected':20,'cold_responses_retained':True}
            baseline = None
            self.selected_config = candidate
            try:
                self.start_candidate(candidate,path)
                records,failures = self.request_sequence('stability',sequence,profile,directory=path,collect=True)
                result.update(diagnostic_requests_completed=len(records),diagnostic_failures=failures)
                # Probe is separate and cannot qualify a candidate or alter formal responses.
                probes = [(pair,v,i) for i,v in enumerate(('event','control','control','event'))]
                _,probe_errors = self.request_sequence('logprobs',probes,profile,directory=path,collect=True,logprobs=True)
                result['logprobs_probe_errors'] = probe_errors
                require(len(records)==20 and not failures,'Twenty-request stability gate failed')
                if not self.diagnostic_only:
                    baseline = self.requests('trace_off',pairs,profile,records)
                    for name in ('trace_off-api.jsonl','trace_off-responses.jsonl','trace_off-failures.json'):
                        (path/name).write_bytes((self.directory/name).read_bytes())
                result['status'] = 'accepted'
            except (TaskTerminated,TaskDeadline) as error:
                result.update(status='interrupted',error=str(error))
                write_json(path/'result.json',result)
                raise
            except Exception as error:
                result.update(status='rejected',error=str(error))
                for file in self.directory.glob('trace_off-*'):
                    if file.is_file():
                        (path/file.name).write_bytes(file.read_bytes())
                        file.unlink()
            finally:
                try:
                    self.command(['docker','logs',self.container],output=path/'service.log',timeout=60)
                except Exception as log_error:
                    result['service_log_error'] = str(log_error)
                self.cleanup()
                self.wait_release()
            self.diagnostics.append(result)
            write_json(path/'result.json',result)
            write_json(self.directory/'diagnostics.json',self.diagnostics)
            self.checkpoint('diagnostic_'+candidate['id'],[self.directory/'diagnostics.json',*[p for p in path.rglob('*') if p.is_file()]])
            if result['status']=='accepted':
                write_json(self.directory/'selected-config.json',dict(candidate,image_id=IMAGE_ID,
                           original_command=self.original['Config']['Cmd'],diagnostic_only=self.diagnostic_only,
                           diagnostic_requests=20,baseline_requests=0 if self.diagnostic_only else 48))
                return baseline
        self.selected_config = None
        if self.repair:
            self.operator_diagnostics(candidates[-1],pair,profile)
        raise ValueError(f'All {"eight" if self.repair else "six"} experimental configurations rejected; see diagnostics.json')

    def operator_diagnostics(self,candidate,pair,profile):
        """Separate observed requests and replay; never qualify an instrumented run."""
        path = self.directory/'operator-diagnostics'
        path.mkdir()
        self.selected_config = candidate
        try:
            self.start_candidate(candidate,path,operator_diagnostics=True)
            records = []
            for index in range(20):
                write_json(path/'request.json',{'ordinal':index,'pair_id':pair['pair_id'],'variant':'event',
                                              'prompt_sha256':pair['event']['prompt_token_ids_sha256_expected']})
                rows,_ = self.request_sequence(f'observed-{index:02d}',[(pair,'event',index)],profile,directory=path,collect=True)
                records.extend(rows)
            signatures = [r['signature'] for r in records]
            write_json(path/'observed-responses.json',records)
            write_json(path/'observation.json',{'requests':len(records),'unstable':any(s!=signatures[0] for s in signatures[1:]),
                'instrumentation_synchronizes_device':True,'root_cause_confirmed':False})
        finally:
            try:
                self.command(['docker','logs',self.container],output=path/'service.log',timeout=60)
            finally:
                self.cleanup()
                self.wait_release()
        for capsule in sorted(path.glob('rank*-first.pt')):
            helper = self.container+'-replay'
            self.owned.add(helper)
            output = capsule.with_suffix('.replay.json')
            self.command(clone_args(self.original,helper,self.run_id,IMAGE,
                env_extra=candidate.get('environment',{}),command=['python3',str(self.code_root/'m0a/deepseek_operator_diagnostics.py'),
                    '--replay',str(capsule),'--output',str(output)]))
            exit_code = self.command(['docker','wait',helper],timeout=900).strip()
            self.command(['docker','logs',helper],output=capsule.with_suffix('.replay.log'),timeout=60)
            if exit_code != '0' and not output.exists():
                write_json(output,{'reproduced':False,'error':'Replay helper exited '+exit_code,'root_cause_confirmed':False})
            self.cleanup()
            self.wait_release()
        write_json(path/'result.json',{'root_cause_confirmed':False,'patch_applied':False,
            'first_observations':[json.loads(p.read_text()) for p in sorted(path.glob('rank*-first.json'))],
            'replays':[json.loads(p.read_text()) for p in sorted(path.glob('*.replay.json'))],
            'conclusion':'No verified operator repair; input drift and observer effects remain suspicions.'})
        self.checkpoint('operator_diagnostics',[p for p in path.rglob('*') if p.is_file()])
        self.selected_config = None

    def start_trace(self,ranges):
        self.update(stage='trace_on_startup')
        self.wait_release()
        trace = self.directory/'raw-traces'
        trace.mkdir()
        hook = self.code_root/'m0a/source/m0a_selected_trace.py'
        self.owned.add(self.container)
        args = clone_args(self.original,self.container,self.run_id,IMAGE,
                          env_extra={**self.candidate_environment(self.selected_config,trace),
                                     'VLLM_ASCEND_M0A_TRACE_DIR':str(trace), 'VLLM_ASCEND_M0A_RUN_ID':self.run_id,
                                     'VLLM_ASCEND_M0A_TRACE_POSITIONS':','.join(f'{a}:{b}' for a,b in ranges)},
                          mounts=[str(hook)+':'+ASCEND+'vllm_ascend/attention/context_parallel/m0a_selected_trace.py:ro']+
                                  (self.experimental_mounts() if self.repair else [
                                   str(self.directory/'capture-source/dsa_cp.py')+':'+SOURCE_PATHS['dsa_cp.py']+':ro',
                                   str(self.directory/'capture-source/model_runner_v1.py')+':'+SOURCE_PATHS['model_runner_v1.py']+':ro']),
                          command=self.selected_config['command'])
        write_json(self.directory/'trace-launch.json',{'args':args,'hook_sha256':sha256_file(hook),'configuration':self.selected_config,
                   'original_command':self.original['Config']['Cmd']})
        self.command(args,output=self.directory/'trace_on-launcher.log')
        self.service_health(self.container)
        if self.repair:
            from m0a.deepseek_repair import validate_deterministic_ranks
            write_json(self.directory/'trace-deterministic-ranks.json',validate_deterministic_ranks(trace/'rank-settings',self.selected_config))

    def trace_replay_baseline(self,pairs,profile):
        """Capture the unmodified deployment command without a stability gate."""
        candidate={'id':'original_fresh','command':list(self.original['Config']['Cmd']),'changes':[]}
        self.selected_config=candidate
        write_json(self.directory/'selected-config.json',dict(candidate,image_id=IMAGE_ID,
                   original_command=self.original['Config']['Cmd'],validation_mode=self.validation_mode,
                   diagnostic_requests=0,baseline_requests=48))
        path=self.directory/'trace-off-service'
        path.mkdir()
        try:
            self.start_candidate(candidate,path)
            return self.requests('trace_off',pairs,profile,soft_output_differences=True)
        finally:
            try:
                self.command(['docker','logs',self.container],output=path/'service.log',timeout=60)
            finally:
                self.cleanup()
                self.wait_release()

    def validate_engineering_evidence(self):
        """Check completed local evidence before reporting engineering success."""
        pairs = json.loads((self.directory/'pairs.json').read_text())
        expected_requests = 4 * len(pairs['pairs'])
        required=['pairs.json','selected-config.json','trace_off-api.jsonl','trace_on-api.jsonl',
                  'trace_off-responses.jsonl','trace_on-responses.jsonl','trace_off-failures.json',
                  'trace_on-failures.json','trace_off-output-differences.json',
                  'trace_on-output-differences.json','compressor-contract.json',
                  'trace-validation.json','sidecar.json','restoration.json']
        for scope in ('per_rank','aggregate'):
            for capacity in (64,128):
                label=f'{scope}-{capacity}mib'
                required += [f'cache-{label}.json',f'replay-{label}/report.json',f'replay-{label}/events.jsonl']
        require(all((self.directory/name).is_file() for name in required), 'Missing engineering evidence')
        require(json.loads((self.directory/'restoration.json').read_text()).get('restored') is True,
                'Original service has not been restored')
        require(json.loads((self.directory/'compressor-contract.json').read_text()).get('passed') is True,
                'Compressor contract failed')
        for phase in ('trace_off','trace_on'):
            require(len((self.directory/(phase+'-responses.jsonl')).read_text().splitlines())==expected_requests,
                    f'Incomplete {phase} responses')
            require(len((self.directory/(phase+'-api.jsonl')).read_text().splitlines())==expected_requests,
                    f'Incomplete {phase} API evidence')
            require(not json.loads((self.directory/(phase+'-failures.json')).read_text()),
                    f'{phase} has hard request failures')
        trace=json.loads((self.directory/'trace-validation.json').read_text())
        require(trace['requests']==expected_requests and trace['window_rows']==expected_requests*33*21 and trace['causal_scope_checked'],
                'Native trace coverage or causal contract failed')
        sidecar=json.loads((self.directory/'sidecar.json').read_text())
        require(sidecar['selection_provenance']=='real_npu_native' and len(sidecar['events'])==expected_requests,
                'Sidecar provenance or event count failed')
        for scope in ('per_rank','aggregate'):
            for capacity in (64,128):
                label=f'{scope}-{capacity}mib'
                path=self.directory/f'replay-{label}'
                replay=json.loads((path/'report.json').read_text())
                require(replay['config']['budget_scope']==scope and replay['config']['capacity_bytes']==capacity*1024**2,
                        'Replay configuration mismatch')
                require(replay['training_events']==expected_requests//2 and replay['evaluation_events']==expected_requests//2 and
                        len((path/'events.jsonl').read_text().splitlines())==expected_requests*14,
                        'Incomplete replay result')
                require(all(sha256_file(Path(name))==digest for name,digest in replay['input_sha256'].items()),
                        'Replay input hash mismatch')
        return True

    def report(self,status,error=None):
        expected = ['pairs.json','trace_off-responses.jsonl','trace_on-responses.jsonl','compressor-contract.json',
                    'trace-validation.json','sidecar.json','restoration.json']
        expected += [f'replay-{scope}-{cap}mib/report.json' for scope in ('per_rank','aggregate') for cap in (64,128)]
        if self.diagnostic_only:
            expected = ['pairs.json','diagnostics.json','selected-config.json','restoration.json','compressor-contract.json']
        restoration = json.loads((self.directory/'restoration.json').read_text()) if (self.directory/'restoration.json').exists() else {'restored':False}
        trace_path=self.directory/'trace-validation.json'
        trace=json.loads(trace_path.read_text()) if trace_path.exists() else {}
        differences={phase:json.loads((self.directory/(phase+'-output-differences.json')).read_text())
                     if (self.directory/(phase+'-output-differences.json')).exists() else []
                     for phase in ('trace_off','trace_on')}
        data={'run_id':self.run_id,'status':status,'error':error,'completed_stages':self.completed,
              'validation_mode':self.validation_mode,
              'strict_output_acceptance':'not_qualified' if self.validation_mode=='trace-replay' else
                  ('passed' if status=='passed' and not self.diagnostic_only else 'not_qualified'),
              'output_difference_counts':{phase:len(rows) for phase,rows in differences.items()},
              'output_differences':differences,
              'selected_set_consistency':{key:trace.get(key) for key in
                  ('repeat_selected_ids_identical','pair_prefix_selected_ids_identical',
                   'repeat_selected_id_differences','pair_prefix_selected_id_differences',
                   'paired_effect_interpretation')},
              'missing_artifacts':[p for p in expected if not (self.directory/p).exists()], 'restoration':restoration,
              'evidence_level':'synthetic eight-NPU DeepSeek engineering validation',
              'selection_provenance':'real_npu_native' if 'native_trace_and_sidecar' in self.completed else 'not_validated',
              'online_kv_restore':False,'requests_per_phase':4*len(json.loads((self.directory/'pairs.json').read_text())['pairs'])
                  if (self.directory/'pairs.json').exists() else None,'window_tokens':32,'final_sync_required':True,
              'completed_requests_per_phase':{phase:len((self.directory/(phase+'-responses.jsonl')).read_text().splitlines())
                    if (self.directory/(phase+'-responses.jsonl')).exists() else 0 for phase in ('trace_off','trace_on')},
              'replay_reports':[p for p in expected if p.startswith('replay-')],
              'diagnostic_only':self.diagnostic_only,'diagnostics':self.diagnostics,
              'repair':self.repair,'operator_patch_applied':bool(self.operator_repairs),
              'operator_root_cause_confirmed':False,'operator_repairs':self.operator_repairs,
              'selected_configuration':self.selected_config,'request_failures':self.failures,
              'production_configuration_validated':False,
              'limitations':['Synthetic 8K/32K grouping is not real-task generalization evidence.',
                             'Logical inventory is not online KV restoration; CPU replay does not measure DMA stalls.',
                             'Indexer, SWA, C128, compressor state and communication are outside the 1024-byte C4 payload model.']}
        write_json(self.directory/'report.json',data)
        (self.directory/'report.md').write_text('# 八卡 DeepSeek 工程验证\n\n状态：'+status+'\n\n完成阶段：'+', '.join(self.completed)+
            '\n\n原服务恢复：'+str(restoration.get('restored',False))+'\n\n缺失产物：'+', '.join(data['missing_artifacts'])+
            '\n\n选定实验配置：'+str((self.selected_config or {}).get('id','无'))+
            '\n\n诊断结果：'+', '.join(r['configuration']['id']+'='+r['status'] for r in self.diagnostics)+
            '\n\n'+(error or ('仅完成稳定性诊断。' if self.diagnostic_only else '四组预算、七种策略、native/page128 回放详见各 replay 目录。'))+
            '\n\n证据仅限合成输入工程验证，不证明任务泛化、在线 offload 或 DMA stall 收益。\n')

    def run(self):
        self.update()
        thread=threading.Thread(target=self.heartbeat,daemon=True)
        thread.start()
        signal.signal(signal.SIGTERM,terminate_worker)
        signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TaskDeadline('Six-hour task limit exceeded')))
        signal.alarm(self.total_seconds if hasattr(self, 'total_seconds') else TOTAL_SECONDS)
        error=None
        try:
            self.checkpoint('bootstrap',[self.directory/n for n in ('launch.json','code-provenance.json','implementation.bundle','deployment.json')])
            layout=self.preflight()
            self.completed.append('preflight')
            pairs=self.prepare()
            profile=load_profile('deepseek_v4')
            from_ranges = [(p[v]['boundary_position']-1,p[v]['boundary_position']+WINDOW-1) for p in pairs['pairs'] for v in ('event','control')]
            ranges=sorted(set(from_ranges))
            self.checkpoint('inputs',[self.directory/n for n in ('original-service.json','original-models.json','preflight.json','identity.json','layout.json','npu-before.txt','pairs.json','prepare.log')]+list((self.directory/'sources').glob('*'))+list((self.directory/'capture-source').glob('*')))
            if self.repair:
                self.checkpoint('experimental_sources',[self.directory/'experimental-provenance.json',
                    *[p for d in ('repair-source','operator-source') for p in (self.directory/d).rglob('*') if p.is_file()]])
            self.completed.append('inputs')
            self.pause_original()
            self.contract()
            self.completed.append('npu_compressor_contract')
            if self.validation_mode=='trace-replay':
                baseline=self.trace_replay_baseline(pairs,profile)
            else:
                baseline=self.choose_configuration(pairs,profile)
                self.completed.append('stability_diagnostics')
            if self.diagnostic_only:
                self.restore()
                self.completed.append('restore_original')
                self.report('passed')
                self.checkpoint('report',[self.directory/n for n in ('report.json','report.md','selected-config.json','restoration.json')])
                raise DiagnosticComplete()
            self.checkpoint('trace_off',[self.directory/n for n in ('trace_off-responses.jsonl','trace_off-api.jsonl','selected-config.json')])
            self.completed.append('trace_off')
            self.start_trace(ranges)
            responses=self.requests('trace_on',pairs,profile,baseline,
                                    soft_output_differences=self.validation_mode=='trace-replay')
            self.command(['docker','logs',self.container],output=self.directory/'trace_on-service.log',timeout=60)
            self.restore()  # Restore before validation, synchronization or CPU work.
            self.completed += ['trace_on','restore_original']
            evidence=validate_native_trace(self.directory/'raw-traces',self.directory/'traces',responses,ranges,profile,
                                           allow_selected_drift=self.validation_mode=='trace-replay')
            write_json(self.directory/'trace-validation.json',evidence)
            support_evidence=next(e for e in layout['source_evidence'] if e['path'].endswith('compressor_kernel.h'))
            sidecar=export_sidecar(responses,pairs,profile,evidence,support_evidence)
            snapshots,_=validate_sidecar(sidecar,self.directory)
            read_trace(self.directory/'traces',snapshots,sidecar['events'],synthetic=False)
            write_json(self.directory/'sidecar.json',sidecar)
            del snapshots,sidecar
            self.completed.append('native_trace_and_sidecar')
            paths=[self.directory/n for n in ('trace_on-responses.jsonl','trace_on-api.jsonl','trace_on-service.log','trace-launch.json',
                   'restoration.json','resource-release.json','trace-validation.json','sidecar.json','compressor-contract.json','compressor-contract.log')]
            self.checkpoint('npu_capture',paths+list((self.directory/'raw-traces').glob('*.jsonl'))+list((self.directory/'traces').glob('*.jsonl')))
            for scope in ('per_rank','aggregate'):
                for capacity in (64,128):
                    label=f'{scope}-{capacity}mib'
                    self.update(stage='cpu_replay_'+label)
                    config={'schema_version':1,'capacity_bytes':capacity*1024**2,'prefetch_budget_bytes':8*1024**2,
                            'budget_scope':scope,'kv_unit_bytes':{'compressed_kv_token_position':1024},'max_tables':256,'bootstrap_draws':1000,'seed':0}
                    path=self.directory/('cache-'+label+'.json')
                    write_json(path,config)
                    output=self.directory/('replay-'+label)
                    self.command(['python3',str(self.code_root/'m0a/transition_replay.py'),'--trace-dir',str(self.directory/'traces'),
                                  '--sidecar',str(self.directory/'sidecar.json'),'--config',str(path),'--source-root',str(self.directory),
                                  '--output-dir',str(output)],timeout=TOTAL_SECONDS,output=self.directory/('replay-'+label+'.log'))
                    replay_report_path=output/'report.json'
                    replay_report=json.loads(replay_report_path.read_text())
                    replay_report['paired_effect_interpretation']=evidence['paired_effect_interpretation']
                    if evidence['paired_effect_interpretation']=='exploratory':
                        replay_report['limitations'].append('Selected sets drift across repeats or paired prefixes; paired effects are exploratory.')
                    write_json(replay_report_path,replay_report)
                    self.checkpoint('replay_'+label,[path,output/'events.jsonl',output/'report.json',self.directory/('replay-'+label+'.log')])
                    self.completed.append('replay_'+label)
            if self.validation_mode=='trace-replay':
                self.validate_engineering_evidence()
                self.report('engineering_validated')
            else:
                self.report('passed')
            self.checkpoint('report',[self.directory/'report.json',self.directory/'report.md'])
        except DiagnosticComplete:
            pass
        except BaseException as failure:
            traceback.print_exc()
            error=str(failure)
            self.update(status='failed',stage='failure',error=error)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGTERM,signal.SIG_IGN)
            # Recovery gets a fresh budget even after the six-hour run deadline.
            signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TaskDeadline('Restoration budget exceeded')))
            signal.alarm(1800)
            try:
                if self.original and (self.restore_required or not (self.directory/'restoration.json').exists()):
                    self.restore()
                else:
                    self.cleanup()
            except BaseException as failure:
                traceback.print_exc()
                error=(error+'; ' if error else '')+'Restoration failed: '+str(failure)
                write_json(self.directory/'restoration.json',{'at':utc(),'restored':False,'error':str(failure),'original_container_id':self.original['Id'] if self.original else None})
            finally:
                signal.alarm(0)
            if error:
                self.report('failed',error)
            self.update(status='failed' if error else ('engineering_validated' if self.validation_mode=='trace-replay' else 'passed'),
                        stage='finished',error=error,completed_stages=self.completed,finished_at=utc())
            self.stop.set()
            thread.join(timeout=1)
        return 1 if error else 0


def watchdog(root,run_id):
    worker=Worker(root,run_id)
    deadline=time.monotonic()+TOTAL_SECONDS+1800
    while time.monotonic()<deadline:
        state=json.loads((worker.directory/'status.json').read_text())
        try:
            alive=Path(f'/proc/{state["worker_pid"]}/stat').read_text().split()[2]!='Z'
        except FileNotFoundError:
            alive=False
        if state['stage']=='finished' and state.get('original_service_restored'):
            return 0
        if not alive:
            break
        time.sleep(15)
    else:
        os.kill(state['worker_pid'],signal.SIGTERM)
        time.sleep(15)
        try: os.kill(state['worker_pid'],signal.SIGKILL)
        except ProcessLookupError: pass
    worker.original=json.loads((worker.directory/'original-service.json').read_text())
    worker.restore_required=not worker.inspect(worker.original['Id'])['State']['Running']
    # Clean only this task's owned containers, including an interrupted helper.
    names=worker.command(['docker','ps','-a','--filter','label=memecho.validation.run='+run_id,'--format','{{.Names}}']).splitlines()
    worker.owned.update(names)
    error='Worker exited unexpectedly; independent watchdog recovery'
    try:
        worker.restore()
    except BaseException as failure:
        error+='; restoration failed: '+str(failure)
        write_json(worker.directory/'restoration.json',{'restored':False,'error':str(failure),'at':utc()})
    worker.completed=state.get('completed_stages',[])
    if (worker.directory/'diagnostics.json').exists():
        worker.diagnostics=json.loads((worker.directory/'diagnostics.json').read_text())
    if (worker.directory/'selected-config.json').exists():
        worker.selected_config=json.loads((worker.directory/'selected-config.json').read_text())
    if (worker.directory/'operator-repairs.json').exists():
        worker.operator_repairs=json.loads((worker.directory/'operator-repairs.json').read_text())
    worker.report('failed',error)
    restored = json.loads((worker.directory/'restoration.json').read_text()).get('restored',False)
    worker.update(**dict(state,status='failed',stage='finished',error=error,finished_at=utc(),original_service_restored=restored))
    write_json(worker.directory/'watchdog-recovery.json',{'at':utc(),'error':error})
    return 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/workspace/memecho'))
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--watchdog',action='store_true')
    args=parser.parse_args()
    if args.prepare:
        profile=load_profile('deepseek_v4')
        model=args.root/profile['model_path']
        chat=make_chat_tokenizer(profile_name='deepseek_v4',tokenizer_path=model/'tokenizer.json',model_dir=model)
        source=Path(__file__).resolve().parents[1]/'m0a/pairs.json'
        data=json.loads(source.read_text())
        require(sha256_file(model/'tokenizer.json') == data['tokenizer_json_sha256'],'Pinned tokenizer hash mismatch')
        result=extend_pairs(data,chat,profile)
        result['source_file_sha256']=sha256_file(source)
        write_json(args.root/'m0a/runs'/args.run_id/'pairs.json',result)
        return 0
    if args.watchdog:
        return watchdog(args.root,args.run_id)
    return Worker(args.root,args.run_id).run()


if __name__=='__main__':
    raise SystemExit(main())
