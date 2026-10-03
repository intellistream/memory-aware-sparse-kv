"""Eight-card CP coverage, exact inventories, budgets and service recovery regressions."""
import copy
import json
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from contracts import sha256_file
from deepseek_validation import (Worker, extend_pairs, validate_cp, validate_native_trace, export_sidecar,
                                 port_available, clone_args, source_evidence, write_json)
from import_workloads import sha_ids
from model_profiles import load_profile
from transition_replay import read_trace, run_replay, validate_sidecar, replay_event
from working_set import LRU, RankCaches, TransitionTable, support, validate_snapshot


def chat(text): return [0,*text.encode(),999]


def corpus():
    data={'schema_version':1,'tokenizer_json_sha256':'0'*64,'pairs':[]}
    for context in (8192,32768):
        for index in range(6):
            pair={'pair_id':f'{context}-{index}','workload_id':'synthetic','episode_id':f'{context}-{index}',
                  'context_target':context,'event_type':'tool_call'}
            for variant,text in [('event','p'*63+'E'+'q'*70),('control','p'*63+'Cxx'+'q'*70)]:
                ids=chat(text)
                pair[variant]={'prompt':text,'boundary_position':64,'prompt_tokens_expected':len(ids),
                               'prompt_token_ids_sha256_expected':sha_ids(ids)}
            data['pairs'].append(pair)
    return data


def traces(root,pairs):
    raw=root/'raw'
    raw.mkdir()
    outputs={r:(raw/f'rank{r}.jsonl').open('w') for r in range(8)}
    responses=[]
    for pair in pairs['pairs']:
        for repetition in range(2):
            for variant in ('event','control'):
                item=pair[variant]
                response={'request_id':f'{pair["pair_id"]}-{variant}-{repetition}','run_id':'test',
                          'pair_id':pair['pair_id'],'variant':variant,'repetition':repetition,
                          'boundary_position':64,'prompt_tokens':len(item['prompt_token_ids']),
                          'prompt_token_ids':item['prompt_token_ids']}
                responses.append(response)
                for pos in range(63,96):
                    chunk=0 if pos<64 else 64
                    index=pos-chunk
                    rank=index//8
                    for layer in range(2,43,2):
                        ids=[0,1,pos//4-1]+[-1]*509
                        row={'schema_version':1,'run_id':'test','request_id':response['request_id'],
                             'rank':rank,'layer':f'model.layers.{layer}.self_attn.attn','prompt_position':pos,
                             'context_len':pos+1,'request_context_len':response['prompt_tokens'],
                             'raw_index_unit':'compressed_kv_token_position','invalid_sentinel':-1,
                             'compression_ratio':4,'compressed_block_size':128,'selected_width':512,
                             'raw_selected_ids':ids,'logical_compressed_block_ids':[u//128 if u>=0 else -1 for u in ids],
                             'timestamp_ns':1,'cp_world_size':8,'cp_local_start':rank*8,'cp_local_end':rank*8+8,
                             'chunk_start_position':chunk,'chunk_token_count':64,
                             'chunk_context_len':chunk+64,'query_global_index':index}
                        outputs[rank].write(json.dumps(row)+'\n')
    for output in outputs.values(): output.close()
    return responses


class DeepSeekTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temp.name)
        cls.profile=load_profile('deepseek_v4')
        cls.pairs=extend_pairs(corpus(),chat,cls.profile)
        cls.responses=traces(cls.root,cls.pairs)
        cls.evidence=validate_native_trace(cls.root/'raw',cls.root/'traces',cls.responses,[(63,95)],cls.profile)
        source=cls.root/'kernel.h'
        source.write_text('// C4 overlap: previous four and current four tokens.\n')
        cls.sidecar=export_sidecar(cls.responses,cls.pairs,cls.profile,cls.evidence,
                                  source_evidence(source,['overlap'],cls.root))
        cls.snapshots,cls.groups=validate_sidecar(cls.sidecar,cls.root)
        cls.rows=read_trace(cls.root/'traces',cls.snapshots,cls.sidecar['events'],synthetic=False)

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_strict_equal_length_and_original_prefix(self):
        for original,pair in zip(corpus()['pairs'],self.pairs['pairs']):
            self.assertEqual(pair['event']['prompt_tokens_expected'],pair['control']['prompt_tokens_expected'])
            for v in ('event','control'):
                self.assertEqual(pair[v]['prompt_token_ids'][:64],chat(original[v]['prompt'])[:64])
                self.assertGreaterEqual(len(pair[v]['prompt_token_ids'])-64,32)
                self.assertEqual(pair[v]['prompt_token_ids_sha256_expected'],sha_ids(chat(pair[v]['prompt'])))
        self.assertTrue(self.pairs['length_equalization']['changes'])

    def test_pinned_tokens_rejected(self):
        data=corpus()
        data['pairs'][0]['event']['prompt_token_ids_sha256_expected']='1'*64
        with self.assertRaisesRegex(ValueError,'tokenizer contract'): extend_pairs(data,chat,self.profile)

    def test_cp_cross_rank_cross_chunk_union(self):
        self.assertEqual(self.evidence['window_rows'],48*33*21)
        self.assertEqual(self.evidence['ranks_with_rows'],[0,1,2,3,7])
        event=self.sidecar['events'][0]
        for pos,owner in ((63,7),(64,0),(72,1),(80,2),(88,3)):
            self.assertEqual(set(event['query_assignment'][str(pos)].values()),{owner})
        self.assertEqual(len(self.rows[event['event_id']]),33*21)

    def test_selected_drift_is_diagnostic_only_in_trace_replay(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            raw=root/'raw';raw.mkdir()
            target=next(r['request_id'] for r in self.responses if r['variant']=='control' and r['repetition']==0)
            modified=False
            for source in (self.root/'raw').glob('rank*.jsonl'):
                if source.name!='rank7.jsonl':
                    (raw/source.name).symlink_to(source)
                    continue
                with (raw/source.name).open('w') as out, source.open() as incoming:
                    for line in incoming:
                        row=json.loads(line)
                        if row['request_id']==target and row['prompt_position']==63 and row['layer'].endswith('2.self_attn.attn'):
                            row['raw_selected_ids'][1]=4
                            line=json.dumps(row)+'\n'
                            modified=True
                        out.write(line)
            self.assertTrue(modified)
            with self.assertRaisesRegex(ValueError,'selected IDs differ'):
                validate_native_trace(raw,root/'strict',self.responses,[(63,95)],self.profile)
            evidence=validate_native_trace(raw,root/'replay',self.responses,[(63,95)],self.profile,
                                           allow_selected_drift=True)
            self.assertEqual(evidence['requests'],48)
            self.assertGreater(evidence['pair_prefix_selected_id_differences'],0)
            self.assertGreater(evidence['repeat_selected_id_differences'],0)
            self.assertEqual(evidence['paired_effect_interpretation'],'exploratory')
            sidecar=export_sidecar(self.responses,self.pairs,self.profile,evidence,
                                  self.sidecar['snapshots'][0]['lanes'][0]['support_evidence'])
            snapshots,_=validate_sidecar(sidecar,self.root)
            rows=read_trace(root/'replay',snapshots,sidecar['events'],synthetic=False)
            self.assertIn(4,next(row['raw_selected_ids'] for row in rows[target].values()
                                 if row['prompt_position']==63 and row['layer'].endswith('2.self_attn.attn')))
            path=raw/'rank7.jsonl'
            lines=path.read_text().splitlines()
            bad=json.loads(lines[0]);bad['cp_local_end']=999
            lines[0]=json.dumps(bad)
            path.write_text('\n'.join(lines)+'\n')
            with self.assertRaisesRegex(ValueError,'CP ownership interval mismatch'):
                validate_native_trace(raw,root/'invalid',self.responses,[(63,95)],self.profile,
                                      allow_selected_drift=True)

    def test_bad_cp_owner_and_chunk_rejected(self):
        row=next(iter(self.rows[self.sidecar['events'][0]['event_id']].values()))
        for field,value in [('rank',6),('chunk_token_count',0),('query_global_index',-1),
                            ('cp_local_end',999),('chunk_context_len',0)]:
            changed=dict(row,**{field:value})
            with self.assertRaises(ValueError): validate_cp(changed)

    def test_cp_chunk_accepts_only_exact_eight_rank_padding(self):
        row={'rank':7,'cp_world_size':8,'cp_local_start':7147,'cp_local_end':8168,
             'chunk_start_position':0,'chunk_token_count':8168,'chunk_context_len':8165,
             'request_context_len':8165,'query_global_index':8129,'prompt_position':8129}
        validate_cp(row)
        for change in ({'chunk_context_len':8160}, {'chunk_token_count':8176},
                       {'prompt_position':8165}, {'request_context_len':8164}):
            with self.assertRaises(ValueError): validate_cp(dict(row,**change))

    def test_missing_cp_row_and_assignment_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            for source in (self.root/'traces').glob('*.jsonl'):
                lines=source.read_text().splitlines()
                if source.name=='rank0.jsonl': lines=lines[1:]
                (root/source.name).write_text('\n'.join(lines)+'\n' if lines else '')
            with self.assertRaisesRegex(ValueError,'Missing CP owner row'):
                read_trace(root,self.snapshots,self.sidecar['events'],synthetic=False)

    def test_support_first_overlap_and_full_rank_inventory(self):
        snapshot=self.sidecar['snapshots'][1]
        self.assertEqual(len(snapshot['lanes']),8*21)
        for lane in snapshot['lanes']:
            self.assertEqual(lane['generated_ids'],list(range(16)))
            self.assertEqual(support(snapshot,lane,0),(0,4,3))
            self.assertEqual(support(snapshot,lane,1),(0,8,7))
            self.assertEqual(support(snapshot,lane,2),(4,12,11))
        changed=copy.deepcopy(snapshot)
        changed['lanes'][5]['generated_ids'].pop()
        with self.assertRaisesRegex(ValueError,'inventory'): validate_snapshot(changed,self.root)

    def test_non_group_prefix_inventory(self):
        snapshot=copy.deepcopy(self.sidecar['snapshots'][1])
        snapshot['generated_tokens']=63
        for lane in snapshot['lanes']: lane['generated_ids']=list(range(15))
        validate_snapshot(snapshot,self.root)
        snapshot['lanes'][0]['generated_ids'].append(15)
        with self.assertRaisesRegex(ValueError,'inventory'): validate_snapshot(snapshot,self.root)

    def test_group_isolation(self):
        changed=copy.deepcopy(self.sidecar)
        changed['events'][1]['split']='eval'
        with self.assertRaisesRegex(ValueError,'leakage'): validate_sidecar(changed,self.root)
        self.assertEqual(len(set(self.groups.values())),12)

    def test_rank_cache_capacity_is_independent(self):
        cache=RankCaches(8)
        key=('s','m','r','t','v',0,'l','compressed_kv_token_position',0)
        other=key[:5]+(1,)+key[6:]
        cache.put(key,8,{0})
        cache.put(other,8,{0})
        self.assertEqual(cache.used,16)
        self.assertTrue(cache.contains(key,0) and cache.contains(other,0))
        unified=LRU(8)
        unified.put(key,8,{0})
        unified.put(other,8,{0})
        self.assertFalse(unified.contains(key,0))

    def test_cp_readiness_and_both_budget_scopes(self):
        event=self.sidecar['events'][0]
        cur,old=self.snapshots[event['snapshot_id']],self.snapshots[event['previous_snapshot_id']]
        for scope,kind in [('per_rank',RankCaches),('aggregate',LRU)]:
            cfg={'capacity_bytes':64*1024**2,'prefetch_budget_bytes':8*1024,'budget_scope':scope,
                 'kv_unit_bytes':{'compressed_kv_token_position':1024}}
            for granularity in ('native','page128'):
                result=replay_event('sequential_selected_set',granularity,event,old,cur,self.rows[event['event_id']],
                                    kind(cfg['capacity_bytes']),cfg,TransitionTable(),TransitionTable(),'tool_call')
                self.assertEqual(result['attention_readiness_assertions'],32*21)
                self.assertEqual(result['synchronous_recall_bytes'],sum(v['synchronous_recall_bytes'] for v in result['per_rank'].values()))
                if scope=='aggregate': self.assertLessEqual(result['prefetched_bytes'],cfg['prefetch_budget_bytes'])
                for rank in result['per_rank'].values(): self.assertLessEqual(rank['prefetched_bytes'],cfg['prefetch_budget_bytes'])

    def test_live_listener_and_time_wait_restart(self):
        listener=socket.socket()
        listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        listener.bind(('127.0.0.1',0))
        port=listener.getsockname()[1]
        listener.listen()
        self.assertFalse(port_available(port))
        probe,_=listener.accept()
        probe.close()
        client=socket.create_connection(('127.0.0.1',port))
        conn,_=listener.accept()
        conn.close()  # Server actively closes, creating TIME_WAIT on this port.
        client.recv(1)
        client.close()
        listener.close()
        self.assertTrue(port_available(port))
        restarted=socket.socket()
        restarted.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        restarted.bind(('127.0.0.1',port))
        restarted.listen()
        self.assertFalse(port_available(port))
        restarted.close()

    def test_clone_keeps_mtp_cp_communication_devices(self):
        metadata={'Config':{'Cmd':['vllm','serve','model','--speculative-config','mtp','--additional-config','dsa_cp'],
                            'Env':['HCCL_BUFFSIZE=1024'],'WorkingDir':'/workspace'},
                  'HostConfig':{'NetworkMode':'host','AutoRemove':False,'IpcMode':'private','ShmSize':100,'Privileged':True,
                                'SecurityOpt':['label=disable'],'Binds':['/a:/a'],'Devices':[{'PathOnHost':'/dev/davinci0','PathInContainer':'/dev/davinci0','CgroupPermissions':'rwm'}]}}
        args=clone_args(metadata,'test','id','image',env_extra={'TRACE':'1'})
        self.assertEqual(args[-len(metadata['Config']['Cmd']):],metadata['Config']['Cmd'])
        self.assertIn('HCCL_BUFFSIZE=1024',args)
        self.assertIn('/dev/davinci0:/dev/davinci0:rwm',args)
        self.assertIn('memecho.validation.run=id',args)

    def test_four_configurations_all_strategies_are_lossless(self):
        selected=[self.sidecar['events'][i] for i in (0,1,24,25)]
        request_ids={e['request_id'] for e in selected}
        snapshot_ids={e[field] for e in selected for field in ('snapshot_id','previous_snapshot_id','post_window_snapshot_id')}
        sidecar=dict(self.sidecar,events=selected,snapshots=[s for s in self.sidecar['snapshots'] if s['snapshot_id'] in snapshot_ids])
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            outputs={rank:(root/f'rank{rank}.jsonl').open('w') for rank in range(8)}
            for event in selected:
                for row in self.rows[event['event_id']].values(): outputs[row['rank']].write(json.dumps(row)+'\n')
            for stream in outputs.values(): stream.close()
            for scope in ('per_rank','aggregate'):
                for capacity in (64,128):
                    cfg={'schema_version':1,'capacity_bytes':capacity*1024**2,'prefetch_budget_bytes':8*1024**2,
                         'budget_scope':scope,'kv_unit_bytes':{'compressed_kv_token_position':1024},'bootstrap_draws':10}
                    report=run_replay(root,sidecar,cfg,self.root)
                    self.assertEqual(len(report['results']),4*14)
                    self.assertEqual(report['evaluation_table_updates'],0)
                    self.assertTrue(all(r['attention_readiness_assertions']==32*21 for r in report['results']))
                    self.assertTrue(all(r['useful_prefetch_bytes']<=r['prefetched_bytes'] for r in report['results']))

    def test_command_timeout_does_not_block_restoration(self):
        with tempfile.TemporaryDirectory() as name:
            worker=Worker(Path(name),'deepseek_20260929T000000Z_12345678')
            with self.assertRaises(subprocess.TimeoutExpired):
                worker.command([sys.executable,'-c','import time; time.sleep(30)'],timeout=.05)

    def test_restore_cleans_owned_container_before_starting_original(self):
        with tempfile.TemporaryDirectory() as name:
            worker=Worker(Path(name),'deepseek_20260929T000000Z_12345678')
            worker.directory.mkdir(parents=True)
            worker.original={'Id':'original','Config':{'x':1},'HostConfig':{'x':2}}
            worker.restore_required=True
            actions=[]
            with patch.object(worker,'cleanup',side_effect=lambda:actions.append('cleanup')), \
                    patch.object(worker,'wait_release',side_effect=lambda:actions.append('release')), \
                    patch.object(worker,'command',side_effect=lambda args,**kw:actions.append(args)), \
                    patch.object(worker,'service_health',return_value=[('dsv4','model')]), \
                    patch.object(worker,'inspect',return_value=worker.original):
                worker.restore()
            self.assertEqual(actions,['cleanup','release',['docker','start','original']])
            self.assertFalse(worker.restore_required)
            self.assertTrue(json.loads((worker.directory/'restoration.json').read_text())['restored'])

    def test_default_launcher_has_no_remote_side_effect(self):
        root=Path(__file__).resolve().parents[1]
        output=subprocess.check_output([sys.executable,str(root/'scripts/launch_deepseek_validation.py')],text=True)
        plan=json.loads(output)
        self.assertFalse(plan['executed'])
        self.assertFalse(Path(plan['local_directory']).exists())
        self.assertEqual(plan['devices'],list(range(8)))

    def test_exception_restores_original_and_writes_failure_report(self):
        with tempfile.TemporaryDirectory() as name:
            worker=Worker(Path(name),'deepseek_20260929T000000Z_12345678')
            worker.directory.mkdir(parents=True)
            def preflight():
                worker.original={'Id':'original'}
                worker.restore_required=True
                return {}
            def restore():
                write_json(worker.directory/'restoration.json',{'restored':True,'container_id':'original'})
            with patch.object(worker,'checkpoint'),patch.object(worker,'preflight',side_effect=preflight), \
                    patch.object(worker,'prepare',side_effect=RuntimeError('sync/token failure')),patch.object(worker,'restore',side_effect=restore) as restored:
                self.assertEqual(worker.run(),1)
            restored.assert_called_once()
            report=json.loads((worker.directory/'report.json').read_text())
            self.assertEqual(report['status'],'failed')
            self.assertTrue(report['restoration']['restored'])
            self.assertIn('sidecar.json',report['missing_artifacts'])

    def test_restoration_failure_is_separate_evidence(self):
        with tempfile.TemporaryDirectory() as name:
            worker=Worker(Path(name),'deepseek_20260929T000000Z_12345678')
            worker.directory.mkdir(parents=True)
            def preflight():
                worker.original={'Id':'original'}
                worker.restore_required=True
                raise RuntimeError('capture failed')
            with patch.object(worker,'checkpoint'),patch.object(worker,'preflight',side_effect=preflight), \
                    patch.object(worker,'restore',side_effect=RuntimeError('health failed')):
                worker.run()
            report=json.loads((worker.directory/'report.json').read_text())
            self.assertFalse(report['restoration']['restored'])
            self.assertIn('Restoration failed',report['error'])


if __name__=='__main__': unittest.main()
