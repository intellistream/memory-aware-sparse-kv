"""Cancellation, deployed state contracts and evidence-based attribution."""
import copy
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
import importlib.util
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from deepseek_compressor_contract import expected_slots, pinned_block_sizes, ready_count, state_layout, deployment_state_spec, padded_slot
from deepseek_operator_diagnostics import classify_difference, snapshot_call, restore_call, install_model, install_dsa, meaningful_result
from deepseek_repair import (configure_worker, instrument_source, patch_worker, validate_deterministic_ranks,
                             new_request_physical_blocks, patch_compressor_initialization, verify_state_repair)
from deepseek_validation import TaskTerminated, stability_candidates
from test_deepseek_stability import COMMAND


class RepairTests(unittest.TestCase):
    def test_sigterm_exception_bypasses_request_and_candidate_retries(self):
        self.assertTrue(issubclass(TaskTerminated,BaseException))
        self.assertFalse(issubclass(TaskTerminated,Exception))
        with tempfile.TemporaryDirectory() as root:
            code = textwrap.dedent('''
                import json,os,signal,sys
                from pathlib import Path
                sys.path.insert(0,sys.argv[1])
                from deepseek_validation import Worker,write_json
                from test_deepseek_stability import COMMAND,pair
                worker=Worker(Path(sys.argv[2]),'deepseek_20260929T000000Z_12345678')
                worker.directory.mkdir(parents=True)
                worker.original={'Id':'original','Config':{'Cmd':COMMAND}}
                worker.restore_required=True
                actions=[]
                worker.checkpoint=lambda *a:None
                worker.preflight=lambda:{}
                worker.prepare=lambda:{'pairs':[pair()]}
                worker.pause_original=lambda:None
                worker.contract=lambda:None
                worker.start_candidate=lambda c,p:actions.append('candidate:'+c['id'])
                worker.command=lambda *a,**kw:''
                worker.cleanup=lambda:actions.append('cleanup')
                worker.wait_release=lambda:None
                def restore():
                    actions.append('restore')
                    worker.restore_required=False
                    write_json(worker.directory/'restoration.json',{'restored':True,'container_id':'original'})
                    worker.update(original_service_restored=True)
                worker.restore=restore
                import deepseek_validation
                def request(*a,**kw):
                    actions.append('request')
                    os.kill(os.getpid(),signal.SIGTERM)
                    raise AssertionError('Signal was swallowed')
                deepseek_validation.request=request
                code=worker.run()
                (Path(sys.argv[2])/'actions.json').write_text(json.dumps(actions))
                raise SystemExit(code)
            ''')
            result = subprocess.run([sys.executable,'-c',code,str(Path(__file__).resolve().parent),root],
                                    capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,1,result.stderr)
            actions = json.loads((Path(root)/'actions.json').read_text())
            self.assertEqual([a for a in actions if a.startswith('candidate:')],['candidate:original_fresh'])
            self.assertEqual(actions.count('request'),1)
            self.assertEqual(actions.count('restore'),1)
            directory = Path(root)/'m0a/runs/deepseek_20260929T000000Z_12345678'
            report = json.loads((directory/'report.json').read_text())
            self.assertEqual(report['status'],'failed')
            self.assertTrue(report['restoration']['restored'])
            self.assertIn('SIGTERM',report['error'])
            self.assertEqual(json.loads((directory/'diagnostics/original_fresh/result.json').read_text())['status'],'interrupted')
            self.assertIn('termination',json.loads((directory/'diagnostics/original_fresh/stability-api.jsonl').read_text()))

    def test_actual_c4_state_views_have_distinct_padding(self):
        attention = state_layout(8,2048,131072)
        indexer = state_layout(8,512,16640)
        self.assertEqual(attention['operator_shape'],[6,8,2048])
        self.assertEqual(attention['operator_stride'],[32768,2048,1])
        self.assertEqual(attention['padding_bytes'],65536)
        self.assertEqual(indexer['operator_stride'],[4160,512,1])
        self.assertEqual(indexer['padding_bytes'],256)
        for args in ((8,512,16380),(8,512,16641),(0,512,16640)):
            with self.assertRaises(ValueError): state_layout(*args)

    def test_pinned_spec_method_uses_real_constructor_without_model_imports(self):
        from types import SimpleNamespace
        root=Path(__file__).resolve().parents[1]
        actual=Path('/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v4.py')
        retained=root/'m0a/runs/deepseek_20260929T090105Z_ac4ac7d2/sources/model.py'
        if actual.exists() or retained.exists():
            source=(actual if actual.exists() else retained).read_text()
        else:
            source='class AscendCompressorStateCache:\n def get_kv_cache_spec(self, config: MissingType):\n  pads=_dsv4_block_sizes()[config.cache_config.block_size][1]\n  return AscendSlidingWindowMLASpec(block_size=self.block_size,head_size=self.state_dim,page_size_padded=pads[0 if self.state_dim==512 else 1])\n'
        table={128:[[128,128,8,32],[16640,131072]]}
        for dim,padded in ((512,16640),(2048,131072)):
            method,obj=deployment_state_spec(source,SimpleNamespace,table,128,dim)
            spec=method(obj,SimpleNamespace(cache_config=SimpleNamespace(block_size=128)))
            self.assertEqual(spec.block_size,8)
            self.assertEqual(spec.head_size,dim)
            self.assertEqual(spec.page_size_padded,padded)
        with self.assertRaises(ValueError):deployment_state_spec('pass',SimpleNamespace,table,128,512)

    def test_compressor_fingerprints_exclude_incomplete_units_for_c4_and_c128(self):
        class Vector:
            def __init__(self,value):self.value=value
            def detach(self):return self
            def cpu(self):return self
            def tolist(self):return self.value
        for ratio in (4,128):
            kwargs={'start_pos':Vector([ratio-1]),'cu_seqlens':Vector([0,2]),'cmp_ratio':ratio}
            self.assertEqual(meaningful_result('_C_ascend.compressor.default',(),kwargs,[1,2,3]),[1])

    def test_pinned_layout_reader_rejects_ambiguous_or_changed_source(self):
        source='def f():\n _DSV4_BLOCK_SIZES={128:[[128,128,8,32],[16640,131072]]}\n return _DSV4_BLOCK_SIZES\n'
        self.assertEqual(pinned_block_sizes(source)[128][0][2],8)
        with self.assertRaises(ValueError): pinned_block_sizes('DSV4_BLOCK_SIZES={}')
        with self.assertRaises(ValueError): pinned_block_sizes(source+source)

    def test_non_group_chunks_cross_physical_pages_and_keep_sentinel_free(self):
        self.assertEqual(expected_slots([3,1,4],128,509,9),[[3,127],[1,0]])
        for start,count in ((0,3),(3,1),(5,3),(7,9),(509,9)):
            self.assertEqual(len(expected_slots([3,1,4],128,start,count)),ready_count(start,count))
            self.assertTrue(all(block != 0 for block,_ in expected_slots([3,1,4],128,start,count)))
        source='int32_t padOffset = static_cast<int32_t>(kvBlockSize_ - 1);\nslotLocal.SetValue(slotOffset, -1);\nslotLocal.SetValue(slotOffset + 1, padOffset);'
        self.assertEqual(padded_slot(128,source),[-1,127])
        self.assertEqual(padded_slot(64,source),[-1,63])
        with self.assertRaises(ValueError):padded_slot(128,source.replace('kvBlockSize_ - 1','0'))

    def test_only_repair_appends_two_determinism_candidates(self):
        original=copy.deepcopy(COMMAND)
        candidates=stability_candidates(original,repair=True)
        self.assertEqual(original,COMMAND)
        self.assertEqual(candidates[:6],stability_candidates(COMMAND))
        self.assertEqual(candidates[6]['environment'],{'HCCL_DETERMINISTIC':'true'})
        self.assertEqual(candidates[7]['environment'],{'HCCL_DETERMINISTIC':'true','MEMECHO_DETERMINISTIC_LEVEL':'1'})
        self.assertEqual(candidates[5]['command'],candidates[6]['command'])
        self.assertEqual(candidates[6]['command'],candidates[7]['command'])

    def test_setting_is_after_device_binding_before_communication_and_getter_checked(self):
        source='class W:\n def init(self):\n        torch.npu.set_device(device)\n        self._init_worker_distributed_environment()\n'
        result=patch_worker(source,'/tmp/runtime.py')
        self.assertLess(result.index('torch.npu.set_device'),result.index('_memecho_configure(self.rank, device, "device_bound")'))
        self.assertLess(result.index('_memecho_configure(self.rank, device, "device_bound")'),result.index('self._init_worker_distributed_environment()'))
        with self.assertRaises(ValueError): patch_worker(source+source,'/tmp/runtime.py')
        with tempfile.TemporaryDirectory() as root:
            from types import SimpleNamespace
            actions=[]
            module=SimpleNamespace(npu=SimpleNamespace(set_deterministic_level=lambda v:actions.append(v)),
                                   _C=SimpleNamespace(_npu_get_deterministic_level=lambda:1))
            environment={'MEMECHO_DETERMINISTIC_DIR':root,'MEMECHO_DETERMINISTIC_LEVEL':'1','HCCL_DETERMINISTIC':'true'}
            with patch.dict('sys.modules',torch_npu=module),patch.dict('os.environ',environment):
                for rank in range(8):
                    configure_worker(rank,f'npu:{rank}','device_bound')
                    configure_worker(rank,f'npu:{rank}','distributed_initialized')
            self.assertEqual(actions,[1]*8)
            candidate={'environment':{'MEMECHO_DETERMINISTIC_LEVEL':'1','HCCL_DETERMINISTIC':'true'}}
            self.assertEqual(validate_deterministic_ranks(root,candidate)['verified_ranks'],list(range(8)))
            path=Path(root)/'rank7-device_bound.json'
            data=json.loads(path.read_text());data['actual_level']=0;path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'NPU deterministic'):validate_deterministic_ranks(root,candidate)
            path.unlink()
            with self.assertRaises(FileNotFoundError):validate_deterministic_ranks(root,candidate)

    def test_diagnostics_keep_input_drift_and_incomplete_state_as_suspicions(self):
        original={'prefix':'p','input':'i','output':'o','complete':True}
        self.assertEqual(classify_difference(original,dict(original)),'identical')
        self.assertEqual(classify_difference(original,dict(original,prefix='q')),'different_token_prefix')
        self.assertEqual(classify_difference(original,dict(original,input='j',output='x')),'different_inputs_track_upstream')
        self.assertEqual(classify_difference(original,dict(original,output='x',complete=False)),'incomplete_state_no_attribution')
        self.assertEqual(classify_difference(original,dict(original,output='x')),'same_inputs_different_outputs')

    def test_default_repair_launcher_stays_dry(self):
        root=Path(__file__).resolve().parents[1]
        plan=json.loads(subprocess.check_output([sys.executable,str(root/'scripts/launch_deepseek_validation.py'),'--repair'],text=True))
        self.assertTrue(plan['repair'])
        self.assertFalse(plan['executed'])
        self.assertEqual(len(plan['candidate_order']),8)
        self.assertEqual(plan['requests_per_phase'],48)
        self.assertFalse(Path(plan['local_directory']).exists())

    def test_previous_barrier_rejects_active_or_unrestored_run(self):
        import scripts.launch_deepseek_validation as launcher
        from types import SimpleNamespace
        args=SimpleNamespace(host='test',remote_root='/remote')
        with patch.object(launcher,'remote',return_value=b'[]'):
            self.assertEqual(launcher.previous_run_barrier(args),{'prior_runs_checked':True})
        for reason in ('worker active','service restoration unverified'):
            rows=[{'run':'deepseek_old','reason':reason}]
            with patch.object(launcher,'remote',return_value=json.dumps(rows).encode()):
                with self.assertRaisesRegex(ValueError,'Prior Pod validation needs recovery'):
                    launcher.previous_run_barrier(args)

    def test_instrumentation_preserves_future_imports_and_compiles(self):
        source='from __future__ import annotations\nclass Model: pass\n'
        result=instrument_source(source,'/tmp/diagnostic.py','install_model')
        self.assertTrue(result.startswith(source))
        compile(result,'diagnostic.py','exec')
        with self.assertRaises(ValueError):instrument_source(source,'/tmp/diagnostic.py','unknown')

    def test_state_repair_only_initializes_new_requests_nonzero_owned_blocks(self):
        self.assertEqual(new_request_physical_blocks([[0,3,1,-1],[4,2]], [0,7]),[1,3])
        self.assertEqual(new_request_physical_blocks([[0,3],[1,3]], [0,0]),[1,3])
        self.assertEqual(new_request_physical_blocks([[2,3]], [8]),[])
        with self.assertRaises(ValueError):new_request_physical_blocks([[1]],[])
        source='def f():\n return torch.ops._C_ascend.compressor()\ndef g():\n return torch.ops._C_ascend.compressor()\n'
        self.assertEqual(patch_compressor_initialization(source,'/tmp/repair.py').count('_memecho_initialized_compressor('),2)
        with self.assertRaises(ValueError):patch_compressor_initialization('def f():pass','/tmp/repair.py')

    def test_state_patch_requires_before_reproduction_twenty_after_and_reference(self):
        compressor={'sentinel_preserved':True,'padding_preserved':True,'independent_reference_passed':True,'reuse_failures':[]}
        before={'passed':False,'repairable_state_reuse':True,'reuse_repetitions':20,
                'compressors':{'attention':copy.deepcopy(compressor),'indexer':copy.deepcopy(compressor)}}
        before['compressors']['indexer']['reuse_failures']=[{'case':'poisoned_free_blocks'}]
        after=copy.deepcopy(before);after['passed']=True;after['compressors']['indexer']['reuse_failures']=[]
        self.assertEqual(verify_state_repair(before,after)['replays_after'],20)
        for key,value in (('passed',True),('repairable_state_reuse',False),('reuse_repetitions',19)):
            bad=copy.deepcopy(before);bad[key]=value
            with self.assertRaises(ValueError):verify_state_repair(bad,after)
        after['compressors']['attention']['independent_reference_passed']=False
        with self.assertRaises(ValueError):verify_state_repair(before,after)

    def test_model_and_dsa_installers_use_pinned_classes(self):
        class Model:
            def forward(self,input_ids,positions):return input_ids
            def compute_logits(self,x):return x
        class MoE:
            def forward(self,x):return x
        class DSA:
            def _forward(self,x):return x
            def _update_indexer_cache(self,x):return x
            def _indexer_select_topk(self,x):return x
        install_model({'AscendDeepseekV4ForCausalLM':Model,'DeepseekV4MoE':MoE})
        install_dsa({'AscendDSACPImpl':DSA})
        self.assertTrue(hasattr(Model.forward,'__wrapped__'))
        self.assertEqual(Model().forward(7,1),7)
        self.assertEqual(DSA()._forward(3),3)

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('torch_npu'), 'NPU capsule verification runs in experiment container')
    def test_npu_capsule_resets_mutations_and_preserves_aliases_strides_offsets(self):
        import torch
        import torch_npu
        if not Path('/dev/davinci0').exists():
            self.skipTest('No NPU device is mounted in this server')
        torch.npu.set_device(0)
        backing=torch.arange(128,dtype=torch.float32).npu()
        a=backing.as_strided((2,3),(16,2),4)
        b=backing.as_strided((3,2),(2,16),4)
        capsule=snapshot_call((a,b),{'scale':2.})
        for _ in range(20):
            args,kwargs=restore_call(capsule,'npu:0')
            self.assertEqual(args[0].stride(),(16,2))
            self.assertEqual(args[0].storage_offset(),4)
            self.assertEqual(args[0].untyped_storage()._cdata,args[1].untyped_storage()._cdata)
            self.assertTrue(torch.equal(args[0].cpu(),a.cpu()))
            args[0].fill_(99)
            self.assertTrue(torch.all(args[1]==99).item())
            self.assertEqual(kwargs,{'scale':2.})
        with patch('deepseek_operator_diagnostics.MAX_CAPSULE_BYTES',32):
            with self.assertRaisesRegex(ValueError,'budget'):snapshot_call((a,),{})


if __name__ == '__main__': unittest.main()
