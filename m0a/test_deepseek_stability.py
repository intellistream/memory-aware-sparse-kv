"""Candidate qualification retains cold responses and never weakens equivalence."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deepseek_validation import Worker, TaskDeadline, stability_candidates, output_difference, write_json, watchdog
from import_workloads import sha_ids
from model_profiles import load_profile


COMMAND = ['vllm','serve','model','--tensor-parallel-size','8','--max-num-seqs','32',
           '--speculative-config','{"method":"mtp","num_speculative_tokens":1,"enforce_eager":true}',
           '--compilation-config','{"cudagraph_mode":"FULL_DECODE_ONLY","other":7}',
           '--additional-config','{"enable_dsa_cp":true,"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"multistream_overlap_shared_expert":true}']


def pair():
    item = {'prompt':'hello','prompt_token_ids':[1,2], 'prompt_tokens_expected':2,
            'prompt_token_ids_sha256_expected':sha_ids([1,2]),'boundary_position':1}
    return {'pair_id':'8192_tool_call','event':item,'control':copy.deepcopy(item),'context_target':8192}


def api(token=3):
    return {'id':'r','prompt_token_ids':[1,2], 'usage':{'prompt_tokens':2,'completion_tokens':1},
            'choices':[{'token_ids':[token],'finish_reason':'stop','message':{'content':str(token)}}]}


class StabilityTests(unittest.TestCase):
    def worker(self,root):
        worker = Worker(root,'deepseek_20260929T000000Z_12345678')
        worker.directory.mkdir(parents=True)
        worker.original = {'Config':{'Cmd':copy.deepcopy(COMMAND)}}
        return worker

    def test_candidate_changes_are_cumulative_and_original_is_preserved(self):
        original = copy.deepcopy(COMMAND)
        configs = stability_candidates(original)
        self.assertEqual(original,COMMAND)
        self.assertEqual(configs[0]['command'],COMMAND)
        self.assertEqual(len(configs),6)
        for index,config in enumerate(configs):
            self.assertEqual(len(config['changes']),index)
            cmd = config['command']
            self.assertEqual(cmd[cmd.index('--tensor-parallel-size')+1],'8')
            self.assertTrue(json.loads(cmd[cmd.index('--additional-config')+1])['enable_dsa_cp'])
            self.assertEqual('--speculative-config' in cmd,index<3)
            if index:
                self.assertIn('--enforce-eager',cmd)
                self.assertEqual(json.loads(cmd[cmd.index('--compilation-config')+1]),{'cudagraph_mode':'NONE','other':7})
        self.assertIn('--no-async-scheduling',configs[4]['command'])
        self.assertEqual(configs[5]['command'][configs[5]['command'].index('--max-num-seqs')+1],'1')

    def test_diagnostic_retains_all_twenty_and_compares_cold_response(self):
        with tempfile.TemporaryDirectory() as root:
            worker = self.worker(Path(root))
            p = pair()
            seq = [(p,v,cycle*2+(index>=2)) for cycle in range(5) for index,v in enumerate(('event','control','control','event'))]
            count = 0
            def respond(payload):
                nonlocal count
                count += 1
                return api(9 if count==4 else 3)
            with patch('deepseek_validation.request',side_effect=respond):
                records,failures = worker.request_sequence('stability',seq,load_profile('deepseek_v4'),collect=True)
            self.assertEqual(len(records),20)
            self.assertEqual(len(failures),1)
            self.assertEqual(failures[0]['first_differing_token_index'],0)
            self.assertEqual(failures[0]['repetition'],1)
            self.assertEqual([r['variant'] for r in records],['event','control','control','event']*5)
            raw = [json.loads(line) for line in (worker.directory/'stability-api.jsonl').read_text().splitlines()]
            self.assertEqual(raw[0]['request']['max_tokens'],32)
            self.assertEqual(raw[0]['prompt_token_ids'],[1,2])
            self.assertEqual(raw[3]['response']['choices'][0]['token_ids'],[9])

    def test_formal_equivalence_fails_and_saves_request_identity(self):
        with tempfile.TemporaryDirectory() as root:
            worker = self.worker(Path(root))
            p = pair()
            baseline = [{'pair_id':p['pair_id'],'variant':'event','repetition':0,
                         'signature':{'token_ids':[4],'content':'4','reasoning':None,'finish_reason':'stop'}}]
            with patch('deepseek_validation.request',return_value=api(3)):
                with self.assertRaisesRegex(ValueError,'trace_on_off'):
                    worker.request_sequence('trace_on',[(p,'event',0)],load_profile('deepseek_v4'),baseline=baseline)
            failure = json.loads((worker.directory/'trace_on-failures.json').read_text())[0]
            self.assertEqual(failure['prompt_token_ids_sha256'],sha_ids([1,2]))
            self.assertEqual(failure['expected_token'],4)
            self.assertEqual(failure['actual_token'],3)

    def test_trace_replay_records_output_drift_but_rejects_bad_input(self):
        with tempfile.TemporaryDirectory() as root:
            worker=self.worker(Path(root))
            p=pair()
            baseline=[{'pair_id':p['pair_id'],'variant':'event','repetition':0,
                       'signature':{'token_ids':[4],'content':'4','reasoning':None,'finish_reason':'stop'}}]
            with patch('deepseek_validation.request',side_effect=[api(3),api(5)]):
                rows,failures=worker.request_sequence('trace_on',[(p,'event',0),(p,'event',1)],
                    load_profile('deepseek_v4'),baseline=baseline,soft_output_differences=True)
            self.assertEqual(len(rows),2)
            self.assertFalse(failures)
            differences=json.loads((worker.directory/'trace_on-output-differences.json').read_text())
            self.assertEqual([d['kind'] for d in differences],['trace_on_off','repeat'])
            self.assertEqual(differences[0]['response']['choices'][0]['token_ids'],[3])
            self.assertEqual(differences[1]['first_differing_token_index'],0)
            self.assertFalse(json.loads((worker.directory/'trace_on-failures.json').read_text()))
            invalid=api(3)
            invalid['prompt_token_ids']=[7,8]
            with patch('deepseek_validation.request',return_value=invalid):
                with self.assertRaisesRegex(ValueError,'request_or_token_validation'):
                    worker.request_sequence('trace_off',[(p,'event',0)],load_profile('deepseek_v4'),
                                            soft_output_differences=True)
            hard=json.loads((worker.directory/'trace_off-failures.json').read_text())
            self.assertEqual(hard[0]['kind'],'request_or_token_validation')

    def test_trace_replay_cli_is_dry_and_legacy_execute_is_blocked(self):
        script=Path(__file__).resolve().parents[1]/'scripts/launch_deepseek_validation.py'
        dry=subprocess.run([sys.executable,str(script),'--validation-mode','trace-replay'],
                           text=True,capture_output=True,check=True)
        plan=json.loads(dry.stdout)
        self.assertEqual(plan['candidate_order'],['original_fresh'])
        self.assertEqual(plan['remote_directory'].split('/')[1:3],['root','memory-aware-sparse-kv'])
        self.assertFalse(plan['execution_available'])
        self.assertFalse(Path(plan['local_directory']).exists())
        for extra in ('--repair','--diagnostic-only'):
            result=subprocess.run([sys.executable,str(script),'--validation-mode','trace-replay',extra],
                                  text=True,capture_output=True)
            self.assertEqual(result.returncode,2)
            self.assertIn('cannot be combined',result.stderr)
        blocked=subprocess.run([sys.executable,str(script),'--validation-mode','trace-replay','--execute'],
                               text=True,capture_output=True)
        self.assertNotEqual(blocked.returncode,0)
        self.assertIn('Live execution is disabled',blocked.stderr)

    def test_trace_replay_uses_original_command_and_reports_separate_status(self):
        with tempfile.TemporaryDirectory() as root:
            worker=self.worker(Path(root))
            write_json(worker.directory/'launch.json',{'validation_mode':'trace-replay'})
            worker=Worker(Path(root),worker.run_id)
            worker.original={'Config':{'Cmd':copy.deepcopy(COMMAND)}}
            with patch.object(worker,'start_candidate') as started, \
                 patch.object(worker,'requests',return_value=[{}]*48) as requested, \
                 patch.object(worker,'command'),patch.object(worker,'cleanup'),patch.object(worker,'wait_release'):
                rows=worker.trace_replay_baseline({'pairs':[]},load_profile('deepseek_v4'))
            self.assertEqual(len(rows),48)
            self.assertEqual(started.call_args.args[0]['command'],COMMAND)
            self.assertTrue(requested.call_args.kwargs['soft_output_differences'])
            self.assertEqual(json.loads((worker.directory/'selected-config.json').read_text())['diagnostic_requests'],0)
            write_json(worker.directory/'restoration.json',{'restored':True})
            write_json(worker.directory/'trace-validation.json',{'paired_effect_interpretation':'exploratory',
                       'repeat_selected_id_differences':2,'pair_prefix_selected_id_differences':1})
            write_json(worker.directory/'trace_on-output-differences.json',[{'kind':'repeat'}])
            worker.report('engineering_validated')
            report=json.loads((worker.directory/'report.json').read_text())
            self.assertEqual(report['validation_mode'],'trace-replay')
            self.assertEqual(report['strict_output_acceptance'],'not_qualified')
            self.assertEqual(report['output_difference_counts']['trace_on'],1)
            self.assertEqual(report['selected_set_consistency']['paired_effect_interpretation'],'exploratory')

    def test_trace_replay_collects_48_requests_in_each_phase(self):
        with tempfile.TemporaryDirectory() as root:
            worker=self.worker(Path(root))
            pairs=[]
            for index in range(12):
                item=pair()
                item['pair_id']=f'pair-{index}'
                pairs.append(item)
            calls=0
            def respond(_payload):
                nonlocal calls
                calls+=1
                result=api(7 if calls==53 else 3)
                result['id']=f'chatcmpl-{calls}'
                return result
            with patch('deepseek_validation.request',side_effect=respond):
                baseline=worker.requests('trace_off',{'pairs':pairs},load_profile('deepseek_v4'),
                                         soft_output_differences=True)
                traced=worker.requests('trace_on',{'pairs':pairs},load_profile('deepseek_v4'),baseline,
                                       soft_output_differences=True)
            self.assertEqual((len(baseline),len(traced)),(48,48))
            self.assertEqual(calls,96)
            self.assertEqual(len((worker.directory/'trace_on-api.jsonl').read_text().splitlines()),48)
            self.assertTrue(json.loads((worker.directory/'trace_on-output-differences.json').read_text()))
            self.assertFalse(json.loads((worker.directory/'trace_on-failures.json').read_text()))

    def test_logprobs_probe_is_separate_from_repeat_gate(self):
        with tempfile.TemporaryDirectory() as root:
            worker = self.worker(Path(root))
            responses = [api(3),api(4)]
            for response in responses:
                response['choices'][0]['logprobs']={'content':[{'top_logprobs':[{'logprob':-.1},{'logprob':-.15}]}]}
            with patch('deepseek_validation.request',side_effect=responses) as requested:
                records,failures = worker.request_sequence('logprobs',[(pair(),'event',i) for i in range(2)],
                                                           load_profile('deepseek_v4'),collect=True,logprobs=True)
            self.assertFalse(failures)
            self.assertAlmostEqual(records[0]['first_token_top2_gap'],.05)
            self.assertTrue(requested.call_args.args[0]['logprobs'])
            self.assertEqual(requested.call_args.args[0]['top_logprobs'],5)

    def test_failed_full_baseline_advances_and_preserves_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            worker = self.worker(Path(root))
            sequences = []
            def sequence(stage,seq,*args,**kw):
                sequences.append((stage,len(seq),worker.selected_config['id']))
                return ([{}]*len(seq),[])
            def baseline(*args):
                for name in ('trace_off-api.jsonl','trace_off-responses.jsonl','trace_off-failures.json'):
                    (worker.directory/name).write_text(worker.selected_config['id'])
                if worker.selected_config['id']=='original_fresh':
                    raise ValueError('full baseline unstable')
                return [{}]*48
            with patch.object(worker,'start_candidate'),patch.object(worker,'request_sequence',side_effect=sequence), \
                    patch.object(worker,'requests',side_effect=baseline),patch.object(worker,'command'), \
                    patch.object(worker,'cleanup'),patch.object(worker,'wait_release'),patch.object(worker,'checkpoint'):
                result = worker.choose_configuration({'pairs':[pair()]},load_profile('deepseek_v4'))
            self.assertEqual(len(result),48)
            self.assertEqual([r['status'] for r in worker.diagnostics],['rejected','accepted'])
            self.assertEqual(worker.selected_config['id'],'eager')
            self.assertEqual((worker.directory/'diagnostics/original_fresh/trace_off-api.jsonl').read_text(),'original_fresh')
            self.assertEqual((worker.directory/'trace_off-api.jsonl').read_text(),'eager')
            self.assertEqual([n for _,n,_ in sequences],[20,4,20,4])

    def test_global_deadline_is_not_treated_as_candidate_rejection(self):
        with tempfile.TemporaryDirectory() as root:
            worker = self.worker(Path(root))
            with patch.object(worker,'start_candidate',side_effect=TaskDeadline('deadline')), \
                    patch.object(worker,'command'),patch.object(worker,'cleanup'),patch.object(worker,'wait_release'):
                with self.assertRaises(TaskDeadline):
                    worker.choose_configuration({'pairs':[pair()]},load_profile('deepseek_v4'))
            self.assertFalse(worker.diagnostics)

    def test_all_candidates_fail_without_relaxing_gate(self):
        with tempfile.TemporaryDirectory() as root:
            worker = self.worker(Path(root))
            with patch.object(worker,'start_candidate'), \
                    patch.object(worker,'request_sequence',return_value=([{}]*20,[{'kind':'repeat'}])), \
                    patch.object(worker,'requests') as baseline,patch.object(worker,'command'), \
                    patch.object(worker,'cleanup') as cleaned,patch.object(worker,'wait_release'),patch.object(worker,'checkpoint'):
                with self.assertRaisesRegex(ValueError,'All six'):
                    worker.choose_configuration({'pairs':[pair()]},load_profile('deepseek_v4'))
            self.assertEqual(cleaned.call_count,6)
            self.assertEqual(len(worker.diagnostics),6)
            self.assertTrue(all(r['status']=='rejected' for r in worker.diagnostics))
            self.assertIsNone(worker.selected_config)
            baseline.assert_not_called()

    def test_trace_and_baseline_use_the_identical_selected_command(self):
        with tempfile.TemporaryDirectory() as root:
            worker = self.worker(Path(root))
            worker.original['Config']['Env']=[]
            worker.original['HostConfig']={'NetworkMode':'host','AutoRemove':False,'IpcMode':'private','ShmSize':100,
                                          'Privileged':True,'SecurityOpt':[],'Binds':[],'Devices':[]}
            worker.selected_config=stability_candidates(COMMAND)[3]
            path=worker.directory/'candidate'
            path.mkdir()
            with patch.object(worker,'wait_release'),patch.object(worker,'service_health'),patch.object(worker,'command') as command:
                worker.start_candidate(worker.selected_config,path)
                baseline_command=command.call_args.args[0]
                worker.start_trace([(1,33)])
                trace_command=command.call_args.args[0]
            selected=worker.selected_config['command']
            self.assertEqual(baseline_command[-len(selected):],selected)
            self.assertEqual(trace_command[-len(selected):],selected)
            self.assertEqual(worker.original['Config']['Cmd'],COMMAND)

    def test_diagnostic_only_does_not_request_full_baseline(self):
        with tempfile.TemporaryDirectory() as root:
            worker = self.worker(Path(root))
            worker.diagnostic_only = True
            with patch.object(worker,'start_candidate'), \
                    patch.object(worker,'request_sequence',side_effect=lambda stage,seq,*a,**kw:([{}]*len(seq),[])), \
                    patch.object(worker,'requests') as baseline,patch.object(worker,'command'), \
                    patch.object(worker,'cleanup'),patch.object(worker,'wait_release'),patch.object(worker,'checkpoint'):
                self.assertIsNone(worker.choose_configuration({'pairs':[pair()]},load_profile('deepseek_v4')))
            baseline.assert_not_called()
            config = json.loads((worker.directory/'selected-config.json').read_text())
            self.assertEqual(config['baseline_requests'],0)

    def test_prefix_length_difference_and_non_token_difference(self):
        self.assertEqual(output_difference({'token_ids':[1]},{'token_ids':[1,2]})['first_differing_token_index'],1)
        self.assertIsNone(output_difference({'token_ids':[1],'content':'a'},{'token_ids':[1],'content':'b'})['first_differing_token_index'])

    def test_watchdog_preserves_successful_restore_in_terminal_status(self):
        with tempfile.TemporaryDirectory() as root:
            worker = self.worker(Path(root))
            write_json(worker.directory/'original-service.json',{'Id':'original'})
            write_json(worker.directory/'status.json',dict(worker.state,worker_pid=999999999,
                                                         original_service_restored=False,stage='trace_on'))
            def restore():
                write_json(worker.directory/'restoration.json',{'restored':True})
                worker.update(original_service_restored=True)
            with patch('deepseek_validation.Worker',return_value=worker), \
                    patch.object(worker,'inspect',return_value={'State':{'Running':False}}), \
                    patch.object(worker,'command',return_value=''),patch.object(worker,'restore',side_effect=restore):
                self.assertEqual(watchdog(Path(root),worker.run_id),1)
            state=json.loads((worker.directory/'status.json').read_text())
            self.assertTrue(state['original_service_restored'])
            self.assertEqual(state['status'],'failed')


if __name__=='__main__':
    unittest.main()
