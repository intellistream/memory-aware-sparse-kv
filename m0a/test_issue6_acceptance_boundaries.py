"""Additional cross-request, profile and asynchronous payload acceptance checks."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import random
import tempfile
import unittest

from .issue6_async_simulator import AsyncConfig, CostProfile, replay
from .test_issue6_async_simulator import steps
from .test_issue6_simulator import lane


class AcceptanceBoundaryTests(unittest.TestCase):
    def test_rank_group_rejects_mixed_task_run_or_position(self):
        for key,value in (('task_id','another'),('run_id','another'),('token_position',36)):
            schedule=steps([[0,1]],ranks=2)
            schedule[0][1]['row'][key]=value
            with self.assertRaisesRegex(ValueError,'Rank group'):
                replay(schedule,AsyncConfig(ranks=2,capacity=2,budget=2))

    def test_retired_version_cannot_be_reactivated(self):
        schedule=steps([[0,1]]*3)
        schedule[1][0]['row']['context_version']='replacement'
        with self.assertRaisesRegex(ValueError,'Retired context'):
            replay(schedule,AsyncConfig(capacity=2,budget=2))

    def test_profile_is_revalidated_against_actual_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'profile.json'
            path.write_text(json.dumps(dict(schema_version=1,source_kind='assumed',model='m',device='d',
                parallel_ranks=1,provenance='fixture',operations={})))
            profile=CostProfile(path,model='m',device='d')
            with self.assertRaisesRegex(ValueError,'model/device'):
                replay(steps([[0,1]]),AsyncConfig(capacity=2,budget=2),profile)

    def test_transfer_identity_distinguishes_same_query_across_requests(self):
        schedule=steps([[0,1],[0,1]])
        row=schedule[1][0]['row']
        row.update(task_id='second',request_id='second',query=0,token_position=32,position=7,
                   eligible_end=8,prediction_step=0,completion_step=1)
        result=replay(schedule,AsyncConfig(capacity=2,budget=2,native_us=0))
        for event in result['transfers']:
            rec=next(r for r in result['records'] if r['step']==event['step'])
            self.assertEqual(event['namespace'],rec['namespace'])
        self.assertEqual({e['step'] for e in result['transfers']},{0,1})
        self.assertEqual({e['query'] for e in result['transfers']},{0})

    def test_generated_writeback_has_full_identity(self):
        result=replay(steps([[0,1]]*4),AsyncConfig(capacity=2,budget=2))
        for event in result['transfers']:
            self.assertIsNotNone(event['step'])
            self.assertIsNotNone(event['namespace'])

    def test_randomized_capacity_ready_payload_and_byte_conservation(self):
        for seed in range(40):
            rng=random.Random(seed)
            rows=lane([rng.sample(range(8),rng.randint(1,4)) for _ in range(12)])
            schedule=[]
            for row in rows:
                group=[]
                for rank in range(2):
                    copy=deepcopy(row);copy['rank']=rank
                    group.append(dict(row=copy,candidates=rng.sample(range(8),4),strategy='history'))
                schedule.append(group)
            cfg=AsyncConfig(ranks=2,capacity=4,budget=4,chunk_units=rng.choice((1,2,4)),
                native_us=rng.choice((0,.1,1,10)),prediction_us=rng.choice((0,.2,3)),
                bandwidth_GBs=1,launch_us=.2,attention_us=.3,
                host_link=rng.choice(('shared','per_rank')),collective_bytes=0,collective_latency_us=0)
            seen=[]
            def verify(row,payload):
                self.assertEqual(payload,[[key,row['task_id']] for key in row['selected_blocks']])
                seen.append((row['rank'],row['query']))
            result=replay(schedule,cfg,payload_provider=lambda ns,k:[k,ns[0]],attention_check=verify,
                          capture_transitions=False)
            self.assertEqual(len(seen),24)
            self.assertTrue(all(v<=4 for v in result['peak_slots_by_namespace'].values()))
            for event in result['transfers']:
                if event['direction']=='h2d':
                    self.assertEqual(event['bytes'],event['useful_bytes']+event['unused_bytes'])


if __name__=='__main__':
    unittest.main()
