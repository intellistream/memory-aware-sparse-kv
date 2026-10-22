"""Independent analytical scenarios for scheduling, causality and payload safety."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import simpy

from .issue6_async_simulator import AsyncConfig, CostProfile, Fabric, SlotCache, replay, memory_plan
from .test_issue6_simulator import lane


def steps(selections, candidates=None, ranks=1, strategy='none', signal='history'):
    rows = lane(selections)
    result = []
    for q,row in enumerate(rows):
        group=[]
        for rank in range(ranks):
            clone = deepcopy(row)
            clone['rank']=rank
            group.append(dict(row=clone, candidates=[] if candidates is None else candidates[q],
                              strategy=strategy, signal_kind=signal))
        result.append(group)
    return result


class AsyncTests(unittest.TestCase):
    def config(self, **kw):
        return replace(AsyncConfig(capacity=2,budget=2,chunk_units=1,bandwidth_GBs=1,
            launch_us=0,native_us=10,prediction_us=1,attention_us=3), **kw)

    def test_cold_then_warm_analytical(self):
        r=replay(steps([[0,1],[0,1]]), self.config())
        self.assertAlmostEqual(r['records'][0]['wait_us'],2.048)
        self.assertEqual(r['records'][1]['wait_us'],0)
        self.assertEqual(r['total_h2d_bytes'],2048)

    def test_pack_only_missing_ids_into_dma_chunks(self):
        r=replay(steps([[0,2],[0,1,2,3]]),self.config(capacity=4,chunk_units=2))
        transfers=[t for t in r['transfers'] if t['query']==1]
        self.assertEqual(len(transfers),1)
        self.assertEqual(transfers[0]['units'],[1,3])

    def test_perfect_early_prefetch_hides_dma(self):
        r=replay(steps([[0,1]],[[0,1]],strategy='history'), self.config())
        self.assertEqual(r['records'][0]['wait_us'],0)
        self.assertEqual(r['decode_schedule_us'],13)
        self.assertEqual(r['unused_prefetch_bytes'],0)

    def test_late_prediction_does_not_block_demand(self):
        r=replay(steps([[0,1]],[[0,1]],strategy='history'), self.config(prediction_us=100))
        self.assertAlmostEqual(r['decode_schedule_us'],15.048)
        self.assertEqual(r['drained_us'],100)
        self.assertTrue(all(t['reason']=='demand' for t in r['transfers']))

    def test_current_scores_cannot_be_early(self):
        r=replay(steps([[0,1]],[[0,1]],strategy='echo_style',signal='current_scores'), self.config())
        self.assertEqual(r['records'][0]['prediction_ready_us'],11)
        self.assertTrue(all(t['reason']=='demand' for t in r['transfers']))

    def test_no_duplicate_transfer_when_demand_joins_inflight(self):
        r=replay(steps([[0,1]],[[0,1]],strategy='history'),self.config(native_us=.5,prediction_us=0))
        self.assertEqual(r['total_h2d_bytes'],2048)
        self.assertEqual([t['reason'] for t in r['transfers']],['prefetch','demand'])
        self.assertTrue(all(x['peak_slots']<=2 for x in r['records']))

    def test_wrong_prefetch_has_real_pollution_cost(self):
        a=replay(steps([[0,1],[0,1]]),self.config())
        b=replay(steps([[0,1],[0,1]],[[],[2,3]],strategy='history'),self.config())
        self.assertGreater(b['decode_schedule_us'],a['decode_schedule_us'])
        self.assertEqual(b['unused_prefetch_bytes'],2048)

    def test_nonpreemptible_chunk_blocks_but_later_chunks_cancel(self):
        r=replay(steps([[0,1]],[[2,3]],strategy='history'),self.config(native_us=.5,prediction_us=0))
        self.assertEqual([t['units'] for t in r['transfers']],[[2],[0],[1]])
        self.assertAlmostEqual(r['records'][0]['wait_us'],3.072-.5)

    def test_rank_barrier_and_link_topology(self):
        cfg=self.config(ranks=2,collective_latency_us=0,collective_bytes=0)
        shared=replay(steps([[0,1]],ranks=2),cfg)
        private=replay(steps([[0,1]],ranks=2),replace(cfg,host_link='per_rank'))
        self.assertGreater(shared['decode_schedule_us'],private['decode_schedule_us'])
        barrier=shared['collectives'][0]['barrier_us']
        self.assertEqual(barrier,max(r['rank_done_us'] for r in shared['records']))
        self.assertGreater(max(r['barrier_wait_us'] for r in shared['records']),0)

    def test_straggler_and_collective_formula(self):
        cfg=self.config(ranks=2,host_link='per_rank',rank_slowdown=(1,2),collective_bytes=1000,
                        collective_GBs=1,collective_latency_us=2)
        r=replay(steps([[0,1]],ranks=2),cfg)
        self.assertEqual(r['collectives'][0]['duration_us'],5)
        self.assertGreater(r['records'][0]['barrier_wait_us'],0)

    def test_generated_unit_is_d2h_not_h2d(self):
        r=replay(steps([[0,1],[0,1],[0,1],[1,8]]),self.config())
        self.assertEqual(r['total_d2h_bytes'],1024)
        self.assertFalse(any(8 in t['units'] for t in r['transfers'] if t['direction']=='h2d'))

    def test_payload_copy_and_attention_readiness(self):
        host={i:[i,i+1] for i in range(9)}
        checked=[]
        def verify(row, payload):
            self.assertEqual(payload,[host[i] for i in row['selected_blocks']])
            self.assertIsNot(payload[0],host[row['selected_blocks'][0]])
            checked.append(row['query'])
        replay(steps([[0,1],[2,3],[0,1]],[[2,3],[0,1],[4,5]],strategy='history'),
               self.config(),payload_provider=lambda ns,k:host[k],attention_check=verify)
        self.assertEqual(checked,[0,1,2])

    def test_version_change_does_not_reuse_cache(self):
        schedule=steps([[0,1],[0,1]])
        schedule[1][0]['row']['context_version']='replacement'
        r=replay(schedule,self.config())
        self.assertEqual(r['total_h2d_bytes'],4096)
        self.assertTrue(any(t['state']=='namespace_invalidated' for t in r['cache_transitions']))

    def test_invalidated_inflight_cannot_publish_or_read(self):
        env=simpy.Environment()
        cache=SlotCache(env,Fabric(env,self.config(),CostProfile()),0,('old',))
        env.process(cache.fetch([0],'prefetch',0))
        def invalidate():
            yield env.timeout(.1)
            cache.invalidate()
        env.process(invalidate())
        env.run()
        self.assertEqual(cache.entries[0]['state'],'invalid')
        with self.assertRaisesRegex(ValueError,'Invalid namespace'):
            cache.read([0])

    def test_reject_future_and_forged_boundary(self):
        schedule=steps([[0,1]],[[8]],strategy='history')
        with self.assertRaisesRegex(ValueError,'candidate'):
            replay(schedule,self.config())
        schedule[0][0]['row']['eligible_end']=99
        with self.assertRaisesRegex(ValueError,'boundary'):
            replay(schedule,self.config())

    def test_profile_interpolation_and_extrapolation_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'profile.json'
            path.write_text(json.dumps(dict(schema_version=1,source_kind='assumed',model='m',device='d',parallel_ranks=1,
                provenance='analytical test',operations={'h2d':[{'size':0,'duration_us':1},{'size':100,'duration_us':3}]})))
            p=CostProfile(path,model='m',device='d')
            self.assertEqual(p.duration('h2d',50,999),(2,'assumed'))
            with self.assertRaisesRegex(ValueError,'out of range'):
                p.duration('h2d',101,999)
            with self.assertRaisesRegex(ValueError,'mismatch'):
                CostProfile(path,model='wrong',device='d')

    def test_memory_includes_full_indexer_and_no_device_shadow(self):
        cfg=dict(compress_ratios=[0,4,128],num_hidden_layers=3,head_dim=512,
                 index_head_dim=128,sliding_window=128)
        a=memory_plan(cfg,tokens=4096,ranks=8,capacity=512,weights_total_bytes=8000,
                      device_bytes=10**9,workspace_bytes=100)
        b=memory_plan(cfg,tokens=8192,ranks=8,capacity=512,weights_total_bytes=8000,
                      device_bytes=1,workspace_bytes=100)
        self.assertEqual(a['per_rank']['c4_slots'],512*1024)
        self.assertEqual(b['per_rank']['indexer_full_int8_and_scale'],2*a['per_rank']['indexer_full_int8_and_scale'])
        self.assertFalse(b['within_assumed_budget'])
        self.assertEqual(a['per_rank']['weights_balanced_assumption'],1000)


if __name__=='__main__':
    unittest.main()
