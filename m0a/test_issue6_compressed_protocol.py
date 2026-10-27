"""C4 protocol boundaries and offline accounting without simulator dependencies."""
from copy import deepcopy
import unittest

from .issue6_evaluate import evaluate
from .issue6_protocol import context, validate


def compressed_fixture():
    rows=[]
    for task,split in enumerate(('train','validation','test')):
        for q in range(4):
            p=32+q
            rows.append(dict(schema_version=2,unit='compressed_kv_token_position',unit_bytes=1024,
                compression_ratio=4,token_position=p,position=(p+1)//4-1,eligible_end=p//4,
                locally_generated_ids=list(range(p//4,(p+1)//4)),scores=None,
                attention_scope='deepseek_v4_C4_indexed_KV_only',source_kind='synthetic_fixture',
                task_id=f'task{task}',split=split,variant='controlled',run_id='test',request_id=f'req{task}',
                context_version='v1',rank=0,layer=2,phase='decode',query=q,
                prediction_step=2*q,completion_step=2*q+1,stage='one',event_type='controlled',
                query_text='account balance',fragments=[dict(start=0,end=4,text='account balance'),
                    dict(start=4,end=8,text='garden flowers')],selected_blocks=[0,1] if q<3 else [1,8]))
    return rows


class CompressedProtocolTests(unittest.TestCase):
    def test_new_unit_is_selected_but_not_yet_predictable(self):
        rows=compressed_fixture()
        self.assertEqual(len(validate(rows)),3)
        self.assertNotIn(8,context(rows[3]).eligible)
        self.assertIn(8,rows[3]['selected_blocks'])

    def test_reject_future_selection_and_forged_generation(self):
        for field,value in (('selected_blocks',[0,8]),('locally_generated_ids',[8]),('eligible_end',9)):
            rows=compressed_fixture()
            rows[0][field]=value
            with self.assertRaises(ValueError):
                validate(rows)

    def test_reject_mixed_units_and_missing_decode_position(self):
        rows=compressed_fixture()
        changed=deepcopy(rows)
        changed[0]['unit']='native_kv_token'
        with self.assertRaises(ValueError):
            validate(changed)
        del rows[1]
        with self.assertRaises(ValueError):
            validate(rows)

    def test_evaluation_uses_c4_bytes_and_no_local_host_recall(self):
        summary,_,records=evaluate(compressed_fixture(),capacities=(3,),budgets=(0,),strategies=('none',))
        self.assertEqual(summary['unit_bytes'],1024)
        self.assertEqual(summary['capacity_bytes_per_lane'],[3072])
        self.assertEqual(summary['echo_scope'],'unavailable_selected_only')
        self.assertEqual(sum(r['sync_recall_units'] for r in records),2)
        self.assertEqual(sum(r.get('local_generated_units',0) for r in records),1)

    def test_reject_wrong_bytes_and_uncertified_echo(self):
        for kw in (dict(unit_bytes=1152),dict(strategies=('echo_style',))):
            with self.assertRaises(ValueError):
                evaluate(compressed_fixture(),capacities=(3,),budgets=(0,),**kw)


if __name__=='__main__':
    unittest.main()
