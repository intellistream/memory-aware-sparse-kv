"""Tests reject evidence inflation and leakage; fixtures are not Agent results."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from .issue6_generalization_contracts import (components,split_groups,audit_split,check_variant,validate_events)
from .issue6_cpu_readiness import model_inventory


class GeneralizationContractTests(unittest.TestCase):
    def records(self):
        return [dict(id='a',root_ids=['original'],template_ids=['t1'],entity_ids=['e1']),
                dict(id='b',root_ids=['original'],template_ids=['t2'],entity_ids=['e2']),
                dict(id='c',root_ids=['other'],template_ids=['t2'],entity_ids=['e3']),
                dict(id='d',root_ids=['d'],template_ids=['t3'],entity_ids=['e3']),
                dict(id='e',root_ids=['e'],template_ids=['t4'],entity_ids=['e4'])]

    def test_transitive_lineage_and_template_groups(self):
        self.assertEqual(components(self.records()),[['a','b','c'],['d'],['e']])

    def test_entity_holdout_closes_transitive_leak(self):
        self.assertEqual(components(self.records(),entity_holdout=True),[['a','b','c','d'],['e']])

    def test_split_is_deterministic_and_never_breaks_group_for_balance(self):
        records=self.records()
        a,stats=split_groups(records,entity_holdout=True)
        b,_=split_groups(list(reversed(records)),entity_holdout=True)
        self.assertEqual(a,b)
        self.assertFalse(stats['three_nonempty_partitions'])
        self.assertTrue(audit_split(records,a,entity_holdout=True)['leakage_check_passed'])

    def test_reject_duplicate_or_missing_or_leaking_assignments(self):
        records=self.records()
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            components(records+[records[0]])
        with self.assertRaisesRegex(ValueError,'Missing/extra'):
            audit_split(records,{})
        with self.assertRaisesRegex(ValueError,'leakage'):
            audit_split(records,dict(a='train',b='test',c='test',d='test',e='validation'))

    def variant(self):
        return dict(facts={'amount':10},evaluation={'pay':10},tools=['pay'],entities=['person'])

    def test_paraphrase_contract_cannot_certify_semantics(self):
        source=self.variant()
        report=check_variant(source,deepcopy(source),'paraphrase')
        self.assertTrue(report['mechanical_contract_passed'])
        self.assertFalse(report['semantic_equivalence_verified'])
        changed=deepcopy(source);changed['facts']['amount']=100
        with self.assertRaisesRegex(ValueError,'Facts'):
            check_variant(source,changed,'paraphrase')

    def test_different_need_cannot_keep_old_success_criteria(self):
        source=self.variant();changed=deepcopy(source);changed['facts']['amount']=100
        with self.assertRaisesRegex(ValueError,'success criteria'):
            check_variant(source,changed,'different_need')

    def test_entity_mapping_must_cover_original_and_target(self):
        source=self.variant();changed=deepcopy(source)
        changed.update(entity_mapping={'person':'other'},entities=['other'])
        self.assertTrue(check_variant(source,changed,'entity_remap')['mechanical_contract_passed'])
        changed['entities']=['wrong']
        with self.assertRaisesRegex(ValueError,'inconsistent'):
            check_variant(source,changed,'entity_remap')

    def events(self):
        return [dict(event_type=kind,source_kind='synthetic_fixture',request_id='r',context_version='v'+str(i),
                     available_step=2*i,prediction_step=2*i+1) for i,kind in enumerate(
                     ('tool_result','memory_write','memory_merge','memory_replace','task_switch'))]

    def test_all_five_injected_event_types_remain_injected(self):
        report=validate_events(self.events())
        self.assertEqual(len(report['counts']),5)
        self.assertTrue(report['all_injected'])
        self.assertFalse(report['real_tool_execution_verified'])

    def test_events_reject_future_stale_and_unversioned_memory(self):
        for mutation in ('future','stale','unversioned'):
            events=self.events()
            if mutation=='future':
                events[0]['available_step']=99
            elif mutation=='stale':
                events[-1]['context_version']='v0'
            else:
                events[1]['context_version']='v0'
            with self.assertRaises(ValueError):
                validate_events(events)

    def test_observed_event_requires_artifact_reference(self):
        events=self.events();events[0]['source_kind']='observed_agent'
        with self.assertRaisesRegex(ValueError,'source reference'):
            validate_events(events)

    def test_quantized_index_is_checked_and_missing_shards_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'config.json').write_text(json.dumps(dict(model_type='deepseek_v4',index_topk=512)))
            (root/'quant_model_weights.safetensors.index.json').write_text(json.dumps(
                dict(metadata={'total_size':123},weight_map={'weight':'missing.safetensors'})))
            report=model_inventory(root)
            self.assertEqual(report['missing_shards'],['missing.safetensors'])
            self.assertFalse(report['checkpoint_structure_verified'])

    def test_reject_index_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'config.json').write_text('{}')
            (root/'quant_model_weights.safetensors.index.json').write_text(
                json.dumps(dict(weight_map={'w':'../outside.safetensors'})))
            with self.assertRaisesRegex(ValueError,'Unsafe'):
                model_inventory(root)


if __name__=='__main__':
    unittest.main()
