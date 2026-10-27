"""Leakage/variant contracts and conservative audits of existing benchmark tasks.

Structural action signatures are proxies, never human-validated templates.
No Agent episode or paraphrase is generated or certified by this module.
"""
import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def require(condition,message):
    if not condition:
        raise ValueError(message)


def components(records, *, entity_holdout=False):
    ids=[r['id'] for r in records]
    require(len(ids)==len(set(ids)),'Duplicate record identity')
    parent={key:key for key in ids}
    def find(key):
        while parent[key]!=key:
            parent[key]=parent[parent[key]]
            key=parent[key]
        return key
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:
            parent[max(a,b)]=min(a,b)
    owners={}
    fields=['root_ids','template_ids']+(['entity_ids'] if entity_holdout else [])
    for record in records:
        require(record['root_ids'],'Missing source lineage')
        for field in fields:
            for value in record.get(field,[]):
                key=(field,value)
                if key in owners:
                    union(record['id'],owners[key])
                else:
                    owners[key]=record['id']
    result=defaultdict(list)
    for record in records:
        result[find(record['id'])].append(record['id'])
    return sorted([sorted(group) for group in result.values()])


def split_groups(records, *, entity_holdout=False):
    groups=components(records,entity_holdout=entity_holdout)
    # Fixed deterministic allocation. Groups never split to make counts look balanced.
    counts=dict(train=0,validation=0,test=0)
    proportions=dict(train=.6,validation=.2,test=.2)
    assignment={}
    for index,group in enumerate(sorted(groups,key=lambda g:(-len(g),g))):
        split=('train','validation','test')[index] if index<3 else min(counts,key=lambda s:counts[s]/proportions[s])
        for key in group:
            assignment[key]=split
        counts[split]+=len(group)
    return assignment,dict(component_count=len(groups),split_counts=counts,
        three_nonempty_partitions=all(counts.values()),balancing_never_splits_component=True)


def audit_split(records, assignment, *, entity_holdout=False):
    require(set(assignment)=={r['id'] for r in records},'Missing/extra split assignment')
    require(set(assignment.values())<={'train','validation','test'},'Unknown split')
    for group in components(records,entity_holdout=entity_holdout):
        require(len({assignment[key] for key in group})==1,'Lineage/template/entity split leakage')
    return dict(leakage_check_passed=True,template_semantics_validated=False,
        checked_relations=['root_ids','template_ids']+(['entity_ids'] if entity_holdout else []))


def check_variant(original, variant, kind):
    require(kind in {'paraphrase','different_need','entity_remap','history_padding'},'Unknown variant kind')
    if kind in {'paraphrase','history_padding'}:
        require(variant['facts']==original['facts'],'Facts changed')
        require(variant['evaluation']==original['evaluation'],'Evaluation changed')
        require(variant['tools']==original['tools'],'Tool contract changed')
    if kind=='different_need':
        require(variant['facts']!=original['facts'] or variant['evaluation']!=original['evaluation'],
                'Different need has no changed objective')
        require(variant['evaluation']!=original['evaluation'],'Changed need retained old success criteria')
    if kind=='entity_remap':
        mapping=variant.get('entity_mapping',{})
        require(mapping and len(set(mapping.values()))==len(mapping),'Entity mapping must be injective')
        require(set(mapping)==set(original['entities']),'Incomplete entity mapping')
        require(set(variant['entities'])==set(mapping.values()),'Mapped entity set inconsistent')
    return dict(mechanical_contract_passed=True,semantic_equivalence_verified=False,
                environment_executed=False,source_kind='contract_only')


def validate_events(events):
    active={}
    retired=defaultdict(set)
    allowed={'tool_result','memory_write','memory_merge','memory_replace','task_switch'}
    for event in events:
        require(event['event_type'] in allowed,'Unknown event type')
        require(event['source_kind'] in {'synthetic_fixture','observed_agent'},'Unknown event provenance')
        require(event['available_step']<=event['prediction_step'],'Future event visibility')
        owner=event['request_id']
        previous=active.get(owner)
        if previous is not None:
            require(event['prediction_step']>previous['prediction_step'],'Event ordering')
            if event['event_type'].startswith('memory_'):
                require(event['context_version']!=previous['context_version'],'Memory update did not change version')
            if previous['context_version']!=event['context_version']:
                retired[owner].add(previous['context_version'])
        require(event['context_version'] not in retired[owner],'Retired event version reused')
        if event['source_kind']=='observed_agent':
            require(bool(event.get('source_artifact_sha256')) and bool(event.get('source_event_id')),
                    'Observed event lacks source reference')
        active[owner]=event
    return dict(events=len(events),counts=dict(Counter(e['event_type'] for e in events)),
        all_injected=all(e['source_kind']=='synthetic_fixture' for e in events),
        provenance_reference_checked_only=True,real_tool_execution_verified=False)


def stripped(value):
    if isinstance(value,dict):
        return {k:stripped(v) for k,v in value.items() if k not in {'action_id'}}
    if isinstance(value,list):
        return [stripped(v) for v in value]
    return value


def inventory(tau_source):
    records,tasks,receipts=[],{},{}
    for domain in ('banking_knowledge','retail'):
        path=Path(tau_source)/'data/tau2/domains'/domain/'tasks.json'
        raw=path.read_bytes()
        receipts[domain]=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())
        parsed=json.loads(raw)
        for task in parsed:
            key=domain+':'+str(task['id'])
            actions=(task.get('evaluation_criteria') or {}).get('actions') or []
            signature=[(a.get('requestor','agent'),a['name'],sorted((a.get('arguments') or {}).keys())) for a in actions]
            # Empty-action tasks are not automatically certified as one semantic template.
            proxy=domain+':action-schema:'+digest(signature) if actions else domain+':unclassified:'+str(task['id'])
            entities=set()
            for action in actions:
                args=action.get('arguments') or {}
                for field in ('order_id','user_id','customer_name'):
                    if field in args:
                        entities.add(domain+':'+field+':'+str(args[field]))
                if 'first_name' in args and 'last_name' in args:
                    entities.add(domain+':name:'+str(args['first_name'])+' '+str(args['last_name']))
            records.append(dict(id=key,domain=domain,root_ids=[key],template_ids=[proxy],
                template_basis='structural_action_schema_proxy_requires_review',entity_ids=sorted(entities),
                tool_names=sorted({a['name'] for a in actions}),task_sha256=digest(task),
                evaluation_sha256=digest(stripped(task.get('evaluation_criteria'))),
                source_kind='benchmark_authored_task_not_model_trace'))
            tasks[key]=task
    assignments,stats=split_groups(records,entity_holdout=True)
    audit=audit_split(records,assignments,entity_holdout=True)
    per_domain={}
    for domain in receipts:
        subset=[r for r in records if r['domain']==domain]
        per_domain[domain]=dict(tasks=len(subset),partitions=dict(Counter(assignments[r['id']] for r in subset)),
            template_proxies=len({t for r in subset for t in r['template_ids']}))
    # Existing author-written pairs only: no alteration of facts or success criteria.
    pairs=[]
    for i,a in enumerate(records):
        text_a=json.dumps(tasks[a['id']]['user_scenario'],sort_keys=True)
        for b in records[i+1:]:
            if a['domain']!=b['domain'] or a['evaluation_sha256']==b['evaluation_sha256']:
                continue
            if not set(a['template_ids'])&set(b['template_ids']):
                continue
            text_b=json.dumps(tasks[b['id']]['user_scenario'],sort_keys=True)
            similarity=SequenceMatcher(None,text_a,text_b,autojunk=False).ratio()
            if similarity>=.85:
                pairs.append(dict(first=a['id'],second=b['id'],lexical_similarity=similarity,
                    distinct_evaluation=True,semantic_relationship_reviewed=False,
                    split_first=assignments[a['id']],split_second=assignments[b['id']]))
    pairs=sorted(pairs,key=lambda p:(-p['lexical_similarity'],p['first'],p['second']))[:30]
    return dict(schema_version=1,source_receipts=receipts,records=records,assignment=assignments,
        split_statistics=stats,split_audit=audit,per_domain=per_domain,existing_candidate_pairs=pairs,
        template_review_completed=False,Agent_executed=False,native_sparse_trace_available=False,
        tasks_11_to_17_research_acceptance=False,
        limitations=['Action schemas and entity groups are conservative proxies, not semantic template labels.',
                     'Authored benchmark tasks are inventoried; task solvability was not re-executed.',
                     'Candidate lexical pairs require review and their own native model traces.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tau-source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    report=inventory(args.tau_source)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps({k:report[k] for k in ('split_statistics','split_audit','per_domain')},indent=2))


if __name__=='__main__':
    main()
