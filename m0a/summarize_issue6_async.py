"""Audit archived asynchronous runs and draw scope-labelled scientific figures."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    args=p.parse_args()
    run=args.run
    manifest=json.loads((run/'manifest.json').read_text())
    checked=dict(artifacts=0,rank_queries=0,transfers=0,payload_state_transitions=0)
    for name,expected in manifest['artifact_sha256'].items():
        assert hashlib.sha256((run/name).read_bytes()).hexdigest()==expected
        data=json.load(gzip.open(run/name,'rt'))
        checked['artifacts']+=1
        for row in data['records']:
            assert row['attention_start_us']>=row['native_ready_us']
            assert row['wait_us']==row['attention_start_us']-row['native_ready_us']
            assert row['peak_slots']<=data['config']['capacity']
            checked['rank_queries']+=1
        for event in data['transfers']:
            assert event['queued_us']<=event['start_us']<=event['end_us']
            owners=[r for r in data['records'] if r['rank']==event['rank'] and r['step']==event['step']]
            assert len(owners)==1, 'Ambiguous transfer owner'
            assert event['namespace']==owners[0]['namespace'], 'Transfer namespace mismatch'
            assert event['bytes']==len(event['units'])*1024
            assert set(event['used_units'])<=set(event['units'])
            if event['direction']=='h2d':
                assert event['bytes']==event['useful_bytes']+event['unused_bytes']
            if event['reason']=='prefetch':
                row=next(r for r in data['records'] if r['rank']==event['rank'] and r['step']==event['step'])
                assert event['start_us']>=row['prediction_ready_us']
            checked['transfers']+=1
        for link in {e['link'] for e in data['transfers']}:
            events=sorted([e for e in data['transfers'] if e['link']==link],key=lambda e:e['start_us'])
            assert all(a['end_us']<=b['start_us']+1e-9 for a,b in zip(events,events[1:]))
        for event in data['collectives']:
            records=[r for r in data['records'] if r['step']==event['step']]
            assert event['barrier_us']==max(r['rank_done_us'] for r in records)
            assert event['end_us']==event['barrier_us']+event['duration_us']
        states={}
        for event in data['cache_transitions']:
            key=tuple(event['namespace'])+(event['unit'],)
            state=event['state']
            if state=='attention_read':
                assert states[key] in {'ready','generated_ready'}
            elif state=='evicted':
                assert states[key] in {'ready','generated_ready'}
                del states[key]
            else:
                states[key]=state
            checked['payload_state_transitions']+=1
    for name,expected in manifest['source_files_sha256'].items():
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==expected,name
    checked['source_files']=len(manifest['source_files_sha256'])
    checked['passed']=True
    (run/'audit.json').write_text(json.dumps(checked,indent=2))
    cases=sorted({r['case'] for r in manifest['summaries']})
    configs=list(manifest['configurations'])
    strategies=['last_selection','global_frequency','stage_frequency','oracle_diagnostic']
    fig,axes=plt.subplots(1,len(cases),figsize=(14,5),constrained_layout=True)
    for ax,case in zip(axes,cases):
        matrix=np.array([[next(r['service_reduction_percent'] for r in manifest['summaries']
            if r['case']==case and r['scenario']==cfg and r['strategy']==strategy)
            for strategy in strategies] for cfg in configs])
        ax.imshow(matrix,cmap='RdYlGn',aspect='auto',vmin=-30,vmax=30)
        ax.set_xticks(range(len(strategies)),['Last set','Global freq','Stage freq','Oracle*'],rotation=25,ha='right')
        ax.set_yticks(range(len(configs)),configs)
        ax.set_title(case)
        for y in range(len(configs)):
            for x in range(len(strategies)):
                ax.text(x,y,f'{matrix[y,x]:+.1f}%',ha='center',va='center',fontsize=10)
    fig.suptitle('Simulated rank-group service-time reduction vs no prefetch\n'
                 'Assumed costs; controlled hidden states; *oracle uses current labels; NOT measured TPOT')
    fig.savefig(run/'service-reduction.png',dpi=160)
    plt.close(fig)
    name='c4-8192-shift-fp32--rank8-shared--global_frequency.json.gz'
    if (run/name).exists():
        data=json.load(gzip.open(run/name,'rt'))
        fig,ax=plt.subplots(figsize=(13,5),constrained_layout=True)
        query=0
        begin=data['collectives'][query]['begin_us']
        palette={'prefetch':'#2878b5','demand':'#d95f02','write_through':'#9467bd'}
        for r in data['records']:
            if r['query']==query:
                ax.broken_barh([(r['begin_us']-begin,r['native_ready_us']-r['begin_us'])],(r['rank']-.32,.25),facecolors='#999999')
                ax.broken_barh([(r['attention_start_us']-begin,r['attention_end_us']-r['attention_start_us'])],(r['rank']-.32,.25),facecolors='#2ca02c')
        for e in data['transfers']:
            if e['query']==query:
                ax.broken_barh([(e['start_us']-begin,e['end_us']-e['start_us'])],(e['rank'],.25),facecolors=palette[e['reason']])
        ax.axvline(data['collectives'][query]['barrier_us']-begin,color='black',ls='--',label='All ranks ready')
        ax.set(xlabel='Simulated time (us)',ylabel='Logical rank',yticks=range(8),
               title='Cold first query: eight logical ranks contend for one assumed Host link\n'
                     'Gray: native computation; orange: demand H2D; green: attention; no actual NPU execution')
        ax.legend()
        fig.savefig(run/'rank-timeline.png',dpi=160)
        plt.close(fig)
    print(json.dumps(checked,indent=2))
    for row in manifest['summaries']:
        if row['scenario'] in {'rank1-lead50','rank8-shared','rank8-per-rank'} and row['strategy'] in {'none','global_frequency','oracle_diagnostic'}:
            print(row['case'],row['scenario'],row['strategy'],round(row['mean_group_service_us'],3),round(row['service_reduction_percent'],2))
    print('MEMORY',json.dumps(manifest['memory_plans']))


if __name__=='__main__':
    main()
