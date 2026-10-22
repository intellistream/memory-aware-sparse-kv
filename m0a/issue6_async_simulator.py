"""SimPy KV state machine and synchronized rank replay, not an NPU emulator.

Costs are assumptions unless an explicitly sourced profile supplies an operation.
No model scores are synthesized here. Native selected IDs are immutable inputs.
"""
from __future__ import annotations
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import simpy

from .issue6_protocol import context


def check(condition, message):
    if not condition:
        raise ValueError(message)


class CostProfile:
    """Optional operation curves: linear interpolation; no silent extrapolation.

    Each row has size and duration_us. Sizes are bytes for h2d/d2h/collective,
    eligible C4 units for native, attention KV rows for attention, candidate
    count for prediction. Profiles must match the scenario's model/device label.
    """
    def __init__(self, path=None, *, model=None, device=None):
        self.data = None
        self.sha256 = None
        if path is None:
            return
        raw = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.data = json.loads(raw)
        check(self.data['schema_version'] == 1, 'Unsupported cost profile')
        check(self.data['source_kind'] in {'assumed', 'measured', 'external_simulator'}, 'Profile source kind')
        check(self.data['model'] == model and self.data['device'] == device, 'Profile model/device mismatch')
        check(type(self.data.get('parallel_ranks')) is int and self.data['parallel_ranks'] > 0,
              'Missing profile parallel rank count')
        check(bool(self.data.get('provenance')), 'Missing profile provenance')
        if self.data['source_kind'] != 'assumed':
            receipt = self.data.get('source_sha256', '')
            check(len(receipt) == 64 and all(c in '0123456789abcdef' for c in receipt), 'Missing source SHA256')
        for op, rows in self.data['operations'].items():
            check(op in {'h2d', 'd2h', 'native', 'attention', 'prediction', 'collective'}, 'Unknown operation')
            check(rows and all(math.isfinite(r['size']) and r['size'] >= 0 and
                  math.isfinite(r['duration_us']) and r['duration_us'] >= 0 for r in rows), 'Invalid cost curve')
            check(all(a['size'] < b['size'] for a,b in zip(rows, rows[1:])), 'Unsorted/duplicate curve sizes')

    def duration(self, operation, size, fallback):
        if self.data is None or operation not in self.data['operations']:
            return fallback, 'assumed'
        rows = self.data['operations'][operation]
        check(rows[0]['size'] <= size <= rows[-1]['size'], f'Profile out of range: {operation}/{size}')
        for row in rows:
            if size == row['size']:
                return row['duration_us'], self.data['source_kind']
        for a,b in zip(rows, rows[1:]):
            if a['size'] < size < b['size']:
                fraction = (size-a['size'])/(b['size']-a['size'])
                return a['duration_us']+fraction*(b['duration_us']-a['duration_us']), self.data['source_kind']
        raise ValueError('Unreachable cost interpolation')


@dataclass(frozen=True)
class AsyncConfig:
    ranks: int = 1
    capacity: int = 512
    budget: int = 512
    unit_bytes: int = 1024
    chunk_units: int = 128
    bandwidth_GBs: float = 16
    launch_us: float = 5
    native_us: float = 50
    attention_us: float = 20
    prediction_us: float = 5
    signal_delay_us: float = 0
    host_link: str = 'shared'
    collective_bytes: int = 8192
    collective_GBs: float = 100
    collective_latency_us: float = 1
    rank_slowdown: tuple = ()
    model: str = 'DeepSeek-V4-Flash-W8A8'
    device: str = 'unmeasured-logical-ranks'

    def validate(self):
        check(all(type(x) is int and x > 0 for x in
                  (self.ranks, self.capacity, self.unit_bytes, self.chunk_units)), 'Invalid integer capacity/rank/chunk')
        check(type(self.budget) is int and 0 <= self.budget <= self.capacity, 'Invalid budget')
        check(type(self.collective_bytes) is int and self.collective_bytes >= 0, 'Invalid collective size')
        check(self.host_link in {'shared', 'per_rank'}, 'Invalid host topology')
        for name in ('bandwidth_GBs', 'collective_GBs'):
            check(math.isfinite(getattr(self, name)) and getattr(self, name) > 0, 'Invalid bandwidth')
        for name in ('launch_us','native_us','attention_us','prediction_us','signal_delay_us','collective_latency_us'):
            check(math.isfinite(getattr(self, name)) and getattr(self, name) >= 0, 'Invalid timing')
        check(not self.rank_slowdown or len(self.rank_slowdown) == self.ranks, 'Rank multiplier size')
        check(all(math.isfinite(x) and x >= 1 for x in self.rank_slowdown), 'Invalid rank slowdown')


class Fabric:
    def __init__(self, env, config, profile):
        self.env, self.cfg, self.profile = env, config, profile
        self.links = [simpy.PriorityResource(env, capacity=1) for _ in
                      range(1 if config.host_link == 'shared' else config.ranks)]
        self.events = []

    def transfer(self, rank, keys, direction, reason, query, on_start=None):
        link_id = 0 if self.cfg.host_link == 'shared' else rank
        queued = self.env.now
        with self.links[link_id].request(priority=1 if reason == 'prefetch' else 0) as request:
            yield request
            if on_start is not None:
                on_start()
            start = self.env.now
            size = len(keys)*self.cfg.unit_bytes
            duration, source = self.profile.duration(direction, size,
                self.cfg.launch_us + size/(self.cfg.bandwidth_GBs*1000))
            factor = self.cfg.rank_slowdown[rank] if self.cfg.rank_slowdown else 1
            event = dict(rank=rank, link=link_id, query=query, direction=direction, reason=reason,
                         units=list(keys), bytes=size, queued_us=queued, start_us=start,
                         end_us=start+duration*factor, cost_source=source, used_units=[])
            self.events.append(event)
            yield self.env.timeout(duration*factor)
            return event


class SlotCache:
    """Reservations consume slots; queued/inflight slots are never evicted.

    A namespace owns its own slot map. Invalidating it prevents a late DMA
    callback from publishing data into a newer context. Optional payloads make
    the exact same scheduling path usable for numerical checks.
    """
    def __init__(self, env, fabric, rank, namespace, host_payload=None, capture_transitions=True):
        self.env, self.fabric, self.rank, self.namespace = env, fabric, rank, namespace
        self.entries = OrderedDict()
        self.protected = set()
        self.valid = True
        self.peak = 0
        self.changed = env.event()
        self.host_payload = host_payload
        self.transitions = []
        self.capture_transitions = capture_transitions
        self.transition_counts = {}

    def notify(self):
        self.changed.succeed()
        self.changed = self.env.event()

    def log(self, key, state):
        self.transition_counts[state] = self.transition_counts.get(state,0)+1
        if self.capture_transitions:
            self.transitions.append(dict(time_us=self.env.now, unit=key, state=state,
                                         rank=self.rank, namespace=list(self.namespace)))

    def reserve(self, keys):
        while self.valid:
            missing = [k for k in keys if k not in self.entries]
            evictable = [k for k,v in self.entries.items() if v['state']=='ready' and k not in self.protected and k not in keys]
            shortage = len(self.entries)+len(missing)-self.fabric.cfg.capacity
            if shortage <= len(evictable):
                for key in evictable[:max(0,shortage)]:
                    del self.entries[key]
                    self.log(key, 'evicted')
                for key in missing:
                    self.entries[key] = dict(state='reserved', ready=self.env.event(), payload=None, transfer=None)
                    self.log(key, 'reserved')
                self.peak = max(self.peak, len(self.entries))
                return missing
            yield self.changed
        return []

    def copy_payload(self, key):
        if self.host_payload is None:
            return None
        value = self.host_payload(key)
        return value.clone() if hasattr(value, 'clone') else deepcopy(value)

    def fetch(self, keys, reason, query):
        # Chunk reservation is atomic; another worker can request the same key.
        created = yield self.env.process(self.reserve(keys))
        if created:
            def started():
                for key in created:
                    self.entries[key]['state'] = 'inflight'
                    self.log(key, 'inflight')
            event = yield self.env.process(self.fabric.transfer(self.rank, created, 'h2d', reason, query, started))
            event['namespace'] = list(self.namespace)
            event['stale_on_completion'] = not self.valid
            for key in created:
                entry = self.entries[key]
                entry['state'] = 'ready' if self.valid else 'invalid'
                entry['payload'] = self.copy_payload(key) if self.valid else None
                entry['transfer'] = event
                entry['ready'].succeed()
                self.log(key, entry['state'])
            self.notify()
        if self.valid and reason == 'demand':
            yield simpy.AllOf(self.env, [self.entries[k]['ready'] for k in keys])

    def generate(self, keys):
        created = yield self.env.process(self.reserve(keys))
        for key in created:
            entry = self.entries[key]
            entry['state'] = 'ready'
            entry['payload'] = self.copy_payload(key)
            entry['ready'].succeed()
            self.log(key, 'generated_ready')
        self.notify()

    def read(self, keys):
        check(self.valid, 'Invalid namespace read')
        for key in keys:
            check(key in self.entries and self.entries[key]['state']=='ready', 'Attention read before KV ready')
        payloads = []
        for key in keys:
            entry = self.entries[key]
            event = entry['transfer']
            if event is not None and key not in event['used_units']:
                event['used_units'].append(key)
            self.entries.move_to_end(key)
            payloads.append(entry['payload'])
            self.log(key, 'attention_read')
        return payloads

    def invalidate(self):
        self.valid = False
        self.protected.clear()
        for key in self.entries:
            self.log(key, 'namespace_invalidated')
        self.notify()


def replay(steps, config=AsyncConfig(), profile=None, payload_provider=None, attention_check=None,
           capture_transitions=True):
    """steps[step][rank] = {row, candidates, strategy, signal_kind?}.

    Each step is a synchronous rank group for one request/query/layer. This is
    NOT a batching scheduler or a 43-layer forward pass. Callers must declare
    replicated rank fixtures honestly. Signal kind current_scores cannot start
    before native selection. Candidates are supplied by a separate causal API.
    """
    config.validate()
    check(steps, 'Empty schedule')
    last = {}
    for step in steps:
        check(len(step)==config.ranks, 'Missing rank')
        identity = {(w['row']['request_id'],w['row']['context_version'],w['row']['layer'],w['row']['query']) for w in step}
        check(len(identity)==1, 'Rank group query mismatch')
        for rank, work in enumerate(step):
            row, candidates = work['row'], work['candidates']
            check(row['rank']==rank and row['schema_version']==2 and row['phase']=='decode', 'Invalid rank/layout')
            position = row['token_position']
            check(row['eligible_end']==position//4 and row['position']==(position+1)//4-1 and
                  row['locally_generated_ids']==list(range(position//4,(position+1)//4)), 'C4 causal boundary mismatch')
            ns = context(row).namespace
            if ns in last:
                check((row['query'],position)==(last[ns][0]+1,last[ns][1]+1), 'Noncontiguous replay')
            last[ns] = (row['query'],position)
            selected = row['selected_blocks']
            check(len(selected)==len(set(selected)) and 0 < len(selected)<=config.capacity, 'Working set exceeds capacity')
            check(all(type(k) is int and 0 <= k < (row['token_position']+1)//4 for k in selected), 'Future native selection')
            check(config.unit_bytes==row['unit_bytes']==1024, 'C4 unit bytes mismatch')
            check(len(candidates)<=config.budget and len(candidates)==len(set(candidates)) and
                  all(type(k) is int and 0 <= k < row['eligible_end'] for k in candidates), 'Invalid future candidate/budget')
            check(work.get('signal_kind','history') in {'history','current_scores','oracle'}, 'Unknown signal timing')
    env = simpy.Environment()
    profile = profile or CostProfile()
    if profile.data is not None:
        check(profile.data['parallel_ranks']==config.ranks, 'Profile parallelism mismatch')
    fabric = Fabric(env, config, profile)
    caches, active, records, collectives, background = {}, {}, [], [], []

    def rank_step(work, rank, step_id):
        row, candidates = work['row'], work['candidates']
        namespace = context(row).namespace
        identity = (row['task_id'],row['run_id'],row['request_id'],rank,row['layer'],row['phase'])
        previous = active.get(identity)
        if previous is not None and previous != namespace:
            caches[previous].invalidate()
            # Retired allocations cannot be reused while DMA still targets them.
            yield simpy.AllOf(env, [entry['ready'] for entry in caches[previous].entries.values()
                                   if not entry['ready'].triggered])
        active[identity] = namespace
        if namespace not in caches:
            provider = None if payload_provider is None else lambda key: payload_provider(namespace, key)
            caches[namespace] = SlotCache(env, fabric, rank, namespace, provider, capture_transitions)
        cache = caches[namespace]
        begin = env.now
        native, native_source = profile.duration('native', row['eligible_end'], config.native_us)
        factor = config.rank_slowdown[rank] if config.rank_slowdown else 1
        native *= factor
        rec = dict(rank=rank, step=step_id, query=row['query'], namespace=list(namespace),
                   begin_us=begin, strategy=work['strategy'], native_ready_us=begin+native,
                   candidate_count=len(candidates), selected_count=len(row['selected_blocks']),
                   signal_kind=work.get('signal_kind','history'), native_cost_source=native_source)
        state = {'native_known': False}

        def prefetch():
            if work['strategy']=='none' or not config.budget:
                rec['prediction_ready_us'] = begin
                return
            signal = max(config.signal_delay_us, native if rec['signal_kind']=='current_scores' else 0)
            prediction, source = profile.duration('prediction', len(candidates), config.prediction_us)
            rec['prediction_cost_source'] = source
            yield env.timeout(signal+prediction)
            rec['prediction_ready_us'] = env.now
            for start in range(0, len(candidates), config.chunk_units):
                # Cancel unsent speculative chunks after the native result arrives.
                if state['native_known'] or not cache.valid:
                    break
                yield env.process(cache.fetch(candidates[start:start+config.chunk_units], 'prefetch', row['query']))

        background.append(env.process(prefetch()))
        yield env.timeout(native)
        state['native_known'] = True
        selected = row['selected_blocks']
        cache.protected = set(selected)
        local = row['locally_generated_ids']
        # Generated units needed now are locally ready; never H2D them. Other new
        # units use the compressor staging area, not speculative cache slots.
        yield env.process(cache.generate([k for k in local if k in cache.protected]))
        writeback = None
        if local:
            writeback = env.process(fabric.transfer(rank, local, 'd2h', 'write_through', row['query']))
        # Pack missing IDs rather than charging launches for chunks containing
        # mostly resident entries. Join existing reservations without reloading.
        missing = [key for key in selected if key not in cache.entries]
        for start in range(0, len(missing), config.chunk_units):
            yield env.process(cache.fetch(missing[start:start+config.chunk_units], 'demand', row['query']))
        yield simpy.AllOf(env, [cache.entries[key]['ready'] for key in selected])
        rec['attention_start_us'] = env.now
        payloads = cache.read(selected)
        if attention_check is not None:
            attention_check(row, payloads)
        attention, source = profile.duration('attention', 128+len(selected), config.attention_us)
        rec['attention_cost_source'] = source
        yield env.timeout(attention*factor)
        rec['attention_end_us'] = env.now
        cache.protected.clear()
        cache.notify()
        # Write-through must finish before the next query can fetch the unit
        # from Host. This is an explicit conservative policy, not free offload.
        if writeback is not None:
            yield writeback
        rec['rank_done_us'] = env.now
        rec['wait_us'] = rec['attention_start_us']-rec['native_ready_us']
        rec['peak_slots'] = cache.peak
        records.append(rec)

    def schedule():
        for step_id, step in enumerate(steps):
            begin = env.now
            yield simpy.AllOf(env, [env.process(rank_step(work,rank,step_id)) for rank,work in enumerate(step)])
            barrier = env.now
            ring = (2*(config.ranks-1)*config.collective_latency_us +
                    2*(config.ranks-1)/config.ranks*config.collective_bytes/(config.collective_GBs*1000))
            duration, source = profile.duration('collective', config.collective_bytes, ring)
            if config.ranks==1:
                duration = 0
            yield env.timeout(duration)
            collectives.append(dict(step=step_id, begin_us=begin, barrier_us=barrier, end_us=env.now,
                duration_us=duration, payload_bytes=config.collective_bytes, ranks=config.ranks,
                cost_source=source, algorithm='assumed_ring_allreduce_or_supplied_profile'))
            for rec in [r for r in records if r['step']==step_id]:
                rec['barrier_wait_us'] = barrier-rec['rank_done_us']
                rec['group_service_us'] = env.now-begin
        return env.now

    job = env.process(schedule())
    env.run()  # includes late predictions / DMA for complete traffic accounting
    check(job.triggered, 'Schedule deadlocked')
    for cache in caches.values():
        check(cache.peak <= config.capacity, 'Capacity violated')
    for link in range(len(fabric.links)):
        events = sorted([e for e in fabric.events if e['link']==link], key=lambda e:e['start_us'])
        check(all(a['end_us']<=b['start_us']+1e-9 for a,b in zip(events,events[1:])), 'Link overlap')
    for event in fabric.events:
        event['useful_bytes'] = len(event['used_units'])*config.unit_bytes if event['direction']=='h2d' else None
        event['unused_bytes'] = event['bytes']-event['useful_bytes'] if event['direction']=='h2d' else None
    return dict(config=asdict(config), cost_profile_sha256=profile.sha256,
        scope='synchronized_sparse_attention_rank_group_not_full_model',
        timeline_costs_are_hardware_measurements=False, decode_schedule_us=job.value,
        drained_us=env.now, records=records, transfers=fabric.events, collectives=collectives,
        cache_transitions=[t for c in caches.values() for t in c.transitions],
        transition_counts_by_namespace={str(k):c.transition_counts for k,c in caches.items()},
        total_h2d_bytes=sum(e['bytes'] for e in fabric.events if e['direction']=='h2d'),
        total_d2h_bytes=sum(e['bytes'] for e in fabric.events if e['direction']=='d2h'),
        unused_prefetch_bytes=sum(e['unused_bytes'] for e in fabric.events if e['reason']=='prefetch'),
        peak_slots_by_namespace={str(k):c.peak for k,c in caches.items()})


def memory_plan(model_config, *, tokens, ranks, capacity, concurrent_requests=1,
                weights_total_bytes, device_bytes, workspace_bytes, safety_bytes=0):
    """Logical memory ledger; balanced weights and replicated KV are assumptions.

    Indexer remains full resident. No full C4 device shadow is counted. The Host
    C4 copy is shared in this ledger; actual rank duplication needs measurement.
    Compressor states are conservative FP32 KV+gate windows, not an allocator
    prediction. Configured workspace includes allocator/collective/MoE buffers.
    """
    check(all(type(x) is int and x > 0 for x in (tokens,ranks,capacity,concurrent_requests,device_bytes)), 'Invalid memory dimensions')
    check(all(type(x) is int and x >= 0 for x in (weights_total_bytes,workspace_bytes,safety_bytes)), 'Invalid memory budget')
    ratios = model_config['compress_ratios'][:model_config['num_hidden_layers']]
    d, w = model_config['head_dim'], model_config['sliding_window']
    c4, c128 = ratios.count(4), ratios.count(128)
    check(set(ratios)<={0,4,128}, 'Unsupported layer layout')
    entries = dict(weights_balanced_assumption=math.ceil(weights_total_bytes/ranks),
        swa=concurrent_requests*len(ratios)*min(tokens,w)*d*2,
        c4_slots=concurrent_requests*c4*min(capacity,tokens//4)*d*2,
        c128_all=concurrent_requests*c128*(tokens//128)*d*2,
        indexer_full_int8_and_scale=concurrent_requests*c4*(tokens//4)*(model_config['index_head_dim']+4),
        compressor_state_upper_estimate=concurrent_requests*(c4*8*2*(d+model_config['index_head_dim'])*2*4+c128*128*d*2*4),
        workspace=workspace_bytes, safety=safety_bytes)
    total = sum(entries.values())
    return dict(per_rank=entries, per_rank_total=total, device_bytes=device_bytes,
        within_assumed_budget=total<=device_bytes, spare_bytes=device_bytes-total,
        host_c4_shared_bytes=concurrent_requests*c4*(tokens//4)*d*2,
        excluded_full_device_c4_shadow_bytes=concurrent_requests*c4*(tokens//4)*d*2,
        c4_layers=c4,c128_layers=c128,all_layers=len(ratios),
        measured_peak_memory=False, assumptions=['balanced_weight_sharding','replicated_KV_per_rank',
        'INT8_indexer_plus_FP32_scale','supplied_workspace_not_measured','shared_Host_C4_store'])
