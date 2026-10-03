"""Bounded, per-rank operator fingerprints and exact-state replay capsules.

Observation synchronizes device reads and can change a race. A module boundary,
input drift, incomplete state, or an unrepeatable replay never establishes cause.
"""
import argparse
import functools
import hashlib
import inspect
import json
import os
import threading
from pathlib import Path

MAX_FINGERPRINT_BYTES = 64 * 1024**2
MAX_CAPSULE_BYTES = 512 * 1024**2
_LOCAL = threading.local()
_RECORDER = None


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def classify_difference(reference, current):
    if reference['prefix'] != current['prefix']:
        return 'different_token_prefix'
    if reference['input'] != current['input']:
        return 'different_inputs_track_upstream'
    if reference['output'] == current['output']:
        return 'identical'
    if not reference['complete'] or not current['complete']:
        return 'incomplete_state_no_attribution'
    return 'same_inputs_different_outputs'


def fingerprint(value):
    import torch
    complete = True
    def walk(item):
        nonlocal complete
        if isinstance(item,torch.Tensor):
            metadata = {'shape':list(item.shape),'stride':list(item.stride()),'dtype':str(item.dtype),
                        'offset':item.storage_offset(),'device_type':item.device.type}
            if item.numel()*item.element_size() > MAX_FINGERPRINT_BYTES:
                complete = False
                metadata['omitted_bytes'] = item.numel()*item.element_size()
            else:
                data = item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                metadata['sha256'] = hashlib.sha256(data).hexdigest()
            return metadata
        if isinstance(item,(list,tuple)):
            return [walk(v) for v in item]
        if isinstance(item,dict):
            return {str(k):walk(v) for k,v in sorted(item.items(),key=lambda p:str(p[0]))}
        if item is None or isinstance(item,(bool,int,float,str)):
            return item
        complete = False
        return {'unsupported_type':type(item).__qualname__}
    tree = walk(value)
    return {'sha256':digest(tree),'tensors':tree},complete


def snapshot_call(args, kwargs):
    """Copy whole backing stores once, keeping aliases, strides, offsets and CPU args."""
    import torch
    stores,ids,used = [],{},0
    def encode(value):
        nonlocal used
        if isinstance(value,torch.Tensor):
            if value.layout != torch.strided:
                raise ValueError('Unsupported tensor layout for exact replay')
            fmt = None
            if value.device.type == 'npu':
                import torch_npu
                fmt = int(torch_npu.get_npu_format(value))
                if fmt not in (-1,2):
                    raise ValueError('Internal NPU format cannot be replayed exactly: '+str(fmt))
            storage = value.untyped_storage()
            key = (value.device.type,storage._cdata)
            if key not in ids:
                used += storage.nbytes()
                if used > MAX_CAPSULE_BYTES:
                    raise ValueError('Exact pre-call state exceeds capsule budget')
                ids[key] = len(stores)
                raw = torch.empty(0,dtype=torch.uint8,device=value.device).set_(storage,0,(storage.nbytes(),),(1,))
                stores.append({'data':raw.cpu().clone(),'device_type':value.device.type})
            return {'tensor':True,'storage':ids[key],'dtype':str(value.dtype).split('.')[-1],
                    'shape':list(value.shape),'stride':list(value.stride()),'offset':value.storage_offset(),'format':fmt}
        if isinstance(value,tuple): return {'tuple':[encode(v) for v in value]}
        if isinstance(value,list): return {'list':[encode(v) for v in value]}
        if isinstance(value,dict): return {'dict':{k:encode(v) for k,v in value.items()}}
        if value is None or isinstance(value,(bool,int,float,str)): return value
        raise ValueError('Unsupported operator argument '+type(value).__qualname__)
    encoded = encode((args,kwargs))
    return {'arguments':encoded,'storages':stores,'storage_bytes':used,
            'pre_call_fingerprint':fingerprint((args,kwargs))[0],
            'cpu_rng':torch.get_rng_state().clone(),'npu_rng':torch.npu.get_rng_state().cpu().clone()}


def restore_call(capsule, device):
    import torch
    stores = [s['data'].clone().to(device if s['device_type']=='npu' else 'cpu') for s in capsule['storages']]
    def decode(value):
        if isinstance(value,dict) and value.get('tensor'):
            raw = stores[value['storage']]
            return torch.empty(0,dtype=getattr(torch,value['dtype']),device=raw.device).set_(
                raw.untyped_storage(),value['offset'],value['shape'],value['stride'])
        if isinstance(value,dict):
            if 'tuple' in value: return tuple(decode(v) for v in value['tuple'])
            if 'list' in value: return [decode(v) for v in value['list']]
            if 'dict' in value: return {k:decode(v) for k,v in value['dict'].items()}
        return value
    torch.set_rng_state(capsule['cpu_rng'])
    torch.npu.set_rng_state(capsule['npu_rng'],device=device)
    restored = decode(capsule['arguments'])
    if fingerprint(restored)[0] != capsule['pre_call_fingerprint']:
        raise ValueError('Restored operator arguments differ from the pre-call state')
    return restored


def meaningful_result(name, args, kwargs, result):
    """Compressor trailing allocation is not a completed native KV unit."""
    if name == '_C_ascend.compressor.default':
        starts = kwargs['start_pos'].detach().cpu().tolist()
        lengths = kwargs['cu_seqlens'].detach().cpu().tolist()
        if len(starts) != 1 or len(lengths) != 2:
            raise ValueError('Diagnostic compressor output masking requires concurrency one')
        start,n = starts[0],lengths[1]-lengths[0]
        ratio = kwargs['cmp_ratio']
        valid = (start+n)//ratio-start//ratio
        return result[:valid]
    return result


class Recorder:
    def __init__(self, directory, rank):
        self.directory,self.rank = Path(directory),rank
        self.directory.mkdir(parents=True,exist_ok=True)
        self.references,self.histories = {},{}
        self.first = None
        self.path = self.directory/f'rank{rank}-fingerprints.jsonl'
        self.stream = self.path.open('a',buffering=1)

    def scope(self, input_ids, positions):
        marker = json.loads((self.directory/'request.json').read_text())
        ids,pos = input_ids.detach().cpu().reshape(-1).tolist(),positions.detach().cpu().reshape(-1).tolist()
        if len(ids) != len(pos):
            raise ValueError('Diagnostic token/position alignment changed')
        history = self.histories.setdefault(marker['ordinal'],{})
        history.update(zip(pos,ids))
        prefix = digest({'prompt':marker['prompt_sha256'],'observed_prefix':sorted(history.items())})
        with (self.directory/f'rank{self.rank}-steps.jsonl').open('a') as out:
            out.write(json.dumps({'prefix':prefix,'request':marker,'positions':pos,'input_token_ids':ids})+'\n')
        position_key = digest(pos)
        return {'prefix':prefix,'request':marker,'position_key':position_key,
                'positions':{'sha256':position_key,'count':len(pos),'start':min(pos) if pos else None,'end':max(pos) if pos else None},
                'ordinal':0,'schedule':{}}

    def observe(self, name, before, output, complete, *, capsule=None, capsule_error=None, boundary=False):
        context = _LOCAL.context
        key = (context['prefix'],context['position_key'],getattr(_LOCAL,'layer',''),name,context['ordinal'])
        context['ordinal'] += 1
        after,after_complete = fingerprint(output)
        row = {'rank':self.rank,'operator':name,'layer':getattr(_LOCAL,'layer',''),
               'prefix':context['prefix'],'positions':context['positions'],'request':context['request'],
               'ordinal':context['ordinal']-1,'input':before,'output':after,
               'complete':complete and after_complete,'schedule':context['schedule'],'module_boundary':boundary}
        reference = self.references.get(key)
        # Compare with the first request for this exact, still-equal token prefix.
        if reference is None:
            self.references[key] = row
        elif reference['request']['ordinal'] != context['request']['ordinal']:
            row['comparison'] = classify_difference(reference,row)
            if row['comparison'] != 'identical' and self.first is None and not boundary:
                self.first = dict(row,reference=reference,capsule_error=capsule_error)
                if row['comparison'] == 'same_inputs_different_outputs' and capsule:
                    import torch
                    capsule.update(operator=name,rank=self.rank,observation=self.first)
                    torch.save(capsule,self.directory/f'rank{self.rank}-first.pt')
                    self.first['capsule'] = f'rank{self.rank}-first.pt'
                (self.directory/f'rank{self.rank}-first.json').write_text(json.dumps(self.first,indent=2)+'\n')
        self.stream.write(json.dumps(row,separators=(',',':'))+'\n')


def get_recorder():
    global _RECORDER
    if _RECORDER is None:
        from vllm.distributed import get_tensor_model_parallel_rank
        _RECORDER = Recorder(os.environ['MEMECHO_OPERATOR_DIR'],get_tensor_model_parallel_rank())
    return _RECORDER


def dispatch_mode():
    from torch.utils._python_dispatch import TorchDispatchMode
    class OperatorMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            name = str(func)
            selected = any(word in name for word in ('_C_ascend.','npu.','aten.mm.','aten.bmm.','aten.addmm.'))
            if not selected:
                return func(*args,**kwargs)
            recorder = get_recorder()
            before,complete = fingerprint((args,kwargs))
            capsule,capsule_error = None,None
            if recorder.first is None:
                # Retain the pre-call mutable state in memory, export only first divergence.
                try: capsule = snapshot_call(args,kwargs)
                except ValueError as error: capsule_error = str(error)
            result = func(*args,**kwargs)
            # Include side effects; an output pointer alone can hide cache writes.
            recorder.observe(name,before,(meaningful_result(name,args,kwargs,result),args,kwargs),complete,
                             capsule=capsule,capsule_error=capsule_error)
            return result
    return OperatorMode()


def request_scope(function):
    @functools.wraps(function)
    def wrapper(self,*args,**kwargs):
        directory = os.environ.get('MEMECHO_OPERATOR_DIR')
        if not directory or not (Path(directory)/'request.json').exists():
            return function(self,*args,**kwargs)
        values = inspect.signature(function).bind(self,*args,**kwargs).arguments
        ids,positions = values.get('input_ids'),values.get('positions')
        if ids is None or positions is None:
            return function(self,*args,**kwargs)
        recorder = get_recorder()
        _LOCAL.context = recorder.scope(ids,positions)
        try:
            with dispatch_mode(): return function(self,*args,**kwargs)
        finally:
            _LOCAL.last_context = _LOCAL.context
            _LOCAL.context = None
    return wrapper


def layer_scope(label, function):
    @functools.wraps(function)
    def wrapper(self,*args,**kwargs):
        if not getattr(_LOCAL,'context',None):
            return function(self,*args,**kwargs)
        previous = getattr(_LOCAL,'layer','')
        values = inspect.signature(function).bind(self,*args,**kwargs).arguments
        layer = values.get('layer_name',getattr(self,'layer_idx',''))
        _LOCAL.layer = f'{label}:{layer}'
        # Record concrete scheduling/slot inputs used at the DSA boundary.
        metadata = values.get('attn_metadata')
        if isinstance(metadata,(tuple,list)):
            schedule = []
            for m in metadata:
                req = getattr(m,'req_metadata',None)
                if req is not None:
                    fields = {k:getattr(req,k) for k in ('start_pos','block_table','slot_mapping','query_start_loc') if hasattr(req,k)}
                    schedule.append(fingerprint(fields)[0])
            _LOCAL.context['schedule'] = schedule
        before,complete = fingerprint({k:v for k,v in values.items() if k not in ('self','attn_metadata')})
        try:
            result = function(self,*args,**kwargs)
            get_recorder().observe(label,before,result,complete,boundary=True)
            return result
        finally: _LOCAL.layer = previous
    return wrapper


def install_model(namespace):
    cls = namespace['AscendDeepseekV4ForCausalLM']
    cls.forward = request_scope(cls.forward)
    compute_logits = cls.compute_logits
    @functools.wraps(compute_logits)
    def logits(self,*args,**kwargs):
        context = getattr(_LOCAL,'last_context',None)
        if not context:
            return compute_logits(self,*args,**kwargs)
        _LOCAL.context = context
        try:
            with dispatch_mode(): return compute_logits(self,*args,**kwargs)
        finally: _LOCAL.context = None
    cls.compute_logits = logits
    namespace['DeepseekV4MoE'].forward = layer_scope('moe',namespace['DeepseekV4MoE'].forward)


def install_dsa(namespace):
    cls = namespace['AscendDSACPImpl']
    for name,label in [('_forward','attention'),('_update_indexer_cache','indexer_compressor'),('_indexer_select_topk','indexer')]:
        setattr(cls,name,layer_scope(label,getattr(cls,name)))


def independent_reference(operator, args, kwargs):
    """Independent FP64 CPU result for supported arithmetic, never a second custom-op call."""
    import torch
    if operator in ('aten.mm.default','aten.bmm.default') and not kwargs:
        return torch.matmul(args[0].detach().cpu().double(),args[1].detach().cpu().double())
    raise ValueError('No independent reference for captured operator '+operator)


def replay_capsule(path, output, repetitions=20):
    import torch
    import torch_npu  # noqa: F401
    from vllm_ascend.utils import enable_custom_op
    assert enable_custom_op()
    assert repetitions == 20, 'Operator acceptance requires twenty replays'
    capsule = torch.load(path,map_location='cpu',weights_only=False)
    device = f'npu:{capsule["rank"]}'
    torch.npu.set_device(device)
    name = capsule['operator']
    # Communication needs all rank arguments and communicator state, absent here.
    if any(word in name.lower() for word in ('all_reduce','all_to_all','dispatch','combine','hccl')):
        raise ValueError('Distributed operator needs coordinated eight-rank replay')
    parts = name.split('.')
    operator = getattr(getattr(getattr(torch.ops,parts[0]),parts[1]),parts[2])
    signatures,errors,outputs = [],[],[]
    reference,reference_error = None,None
    args,kwargs = restore_call(capsule,device)
    try: reference = independent_reference(name,args,kwargs)
    except ValueError as error: reference_error = str(error)
    for _ in range(repetitions):
        args,kwargs = restore_call(capsule,device)
        torch.npu.synchronize()
        result = operator(*args,**kwargs)
        torch.npu.synchronize()
        signatures.append(fingerprint((meaningful_result(name,args,kwargs,result),args,kwargs))[0]['sha256'])
        if reference is not None:
            value = result.detach().cpu().double()
            errors.append(float((value-reference).abs().max()))
            outputs.append(torch.allclose(value,reference,atol=.035,rtol=.035))
    data = {'operator':name,'rank':capsule['rank'],'repetitions':repetitions,'state_reset_each_time':True,
            'output_and_mutated_state_sha256':signatures,'distinct_results':len(set(signatures)),
            'reproduced':len(set(signatures))>1,'independent_reference_available':reference is not None,
            'reference_error':reference_error,'max_reference_errors':errors,
            'reference_passed':bool(outputs) and all(outputs),'observer_synchronizes_device':True,
            'root_cause_confirmed':False}
    # A reproduction isolates numerical nondeterminism; it does not prove an async
    # production race or authorize a substitute with no independent reference.
    Path(output).write_text(json.dumps(data,indent=2)+'\n')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    try: replay_capsule(args.replay,args.output)
    except BaseException as error:
        args.output.write_text(json.dumps({'reproduced':False,'root_cause_confirmed':False,'error':str(error)},indent=2)+'\n')
        raise


if __name__ == '__main__': main()
