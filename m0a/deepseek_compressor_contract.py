#!/usr/bin/env python3
"""Probe deployed C4 state views, support, padding and physical reuse on NPU."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def state_layout(block_size, state_dim, page_bytes, blocks=6):
    """Same contiguous inner dimensions and padded page stride as model_runner."""
    real_bytes = block_size * state_dim * 4
    if min(block_size, state_dim, blocks) <= 0 or page_bytes < real_bytes or page_bytes % 4:
        raise ValueError('Invalid compressor state page')
    return {'shape':[blocks, block_size, 1, state_dim],
            'stride':[page_bytes//4, state_dim, state_dim, 1],
            'operator_shape':[blocks, block_size, state_dim],
            'operator_stride':[page_bytes//4, state_dim, 1],
            'real_page_bytes':real_bytes, 'page_bytes':page_bytes,
            'padding_bytes':page_bytes-real_bytes, 'state_block_size':block_size}


def pinned_block_sizes(source):
    matches = [n.value for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == '_DSV4_BLOCK_SIZES' for t in n.targets)]
    if len(matches) != 1:
        raise ValueError('Pinned DeepSeek block layout changed')
    return ast.literal_eval(matches[0])


def ready_count(start, count):
    return (start+count)//4-start//4


def expected_slots(table, kv_block_size, start, count):
    return [[table[u//kv_block_size], u%kv_block_size]
            for u in range(start//4, (start+count)//4)]


def padded_slot(block_size, source):
    needles = ('int32_t padOffset = static_cast<int32_t>(kvBlockSize_ - 1);',
               'slotLocal.SetValue(slotOffset, -1);', 'slotLocal.SetValue(slotOffset + 1, padOffset);')
    if any(source.count(n) != 1 for n in needles):
        raise ValueError('Pinned invalid slot layout changed')
    return [-1,block_size-1]


def deployment_state_spec(model_source, constructor, table, block_size, state_dim):
    """Execute the pinned spec method without the model/backend import cycle."""
    methods = [m for c in ast.parse(model_source).body if isinstance(c,ast.ClassDef)
               and c.name == 'AscendCompressorStateCache' for m in c.body
               if isinstance(m,ast.FunctionDef) and m.name == 'get_kv_cache_spec']
    if len(methods) != 1:
        raise ValueError('Pinned state spec method changed')
    method = methods[0]
    module = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),method],type_ignores=[])
    namespace = {'AscendSlidingWindowMLASpec':constructor,'_dsv4_block_sizes':lambda:table}
    exec(compile(ast.fix_missing_locations(module),'pinned-state-spec.py','exec'),namespace)
    instance = SimpleNamespace(block_size=table[block_size][0][2],state_dim=state_dim,
                               compress_ratio=4,sliding_window=8,dtype=None)
    return namespace['get_kv_cache_spec'],instance


def run_contract(*, block_size=128, model_config=None, source_dir=None, initialize_new_blocks=False, artifact_directory=None):
    import torch
    import torch_npu  # noqa: F401
    from vllm_ascend.utils import enable_custom_op
    assert enable_custom_op()
    from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSlidingWindowMLASpec
    from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type
    assert get_ascend_device_type() != AscendDeviceType.A5, 'Fixed BF16 contract targets the retained A3 deployment'
    source_dir = Path(source_dir) if source_dir else Path('/vllm-workspace/vllm-ascend/vllm_ascend/models')
    layer_source = source_dir/'layer.py' if (source_dir/'layer.py').exists() else source_dir/'layer/attention/layer.py'
    model_source = source_dir/'model.py' if (source_dir/'model.py').exists() else source_dir/'deepseek_v4.py'
    block_sizes = pinned_block_sizes(layer_source.read_text())
    config = json.loads(Path(model_config).read_text()) if model_config else {
        'hidden_size':4096,'head_dim':512,'index_head_dim':128,'qk_rope_head_dim':64,'rms_norm_eps':1e-6}
    assert config['head_dim'] == 512
    spec = AscendMLAAttentionSpec(block_size=block_size,num_kv_heads=1,head_size=512,dtype=torch.bfloat16,
                                compress_ratio=4,model_version='deepseek_v4',cache_dtype_str='auto')
    assert spec.page_size_bytes == block_size*1024
    torch.npu.set_device(0)
    torch.manual_seed(37)
    dtype = torch.bfloat16
    x = (torch.randn(32,config['hidden_size'])*.2).to(dtype)
    results = {}
    from m0a.deepseek_repair import initialize_compressor_blocks
    operator = initialize_compressor_blocks if initialize_new_blocks else torch.ops._C_ascend.compressor
    for kind, head_dim in [('attention',config['head_dim']),('indexer',config['index_head_dim'])]:
        # Same pinned method and actual spec class; avoid registering/importing a model.
        method,obj = deployment_state_spec(model_source.read_text(),AscendSlidingWindowMLASpec,block_sizes,block_size,4*head_dim)
        obj.dtype = torch.float32
        state_spec = method(obj,SimpleNamespace(cache_config=SimpleNamespace(block_size=block_size)))
        layout = state_layout(state_spec.block_size,state_spec.head_size,state_spec.page_size_bytes)
        w = (torch.randn(2*head_dim,config['hidden_size'])*.02).to(dtype)
        weights,gate = w.npu(),torch.zeros_like(w).npu()
        ape,norm = torch.zeros(4,2*head_dim,dtype=torch.float32).npu(),torch.ones(head_dim,dtype=dtype).npu()
        physical = [3,1,4,2]  # Permuted ownership; block zero remains reserved.
        table = torch.tensor([physical],dtype=torch.int32).npu()
        def state(poison=0):
            raw = torch.full((layout['shape'][0],layout['page_bytes']//4),23.,dtype=torch.float32).npu()
            view = torch.as_strided(raw.flatten(),size=layout['shape'],stride=layout['stride']).squeeze(-2)
            view.fill_(poison)
            raw[0].zero_()
            return raw,view
        capture_reuse = False
        first_exported = False
        def call(values,start,cache,mapping=table):
            nonlocal first_exported
            raw,view = cache
            n,valid = len(values),ready_count(start,len(values))
            count = min(n,n//4+1)
            assert valid <= count
            sin = torch.zeros(count,config['qk_rope_head_dim'],dtype=dtype).npu()
            cos = torch.ones_like(sin)
            args = (values.npu(),weights,gate,view,ape,norm,sin,cos)
            kwargs = dict(state_block_table=mapping,cu_seqlens=torch.tensor([0,n],dtype=torch.int32).npu(),seqused=None,
                start_pos=torch.tensor([start],dtype=torch.int32).npu(),rope_head_dim=config['qk_rope_head_dim'],
                cmp_ratio=4,coff=2,norm_eps=config['rms_norm_eps'],rotary_mode=2,cache_mode=1)
            capsule = None
            if capture_reuse and not first_exported and artifact_directory:
                from m0a.deepseek_operator_diagnostics import snapshot_call
                capsule = snapshot_call(args,kwargs)
            result = operator(*args,**kwargs)
            torch.npu.synchronize()
            assert torch.count_nonzero(raw[0]).item() == 0, ('Sentinel overwritten',kind)
            assert torch.all(raw[1:,layout['real_page_bytes']//4:] == 23).item(), ('Padding overwritten',kind)
            output = result[:valid].cpu().float()
            if capsule and valid and not torch.allclose(output,baseline[start//4:start//4+valid],atol=.035,rtol=.035):
                capsule.update(operator='_C_ascend.compressor.default',rank=0,kind=kind,
                               observation={'comparison':'state_reuse_reference_mismatch','start_position':start})
                torch.save(capsule,Path(artifact_directory)/f'contract-{kind}-first.pt')
                first_exported = True
            return output
        baseline = call(x,0,state())
        projection = x.float() @ w.float().T
        expected = []
        for i in range(len(x)//4):
            groups = [projection[4*i:4*(i+1),head_dim:]]
            if i: groups.insert(0,projection[4*i-4:4*i,:head_dim])
            v = torch.cat(groups).mean(0)
            expected.append(v/torch.sqrt((v*v).mean()+config['rms_norm_eps']))
        expected = torch.stack(expected)
        def compare(output,target,label):
            assert output.shape == target.shape and torch.allclose(output,target,atol=.035,rtol=.035), (
                kind,label,float((output-target).abs().max()) if output.numel() else 0)
        compare(baseline,expected,'Independent FP32 reference')
        dependencies = []
        for position in range(16):
            changed = x.clone()
            changed[position] = (changed[position].float()+torch.randn(config['hidden_size'])*.7).to(dtype)
            output = call(changed,0,state())
            actual = [i for i in range(8) if float((output[i]-baseline[i]).abs().max()) > .02]
            wanted = [i for i in range(8) if max(0,4*i-4) <= position < 4*(i+1)]
            assert actual == wanted, ('Support mismatch',kind,position,actual,wanted)
            dependencies.append({'token_position':position,'dependent_native_ids':actual})
        chunks = []
        for sizes in ((3,1,3,1,5,3,7,1,8),(5,2,1,8,3,5,8),(4,)*8):
            cache,start,outputs,counts = state(),0,[],[]
            for n in sizes:
                output = call(x[start:start+n],start,cache)
                outputs.append(output);counts.append(len(output));start += n
            merged = torch.cat(outputs)
            compare(merged,baseline,('Cross-block/chunk state',sizes))
            chunks.append({'chunk_sizes':list(sizes),'valid_units_per_chunk':counts,
                           'max_error':float((merged-baseline).abs().max())})
        prefixes = []
        for n in (1,3,4,5,7,8,9,15,16,17,31,32):
            output = call(x[:n],0,state());compare(output,baseline[:n//4],('Ready prefix',n))
            prefixes.append({'prefix_tokens':n,'ready_units':len(output)})
        reuse, reuse_failures = [],[]
        for label in ('released_request_blocks','poisoned_free_blocks'):
            errors = []
            for repetition in range(20):
                cache = state(17 if label == 'poisoned_free_blocks' else 0)
                if label == 'released_request_blocks': call(x.flip(0),0,cache)
                capture_reuse = True
                start,outputs = 0,[]
                for n in (3,2,4,7,1,7,8):
                    outputs.append(call(x[start:start+n],start,cache));start += n
                capture_reuse = False
                merged = torch.cat(outputs)
                error = float((merged-baseline).abs().max());errors.append(error)
                if not torch.allclose(merged,baseline,atol=.035,rtol=.035):
                    reuse_failures.append({'case':label,'repetition':repetition,'max_error':error})
            reuse.append({'case':label,'repetitions':20,'max_errors':errors,'sentinel_zero':True})
        call(x[:4],0,state(),torch.zeros_like(table))
        results[kind] = dict(layout=layout,actual_cache_spec=repr(state_spec),physical_block_table=[physical],
            max_reference_error=float((baseline-expected).abs().max()),dependencies=dependencies,
            chunk_state_tests=chunks,prefix_tests=prefixes,reuse_tests=reuse,reuse_failures=reuse_failures,
            independent_reference_passed=True,sentinel_preserved=True,padding_preserved=True)
        if artifact_directory:
            (Path(artifact_directory)/'compressor-contract-progress.json').write_text(json.dumps({
                'completed_compressors':results,'reuse_repetitions':20,'model_config':config,
                'initializes_new_request_blocks':initialize_new_blocks},indent=2)+'\n')
    slots,kv_physical = [],[3,1,4]
    kv_table = torch.tensor([kv_physical],dtype=torch.int32).npu()
    invalid_slot = padded_slot(block_size,(source_dir/'slot_mapping.h').read_text())
    for start,n in ((0,3),(0,4),(0,7),(3,1),(5,3),(7,9),(4*block_size-3,9)):
        count = n//4+1
        cos = torch.ones(start+n,64,dtype=dtype).npu();sin = torch.zeros_like(cos)
        output = torch.ops._C_ascend.compressor_metadata(cos,sin,
            torch.tensor([0,n],dtype=torch.int32).npu(),torch.tensor([start],dtype=torch.int32).npu(),
            kv_table,block_size,2,4,count,1)
        torch.npu.synchronize()
        mapping = output[2].cpu().tolist()
        positive = [v for v in mapping if v[0]>=0 and v[1]>=0]
        wanted = expected_slots(kv_physical,block_size,start,n)
        assert positive == wanted, ('Slot/ready mapping mismatch',start,n,mapping,wanted)
        assert all(v == invalid_slot for v in mapping[len(positive):]), ('Invalid padded slot',mapping,invalid_slot)
        slots.append({'start_position':start,'chunk_tokens':n,'slot_mapping':mapping})
    failures = any(c['reuse_failures'] for c in results.values())
    return {'passed':not failures,'repairable_state_reuse':failures,'reuse_repetitions':20,
            'initializes_new_request_blocks':initialize_new_blocks,'support_contract':'c4_overlap_v1','device':'npu:0','dtype':'bfloat16',
            'operator_schema':str(torch.ops._C_ascend.compressor.default._schema),'compressors':results,
            'slot_mapping_tests':slots,'native_kv_payload_bytes_per_layer':1024,'page128_bytes':128*1024,
            'attention_page_bytes':spec.page_size_bytes,'actual_cache_spec':repr(spec),
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(source_dir).glob('*') if p.is_file()} if source_dir else {}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--block-size',type=int,default=128)
    parser.add_argument('--model-config',type=Path)
    parser.add_argument('--source-dir',type=Path)
    parser.add_argument('--initialize-new-blocks',action='store_true')
    parser.add_argument('--artifact-directory',type=Path)
    args = parser.parse_args()
    try:
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        result = run_contract(block_size=args.block_size,model_config=args.model_config,source_dir=args.source_dir,
                              initialize_new_blocks=args.initialize_new_blocks,artifact_directory=args.artifact_directory)
    except BaseException as error:
        partial = args.artifact_directory/'compressor-contract-progress.json' if args.artifact_directory else None
        data = {'passed':False,'error':str(error)}
        if partial and partial.exists():
            data['partial_contract'] = json.loads(partial.read_text())
        args.output.write_text(json.dumps(data,indent=2)+'\n');raise
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
