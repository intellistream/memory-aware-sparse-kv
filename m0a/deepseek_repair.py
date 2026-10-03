"""Pinned experimental source preparation and per-rank deterministic evidence."""
import ast
import json
import os
from pathlib import Path


def patch_worker(source, runtime_path):
    needle = '        torch.npu.set_device(device)'
    if source.count(needle) != 1:
        raise ValueError('Pinned worker device binding changed')
    replacement = (needle + '\n        import runpy\n'
        '        _memecho_configure = runpy.run_path(' + repr(str(runtime_path)) + ')["configure_worker"]\n'
        '        _memecho_configure(self.rank, device, "device_bound")')
    source = source.replace(needle,replacement)
    needle = '        self._init_worker_distributed_environment()'
    if source.count(needle) != 1:
        raise ValueError('Pinned distributed initialization changed')
    source = source.replace(needle,needle+'\n        _memecho_configure(self.rank, device, "distributed_initialized")')
    compile(source,'experimental-worker.py','exec')
    return source


def configure_worker(rank, device, stage):
    directory = os.environ.get('MEMECHO_DETERMINISTIC_DIR')
    if not directory:
        return
    import torch_npu
    level = os.environ.get('MEMECHO_DETERMINISTIC_LEVEL')
    if level and stage == 'device_bound':
        torch_npu.npu.set_deterministic_level(int(level))
    # This getter exists in the fixed deployed torch_npu, checked before patching.
    actual = int(torch_npu._C._npu_get_deterministic_level())
    if level and actual != int(level):
        raise RuntimeError('Rank deterministic setting did not take effect')
    data = {'rank':int(rank),'device':str(device),'pid':os.getpid(),'stage':stage,
            'hccl_deterministic':os.environ.get('HCCL_DETERMINISTIC'),
            'requested_level':int(level) if level else None,'actual_level':actual}
    path = Path(directory);path.mkdir(parents=True,exist_ok=True)
    target = path/f'rank{rank}-{stage}.json'
    temporary = target.with_suffix('.tmp');temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(target)


def validate_deterministic_ranks(directory, candidate):
    environment = candidate.get('environment',{})
    rows = []
    for rank in range(8):
        for stage in ('device_bound','distributed_initialized'):
            row = json.loads((Path(directory)/f'rank{rank}-{stage}.json').read_text())
            if row['rank'] != rank or row['stage'] != stage or row['device'] != f'npu:{rank}':
                raise ValueError('Deterministic rank/device evidence mismatch')
            if 'HCCL_DETERMINISTIC' in environment and row['hccl_deterministic'] != environment['HCCL_DETERMINISTIC']:
                raise ValueError('HCCL deterministic setting mismatch')
            expected = environment.get('MEMECHO_DETERMINISTIC_LEVEL')
            if expected and (row['requested_level'] != int(expected) or row['actual_level'] != int(expected)):
                raise ValueError('NPU deterministic level mismatch')
            rows.append(row)
    return {'verified_ranks':list(range(8)),'before_and_after_distributed_init':True,'rows':rows}


def instrument_source(source, runtime_path, installer):
    if installer not in ('install_model','install_dsa'):
        raise ValueError('Unknown diagnostic installer')
    ast.parse(source)
    root = str(Path(runtime_path).resolve().parents[1])
    result = source+'\n# Isolated operator diagnostics; never mounted for acceptance requests.\nimport sys as _memecho_sys\n_memecho_sys.path.insert(0,'+repr(root)+')\nfrom m0a import deepseek_operator_diagnostics as _memecho_diagnostics\n_memecho_diagnostics.'+installer+'(globals())\n'
    compile(result,'diagnostic-source.py','exec')
    return result


def new_request_physical_blocks(table, starts):
    if len(table) != len(starts):
        raise ValueError('State block table/request length mismatch')
    return sorted({block for row,start in zip(table,starts) if start == 0 for block in row if block > 0})


def initialize_compressor_blocks(*args, **kwargs):
    """Clear owned new-request blocks; preserve sentinel and page padding."""
    import torch
    cache = args[3]
    blocks = new_request_physical_blocks(kwargs['state_block_table'].detach().cpu().tolist(),
                                         kwargs['start_pos'].detach().cpu().tolist())
    for block in blocks:
        if block >= cache.shape[0]:
            raise ValueError('Physical compressor block outside cache')
        cache[block].zero_()
    return torch.ops._C_ascend.compressor(*args,**kwargs)


def patch_compressor_initialization(source, runtime_path):
    needle = 'torch.ops._C_ascend.compressor('
    if source.count(needle) != 2:
        raise ValueError('Pinned attention/indexer compressor call sites changed')
    source = source.replace(needle,'_memecho_initialized_compressor(')
    source += '\nimport runpy as _memecho_repair_runpy\n_memecho_initialized_compressor = _memecho_repair_runpy.run_path('+repr(str(runtime_path))+')["initialize_compressor_blocks"]\n'
    compile(source,'repaired-dsa.py','exec')
    return source


def verify_state_repair(before, after):
    if before.get('passed') or not before.get('repairable_state_reuse') or not after.get('passed'):
        raise ValueError('No reproduced and repaired state reuse defect')
    for data in (before,after):
        if data.get('reuse_repetitions') != 20 or set(data.get('compressors',{})) != {'attention','indexer'}:
            raise ValueError('Incomplete twenty-replay independent state verification')
        for compressor in data['compressors'].values():
            if not compressor['sentinel_preserved'] or not compressor['padding_preserved']:
                raise ValueError('State repair damaged sentinel/padding')
            if not compressor.get('independent_reference_passed'):
                raise ValueError('State repair lacks independent reference')
    if not any(c['reuse_failures'] for c in before['compressors'].values()):
        raise ValueError('State reuse defect was not reproduced')
    if any(c['reuse_failures'] for c in after['compressors'].values()):
        raise ValueError('State repair still fails reference')
    return {'operator':'_C_ascend.compressor','kind':'new_request_physical_block_initialization',
            'reproduced_before':True,'independent_reference_passed':True,'replays_after':20,
            'root_cause_scope':'synthetic C4 state reuse; formal model stability still requires full acceptance'}
