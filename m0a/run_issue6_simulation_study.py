"""Reproducible no-NPU study; run in an isolated environment, with a new output directory."""
import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import platform
import subprocess

from .issue6_simulator import Timing, run_grid, synthetic, load_trace, sha
from .issue6_predictors import STRATEGIES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--native-trace', type=Path, required=True)
    p.add_argument('--native-audit', type=Path, required=True)
    p.add_argument('--model-config', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.model_config.read_text())
    assert config['model_type'] == 'deepseek_v4' and config['index_topk'] == 512
    assert config['head_dim'] == 512 and config['num_key_value_heads'] == 1
    manifest = dict(model_config_path=str(args.model_config), model_config_sha256=sha(args.model_config),
        model_config=config, platform=platform.platform(), npu_used=False,
        C4_layers=[i for i, v in enumerate(config['compress_ratios'][:config['num_hidden_layers']]) if v == 4],
        BF16_C4_unit_bytes_assumption=1024, model_weights_loaded=False,
        requirements_sha256=sha(Path(__file__).with_name('issue6-simulation-requirements.txt')),
        study_code_sha256=sha(__file__))
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    subprocess.run([__import__('sys').executable, '-m', 'pip', 'freeze'], stdout=(args.output / 'pip-freeze.txt').open('w'), check=True)
    strategies = [s for s in STRATEGIES if s != 'echo_style']
    for scenario in ('stable', 'shift', 'random', 'misleading'):
        for seed in (7, 19, 31):
            name = f'{scenario}-seed{seed}'
            print('START', name, flush=True)
            rows = synthetic(seed=seed, scenario=scenario)
            run_grid(rows, args.output / name, capacities=[512, 1024], budgets=[128, 512],
                     timings=[Timing()], strategies=strategies)
            with gzip.open(args.output / name / 'synthetic-trace.jsonl.gz', 'wt') as f:
                for row in rows:
                    f.write(json.dumps(row) + '\n')
            (args.output / name / 'input-receipt.json').write_text(json.dumps(dict(
                scenario=scenario, seed=seed, tasks=12, queries=32, units=4096, topk=512,
                trace_sha256=sha(args.output / name / 'synthetic-trace.jsonl.gz'))))
            print('DONE', name, flush=True)
    print('START sensitivity', flush=True)
    rows = synthetic(seed=7, scenario='shift')
    run_grid(rows, args.output / 'sensitivity', capacities=[512], budgets=[512],
        timings=[Timing(bandwidth_GBs=b, lead_us=l, prediction_us=c)
                 for b in (8, 16, 32) for l in (0, 10, 50, 200) for c in (0, 5, 50)],
        strategies=['none', 'last_selection', 'global_frequency', 'adaptive', 'oracle_diagnostic'])
    for streams in (2, 4):
        print('START contention', streams, flush=True)
        run_grid(rows, args.output / f'contention-{streams}', capacities=[512], budgets=[512],
            timings=[Timing()], strategies=['none', 'last_selection', 'adaptive', 'oracle_diagnostic'], concurrency=streams)
    for units in (2048, 16384):
        print('START length', units, flush=True)
        run_grid(synthetic(seed=7, scenario='shift', units=units), args.output / f'length-{units}',
            capacities=[512], budgets=[512], timings=[Timing()],
            strategies=['none', 'last_selection', 'global_frequency', 'semantic_lsa', 'event_history', 'adaptive', 'oracle_diagnostic'])
    print('START audited-GLM-replay', flush=True)
    rows, receipt = load_trace(args.native_trace, args.native_audit)
    run_grid(rows, args.output / 'audited-GLM-replay', capacities=[4096], budgets=[512],
        timings=[Timing(lead_us=0), Timing(lead_us=50)],
        strategies=['none', 'last_selection', 'echo_style', 'global_frequency', 'stage_frequency',
                    'semantic_lsa', 'event_history', 'adaptive', 'oracle_diagnostic'])
    (args.output / 'audited-GLM-replay' / 'input-receipt.json').write_text(json.dumps(receipt, indent=2))
    print('DONE all experiments', flush=True)


if __name__ == '__main__':
    main()
