"""Validate a trace-off model baseline before a trace-enabled pilot."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--response', type=Path, required=True)
    parser.add_argument('--model-profile', required=True)
    args = parser.parse_args()
    data = json.loads(args.response.read_text())
    if data.get('kind') != 'model_baseline':
        raise RuntimeError('Not a model baseline artifact')
    if data.get('model_profile') != args.model_profile:
        raise RuntimeError('Model baseline profile mismatch')
    if data.get('trace_enabled') is not False:
        raise RuntimeError('Model baseline must be captured with trace disabled')
    if data.get('repetitions') != 3 or len(data.get('records', [])) != 3:
        raise RuntimeError('Model baseline requires exactly three repetitions')
    if not data.get('identical'):
        raise RuntimeError('Model baseline signatures differ')
    print(json.dumps({
        'model_profile': args.model_profile,
        'repetitions': 3,
        'identical': True,
    }, sort_keys=True))


if __name__ == '__main__':
    main()
