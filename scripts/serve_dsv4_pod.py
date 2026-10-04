#!/usr/bin/env python3
"""Install, start and verify the persistent DeepSeek service in the current Pod."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from m0a.pod_runtime import SupervisorService, health, port_available, preflight, service_command, sha256
from m0a.run_requests import request
from m0a.working_set import require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/root/memory-aware-sparse-kv'))
    parser.add_argument('--model-dir', type=Path, default=Path('/models/DeepSeek-V4-Flash-W8A8'))
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan = {'root': str(args.root), 'model_dir': str(args.model_dir),
            'command': service_command(args.model_dir), 'supervisor': str(args.root / 'runtime/deepseek-pod/supervisord.conf')}
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    require(args.root.is_absolute() and args.model_dir.is_absolute(), 'Absolute paths required')
    free = port_available()
    inventory = preflight(args.model_dir, expect_port_free=free)
    service = SupervisorService(args.root, args.model_dir)
    if free:
        service.install()
        service.control('start')
    else:
        saved = json.loads((service.runtime / 'service-command.json').read_text())
        require(saved['command'] == plan['command'] and saved['model_dir'] == str(args.model_dir),
                'Existing service configuration differs')
    models = health(args.model_dir)
    result = request({'model': 'dsv4', 'messages': [{'role': 'user', 'content': 'Reply with OK.'}],
                      'temperature': 0, 'max_tokens': 8})
    require(result['choices'] and result['usage']['completion_tokens'] > 0, 'Short request failed')
    evidence = {'service': service.identity(), 'models': models, 'short_request_id': result['id'],
                'supervisor_config_sha256': sha256(service.conf), 'model_metadata': inventory['metadata'],
                'shards': len(inventory['weights']), 'read_only_mount': inventory['read_only_mount']}
    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
