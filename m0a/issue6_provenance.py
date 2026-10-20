"""Capture non-secret, post-run software provenance linked to a sealed native manifest."""
import argparse
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from .glm_agent_cases import sha_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    packages = {}
    for name in ('torch', 'torch-npu', 'vllm', 'vllm-ascend', 'transformers', 'numpy', 'matplotlib'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    files = [Path('/vllm-workspace/vllm-ascend/vllm_ascend/device/device_op.py'),
             Path('/root/models/echo-glm-model/chat_template.jinja')]
    result = {'observed_at_utc': datetime.now(timezone.utc).isoformat(),
              'observation_scope': 'post-run receipt; sealed capture manifest is not rewritten',
              'native_manifest_sha256': sha_file(args.native_manifest),
              'python': platform.python_version(), 'machine': platform.machine(), 'packages': packages,
              'ascend_rt_visible_devices': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
              'mounted_npus': sorted(p.name for p in Path('/dev').glob('davinci[0-9]*')),
              'runtime_files_sha256': {str(p): sha_file(p) for p in files if p.is_file()}}
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
