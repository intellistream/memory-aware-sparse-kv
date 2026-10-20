"""Serve the pinned GLM auxiliary model on loopback for real benchmark decisions."""
import json
import os
import sys
import time
import signal
from pathlib import Path
from .glm_agent_cases import verify_model, REVISION

# Spawned engine processes re-import this module but do not execute main().
# Apply the pinned-config compatibility patch in each interpreter.
import transformers.configuration_utils as config
if 'deepseek_sparse_attention' not in config.ALLOWED_LAYER_TYPES:
    config.ALLOWED_LAYER_TYPES += ('deepseek_sparse_attention',)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--stop':
        stop_server(Path(sys.argv[2]))
        return
    model = Path('/root/models/echo-glm-model')
    verify_model(model)
    physical = int(os.environ['ISSUE6_PHYSICAL_NPU'])
    if not Path(f'/dev/davinci{physical}').exists():
        raise ValueError('Physical NPU is not mounted')
    if os.environ.get('ASCEND_RT_VISIBLE_DEVICES') != os.environ['ISSUE6_RUNTIME_DEVICE']:
        raise ValueError('Runtime visibility differs from verified mapping')
    import torch
    import torch_npu
    if torch.npu.device_count() != 1:
        raise ValueError('Expected one visible NPU')
    torch.npu.set_device(0)
    receipt = {'pid': os.getpid(), 'model': str(model), 'revision': REVISION,
               'physical_npu': physical, 'runtime_device': os.environ['ISSUE6_RUNTIME_DEVICE'],
               'process_device': 0, 'argv': sys.argv[1:]}
    Path(os.environ['ISSUE6_SERVER_RECEIPT']).write_text(json.dumps(receipt, indent=2))
    from vllm.entrypoints.cli.main import main as cli_main
    cli_main()


def stop_server(receipt_path):
    """Stop only processes carrying this run's exact server receipt marker."""
    from .pod_runtime import process_identity, same_process
    receipt_path = receipt_path.resolve()
    receipt = json.loads(receipt_path.read_text())
    marker = ('ISSUE6_SERVER_RECEIPT=' + str(receipt_path)).encode()
    ancestors, pid = set(), os.getpid()
    while pid > 1:
        ancestors.add(pid)
        try:
            pid = int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[1])
        except FileNotFoundError:
            break
    targets = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit() or int(directory.name) in ancestors:
            continue
        try:
            identity = process_identity(int(directory.name))
            if identity and marker in (directory / 'environ').read_bytes().split(b'\0'):
                targets.append(identity)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    # Verify this marker actually includes the launched server if it is still live.
    if process_identity(receipt['pid']) and receipt['pid'] not in {t['pid'] for t in targets}:
        raise ValueError('Live server does not carry the expected run marker')
    for identity in targets:
        if same_process(identity):
            os.kill(identity['pid'], signal.SIGTERM)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and any(same_process(t) for t in targets):
        time.sleep(0.2)
    for identity in targets:
        if same_process(identity):
            os.kill(identity['pid'], signal.SIGKILL)
    print(json.dumps({'stopped_owned_server_pids': [t['pid'] for t in targets]}))


if __name__ == '__main__':
    main()
