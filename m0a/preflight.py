"""Require healthy requested 910B2 devices with enough free HBM."""

import argparse
import re
import subprocess

MAX_USED_HBM_MB = 6000  # 60.96 GiB usable - 0.90 * 60.96 GiB, rounded down.


def parse_devices(value: str) -> tuple[int, ...]:
    devices = tuple(int(item) for item in value.split(',') if item != '')
    if not devices or len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError('devices must be a unique comma-separated list')
    if any(device < 0 or device > 7 for device in devices):
        raise argparse.ArgumentTypeError('device IDs must be between 0 and 7')
    return devices


def get_hbm_usage() -> dict[int, int]:
    text = subprocess.check_output(['npu-smi', 'info'], text=True)
    lines = text.splitlines()
    usage = {}
    for index, line in enumerate(lines[:-1]):
        card = re.match(r'\|\s*([0-7])\s+910B2\s+\|\s+OK\b', line)
        if not card:
            continue
        pairs = re.findall(r'(\d+)\s*/\s*(\d+)', lines[index + 1])
        if not pairs:
            raise RuntimeError(f'Cannot parse HBM usage for NPU {card.group(1)}')
        usage[int(card.group(1))] = int(pairs[-1][0])
    return usage


def check(devices: tuple[int, ...], max_used_hbm_mb: int) -> dict[int, int]:
    usage = get_hbm_usage()
    missing = sorted(set(devices) - usage.keys())
    if missing:
        raise RuntimeError(
            f'Expected healthy 910B2 devices {list(devices)}, missing {missing}; '
            f'found {usage}'
        )
    requested = {device: usage[device] for device in devices}
    occupied = {
        device: used for device, used in requested.items()
        if used > max_used_hbm_mb
    }
    print('HBM used (MB):', requested)
    if occupied:
        raise RuntimeError(
            f'Preflight limit {max_used_hbm_mb} MB exceeded: {occupied}'
        )
    return requested


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', type=parse_devices, default=tuple(range(8)))
    parser.add_argument('--max-used-hbm-mb', type=int, default=MAX_USED_HBM_MB)
    args = parser.parse_args()
    if args.max_used_hbm_mb < 0:
        parser.error('--max-used-hbm-mb must be non-negative')
    check(args.devices, args.max_used_hbm_mb)


if __name__ == '__main__':
    main()
