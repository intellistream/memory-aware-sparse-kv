"""Create exact, hash-checked offline τ³ source and wheel archives."""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

TAU_COMMIT = 'fc0055dc4e0a316c3f83133267fbd6faaa770992'
TAU_URL = 'https://github.com/sierra-research/tau2-bench.git'
SOURCE_PREFIXES = ('src', 'data/tau2/domains/retail',
                   'data/tau2/domains/banking_knowledge')
TARGET_PLATFORMS = ('manylinux_2_38_aarch64', 'manylinux_2_36_aarch64',
                    'manylinux_2_34_aarch64', 'manylinux_2_31_aarch64',
                    'manylinux_2_28_aarch64', 'manylinux_2_17_aarch64')


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def pack(directory: Path, source: Path, names: list[str], manifest_name: str,
         provenance: dict) -> dict:
    records = [{'path': name, 'size': (source / name).stat().st_size,
                'sha256': sha(source / name)} for name in sorted(names)]
    manifest = {**provenance, 'files': records}
    data = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
    target = directory / (manifest_name.removesuffix('.json') + '.tar.gz')
    with tarfile.open(target, 'w:gz') as archive:
        info = tarfile.TarInfo(manifest_name)
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
        for record in records:
            info = tarfile.TarInfo(record['path'])
            info.size = record['size']
            with (source / record['path']).open('rb') as content:
                archive.addfile(info, content)
    return {'archive': target.name, 'sha256': sha(target), 'size': target.stat().st_size,
            'manifest': manifest_name, 'files': len(records)}


def prepare_uv(cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    wheels = list(cache.glob('uv-0.11.8-*.whl'))
    if not wheels:
        subprocess.run([sys.executable, '-m', 'pip', 'download', '--only-binary=:all:',
                        '--no-deps', '--dest', str(cache), '--platform',
                        'manylinux_2_17_aarch64', '--python-version', '3.12',
                        '--implementation', 'cp', '--abi', 'cp312', 'uv==0.11.8'],
                       check=True, timeout=1200)
        wheels = list(cache.glob('uv-0.11.8-*.whl'))
    if len(wheels) != 1 or 'aarch64' not in wheels[0].name:
        raise RuntimeError('Expected exactly one pinned aarch64 uv wheel')
    target = cache / 'uv'
    with zipfile.ZipFile(wheels[0]) as archive:
        member = 'uv-0.11.8.data/scripts/uv'
        with archive.open(member) as source, target.open('wb') as output:
            shutil.copyfileobj(source, output)
    target.chmod(0o755)
    with target.open('rb') as source:
        header = source.read(20)
    if header[:4] != b'\x7fELF' or int.from_bytes(header[18:20], 'little') != 183:
        raise RuntimeError('Pinned uv binary is not aarch64 ELF')
    return target


def build(directory: Path, requirements: Path, *, tau_source: Path,
          wheelhouse: Path, uv_cache: Path) -> dict:
    uv_source = prepare_uv(uv_cache)
    if not tau_source.exists():
        subprocess.run(['git', 'clone', '--depth', '1', '--branch', 'v1.0.1',
                        TAU_URL, str(tau_source)], check=True, timeout=1200)
    head = subprocess.check_output(['git', '-C', str(tau_source), 'rev-parse', 'HEAD'], text=True).strip()
    if head != TAU_COMMIT:
        raise RuntimeError('Local τ³ checkout is not pinned v1.0.1')
    dirty = subprocess.check_output(['git', '-C', str(tau_source), 'status', '--porcelain', '--',
                                     *SOURCE_PREFIXES], text=True).strip()
    if dirty:
        raise RuntimeError('Local τ³ source or domain data is modified')
    tracked = subprocess.check_output(['git', '-C', str(tau_source), 'ls-files', '-z', '--',
                                       *SOURCE_PREFIXES]).decode().split('\0')
    source_names = [name for name in tracked if name and (tau_source / name).is_file()]
    if not source_names:
        raise RuntimeError('No tracked τ³ workload files')
    if not wheelhouse.exists() or not list(wheelhouse.glob('*.whl')):
        wheelhouse.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, '-m', 'pip', 'download', '--only-binary=:all:',
                        '--dest', str(wheelhouse),
                        *[option for platform in TARGET_PLATFORMS
                          for option in ('--platform', platform)],
                        '--python-version', '3.12', '--implementation', 'cp',
                        '--abi', 'cp312', '--abi', 'abi3', '--abi', 'none',
                        '-r', str(requirements)], check=True, timeout=2400)
    wheels = [p.name for p in wheelhouse.glob('*.whl')]
    if len(wheels) < 70 or not any('aarch64' in name for name in wheels) or any('x86_64' in name for name in wheels):
        raise RuntimeError('Pinned aarch64 τ³ wheelhouse is incomplete or mixed-architecture')
    uv_target = directory / 'tau3-uv'
    shutil.copy2(uv_source, uv_target)
    uv_target.chmod(0o755)
    return {'source': pack(directory, tau_source, source_names,
                           'tau3-source-manifest.json', {'commit': head}),
            'wheels': pack(directory, wheelhouse, wheels,
                           'tau3-wheelhouse-manifest.json',
                           {'requirements_sha256': sha(requirements)}),
            'uv': {'archive': uv_target.name, 'sha256': sha(uv_target),
                   'size': uv_target.stat().st_size, 'version': '0.11.8',
                   'machine': 'aarch64'}}
