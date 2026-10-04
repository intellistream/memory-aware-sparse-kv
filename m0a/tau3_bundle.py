"""Create exact, hash-checked offline τ³ source and wheel archives."""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

TAU_COMMIT = 'fc0055dc4e0a316c3f83133267fbd6faaa770992'
TAU_URL = 'https://github.com/sierra-research/tau2-bench.git'
SOURCE_PREFIXES = ('src', 'data/tau2/domains/retail',
                   'data/tau2/domains/banking_knowledge')


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


def build(directory: Path, requirements: Path, *, tau_source: Path,
          wheelhouse: Path) -> dict:
    uv_source = shutil.which('uv')
    if not uv_source:
        raise RuntimeError('uv executable is required for the offline Pod environment')
    uv_version = subprocess.check_output([uv_source, '--version'], text=True).strip()
    if uv_version != 'uv 0.11.8 (x86_64-unknown-linux-gnu)':
        raise RuntimeError('Expected the pinned Linux uv 0.11.8 executable')
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
                        '--dest', str(wheelhouse), '-r', str(requirements)], check=True, timeout=2400)
    wheels = [p.name for p in wheelhouse.glob('*.whl')]
    if len(wheels) < 70:
        raise RuntimeError('Pinned τ³ wheelhouse is incomplete')
    uv_target = directory / 'tau3-uv'
    shutil.copy2(uv_source, uv_target)
    uv_target.chmod(0o755)
    return {'source': pack(directory, tau_source, source_names,
                           'tau3-source-manifest.json', {'commit': head}),
            'wheels': pack(directory, wheelhouse, wheels,
                           'tau3-wheelhouse-manifest.json',
                           {'requirements_sha256': sha(requirements)}),
            'uv': {'archive': uv_target.name, 'sha256': sha(uv_target),
                   'size': uv_target.stat().st_size, 'version': uv_version}}
