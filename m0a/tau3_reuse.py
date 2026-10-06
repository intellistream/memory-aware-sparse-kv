"""Hash checked reuse of a completed τ³ collection and optional sealed pairs."""
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

from .contracts import sha256_file
from .deepseek_validation import write_json
from .working_set import require

METADATA = ('episodes.json', 'collection-summary.json', 'rejected-episodes.json',
            'tau3-provenance.json', 'pairs.json', 'event-audit.json',
            'context-feasibility.json')


def _archive_name(name: str) -> str:
    return 'old-pairs.json' if name == 'pairs.json' else name


def source_records(source: Path) -> dict:
    source = Path(source)
    run_id = source.name
    final = source / 'final-checksums.json'
    status = json.loads((source / 'final-status.json').read_text())
    require(status['run_id'] == run_id and status['hash_sync'] == 'verified',
            'Source run has no verified final synchronization')
    require(status['manifest_sha256'] == sha256_file(final), 'Source final manifest changed')
    inventory = json.loads(final.read_text())['files']
    indexed = {row['path']: row for row in inventory}
    require(len(indexed) == len(inventory), 'Duplicate source inventory paths')
    for row in inventory:
        name = row['path']
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,
                'Unsafe source inventory path')
        path = source / name
        require(path.is_file() and path.stat().st_size == row['size'] and
                sha256_file(path) == row['sha256'], 'Source final inventory mismatch: ' + name)
    episodes = json.loads((source / 'episodes.json').read_text())
    require(len(episodes) >= 96, 'Source collection is incomplete')
    from .tau3_workload import DOMAINS
    for domain in DOMAINS:
        selected = [e['task_id'] for e in episodes if e['domain'] == domain]
        require(len(set(selected)) >= 48 and len(selected) == len(set(selected)),
                f'Insufficient independent source tasks in {domain}')
    log_prefix = '' if any(name.startswith('episode-') for name in indexed) else 'source-inputs/'
    names = sorted(name for name in indexed if name.startswith(log_prefix + 'episode-')
                   and name.endswith('.jsonl'))
    require(names and set(METADATA) <= set(indexed), 'Source input inventory incomplete')
    rejected = json.loads((source / 'rejected-episodes.json').read_text())
    expected_logs = {f'episode-{e["domain"]}-{e["task_id"]}.jsonl' for e in episodes}
    expected_logs.update(Path(row['episode_log']).name for row in rejected)
    log_root = source / log_prefix
    require({Path(name).name for name in names} == expected_logs ==
            {p.name for p in log_root.glob('episode-*.jsonl')},
            'Complete episode log set differs from source inventory')
    rows = []
    for name in (*METADATA, *names):
        row = indexed[name]
        path = source / name
        require(path.is_file() and path.stat().st_size == row['size'] and
                sha256_file(path) == row['sha256'], 'Source file hash mismatch: ' + name)
        rows.append({'path': _archive_name(Path(name).name), 'source_path': name,
                     'size': row['size'], 'sha256': row['sha256']})
    return {'source_run_id': run_id, 'source_final_manifest_sha256': sha256_file(final),
            'files': rows, 'old_pairs_sha256': indexed['pairs.json']['sha256']}


def build_archive(source: Path, directory: Path) -> dict:
    manifest = source_records(source)
    archive = Path(directory) / 'tau3-source-inputs.tar.gz'
    with tarfile.open(archive, 'w:gz') as output:
        for row in manifest['files']:
            output.add(Path(source) / row['source_path'], arcname=row['path'], recursive=False)
    write_json(Path(directory) / 'tau3-source-inputs-manifest.json', manifest)
    return {'archive': archive.name, 'size': archive.stat().st_size,
            'sha256': sha256_file(archive),
            'manifest_sha256': sha256_file(Path(directory) / 'tau3-source-inputs-manifest.json'),
            'source_run_id': manifest['source_run_id'], 'old_pairs_sha256': manifest['old_pairs_sha256']}


def verify_snapshot(directory: Path, *, complete: bool = True) -> dict:
    directory = Path(directory)
    manifest = json.loads((directory / 'tau3-source-inputs-manifest.json').read_text())
    rows = manifest['files']
    paths = {row['path'] for row in rows}
    require(len(paths) == len(rows) and paths == {_archive_name(Path(row['source_path']).name) for row in rows},
            'Invalid source snapshot manifest')
    require(paths >= {'episodes.json', 'collection-summary.json', 'rejected-episodes.json',
                      'tau3-provenance.json', 'old-pairs.json', 'event-audit.json',
                      'context-feasibility.json'}, 'Source snapshot lacks metadata')
    if complete:
        actual = {str(path.relative_to(directory)) for path in directory.rglob('*') if path.is_file()}
        require(actual == paths | {'tau3-source-inputs-manifest.json'},
                'Source snapshot contains missing or unexpected files')
    for row in rows:
        path = directory / row['path']
        require(path.is_file() and path.stat().st_size == row['size'] and
                sha256_file(path) == row['sha256'], 'Source snapshot hash mismatch: ' + row['path'])
    return manifest


def unpack_archive(archive: Path, manifest_path: Path, destination: Path, expected: dict) -> dict:
    archive, manifest_path, destination = Path(archive), Path(manifest_path), Path(destination)
    require(archive.stat().st_size == expected['size'] and sha256_file(archive) == expected['sha256'],
            'Source archive hash mismatch')
    require(sha256_file(manifest_path) == expected['manifest_sha256'],
            'Source manifest hash mismatch')
    manifest = json.loads(manifest_path.read_text())
    require(manifest['source_run_id'] == expected['source_run_id'] and
            manifest['old_pairs_sha256'] == expected['old_pairs_sha256'], 'Source identity mismatch')
    rows = {row['path']: row for row in manifest['files']}
    require(len(rows) == len(manifest['files']), 'Duplicate source archive paths')
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, 'r:gz') as stream:
        members = stream.getmembers()
        require({m.name for m in members} == set(rows) and len(members) == len(rows),
                'Source archive inventory mismatch')
        for member in members:
            require(member.isfile() and not Path(member.name).is_absolute() and
                    '..' not in Path(member.name).parts and member.size == rows[member.name]['size'],
                    'Unsafe or invalid source archive member')
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with stream.extractfile(member) as source, target.open('wb') as output:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    output.write(block)
                    digest.update(block)
            require(digest.hexdigest() == rows[member.name]['sha256'],
                    'Source archive member hash mismatch: ' + member.name)
    (destination / manifest_path.name).write_bytes(manifest_path.read_bytes())
    return verify_snapshot(destination)


def regenerate(inputs: Path, output: Path, tau_root: Path, tokenizer_sha256: str) -> dict:
    from .tau3_workload import DOMAINS, make_pairs, verify_tau
    manifest = verify_snapshot(inputs)
    pinned = verify_tau(tau_root)
    provenance = json.loads((inputs / 'tau3-provenance.json').read_text())
    require(provenance['commit'] == pinned['commit'] and
            provenance['data_sha256'] == pinned['data_sha256'], 'Pinned τ³ source differs from collected input')
    episodes = json.loads((inputs / 'episodes.json').read_text())
    require(all(sum(e['domain'] == d for e in episodes) >= 48 for d in DOMAINS),
            'Source episodes do not cover both domains')
    output.mkdir(parents=True, exist_ok=True)
    for name in ('episodes.json', 'collection-summary.json', 'rejected-episodes.json',
                 'tau3-provenance.json'):
        (output / name).write_bytes((inputs / name).read_bytes())
    result = make_pairs(episodes, output, tokenizer_sha256)
    launch = json.loads((output / 'launch.json').read_text())
    archive = launch['source_archive']
    require(archive['source_run_id'] == manifest['source_run_id'] and
            archive['old_pairs_sha256'] == manifest['old_pairs_sha256'],
            'Launch source archive identity mismatch')
    write_json(output / 'source-lineage.json', {
        'source_run_id': manifest['source_run_id'],
        'source_archive_sha256': archive['sha256'],
        'source_final_manifest_sha256': manifest['source_final_manifest_sha256'],
        'source_input_manifest_sha256': sha256_file(inputs / 'tau3-source-inputs-manifest.json'),
        'old_pairs_sha256': manifest['old_pairs_sha256'],
        'new_pairs_sha256': sha256_file(output / 'pairs.json'),
        'reused': 'original episodes, task logs, and collection metadata',
        'regenerated': ['pairs.json', 'event-audit.json', 'context-feasibility.json']})
    return result


def reuse_sealed_pairs(inputs: Path, output: Path, tau_root: Path,
                       tokenizer_sha256: str) -> dict:
    """Carry the exact archived prompts forward, without calling make_pairs."""
    from .tau3_workload import verify_tau
    manifest = verify_snapshot(inputs)
    pinned = verify_tau(tau_root)
    provenance = json.loads((inputs / 'tau3-provenance.json').read_text())
    require(provenance['commit'] == pinned['commit'] and
            provenance['data_sha256'] == pinned['data_sha256'],
            'Pinned τ³ source differs from archived input')
    pairs = json.loads((inputs / 'old-pairs.json').read_text())
    require(pairs['tokenizer_json_sha256'] == tokenizer_sha256,
            'Archived pair tokenizer hash mismatch')
    output.mkdir(parents=True, exist_ok=True)
    for name in ('episodes.json', 'collection-summary.json', 'rejected-episodes.json',
                 'tau3-provenance.json', 'event-audit.json', 'context-feasibility.json'):
        (output / name).write_bytes((inputs / name).read_bytes())
    (output / 'pairs.json').write_bytes((inputs / 'old-pairs.json').read_bytes())
    require(sha256_file(output / 'pairs.json') == manifest['old_pairs_sha256'],
            'Archived pair bytes changed')
    write_json(output / 'source-lineage.json', {
        'source_run_id': manifest['source_run_id'],
        'source_final_manifest_sha256': manifest['source_final_manifest_sha256'],
        'source_input_manifest_sha256': sha256_file(inputs / 'tau3-source-inputs-manifest.json'),
        'old_pairs_sha256': manifest['old_pairs_sha256'],
        'new_pairs_sha256': sha256_file(output / 'pairs.json'),
        'reused': 'sealed pairs, episode logs, collection metadata, event audit, and context feasibility',
        'regenerated': []})
    return pairs
