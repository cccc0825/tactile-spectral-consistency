#!/usr/bin/env python3
"""Prepare a fresh full-source study tree without importing historical results."""
from __future__ import annotations
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
V2 = Path('outputs/research_v2')
REFERENCE_ONLY = {
    'outputs/research_v2/analysis/report.json',
    'outputs/research_v2/experiments/calibration/external/external40_corrected_predictions.csv',
    'outputs/research_v2/experiments/calibration/external/external40_spatial_predictions.csv',
}

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def require(condition, message):
    if not condition:
        raise ValueError(message)

def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')

def safe_path(root, name):
    path = Path(name)
    require(not path.is_absolute() and '..' not in path.parts, 'Unsafe manifest path')
    result = root / path
    require(result.resolve().is_relative_to(root.resolve()), 'Path leaves study root')
    return result

def destination_path(name):
    # Preserve published reference values without seeding any new target gate.
    reference = name in REFERENCE_ONLY or name.startswith((
        'outputs/research_v2/revision_20260914/',
        'outputs/research_v2/consistency_20260915/',
    ))
    return Path('reference_release') / name if reference else Path(name)

def verify_study(root):
    root = root.resolve()
    setup = read(root / 'STUDY_SETUP.json')
    require(setup['kind'] == 'new_full_source_study_tree', 'Not a prepared full study tree')
    require(Path(setup['source_root']).resolve() != root, 'Fresh tree must differ from release')
    for name, digest in setup['files_sha256'].items():
        require(sha(safe_path(root, name)) == digest, 'Prepared source changed: ' + name)
    return setup

def prepare(source, root):
    source, root = source.resolve(), root.resolve()
    require(not root.exists(), 'Destination must not exist; preserve earlier runs')
    require(not root.is_relative_to(source), 'Destination must be outside the release')
    manifest_path = source / 'release_metadata/release_manifest.json'
    published = read(manifest_path)['files']
    required = ['scripts/prepare_study.py', 'scripts/accept_source_audits.py',
                'docs/FULL_STUDY.md',
                'outputs/research_v2/audits/replay/audit_external_data.py',
                'outputs/research_v2/audits/replay/audit_external_svm_source.py']
    require(all(name in published for name in required), 'Release manifest must include all study tools')
    for name, record in published.items():
        path = safe_path(source, name)
        require(path.is_file() and sha(path) == record['sha256'], 'Release hash mismatch: ' + name)
    # The unchanged helper supplies its own 36-file, source-only manifest.
    subprocess.run([sys.executable, str(source / 'scripts/bootstrap_public.py')], check=True, cwd=source)
    subprocess.run([sys.executable, str(source / V2 / 'tools/prepare_fresh_run.py'),
                    'prepare', '--source-root', str(source), '--project-root', str(root)], check=True, cwd=source)
    copied = {}
    relocated = []
    for name, record in published.items():
        relative = destination_path(name)
        target = safe_path(root, relative)
        if target.exists():
            require(sha(target) == record['sha256'], 'Prepared file conflicts with release: ' + name)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target)
        require(sha(target) == record['sha256'], 'Copied file hash mismatch: ' + name)
        copied[str(relative)] = record['sha256']
        if str(relative) != name:
            relocated.append({'published': name, 'reference_path': str(relative)})
    # release_manifest excludes its own digest; preserve it as provenance.
    target_manifest = root / 'release_metadata/release_manifest.json'
    if not target_manifest.exists():
        shutil.copyfile(manifest_path, target_manifest)
    copied['release_metadata/release_manifest.json'] = sha(manifest_path)
    subprocess.run([sys.executable, str(root / 'scripts/bootstrap_public.py')], check=True, cwd=root)
    write_new(root / 'STUDY_SETUP.json', {
        'kind': 'new_full_source_study_tree', 'created_utc': now(), 'source_root': str(source),
        'files_sha256': copied, 'reference_only_relocations': relocated,
        'source_subset_manifest': 'FRESH_RUN.json', 'training_executed': False,
        'target_prediction_executed': False, 'historical_acceptance_copied': False,
        'scope': 'Source/metadata preparation only; published outcomes relocated as reference, never accepted as new outputs.',
    })
    print(json.dumps({'status': 'prepared', 'root': str(root), 'files': len(copied),
                      'reference_only_files': len(relocated)}, indent=2))

def augment_handoff(root):
    """Add independently recomputable class counts required by the old audit."""
    verify_study(root)
    cnn = root / V2 / 'experiments/cnn'
    require(not (cnn / 'external_config.json').exists(), 'Must precede external CNN configuration')
    require(not (cnn / 'external/freeze.json').exists(), 'Must precede external CNN selection')
    path = cnn / 'external_split_preflight.json'
    backup = cnn / 'external_split_preflight.source_subset.json'
    receipt = cnn / 'full_study_handoff_extension.json'
    require(not backup.exists() and not receipt.exists(), 'Handoff extension already exists')
    with (root / V2 / 'data/external/processed/metadata.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    source = [r for r in rows if int(r['speed']) in (30, 35)]
    require(len(source) == 2354, 'Unexpected source count')
    value = read(path)
    for held in range(3):
        tr = [r for r in source if int(r['trial']) % 3 != held]
        va = [r for r in source if int(r['trial']) % 3 == held]
        saved = value['folds'][str(held)]
        for prefix, subset in [('training', tr), ('validation', va)]:
            digest = hashlib.sha256('\n'.join(r['id'] for r in subset).encode()).hexdigest()
            require(saved[prefix + '_ids_sha256'] == digest and saved[prefix + '_n'] == len(subset),
                    'Existing preflight IDs/counts disagree with metadata')
            counts = {str(c): sum(int(r['class_id']) == c for r in subset) for c in range(12)}
            require(all(n > 0 for n in counts.values()), 'Missing source class')
            saved[prefix + '_by_class'] = counts
    previous = sha(path)
    shutil.copyfile(path, backup)
    temporary = path.with_suffix('.extended.json')
    write_new(temporary, value)
    temporary.replace(path)
    write_new(receipt, {'created_utc': now(), 'status': 'source_metadata_extended',
                       'previous_sha256': previous, 'extended_sha256': sha(path),
                       'metadata_sha256': sha(root / V2 / 'data/external/processed/metadata.csv'),
                       'class_counts_from_source_rows_only': True,
                       'classification_outcomes_read': False,
                       'scope': 'Add per-class counts before CNN configuration; original IDs/folds remain unchanged.'})
    print(json.dumps({'status': 'handoff_extended', 'receipt': str(receipt)}))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'check', 'augment-handoff'])
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.project_root.resolve()
    if args.mode == 'prepare':
        prepare(args.source_root, root)
    elif args.mode == 'augment-handoff':
        augment_handoff(root)
    else:
        setup = verify_study(root)
        print(json.dumps({'status': 'source_files_verified', 'files': len(setup['files_sha256']),
                          'data_or_training_verified': False}))

if __name__ == '__main__':
    main()
