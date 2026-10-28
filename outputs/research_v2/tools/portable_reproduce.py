#!/usr/bin/env python3
"""Copy selected completed artifacts to a new directory and replay them locally.

No network, installation, training, raw extraction, target release, symlinks, or
in-place historical-output writes. See ../REPRODUCTION.md for scope and commands.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '2'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
sys.dont_write_bytecode = True
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import shutil
import subprocess
import time
import traceback

V2 = pathlib.Path('outputs/research_v2')
V1 = pathlib.Path('outputs/research')


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(pathlib.Path(path).read_text())


def write(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2)


def safe_relative(value):
    path = pathlib.Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Unsafe manifest-relative path: {value}')
    return path


def historical_relative(value):
    """Map only a declared historical outputs path; never modify its manifest."""
    path = pathlib.Path(value)
    if not path.is_absolute():
        return safe_relative(value)
    positions = [i for i, part in enumerate(path.parts) if part == 'outputs']
    if len(positions) != 1:
        raise ValueError(f'Cannot map historical path: {value}')
    return safe_relative(pathlib.Path(*path.parts[positions[0]:]))


class Snapshot:
    def __init__(self, source, destination):
        self.source = source
        self.destination = destination
        self.files = {}
        self.pending = []
        self.remappings = []

    def copy(self, relative, expected=None, destination=None):
        relative = safe_relative(relative)
        target_rel = safe_relative(destination or relative)
        source = self.source / relative
        if not source.is_file() or not source.resolve().is_relative_to(self.source):
            raise FileNotFoundError(f'Missing or nonlocal required artifact: {relative}')
        digest = sha(source)
        if expected is not None and digest != expected:
            raise ValueError(f'Frozen digest mismatch: {relative}')
        target = self.destination / target_rel
        key = str(target_rel)
        if key in self.files:
            if self.files[key]['sha256'] != digest:
                raise RuntimeError(f'Source changed during snapshot: {relative}')
            return read(target) if target.suffix == '.json' else target
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if digest != sha(target) or digest != sha(source):
            raise RuntimeError(f'Source changed during copy: {relative}')
        self.files[key] = dict(source_relative=str(relative), sha256=digest,
                               bytes=target.stat().st_size,
                               expected_frozen_sha256=expected)
        return read(target) if target.suffix == '.json' else target

    def optional(self, relative, destination=None):
        if (self.source / relative).is_file():
            return self.copy(relative, destination=destination)
        return None

    def calibration(self, dataset, models):
        directory = V2 / 'experiments/calibration' / dataset
        if not (self.source / directory / 'evaluation_complete.json').is_file():
            self.pending.append(dataset + '_calibration')
            return
        complete = self.copy(directory / 'evaluation_complete.json')
        freeze = self.copy(directory / 'freeze.json', complete['selection_freeze_sha256'])
        selected = self.copy(directory / 'selected.json', freeze['files']['selected.json'])
        for record in complete['records']:
            self.copy(directory / safe_relative(record['file']), record['sha256'])
        if not models:
            return
        for record in selected:
            self.copy(directory / safe_relative(record['model']), record['model_sha256'])
        # Selectively verify the declared PSD/code/protocol dependencies. Absolute
        # historical names stay intact; only the audited lookup root changes.
        for original, expected in freeze['inputs'].items():
            relative = historical_relative(original)
            self.copy(relative, expected)
            self.remappings.append(dict(original=original, relocated=str(relative), sha256=expected))

    def diagnostics(self, models):
        directory = V2 / 'experiments/diagnostics'
        manifest = self.copy(directory / 'real_results_manifest.json')
        freeze = self.copy(directory / 'real_selection_freeze.json')
        selected = self.copy(directory / 'real_selection.json', freeze['selection_sha256'])
        self.copy(V2 / 'experiments/estimator_diagnostics.py', freeze['script_sha256'])
        self.copy(V2 / 'research/diagnostic_protocol.md', freeze['protocol_sha256'])
        for record in manifest['records']:
            self.copy(directory / safe_relative(record['file']), record['sha256'])
        synthetic = self.copy(directory / 'synthetic_results.json')
        for record in synthetic['records']:
            self.copy(directory / safe_relative(record['file']), record['sha256'])
        if models:
            for record in selected:
                self.copy(directory / f"{record['split']}_{record['method']}.joblib", record['model_sha256'])
            spatial = self.copy(V2 / 'data/spatial_resolution/manifest.json')
            self.copy(V2 / 'data/spatial_resolution/psd.npz', spatial['file_sha256'])

    def cnn(self, dataset):
        directory = V2 / 'experiments/cnn' / dataset
        if not (self.source / directory / 'predictions/completion.json').is_file():
            self.pending.append(dataset + '_cnn')
            return
        complete = self.copy(directory / 'predictions/completion.json')
        summary = self.copy(directory / 'summary.json', complete['summary_sha256'])
        self.copy(directory / 'predictions/manifest.json', complete['manifest_sha256'])
        freeze = self.copy(directory / 'freeze.json', summary['selection_freeze_sha256'])
        selection_path = directory / 'selection.json'
        self.copy(selection_path, freeze['artifacts_sha256'][str(selection_path)])
        for record in summary['per_seed']:
            self.copy(safe_relative(record['prediction_file']), record['prediction_sha256'])


def environment():
    packages = {}
    for name in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl', 'matplotlib'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                packages=packages, cpu_threads=2, network_used=False,
                clean_environment_install_tested=False)


def replay_svm(project, max_rows):
    """Saved classifiers; recomputed author features; no SVM fitting or CV search."""
    import numpy as np
    import pandas as pd
    import joblib
    from threadpoolctl import threadpool_limits
    sys.path.insert(0, str(project / V2 / 'experiments'))
    import response_calibration as features
    import run_calibration as runner
    records = []

    def predict(model, x, result, directory, y, v, config):
        frame = pd.read_csv(directory / result['file'])
        index = np.flatnonzero(config['test'])
        if not np.array_equal(frame['index'], index):
            raise AssertionError('Prediction row order does not match frozen split')
        if not np.array_equal(frame.texture, y[index]) or not np.array_equal(frame.speed, v[index]):
            raise AssertionError('Prediction labels do not match PSD labels')
        if not np.array_equal(frame.domain, config['domains'][index]):
            raise AssertionError('Prediction domains do not match frozen split')
        if max_rows and len(index) > max_rows:
            use = np.linspace(0, len(index) - 1, max_rows, dtype=int)
            index, frame = index[use], frame.iloc[use]
        started = time.monotonic()
        predicted = model.predict(x[index])
        mismatch = int(np.count_nonzero(predicted != frame.prediction.to_numpy()))
        rec = dict(split=result['split'], method=result['method'], rows=len(index),
                   full_target_rows=result['rows'], mismatches=mismatch,
                   seconds=time.monotonic() - started)
        records.append(rec)
        print(json.dumps(rec), flush=True)
        if mismatch:
            raise AssertionError(f'Saved-model prediction differs: {rec}')

    with threadpool_limits(limits=2):
        cluster_data = None
        for dataset in ('cluster', 'external'):
            directory = project / V2 / 'experiments/calibration' / dataset
            if not (directory / 'evaluation_complete.json').exists():
                continue
            if dataset == 'cluster':
                data, feature_function = runner.load_cluster(), features.features_for_split
                cluster_data = data
            else:
                import run_external_calibration as external
                data, feature_function = external.load_external(), external.external_features
            p, f, y, v, labels, configs, settings, data_path = data
            selected = read(directory / 'selected.json')
            results = read(directory / 'evaluation_complete.json')['records']
            for result in results:
                config = next(c for c in configs if c['name'] == result['split'])
                choice = next(r for r in selected if (r['split'], r['method']) == (result['split'], result['method']))
                x, diagnostic = feature_function(p, f, v, y, config['train'], settings, result['method'])
                if diagnostic != choice['source_fit_diagnostics']:
                    raise AssertionError('Recomputed source feature parameters differ')
                model = joblib.load(directory / choice['model'])
                predict(model, x, result, directory, y, v, config)
                records[-1]['dataset'] = dataset
        if cluster_data is not None:
            p, f, y, v, labels, configs, settings, data_path = cluster_data
            directory = project / V2 / 'experiments/diagnostics'
            spatial = np.load(project / V2 / 'data/spatial_resolution/psd.npz')
            if not np.array_equal(spatial['labels'], labels):
                raise AssertionError('Fixed-space PSD row order differs')
            edge = np.geomspace(40, 800, 129)
            matrices = {}
            for mode in ('temporal', 'spatial'):
                temporal_edge = edge[None, :] * (v[:, None] / 40 if mode == 'spatial' else np.ones((len(v), 1)))
                queries = {'common_band_' + mode: (p, f, np.clip(temporal_edge, 60, 400)),
                           'fixed_space_' + mode: (spatial['psd'], spatial['frequency'], edge[None, :] / (40 if mode == 'spatial' else v[:, None]))}
                for name, (power, frequency, bins) in queries.items():
                    mass = np.maximum(np.diff(features.integral(power, frequency, bins), axis=1), 0)
                    if not (mass.sum(1) > 0).all():
                        raise AssertionError('Zero-mass diagnostic feature')
                    matrices[name] = np.log(np.maximum(mass / mass.sum(1, keepdims=True), 1e-12))
            for result in read(directory / 'real_results_manifest.json')['records']:
                config = next(c for c in configs if c['name'] == result['split'])
                model = joblib.load(directory / f"{result['split']}_{result['method']}.joblib")
                predict(model, matrices[result['method']], result, directory, y, v, config)
                records[-1]['dataset'] = 'cluster'
    write(project / 'reproduction/svm_replay.json', dict(timestamp_utc=utc(), status='passed',
          scope='author feature reconstruction and saved SVM inference; no independent estimator derivation or refit',
          sample_policy='all target rows' if max_rows == 0 else 'evenly spaced deterministic row positions',
          records=records, compared_rows=sum(r['rows'] for r in records),
          mismatches=sum(r['mismatches'] for r in records)))


def compare_tables(project):
    import pandas as pd
    results = []
    for name in ('scores.csv', 'contrasts.csv', 'specimen_recalls.csv', 'source_cv_folds.csv',
                 'prediction_costs.csv', 'synthetic_diagnostics.csv'):
        actual_path = project / V2 / 'analysis' / name
        expected_path = project / 'reproduction/reference' / name
        if not expected_path.exists():
            results.append(dict(file=name, status='reference_missing'))
            continue
        actual, expected = pd.read_csv(actual_path), pd.read_csv(expected_path)
        if 'dataset' in actual and 'dataset' in expected:
            expected = expected[expected.dataset.isin(actual.dataset.unique())]
        try:
            pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True),
                                          check_dtype=False, check_exact=False, atol=1e-12, rtol=0)
            results.append(dict(file=name, status='passed', rows=len(actual),
                                byte_identical=sha(actual_path) == sha(expected_path), atol=1e-12))
        except AssertionError as error:
            results.append(dict(file=name, status='failed', rows=len(actual), error=str(error)[:3000]))
    write(project / 'reproduction/table_comparison.json', results)
    if any(r['status'] != 'passed' for r in results):
        raise AssertionError('Table comparison did not pass; inspect table_comparison.json')


def check_artifacts(project):
    """Check that the generator's numeric macros exactly match its lineage map."""
    root = project / V2
    lineage = read(root / 'analysis/numeric_lineage.json')
    expected = {'\\newcommand{\\' + key + '}{' + value['printed'] + '}'
                for key, value in lineage.items()}
    actual = set((root / 'paper/numbers.tex').read_text().splitlines())
    if actual != expected:
        raise AssertionError('Generated numeric macros differ from numeric lineage')
    manifest = read(root / 'analysis/figure_manifest.json')
    if not manifest or any(sha(root / safe_relative(path)) != digest for path, digest in manifest.items()):
        raise AssertionError('Generated figure manifest mismatch')
    write(project / 'reproduction/artifact_checks.json', dict(timestamp_utc=utc(), status='passed',
          numeric_macros=len(lineage), figure_files=len(manifest),
          macro_lineage_sha256=sha(root / 'analysis/numeric_lineage.json'),
          numbers_tex_sha256=sha(root / 'paper/numbers.tex'),
          scope='generated macro/lineage and figure-file hashes; no PDF visual QA or LaTeX compilation'))


def relocate_handoff(original, project):
    """Rebind paths only when every already-staged artifact is byte-identical.

    This does not copy large arrays, relax QC, change file hashes, or authorize
    targets. It supplies the historical configure_external.py absolute-path API
    in a new tree for source refits from the same sealed processed inputs.
    """
    original, project = original.resolve(), project.resolve()
    destination = project / V2 / 'data/external/handoff.json'
    if destination.exists() or destination == original:
        raise FileExistsError('Relocated handoff destination must not exist')
    document = read(original)
    original_digest = sha(original)
    if document.get('status') != 'sealed' or not document.get('checks_passed') or not document.get('files'):
        raise ValueError('Handoff must be sealed and have passed checks')
    remappings = []
    for key, record in document['files'].items():
        relative = historical_relative(record['path'])
        target = project / relative
        if not target.is_file() or not target.resolve().is_relative_to(project):
            raise FileNotFoundError(f'Stage the required artifact inside the new tree: {relative}')
        if sha(target) != record['sha256']:
            raise ValueError(f'Artifact bytes changed; path-only relocation cannot approve new data/QC: {relative}')
        remappings.append(dict(artifact=key, original_path=record['path'], relocated_path=str(target),
                               sha256=record['sha256']))
        record['path'] = str(target)
    if sha(original) != original_digest:
        raise RuntimeError('Original handoff changed during relocation')
    # Each field except the pathname remains exactly as supplied by the verified
    # handoff. A separate receipt carries relocation provenance.
    write(destination, document)
    receipt = project / 'reproduction' / ('handoff_relocation_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f') + '.json')
    write(receipt, dict(timestamp_utc=utc(), status='passed', original_handoff=str(original),
          original_handoff_sha256=original_digest, relocated_handoff=str(destination),
          relocated_handoff_sha256=sha(destination), path_remappings=remappings,
          wrapper_sha256=sha(__file__), model_training_started=False, target_prediction_started=False,
          scope='path-only relocation of already staged identical artifacts for a new source-fit run; no new QC decision'))
    print(json.dumps(dict(status='passed', artifacts=len(remappings), handoff=str(destination), receipt=str(receipt)), indent=2))


def main(args):
    started = time.monotonic()
    source = args.source_root.resolve()
    destination = args.destination.resolve()
    if destination.exists():
        raise FileExistsError('Destination must not exist; every run keeps its own audit trail')
    if destination == source or any(destination.is_relative_to(source / p) for p in (V1, V2)):
        raise ValueError('Destination cannot be inside either historical research tree')
    destination.mkdir(parents=True)
    snap = Snapshot(source, destination)
    report = dict(started_utc=utc(), source_root=str(source), destination=str(destination),
                  command=sys.argv, level=args.level, dataset=args.dataset,
                  environment=environment(), commands=[], status='running',
                  pending=[], adaptations=['Original relative output tree retained.',
                    'Historical absolute inputs are verified through explicit suffix remapping; their JSON bytes are unchanged.',
                    'Analysis writes to a new unfrozen directory. Historical analysis freeze is not copied.',
                    'CNN tables use completed predictions only; CNN tensor/checkpoint replay is excluded.'])
    try:
        models = args.level == 'svm'
        snap.copy(V2 / 'analysis/analyze_revision.py')
        snap.copy(V2 / 'analysis/make_artifacts.py')
        snap.copy(V2 / 'tools/portable_reproduce.py')
        for name in ('scores.csv', 'contrasts.csv', 'posthoc_grid_contrasts.csv'):
            snap.copy(V1 / 'analysis' / name)
        snap.copy(V2 / 'analysis/alignment_summary.csv')
        snap.copy(V2 / 'research/external_validity_decision.json')
        snap.optional(V2 / 'analysis/alignment_manifest.json')
        snap.optional(V2 / 'research/external_validity_decision.freeze.json')
        for relative in ('environment/requirements.txt', 'environment/runtime.json',
                         'experiments/response_calibration.py', 'experiments/run_calibration.py',
                         'experiments/run_external_calibration.py'):
            snap.optional(V2 / relative)
        for name in ('report.json', 'scores.csv', 'contrasts.csv', 'specimen_recalls.csv',
                     'source_cv_folds.csv', 'prediction_costs.csv', 'synthetic_diagnostics.csv'):
            snap.optional(V2 / 'analysis' / name, pathlib.Path('reproduction/reference') / name)
        datasets = ('cluster',) if args.dataset == 'cluster' else ('cluster', 'external')
        for dataset in datasets:
            snap.calibration(dataset, models)
            snap.cnn(dataset)
        snap.diagnostics(models)
        report['pending'] = snap.pending
        if args.require_complete and (args.dataset != 'all' or snap.pending):
            raise RuntimeError(f'Complete reproduction requires both datasets and all completion manifests: {snap.pending}')
        write(destination / 'reproduction/snapshot.json', dict(timestamp_utc=utc(), files=snap.files,
              path_remappings=snap.remappings, pending=snap.pending))
        operations = [[sys.executable, '-B', str(destination / V2 / 'analysis/analyze_revision.py')]]
        if args.require_complete:
            operations[0].append('--require-complete')
        operations.append([sys.executable, '-B', str(destination / V2 / 'tools/portable_reproduce.py'),
                           '--worker', 'compare', '--project-root', str(destination)])
        artifact_command = [sys.executable, '-B', str(destination / V2 / 'analysis/make_artifacts.py')]
        if args.require_complete:
            artifact_command.append('--require-complete')
        operations.append(artifact_command)
        operations.append([sys.executable, '-B', str(destination / V2 / 'tools/portable_reproduce.py'),
                           '--worker', 'artifacts', '--project-root', str(destination)])
        if models:
            operations.append([sys.executable, '-B', str(destination / V2 / 'tools/portable_reproduce.py'),
                               '--worker', 'svm', '--project-root', str(destination), '--max-rows', str(args.max_rows)])
        for i, command in enumerate(operations):
            log = destination / f'reproduction/command_{i + 1}.log'
            start = time.monotonic()
            record = dict(argv=command, cwd=str(destination), started_utc=utc(), log=str(log.relative_to(destination)))
            report['commands'].append(record)
            with log.open('x') as handle:
                try:
                    completed = subprocess.run(command, cwd=destination, stdout=handle, stderr=subprocess.STDOUT,
                                               timeout=max(1, args.timeout_seconds - (time.monotonic() - started)),
                                               env={**os.environ, 'PYTHONPATH': '', 'PYTHONDONTWRITEBYTECODE': '1'})
                    record['exit_code'] = completed.returncode
                except subprocess.TimeoutExpired:
                    record['exit_code'] = 124
                    raise
                finally:
                    record.update(ended_utc=utc(), seconds=time.monotonic() - start, log_sha256=sha(log))
            if record['exit_code']:
                raise RuntimeError(f'Child command failed ({record["exit_code"]}); inspect {log}')
        analysis = read(destination / V2 / 'analysis/report.json')
        report['analysis_pending'] = analysis['pending']
        report['prediction_models'] = analysis['prediction_models']
        report['source_inputs_unchanged'] = all(sha(source / rec['source_relative']) == rec['sha256'] for rec in snap.files.values())
        if not report['source_inputs_unchanged']:
            raise RuntimeError('A selected source input changed after snapshot; retry only after writer completes')
        report['status'] = 'passed_partial' if analysis['pending'] else 'passed'
    except Exception:
        report.update(status='failed', exception=traceback.format_exc(), pending=snap.pending)
    finally:
        report.update(ended_utc=utc(), elapsed_seconds=time.monotonic() - started,
                      copied_files=len(snap.files), copied_bytes=sum(r['bytes'] for r in snap.files.values()),
                      selected_input_hashes=snap.files, path_remappings=snap.remappings)
        report['artifacts'] = {str(path.relative_to(destination)): sha(path)
                               for directory in (destination / 'reproduction', destination / V2 / 'analysis',
                                                 destination / V2 / 'paper', destination / V2 / 'figures')
                               if directory.exists() for path in directory.rglob('*') if path.is_file()}
        write(destination / 'reproduction/run.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'pending', 'copied_files', 'copied_bytes', 'elapsed_seconds')}, indent=2))
    print(str(destination / 'reproduction/run.json'))
    return 1 if report['status'] == 'failed' else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[3])
    parser.add_argument('--destination', type=pathlib.Path)
    parser.add_argument('--level', choices=('tables', 'svm'), default='tables', help='svm also rebuilds tables')
    parser.add_argument('--dataset', choices=('cluster', 'all'), default='all')
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--max-rows', type=int, default=0, help='SVM target rows per model; 0 means all')
    parser.add_argument('--timeout-seconds', type=int, default=840)
    parser.add_argument('--relocate-handoff', type=pathlib.Path,
                        help='Write a new path-only handoff after verifying all staged artifacts match its original digests')
    parser.add_argument('--worker', choices=('svm', 'compare', 'artifacts'), help=argparse.SUPPRESS)
    parser.add_argument('--project-root', type=pathlib.Path, help='New tree root for handoff relocation')
    args = parser.parse_args()
    if args.max_rows < 0 or args.timeout_seconds <= 0:
        parser.error('Row cap must be nonnegative and timeout must be positive')
    if args.relocate_handoff:
        if args.project_root is None or args.worker or args.destination:
            parser.error('Handoff relocation requires --project-root and cannot combine with --destination/--worker')
        relocate_handoff(args.relocate_handoff, args.project_root)
    elif args.worker:
        if args.project_root is None:
            parser.error('Internal worker requires project root')
        if args.worker == 'svm':
            replay_svm(args.project_root.resolve(), args.max_rows)
        elif args.worker == 'compare':
            compare_tables(args.project_root.resolve())
        else:
            check_artifacts(args.project_root.resolve())
    else:
        if args.destination is None:
            parser.error('--destination is required and must not already exist')
        sys.exit(main(args))
