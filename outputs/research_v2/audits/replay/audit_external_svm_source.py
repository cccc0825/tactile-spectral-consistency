"""Read-only external SVM source-selection replay; never infer target rows.

Uses the existing environment and author deterministic feature functions on an
explicitly source-only array. Metrics, folds, model refits and selected feature
spot calculations are independently implemented below. Writes only this audit
directory, and has an eight-minute execution deadline.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '2'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
sys.dont_write_bytecode = True
import datetime
import hashlib
import itertools
import json
import pathlib
import time
from fractions import Fraction
import joblib
import numpy as np
import scipy
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits, threadpool_info

ROOT = pathlib.Path(__file__).resolve().parents[4]
V2 = ROOT / 'outputs/research_v2'
OUT = V2 / 'audits/replay'
RUN = V2 / 'experiments/calibration/external'
sys.path.insert(0, str(V2 / 'experiments'))
from run_external_calibration import external_features

COMMAND = ('PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 '
           'MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 work/research/venv/bin/python '
           'outputs/research_v2/audits/replay/audit_external_svm_source.py')
METHODS = ['temporal', 'spatial', 'corrected', 'pooled_whitening', 'mel', 'mfcc',
           'corrected_weak', 'corrected_strong', 'wrong_order']

def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(pathlib.Path(path).read_text())

def bitwise(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()

def array_hash(a):
    a = np.asarray(a)
    return hashlib.sha256(str(a.dtype).encode() + repr(a.shape).encode() + a.tobytes()).hexdigest()

def macro_recall(y, prediction):
    """Direct twelve-class correct/denominator fractions, no metric library."""
    counts = []
    for c in range(12):
        mask = y == c
        counts.append((int(np.sum(prediction[mask] == c)), int(mask.sum())))
    assert all(total > 0 for _, total in counts)
    exact = sum((Fraction(correct, total) for correct, total in counts), Fraction()) / 12
    floating = float(np.mean([correct / total for correct, total in counts]))
    return floating, exact, counts

def independent_log_bands(psd, frequency, speed, spatial):
    """Direct integration at every knot, independent of cumulative-integral code."""
    edges = np.geomspace(5., 100., 129) * (speed / 35. if spatial else 1.)
    masses = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        knots = np.r_[lo, frequency[(frequency > lo) & (frequency < hi)], hi]
        values = np.interp(knots, frequency, psd)
        masses.append(float(np.sum(np.diff(knots) * (values[:-1] + values[1:]) / 2.)))
    masses = np.maximum(masses, 0.)
    return np.log(np.maximum(masses / masses.sum(), 1e-12))

def independent_mel(psd, frequency):
    low, high = 2595. * np.log10(1. + np.array([5., 100.]) / 700.)
    edges = 700. * (10. ** ((low + np.arange(34) * (high - low) / 33.) / 2595.) - 1.)
    energies = []
    for i in range(32):
        total = 0.
        for power, freq in zip(psd, frequency):
            if edges[i] <= freq <= edges[i + 1]:
                total += power * (freq - edges[i]) / (edges[i + 1] - edges[i])
            elif edges[i + 1] < freq <= edges[i + 2]:
                total += power * (edges[i + 2] - freq) / (edges[i + 2] - edges[i + 1])
        energies.append(total * (frequency[1] - frequency[0]))
    energies = np.asarray(energies)
    logs = np.log(np.maximum(energies / energies.sum(), 1e-12))
    cosines = np.cos(np.pi / 32. * (np.arange(32) + .5)[None, :] * np.arange(1, 21)[:, None])
    cepstra = np.sqrt(2. / 32.) * (cosines @ logs)
    return logs, cepstra

def main():
    start = time.monotonic()
    checks, hash_checks, fold_records, cv_records, model_records = [], [], [], [], []
    feature_spots, invariance = [], []
    calls = {'source_feature_calls': 0, 'source_cv_fits': 0, 'source_cv_predictions': 0,
             'source_final_fits': 0, 'external_target_inferences': 0,
             'external_target_outcome_files_read': 0}

    def check(name, passed, **detail):
        checks.append({'name': name, 'passed': bool(passed), **detail})

    def deadline():
        if time.monotonic() - start > 480:
            raise TimeoutError('Eight-minute source replay execution cap')

    freeze_path = RUN / 'freeze.json'
    freeze = read(freeze_path)
    observed = {str(freeze_path): sha(freeze_path)}
    check('freeze_stage', freeze['stage'] == 'before extension target prediction')
    check('nine_method_order', freeze['method_order'] == METHODS)
    for group, base in (('files', RUN), ('inputs', None)):
        for filename, expected in freeze[group].items():
            # The sealed manifest enumerates source artifacts only. Do not open
            # any evaluation artifact should an unexpected one appear here.
            assert 'prediction' not in pathlib.Path(filename).name and 'evaluation' not in pathlib.Path(filename).name
            path = base / filename if base is not None else pathlib.Path(filename)
            actual = sha(path)
            observed[str(path)] = actual
            row = {'group': group, 'path': str(path), 'expected_sha256': expected,
                   'actual_sha256': actual, 'passed': actual == expected}
            hash_checks.append(row)
            check('freeze_hash:' + filename, row['passed'])
    assert all(a['passed'] for a in hash_checks), 'Do not execute changed frozen code or models'
    check('input_hash_manifest_exact', read(RUN / 'input_hashes.json') == freeze['inputs'])
    selected = read(RUN / 'selected.json')
    check('selected_nine_unique_methods', [r['method'] for r in selected] == METHODS)
    settings = read(V2 / 'research/external_settings.json')['feature_settings']
    check('external_feature_settings', settings == {'lo': 5, 'hi': 100, 'ref': 35, 'support': [4, 120], 'mel_hi': 100})

    # The archive contains both partitions, so its whole-file hash and label
    # metadata are checked; target spectra are discarded immediately at slicing.
    # Only the source-only p/y/v below can reach feature/fit/predict functions.
    with np.load(V2 / 'data/external/processed/psd.npz') as archive:
        labels = archive['labels']
        source_mask = np.isin(labels[:, 1], [30, 35])
        source_indices = np.flatnonzero(source_mask)
        target_indices = np.flatnonzero(labels[:, 1] == 40)
        p = archive['psd'][source_indices].copy()
        frequency = archive['frequency'].copy()
        source_labels = labels[source_indices].copy()
    y, v, trial = source_labels.T
    assert len(p) == 2354 and set(v) == {30, 35}
    check('partition_counts', len(labels) == 3550 and len(source_indices) == 2354 and len(target_indices) == 1196)
    check('all_episode_identifiers_unique', len(set(map(tuple, labels))) == 3550)
    check('partition_cover_disjoint', len(np.intersect1d(source_indices, target_indices)) == 0 and
          np.array_equal(np.sort(np.r_[source_indices, target_indices]), np.arange(3550)))
    check('source_psd_finite_positive_mass', np.isfinite(p).all() and (p.sum(1) > 0).all())
    split = read(RUN / 'external40_split.json')
    check('published_train_indices', np.array_equal(split['train_indices'], source_indices))
    check('published_test_indices_metadata_only', np.array_equal(split['test_indices'], target_indices))
    check('five_folds', len(split['folds']) == 5)
    validation_counts = np.zeros(len(p), dtype=int)
    folds = []
    for k, record in enumerate(split['folds']):
        tr, va = trial % 5 != k, trial % 5 == k
        validation_counts += va
        tests = {'name': record['name'] == f'trial_mod5_{k}',
                 'train_indices': np.array_equal(record['train_indices'], source_indices[tr]),
                 'validation_indices': np.array_equal(record['validation_indices'], source_indices[va]),
                 'disjoint': not np.any(tr & va), 'cover_source': np.all(tr | va),
                 'class_coverage': set(y[tr]) == set(y[va]) == set(range(12)),
                 'class_speed_coverage': all(np.any(tr & (y == c) & (v == s)) and np.any(va & (y == c) & (v == s))
                                             for c in range(12) for s in (30, 35)),
                 'no_target': set(v[tr]) == set(v[va]) == {30, 35}}
        fold_records.append({'fold': k, 'train_rows': int(tr.sum()), 'validation_rows': int(va.sum()),
                             'validation_class_counts': [int(np.sum(va & (y == c))) for c in range(12)],
                             'checks': {name: bool(passed) for name, passed in tests.items()}})
        for key, passed in tests.items():
            check(f'fold_{k}:{key}', passed)
        folds.append((tr, va))
    check('source_each_validation_once', np.all(validation_counts == 1))

    def features(method, train, psd=None, classes=None):
        deadline()
        psd = p if psd is None else psd
        classes = y if classes is None else classes
        assert len(psd) == len(y) == 2354 and set(v) == {30, 35}
        assert train.shape == (2354,) and train.dtype == bool
        calls['source_feature_calls'] += 1
        return external_features(psd, frequency, v, classes, train, settings, method)

    # One fixed retained episode per source class x speed, 24 distinct rows.
    spots = np.array([np.flatnonzero((y == c) & (v == speed))[0]
                      for c in range(12) for speed in (30, 35)])
    independent = {'temporal': [], 'spatial': [], 'mel': [], 'mfcc': []}
    for index in spots:
        independent['temporal'].append(independent_log_bands(p[index], frequency, v[index], False))
        independent['spatial'].append(independent_log_bands(p[index], frequency, v[index], True))
        mel, mfcc = independent_mel(p[index], frequency)
        independent['mel'].append(mel)
        independent['mfcc'].append(mfcc)

    for record in selected:
        deadline()
        method = record['method']
        check(method + ':standalone_record', read(RUN / f'external40_{method}_selection.json') == record)
        check(method + ':source_metadata', record['split'] == 'external40' and record['source_acquisitions'] == 2354 and record['settings'] == settings)
        check(method + ':model_hash_in_record', sha(RUN / record['model']) == record['model_sha256'])
        check(method + ':candidate_grid', [(r['C'], r['g']) for r in record['candidates']] == list(itertools.product((1, 10, 100), (.25, 1, 4))))
        fold_data = []
        for k, (tr, va) in enumerate(folds):
            x, info = features(method, tr)
            check(f'{method}:fold_{k}:fit_diagnostics', {'fold': f'trial_mod5_{k}', **info} == record['fold_diagnostics'][k])
            fold_data.append((x[tr], x[va], y[tr], y[va]))
            if method in ('corrected', 'corrected_weak', 'corrected_strong', 'wrong_order', 'pooled_whitening'):
                altered_p, altered_y = p.copy(), y.copy()
                altered_p[va, :40] *= 2.
                altered_y[va] = 9999
                changed_x, changed_info = features(method, tr, altered_p, altered_y)
                passed = info == changed_info and bitwise(x[tr], changed_x[tr])
                invariance.append({'method': method, 'fold': k, 'passed': passed,
                                   'perturbation': 'Held-out SOURCE-fold labels set to 9999 and first 40 PSD bins doubled; no external target row used.'})
                check(f'{method}:fold_{k}:source_validation_invariance', passed)

        candidates = []
        for stored in record['candidates']:
            C, g = stored['C'], stored['g']
            fold_scores, exact_scores, counts = [], [], []
            for k, (xt, xv, yt, yv) in enumerate(fold_data):
                deadline()
                replica = make_pipeline(StandardScaler(), SVC(C=C, gamma=g / xt.shape[1], cache_size=200))
                replica.fit(xt, yt)
                calls['source_cv_fits'] += 1
                prediction = replica.predict(xv)
                calls['source_cv_predictions'] += 1
                score, exact, cell_counts = macro_recall(yv, prediction)
                fold_scores.append(score)
                exact_scores.append(exact)
                counts.append(cell_counts)
                check(f'{method}:C{C}:g{g}:fold_{k}:score_exact', score == stored['fold_scores'][k],
                      observed=score, expected=stored['fold_scores'][k])
            candidate = {'C': C, 'g': g, 'cv_mean': float(np.mean(fold_scores)), 'fold_scores': fold_scores,
                         'exact_mean': sum(exact_scores, Fraction()) / 5, 'counts': counts}
            candidates.append(candidate)
            check(f'{method}:C{C}:g{g}:macro_fold_mean', candidate['cv_mean'] == stored['cv_mean'])
            cv_records.append({'method': method, 'C': C, 'g': g, 'cv_mean': candidate['cv_mean'],
                               'fold_scores': fold_scores, 'exact_rational_mean': str(candidate['exact_mean']),
                               'fold_class_correct_total': counts})
        # Equal-weight fold means; lower C, then gamma multiplier nearer 1, then
        # lower multiplier break ties. Independent rational check also included.
        choice = min(candidates, key=lambda a: (-a['cv_mean'], a['C'], abs(np.log(a['g'])), a['g']))
        exact_choice = min(candidates, key=lambda a: (-a['exact_mean'], a['C'], abs(np.log(a['g'])), a['g']))
        check(method + ':selection_recomputed', {k: choice[k] for k in ('C', 'g', 'cv_mean', 'fold_scores')} == record['selected'])
        check(method + ':exact_rational_selection', (exact_choice['C'], exact_choice['g']) == (record['selected']['C'], record['selected']['g']))

        x, info = features(method, np.ones(len(y), dtype=bool))
        check(method + ':final_fit_diagnostics', info == record['source_fit_diagnostics'])
        check(method + ':feature_dimension', x.shape == (2354, record['feature_dimension']))
        if method in independent:
            delta = float(np.max(np.abs(np.asarray(independent[method]) - x[spots])))
            feature_spots.append({'method': method, 'rows': source_indices[spots].tolist(), 'maximum_absolute_error': delta,
                                  'atol': 2e-9, 'passed': delta < 2e-9})
            check(method + ':independent_feature_math_spots', delta < 2e-9, maximum_absolute_error=delta)
        final = make_pipeline(StandardScaler(), SVC(C=choice['C'], gamma=choice['g'] / x.shape[1], cache_size=200))
        final.fit(x, y)
        calls['source_final_fits'] += 1
        saved = joblib.load(RUN / record['model'])
        parameters, parameter_hashes = {}, {}
        for step, names in (
            ('standardscaler', ['mean_', 'var_', 'scale_', 'n_samples_seen_', 'n_features_in_']),
            ('svc', ['classes_', 'support_', 'support_vectors_', 'dual_coef_', 'intercept_', 'n_iter_',
                     'n_support_', 'n_features_in_', 'fit_status_', 'shape_fit_', '_dual_coef_', '_intercept_', '_probA', '_probB'])):
            original, replica = saved.named_steps[step], final.named_steps[step]
            for name in names:
                a, b = getattr(original, name), getattr(replica, name)
                parameters[f'{step}.{name}'] = bitwise(a, b)
                parameter_hashes[f'{step}.{name}'] = {'saved': array_hash(a), 'replayed': array_hash(b)}
            parameters[step + '.hyperparameters'] = original.get_params() == replica.get_params()
        transformed = saved.named_steps['standardscaler'].transform(x)
        svc = saved.named_steps['svc']
        parameters['support_vectors_from_declared_source'] = bitwise(svc.support_vectors_, transformed[svc.support_])
        for name, passed in parameters.items():
            check(f'{method}:final_model:{name}', passed)
        model_records.append({'method': method, 'selected': record['selected'], 'all_parameters_bitwise_equal': all(parameters.values()),
                              'checks': parameters, 'parameter_sha256': parameter_hashes})
        print(json.dumps({'method': method, 'source_cv_fits': calls['source_cv_fits'],
                          'final_model_bitwise_equal': all(parameters.values()),
                          'failed_checks_so_far': sum(not c['passed'] for c in checks),
                          'elapsed_seconds': time.monotonic() - start}), flush=True)

    for path, before in observed.items():
        check('unchanged_after:' + path, sha(path) == before)
    check('all_405_source_cv_fits', calls['source_cv_fits'] == calls['source_cv_predictions'] == 405)
    check('nine_source_final_fits', calls['source_final_fits'] == 9)
    check('source_only_execution', calls['external_target_inferences'] == calls['external_target_outcome_files_read'] == 0)
    failures = [c for c in checks if not c['passed']]
    result = {'status': 'pass_source_selection_only' if not failures else 'failed',
              'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'elapsed_seconds': time.monotonic() - start,
              'command': COMMAND, 'script_sha256': sha(__file__), 'checks_total': len(checks),
              'failed_checks': failures, 'checks': checks, 'freeze_hash_checks': hash_checks,
              'read_scope': {'source_rows_used': 2354, 'target_rows_used_for_features_fits_or_predictions': 0,
                             'whole_mixed_archive_hash_and_partition_metadata_checked': True,
                             'target_outcome_files_read': False,
                             'note': 'Whole NPZ PSD member is decoded by NumPy before immediate source slicing. No external target row reaches a feature, fit or inference call; only partition metadata and opaque whole-file hashes include target records.'},
              'calls': calls, 'folds': fold_records, 'cv_replays': cv_records, 'final_model_replays': model_records,
              'source_validation_invariance': invariance, 'independent_feature_spots': feature_spots,
              'observed_inputs_and_source_artifacts_sha256': observed,
              'independence': {'all_cv_and_final_features': 'Author deterministic feature functions restricted to source-only arrays.',
                               'independent_feature_math': '24 deterministic source episodes; direct knot-wise integration for temporal/spatial bands, explicit mel triangles and orthonormal cosine sum for MFCC.',
                               'metrics_and_selection': 'Independent per-class correct/total counts, exact rational macro means and source fold construction; independently instantiated sklearn scaler/SVC fits.',
                               'not_replayed': 'Full source training under another numerical environment; target predictions; CNN source training; full external raw data extraction in this audit.'},
              'environment': {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
                              'sklearn': sklearn.__version__, 'joblib': joblib.__version__, 'threadpools': threadpool_info()},
              'gate': 'Source-only technical acceptance. External target release remains a separate parent global gate requiring all SVM and CNN source freezes.'}
    (OUT / 'external_svm_source_audit.json').write_text(json.dumps(result, indent=2))
    text = (f"# External SVM source-selection technical audit\n\nStatus: `{result['status']}`. "
            f"{len(checks)} checks, {len(hash_checks)} original freeze hashes, {len(failures)} failures. "
            f"Elapsed {result['elapsed_seconds']:.2f} seconds.\n\n"
            "The replay uses only the 2,354 source episodes at 30 and 35 mm/s. No external target feature, fit, "
            "prediction or outcome file was used. Whole-file hashes and partition metadata include the mixed archive; "
            "NumPy decodes its PSD member before immediate source slicing.\n\n"
            "All five trial-index-modulo-5 folds were reconstructed, checked for episode disjointness, exact coverage "
            "and class-by-speed coverage. All 405 source CV fits were repeated. Twelve-class macro recall and "
            "equal-weight fold means were computed from integer correct/total counts. Both the floating implementation "
            "and exact-rational comparison select the saved candidate under the declared C/gamma tie-break. "
            "All nine final models were independently refit using the saved hyperparameters; fitted scaler and SVC "
            "array bytes and hyperparameters were compared.\n\n"
            "The full fit replay uses the author's deterministic features on source-only arrays. Independent direct "
            "integration, mel triangle sums and cosine-transform calculations additionally cover 24 source episodes "
            "for temporal, spatial, mel and MFCC features. Twenty-five held-out source-fold perturbations test that "
            "calibration parameters and training features do not depend on validation labels or spectra.\n\n"
            "The audit uses the existing numerical environment with at most two BLAS/OpenMP threads. It is technical "
            "source-selection acceptance, not external target acceptance or a statistical performance claim. "
            "Target release remains subject to the parent's global all-SVM/all-CNN-source freeze gate.\n\n"
            f"Command:\n\n```sh\n{COMMAND}\n```\n\n"
            "Detailed candidate counts, parameters, source masks and hash evidence are in `external_svm_source_audit.json`.\n")
    if failures:
        text += '\nFailed checks:\n\n' + '\n'.join('- ' + c['name'] for c in failures) + '\n'
    (OUT / 'external_svm_source_audit.md').write_text(text)
    print(json.dumps({'status': result['status'], 'checks': len(checks), 'failed_checks': len(failures),
                      'seconds': result['elapsed_seconds']}), flush=True)
    return not failures

if __name__ == '__main__':
    with threadpool_limits(limits=2):
        sys.exit(0 if main() else 1)
