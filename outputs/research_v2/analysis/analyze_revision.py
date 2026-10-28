"""Rebuild revision tables from hash-verified, completed prediction artifacts.

Percentile intervals resample physical specimen recalls conditional on the saved
models. They do not estimate uncertainty from future specimens or retraining.
Neural seeds are averaged within specimen; they are never statistical units.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
import argparse
import datetime
import hashlib
import json
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT.parents[1]
OUT = ROOT / 'analysis'
SEED = 20270913
inputs = {}
pending = []
frames = {}
scores = []
specimens = []
selections = []
costs = []


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def record(path, expected=None):
    path = pathlib.Path(path)
    digest = sha(path)
    if expected is not None:
        assert digest == expected, str(path)
    inputs[str(path.relative_to(BASE))] = digest
    return path


def read_json(path):
    return json.loads(record(path).read_text())


def interval(values):
    values = np.asarray(values, dtype=float)
    assert np.isfinite(values).all()
    draws = np.random.default_rng(SEED).integers(0, len(values), (10000, len(values)))
    return np.quantile(values[draws].mean(axis=1), [.025, .975])


def add_frame(dataset, split, method, path, digest, seed='deterministic'):
    d = pd.read_csv(record(path, digest))
    if 'class_id' in d:
        d['texture'] = d['class_id']
    if 'index' not in d:
        d['index'] = d['row']
    assert d['index'].is_unique
    assert np.array_equal(d.correct.to_numpy(), (d.texture == d.prediction).astype(int).to_numpy())
    assert d.prediction.between(0, 117 if dataset == 'cluster' else 11).all()
    d = d.sort_values('index')
    frames[(dataset, split, method, str(seed))] = d
    add_scores(dataset, split, method, str(seed), d)


def add_scores(dataset, split, method, seed, d):
    for domain in sorted(d.domain.unique()):
        part = d[d.domain == domain]
        by_class = part.groupby('texture').correct.mean().sort_index()
        assert len(by_class) == (118 if dataset == 'cluster' else 12)
        lo, hi = interval(by_class)
        scores.append(dict(dataset=dataset, split=split, method=method, seed=seed,
                           domain=domain, episodes=len(part), physical_specimens=len(by_class),
                           macro_accuracy=by_class.mean(), micro_accuracy=part.correct.mean(),
                           conditional_low=lo, conditional_high=hi))
        for texture, recall in by_class.items():
            specimens.append(dict(dataset=dataset, split=split, method=method,
                                  seed=seed, domain=domain, texture=int(texture), recall=recall))
        if domain == 'speed':
            for speed, endpoint in part.groupby('speed'):
                rc = endpoint.groupby('texture').correct.mean()
                low, high = interval(rc)
                scores.append(dict(dataset=dataset, split=split, method=method, seed=seed,
                                   domain=f'speed_{int(speed)}', episodes=len(endpoint),
                                   physical_specimens=len(rc), macro_accuracy=rc.mean(),
                                   micro_accuracy=endpoint.correct.mean(), conditional_low=low,
                                   conditional_high=high))


def source_selection(dataset, rec):
    choice = rec['selected']
    for i, value in enumerate(choice['fold_scores']):
        selections.append(dict(dataset=dataset, split=rec['split'], method=rec['method'],
                               fold=i, macro_accuracy=value, cv_mean=choice['cv_mean'],
                               C=choice['C'], gamma_multiplier=choice['g']))


def load_calibration(dataset):
    directory = ROOT / 'experiments/calibration' / dataset
    completion = directory / 'evaluation_complete.json'
    if not completion.exists():
        pending.append(f'{dataset}_calibration')
        return
    manifest = read_json(completion)
    record(directory / 'freeze.json', manifest['selection_freeze_sha256'])
    for rec in manifest['records']:
        add_frame(dataset, rec['split'], rec['method'], directory / rec['file'], rec['sha256'])
        costs.append(dict(dataset=dataset, split=rec['split'], method=rec['method'],
                          prediction_seconds=rec['prediction_seconds'], episodes=rec['rows']))
    for rec in read_json(directory / 'selected.json'):
        source_selection(dataset, rec)


def load_diagnostics():
    directory = ROOT / 'experiments/diagnostics'
    manifest = read_json(directory / 'real_results_manifest.json')
    for rec in manifest['records']:
        add_frame('cluster', rec['split'], rec['method'], directory / rec['file'], rec['sha256'])
    for rec in read_json(directory / 'real_selection.json'):
        source_selection('cluster', rec)
    synthetic = read_json(directory / 'synthetic_results.json')
    for rec in synthetic['records']:
        d = pd.read_csv(record(directory / rec['file'], rec['sha256']))
        assert np.array_equal(d.correct, (d.texture == d.prediction).astype(int))
        assert abs(d.groupby('texture').correct.mean().mean() - rec['target_accuracy']) < 1e-12
    pd.DataFrame([{k: v for k, v in rec.items() if k not in ['all_source_candidates', 'source_selection']}
                  for rec in synthetic['records']]).to_csv(OUT / 'synthetic_diagnostics.csv', index=False)


def load_cnn(dataset):
    directory = ROOT / 'experiments/cnn' / dataset
    complete = directory / 'predictions/completion.json'
    if not complete.exists():
        pending.append(f'{dataset}_cnn')
        return
    completion = read_json(complete)
    record(directory / 'summary.json', completion['summary_sha256'])
    summary = read_json(directory / 'summary.json')
    record(directory / 'predictions/manifest.json', completion['manifest_sha256'])
    split = 'repeat0' if dataset == 'cluster' else 'external40'
    for rec in summary['per_seed']:
        add_frame(dataset, split, 'resnet34', BASE / rec['prediction_file'],
                  rec['prediction_sha256'], rec['seed'])
        costs.append(dict(dataset=dataset, split=split, method=f'resnet34_seed{rec["seed"]}',
                          prediction_seconds=rec['metrics']['seconds'], episodes=rec['metrics']['n']))
    ds = [frames[(dataset, split, 'resnet34', str(seed))] for seed in summary['seeds']]
    for d in ds[1:]:
        assert np.array_equal(ds[0][['index', 'texture', 'speed', 'domain']], d[['index', 'texture', 'speed', 'domain']])
    mean = ds[0].copy()
    mean['correct'] = np.mean([d.correct.to_numpy() for d in ds], axis=0)
    frames[(dataset, split, 'resnet34', 'seed_mean')] = mean
    add_scores(dataset, split, 'resnet34', 'seed_mean', mean)
    selected = summary['selected']
    (OUT / f'{dataset}_cnn_selection.json').write_text(json.dumps(selected, indent=2))


def contrast(dataset, split, a, b, kind='descriptive', seed_a='deterministic', seed_b='deterministic'):
    ka, kb = (dataset, split, a, seed_a), (dataset, split, b, seed_b)
    if ka not in frames or kb not in frames:
        return None
    left, right = frames[ka], frames[kb]
    assert np.array_equal(left[['index', 'texture', 'speed', 'domain']], right[['index', 'texture', 'speed', 'domain']])
    select = left.domain == 'speed'
    paired = left.loc[select, ['texture', 'correct']].copy()
    paired['difference'] = left.loc[select, 'correct'].to_numpy() - right.loc[select, 'correct'].to_numpy()
    vector = paired.groupby('texture').difference.mean()
    lo, hi = interval(vector)
    return dict(dataset=dataset, split=split, method_a=a, method_b=b, seed_a=seed_a,
                seed_b=seed_b, domain='speed', analysis_status=kind, specimens=len(vector),
                difference_pp=100 * vector.mean(), conditional_low_pp=100 * lo,
                conditional_high_pp=100 * hi, resampling_seed=SEED,
                specimens_improved=int((vector > 0).sum()),
                specimens_worsened=int((vector < 0).sum()), specimens_tied=int((vector == 0).sum()))


def main(require_complete=False):
    if (OUT / 'freeze.json').exists():
        raise RuntimeError('Analysis is frozen; verify in an isolated copy rather than overwrite.')
    OUT.mkdir(exist_ok=True)
    load_calibration('cluster')
    load_calibration('external')
    load_diagnostics()
    load_cnn('cluster')
    load_cnn('external')
    contrasts = []
    for dataset, splits in [('cluster', ['repeat0', 'repeat1']), ('external', ['external40'])]:
        for split in splits:
            for a, b in [('spatial', 'temporal'), ('corrected', 'spatial'),
                         ('corrected', 'pooled_whitening'), ('corrected', 'mel'),
                         ('mfcc', 'mel'), ('corrected_weak', 'corrected'),
                         ('corrected_strong', 'corrected'), ('wrong_order', 'corrected'),
                         ('fixed_space_spatial', 'fixed_space_temporal'),
                         ('common_band_spatial', 'common_band_temporal')]:
                status = 'exploratory Cluster revision' if dataset == 'cluster' else 'descriptive external'
                if dataset == 'external' and (a, b) == ('corrected', 'spatial'):
                    status = 'locked external primary'
                item = contrast(dataset, split, a, b, status)
                if item:
                    contrasts.append(item)
            item = contrast(dataset, split, 'resnet34', 'mfcc', 'descriptive model-seed mean', 'seed_mean')
            if item:
                contrasts.append(item)
    primary = next((x for x in contrasts if x['analysis_status'] == 'locked external primary'), None)
    support = None if primary is None else bool(primary['difference_pp'] >= 3 and primary['conditional_low_pp'] > 0)
    for item in contrasts:
        if item['dataset'] == 'external' and item['method_a'] == 'corrected' and item['method_b'] == 'mel':
            item['analysis_status'] = 'gatekept secondary' if support else 'descriptive; secondary benefit claim not opened'
    pd.DataFrame(scores).to_csv(OUT / 'scores.csv', index=False)
    pd.DataFrame(specimens).to_csv(OUT / 'specimen_recalls.csv', index=False)
    pd.DataFrame(selections).to_csv(OUT / 'source_cv_folds.csv', index=False)
    pd.DataFrame(contrasts).to_csv(OUT / 'contrasts.csv', index=False)
    pd.DataFrame(costs).to_csv(OUT / 'prediction_costs.csv', index=False)
    report = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  pending=pending, external_primary_support=support, external_primary=primary,
                  prediction_models=len(frames), resampling='paired physical-specimen recalls; 10000 percentile draws',
                  interval_scope='conditional on saved models and empirical specimen panel; no future-sensor, new-specimen or retraining coverage claim',
                  neural_summary='mean per-specimen recall over all three seeds; not an ensemble prediction',
                  input_sha256=inputs, script_sha256=sha(__file__))
    (OUT / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ['pending', 'external_primary_support', 'prediction_models']}, indent=2))
    print(pd.DataFrame(contrasts).to_string(index=False))
    if require_complete:
        assert not pending, pending


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--require-complete', action='store_true')
    main(ap.parse_args().require_complete)
