#!/usr/bin/env python3
"""Analyze a completed fresh external run and its sealed synthetic diagnostic.

Post-hoc specimen influence is descriptive. No fitting, exclusions, tuning or
new acceptance rule is performed. Historical reference outputs are never inputs.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import os
import numpy as np
import make_synthetic_table as synthetic
from prepare_study import verify_study

V2 = Path('outputs/research_v2')
METHODS = {'temporal', 'spatial', 'corrected', 'pooled_whitening', 'mel', 'mfcc',
           'corrected_weak', 'corrected_strong', 'wrong_order'}
SEED, DRAWS = 20270913, 10000


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def csv_rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def checked_path(root, name):
    path = Path(name)
    require('..' not in path.parts, 'Unsafe artifact path')
    path = path if path.is_absolute() else root / path
    path = path.resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file(), 'Artifact missing or outside study root')
    require('reference_release' not in path.relative_to(root.resolve()).parts,
            'Published reference output cannot represent a fresh run')
    return path


def verify_external(root):
    root = root.resolve()
    setup = verify_study(root)
    v2 = root / V2
    directory = v2 / 'experiments/calibration/external'
    bindings = {}

    def bind(path, expected=None):
        path = checked_path(root, path)
        digest = sha(path)
        require(expected is None or digest == expected, 'Hash mismatch: ' + str(path.relative_to(root)))
        bindings[str(path.relative_to(root))] = digest
        return path

    bind(root / 'STUDY_SETUP.json')
    seal_path = bind(v2 / 'research/external_global_model_freeze.json')
    seal = read(seal_path)
    require(seal['external_target_outputs_present'] is False and seal['svm_models'] == 9 and
            seal['cnn_models'] == 3, 'Global source seal is incomplete')
    require(seal['no_target_dependent_changes_authorized'] is True, 'Source-only rule is missing')
    require(seal['verified_files_sha256'], 'Empty global source inventory')
    for name, digest in seal['verified_files_sha256'].items():
        bind(root / name, digest)
    completion_path = bind(directory / 'evaluation_complete.json')
    completion = read(completion_path)
    freeze_path = bind(directory / 'freeze.json', completion['selection_freeze_sha256'])
    require(seal['verified_files_sha256'].get(str(freeze_path.relative_to(root))) == sha(freeze_path),
            'Global seal does not bind the evaluated SVM selection')
    timestamps = [dt.datetime.fromisoformat(x) for x in
                  [setup['created_utc'], seal['utc'], completion['utc']]]
    require(timestamps == sorted(timestamps), 'Fresh setup/source/target chronology is inconsistent')
    freeze = read(freeze_path)
    require(set(freeze['method_order']) == METHODS, 'Unexpected source method inventory')
    require(len([name for name in freeze['files'] if name.endswith('.joblib')]) == 9 and
            freeze['inputs'], 'Source model/input inventory is incomplete')
    for name, digest in freeze['inputs'].items():
        bind(name, digest)
    for name, digest in freeze['files'].items():
        bind(directory / name, digest)
    records = completion['records']
    require(len(records) == len(METHODS) and {r['method'] for r in records} == METHODS,
            'External target evaluation is incomplete')
    require(all(r['split'] == 'external40' and r['rows'] == 1196 for r in records),
            'External target split/count differs')
    predictions = {}
    for record in records:
        expected_name = 'external40_' + record['method'] + '_predictions.csv'
        require(record['file'] == expected_name, 'Unexpected prediction artifact name')
        predictions[record['method']] = bind(directory / record['file'], record['sha256'])
    processed = v2 / 'data/external/processed'
    publication_path = bind(processed / 'manifest.json')
    publication = read(publication_path)
    require(publication['status'] == 'complete' and publication['rows'] == 3550 and
            publication['source_rows'] == 2354 and publication['target_rows'] == 1196,
            'Eligible panel is incomplete')
    require(publication['classification_outcomes_computed'] is False, 'Eligibility uses outcomes')
    require(str(publication_path.relative_to(root)) in seal['verified_files_sha256'],
            'Publication is not bound by the source seal')
    metadata_path = bind(processed / 'metadata.csv', publication['artifacts']['metadata.csv']['sha256'])
    metadata = csv_rows(metadata_path)
    require(len(metadata) == 3550 and [int(r['row']) for r in metadata] == list(range(3550)),
            'Metadata row order/count differs')
    require(len({r['id'] for r in metadata}) == 3550, 'Duplicate eligible acquisition ID')
    decision_path = bind(v2 / 'research/external_validity_decision.json', publication['decision_sha256'])
    decision = read(decision_path)
    require([r['id'] for r in metadata] == decision['eligible_ids'], 'Eligibility IDs differ')
    split_path = bind(directory / 'external40_split.json')
    split = read(split_path)
    target = {int(r['row']): r for r in metadata if int(r['speed']) == 40}
    source = {int(r['row']) for r in metadata if int(r['speed']) in (30, 35)}
    require(len(target) == 1196 and len(source) == 2354 and not source.intersection(target),
            'Source/target panel differs')
    require(set(split['test_indices']) == set(target) and set(split['train_indices']) == source,
            'Saved split differs from eligible panel')
    return predictions, target, bindings


def paired_rows(corrected, spatial, target):
    result = []
    for path in [corrected, spatial]:
        data = csv_rows(path)
        require(len(data) == len(target) == 1196, 'Target row count differs')
        for r in data:
            for name in ['index', 'texture', 'speed', 'prediction', 'correct']:
                r[name] = int(r[name])
            require(r['index'] in target, 'Prediction includes a non-target row')
            meta = target[r['index']]
            require(r['texture'] == int(meta['class_id']) and r['speed'] == int(meta['speed']) == 40
                    and r['domain'] == 'speed', 'Prediction truth/domain differs from eligible data')
            expected_id = '_'.join(str(int(meta[k])) for k in ['class_id', 'speed', 'trial'])
            require(r['acquisition_id'] == expected_id, 'Prediction acquisition identity differs')
            require(0 <= r['prediction'] < 12 and r['correct'] == int(r['prediction'] == r['texture']),
                    'Invalid prediction/correct field')
        require({r['index'] for r in data} == set(target) and
                len({r['acquisition_id'] for r in data}) == len(data), 'Duplicate or missing target keys')
        result.append(sorted(data, key=lambda r: r['index']))
    a, b = result
    require(all(all(x[k] == y[k] for k in ['index', 'acquisition_id', 'texture', 'speed', 'domain'])
                for x, y in zip(a, b)), 'Methods are not paired')
    return a, b


def summarize(a, b):
    ids = sorted({r['texture'] for r in a})
    require(ids == list(range(12)), 'All twelve specimens are required')
    rows = []
    for specimen in ids:
        left = [r for r in a if r['texture'] == specimen]
        right = [r for r in b if r['texture'] == specimen]
        require(len(left) == len(right) and len(left) > 0, 'Unpaired specimen')
        n, ca, cb = len(left), sum(r['correct'] for r in left), sum(r['correct'] for r in right)
        delta = 100 * (ca - cb) / n
        rows.append(dict(specimen_id=specimen, n=n, corrected_correct=ca, spatial_correct=cb,
                         corrected_recall_pct=100*ca/n, spatial_recall_pct=100*cb/n,
                         difference_pp=delta, contribution_to_12_specimen_macro_pp=delta/12))
    diffs = np.array([r['difference_pp'] for r in rows])
    mean = float(diffs.mean())
    draws = np.random.default_rng(SEED).integers(0, 12, (DRAWS, 12))
    low, high = np.quantile(diffs[draws].mean(axis=1), [.025, .975])
    for row in rows:
        loo = float(np.mean([r['difference_pp'] for r in rows if r is not row]))
        row.update(leave_one_specimen_out_difference_pp=loo, change_from_all12_pp=loo-mean,
                   removal_changes_sign=bool(np.sign(loo) != np.sign(mean)))
    signs = dict(improved=int(np.sum(diffs > 0)), worsened=int(np.sum(diffs < 0)), tied=int(np.sum(diffs == 0)))
    summary = dict(status='post-hoc descriptive influence analysis of the specified completed run',
        n_target_acquisitions=len(a), n_specimens=12,
        corrected_macro_recall_pct=float(np.mean([r['corrected_recall_pct'] for r in rows])),
        spatial_macro_recall_pct=float(np.mean([r['spatial_recall_pct'] for r in rows])),
        all12_difference_pp=mean, conditional_interval_pp=[float(low), float(high)],
        resampling_seed=SEED, resampling_draws=DRAWS,
        interval_scope='paired physical-specimen recalls conditional on saved models and empirical panel; not future-specimen or retraining uncertainty',
        specimen_signs=signs, primary_support=bool(mean >= 3 and low > 0),
        primary_rule='difference >= 3 pp and conditional lower bound > 0; unchanged',
        leave_one_specimen_out_range_pp=[min(r['leave_one_specimen_out_difference_pp'] for r in rows), max(r['leave_one_specimen_out_difference_pp'] for r in rows)],
        removals_changing_sign=[r['specimen_id'] for r in rows if r['removal_changes_sign']],
        all_specimens_retained=True, fitting_performed=False,
        limitations=['Post-hoc influence; no specimen is excluded.', 'Technical verification is not independent peer review.',
                     'No new material-instance or isolated-session validation.'])
    return rows, summary


def plot_limits(values):
    lo, hi = min(0., min(values)), max(0., max(values))
    margin = max(3., .12 * (hi - lo))
    return lo - margin, hi + margin


def render(rows, summary, synthetic_rows, output):
    os.environ.setdefault('MPLCONFIGDIR', str(output / 'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.labelsize': 7,
                         'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    blue, orange = '#3175A5', '#D48B46'
    values = {condition: [100 * float(synthetic_rows[('B', 'Cluster', condition, method)]['target_macro_accuracy'])
              for method in ['spatial', 'corrected']] for condition in ['ideal', 'lowpass_200Hz']}
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.0), gridspec_kw={'width_ratios': [1.15, 1, 1.25]})
    a, b, c = axes
    a.set_xlim(0, 1); a.set_ylim(0, 1); a.axis('off')
    a.set_title('(a) Source-only calibration', fontsize=8, fontweight='bold')
    for x, labels, color in [(.01, ['Source spectra\n+ labels', 'Class balance\n+ double centering', 'Cosine ridge fit'], blue),
                              (.53, ['Target spectra\n+ speed', 'Subtract\n+ renormalize', 'Source scaler\n+ SVM'], orange)]:
        for y, label in zip([.72, .43, .14], labels):
            a.add_patch(FancyBboxPatch((x, y), .46, .19, boxstyle='round,pad=.01', facecolor='white', edgecolor=color, linewidth=.7))
            a.text(x+.23, y+.095, label, ha='center', va='center', fontsize=5.5)
        for y in [.72, .43]:
            a.annotate('', xy=(x+.23, y-.09), xytext=(x+.23, y), arrowprops={'arrowstyle': '->', 'lw': .7})
    a.annotate('', xy=(.53, .52), xytext=(.47, .24), arrowprops={'arrowstyle': '->', 'color': blue, 'lw': .8})
    a.text(.5, .015, 'Target labels/distribution excluded from fitting', fontsize=5.1, ha='center')
    b.set_title('(b) Synthetic observations', fontsize=8, fontweight='bold')
    for k, (label, color) in enumerate([('Spatial', orange), ('Calibrated', blue)]):
        heights = [values[o][k] for o in values]
        positions = np.arange(2) + (k-.5)*.32
        b.bar(positions, heights, .30, color=color, label=label, hatch='' if k == 0 else '///', edgecolor='white', linewidth=.4)
        for x, y in zip(positions, heights):
            b.text(x, y+2, f'{y:.2f}', fontsize=5.8, ha='center')
    b.set_xticks([0, 1], ['Ideal', 'Low-pass']); b.set_ylim(0, 115); b.set_yticks([0, 50, 100])
    b.set_ylabel('Accuracy (%)'); b.legend(frameon=False, ncol=2, fontsize=5.8, loc='lower center', bbox_to_anchor=(.5, -.36))
    c.set_title('(c) External specimen effects', fontsize=8, fontweight='bold')
    ids = [r['specimen_id'] for r in rows]; diffs = [r['difference_pp'] for r in rows]
    mean = summary['all12_difference_pp']; low, high = summary['conditional_interval_pp']
    c.bar(ids, diffs, color=blue, width=.72)
    c.axhline(0, color='.25', linewidth=.7); c.axvline(12.1, color='.8', linewidth=.6)
    c.errorbar(13.2, mean, yerr=[[mean-low], [high-mean]], fmt='D', color='.2', markersize=3, capsize=2)
    c.set_xticks(ids+[13.2], [str(i) for i in ids]+['All']); c.set_xlim(-.8, 14.2)
    c.set_ylim(*plot_limits(diffs+[mean, low, high])); c.set_ylabel('Calibrated − spatial (pp)')
    c.set_xlabel('Specimen ID; all retained')
    for ax in [b, c]:
        ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.tight_layout(pad=.6, w_pad=1.1)
    for extension in ['pdf', 'svg', 'png']:
        fig.savefig(output / ('overview.'+extension), dpi=300)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 3.2), sharey=True)
    y = np.arange(12); loo = [r['leave_one_specimen_out_difference_pp'] for r in rows]
    axes[0].hlines(y, 0, diffs, color='.6'); axes[0].scatter(diffs, y, color=blue, s=15)
    axes[1].scatter(loo, y, color=blue, s=15); axes[1].axvline(mean, color=orange, linestyle='--', label='All 12')
    for ax in axes:
        ax.axvline(0, color='.25', linewidth=.7); ax.set_ylim(11.6, -.6); ax.grid(axis='x', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_yticks(y, ids); axes[0].set_ylabel('Specimen ID')
    axes[0].set_title('Paired recall difference'); axes[1].set_title('Omit indicated specimen (influence only)')
    axes[0].set_xlabel('Calibrated − spatial (pp)'); axes[1].set_xlabel('Panel mean difference (pp)'); axes[1].legend(frameon=False)
    fig.tight_layout(pad=.6)
    for extension in ['pdf', 'svg', 'png']:
        fig.savefig(output / ('specimen_influence.'+extension), dpi=220)
    plt.close(fig)
    dump(output / 'figure_source_data.json', {'synthetic_accuracy_pct': values, 'specimens': rows,
                                             'external_summary': summary})


def export(root, run, output):
    root, run, output = root.resolve(), run.resolve(), output.resolve()
    require(not output.exists(), 'Output must be a new directory')
    require(run.is_relative_to(root) and 'reference_release' not in run.relative_to(root).parts,
            'Synthetic run must be inside the new study, outside published references')
    require(not output.is_relative_to(run) and not output.is_relative_to(root / V2 / 'experiments'),
            'Output cannot modify sealed experiment directories')
    predictions, target, bindings = verify_external(root)
    a, b = paired_rows(predictions['corrected'], predictions['spatial'], target)
    rows, summary = summarize(a, b)
    synthetic.STUDY = root / V2
    _, synthetic_rows, _, _ = synthetic.verify_run(run)
    initialized = read(run / 'initialization.json')
    setup = read(root / 'STUDY_SETUP.json')
    require(dt.datetime.fromisoformat(initialized['utc']) >= dt.datetime.fromisoformat(setup['created_utc']),
            'Synthetic run predates the fresh source tree')
    for name in ['initialization.json', 'source_seal.json', 'result_seal.json', 'verification.json']:
        bindings[str((run / name).relative_to(root))] = sha(run / name)
    output.mkdir(parents=True)
    with (output / 'per_specimen_effects.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    dump(output / 'summary.json', summary)
    table = ['% Generated from completed fresh predictions; post-hoc influence only.', r'\begin{tabular}{rrrrrr}',
             r'\hline', r'ID & $n$ & Corrected & Spatial & $\Delta$ (pp) & Omit ID (pp) \\', r'\hline']
    for r in rows:
        table.append(f"{r['specimen_id']} & {r['n']} & {r['corrected_correct']} & {r['spatial_correct']} & {r['difference_pp']:+.2f} & {r['leave_one_specimen_out_difference_pp']:+.2f} " + r'\\')
    table += [r'\hline', r'\end{tabular}']
    (output / 'specimen_influence_table.tex').write_text('\n'.join(table)+'\n')
    signs = summary['specimen_signs']
    macros = {'SpecimenImproved': signs['improved'], 'SpecimenWorsened': signs['worsened'],
              'SpecimenTied': signs['tied'], 'InfluenceAllPanel': f"{summary['all12_difference_pp']:.2f}",
              'InfluenceLow': f"{summary['leave_one_specimen_out_range_pp'][0]:.2f}",
              'InfluenceHigh': f"{summary['leave_one_specimen_out_range_pp'][1]:.2f}"}
    (output / 'specimen_influence_numbers.tex').write_text(
        '% Generated from this run; post-hoc influence only.\n' +
        ''.join('\\newcommand{\\'+name+'}{'+str(value)+'}\n' for name, value in macros.items()))
    render(rows, summary, synthetic_rows, output)
    require(all(sha(root / name) == digest for name, digest in bindings.items()), 'Input changed during analysis')
    dump(output / 'manifest.json', {'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'status': 'completed_fresh_run_descriptive_analysis', 'independent_review': False,
        'new_model_fitting': False, 'input_sha256': bindings, 'adapter_sha256': sha(Path(__file__)),
        'outputs': {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return {'status': 'complete', 'target_rows': len(a), 'specimens': len(rows),
            'primary_support': summary['primary_support'], 'output': str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--synthetic-run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = export(args.project_root, args.synthetic_run_dir, args.output_dir)
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.exit(1, f'Fresh-result verification failed: {error}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
