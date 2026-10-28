"""Generate all revision plots and manuscript numeric inputs without hand edits."""
import os
os.environ['MPLCONFIGDIR'] = str(__import__('pathlib').Path(__file__).resolve().parents[3] / 'work/revision/matplotlib')
import argparse
import hashlib
import json
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
OLD = ROOT.parent / 'research'
AN = ROOT / 'analysis'
FIG = ROOT / 'figures'
PAPER = ROOT / 'paper'
LABELS = {'temporal': 'Temporal', 'spatial': 'Spatial', 'corrected': 'Calibrated spatial',
          'pooled_whitening': 'Pooled whitening', 'mel': 'Log-mel', 'mfcc': 'MFCC summary',
          'resnet34': 'ResNet34', 'corrected_weak': r'Calibration $\lambda=.001$',
          'corrected_strong': r'Calibration $\lambda=.1$', 'wrong_order': 'Reversed source order'}


def main(require_complete=False):
    report = json.loads((AN / 'report.json').read_text())
    if require_complete:
        assert not report['pending'], report['pending']
    FIG.mkdir(exist_ok=True)
    PAPER.mkdir(exist_ok=True)
    scores = pd.read_csv(AN / 'scores.csv')
    contrasts = pd.read_csv(AN / 'contrasts.csv')
    cv = pd.read_csv(AN / 'source_cv_folds.csv')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8,
                         'axes.titlesize': 9, 'legend.fontsize': 7, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    def score(dataset, split, method, domain='speed'):
        seed = 'seed_mean' if method == 'resnet34' else 'deterministic'
        d = scores[(scores.dataset == dataset) & (scores.split == split) &
                   (scores.method == method) & (scores.domain == domain) & (scores.seed == seed)]
        return None if len(d) == 0 else float(d.iloc[0].macro_accuracy * 100)

    def cv_score(dataset, split, method):
        if method == 'resnet34':
            p = ROOT / 'experiments/cnn' / dataset / 'selection.json'
            if not p.exists():
                return None
            d = json.loads(p.read_text())
            choice = d.get('selected', d)
            for key in ['mean_cv_accuracy', 'mean_validation_accuracy', 'mean_accuracy', 'cv_mean_accuracy', 'cv_mean']:
                if key in choice:
                    return choice[key] * 100
            raise KeyError(('Unexpected CNN selection fields', choice.keys()))
        d = cv[(cv.dataset == dataset) & (cv.split == split) & (cv.method == method)]
        return None if len(d) == 0 else float(d.iloc[0].cv_mean * 100)

    fmt = lambda value: '--' if value is None else f'{value:.2f}'
    methods = ['temporal', 'spatial', 'corrected', 'pooled_whitening', 'mel', 'mfcc', 'resnet34']
    lines = [r'\begin{tabular}{lrrrrrr}', r'\hline',
             r'& \multicolumn{4}{c}{Cluster (118 specimens)} & \multicolumn{2}{c}{Lima (12 specimens)} \\',
             r'Representation & Source CV & 20 mm/s & 60 mm/s & Pooled R0 / R1 & Source CV & 40 mm/s \\',
             r'\hline']
    table_rows = []
    for method in methods:
        values = [cv_score('cluster', 'repeat0', method), score('cluster', 'repeat0', method, 'speed_20'),
                  score('cluster', 'repeat0', method, 'speed_60'), score('cluster', 'repeat0', method),
                  score('cluster', 'repeat1', method), cv_score('external', 'external40', method),
                  score('external', 'external40', method)]
        lines.append(LABELS[method] + ' & ' + ' & '.join([fmt(values[0]), fmt(values[1]), fmt(values[2]),
                     fmt(values[3]) + ' / ' + fmt(values[4]), fmt(values[5]), fmt(values[6])]) + r' \\')
        table_rows.append(dict(method=method, **dict(zip(['cluster_cv', 'cluster20', 'cluster60',
                               'cluster_r0', 'cluster_r1', 'external_cv', 'external40'], values))))
    lines += [r'\hline', r'\end{tabular}']
    (PAPER / 'main_table.tex').write_text('\n'.join(lines) + '\n')
    pd.DataFrame(table_rows).to_csv(AN / 'main_table_values.csv', index=False)

    # All supplemental metrics are emitted, not just favorable contrasts.
    for name, data in [('all_scores', scores), ('source_cv_folds', cv), ('all_contrasts', contrasts)]:
        data.to_csv(PAPER / f'{name}.csv', index=False)
    old_scores = pd.read_csv(OLD / 'analysis/scores.csv')
    signed = old_scores[(old_scores['mode'] == 0) & (old_scores.training_repeat == 0) &
                        (old_scores.domain == 'speed') & old_scores.method.isin(['spatial', 'partial'])].copy()
    signed = signed[['method', 'speed_error', 'n', 'accuracy', 'ci_low', 'ci_high']].sort_values(['method', 'speed_error'])
    signed.to_csv(AN / 'signed_speed_error.csv', index=False)

    # C groups are genuinely separated; no line connects gamma resets.
    grid = pd.read_csv(OLD / 'analysis/posthoc_grid_contrasts.csv')
    grid = grid[grid.domain == 'speed']
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.2), sharey=True)
    for ax, C in zip(axes, [1, 10, 100]):
        for repeat, color, marker, offset in [(0, '#235C9C', 'o', -.055), (1, '#B25332', 's', .055)]:
            d = grid[(grid.C == C) & (grid.training_repeat == repeat)].sort_values('gamma_multiplier')
            y = d.difference_pp.to_numpy()
            ax.errorbar(np.arange(3) + offset, y,
                        yerr=[y - d.ci_low_pp.to_numpy(), d.ci_high_pp.to_numpy() - y],
                        fmt=marker, color=color, markersize=4, capsize=2, linewidth=.8, label=f'Repeat {repeat}')
        ax.axhline(0, color='0.35', linewidth=.7)
        ax.set_title(f'C = {C}')
        ax.set_xticks(range(3), ['0.25', '1', '4'])
        ax.set_xlabel(r'$g$ in $\gamma=g/128$')
        ax.grid(axis='y', alpha=.15)
    axes[0].set_ylabel('Spatial − temporal (pp)')
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc='lower center', ncol=2,
               bbox_to_anchor=(.5, -.015))
    fig.tight_layout(rect=[0, .10, 1, 1])
    fig.savefig(FIG / 'matched_grid.pdf')
    fig.savefig(FIG / 'matched_grid.png', dpi=200)
    plt.close(fig)

    rows = []
    for method, label in [('spatial', 'Original 1 s'), ('common_band_spatial', 'Shared 60–400 Hz'),
                          ('fixed_space_spatial', 'Fixed spatial resolution')]:
        for repeat in [0, 1]:
            r = contrasts[(contrasts.dataset == 'cluster') & (contrasts.split == f'repeat{repeat}') &
                          (contrasts.method_a == method)].iloc[0]
            rows.append((label, repeat, r))
    fig, ax = plt.subplots(figsize=(3.35, 2.3))
    for i, (label, repeat, r) in enumerate(rows):
        color = '#235C9C' if repeat == 0 else '#B25332'
        ax.errorbar(r.difference_pp, i, xerr=[[r.difference_pp - r.conditional_low_pp],
                    [r.conditional_high_pp - r.difference_pp]], fmt='o' if repeat == 0 else 's',
                    color=color, capsize=2, markersize=4, linewidth=1)
    ax.set_yticks(range(len(rows)), [f'{label}, R{repeat}' for label, repeat, _ in rows])
    ax.axvline(0, color='0.4', linewidth=.8)
    ax.invert_yaxis()
    ax.set_xlabel('Spatial − temporal macro accuracy (pp)')
    ax.grid(axis='x', alpha=.15)
    fig.tight_layout(pad=.5)
    fig.savefig(FIG / 'estimator_controls.pdf', bbox_inches='tight')
    fig.savefig(FIG / 'estimator_controls.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    synthetic = pd.read_csv(AN / 'synthetic_diagnostics.csv')
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.45))
    for rep, color, offset in [('temporal', '#235C9C', -.17), ('spatial', '#B25332', .17)]:
        d = synthetic[synthetic.representation == rep].set_index('observation').loc[['ideal', 'lowpass_200Hz']]
        for ax, metric, factor in [(axes[0], 'target_accuracy', 100), (axes[1], 'mean_speed_JS_divergence', 1)]:
            ax.bar(np.arange(2) + offset, d[metric] * factor, width=.32, label=rep.capitalize(), color=color)
    for ax in axes:
        ax.set_xticks([0, 1], ['Ideal observation', '200 Hz low-pass'])
    axes[0].set_ylabel('Synthetic target accuracy (%)')
    axes[1].set_ylabel('Five-speed JS divergence (nats)')
    axes[0].legend(frameon=False)
    fig.suptitle('Constructed two-band signals; no physical measurements', fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / 'synthetic_diagnostics.pdf')
    fig.savefig(FIG / 'synthetic_diagnostics.png', dpi=200)
    plt.close(fig)

    # Every macro records the exact table/row source used to create it.
    macros = {}
    lineage = {}
    def put(name, value, source, percent=False, digits=2):
        if value is None:
            return
        macros[name] = f'{value * (100 if percent else 1):.{digits}f}'
        lineage[name] = {'value': value, 'printed': macros[name], 'source': source}
    for rec in table_rows:
        prefix = {'temporal': 'Temporal', 'spatial': 'Spatial', 'corrected': 'Corrected',
                  'pooled_whitening': 'Whitening', 'mel': 'Mel', 'mfcc': 'Mfcc', 'resnet34': 'Cnn'}[rec['method']]
        for key, suffix in [('cluster_r0', 'ClusterZero'), ('cluster_r1', 'ClusterOne'),
                            ('external40', 'External'), ('cluster_cv', 'SourceCv')]:
            put(prefix + suffix, rec[key], f'main_table_values.csv:{rec["method"]}:{key}')
    for split, suffix in [('repeat0', 'Zero'), ('repeat1', 'One')]:
        put('TemporalId' + suffix, score('cluster', split, 'temporal', 'id'),
            f'scores.csv:cluster:{split}:temporal:id:macro_accuracy*100')
    put('ClusterChance', 100 / 118, 'derived uniform prediction expectation: 100/118 classes')
    put('ExternalChance', 100 / 12, 'derived uniform prediction expectation: 100/12 classes')
    for dataset, split, prefix in [('cluster', 'repeat0', 'CalibrationZero'),
                                  ('cluster', 'repeat1', 'CalibrationOne'),
                                  ('external', 'external40', 'CalibrationExternal')]:
        d = contrasts[(contrasts.dataset == dataset) & (contrasts.split == split) &
                      (contrasts.method_a == 'corrected') & (contrasts.method_b == 'spatial')]
        if len(d):
            for key, suffix in [('difference_pp', 'Delta'), ('conditional_low_pp', 'Low'), ('conditional_high_pp', 'High')]:
                put(prefix + suffix, d.iloc[0][key], f'contrasts.csv:{dataset}:{split}:corrected-spatial:{key}')
    old_contrasts = pd.read_csv(OLD / 'analysis/contrasts.csv')
    primary = old_contrasts[(old_contrasts['mode'] == 0) & (old_contrasts.training_repeat == 0) &
                            (old_contrasts.method == 'spatial') & (old_contrasts.reference == 'temporal') &
                            (old_contrasts.domain == 'speed')].iloc[0]
    for key, suffix in [('difference_pp', 'Delta'), ('ci_low_pp', 'Low'), ('ci_high_pp', 'High')]:
        put('OriginalPrimary' + suffix, primary[key], f'../research/analysis/contrasts.csv:primary:{key}')
    for observation, method, name in [('ideal', 'spatial', 'SyntheticIdealSpatial'),
                                      ('ideal', 'temporal', 'SyntheticIdealTemporal'),
                                      ('lowpass_200Hz', 'spatial', 'SyntheticFilteredSpatial')]:
        value = synthetic[(synthetic.observation == observation) &
                          (synthetic.representation == method)].iloc[0].target_accuracy
        put(name, value, f'synthetic_diagnostics.csv:{observation}:{method}:target_accuracy', percent=True)
    alignment = pd.read_csv(AN / 'alignment_summary.csv')
    for method, prefix in [('spatial', 'Spatial'), ('corrected', 'Corrected')]:
        rec = alignment[(alignment.dataset == 'cluster') & (alignment.method == method)].iloc[0]
        for key, suffix in [('mean_within_specimen_speed_js_nats', 'WithinJs'),
                            ('between_specimen_centroid_js_nats', 'BetweenJs')]:
            put(prefix + suffix, rec[key], f'alignment_summary.csv:cluster:{method}:{key}', digits=3)
    decision = json.loads((ROOT / 'research/external_validity_decision.json').read_text())
    for key, name in [('planned_episodes', 'ExternalPlanned'), ('quality_excluded', 'ExternalQcExcluded'),
                      ('additional_exact_duplicates_excluded', 'ExternalDuplicateExcluded'),
                      ('eligible_episodes', 'ExternalEligible'), ('source_episodes', 'ExternalSourceN'),
                      ('target_episodes', 'ExternalTargetN')]:
        put(name, decision[key], f'research/external_validity_decision.json:{key}', digits=0)
    put('ExternalQcPercent', decision['quality_excluded'] / decision['planned_episodes'],
        'research/external_validity_decision.json:quality_excluded/planned_episodes', percent=True)
    (PAPER / 'numbers.tex').write_text('\n'.join('\\newcommand{\\' + key + '}{' + value + '}' for key, value in macros.items()) + '\n')
    (AN / 'numeric_lineage.json').write_text(json.dumps(lineage, indent=2))
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(FIG.iterdir()) if p.is_file()}
    (AN / 'figure_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'figures': len(manifest), 'numeric_macros': len(macros), 'pending': report['pending']}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--require-complete', action='store_true')
    main(ap.parse_args().require_complete)
