"""Emit supplementary tables exclusively from recorded result and protocol files."""
import argparse
import hashlib
import json
import pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
OLD = ROOT.parent / 'research'
OUT = ROOT / 'paper/supplement_tables'
NAMES = {'temporal': 'Temporal', 'spatial': 'Spatial', 'corrected': 'Calibrated',
         'pooled_whitening': 'Whitening', 'mel': 'Log-mel', 'mfcc': 'MFCC',
         'corrected_weak': 'Calib. weak', 'corrected_strong': 'Calib. strong',
         'wrong_order': 'Reversed order', 'resnet34': 'ResNet34',
         'common_band_temporal': 'Shared-band temporal', 'common_band_spatial': 'Shared-band spatial',
         'fixed_space_temporal': 'Fixed-space temporal', 'fixed_space_spatial': 'Fixed-space spatial',
         'partial': 'Partial', 'dual': 'Dual', 'metadata': 'Metadata only',
         'endpoint_union': 'Union band', 'temporal_gain': 'Temporal + gain',
         'spatial_gain': 'Spatial + gain'}
DOMAINS = ['id', 'speed', 'direction', 'load', 'speed_direction', 'speed_load',
           'direction_load', 'speed_direction_load']
INPUTS = {}


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_csv(p):
    INPUTS[str(p.relative_to(ROOT.parent))] = digest(p)
    return pd.read_csv(p)


def read_json(p):
    INPUTS[str(p.relative_to(ROOT.parent))] = digest(p)
    return json.loads(p.read_text())


def pct(value):
    return '--' if pd.isna(value) else f'{100 * float(value):.2f}'


def esc(value):
    text = str(value)
    return text.replace('&', r'\&').replace('_', r'\_').replace('%', r'\%')


def primary_caption(support):
    if support is True:
        return ('The primary benefit criterion was met, so calibrated-minus-log-mel '
                'is eligible for the predeclared secondary comparison. ')
    if support is False:
        return ('The primary benefit criterion was not met, so calibrated-minus-log-mel '
                'remains descriptive; its secondary gate did not open. ')
    return 'The primary comparison is unavailable; its secondary gate has not opened. '


def table(name, caption, headers, rows, align=None):
    align = align or ('l' + 'r' * (len(headers) - 1))
    h = ' & '.join(esc(x) for x in headers) + r' \\'
    lines = [r'\begingroup\normalsize', r'\setlength{\tabcolsep}{4pt}',
             r'\begin{longtable}{' + align + '}',
             r'\caption{' + caption + r'}\label{tab:s-' + name + r'}\\',
             r'\toprule', h, r'\midrule\endfirsthead',
             r'\toprule', h, r'\midrule\endhead',
             r'\bottomrule\endfoot']
    lines += [' & '.join(esc(x) for x in row) + r' \\' for row in rows]
    lines += [r'\end{longtable}', r'\endgroup']
    (OUT / (name + '.tex')).write_text('\n'.join(lines) + '\n')


def main(require_complete=False):
    report = read_json(ROOT / 'analysis/report.json')
    if require_complete:
        assert not report['pending'], report['pending']
    OUT.mkdir(exist_ok=True)
    scores = read_csv(ROOT / 'analysis/scores.csv')
    cv = read_csv(ROOT / 'analysis/source_cv_folds.csv')
    contrasts = read_csv(ROOT / 'analysis/contrasts.csv')
    costs = read_csv(ROOT / 'analysis/prediction_costs.csv')
    for repeat in [0, 1]:
        d = scores[(scores.dataset == 'cluster') & (scores.split == f'repeat{repeat}') &
                   scores.seed.isin(['deterministic', 'seed_mean'])]
        rows = []
        for method in NAMES:
            a = d[d.method == method].set_index('domain')
            if not len(a):
                continue
            rows.append([NAMES[method]] + [pct(a.loc[x].macro_accuracy) if x in a.index else '--'
                                          for x in DOMAINS])
        table(f'cluster_r{repeat}', f'All revision Cluster target domains, training repeat {repeat}. '
              r'Values are macro accuracy (\%). S, D and L denote speed, direction and load shifts. ' +
              (r'ResNet34 is a within-specimen average of three seeds.' if repeat == 0 else r'No CNN R1 fit was planned.'),
              ['Method', 'ID', 'S', 'D', 'L', 'SD', 'SL', 'DL', 'SDL'], rows)
    rows = []
    for (dataset, split, method), d in cv.groupby(['dataset', 'split', 'method'], sort=False):
        d = d.sort_values('fold')
        folds = [pct(x) for x in d.macro_accuracy] + ['--'] * (5 - len(d))
        rows.append([('Cl. R' + split[-1]) if dataset == 'cluster' else 'Lima', NAMES[method],
                     f'{d.iloc[0].C:g}', f'{d.iloc[0].gamma_multiplier:g}', *folds, pct(d.iloc[0].cv_mean)])
    table('svm_cv', r'All selected SVM source-fold macro recalls (\%). Cluster folds hold out '
          r'30/40/50 mm/s, in that order. Lima folds are original trial index modulo five, 0--4. '
          r'Each row independently selected from the same nine $C,g$ pairs, with $\gamma=g/d$.',
          ['Set', 'Method', 'C', 'g', 'F0', 'F1', 'F2', 'F3', 'F4', 'Mean'], rows, 'llrrrrrrrr')
    d = scores[(scores.dataset == 'external') & (scores.domain == 'speed') &
               scores.seed.isin(['deterministic', 'seed_mean'])]
    if len(d):
        assert d.method.is_unique, 'External table must contain one pooled target row per method.'
        rows = [[NAMES[r.method], int(r.episodes), pct(r.macro_accuracy), pct(r.micro_accuracy),
                 f'[{pct(r.conditional_low)}, {pct(r.conditional_high)}]'] for r in d.itertuples()]
        table('external', r'All externally locked target methods at 40 mm/s. Macro/micro accuracy '
              r'and conditional specimen-bootstrap intervals are in percent. The interval resamples '
              r'twelve fixed specimen recalls, not 1,196 independent specimens.',
              ['Method', 'Episodes', 'Macro', 'Micro', 'Conditional interval'], rows, 'lrrrl')
    rows = []
    for r in contrasts.itertuples():
        rows.append([('Cl. R' + r.split[-1]) if r.dataset == 'cluster' else 'Lima',
                     NAMES[r.method_a] + ' - ' + NAMES[r.method_b], f'{r.difference_pp:.2f}',
                     f'[{r.conditional_low_pp:.2f}, {r.conditional_high_pp:.2f}]',
                     f'{r.specimens_improved}/{r.specimens_worsened}/{r.specimens_tied}'])
    table('contrasts', r'All revision speed contrasts (percentage points). $+/-/=$ counts specimens '
          r'with larger/smaller/equal recall. Limits are the 2.5th and 97.5th percentiles of conditional '
          r'specimen reweighting. Only Lima calibrated-minus-spatial is the locked primary. ' +
          primary_caption(report['external_primary_support']) +
          r'All other intervals are descriptive. Cluster repeat checks share specimens and targets.',
          ['Set', 'Contrast', 'Difference', 'Conditional interval', '+/-/='], rows, 'llrll')
    old = read_csv(OLD / 'analysis/scores.csv')
    d = old[(old.speed_error == 0) & (old.domain == 'speed')]
    rows = [[('1 s' if r.mode == 0 else '20 mm'), int(r.training_repeat), NAMES[r.method],
             pct(r.accuracy), f'[{pct(r.ci_low)}, {pct(r.ci_high)}]'] for r in d.itertuples()]
    table('original_controls', r'Every original source-selected speed-target model, including all '
          r'alternatives and the original constant-distance check. These are frozen v1 values. '
          r'The 20 mm check alone does not fix spatial Welch resolution.',
          ['Window', 'Repeat', 'Method', 'Accuracy (%)', 'Conditional interval (%)'], rows, 'lr lrl'.replace(' ', ''))
    signed = read_csv(ROOT / 'analysis/signed_speed_error.csv')
    rows = [[NAMES[r.method], f'{100*r.speed_error:+.0f}', pct(r.accuracy),
             f'[{pct(r.ci_low)}, {pct(r.ci_high)}]'] for r in signed.itertuples()]
    table('signed_errors', r'All signed nominal-speed perturbations for the frozen original R0 '
          r'models. The error changes only the supplied target speed. Zero is the unperturbed reference. '
          r'No refitting occurs; intervals retain the original conditional resampling.',
          ['Method', 'Speed error (%)', 'Accuracy (%)', 'Conditional interval (%)'], rows, 'lrrl')
    decision = read_json(ROOT / 'research/external_validity_decision.json')
    rows = [[c + 1] + [decision['eligible_by_class_speed'][f'{c:02d}_{v}'] for v in [30, 35, 40]]
            for c in range(12)]
    table('eligibility', 'Final external unique eligible episodes per texture folder and speed. '
          'The planned count in every cell was 100. Source speeds are 30/35; 40 is target.',
          ['Texture folder', '30 mm/s', '35 mm/s', '40 mm/s'], rows)
    cnn_cv, cnn_seeds = [], []
    for dataset in ['cluster', 'external']:
        p = ROOT / f'experiments/cnn/{dataset}/selection.json'
        if not p.exists():
            continue
        choice = read_json(p)['selected']
        cnn_cv.append([dataset, f'{choice["learning_rate"]:g}', choice['epoch'],
                       *[pct(x) for x in choice['fold_accuracy']], pct(choice['mean_cv_accuracy'])])
        q = ROOT / f'experiments/cnn/{dataset}/source_eval_audit.json'
        a = read_json(q) if q.exists() else None
        for seed in [20270913, 20270914, 20270915]:
            d = scores[(scores.dataset == dataset) & (scores.method == 'resnet34') &
                       (scores.seed.astype(str) == str(seed)) & (scores.domain == 'speed')]
            if not len(d):
                continue
            rec = next(x for x in a['records'] if x['seed'] == seed)
            r = d.iloc[0]
            cnn_seeds.append([dataset, seed, pct(rec['last_online_training_metrics']['accuracy']),
                              pct(rec['source_eval_metrics']['accuracy']), pct(r.macro_accuracy),
                              f'[{pct(r.conditional_low)}, {pct(r.conditional_high)}]'])
    table('cnn_cv', r'Selected CNN source fold accuracy (\%). The complete three-learning-rate '
          r'by six-epoch validation grid is preserved in each selection.json and epoch-logit archive. '
          r'Cluster holds out source speeds; Lima uses three episode-modulo folds.',
          ['Dataset', 'Learning rate', 'Epochs', 'F0', 'F1', 'F2', 'Mean'], cnn_cv)
    table('cnn_seeds', r'Every fixed CNN final seed (\%). Online training accuracy uses evolving '
          r'parameters and train mode. Source evaluation uses the sealed final model in eval mode. '
          r'These source diagnostics are not new selection criteria. The target column is specimen-macro accuracy.',
          ['Dataset', 'Seed', 'Online train', 'Source eval', 'Target', 'Conditional interval'], cnn_seeds, 'lrrrrl')
    align = read_csv(ROOT / 'analysis/alignment_summary.csv')
    rows = [[r.dataset, NAMES[r.method], f'{r.mean_within_specimen_speed_js_nats:.4f}',
             f'{r.between_specimen_centroid_js_nats:.4f}', f'{r.within_between_ratio:.3f}']
            for r in align.itertuples()]
    table('alignment', 'Exploratory coordinate-dependent spectral descriptors. Within is mean '
          'generalized speed Jensen--Shannon divergence within specimen; between is generalized divergence '
          'of specimen centroids. Both use natural logarithms. Their ratio has no validated decision threshold.',
          ['Dataset', 'Method', 'Within (nats)', 'Between (nats)', 'Ratio'], rows, 'llrrr')
    rows = [[r.dataset, r.split, NAMES.get(r.method, r.method.replace('resnet34_seed', 'CNN ')),
             int(r.episodes), f'{r.prediction_seconds:.3f}', f'{1000*r.prediction_seconds/r.episodes:.3f}']
            for r in costs.itertuples()]
    table('prediction_costs', 'Recorded batch-prediction timing, excluding raw download and feature extraction. '
          'Per-episode time is batch duration divided by count; it is not a real-time latency guarantee. '
          'CPU with two threads; no repeated timing benchmark or energy measurement was performed.',
          ['Dataset', 'Split', 'Model', 'Episodes', 'Seconds', 'ms/episode'], rows, 'lllrrr')
    manifest = {'inputs_sha256': INPUTS, 'script_sha256': digest(pathlib.Path(__file__)),
                'outputs_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sorted(OUT.glob('*.tex'))},
                'pending': report['pending']}
    (ROOT / 'analysis/supplement_table_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'tables': len(manifest['outputs_sha256']), 'pending': report['pending']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--require-complete', action='store_true')
    main(p.parse_args().require_complete)
