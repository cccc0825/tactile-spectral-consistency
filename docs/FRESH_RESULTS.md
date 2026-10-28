# Figures and specimen analysis for a newly completed run

Use this step after the full-study workflow has completed its fresh source selections, audits, global source seal and external target evaluation. It analyzes the new predictions; it does not fit models, change the benefit rule or provide an additional independent validation sample.

The compact README replay remains the entry point for the original published prediction files. The fresh entry point below rejects `reference_release/` inputs and an incomplete or historical run, and writes to a new directory.

```sh
python scripts/test_fresh_analysis.py
python scripts/analyze_fresh_results.py \
  --project-root /tmp/tactile-full-study-001 \
  --synthetic-run-dir /tmp/tactile-full-study-001/outputs/research_v2/experiments/calibrator_diagnostic/run_001 \
  --output-dir /tmp/tactile-fresh-figures-001
```

Run the command from the downloaded repository or from its prepared full source tree, with the same scientific environment active. Replace the example project/run paths with those of the actual new experiment. The synthetic run must have been initialized in that tree after its setup and completed all four diagnostic stages. Choose a new output directory outside sealed experiment directories.

The verifier checks:

- The full-study source manifest, new setup chronology, global source seal, evaluated SVM selection freeze and all nine external evaluation records.
- Every hash in the global source inventory and SVM input/model inventory. The processed metadata, eligibility decision and source/target split must agree.
- Exactly 1,196 paired target episodes, all 12 physical specimens, acquisition identities, truth labels, correct fields and unchanged source/target membership.
- The synthetic diagnostic's source/result seals, completed stage records and prediction-derived metrics, through `make_synthetic_table.py`.

Outputs include `per_specimen_effects.csv`, `summary.json`, a specimen table, overview and influence figures in PDF/SVG/PNG, `figure_source_data.json`, and a hash manifest. The conditional interval is recalculated from the current predictions using the original paired physical-specimen scheme: 10,000 percentile draws, seed 20270913. The original benefit rule remains a difference of at least 3 percentage points and a strictly positive conditional lower bound. It is reported without tuning or revising the rule.

Every specimen is retained, in ID order. Leave-one-specimen-out values are post-hoc influence diagnostics and do not justify exclusion. Improvement/decline/tie counts are computed from the supplied run; the original 6/3/3 counts or negative mean are not required. Plot limits expand to display every specimen effect and the all-specimen interval.

A successful export establishes artifact integrity and descriptive computation, not independent peer review, new-instance generalization, a causal explanation, or an externally preregistered study. The guard tests use artificial temporary records solely to check validation behavior; those test records are never presented as observations.

The supplementary contrast caption also follows the current `external_primary_support` value in the new analysis report. A missing comparison is labeled unavailable; a positive criterion is not relabeled as failure. The original archived paper and stored result files are unchanged.
