# Fresh full-study dependency map

This workflow prepares every published source dependency, acquires public inputs, and separates source fitting, technical audits, target release and analysis. It is a procedure for a **new study run**. A public-clone execution completed all stages on Python 3.12/macOS arm64; its [validation receipt](../release_metadata/full_study_validation.json) records the tested environment, resource allowance, 239 matching scientific artifact comparisons and five inaccessible historical reference files. Numerical equivalence across other environments must still be checked; failed checks remain failures. Source/target settings and the original numerical producers are unchanged.

Use Python 3.12, the recorded dependencies, a macOS/Linux environment with `curl`, sufficient disk for multi-GB recordings/tensors/checkpoints, and the documented compute limits. CUDA training is not implemented by this trainer: it selects Apple MPS when available, otherwise CPU. A slower machine may hit a frozen time or memory limit. Preserve any failure. If source-only throughput indicates that more local CPU time is needed, use the separately recorded [reproduction resource allowance](REPRODUCTION_RESOURCES.md) before target evaluation; a run exceeding 180 minutes is not a reproduction of the historical runtime bound.

## 1. Prepare all sources in a new directory

From the downloaded repository, after creating the Python environment and installing its requirements:

```sh
python scripts/test_study_guards.py
python scripts/prepare_study.py prepare --project-root /tmp/tactile-full-study-001
python scripts/prepare_study.py check --project-root /tmp/tactile-full-study-001
cd /tmp/tactile-full-study-001
```

The wrapper first verifies the published manifest, then invokes the original 36-file source helper and copies the remaining published sources/configuration. It runs the copied metadata bootstrap in the new root. `FRESH_RUN.json` continues to bind the unchanged 36-file subset; `STUDY_SETUP.json` binds the additional files. Never edit frozen inputs in place.

The two published target-prediction CSVs, old analysis report, specimen-analysis directory and overview directory are relocated under `reference_release/`. They cannot satisfy a fresh experiment's input or target gate. Use the **original downloaded repository** for the compact saved-result replay commands in the README. No historical audit PASS, acceptance receipt, model or target release is copied into this fresh study tree.

Run all remaining commands from the new tree with the same environment active. In the tables below, a source-helper stage means:

```sh
python outputs/research_v2/tools/prepare_fresh_run.py run \
  --project-root /tmp/tactile-full-study-001 --step STAGE
```

Replace `STAGE` with one table entry at a time, in order. Each call records its own log, exit code and elapsed time. Do not treat a failed or `needs_qc_disposition` receipt as successful extraction.

## 2. Cluster acquisition and source fits

| Order | Source-helper stage / command | Dependency produced or checked |
|---|---|---|
| 1 | `acquire-cluster` | Public accelerometer and stage-position ranges |
| 2 | `extract-cluster` | PSD arrays, labels, quality and extraction manifest |
| 3 | `python outputs/research/experiments/verify_data.py` | Range/cache hashes, count and duplicate checks |
| 4 | `python outputs/research/experiments/audit_positions.py` | Recorded stage-position diagnostics; no exclusions |
| 5 | `extract-fixed-space` | Spatial-resolution control PSDs |
| 6 | `select-cluster-svm` | Source-selected calibration/comparator models and freeze |
| 7 | `extract-cluster-cnn` | CNN tensors/labels/manifest |
| 8 | Download the initialization below | Original ResNet34 initialization |
| 9 | `configure-cluster-cnn` | Hash-bound tensor/split/training configuration |
| 10 | Defer `select-cluster-cnn` until section 3 configuration is complete | Both configurations must exist before CNN training; the external configuration initializes the shared 180-minute allowance |

Download the original initialization into this fresh tree, then verify the entire SHA-256 before configuration:

```sh
curl --fail --location --output outputs/research_v2/experiments/cnn/resnet34-b627a593.pth \
  https://download.pytorch.org/models/resnet34-b627a593.pth
python outputs/research_v2/tools/prepare_fresh_run.py check \
  --project-root /tmp/tactile-full-study-001
```

The resulting preflight receipt must report `imagenet_weights.hash_matches: true`. Expected SHA-256: `b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f`. The source helper prints its receipt path. Read that receipt; its generic `source_setup_passed` status alone does not establish that weights or datasets exist.

After both source CNN selections in section 3, and before any target evaluation, compute the Cluster saved-model source diagnostic:

```sh
python outputs/research_v2/experiments/cnn/source_eval_audit.py \
  --config outputs/research_v2/experiments/cnn/cluster_config.json
```

## 3. External acquisition, QC and source fits

| Order | Source-helper stage / command | Dependency produced or checked |
|---|---|---|
| 1 | `acquire-external` | Public Mendeley IMU range data and logs |
| 2 | `smoke-external` | Restricted input-decoding/preprocessing checks |
| 3 | `extract-external` | Full QC; the recorded panel stops for QC disposition |
| 4 | `external-qc-decision` | New hashes under unchanged 49-invalid/one-duplicate policy |
| 5 | `publish-external` | New common eligible PSD/CNN arrays |
| 6 | `external-handoff` | New metadata/label/source-fold handoff |
| 7 | Handoff extension command below | Per-class source-fold counts required by the original data audit |
| 8 | `configure-external-cnn` | New external configuration; same recorded compute addendum |
| 9 | `select-external-svm` | Nine source-only SVM selections and models |
| 10 | Run `select-cluster-cnn`, then `select-external-cnn`, serially | Source CV and all three final CNN models for each dataset |

Complete both CNN configurations before step 10. For a slower machine, the optional [resource allowance command](REPRODUCTION_RESOURCES.md) belongs after step 8 and before either CNN selection, with no running ledger writer. The two selections and later source audits/evaluations share one ledger and must run serially.

Execute step 7 **before** external CNN configuration:

```sh
python scripts/prepare_study.py augment-handoff \
  --project-root /tmp/tactile-full-study-001
```

This adds source per-class counts to `external_split_preflight.json`, after matching its existing source IDs, folds and counts against the new metadata. The original preflight is preserved as `external_split_preflight.source_subset.json`. No split, feature, model or outcome changes. Running it after CNN configuration is rejected because the configuration already binds its dependencies.

The data and SVM audits require the saved SVM split, so execute them after step 9. They may also be run after step 10, before any external target command:

```sh
python scripts/accept_source_audits.py data
python scripts/accept_source_audits.py svm
python outputs/research_v2/experiments/cnn/source_eval_audit.py \
  --config outputs/research_v2/experiments/cnn/external_config.json
python outputs/research_v2/experiments/cnn/verify_external_outputs.py source
```

`accept_source_audits.py` executes the two original audit programs; it does not accept pre-existing audit results. The data audit reparses 3,600 raw episodes, checks acquired range hashes and the common eligibility mask, and replays 36 spectral samples. It includes target recording QC/metadata, but does not read external classification outcomes. The SVM audit repeats 405 source CV fits and nine final source fits, checking the saved parameters bitwise; no target row enters its feature fitting or classification. Original audit numerical/coverage limits remain unchanged.

Successful commands create **new local technical receipts**, with the actual subprocess command, exit code, timing, output hashes and unchanged input bindings. They are not independent peer review or a statistical performance claim. Historical command strings embedded in the unchanged original audit reports are provenance text; the `.execution.json` and acceptance receipt contain the command actually executed now. The wrapper fails closed on an existing audit output, changed input, nonzero exit, missing check, target output/release, or previous external-evaluation budget entry. Preserve any failed attempt; investigate before starting a separately named fresh tree. The wrapper cannot prove that someone has never viewed the public paper's existing results; the new run is a reproduction, not a first unseen-test confirmation.

## 4. Freeze, release and execute target evaluation

All data/source audits above must succeed. Inspect their new receipts and keep all outcomes, including losses. If a reproduction resource allowance was applied, run its `check --pre-target` command now and preserve the output before writing the global seal. The original global seal is still required:

```sh
python outputs/research_v2/research/seal_external_models.py
python outputs/research_v2/research/seal_external_models.py --write
```

The first command verifies readiness without writing. The second writes a new global freeze only after verifying both source selections, all model/code/data hashes, new data/SVM receipts, CNN source evaluation and integrity checks, and the compute budget. A missing or failed requirement stops the gate; never create an empty JSON file as a substitute for this freeze. Inspect `research/external_global_model_freeze.json` and then run each target producer once:

```sh
python outputs/research_v2/experiments/run_calibration.py cluster evaluate
python outputs/research_v2/experiments/cnn_baseline.py evaluate
python outputs/research_v2/experiments/run_external_calibration.py evaluate
python outputs/research_v2/experiments/cnn/run_external_baseline.py evaluate \
  --global-release outputs/research_v2/research/external_global_model_freeze.json
python outputs/research_v2/experiments/cnn/verify_external_outputs.py complete
```

The SVM entrypoint checks its source freeze; the guide's global-seal step must precede both external evaluations. The old `run_calibration.py external` loader is not the final external split and must not be substituted. None of these target commands authorizes tuning on the resulting scores or dropping a specimen.

## 5. Additional original controls and analysis dependencies

The v2 manuscript generators also refer to v1 controls and estimator diagnostics. They are separate from the primary external comparison and must not be silently replaced by hand-entered values.

After Cluster extraction, integrity and position checks, run the original v1 controls in the same fresh tree:

```sh
python scripts/fetch_texture_metadata.py
python outputs/research/experiments/run_experiment.py select
python outputs/research/experiments/run_experiment.py evaluate
python outputs/research/experiments/posthoc_grid.py
python outputs/research/analysis/analyze.py
python outputs/research/analysis/posthoc_analyze.py
```

The public workbook getter verifies the used ZIP member and produces `outputs/research/corpus/texture_list.xlsx`, required for category metadata. The v1 experiment retains its original **90-minute persistent wall-clock** budget across selection/evaluation/post-hoc work; execute that sequence without an unrelated lengthy pause, and preserve a timeout as a failed run. Running the source-selection phase does not grant permission to change frozen hyperparameters after viewing its evaluation. The grid is explicitly post-hoc and remains so in any new report.

Run the remaining estimator controls and analysis in this order, after the relevant data and target outputs exist:

```sh
python outputs/research_v2/experiments/estimator_diagnostics.py real
python outputs/research_v2/experiments/estimator_diagnostics.py synthetic
python outputs/research_v2/analysis/analyze_revision.py --require-complete
python outputs/research_v2/analysis/alignment_diagnostic.py
python outputs/research_v2/analysis/make_artifacts.py --require-complete
python outputs/research_v2/analysis/make_supplement.py --require-complete
```

`real` produces `real_selection.json` and `real_results_manifest.json`; `synthetic` produces `synthetic_results.json`. They are required by `analyze_revision.py`. `alignment_diagnostic.py` produces `alignment_summary.csv`; `make_artifacts.py` also produces `signed_speed_error.csv` required by the supplement. Source-evaluation diagnostics for **both** CNN datasets must have been generated before the supplement's seed table. All results from these new executions are new-run outputs; compare with, but never overwrite, the original published references.

## 6. Calibrator simulation and new-run table generation

The finite-observation calibrator simulation remains a separate explicitly synthetic chain:

```sh
python outputs/research_v2/experiments/calibrator_diagnostic.py all \
  --run-dir outputs/research_v2/experiments/calibrator_diagnostic/run_001
```

Use the frozen included fixture for exact recorded-fixture replay (README command in the original repository). The `estimator_diagnostics.py synthetic` command above regenerates that fixture in the fresh study tree; the diagnostic verifies its frozen hash and stops if this regeneration differs. Such a mismatch requires investigation, not a hash update. The simulation writes its own `evaluation/summary.csv`, `evaluation/summary_table.tex`, test records and result seal.

Generate a standalone table from this completed new simulation with the public adapter:

```sh
python scripts/make_synthetic_table.py \
  --run-dir outputs/research_v2/experiments/calibrator_diagnostic/run_001 \
  --output-dir /tmp/tactile-synthetic-table-001
```

Use a new output directory and retain the adapter's input/result verification records. This table comes from the actual completed run; no methodological acceptance is invented or required merely to render its recorded values. Review the simulation's protocol, truth-alignment boundaries and statistical scope before using its results in scientific conclusions.

The historical `analysis/make_calibrator_artifacts.py` and `tools/check_calibrator_clean_replay.py` are archived-study adapters with historical paths and receipt dependencies. They are not the new-run table entrypoint and should not be made runnable by copying an old acceptance receipt into this tree.

## 7. Current-run specimen analysis and figures

After the complete external target stages and the synthetic run above, generate the latest specimen analysis and overview from this run's sealed predictions:

```sh
python scripts/test_fresh_analysis.py
python scripts/analyze_fresh_results.py \
  --project-root /tmp/tactile-full-study-001 \
  --synthetic-run-dir /tmp/tactile-full-study-001/outputs/research_v2/experiments/calibrator_diagnostic/run_001 \
  --output-dir /tmp/tactile-fresh-figures-001
```

[Fresh-result details](FRESH_RESULTS.md) describe the exact bindings, paired bootstrap, complete specimen retention and new outputs. The figure adapter checks the external SVM targets it actually uses; its success does not replace CNN target completion or the other full-study stages. A different sign or magnitude is retained and reported. Do not copy historical plotting tables into fresh result directories.

This repository publishes the research programs and compact evidence, not a complete historical archive or the paper/template sources. `tools/portable_reproduce.py` still requires the separate full archived prediction/model package. Successful preparation or guard tests alone do not establish full experimental reproduction, independent review, or a submission-ready manuscript.
