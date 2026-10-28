# Reproduction entry points

Run the root README's installation, file verification and bootstrap commands first. Python 3.12/macOS arm64 was tested. Acquisition uses `curl` and POSIX facilities; native Windows is untested. The source archive does not contain raw recordings or trained weights.

## Compact workflows

The root README runs the following without downloading sensor recordings:

- 15 pipeline unit checks and the calibration algebra check.
- The complete synthetic diagnostic, using the included, explicitly synthetic PSD fixture.
- `scripts/make_synthetic_table.py --run-dir RUN --output-dir NEW_DIRECTORY`, which verifies that particular completed diagnostic and recomputes its six classification summaries from saved predictions.
- Paired external specimen analysis from the two included original prediction CSVs, with 1,196 rows each.
- The overview figure from its frozen numerical source tables.

Use a new diagnostic and output directory for each execution. The paired replay and overview do not retrain a classifier or add independent observations. The synthetic table adapter records numerical verification, not independent peer review. The archived `analysis/make_calibrator_artifacts.py` remains bound to the historical study; use the new adapter for a fresh diagnostic run.

## Full source-only working tree

Do not run a fresh physical-data study in the release root: it contains published external predictions for the compact replay. Prepare a separate tree instead:

```sh
python scripts/prepare_study.py prepare --project-root /tmp/tactile-study-001
python scripts/prepare_study.py check --project-root /tmp/tactile-study-001
```

The new wrapper retains the original source-selection helper, copies the additional analysis/audit producers, and relocates published reference outputs outside active experiment paths. It never copies historical acceptance receipts or trained weights. `check` validates prepared source hashes; it does not certify data, training or target evaluation.

Continue with [FULL_STUDY.md](FULL_STUDY.md). It gives the ordered Cluster and external acquisition, source selection, audit, sealing, target evaluation and table-generation steps. The missing workbook is now acquired with `python scripts/fetch_texture_metadata.py` inside the fresh tree: only 16,618 bytes are downloaded, and the resulting public workbook is checked against its original SHA-256, CRC and size. This metadata is needed by the original v1 analysis, which feeds the later tables.

Each new source selection must pass its own technical checks before external target inference. The local receipt tool runs the included audit producers, checks their outputs and bindings, and refuses pre-existing target predictions. It does not convert historical acceptance into fresh evidence or constitute an independent scientific review. Keep the original split, tuning budget and benefit criterion; retain negative outcomes.

The original `outputs/research_v2/tools/prepare_fresh_run.py` is a legacy 36-file source-selection subset, not the complete analysis tree. Use the new wrapper for full-study preparation.

## Archived-package adapter

`outputs/research_v2/tools/portable_reproduce.py` supports `--level tables` and `--level svm` for the separate complete archived experiment package. It requires all archived predictions/manifests and, for SVM inference, PSDs and saved models. That large package is not in this repository. The adapter deliberately rejects missing or mismatched inputs; it is not the entry point for the compact source download.

## Verification boundary

The download audit installs the published requirements in a new environment and executes the compact workflows. Full raw acquisition and physical-data/CNN retraining require additional network, disk and compute; they were not rerun for this release. The full-study instructions and gate tests establish available dependencies and fail-closed behavior, not an already completed clean-machine training experiment.

Original study files and protocol hashes are retained in `release_metadata/source_inventory.json`. `release_manifest.json` covers the current published bytes, including packaging additions. Packaging adaptations and the latest downloaded-source checks are recorded separately. The research hypotheses, test-exposure chronology, numerical results and protocol freezes are unchanged.
