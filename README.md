# Beyond Spectral Consistency in Source-Speed Calibration for Tactile Texture Recognition

Research code for analyzing the relationship between cross-speed spectral consistency and tactile texture recognition. The study connects response identifiability, finite observation, and downstream discrimination through a source-speed calibration operator, controlled synthetic signals, and recorded tactile datasets.

The repository name is **tactile-spectral-consistency**. All paths below refer to this repository only.

## Contents

- `outputs/research/experiments/`: Cluster acquisition, PSD extraction, original recognition pipeline and unit tests.
- `outputs/research_v2/experiments/`: response calibration, SVM comparisons, finite-observation diagnostics, external acquisition/preprocessing and CNN training.
- `outputs/research_v2/research/`: experiment protocols, settings and model-release checks.
- `outputs/research_v2/analysis/`: analysis and manuscript table/figure generation.
- `outputs/research_v2/revision_20260914/`: paired external specimen analysis and its recorded inputs/reference outputs.
- `outputs/research_v2/consistency_20260915/`: overview plotting script and compact source data.
- `work/revision/vendor/cluster-official/`: attributed reference code and its Apache-2.0 license.

Original relative paths are preserved because the research programs resolve inputs from this layout. The source manifest records file hashes. Raw sensor archives, trained checkpoints and Python environments are not stored here.

## Environment

Use Python 3.12 in a separate environment. The download audit was performed on macOS arm64; the acquisition code also requires `curl` and POSIX facilities (`fcntl` and `resource`). Native Windows execution has not been verified. The dependency file records versions from the original study; availability and exact numerical behavior on another platform must be checked locally.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r outputs/research_v2/environment/requirements.txt
```

Verify the downloaded files and initialize source metadata (standard library only):

```sh
python scripts/verify_release.py
python scripts/bootstrap_public.py
```

This first decompresses and SHA-256 checks the public ZIP member index, then creates local compatibility documents from public dataset citation metadata and records the current environment. It does not reproduce the historical environment receipt, download recordings or fit models. Existing files are preserved. It deliberately excludes webpage application/session state.

## Small runnable checks

From the repository root, with the scientific dependencies installed:

```sh
python outputs/research/experiments/test_pipeline.py
python outputs/research_v2/experiments/response_calibration.py
python outputs/research_v2/experiments/calibrator_diagnostic.py all \
  --run-dir outputs/research_v2/experiments/calibrator_diagnostic/run_reproduction_001
```

The third command executes the guarded source-fitting and evaluation stages for the explicitly synthetic diagnostic, using its included frozen PSD fixture. Use a new run directory for every execution. Synthetic inputs and outcomes are not physical measurements. Generate a table from that particular run with:

```sh
python scripts/make_synthetic_table.py \
  --run-dir outputs/research_v2/experiments/calibrator_diagnostic/run_reproduction_001 \
  --output-dir /tmp/tactile-synthetic-table-001
```

This adapter verifies the run files and computes the table from their predictions; it does not reuse historical acceptance records.

Recompute the paired external specimen analysis from the two included, original 1,196-row prediction files:

```sh
python outputs/research_v2/revision_20260914/tools/replay_specimen_influence.py \
  /tmp/tactile-specimen-replay-001
```

Generate the overview figure from its frozen source data:

```sh
python outputs/research_v2/consistency_20260915/tools/plot_overview.py
```

This renders an existing result; it does not run a new experiment.

## Recorded outcomes and scope

The finite synthetic comparison changes accuracy from 100.00% to 86.46% under ideal observations, and from 54.17% to 81.25% under the constructed low-pass observation. In the separate external tactile dataset, calibrated-minus-spatial macro recall is −0.66 percentage points with the original conditional interval [−6.49, 4.51]. These values describe the recorded study, not new results from cloning the repository.

The experiments concern held-out acquisition conditions on fixed specimen panels, with models fitted separately for the two datasets. Spectral consistency is evaluated separately from task effectiveness. The external interval spans gains and losses; the original benefit criterion was not met. The specimen influence analysis is post-hoc and retains every specimen.

## Full data and training workflow

See [docs/REPRODUCTION.md](docs/REPRODUCTION.md) for the entry points and [docs/FULL_STUDY.md](docs/FULL_STUDY.md) for the ordered acquisition, source audit, target evaluation and manuscript-analysis dependencies. Use [the fresh-result analyzer](docs/FRESH_RESULTS.md) to produce the latest specimen analysis and figures from a completed new run. Use `scripts/prepare_study.py` to build the full source-only working tree; the original 36-file helper remains a legacy source-selection subset. The full public-data workflow has now been executed in a separately prepared tree, including both datasets, all source fits, six final CNN models, sealed target predictions, controls and result generation. The recorded execution and comparison limits are in [the full-study validation receipt](release_metadata/full_study_validation.json).

## Release validation

A fresh public clone was used to reacquire and process 18,880 Cluster records and all 3,600 external episodes; the unchanged QC rule retained 3,550 external episodes. All 42 experimental/analysis stages completed, followed by the new-run supplement and figure adapters. The audit passed 1,325 integrity checks and 836 hash checks. All 239 available scientific artifact comparisons matched, including prediction outputs, main numerical macros and synthetic diagnostics. Five older locally offloaded reference files could not be read within bounded attempts; those old-file comparisons remain explicitly unverified, while their newly generated outputs exist. See [full_study_validation.json](release_metadata/full_study_validation.json) for the file list, commands, results and limitations.

The new external result is 12.5926% calibrated versus 13.25% spatial macro recall, with difference −0.6574 percentage points and conditional interval [−6.4932, 4.5068]. The original benefit criterion remains unmet. All specimens and outcomes were retained. The synthetic target predictions and recorded core numerical results were reproduced; this adds a reproducibility check, not new independent physical samples.

The run used Python 3.12 on macOS arm64, two CPU threads and the full pinned environment. CNN active time was 158.57 minutes. A separate six-hour reproduction allowance had been recorded because of variable CPU throughput; actual use remained below the historical 180-minute cap. The original protocols and spent time were preserved. Use [the resource instructions](docs/REPRODUCTION_RESOURCES.md) if a slower machine needs a separately disclosed allowance. The complete study tree occupied about 8.85 GiB, excluding the Python environment; reserve at least 20 GiB of free disk for acquisition and temporary files.

Earlier compact checks and their repaired failures remain in [validation.json](release_metadata/validation.json) and [download_validation.json](release_metadata/download_validation.json). New safeguards passed 15 fresh-analysis and 23 resource-ledger tests. Native Windows and other hardware/platforms have not been executed.

## Sources and license scope

Dataset creators, licenses and preprocessing attribution are in [THIRD_PARTY.md](THIRD_PARTY.md). Pretrained weights should be obtained from their original provider. This source release does not grant a new blanket license over third-party materials; the included Apache-2.0 notice applies to its own reference-code directory. No project-wide open-source license has yet been selected.
