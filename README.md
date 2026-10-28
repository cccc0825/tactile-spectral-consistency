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

Use Python 3.12 in a separate environment. The dependency file records versions from the original study; availability and exact numerical behavior on another platform must be checked locally.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r outputs/research_v2/environment/requirements.txt
```

For source metadata initialization (standard library only):

```sh
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

The third command executes the guarded source-fitting and evaluation stages for the explicitly synthetic diagnostic, using its included frozen PSD fixture. Use a new run directory for every execution. Synthetic inputs and outcomes are not physical measurements.

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

See [docs/REPRODUCTION.md](docs/REPRODUCTION.md) for acquisition, isolated source fits, target-stage requirements and the distinction between fresh training and archived-result replay. The full raw acquisition and CNN retraining workflow is not claimed to have been rerun for this source release.

## Release validation

The release checks passed 15 pipeline unit tests, the calibration algebra check, all synthetic diagnostic stages, the paired specimen replay and overview generation. The diagnostic reproduced all 576 target predictions and 10 CSVs byte for byte. Initial import/font-cache timeouts and successful retries are summarized in [release_metadata/validation.json](release_metadata/validation.json). These checks do not constitute a new full physical-data/CNN experiment.

## Sources and license scope

Dataset creators, licenses and preprocessing attribution are in [THIRD_PARTY.md](THIRD_PARTY.md). Pretrained weights should be obtained from their original provider. This source release does not grant a new blanket license over third-party materials; the included Apache-2.0 notice applies to its own reference-code directory. No project-wide open-source license has yet been selected.
