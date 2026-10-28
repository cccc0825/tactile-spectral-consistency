# Source-selected ResNet34 baseline protocol

Frozen: 2026-09-12T16:27:52Z. Formal fits performed before this protocol: **0**. This protocol follows an outcome-blind synthetic resource benchmark and precedes all source cross-validation. Cluster results are exploratory revision evidence because the v1 target outcomes have already been inspected in the overall project. They cannot retrospectively become a confirmatory test.

## Provenance and reproduction boundary

The official Cluster authors' repository is `https://github.com/cluster-lab/Cluster-Haptic-Texture-Dataset`, Apache-2.0, commit `e05d6b022d127e24f73583146f0aa229c6934449`. Read-only source inspection covers `train/classification/ResnetClassification.py`, `train/datasets/accel_dataset_mel.py`, and `train/haptics_texture_classification.py`; their individual hashes are frozen in the selection context. The implementation here is an independently written protocol adaptation of the authors' ResNet34 plus acceleration log-mel baseline. The released random-split benchmark accuracy is not reproduced or imported.

ImageNet initialization is torchvision `ResNet34_Weights.IMAGENET1K_V1` (also the ResNet34 `DEFAULT` in torchvision 0.23.0), fetched from `https://download.pytorch.org/models/resnet34-b627a593.pth`. Full SHA-256: `b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f`. All three original pretrained convolution input channels are retained and the final fully connected layer is replaced with 118 outputs for Cluster. The authors instantiate pretrained ResNet34 but replace its first convolution with a randomly initialized layer even when input channels equal three. Retaining that pretrained convolution is a deliberate documented adaptation. All parameters are trained. No claim of an exact author-code reproduction is permitted.

## Frozen acquisition and preprocessing protocol

All 18,880 accelerometer ZIP entries are decoded from the preserved HTTP archive ranges, with per-entry uncompressed length and CRC32 checked. Each raw CSV receives a SHA-256 record. The complete factorial design is 118 textures × 8 directions (0:45:315 degrees) × 5 speeds (20, 30, 40, 50, 60 mm/s) × 2 loads (500, 1000 mN) × 2 repetitions. Row order and acquisition labels must exactly equal v1's processed labels. An invalid acquisition stops extraction; no condition-dependent exclusion is allowed.

For each acquisition, average all acceleration axes at duplicate timestamps; reject nonfinite or decreasing timestamps, spans shorter than 1.1 s, and central timestamp gaps exceeding 20 ms. Use the same central one-second interval as v1, from the midpoint of the first/last unique timestamps. Interpolate each axis linearly to exactly 6,000 samples on the grid `center − 0.5 + arange(6000)/6000` seconds. No raw full-acquisition arrays are retained.

Compute scipy STFT independently for X/Y/Z: 6 kHz sample rate, periodic Hann window, 1,024-point FFT, 256-sample hop, per-frame constant detrending, no boundary extension and no end padding. Use squared complex magnitude as power. Project to 128 unnormalized triangular HTK mel bands spanning 20–1,200 Hz. For each axis separately, reference the maximum power across its mel/time array, convert by `10 log10`, clip to [−80, 0] dB, and map by `dB/40 + 1` to [−1, 1]. Resize to 128 × 64 with PyTorch bilinear interpolation, `align_corners=False`, `antialias=False`. Store inputs as float16 `.npy`; convert to float32 when loading a batch. No population-fitted preprocessing, augmentation, flips, erasing, ImageNet color normalization, or target-dependent transform is applied.

The authors instead use nominal 5 kHz nearest/searchsorted resampling, whole recording handling with a minimum one-second padded duration, 512-point librosa STFT/128-hop mel processing, and random acquisition splits. Their transforms include horizontal/vertical flips and random erasing for neural training. The new preprocessing fixes the one-second observation budget and source-condition split to match this research question. These choices, and the bounded training horizon below, prevent numerical comparison with the published benchmark from being interpreted as reproduction success/failure.

## Source-only selection and final fits

Cluster source training uses repetition 0 only at speeds 30/40/50 mm/s, cardinal directions 0/90/180/270 degrees, and 500 mN load: 1,416 acquisitions, 12 per texture. Source cross-validation leaves one of the three source speeds out at a time. Every fold has 944 training and 472 validation acquisitions, with all 118 classes present. No other acquisition enters a fitting or selection call.

Use full ResNet34 fine-tuning, cross-entropy loss, Adam with weight decay 0.0001, batch size 16, constant learning rate, no scheduler, and exactly six epochs per source fold. The learning-rate candidates are 0.0001, 0.0003, and 0.001. All nine fold/candidate runs use seed 20270913 for model initialization and batch permutations, with deterministic PyTorch algorithms requested. There is no score-dependent early stopping. Select one learning rate and one epoch (1–6) by the largest mean accuracy over the three source folds; ties resolve to the earlier epoch, then smaller learning rate. Source folds are balanced across texture, so accuracy equals macro recall. Preserve every candidate's per-epoch validation logits and metrics.

After completing the entire source grid, independently refit the selected configuration on all 1,416 source acquisitions using the three fixed seeds 20270913, 20270914, and 20270915. Each refit starts from the same ImageNet archive, with a fresh classifier initialization. All three final model files, the complete source ledger, configuration, code, data, preprocessing manifest, and environment must be hashed and sealed before any target prediction. A missing or failed required final model blocks the target phase. Failed runs and partial state are retained; they cannot silently disappear from the planned matrix.

## Evaluation and reporting

After the freeze, evaluate each of the three frozen models on all 17,464 remaining Cluster acquisitions. Domains match v1 exactly: ID 1,416; speed 1,888; direction 2,832; load 2,832; speed+direction 1,888; speed+load 1,888; direction+load 2,832; speed+direction+load 1,888. The three deviations mean respectively speed outside {30,40,50}, direction outside {0,90,180,270}, and load other than 500 mN. Save acquisition-level predictions and logits for every seed, including all failures. Report every seed and the arithmetic seed mean; seeds measure training stochasticity and are not independent data samples for significance claims. No target result can change preprocessing, initialization, learning rates, epoch choice, inclusion rules, or seeds.

## Compute amendment, checkpoints, and execution

The provisional delegated plan was at most 15 epochs and a 90-minute shared cap. The synthetic benchmark used no real data or labels: CPU with two threads, float32 ResNet34, batch 16 × 3 × 128 × 64, Adam, and squared-logit loss. Five steps took 0.7185, 0.5458, 0.5406, 0.5233, and 0.5283 s. Steady mean was 0.5345 s and peak RSS 703,692,800 bytes. MPS was built but unavailable to the process. The benchmark indicated that the original complete grid, worst-case final refits, and evaluation would exceed the provisional budget. Before any formal fit, the parent made the outcome-blind amendment to **six epochs per fold and 150 minutes cumulative active compute for Cluster plus external CNN runs**. This preserves all three learning rates, all source folds, and all three final seeds.

The 150-minute ledger counts source training, validation, final refits, target inference, initialization and checkpoint I/O within those phases. It excludes acquisition download/extraction and the synthetic resource benchmark. Enforce two CPU and two inter-op threads, a 3,000,000,000-byte peak process RSS ceiling, batch 16, and no paid/external compute. Stop and record missing planned runs if a budget or integrity limit is exceeded. Do not adapt based on target accuracy.

Available disk at planning was about 13 GiB. Preserve an exact float32 terminal model checkpoint for each of nine CV runs and each of three final refits (about 1 GiB total), all epoch logs, and all epoch validation logits. A failure checkpoint includes the optimizer state. Successful CV checkpoints represent the terminal sixth epoch, not necessarily the subsequently selected epoch; final refits train afresh to the selected epoch. No successful optimizer-state archive or redundant full-model checkpoint for every epoch is retained. No target checkpoint is selected after evaluation.

Entry points, from the workspace root, using `work/research/venv/bin/python`:

```text
outputs/research_v2/experiments/extract_cluster_spectrograms.py
outputs/research_v2/experiments/cnn_baseline.py configure-cluster
outputs/research_v2/experiments/cnn_baseline.py select
outputs/research_v2/experiments/cnn_baseline.py evaluate
```

The source and target phases are separate commands. Re-running a completed extraction, configuration, freeze, checkpoint, or target-output directory is refused. Source runs with already-complete, unchanged terminal checkpoints can be reused after an interruption; a failed/incomplete run requires explicit recorded recovery. The shared ledger cannot be silently reset.

## External dataset interface, pending its separate freeze

The same trainer accepts `--config PATH` and a frozen three-channel 128 × 64 input `.npy` plus an acquisition metadata CSV containing unique `id`, integer `class_id`, binary `source`, source `cv_fold`, and `domain` columns. All required file hashes and dataset counts are supplied in its config. The external dataset/config is not yet available at this freeze and no external fitting is authorized by this document alone.

The intended external outer split is source speeds 30/40 versus untouched target speed 45 mm/s, with three inner CNN folds using source episode-index modulo 3. The SVM external plan may use five modulo-5 folds; these are separate tuning budgets sharing an outer test, and must not be described as identical search protocols. The parent must supply the external extraction/schema and freeze its protocol before calling the trainer. Every external seed model must be sealed before any external target prediction. The shared active compute cap remains 150 minutes.
