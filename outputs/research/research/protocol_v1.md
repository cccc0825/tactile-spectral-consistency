# Frozen internal protocol v1 — motion-shift tactile spectra

Timestamp is stored in protocol_freeze.json. This is a local prospective record, not external preregistration. Written before any classifier fitting or held-out evaluation. Archive metadata and one training-domain acquisition (texture 0, direction 0, speed 40, load 500, repeat 0) were inspected solely for schema/timing. No accuracy or discriminative feature was examined.

## Question and scope

For closed-set recognition of the same 118 physical surface specimens, does physical speed canonicalization of accelerometer spectra improve held-out-speed recognition relative to an otherwise identical temporal-spectrum classifier? Does a source-selected partial warp or two-coordinate representation avoid the cost of forcing invariance? How do effects change under direction/load shifts?

This is an empirical signal-representation study. Time/speed normalization is established prior art, not a new invention. No claims of grasp improvement, human perception, unseen-specimen recognition, closed-loop safety or universal invariance are planned.

H1: full speed canonicalization improves speed-only domain-shift macro accuracy over temporal spectra by at least 3 percentage points. H1 is not supported if the paired specimen bootstrap 95% CI includes zero or the point effect is below 3pp. H2 (secondary): selecting warp strength only within source speeds reduces the cost of forced full invariance across domains. Null/negative findings are reportable. No method pivot based on final test performance.

## Data and units

Cluster Haptic Texture Dataset v5, figshare 29438288, archive file 61118716, CC BY 4.0. 118 specimens, 8 directions, 5 speeds, 2 normal-force levels and 2 recorded repeats = 18,880 acquisitions. Only acceleration and position are downloaded. The independent unit for paired descriptive bootstrap is the physical specimen, not window, sample, repeat, direction or seed. One specimen per named texture limits generalization; bootstrap is conditional on this specimen panel and trained models, not population-level proof.

Use all acquisitions passing predeclared integrity checks. Parse original timestamps; never assume rows are perfectly uniformly spaced. Require finite values, strictly increasing time after merging exact duplicate timestamps by averaging, no gap >20 ms intersecting the selected window, at least 1.1 seconds available, all three axes present. Record every exclusion and its cause. If >1% acquisitions excluded or a class/domain is missing, stop formal evaluation and inspect rather than silently changing the set.

Use the central 1.0-second interval of each acquisition, away from start/stop. Positions will audit whether the central interval is in motion; flag median central speed <80% of commanded speed or any long stop. Velocity labels are commanded metadata known to a motion controller. They are allowed inputs for canonicalization. No label is inferred from held-out outcomes. Position signals are used for data-quality audit only in the primary experiment.

Uniformly interpolate timestamped acceleration at 6000 Hz over the center interval; linearly interpolate after merging duplicate times. Analysis is restricted to at most1200Hz, below the nominal Nyquist. Do not treat interpolation as recovery of lost high-frequency content. Remove each axis mean and use Welch PSD with Hann window, nperseg=1024, noverlap=512, detrend=constant, scaling=density; sum the three axis PSDs for an orientation-tolerant power spectrum.

## Fixed acquisition-level split

Source conditions: velocities {30,40,50} mm/s, directions {0,90,180,270} degrees, load 500 mN. Primary training uses only repeat 0 (1416 acquisitions). Hyperparameters are chosen by leave-one-source-velocity-out CV within these training acquisitions (3 folds), using macro accuracy, then refitted to all primary training data. No repeat-1 or target-domain data enter selection, normalization, feature fitting or thresholds.

Formal evaluations:
- ID repeat holdout: source conditions, repeat1.
- Speed only: velocities{20,60}, source directions, load500, both repeats.
- Direction only: source velocities, directions{45,135,225,315}, load500, both repeats.
- Load only: source velocities, source directions, load1000, both repeats.
- All three shifts: velocities{20,60}, diagonal directions, load1000, both repeats.
- Remaining mixed shifts (speed+direction, speed+load, direction+load) are descriptive supplemental analyses.

Acquisition IDs are mutually exclusive across training and evaluation; no segment augmentation crosses a split. Repeat the full selection/training/evaluation with repeat1 training and repeat0 ID holdout as a predefined acquisition-repeat robustness check. These are two correlated experiments, not independent datasets. Primary hypothesis uses the repeat0-trained model; replicate is reported separately.

## Methods and equal budgets

Reference speed v0=40 mm/s. 128 logarithmic bins have temporal reference edges 40…800Hz. For warp strength beta, edges for acquisition v are multiplied by (v/v0)^beta. beta=0 gives temporal; beta=1 gives spatial-frequency bins1…20cycles/mm. Across all prescribed velocities/fixed beta values, edges remain inside20…1200Hz, avoiding unsupported high-frequency extrapolation. Integrate PSD over bin edges, normalize bin masses by total mass and take log with fixed1e-12 floor. StandardScaler is fitted only on each training fold.

Core representations:
1. Temporal (beta0).
2. Spatial (beta1).
3. Partial warp, beta in {0,0.25,0.5,0.75,1}, selected on source CV.
4. Dual-coordinate concatenation of temporal and spatial features, with each block divided by sqrt(2).
5. Temporal plus metadata (velocity/v0, sin(direction),cos(direction),force/500); explicit metadata comparator, no material-dependent motion identity.
6. Log-mel baseline:128 triangular mel bands20…1200Hz applied to the same PSD, log normalized masses, same classifier. This is a compact PSD baseline inspired by published log-mel processing, not an exact reproduction of the publisher's resized spectrogram/CNN pipeline.

All use RBF SVC, C in {1,10,100}, gamma multiplier in {0.25,1,4} divided by input dimension after standardization. Nine classifier choices per fixed representation. Partial warp has45 joint candidates; to make the extra representation search explicit, report its entire search table and compare an equally selected union (temporal versus spatial selected by same source CV) with18 candidates. No claim of equal total search count for the partial selector. Tie-break by lower C, then gamma closest1, then lower beta. Record all CV results, fit/predict wall time, support-vector counts and memory.

Mechanism/robustness controls, fixed before test:
- Wrong/constant speed control is the temporal beta0 representation (no actual speed adjustment).
- Speed errors: evaluate the frozen spatial and partial classifiers using speeds perturbed by ±5% and ±10%; source training remains unchanged. These are explicitly synthetic metadata perturbations to real acquired signals.
- Observation length: rerun fixed beta0 and1 under a central constant spatial span20mm (duration20/v seconds) using same native timestamp-resampling and PSD rules, then source-only hyperparameter selection. This separates duration from distance, without claiming equal information content.
- Gain ablation: add log total band power to temporal/spatial representations (no manually tuned amplitude normalization).
- Low/high held-out speeds and per-category/per-texture paired errors are descriptive, including all failures.

## Statistics and stopping

Primary endpoint: macro accuracy (mean per-specimen recall) on speed-only target, spatial minus temporal, primary training repeat0. Report paired10,000-resample specimen bootstrap95% percentile interval using seed20270912; include 3pp practical threshold. No significance testing of technical repeats. Secondary intervals are descriptive, not family-wise discoveries. Also report all-domain macro accuracy, per-specimen confusion, pooled accuracy, inference milliseconds/acquisition and storage.

Before the first formal target inference, save source-selected parameters and hashes of protocol, code, feature manifest, splits and chosen models. Formal results cannot be used for method choices. Failures are appended. Stop at90minutes formal CPU time,3GB single-process memory or integrity failures; use at most2 compute threads. Downloads may use4 I/O workers with6GB total acquisition budget and no paid resources. No unbounded retries or result-driven additional seeds.

## Deliverables and quality gate

Reproducible code, checksums, raw byte ranges, extracted PSD/features and acquisition quality logs, exact split/CV tables, all predictions, derived tables/figures,20-ish verified references (17–23 total), editable English LaTeX, original2027 style files and exactly4+1-page PDF. Full research draft status requires actual completed core experiments and evidence-constrained claims. A high conference-quality/novelty verdict is separate and may remain negative.
