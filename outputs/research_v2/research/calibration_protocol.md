# Source response calibration extension

Recorded 2026-09-13 Asia/Shanghai, before fitting this extension. This is a local time-stamped protocol, not an external preregistration. The full Cluster v1 test results were already known. All new Cluster analyses are consequently exploratory. The external dataset's outcomes have not been evaluated.

## Question and model

Does a source-only estimate of a shared frequency response reduce the transfer cost of spatial normalization? Consider the approximation z_cv,j = u_c,j + h(log(v*q_j)) + a_cv + epsilon_cv,j for logarithmic bin masses. It assumes a separable response, narrow enough bins, and balanced texture distributions across source speeds. It does not assert that the response is physically identifiable in these recordings.

Represent h by 12 cosine functions of log frequency over a declared support. Average logarithmic spatial features within each class and speed, then average classes equally. Subtract the average across source speeds and project out the across-bin mean. Let d be the resulting stacked contrast vector and A the identical contrasts of the basis evaluations. Estimate theta = (A' A + lambda*s*D)^(-1) A' d, where s=trace(A'A)/12, D_jj=j^4/mean(k^4), and lambda=0.01. Subtract the fitted h at each trial's warped bin centers and re-normalize with log-sum-exp before the same source-fitted StandardScaler and RBF SVM. Ridge selects a regularized representative; a physical sensor transfer function is not uniquely recovered. Neither test labels nor test-distribution summaries enter fitting.

## Locked primary comparisons

Temporal, spatial, corrected spatial (lambda 0.01), pooled-spectrum whitening followed by spatial mapping, and log-mel features. Add 20-coefficient DCT log-mel features (MFCC summary, not framewise MFCC dynamics). All use exactly the same nine SVM candidates C={1,10,100}, gamma={0.25,1,4}/dimension and the same source folds. Model selection uses mean per-class recall, then smaller C, gamma nearest 1, then smaller gamma. No representation winner will be selected from target results.

Sensitivity only, not alternative winners: correction lambda 0.001 and 0.1, and a class-within-speed label-permutation check is unnecessary because the balanced estimator is label-insensitive once complete. Instead use a declared wrong-speed ordering control (reverse sorted source speed labels in calibration only). This is a deliberately invalid-mechanism control, not a deployable method. All are reported regardless of effect direction.

## Data and splits

Cluster retains every v1 acquisition, PSD extraction and all disjoint masks. Primary extension train is repeat 0 with source speeds 30/40/50, cardinal directions and 500mN. Three leave-one-source-speed-out folds. Repeat 1 is a correlated sensitivity analysis with its own source selection. Band edges 40..800Hz, reference40mm/s; response support20..1200Hz. The original 118-class test results are development evidence, never newly confirmatory.

External ingestion is separate and outcome-blind. Source dataset is Lima et al., Multimodal Tactile Texture Dataset v1, DOI10.17632/n666tk4mw9.1, CC BY4.0. Data-level preprocessing/split addendum will be saved after schema/quality inspection but before any target inference or source classification. No outcome-based changes to the method above are permitted after the external lock. Nominal directory speeds must be distinguished from measured velocity; known article-table inconsistency will be disclosed.

## Inference and failure rules

Primary external contrast will be corrected minus spatial macro accuracy at the held-out speed. A positive value of at least3percentage points with paired specimen bootstrap interval excluding zero is the operational support rule, not a guarantee that the true gain is above3points. Corrected versus log-mel is the secondary practical comparison; if the primary is unsupported, there is no secondary confirmatory superiority claim. Report all per-specimen scores and acquisition predictions. Seed repeats in the CNN are technical variation, not independent specimens. Bootstrap over specimens,10000replicates seed20270913, conditions on the trained classifiers and fixed panel. With12external specimens, intervals are descriptive and population generalization is limited.

The response model is falsified as a generally useful solution if external accuracy fails to improve. Negative results cannot be replaced by a target-selected method. Compare source residual structure and the same estimator's held-source-speed validation to test its approximation, without equating predictive improvements with identification of physical filtering.

## Budget and records

CPU2threads, peak process3GB; no paid resources. Maximum120minutes active SVM fit/analysis time, all predetermined fits and raw outputs retained. Neural baseline has a separate90minute active training cap and protocol. Data acquisition can retry bounded failures but may not replace unavailable real observations with simulations. Computational time does not constrain literature/authoring to a deadline. Tests use explicitly synthetic log-spectrum arrays only to check linear algebra and leakage invariants, never as evidence of robotic performance.
