# Exploratory matched-hyperparameter diagnostic

Written 2026-09-12 after inspecting all frozen primary result summaries. This is explicitly post hoc; it does not replace H1, revise selected models, or count as prospective confirmation.

Motivation: source-only tuning selected C=1/gamma=1 for temporal but C=10/gamma=1 for spatial in the primary repeat, with large ID differences. The negative speed contrast could depend on different hyperparameter choices. Diagnose that alternative by fitting both 128-dimensional representations at every matched (C,gamma multiplier) pair from the original 3x3 grid, under both original training-repeat choices. No setting is selected based on these target scores; report all nine paired contrasts for both repeats. Use unchanged data, preprocessing, training IDs and target domains. Retain all predictions, including unfavorable ones. No new data, model family or tuned threshold.

36 deterministic model fits/evaluations. Stop at the existing persistent 90-minute deadline and 3 GB process limit. Results are diagnostic robustness, not independent replication or a new method comparison. Primary study and its hashes remain unchanged; the diagnostic has its own prospective execution freeze below. Intended output is the fraction and range of matched-grid contrasts, not a best-case score.
