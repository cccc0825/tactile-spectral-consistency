# Resource allowance for a new reproduction

The archived CNN protocol used one shared **10,800-second (180-minute) active-compute limit**. CPU throughput can differ even with the same software and two-thread settings. `scripts/reproduction_resources.py` can record one explicitly chosen **new local reproduction allowance**, up to 21,600 seconds, for exactly the existing fixed operations. It does not change epochs, folds, learning rates, seeds, data, models, metrics, the 3 GB process limit, or the original protocols/results. A run exceeding 180 minutes must not be described as completing within the historical limit.

This is optional. A run that finishes under the original limit does not need an allowance. The tool does not repair failed or killed sessions, resume partial training, or reset a ledger. Preserve such failures and investigate them separately.

## Establish the boundary

Prepare the full fresh study normally. Both `cluster_config.json` and `external_config.json` and the original amendment-02 receipts must exist. New users who expect a slower CPU may run the allowance command **after both CNN configurations and before either CNN selection**, avoiding a training interruption. The existing source-only preparation files and `STUDY_SETUP.json`/`FRESH_RUN.json` remain unchanged. The utility may also be invoked from a newer release against an older prepared study, using `--project-root`; do not overwrite the older study's bound source files.

Before applying an allowance, stop the scheduler from dispatching another stage and let any current training/check process exit normally. Verify that **no process can write the CNN ledger**, `active_session` is null, and no target evaluation or global release/seal exists. Keep the scheduler stopped until the command returns successfully. The acknowledgement flag records this operator assertion; a null ledger flag cannot prove operating-system process inactivity. Do not stop a trainer mid-epoch or edit an active ledger.

The command rejects an existing allowance directory, an unknown limit, an exhausted original cap, incomplete time accounting, changed configured inputs, modified prepared sources, changed scientific settings, target outputs/history, and a global seal. It requires the original amendment to remain intact. An interrupted application leaves evidence and fails closed; do not delete that evidence and retry automatically.

## Apply once and verify

From the release directory, with the inactive fresh study at `../fresh-study`:

```sh
python scripts/reproduction_resources.py allow \
  --project-root ../fresh-study \
  --limit-seconds 21600 \
  --reason "Measured CPU throughput needs additional time for the unchanged fixed matrix." \
  --acknowledge-no-running-writers
```

Choose a finite limit above 10,800 and at most 21,600 seconds from source-only throughput/resource evidence, before viewing fresh target results. This is a resource decision, not permission to add experiments or tune the scientific configuration. Normal authorization for local computation still applies; the tool does not acquire paid compute or authorize external services.

The utility writes `reproduction/resource_allowance/` containing the original ledger bytes, the independent allowance receipt, an applied-ledger snapshot and an application receipt. It atomically updates the current reproduction ledger's effective limit and receipt link. All previous elapsed time, history and other ledger fields are retained. Neither amendment-02 file is rewritten. The before/after snapshots and hashes expose the exact transformation; these hashes are integrity checks, not cryptographic proof of physical process inactivity or an external preregistration.

The frozen external configuration still records its original `active_compute_limit_seconds: 10800`. The unchanged trainer obtains the effective runtime limit from the mutable ledger; the new receipt explains the resource-only override. Do not edit the frozen configuration to make the two numbers appear identical.

Before the normal global seal, at another inactive boundary:

```sh
python scripts/reproduction_resources.py check --project-root ../fresh-study --pre-target
```

Keep this check's JSON output with the run's audit records and then use the unchanged global-seal command from `FULL_STUDY.md`. The existing seal records the effective limit and consumed time. It does not automatically invoke this new companion check or bind its receipt; retain the allowance directory with the final reproducibility records. The resource check is **not** proof that the scientific run or all source audits are complete.

After every prescribed CNN phase and target check has finished:

```sh
python scripts/reproduction_resources.py check --project-root ../fresh-study
```

This verifies the immutable inputs/receipt chain, conserved history and elapsed time, effective limit, and any global seal's resource fields. It permits completed target outputs and reports `original_180_minutes_exceeded` explicitly. Preserve this output and report the actual resource consumption and effective allowance alongside numerical results. The tool refuses while a ledger session is active; use ordinary progress logs during training.

The preparation helper also has a **per-command** `--timeout-seconds` (default 10,800), separate from this shared cumulative allowance. A private scheduler may impose another per-stage timeout. If a single fixed stage needs more time, configure those outer timeouts before launching it and record the choice. This utility does not change them. Do not confuse the CNN active-compute allowance with the separate v1 90-minute persistent wall-clock contract.

## Check the utility

```sh
python scripts/test_reproduction_resources.py
```

These tests use tiny artificial fixture ledgers and files only. They test the safeguards and accounting, not CNN runtime, numerical equivalence or scientific outcomes.
