#!/usr/bin/env python3
"""Export a completed synthetic run without requiring a historical acceptance receipt.

Uses the standard library only. This verifies recorded numerical replay and
artifact integrity; it neither performs independent peer review nor reruns fits.
"""
from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1] / "outputs/research_v2"
SOURCES = {
    "protocol": "research/calibrator_diagnostic_protocol.md",
    "calibrator": "experiments/response_calibration.py",
    "input": "experiments/diagnostics/synthetic_signals_psd.npz",
    "diagnostic_code": "experiments/calibrator_diagnostic.py",
    "protocol_freeze": "research/calibrator_diagnostic_protocol.freeze.json",
    "original_generator_code": "experiments/estimator_diagnostics.py",
}
STAGES = ("test", "source", "evaluate", "verify")
KEYS = ("part", "design", "observation", "method")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def member(root, name):
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts,
            "Unsafe artifact member path")
    path = root / relative
    require(path.resolve().is_relative_to(root.resolve()), "Artifact escapes run directory")
    require(path.is_file(), f"Missing artifact: {relative.as_posix()}")
    return path


def verify_inventory(run, records, expected):
    require(len(records) == len({r["path"] for r in records}), "Duplicate sealed artifact")
    require({r["path"] for r in records} == expected, "Sealed artifact inventory differs")
    for record in records:
        path = member(run, record["path"])
        require(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"],
                f"Artifact changed: {record['path']}")


def file_set(run, directory):
    return {p.relative_to(run).as_posix() for p in (run / directory).rglob("*") if p.is_file()}


def literal(node):
    """Read frozen literals and list concatenation/repetition without executing code."""
    if isinstance(node, ast.BinOp):
        left, right = literal(node.left), literal(node.right)
        if isinstance(node.op, ast.Add) and isinstance(left, list) and isinstance(right, list):
            return left + right
        if isinstance(node.op, ast.Mult) and isinstance(left, list) and isinstance(right, int):
            require(0 <= right <= 100, "Unexpected configuration list size")
            return left * right
        raise ValueError("Unsupported frozen configuration expression")
    if isinstance(node, ast.Dict):
        return {literal(k): literal(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.List):
        return [literal(v) for v in node.elts]
    return ast.literal_eval(node)


def frozen_constants():
    tree = ast.parse((STUDY / SOURCES["diagnostic_code"]).read_text())
    return {target.id: literal(statement.value)
            for statement in tree.body if isinstance(statement, ast.Assign)
            for target in statement.targets
            if isinstance(target, ast.Name) and target.id in {"CONFIG", "EXPECTED"}}


def verify_run(run):
    constants = frozen_constants()
    hashes = {name: sha(STUDY / path) for name, path in SOURCES.items()}
    for name, expected in constants["EXPECTED"].items():
        require(hashes[name] == expected, f"Frozen source input changed: {name}")
    freeze = read_json(STUDY / SOURCES["protocol_freeze"])
    require(freeze["protocol_sha256"] == hashes["protocol"] and
            freeze["input_sha256"] == hashes["input"], "Protocol freeze binding differs")
    config = read_json(member(run, "config.json"))
    require(config == constants["CONFIG"], "Run configuration differs from frozen producer")
    initialization = read_json(member(run, "initialization.json"))
    source = read_json(member(run, "source_seal.json"))
    result = read_json(member(run, "result_seal.json"))
    verification = read_json(member(run, "verification.json"))
    config_sha = sha(run / "config.json")
    for record in (initialization, source, result):
        require(record["input_hashes"] == hashes, "Run code/input binding differs")
    for record in (initialization, source):
        require(record["configuration_sha256"] == config_sha, "Configuration hash differs")
    require(source["before_any_B_target_prediction"] is True and source["final_model_count"] == 6,
            "Missing pre-target source freeze")
    source_files = file_set(run, "source") | file_set(run, "code") | {
        "config.json", "initialization.json", "environment.json", "test_report.json"}
    verify_inventory(run, source["artifacts"], source_files)
    verify_inventory(run, result["artifacts"], file_set(run, "evaluation"))
    require(sha(member(run, "code/calibrator_diagnostic.py")) == hashes["diagnostic_code"] and
            sha(member(run, "code/response_calibration.py")) == hashes["calibrator"],
            "Code snapshot differs from current frozen producer")
    source_sha, result_sha = sha(run / "source_seal.json"), sha(run / "result_seal.json")
    require(result["source_seal_sha256"] == source_sha and
            verification["source_seal_sha256"] == source_sha and
            verification["result_seal_sha256"] == result_sha, "Seal chain differs")
    require(result["status"] == "completed_pending_independent_check" and
            verification["status"] == "internal_checks_passed_pending_independent_checker",
            "Run verification is incomplete")
    test = read_json(run / "test_report.json")
    require(test["status"] == "passed" and test["n_checks"] == 5 and
            len(test["checks"]) == 5 and test["code_sha256"] == hashes["diagnostic_code"] and
            test["input_hashes"] == hashes, "Implementation checks are incomplete")
    require(len(verification["checks"]) == 5, "Verification checks are incomplete")
    require(not file_set(run, "failures"), "Run contains failure records")
    for stage in STAGES:
        timing = read_json(member(run, f"timing/{stage}.json"))
        completion = read_json(member(run, f"logs/{stage}_completion.json"))
        command = read_json(member(run, f"commands/{stage}.json"))
        require(timing["stage"] == command["stage"] == stage and
                timing["status"] == completion["status"] == "complete",
                f"Stage is incomplete: {stage}")
        require(command["input_hashes"] == hashes and completion["stage_timing_sha256"] ==
                sha(run / f"timing/{stage}.json"), f"Stage receipt binding differs: {stage}")
        require(0 <= timing["active_seconds"] <= timing["cumulative_active_seconds"] <=
                config["budget_active_seconds"], f"Invalid stage timing: {stage}")
    guard = read_json(member(run, "evaluation/target_access_guard.json"))
    require(guard["source_seal_sha256"] == source_sha and
            guard["source_artifacts_verified"] == len(source["artifacts"]) and
            guard["all_six_final_models_verified_before_target_loading"] is True,
            "Pre-target model verification receipt differs")
    times = [dt.datetime.fromisoformat(r["utc"]) for r in (source, guard, result, verification)]
    require(times == sorted(times), "Source/target/result receipt chronology differs")
    selections = read_json(member(run, "source/B/selection.json"))
    require(len(selections) == 6 and len(list((run / "source/B/final_models").glob("*.joblib"))) == 6
            and len(list((run / "source/B/cv_models").glob("*.joblib"))) == 162,
            "Retained model count differs")
    for selection in selections:
        require(sha(member(run, selection["model_path"])) == selection["model_sha256"],
                "Selected model binding differs")
    require(len(read_csv(member(run, "source/B/candidates.csv"))) == 54 and
            len(read_csv(member(run, "source/B/cv_predictions.csv"))) == 3888,
            "Source selection record count differs")
    summary = read_csv(member(run, "evaluation/summary.csv"))
    expected = {("A", d, "exact_additive", "corrected") for d in ("Cluster", "Lima")} | {
        ("B", "Cluster", o, m) for o in config["observations_B"] for m in config["methods_B"]}
    rows = {tuple(row[k] for k in KEYS): row for row in summary}
    require(len(summary) == len(rows) == 8 and set(rows) == expected, "Summary conditions differ")
    summary_json = read_json(member(run, "evaluation/summary.json"))
    require(summary_json["status"] == "completed_pending_independent_check" and
            summary_json["population_inference"] is False, "Summary scope differs")
    require(len(summary_json["records"]) == 8, "Summary JSON row count differs")
    for record in summary_json["records"]:
        row = rows[tuple(record[k] for k in KEYS)]
        for key, value in record.items():
            if isinstance(value, (int, float)):
                require(math.isclose(float(row[key]), value, rel_tol=0, abs_tol=1e-12),
                        f"CSV/JSON value differs: {key}")
            elif isinstance(value, str):
                require(row[key] == value, f"CSV/JSON label differs: {key}")
    predictions = read_csv(member(run, "evaluation/B/all_target_predictions.csv"))
    require(len(predictions) == 576, "Target prediction count differs")
    replay = []
    for observation in config["observations_B"]:
        for method in config["methods_B"]:
            group = [r for r in predictions if r["observation"] == observation and r["method"] == method]
            require(len(group) == 96 and all(r["part"] == "B" for r in group), "Prediction group differs")
            coordinates = [(int(r["template"]), int(r["speed"]), int(r["repeat"])) for r in group]
            require(len(set(coordinates)) == 96 and set(coordinates) == {
                (t, s, q) for t in range(12) for s in (20, 60) for q in range(4)},
                "Target template/speed/repetition coordinates differ")
            require(len({r["index"] for r in group}) == 96, "Duplicate target indices")
            for row in group:
                require(0 <= int(row["prediction"]) < 12 and int(row["correct"]) ==
                        int(row["template"] == row["prediction"]), "Invalid prediction/correct field")
            per_method = read_csv(member(run, f"evaluation/B/predictions/{observation}_{method}.csv"))
            require(per_method == group, "Per-method prediction file differs from pooled CSV")
            totals = Counter(int(r["template"]) for r in group)
            correct = Counter(int(r["template"]) for r in group if int(r["correct"]))
            accuracy = math.fsum(correct[t] / totals[t] for t in range(12)) / 12
            stored = float(rows[("B", "Cluster", observation, method)]["target_macro_accuracy"])
            require(math.isclose(accuracy, stored, rel_tol=0, abs_tol=1e-12), "Target accuracy differs")
            replay.append({"observation": observation, "method": method,
                           "predictions": len(group), "recomputed_macro_accuracy": accuracy,
                           "stored_macro_accuracy": stored})
    return summary, rows, replay, hashes


def export(run, output):
    require(not output.exists(), "Output directory already exists; choose a new directory")
    require(not output.resolve().is_relative_to(run.resolve()), "Output must be outside the sealed run")
    summary, rows, replay, hashes = verify_run(run)
    mappings = {
        "CalibratorIdealAccuracy": (("B", "Cluster", "ideal", "corrected"), "target_macro_accuracy", 100, ".2f"),
        "CalibratorFilteredAccuracy": (("B", "Cluster", "lowpass_200Hz", "corrected"), "target_macro_accuracy", 100, ".2f"),
        "CalibratorIdealUncorrectedAccuracy": (("B", "Cluster", "ideal", "spatial"), "target_macro_accuracy", 100, ".2f"),
        "CalibratorFilteredUncorrectedAccuracy": (("B", "Cluster", "lowpass_200Hz", "spatial"), "target_macro_accuracy", 100, ".2f"),
        "CalibratorAdditiveClusterError": (("A", "Cluster", "exact_additive", "corrected"), "target_response_rms_gauge_adjusted", 1, ".5f"),
        "CalibratorAdditiveLimaError": (("A", "Lima", "exact_additive", "corrected"), "target_response_rms_gauge_adjusted", 1, ".5f"),
        "CalibratorFiniteIdealError": (("B", "Cluster", "ideal", "corrected"), "target_response_rms_gauge_adjusted", 1, ".5f"),
        "CalibratorFiniteFilteredError": (("B", "Cluster", "lowpass_200Hz", "corrected"), "target_response_rms_gauge_adjusted", 1, ".5f"),
    }
    macros = ["% Generated from the specified run's sealed CSV; verified numerical replay."]
    values = {}
    for name, (key, field, scale, fmt) in mappings.items():
        raw = float(rows[key][field])
        require(math.isfinite(raw), f"Non-finite macro value: {name}")
        rendered = format(raw * scale, fmt)
        macros.append(f"\\newcommand{{\\{name}}}{{{rendered}}}")
        values[name] = {"row": dict(zip(KEYS, key)), "field": field,
                        "raw_value": raw, "scale": scale, "rendered": rendered}
    table = ["% Generated directly from evaluation/summary.csv.",
             r"\begin{tabular}{lllrrr}", r"\hline",
             r"Part/design & Observation & Method & Target acc. & Contrast RMSE & Target gauge RMSE \\",
             r"\hline"]
    labels = {"exact_additive": "additive", "ideal": "ideal", "lowpass_200Hz": "low-pass, 200 Hz",
              "corrected": "estimated correction", "spatial": "spatial (uncorrected)", "temporal": "temporal"}
    for row in summary:
        accuracy = row["target_macro_accuracy"]
        contrast = row["source_contrast_error_rms"]
        response = row["target_response_rms_gauge_adjusted"]
        cells = [f"{row['part']}/{row['design']}", labels[row["observation"]], labels[row["method"]],
                 f"{100 * float(accuracy):.2f}\\%" if accuracy else "--",
                 f"{float(contrast):.5f}" if contrast else "--",
                 f"{float(response):.5f}" if response else "--"]
        table.append(" & ".join(cells) + r" \\")
    table.extend([r"\hline", r"\end{tabular}",
                  "% Accuracies are descriptive template macro recall, not population inference.",
                  "% Response truth/gauge alignment is evaluation-only; B uses a pointwise finite-PSD reference."])
    evidence_names = ["config.json", "initialization.json", "source_seal.json", "result_seal.json",
                      "test_report.json", "verification.json", "evaluation/target_access_guard.json",
                      "evaluation/summary.csv", "evaluation/summary.json", "evaluation/B/all_target_predictions.csv"]
    evidence_names += [f"{directory}/{stage}{suffix}" for stage in STAGES
                       for directory, suffix in (("timing", ".json"), ("commands", ".json"),
                                                 ("logs", "_completion.json"))]
    lineage = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "status": "verified_numerical_replay", "independent_acceptance_verified": False,
               "verification_scope": "Artifact integrity, completed stage receipts, frozen inputs and CSV metric replay; not peer review or retraining.",
               "source_hashes": hashes, "run_artifacts": {p: sha(run / p) for p in evidence_names},
               "source_of_values": "evaluation/summary.csv", "prediction_replay": replay,
               "macros": values, "summary_rows": summary,
               "scope": "Synthetic fixed-template diagnostic; no physical-data benefit or population inference.",
               "adapter_sha256": sha(Path(__file__))}
    output.mkdir(parents=True)
    (output / "calibrator_numbers.tex").write_text("\n".join(macros) + "\n")
    (output / "calibrator_table.tex").write_text("\n".join(table) + "\n")
    (output / "synthetic_table_lineage.json").write_text(json.dumps(lineage, indent=2, allow_nan=False) + "\n")
    return {"status": lineage["status"], "summary_rows": len(summary), "target_predictions": 576,
            "independent_acceptance_verified": False,
            "files": ["calibrator_numbers.tex", "calibrator_table.tex", "synthetic_table_lineage.json"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed calibrator_diagnostic run")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory outside the run")
    arguments = parser.parse_args()
    try:
        result = export(arguments.run_dir.resolve(), arguments.output_dir.resolve())
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.exit(1, f"Synthetic table verification failed: {error}\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
