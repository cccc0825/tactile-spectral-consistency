"""Frozen post-review simulation diagnostic; never modifies physical-data artifacts.

Run ``python experiments/calibrator_diagnostic.py all --run-dir <new-directory>``
from research_v2, or invoke test, source, evaluate, verify separately. The evaluate
entry point refuses target inference until every source artifact has been sealed.
"""
from __future__ import annotations

import os
import sys
import time

BOOT_TIME = time.monotonic()
sys.dont_write_bytecode = True
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = "2"

import argparse
import csv
import datetime as dt
import hashlib
import importlib.metadata
import io
import itertools
import json
import pathlib
import platform
import shlex
import signal
import subprocess
import tempfile
import traceback
import uuid

import joblib
import numpy as np
from scipy.signal import butter, sosfreqz
from scipy.special import logsumexp
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_info, threadpool_limits

from response_calibration import ResponseCalibration, log_masses

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
CODE = pathlib.Path(__file__).resolve()
CALIBRATOR = ROOT / "experiments/response_calibration.py"
PROTOCOL = ROOT / "research/calibrator_diagnostic_protocol.md"
PROTOCOL_FREEZE = ROOT / "research/calibrator_diagnostic_protocol.freeze.json"
INPUT = ROOT / "experiments/diagnostics/synthetic_signals_psd.npz"
GENERATOR = ROOT / "experiments/estimator_diagnostics.py"
EXPECTED = {
    "protocol": "501c293f001994042105bdcef7e793457fee637e1865d811eacca9a200a418a1",
    "calibrator": "b2096e13e7978dc8e2b4d5280488f924f813e57b1f7701b02ead2687241c668a",
    "input": "179f5f1ebaca701bf0aa7f0f47274768e72acc240b9d13d93d5ef213e84c9f2b",
}
CONFIG = {
    "scope": "post-review exploratory simulation and implementation evidence only",
    "seed_A": 20270914, "stored_signal_seed_B": 20270913,
    "rng_A": "NumPy Generator PCG64; draw u once, then offsets in Cluster,Lima order",
    "classes": 12, "repetitions": [0, 1, 2, 3], "fit_repetitions": [0, 1],
    "bins": 128, "basis_cosines": 12, "lambda": 0.01,
    "theta_true_A": [0.6, -0.3] + [0.0] * 10,
    "class_log_spectrum_sd_A": 0.5, "scalar_offset_sd_A": 0.2,
    "row_order_A": "template, ascending speed, repeat",
    "designs": {
        "Cluster": {"lo": 40, "hi": 800, "ref": 40, "support": [20, 1200],
                    "source_speeds": [30, 40, 50], "target_speeds": [20, 60]},
        "Lima": {"lo": 5, "hi": 100, "ref": 35, "support": [4, 120],
                 "source_speeds": [30, 35], "target_speeds": [40]},
    },
    "observations_B": ["ideal", "lowpass_200Hz"],
    "methods_B": ["temporal", "spatial", "corrected"],
    "C": [1, 10, 100], "gamma_multiplier": [0.25, 1, 4], "gamma_divisor": 128,
    "selection": "mean held-source-speed fold macro recall; lower C then multiplier ties",
    "filter_reference": {"kind": "discrete fourth-order Butterworth lowpass",
                         "cutoff_Hz": 200, "fs_Hz": 6000, "response": "log(abs(sosfreqz)**2)"},
    "gauge": "OLS estimated-minus-truth on [1,log(frequency_Hz)] at unique source-speed bin centers only; subtract this fixed affine function at both source and target",
    "target_feature_use_of_gauge": False,
    "budget_active_seconds": 300, "cpu_threads": 2,
    "uncertainty": "descriptive deterministic-template/technical-repetition outcomes; no population p-values",
    "expected_B_counts": {"input_per_observation": 240, "source": 72, "target": 96,
                          "unused_source": 72, "final_models": 6, "candidates": 54,
                          "cv_fits": 162, "cv_predictions": 3888, "target_predictions": 576},
}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def array_sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.dtype.str, a.shape)).encode() + a.tobytes()).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write_bytes(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_json(path, value):
    write_bytes(path, json_bytes(value))


def save_npz(path, **arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    write_bytes(path, stream.getvalue())


def save_model(path, model):
    stream = io.BytesIO()
    joblib.dump(model, stream, compress=3)
    write_bytes(path, stream.getvalue())


def save_csv(path, rows):
    assert rows, "Cannot emit an empty scientific table"
    fields = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    write_bytes(path, stream.getvalue().encode())


def read_json(path):
    return json.loads(pathlib.Path(path).read_text())


def checked_inputs():
    actual = {"protocol": sha(PROTOCOL), "calibrator": sha(CALIBRATOR), "input": sha(INPUT)}
    assert actual == EXPECTED, (actual, EXPECTED)
    frozen = read_json(PROTOCOL_FREEZE)
    assert frozen["protocol_sha256"] == actual["protocol"]
    assert frozen["input_sha256"] == actual["input"]
    return {**actual, "diagnostic_code": sha(CODE), "protocol_freeze": sha(PROTOCOL_FREEZE),
            "original_generator_code": sha(GENERATOR)}


def record_file(path, root):
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha(path)}


def verify_record(root, record):
    path = root / record["path"]
    assert path.is_file(), f"Missing frozen artifact: {path}"
    assert path.stat().st_size == record["bytes"], f"Frozen size drift: {path}"
    assert sha(path) == record["sha256"], f"Frozen hash drift: {path}"


def inventory(root, directories):
    files = [path for directory in directories for path in (root / directory).rglob("*") if path.is_file()]
    return [record_file(path, root) for path in sorted(set(files))]


def budget_spent(run):
    return sum(read_json(path)["active_seconds"] for path in (run / "timing").glob("*.json"))


def macro_recall(y, predicted):
    classes = np.unique(y)
    assert len(classes) > 0
    return float(np.mean([np.mean(predicted[y == c] == c) for c in classes]))


def double_center(a):
    a = np.asarray(a, dtype=float)
    return a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def settings(name):
    return {key: CONFIG["designs"][name][key] for key in ("lo", "hi", "ref", "support")}


def new_calibrator(name):
    model = ResponseCalibration(**settings(name), lam=CONFIG["lambda"])
    assert model.lam == 0.01
    return model


def temporal_centers(model, speeds):
    edge = np.geomspace(model.lo, model.hi, 129)
    return np.asarray(speeds)[:, None] / model.ref * np.sqrt(edge[:-1] * edge[1:])[None, :]


def balanced_source_contrast(z, v, y):
    means = np.stack([np.mean([z[(v == speed) & (y == c)].mean(0) for c in np.unique(y)], axis=0)
                      for speed in np.unique(v)])
    assert np.isfinite(means).all()
    return double_center(means)


def gauge_fit(source_frequency, source_error):
    design = np.column_stack([np.ones(source_frequency.size), np.log(source_frequency).ravel()])
    coefficients, _, rank, _ = np.linalg.lstsq(design, source_error.ravel(), rcond=None)
    assert rank == 2
    return coefficients


def gauge_function(frequency, coefficients):
    return coefficients[0] + coefficients[1] * np.log(frequency)


def response_diagnostic(model, source_z, source_v, source_y, target_speeds, truth, destination):
    src_speeds = np.unique(source_v)
    target_speeds = np.asarray(target_speeds)
    src_f = temporal_centers(model, src_speeds)
    target_f = temporal_centers(model, target_speeds)
    src_est = np.einsum("nbk,k->nb", model.basis(src_speeds), model.theta)
    target_est = np.einsum("nbk,k->nb", model.basis(target_speeds), model.theta)
    if truth == "direct":
        true_theta = np.asarray(CONFIG["theta_true_A"])
        src_true = np.einsum("nbk,k->nb", model.basis(src_speeds), true_theta)
        target_true = np.einsum("nbk,k->nb", model.basis(target_speeds), true_theta)
    elif truth == "ideal":
        src_true, target_true = np.zeros_like(src_f), np.zeros_like(target_f)
    else:
        assert truth == "lowpass_200Hz"
        sos = butter(4, 200, fs=6000, output="sos")
        src_true = np.log(np.abs(sosfreqz(sos, worN=src_f.ravel(), fs=6000)[1]) ** 2).reshape(src_f.shape)
        target_true = np.log(np.abs(sosfreqz(sos, worN=target_f.ravel(), fs=6000)[1]) ** 2).reshape(target_f.shape)
    src_error, target_error = src_est - src_true, target_est - target_true
    gauge = gauge_fit(src_f, src_error)
    src_adjusted = src_error - gauge_function(src_f, gauge)
    target_adjusted = target_error - gauge_function(target_f, gauge)
    observed_contrast = balanced_source_contrast(source_z, source_v, source_y)
    estimated_contrast, true_contrast = double_center(src_est), double_center(src_true)
    source_residual = observed_contrast - estimated_contrast
    explained = float(1 - np.sum(source_residual ** 2) / np.sum(observed_contrast ** 2))
    assert np.isclose(explained, model.fit_diagnostics["contrast_variance_explained"], atol=1e-12)
    # Affine log-frequency gauge shifts disappear under the same source contrast.
    assert np.max(np.abs(double_center(gauge_function(src_f, gauge)))) < 1e-10
    metrics = {
        "truth": truth, "lambda": model.lam,
        "source_contrast_residual_rms": rms(source_residual),
        "source_contrast_variance_explained": explained,
        "source_contrast_error_rms": rms(estimated_contrast - true_contrast),
        "source_response_rms_unadjusted": rms(src_error),
        "target_response_rms_unadjusted": rms(target_error),
        "source_response_rms_gauge_adjusted": rms(src_adjusted),
        "target_response_rms_gauge_adjusted": rms(target_adjusted),
        "gauge_intercept": float(gauge[0]), "gauge_log_frequency_slope": float(gauge[1]),
        "gauge_fit_source_grid_points": int(src_f.size),
        "source_speeds": src_speeds.tolist(), "target_speeds": target_speeds.tolist(),
        "truth_reference_status": "exact additive construction" if truth == "direct" else "pointwise reference; finite PSD factorization is approximate",
    }
    assert all(np.isfinite(value) for value in metrics.values() if isinstance(value, float))
    save_npz(destination.with_suffix(".npz"), source_speeds=src_speeds, target_speeds=target_speeds,
             source_frequency=src_f, target_frequency=target_f,
             source_estimated=src_est, target_estimated=target_est,
             source_true=src_true, target_true=target_true,
             source_error=src_error, target_error=target_error,
             source_gauge_adjusted_error=src_adjusted, target_gauge_adjusted_error=target_adjusted,
             gauge_coefficients=gauge, source_observed_contrast=observed_contrast,
             source_estimated_contrast=estimated_contrast, source_true_contrast=true_contrast,
             source_contrast_residual=source_residual, fitted_theta=model.theta)
    save_json(destination.with_suffix(".json"), metrics)
    return metrics


def masks(labels, design_name):
    y, v, repeat = labels.T
    design = CONFIG["designs"][design_name]
    source = np.isin(v, design["source_speeds"]) & np.isin(repeat, CONFIG["fit_repetitions"])
    target = np.isin(v, design["target_speeds"])
    assert not np.any(source & target)
    assert len(np.unique(labels, axis=0)) == len(labels)
    for c, speed in itertools.product(range(12), design["source_speeds"]):
        assert np.sum(source & (y == c) & (v == speed)) == 2
    for c, speed in itertools.product(range(12), design["target_speeds"]):
        assert np.sum(target & (y == c) & (v == speed)) == 4
    return source, target


def source_b_input():
    # Only these returned source slices can reach fitting or source CV.
    with np.load(INPUT, allow_pickle=False) as data:
        labels = data["labels"]
        assert labels.shape == (240, 3)
        expected_labels = np.array(list(itertools.product(range(12), [20, 30, 40, 50, 60], range(4))))
        assert np.array_equal(labels, expected_labels)
        source, target = masks(labels, "Cluster")
        p = data["psd"][:, source]
        frequency = data["frequency"]
    assert p.shape == (2, 72, 513)
    assert np.array_equal(frequency, np.arange(513) * 6000 / 1024)
    assert np.isfinite(p).all() and np.all(p >= 0)
    assert source.sum() == 72 and target.sum() == 96 and (~(source | target)).sum() == 72
    return p, frequency, labels[source], np.flatnonzero(source), {
        "source_indices": np.flatnonzero(source).tolist(), "target_indices": np.flatnonzero(target).tolist(),
        "unused_indices": np.flatnonzero(~(source | target)).tolist(),
        "source_n": 72, "target_n": 96, "unused_n": 72,
        "source_repetitions": [0, 1], "target_repetitions": [0, 1, 2, 3],
    }


def fit_representation(p, frequency, v, y, method):
    kw = {key: settings("Cluster")[key] for key in ("lo", "hi", "ref")}
    z = log_masses(p, frequency, v, beta=0 if method == "temporal" else 1, **kw)
    calibration = new_calibrator("Cluster").fit(z, v, y) if method == "corrected" else None
    x = calibration.transform(z, v) if calibration is not None else z
    assert np.isfinite(x).all()
    return x, calibration


def transform_representation(p, frequency, v, method, calibration):
    # No class labels, fitting, truth or gauge coefficients are accepted here.
    kw = {key: settings("Cluster")[key] for key in ("lo", "hi", "ref")}
    z = log_masses(p, frequency, v, beta=0 if method == "temporal" else 1, **kw)
    before = None if calibration is None else calibration.theta.copy()
    x = z if calibration is None else calibration.transform(z, v)
    if calibration is not None:
        assert np.array_equal(before, calibration.theta)
    assert np.isfinite(x).all()
    return x


def pipeline(C, multiplier):
    return make_pipeline(StandardScaler(), SVC(C=C, gamma=multiplier / 128, cache_size=200))


def require_source_seal(run):
    seal_path = run / "source_seal.json"
    assert seal_path.is_file(), "Target inference requires the source/model seal"
    seal = read_json(seal_path)
    assert seal["before_any_B_target_prediction"] is True
    assert seal["final_model_count"] == 6
    assert seal["input_hashes"] == checked_inputs(), "Code, protocol or input changed after selection"
    assert sha(run / "config.json") == seal["configuration_sha256"]
    for record in seal["artifacts"]:
        verify_record(run, record)
    selected = read_json(run / "source/B/selection.json")
    assert len(selected) == 6
    for record in selected:
        assert sha(run / record["model_path"]) == record["model_sha256"]
    return seal


def initialize(run):
    assert str(run).startswith(str(ROOT / "experiments/calibrator_diagnostic") + os.sep), "Run artifacts must remain in the owned diagnostic directory"
    if (run / "config.json").exists():
        assert read_json(run / "config.json") == CONFIG
        assert read_json(run / "initialization.json")["input_hashes"] == checked_inputs()
        return
    assert not run.exists() or not any(run.iterdir()), "Use a new run directory; preserve old attempts"
    run.mkdir(parents=True, exist_ok=True)
    save_json(run / "config.json", CONFIG)
    save_json(run / "initialization.json", {"run_id": run.name, "utc": now(), "input_hashes": checked_inputs(),
              "configuration_sha256": sha(run / "config.json"), "git": "workspace is not a Git checkout; file hashes are version identities",
              "independent_checker_status": "pending", "physical_data_changes": False})
    write_bytes(run / "code/calibrator_diagnostic.py", CODE.read_bytes())
    write_bytes(run / "code/response_calibration.py", CALIBRATOR.read_bytes())
    save_json(run / "environment.json", {"python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "threadpools": threadpool_info(),
              "dependencies": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions() if d.metadata.get("Name"))),
              "thread_environment": {key: os.environ[key] for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")}})


def tests(run):
    checks = []
    assert macro_recall(np.array([0, 0, 1]), np.array([0, 0, 0])) == 0.5
    checks.append("macro recall gives equal weight to templates")
    m = new_calibrator("Cluster")
    f = temporal_centers(m, np.array([30, 40, 50]))
    known = np.array([1.7, -0.23])
    error = gauge_function(f, known)
    assert np.allclose(gauge_fit(f, error), known, atol=1e-12)
    tf = temporal_centers(m, np.array([20, 60]))
    target_error = gauge_function(tf, known)
    assert rms(target_error - gauge_function(tf, gauge_fit(f, error))) < 1e-12
    assert np.array_equal(gauge_fit(f, error), gauge_fit(f, error))
    assert np.max(np.abs(double_center(error))) < 1e-12
    checks.append("source OLS gauge recovers known affine error and extrapolates to held-out bins")
    v = np.tile([30, 40, 50], 12)
    y = np.repeat(np.arange(12), 3)
    u = np.sin(np.arange(12 * 128).reshape(12, 128) / 37) * 0.5
    z = u[y] + np.einsum("nbk,k->nb", m.basis(v), np.array(CONFIG["theta_true_A"]))
    z -= logsumexp(z, axis=1, keepdims=True)
    m.fit(z, v, y)
    basis = m.basis(np.unique(v))
    matrix = np.stack([double_center(basis[:, :, k]) for k in range(12)], axis=2).reshape(-1, 12)
    penalty = np.arange(1, 13, dtype=float) ** 4
    gram = matrix.T @ matrix
    expected = np.linalg.solve(gram + 0.01 * np.trace(gram) / 12 * np.diag(penalty / penalty.mean()),
                               matrix.T @ balanced_source_contrast(z, v, y).ravel())
    assert np.allclose(m.theta, expected, rtol=1e-10, atol=1e-10)
    before = m.theta.copy()
    m.transform(np.full((2, 128), -np.log(128)), np.array([20, 60]))
    assert np.array_equal(before, m.theta)
    checks.append("actual lambda 0.01 matches an independently assembled ridge system; target transform leaves theta unchanged")
    scratch = WORKSPACE / "work/revision/calibrator_diagnostic"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as temp:
        empty = pathlib.Path(temp)
        try:
            require_source_seal(empty)
        except AssertionError as exc:
            assert "requires" in str(exc)
        else:
            raise AssertionError("Unsealed target evaluation was allowed")
        item = empty / "artifact"
        item.write_bytes(b"original")
        record = record_file(item, empty)
        item.write_bytes(b"modified")
        try:
            verify_record(empty, record)
        except AssertionError:
            pass
        else:
            raise AssertionError("Artifact drift was accepted")
    checks.append("unsealed target evaluation and frozen artifact drift are rejected")
    p, _, labels, indices, split = source_b_input()
    assert len(indices) == len(labels) == p.shape[1] == split["source_n"] == 72
    checks.append("immutable PSD, exact labels, source/target/unused counts and mutual exclusion")
    save_json(run / "test_report.json", {"status": "passed", "utc": now(), "checks": checks,
              "code_sha256": sha(CODE), "input_hashes": checked_inputs(), "n_checks": len(checks)})
    return {"checks": len(checks), "status": "passed"}


def source(run):
    report = read_json(run / "test_report.json")
    assert report["status"] == "passed" and report["code_sha256"] == sha(CODE)
    assert not (run / "source").exists(), "Source stage cannot overwrite or retry an existing attempt"
    assert not (run / "evaluation").exists(), "Source selection must precede target inference"
    rng = np.random.default_rng(CONFIG["seed_A"])
    u = rng.normal(0, 0.5, (12, 128))
    save_npz(run / "source/A/class_templates.npz", class_log_spectra=u, true_theta=np.array(CONFIG["theta_true_A"]))
    a_manifest = []
    for name in CONFIG["designs"]:
        design = CONFIG["designs"][name]
        speeds = sorted(design["source_speeds"] + design["target_speeds"])
        labels = np.array(list(itertools.product(range(12), speeds, range(4))))
        y, v, _ = labels.T
        source_mask, target_mask = masks(labels, name)
        model = new_calibrator(name)
        offsets = rng.normal(0, 0.2, (len(labels), 1))
        base = u[y] + np.einsum("nbk,k->nb", model.basis(v), np.array(CONFIG["theta_true_A"]))
        raw = base + offsets
        z = raw - logsumexp(raw, axis=1, keepdims=True)
        assert np.allclose(np.exp(z).sum(axis=1), 1, atol=1e-12)
        assert np.max(np.abs(z - (base - logsumexp(base, axis=1, keepdims=True)))) < 3e-15
        model.fit(z[source_mask], v[source_mask], y[source_mask])
        save_npz(run / f"source/A/{name}_construction.npz", log_spectra=z, scalar_offsets=offsets,
                 labels=labels, source_indices=np.flatnonzero(source_mask), target_indices=np.flatnonzero(target_mask))
        model_path = run / f"source/A/{name}_calibrator.joblib"
        save_model(model_path, model)
        a_manifest.append({"design": name, "source_n": int(source_mask.sum()), "target_n": int(target_mask.sum()),
                           "total_n": len(labels), "model_path": str(model_path.relative_to(run)),
                           "model_sha256": sha(model_path), "fit_diagnostics": model.fit_diagnostics})
    save_json(run / "source/A/models.json", a_manifest)

    psd, frequency, labels, original_indices, split = source_b_input()
    y, v, _ = labels.T
    save_npz(run / "source/B/input_source_only.npz", psd=psd, frequency=frequency,
             labels=labels, original_indices=original_indices)
    save_json(run / "source/B/split.json", split)
    selections, flat_candidates, cv_predictions = [], [], []
    for obs_idx, observation in enumerate(CONFIG["observations_B"]):
        for method in CONFIG["methods_B"]:
            candidates = []
            for candidate_index, (C, multiplier) in enumerate(itertools.product(CONFIG["C"], CONFIG["gamma_multiplier"])):
                folds = []
                for held_speed in CONFIG["designs"]["Cluster"]["source_speeds"]:
                    fit_mask, validation_mask = v != held_speed, v == held_speed
                    assert fit_mask.sum() == 48 and validation_mask.sum() == 24
                    x_train, calibration = fit_representation(psd[obs_idx, fit_mask], frequency, v[fit_mask], y[fit_mask], method)
                    x_validation = transform_representation(psd[obs_idx, validation_mask], frequency, v[validation_mask], method, calibration)
                    svm = pipeline(C, multiplier)
                    svm.fit(x_train, y[fit_mask])
                    predicted = svm.predict(x_validation)
                    fit_name = f"{observation}_{method}_candidate{candidate_index:02d}_held{held_speed}"
                    model_path = run / f"source/B/cv_models/{fit_name}.joblib"
                    bundle = {"calibrator": calibration, "classifier": svm, "method": method,
                              "observation": observation, "C": C, "gamma_multiplier": multiplier,
                              "train_indices": original_indices[fit_mask], "validation_indices": original_indices[validation_mask],
                              "training_feature_sha256": array_sha(x_train), "validation_feature_sha256": array_sha(x_validation)}
                    save_model(model_path, bundle)
                    fold = {"held_source_speed": held_speed, "macro_recall": macro_recall(y[validation_mask], predicted),
                            "train_n": int(fit_mask.sum()), "validation_n": int(validation_mask.sum()),
                            "model_path": str(model_path.relative_to(run)), "model_sha256": sha(model_path),
                            "calibration": None if calibration is None else calibration.fit_diagnostics}
                    folds.append(fold)
                    for idx, lab, pred in zip(original_indices[validation_mask], labels[validation_mask], predicted):
                        cv_predictions.append({"observation": observation, "method": method,
                          "candidate_index": candidate_index, "C": C, "gamma_multiplier": multiplier,
                          "held_source_speed": held_speed, "index": int(idx), "template": int(lab[0]),
                          "speed": int(lab[1]), "repeat": int(lab[2]), "prediction": int(pred), "correct": int(lab[0] == pred)})
                candidate = {"candidate_index": candidate_index, "C": C, "gamma_multiplier": multiplier,
                             "gamma": multiplier / 128, "mean_fold_macro_recall": float(np.mean([f["macro_recall"] for f in folds])), "folds": folds}
                candidates.append(candidate)
                flat_candidates.append({"observation": observation, "method": method,
                    **{key: candidate[key] for key in ("candidate_index", "C", "gamma_multiplier", "gamma", "mean_fold_macro_recall")},
                    **{f"held_{fold['held_source_speed']}_macro_recall": fold["macro_recall"] for fold in folds}})
            chosen = sorted(candidates, key=lambda c: (-c["mean_fold_macro_recall"], c["C"], c["gamma_multiplier"]))[0]
            x, calibration = fit_representation(psd[obs_idx], frequency, v, y, method)
            svm = pipeline(chosen["C"], chosen["gamma_multiplier"])
            svm.fit(x, y)
            model_path = run / f"source/B/final_models/{observation}_{method}.joblib"
            bundle = {"calibrator": calibration, "classifier": svm, "method": method,
                      "observation": observation, "selected": chosen, "train_indices": original_indices,
                      "settings": settings("Cluster"), "training_feature_sha256": array_sha(x)}
            save_model(model_path, bundle)
            save_npz(run / f"source/B/final_features/{observation}_{method}.npz", features=x, labels=labels, original_indices=original_indices)
            selections.append({"observation": observation, "method": method, "selected": chosen,
                               "candidates": candidates, "model_path": str(model_path.relative_to(run)),
                               "model_sha256": sha(model_path), "source_n": len(y),
                               "calibration": None if calibration is None else calibration.fit_diagnostics})
            print(json.dumps({"stage": "source", "observation": observation, "method": method,
                              "selected_C": chosen["C"], "selected_gamma_multiplier": chosen["gamma_multiplier"],
                              "mean_fold_macro_recall": chosen["mean_fold_macro_recall"]}), flush=True)
    assert len(selections) == 6 and len(flat_candidates) == 54 and len(cv_predictions) == 3888
    save_json(run / "source/B/selection.json", selections)
    save_csv(run / "source/B/candidates.csv", flat_candidates)
    save_csv(run / "source/B/cv_predictions.csv", cv_predictions)
    save_json(run / "source/manifest.json", {"A": a_manifest, "B_final_models": 6,
              "B_candidates": 54, "B_cv_fits": 162, "B_cv_predictions": 3888,
              "B_source_selection_only": True, "no_B_target_prediction_yet": True})
    return {"A_models": 2, "B_models": 6, "candidates": 54, "cv_fits": 162}


def evaluate(run):
    seal = require_source_seal(run)
    assert not (run / "evaluation").exists(), "Evaluation cannot overwrite a prior attempt"
    save_json(run / "evaluation/target_access_guard.json", {"utc": now(), "source_seal_sha256": sha(run / "source_seal.json"),
              "source_artifacts_verified": len(seal["artifacts"]), "all_six_final_models_verified_before_target_loading": True})
    summary = []
    for record in read_json(run / "source/A/models.json"):
        name = record["design"]
        model = joblib.load(run / record["model_path"])
        data = np.load(run / f"source/A/{name}_construction.npz", allow_pickle=False)
        labels, z = data["labels"], data["log_spectra"]
        y, v, _ = labels.T
        src, target = data["source_indices"], data["target_indices"]
        before = model.theta.copy()
        source_corrected = model.transform(z[src], v[src])
        target_corrected = model.transform(z[target], v[target])
        assert np.array_equal(before, model.theta)
        save_npz(run / f"evaluation/A/{name}_corrected.npz", source_features=source_corrected,
                 target_features=target_corrected, source_indices=src, target_indices=target)
        metrics = response_diagnostic(model, z[src], v[src], y[src], CONFIG["designs"][name]["target_speeds"],
                                      "direct", run / f"evaluation/responses/A_{name}")
        summary.append({"part": "A", "design": name, "observation": "exact_additive", "method": "corrected",
                        "source_n": len(src), "target_n": len(target), **metrics})

    # First access to B held-out PSD after verification of all six model hashes.
    with np.load(INPUT, allow_pickle=False) as data:
        labels = data["labels"]
        _, target_mask = masks(labels, "Cluster")
        target_labels, target_indices = labels[target_mask], np.flatnonzero(target_mask)
        target_psd, frequency = data["psd"][:, target_mask], data["frequency"]
    assert target_psd.shape == (2, 96, 513)
    source_data = np.load(run / "source/B/input_source_only.npz", allow_pickle=False)
    source_y, source_v, _ = source_data["labels"].T
    target_y, target_v, _ = target_labels.T
    predictions, per_template, per_template_speed = [], [], []
    for selection in read_json(run / "source/B/selection.json"):
        observation, method = selection["observation"], selection["method"]
        obs_idx = CONFIG["observations_B"].index(observation)
        model_path = run / selection["model_path"]
        assert sha(model_path) == selection["model_sha256"]
        bundle = joblib.load(model_path)
        calibration = bundle["calibrator"]
        x_target = transform_representation(target_psd[obs_idx], frequency, target_v, method, calibration)
        predicted = bundle["classifier"].predict(x_target)
        rows = [{"part": "B", "observation": observation, "method": method, "index": int(idx),
                 "template": int(lab[0]), "speed": int(lab[1]), "repeat": int(lab[2]),
                 "prediction": int(pred), "correct": int(lab[0] == pred)}
                for idx, lab, pred in zip(target_indices, target_labels, predicted)]
        predictions.extend(rows)
        save_csv(run / f"evaluation/B/predictions/{observation}_{method}.csv", rows)
        save_npz(run / f"evaluation/B/features/{observation}_{method}.npz", features=x_target,
                 labels=target_labels, original_indices=target_indices, predicted=predicted)
        metrics = {}
        if calibration is not None:
            z_source = log_masses(source_data["psd"][obs_idx], frequency, source_v)
            metrics = response_diagnostic(calibration, z_source, source_v, source_y, [20, 60], observation,
                                           run / f"evaluation/responses/B_{observation}")
            metrics["mean_selected_fold_contrast_variance_explained"] = float(np.mean([
                fold["calibration"]["contrast_variance_explained"] for fold in selection["selected"]["folds"]]))
        summary.append({"part": "B", "design": "Cluster", "observation": observation, "method": method,
                        "source_n": 72, "target_n": 96, "C": selection["selected"]["C"],
                        "gamma_multiplier": selection["selected"]["gamma_multiplier"],
                        "source_cv_macro_recall": selection["selected"]["mean_fold_macro_recall"],
                        "target_macro_accuracy": macro_recall(target_y, predicted), **metrics})
        for template in range(12):
            mask = target_y == template
            per_template.append({"observation": observation, "method": method, "template": template,
                                 "n": int(mask.sum()), "correct": int(np.sum(predicted[mask] == template)),
                                 "accuracy": float(np.mean(predicted[mask] == template))})
            for speed in [20, 60]:
                mask = (target_y == template) & (target_v == speed)
                per_template_speed.append({"observation": observation, "method": method, "template": template,
                                           "speed": speed, "n": int(mask.sum()), "correct": int(np.sum(predicted[mask] == template)),
                                           "accuracy": float(np.mean(predicted[mask] == template))})
        assert sha(model_path) == selection["model_sha256"]
    assert len(predictions) == 576 and len(per_template) == 72 and len(per_template_speed) == 144
    save_csv(run / "evaluation/B/all_target_predictions.csv", predictions)
    save_csv(run / "evaluation/B/per_template.csv", per_template)
    save_csv(run / "evaluation/B/per_template_speed.csv", per_template_speed)
    corrected = [r for r in summary if r["part"] == "B" and r["method"] == "corrected"]
    aggregates = {"mean_B_final_source_contrast_variance_explained": float(np.mean([r["source_contrast_variance_explained"] for r in corrected])),
                  "mean_B_selected_CV_fold_contrast_variance_explained": float(np.mean([r["mean_selected_fold_contrast_variance_explained"] for r in corrected]))}
    save_json(run / "evaluation/summary.json", {"status": "completed_pending_independent_check", "records": summary,
              "aggregates": aggregates, "scope": CONFIG["scope"], "population_inference": False,
              "gauge_applied_to_features": False, "construction_changed_after_results": False})
    save_csv(run / "evaluation/summary.csv", summary)
    # The LaTeX table reads the just-written machine summary, not transcribed values.
    rows = read_json(run / "evaluation/summary.json")["records"]
    latex = ["% Machine generated by calibrator_diagnostic.py from evaluation/summary.json.",
             "\\begin{tabular}{lllrrr}", "\\hline",
             "Part/design & Observation & Method & Target acc. & Contrast RMSE & Target gauge RMSE \\\\", "\\hline"]
    for row in rows:
        label = row["observation"].replace("_", "\\_")
        acc = "--" if "target_macro_accuracy" not in row else f"{100 * row['target_macro_accuracy']:.2f}\\%"
        contrast = "--" if "source_contrast_error_rms" not in row else f"{row['source_contrast_error_rms']:.5f}"
        gauge = "--" if "target_response_rms_gauge_adjusted" not in row else f"{row['target_response_rms_gauge_adjusted']:.5f}"
        latex.append(f"{row['part']}/{row['design']} & {label} & {row['method']} & {acc} & {contrast} & {gauge} \\\\")
    latex.extend(["\\hline", "\\end{tabular}", "% All accuracies are descriptive template macro recall. B response truth is a pointwise finite-PSD reference."])
    write_bytes(run / "evaluation/summary_table.tex", ("\n".join(latex) + "\n").encode())
    require_source_seal(run)
    return {"target_predictions": len(predictions), "summary_rows": len(summary), **aggregates}


def verify(run):
    source_seal = require_source_seal(run)
    result_seal = read_json(run / "result_seal.json")
    assert result_seal["source_seal_sha256"] == sha(run / "source_seal.json")
    for record in result_seal["artifacts"]:
        verify_record(run, record)
    assert len(list((run / "source/B/final_models").glob("*.joblib"))) == 6
    assert len(list((run / "source/B/cv_models").glob("*.joblib"))) == 162
    with (run / "source/B/candidates.csv").open() as handle:
        candidates = list(csv.DictReader(handle))
    with (run / "source/B/cv_predictions.csv").open() as handle:
        cv_rows = list(csv.DictReader(handle))
    with (run / "evaluation/B/all_target_predictions.csv").open() as handle:
        predicted_rows = list(csv.DictReader(handle))
    assert len(candidates) == 54 and len(cv_rows) == 3888 and len(predicted_rows) == 576
    summary = read_json(run / "evaluation/summary.json")["records"]
    for row in summary:
        if row["part"] == "B":
            selected = [p for p in predicted_rows if p["observation"] == row["observation"] and p["method"] == row["method"]]
            assert len(selected) == 96
            y = np.array([int(p["template"]) for p in selected])
            p = np.array([int(p["prediction"]) for p in selected])
            assert macro_recall(y, p) == row["target_macro_accuracy"]
    checks = ["source/model seal reverified", "result artifact hashes reverified", "six final and 162 fold models retained",
              "54 candidates, 3888 CV and 576 target predictions retained", "all target macro recalls recomputed from prediction rows"]
    save_json(run / "verification.json", {"utc": now(), "status": "internal_checks_passed_pending_independent_checker",
              "checks": checks, "source_seal_sha256": sha(run / "source_seal.json"),
              "result_seal_sha256": sha(run / "result_seal.json"), "active_seconds_prior_to_verification": budget_spent(run)})
    return {"status": "passed", "checks": len(checks), "independent_checker": "pending"}


def seal_source(run):
    artifacts = inventory(run, ["source", "code"])
    for name in ("config.json", "initialization.json", "environment.json", "test_report.json"):
        artifacts.append(record_file(run / name, run))
    save_json(run / "source_seal.json", {"utc": now(), "run_id": run.name, "input_hashes": checked_inputs(),
              "configuration_sha256": sha(run / "config.json"), "final_model_count": 6,
              "before_any_B_target_prediction": True, "artifacts": artifacts})


def seal_results(run):
    save_json(run / "result_seal.json", {"utc": now(), "run_id": run.name,
              "source_seal_sha256": sha(run / "source_seal.json"), "input_hashes": checked_inputs(),
              "status": "completed_pending_independent_check", "artifacts": inventory(run, ["evaluation"]),
              "active_seconds_at_result_freeze": budget_spent(run)})


def execute_stage(stage, run):
    initialize(run)
    already = run / f"timing/{stage}.json"
    assert not already.exists(), f"Stage already ran: {stage}; preserve this run and use verify or a new run directory"
    spent = budget_spent(run)
    remaining = CONFIG["budget_active_seconds"] - spent
    assert remaining > 0, "Active compute budget exhausted"
    request = {"stage": stage, "utc": now(), "argv": sys.argv, "cwd": os.getcwd(),
               "command": shlex.join([sys.executable, *sys.argv]), "prior_active_seconds": spent,
               "remaining_active_seconds": remaining, "input_hashes": checked_inputs()}
    save_json(run / f"commands/{stage}.json", request)
    def timeout(_signum, _frame):
        raise TimeoutError("Frozen 300 active second budget exhausted")
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, max(0.01, remaining - (time.monotonic() - BOOT_TIME)))
    start_cpu = time.process_time()
    status, result = "failed", None
    try:
        result = {"test": tests, "source": source, "evaluate": evaluate, "verify": verify}[stage](run)
        status = "complete"
    except BaseException:
        failure = {"stage": stage, "utc": now(), "traceback": traceback.format_exc(), "input_hashes": checked_inputs()}
        save_json(run / f"failures/{stage}_{uuid.uuid4().hex}.json", failure)
        print(json.dumps({"stage": stage, "status": "failed", "run_dir": str(run)}), flush=True)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        elapsed = time.monotonic() - BOOT_TIME
        save_json(already, {"stage": stage, "status": status, "utc": now(), "active_seconds": elapsed,
                  "process_cpu_seconds": time.process_time() - start_cpu,
                  "cumulative_active_seconds": spent + elapsed, "budget_active_seconds": 300})
    if stage == "source":
        seal_source(run)
    elif stage == "evaluate":
        seal_results(run)
    save_json(run / f"logs/{stage}_completion.json", {"utc": now(), "status": status, "result": result,
              "stage_timing_sha256": sha(already)})
    print(json.dumps({"stage": stage, "status": status, "run_dir": str(run), "result": result,
                      "cumulative_active_seconds": budget_spent(run)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["test", "source", "evaluate", "verify", "all"])
    parser.add_argument("--run-dir", type=pathlib.Path, default=ROOT / "experiments/calibrator_diagnostic/run_001")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if args.stage == "all":
        assert not run.exists(), "An all-stage replay requires a fresh directory"
        for stage in ("test", "source", "evaluate", "verify"):
            subprocess.run([sys.executable, str(CODE), stage, "--run-dir", str(run)], check=True)
        return
    with threadpool_limits(limits=2):
        assert all(pool["num_threads"] <= 2 for pool in threadpool_info())
        execute_stage(args.stage, run)


if __name__ == "__main__":
    main()
