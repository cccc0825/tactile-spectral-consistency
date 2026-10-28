"""Seal every externally locked source model before any external target inference.

This script only verifies hashes and records a local gate. It never fits a model,
reads classifier target predictions, or alters an earlier freeze.
"""
import argparse
import datetime
import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT.parents[1]
DEST = ROOT / 'research/external_global_model_freeze.json'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def main(write=False):
    if DEST.exists():
        raise RuntimeError('Global freeze already exists; it must not be overwritten.')
    svm_dir = ROOT / 'experiments/calibration/external'
    cnn_dir = ROOT / 'experiments/cnn/external'
    if list(svm_dir.glob('*_predictions.csv')) or (svm_dir / 'evaluation_complete.json').exists():
        raise RuntimeError('External SVM target outputs already exist before this global gate.')
    pred_dir = cnn_dir / 'predictions'
    if pred_dir.exists() and any(pred_dir.iterdir()):
        raise RuntimeError('External CNN target output directory is not empty before this gate.')
    if (cnn_dir / 'summary.json').exists():
        raise RuntimeError('External CNN summary already exists before this global gate.')
    paths = [svm_dir / 'freeze.json', cnn_dir / 'freeze.json',
             cnn_dir / 'source_eval_audit.json',
             ROOT / 'audits/replay/external_data_acceptance_receipt.json',
             ROOT / 'audits/replay/external_svm_source_acceptance_receipt.json']
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.exists()]
    if missing:
        raise RuntimeError('Required source seals/audits not complete: ' + ', '.join(missing))
    svm, cnn, source_audit, data_audit, svm_audit = [json.loads(p.read_text()) for p in paths]
    assert len(svm['method_order']) == 9
    assert len(list(svm_dir.glob('*.joblib'))) == 9
    assert cnn['status'] == 'all_source_selection_and_final_models_sealed_before_target_inference'
    assert cnn['target_prediction_started'] is False
    assert cnn['model_seeds'] == [20270913, 20270914, 20270915]
    assert source_audit['status'] == 'complete'
    assert source_audit['target_inputs_or_metrics_used'] is False
    assert source_audit['models_or_selection_changed'] is False
    assert {r['seed'] for r in source_audit['records']} == set(cnn['model_seeds'])
    assert all(r['saved_model_unchanged'] for r in source_audit['records'])
    assert source_audit['selection_freeze_sha256'] == sha(cnn_dir / 'freeze.json')
    assert data_audit['status'] == 'pass_data_acceptance_only'
    assert data_audit['external_classification_outcomes_read'] is False
    assert svm_audit['status'] == 'pass_source_selection_only'
    assert svm_audit['source_cv_fits'] == 405 and svm_audit['source_final_fits'] == 9
    assert svm_audit['all_final_model_parameters_bitwise_equal'] is True
    assert svm_audit['external_target_rows_used_for_features_fitting_or_inference'] == 0
    assert svm_audit['external_target_outcome_files_read'] is False
    integrity_path = cnn_dir / 'source_integrity_check.json'
    integrity = json.loads(integrity_path.read_text())
    assert integrity['phase'] == 'source' and integrity['status'] == 'pass'
    assert all(integrity['checks'].values())
    assert integrity['target_predictions_started'] is False
    assert integrity['selection_freeze_sha256'] == sha(cnn_dir / 'freeze.json')
    paths.append(integrity_path)
    files = {}

    def verify(path, expected):
        path = path.resolve()
        actual = sha(path)
        if actual != expected:
            raise ValueError('Digest mismatch: ' + str(path))
        key = str(path.relative_to(BASE))
        if key in files and files[key] != expected:
            raise ValueError('Conflicting source seals: ' + key)
        files[key] = expected

    for name, digest in svm['inputs'].items():
        verify(pathlib.Path(name), digest)
    for name, digest in svm['files'].items():
        verify(svm_dir / name, digest)
    for mapping in [cnn['context']['files_sha256'], cnn['artifacts_sha256'],
                    data_audit['audited_current_inputs_sha256']]:
        for name, digest in mapping.items():
            verify(BASE / name, digest)
    for rec in source_audit['records']:
        verify(BASE / rec['raw_prediction_file'], rec['raw_prediction_sha256'])
    for name, digest in data_audit['audit_artifacts_sha256'].items():
        verify(ROOT / 'audits/replay' / name, digest)
    for name, digest in svm_audit['audit_artifacts_sha256'].items():
        verify(ROOT / 'audits/replay' / name, digest)
    for name, digest in svm_audit['source_freeze_and_inputs_sha256'].items():
        verify(pathlib.Path(name), digest)
    for path in paths + [pathlib.Path(__file__)]:
        verify(path, sha(path))
    budget = json.loads((ROOT / 'experiments/cnn/compute_budget.json').read_text())
    assert budget['consumed_seconds'] <= budget['limit_seconds']
    record = {
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'stage': 'all source-only external selections, final models and pre-target audits sealed',
        'scope': 'local protocol record; not an external preregistration or physical observation attestation',
        'external_target_outputs_present': False,
        'svm_models': 9, 'cnn_models': 3,
        'primary': 'corrected minus spatial specimen-macro accuracy >= 3 pp and conditional paired bootstrap lower bound > 0',
        'secondary': 'corrected minus log-mel, gated on primary support',
        'no_target_dependent_changes_authorized': True,
        'budget_consumed_seconds_before_target': budget['consumed_seconds'],
        'budget_limit_seconds': budget['limit_seconds'],
        'verified_files_sha256': files,
        'release': 'One external target evaluation for every sealed model; preserve every outcome.'
    }
    if write:
        with DEST.open('x') as f:
            json.dump(record, f, indent=2)
            f.write('\n')
    print(json.dumps({'status': 'sealed' if write else 'ready_not_written',
                      'verified_files': len(files), 'path': str(DEST),
                      'sha256': sha(DEST) if write else None}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    main(parser.parse_args().write)
