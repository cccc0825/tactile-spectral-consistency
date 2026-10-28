#!/usr/bin/env python3
"""Execute fresh source audits and bind successful technical receipts to this run.

These receipts are local executable checks, not independent peer review. Failed
or interrupted attempts are preserved and never produce an acceptance receipt.
"""
from __future__ import annotations
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from prepare_study import ROOT, V2, read, require, safe_path, sha, verify_study, write_new

SCOPES = {
    'data': ('audit_external_data.py', 'external_data_audit', 'external_data_acceptance_receipt.json'),
    'svm': ('audit_external_svm_source.py', 'external_svm_source_audit', 'external_svm_source_acceptance_receipt.json'),
}

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def forbid_target_outputs(root):
    svm = root / V2 / 'experiments/calibration/external'
    cnn = root / V2 / 'experiments/cnn/external'
    forbidden = list(svm.glob('*_predictions.csv')) + list(svm.glob('failure_evaluate*'))
    forbidden += list((cnn / 'predictions').glob('*'))
    forbidden += [p for p in [svm / 'evaluation_complete.json', cnn / 'summary.json',
                              cnn / 'external_global_release.json',
                              root / V2 / 'research/external_global_model_freeze.json'] if p.exists()]
    require(not forbidden, 'External target exposure/release already present; source audit is closed')
    budget = root / V2 / 'experiments/cnn/compute_budget.json'
    if budget.exists():
        ledger = read(budget)
        require(not any(r.get('label') == 'evaluate_external' for r in ledger.get('history', [])),
                'Earlier external target execution appears in compute history')
        active = ledger.get('active_session')
        require(not active, 'A CNN process is still active; source audit requires a stage boundary')

def bound_path(root, path):
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    require(path.resolve().is_relative_to(root.resolve()), 'Audit input escapes fresh study tree')
    return path

def data_inputs(root):
    names = [
        'data/external/processed/manifest.json', 'data/external/zip_index.json',
        'data/external/quality_audit.json', 'data/external/trial_integrity.json',
        'data/external/final_eligibility_audit.json', 'data/external/acquisition_plan.json',
        'research/external_validity_decision.json', 'research/external_validity_decision.freeze.json',
        'research/external_protocol.md', 'research/external_protocol.freeze.json',
        'experiments/cnn/external_split_preflight.json',
        'experiments/calibration/external/external40_split.json',
    ]
    return {str(V2 / name): sha(root / V2 / name) for name in names}

def svm_inputs(root):
    directory = root / V2 / 'experiments/calibration/external'
    freeze = directory / 'freeze.json'
    result = {str(freeze): sha(freeze)}
    record = read(freeze)
    for group in ('inputs', 'files'):
        for name, digest in record[group].items():
            path = bound_path(root, directory / name if group == 'files' else name)
            require('prediction' not in path.name and 'evaluation' not in path.name,
                    'Unexpected target artifact in source freeze')
            require(sha(path) == digest, 'Source freeze hash mismatch: ' + name)
            result[str(path)] = digest
    return result

def validate_checks(report, script):
    require(report['script_sha256'] == sha(script), 'Audit reports a different producer')
    require(report.get('failed_checks') == [], 'Audit has failed checks')
    checks = report.get('checks', [])
    require(checks and all(c.get('passed') is True for c in checks), 'Audit checks absent or not all passed')
    require(report.get('checks_total', report.get('check_count')) == len(checks), 'Audit count mismatch')

def validate_report(root, stage, report, script, before):
    validate_checks(report, script)
    for name, digest in before.items():
        require(sha(bound_path(root, name)) == digest, 'Audit input changed during execution: ' + name)
    if stage == 'data':
        require(report['status'] == 'pass_data_acceptance_only', 'Data audit did not pass')
        for key, expected in [('raw_episodes_reparsed', 3600), ('eligible', 3550),
                              ('source', 2354), ('target', 1196), ('quality_exclusions', 49),
                              ('additional_exact_duplicate_exclusions', 1)]:
            require(report[key] == expected, 'Unexpected data audit count: ' + key)
        require(len(report['independent_spectral_spots']) == 36 and len(report['folds']) == 8,
                'Incomplete data replay coverage')
        hashes = report['hash_checks']
        require(hashes and all(h['passed'] is True for h in hashes), 'Incomplete data hash checks')
        bindings = dict(before)
        for h in hashes:
            path = bound_path(root, h['path'])
            require(h['expected'] == h['actual'] == sha(path), 'Data hash changed since audit')
            bindings[str(path.relative_to(root))] = h['actual']
        return {'status': 'pass_data_acceptance_only', 'exit_code': 0,
                'external_classification_outcomes_read': False,
                'audited_current_inputs_sha256': bindings}
    require(report['status'] == 'pass_source_selection_only', 'SVM source audit did not pass')
    calls = report['calls']
    require(calls['source_cv_fits'] == calls['source_cv_predictions'] == 405,
            'Incomplete source CV replay')
    require(calls['source_final_fits'] == 9, 'Incomplete final refits')
    require(calls['external_target_inferences'] == calls['external_target_outcome_files_read'] == 0,
            'Target outcome use was reported')
    scope = report['read_scope']
    require(scope['source_rows_used'] == 2354 and scope['target_rows_used_for_features_fits_or_predictions'] == 0
            and scope['target_outcome_files_read'] is False, 'Source-only scope not established')
    models = report['final_model_replays']
    require(len(models) == 9 and all(r['all_parameters_bitwise_equal'] is True for r in models),
            'Final models do not match source replay')
    require(report['observed_inputs_and_source_artifacts_sha256'] == before,
            'Observed source artifacts differ from pre-execution bindings')
    require(report['freeze_hash_checks'] and all(h['passed'] is True for h in report['freeze_hash_checks']),
            'SVM freeze checks failed')
    return {'status': 'pass_source_selection_only', 'successful_exit_code': 0,
            'source_cv_fits': 405, 'source_final_fits': 9,
            'all_final_model_parameters_bitwise_equal': True,
            'external_target_rows_used_for_features_fitting_or_inference': 0,
            'external_target_outcome_files_read': False, 'read_scope': scope,
            'source_freeze_and_inputs_sha256': before}

def validate_data_receipt(root):
    directory = root / V2 / 'audits/replay'
    path = directory / SCOPES['data'][2]
    receipt = read(path)
    require(receipt.get('receipt_kind') == 'fresh_local_technical_check', 'Not a new local data audit receipt')
    require(receipt['status'] == 'pass_data_acceptance_only', 'Data audit not accepted')
    for name, digest in receipt['audited_current_inputs_sha256'].items():
        require(sha(bound_path(root, name)) == digest, 'Accepted data changed')
    for name, digest in receipt['audit_artifacts_sha256'].items():
        require(sha(safe_path(directory, name)) == digest, 'Accepted data audit artifact changed')
    return sha(path)

def execute(root, stage):
    verify_study(root)
    forbid_target_outputs(root)
    directory = root / V2 / 'audits/replay'
    script_name, stem, receipt_name = SCOPES[stage]
    script = directory / script_name
    outputs = [directory / (stem + suffix) for suffix in ('.json', '.md', '.log', '.execution.json')]
    receipt_path = directory / receipt_name
    require(not any(p.exists() for p in outputs + [receipt_path]),
            'Audit output already exists; preserve it and investigate before preparing another run')
    prior = validate_data_receipt(root) if stage == 'svm' else None
    before = data_inputs(root) if stage == 'data' else svm_inputs(root)
    command = [sys.executable, '-B', str(script)]
    started = time.monotonic()
    execution = {'started_utc': now(), 'stage': stage, 'command': command,
                 'cwd': str(root), 'script_sha256': sha(script), 'status': 'running',
                 'target_outputs_absent_before': True, 'wrapper_sha256': sha(__file__)}
    accepted = None
    try:
        with outputs[2].open('x') as stream:
            result = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=600 if stage == 'data' else 540)
        execution['exit_code'] = result.returncode
        require(result.returncode == 0, 'Audit process failed; inspect preserved log')
        forbid_target_outputs(root)
        verify_study(root)
        accepted = validate_report(root, stage, read(outputs[0]), script, before)
        if prior is not None:
            require(validate_data_receipt(root) == prior, 'Prior data acceptance changed')
            accepted['prior_external_data_receipt_sha256'] = prior
        execution.update(status='passed', target_outputs_absent_after=True)
    except BaseException:
        execution.update(status='failed', exception=traceback.format_exc())
        raise
    finally:
        execution.update(ended_utc=now(), elapsed_seconds=time.monotonic() - started)
        if outputs[2].exists():
            execution['log_sha256'] = sha(outputs[2])
        write_new(outputs[3], execution)
    accepted.update(utc=now(), receipt_kind='fresh_local_technical_check',
                    scope='Executed local technical replay on this fresh tree; not independent peer review or target performance validation.',
                    active_seconds=execution['elapsed_seconds'], command=command,
                    audit_artifacts_sha256={p.name: sha(p) for p in [script] + outputs},
                    next_gate='Run the unchanged global SVM/CNN source seal before any external target evaluation.')
    write_new(receipt_path, accepted)
    print(json.dumps({'status': accepted['status'], 'receipt': str(receipt_path),
                      'receipt_kind': accepted['receipt_kind']}, indent=2))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=SCOPES)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    args = parser.parse_args()
    execute(args.project_root.resolve(), args.stage)

if __name__ == '__main__':
    main()
