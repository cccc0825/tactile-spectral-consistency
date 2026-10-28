#!/usr/bin/env python3
"""Record one explicit resource allowance for a fixed, separately prepared run.

This does not alter scientific producers, repair failed runs, resume training,
or establish that an operating-system process has stopped. The operator must
establish a zero-running-writer boundary before using ``allow``.
"""
from __future__ import annotations
import argparse
import copy
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CNN = Path('outputs/research_v2/experiments/cnn')
LEDGER = CNN / 'compute_budget.json'
DIRECTORY = Path('reproduction/resource_allowance')
ORIGINAL_LIMIT = 10800
MAXIMUM_LIMIT = 21600
LINK = 'reproduction_resource_allowance'
FILES = {'ledger_before.json', 'allowance.json', 'ledger_applied.json', 'application.json'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def inside(root, name):
    name = Path(name)
    require(not name.is_absolute() and '..' not in name.parts, 'Unsafe relative path')
    path = root / name
    require(path.resolve().is_relative_to(root), 'Path leaves study root')
    return path


def write_new(path, data):
    with path.open('xb') as stream:
        stream.write(data)


def finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def ledger_valid(data):
    require(isinstance(data, dict) and 'active_session' in data and data['active_session'] is None,
            'Active session: establish a zero-running-writer boundary first')
    require(type(data.get('limit_seconds')) is int and data['limit_seconds'] > 0,
            'Invalid numeric compute limit')
    require(finite_nonnegative(data.get('consumed_seconds')), 'Invalid consumed time')
    history = data.get('history')
    require(isinstance(history, list), 'Missing compute history')
    for row in history:
        require(isinstance(row, dict) and finite_nonnegative(row.get('elapsed_seconds')),
                'Invalid elapsed time in history')
        require(row.get('status') in ('complete', 'failed') and row.get('label'),
                'Invalid phase status or label')
    require(abs(sum(r['elapsed_seconds'] for r in history) - data['consumed_seconds']) <= 1e-3,
            'Consumed time does not equal closed-session history; recovery is not supported')
    require(data['consumed_seconds'] <= data['limit_seconds'],
            'Compute limit already exceeded; preserve failure, this is not a recovery tool')


def no_target(root, ledger):
    forbidden = list((root / 'outputs/research/experiments/raw_results').glob('*'))
    for dataset in ('cluster', 'external'):
        cnn = root / CNN / dataset
        svm = root / 'outputs/research_v2/experiments/calibration' / dataset
        forbidden += list((cnn / 'predictions').glob('*'))
        forbidden += list(svm.glob('*_predictions.csv')) + list(svm.glob('failure_evaluate*'))
        forbidden += [p for p in (cnn / 'summary.json', cnn / 'external_global_release.json',
                                  svm / 'evaluation_complete.json') if p.exists()]
    global_seal = root / 'outputs/research_v2/research/external_global_model_freeze.json'
    if global_seal.exists():
        forbidden.append(global_seal)
    require(not forbidden, 'Target output or global release/seal already exists')
    require(not any('evaluate' in r['label'] for r in ledger['history']),
            'Target execution appears in compute history')


def input_bindings(root):
    """Validate immutable prepared sources, configs and their actual inputs."""
    result = {}

    def bind(name, expected=None):
        name = str(name)
        path = inside(root, name)
        require(path.is_file(), 'Missing bound input: ' + name)
        actual = result.get(name) or sha(path)
        require(expected is None or actual == expected, 'Input hash mismatch: ' + name)
        result[name] = actual

    for name, kind, field in [('FRESH_RUN.json', 'new_source_fit_tree', 'source_files_sha256'),
                              ('STUDY_SETUP.json', 'new_full_source_study_tree', 'files_sha256')]:
        bind(name)
        record = read(root / name)
        require(record.get('kind') == kind and record.get(field), 'Invalid prepared source manifest')
        source = record.get('original_source_root', record.get('source_root'))
        require(source and Path(source).resolve() != root, 'Not a separately prepared study')
        for path, expected in record[field].items():
            bind(path, expected)
    auth_path = CNN / 'compute_budget_amendment_02.json'
    applied_path = CNN / 'compute_budget_amendment_02.applied.json'
    bind(auth_path); bind(applied_path)
    auth, applied = read(root / auth_path), read(root / applied_path)
    require(auth.get('status') == 'new_run_same_frozen_resource_bound'
            and applied.get('status') == 'applied_to_new_run', 'Unexpected original amendment status')
    require(all(r.get('previous_limit_seconds') == 9000 and r.get('new_limit_seconds') == ORIGINAL_LIMIT
                for r in (auth, applied)), 'Unknown original amendment limits')
    require(applied.get('authorization_sha256') == result[str(auth_path)],
            'Original amendment receipt mismatch')
    require(isinstance(auth.get('source_addendum_sha256'), str), 'Missing original addendum digest')
    bind(CNN / 'external_baseline_protocol.md', auth['source_addendum_sha256'])
    for dataset, folds, classes, rows, source in [('cluster', ['30', '40', '50'], 118, 18880, 1416),
                                                ('external', ['0', '1', '2'], 12, 3550, 2354)]:
        config_path = CNN / f'{dataset}_config.json'
        bind(config_path)
        cfg = read(root / config_path)
        expected = dict(dataset=dataset, folds=folds, num_classes=classes, expected_rows=rows,
                        expected_source_rows=source, learning_rates=[0.0001, 0.0003, 0.001],
                        max_epochs=6, batch_size=16, weight_decay=0.0001, cv_seed=20270913,
                        final_seeds=[20270913, 20270914, 20270915], optimizer='Adam',
                        scheduler=None, augmentation=None,
                        selection='largest mean source-CV accuracy; ties earlier epoch, then smaller learning rate')
        require(all(key in cfg and cfg[key] == value for key, value in expected.items()),
                'Frozen scientific settings differ: ' + dataset)
        for key in ('input', 'metadata', 'data_manifest', 'weights'):
            require(cfg.get(key + '_sha256'), 'Missing configured input digest')
            bind(cfg[key + '_file'], cfg[key + '_sha256'])
        for path in cfg.get('additional_context_files', []):
            bind(path)
        context = root / CNN / dataset / 'selection_context.json'
        if context.exists():
            # A later source selection may create this file; bind its input map
            # now, without requiring the optional context file to exist later.
            for path, expected_hash in read(context)['files_sha256'].items():
                bind(path, expected_hash)
    return result


def allow(root, limit, reason, acknowledge):
    root = root.resolve()
    require(acknowledge is True, 'Explicit --acknowledge-no-running-writers is required')
    require(type(limit) is int and ORIGINAL_LIMIT < limit <= MAXIMUM_LIMIT,
            'New finite allowance must be above 10800 and at most 21600 seconds')
    require(isinstance(reason, str) and 8 <= len(reason.strip()) <= 2000, 'Provide a concrete resource reason')
    directory = inside(root, DIRECTORY)
    require(not directory.exists(), 'Allowance directory already exists; no duplicate or automatic repair')
    ledger_path = inside(root, LEDGER)
    original_bytes = ledger_path.read_bytes()
    old = json.loads(original_bytes)
    ledger_valid(old)
    require(old['limit_seconds'] == ORIGINAL_LIMIT and LINK not in old,
            'Unknown original limit or existing allowance link')
    no_target(root, old)
    bindings = input_bindings(root)
    record = dict(kind='fixed_operations_reproduction_resource_allowance_v1', created_utc=now(),
                  original_limit_seconds=ORIGINAL_LIMIT, new_limit_seconds=limit,
                  consumed_seconds_preserved=old['consumed_seconds'],
                  original_ledger_sha256=digest(original_bytes), inputs_sha256=bindings,
                  operator_acknowledged_zero_running_writers=True,
                  no_target_outputs_or_global_seal_at_application=True,
                  reason=reason.strip(), scientific_settings_changed=False,
                  scope='New local reproduction allowance only; historical protocols and original results are unchanged.',
                  tool_sha256=sha(Path(__file__).resolve()))
    receipt_bytes = encoded(record)
    new = copy.deepcopy(old)
    new.update(limit_seconds=limit, last_update_utc=record['created_utc'])
    new[LINK] = dict(receipt=str(DIRECTORY / 'allowance.json'), sha256=digest(receipt_bytes),
                     original_limit_seconds=ORIGINAL_LIMIT, new_limit_seconds=limit)
    new_bytes = encoded(new)
    application = dict(kind='resource_allowance_application_v1', created_utc=now(),
                       allowance_sha256=digest(receipt_bytes),
                       ledger_before_sha256=digest(original_bytes), ledger_applied_sha256=digest(new_bytes),
                       changed_fields=['limit_seconds', 'last_update_utc', LINK])
    # Exclusive directory creation prevents a second utility invocation. It is
    # deliberately not a lock understood by the unchanged trainer: the operator
    # must keep the scheduler stopped until this command finishes.
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()
    write_new(directory / 'ledger_before.json', original_bytes)
    write_new(directory / 'allowance.json', receipt_bytes)
    write_new(directory / 'ledger_applied.json', new_bytes)
    require(ledger_path.read_bytes() == original_bytes, 'Ledger changed during preparation; do not apply')
    no_target(root, read(ledger_path))
    temporary = ledger_path.with_name(ledger_path.name + '.resource-allowance-new')
    write_new(temporary, new_bytes)
    temporary.replace(ledger_path)
    write_new(directory / 'application.json', encoded(application))
    return check(root, pre_target=True)


def check(root, pre_target=False):
    root = root.resolve()
    current = read(inside(root, LEDGER))
    ledger_valid(current)
    bindings = input_bindings(root)
    directory = inside(root, DIRECTORY)
    if not directory.exists():
        require(LINK not in current and current['limit_seconds'] == ORIGINAL_LIMIT,
                'Missing allowance chain for non-original limit or ledger link')
        effective = ORIGINAL_LIMIT
        allowance_hash = None
    else:
        require({p.name for p in directory.iterdir()} == FILES, 'Incomplete or unexpected allowance chain files')
        old = read(directory / 'ledger_before.json')
        applied = read(directory / 'ledger_applied.json')
        receipt = read(directory / 'allowance.json')
        application = read(directory / 'application.json')
        ledger_valid(old); ledger_valid(applied)
        require(old['limit_seconds'] == ORIGINAL_LIMIT and LINK not in old, 'Unknown original allowance state')
        require(receipt.get('kind') == 'fixed_operations_reproduction_resource_allowance_v1'
                and application.get('kind') == 'resource_allowance_application_v1', 'Unexpected allowance schema')
        effective = receipt.get('new_limit_seconds')
        require(type(effective) is int and ORIGINAL_LIMIT < effective <= MAXIMUM_LIMIT,
                'Invalid recorded allowance')
        allowance_hash = sha(directory / 'allowance.json')
        require(receipt.get('original_limit_seconds') == ORIGINAL_LIMIT
                and receipt.get('consumed_seconds_preserved') == old['consumed_seconds']
                and receipt.get('original_ledger_sha256') == sha(directory / 'ledger_before.json'),
                'Original ledger binding or preserved time differs')
        require(receipt.get('operator_acknowledged_zero_running_writers') is True
                and receipt.get('no_target_outputs_or_global_seal_at_application') is True
                and receipt.get('scientific_settings_changed') is False, 'Missing boundary or fixed-operation declaration')
        require(receipt.get('tool_sha256') == sha(Path(__file__).resolve()), 'Resource tool changed since allowance')
        require(receipt.get('inputs_sha256') and all(bindings.get(p) == h for p, h in receipt['inputs_sha256'].items()),
                'Bound source/config/input changed since allowance')
        require(application.get('allowance_sha256') == allowance_hash
                and application.get('ledger_before_sha256') == sha(directory / 'ledger_before.json')
                and application.get('ledger_applied_sha256') == sha(directory / 'ledger_applied.json'),
                'Allowance application hash mismatch')
        expected = copy.deepcopy(old)
        expected.update(limit_seconds=effective, last_update_utc=receipt['created_utc'])
        expected[LINK] = dict(receipt=str(DIRECTORY / 'allowance.json'), sha256=allowance_hash,
                             original_limit_seconds=ORIGINAL_LIMIT, new_limit_seconds=effective)
        require(applied == expected, 'Applied ledger changed fields beyond the resource allowance')
        require(current['limit_seconds'] == effective and current.get(LINK) == expected[LINK],
                'Current ledger has a contradictory or missing allowance')
        require(current['history'][:len(old['history'])] == old['history']
                and current['consumed_seconds'] >= old['consumed_seconds'], 'Historical time or phase history was reset')
        mutable = {'consumed_seconds', 'history', 'active_session', 'last_update_utc', 'last_resource'}
        require({k: v for k, v in current.items() if k not in mutable}
                == {k: v for k, v in applied.items() if k not in mutable}, 'Unexpected ledger fields changed')
    if pre_target:
        no_target(root, current)
    seal = root / 'outputs/research_v2/research/external_global_model_freeze.json'
    if seal.exists():
        frozen = read(seal)
        require(frozen.get('budget_limit_seconds') == effective
                and finite_nonnegative(frozen.get('budget_consumed_seconds_before_target'))
                and frozen['budget_consumed_seconds_before_target'] <= current['consumed_seconds'],
                'Global seal resource fields disagree with current ledger')
    return dict(status='resource_chain_valid', check_scope='resources only; not proof of scientific completion or process inactivity',
                original_limit_seconds=ORIGINAL_LIMIT, effective_limit_seconds=effective,
                consumed_seconds=current['consumed_seconds'],
                original_180_minutes_exceeded=current['consumed_seconds'] > ORIGINAL_LIMIT,
                allowance_sha256=allowance_hash, pre_target_checked=pre_target,
                failed_sessions=sum(r['status'] == 'failed' for r in current['history']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['allow', 'check'])
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--limit-seconds', type=int)
    parser.add_argument('--reason')
    parser.add_argument('--acknowledge-no-running-writers', action='store_true')
    parser.add_argument('--pre-target', action='store_true')
    args = parser.parse_args()
    try:
        if args.command == 'allow':
            result = allow(args.project_root, args.limit_seconds, args.reason, args.acknowledge_no_running_writers)
        else:
            require(args.limit_seconds is None and args.reason is None and not args.acknowledge_no_running_writers,
                    'Allowance options are only valid with allow')
            result = check(args.project_root, args.pre_target)
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({'status': 'refused', 'error': str(exc)}), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
