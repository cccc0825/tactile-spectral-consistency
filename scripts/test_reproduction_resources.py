#!/usr/bin/env python3
"""Resource-allowance guards on artificial records; no training or real ledgers."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import reproduction_resources as resources


class ResourceAllowanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.cnn = self.root / resources.CNN
        self.cnn.mkdir(parents=True)
        self.file('scientific_producer.py', b'# fixture only\n')
        protocol = resources.CNN / 'external_baseline_protocol.md'
        self.file(protocol, b'fixture protocol, not a scientific observation\n')
        auth = dict(status='new_run_same_frozen_resource_bound', previous_limit_seconds=9000,
                    new_limit_seconds=10800, source_addendum_sha256=resources.sha(self.root / protocol))
        self.put(resources.CNN / 'compute_budget_amendment_02.json', auth)
        self.put(resources.CNN / 'compute_budget_amendment_02.applied.json',
                 dict(status='applied_to_new_run', previous_limit_seconds=9000, new_limit_seconds=10800,
                      authorization_sha256=resources.sha(self.cnn / 'compute_budget_amendment_02.json')))
        for dataset, folds, classes, rows, source in [('cluster', ['30', '40', '50'], 118, 18880, 1416),
                                                    ('external', ['0', '1', '2'], 12, 3550, 2354)]:
            cfg = dict(dataset=dataset, folds=folds, num_classes=classes, expected_rows=rows,
                       expected_source_rows=source, learning_rates=[0.0001, 0.0003, 0.001],
                       max_epochs=6, batch_size=16, weight_decay=0.0001, cv_seed=20270913,
                       final_seeds=[20270913, 20270914, 20270915], optimizer='Adam', scheduler=None,
                       augmentation=None,
                       selection='largest mean source-CV accuracy; ties earlier epoch, then smaller learning rate')
            for key in ('input', 'metadata', 'data_manifest', 'weights'):
                name = f'inputs/{dataset}_{key}.fixture'
                self.file(name, ('artificial ' + name).encode())
                cfg[key + '_file'] = name
                cfg[key + '_sha256'] = resources.sha(self.root / name)
            self.put(resources.CNN / f'{dataset}_config.json', cfg)
        sources = {'scientific_producer.py': resources.sha(self.root / 'scientific_producer.py'),
                   str(protocol): resources.sha(self.root / protocol)}
        self.put('FRESH_RUN.json', dict(kind='new_source_fit_tree', original_source_root='/fixture/source',
                                       source_files_sha256=sources))
        self.put('STUDY_SETUP.json', dict(kind='new_full_source_study_tree', source_root='/fixture/source',
                                         files_sha256=sources))
        self.original = dict(limit_seconds=10800, consumed_seconds=100.0, active_session=None,
                             history=[dict(label='select_cluster', status='complete', elapsed_seconds=100.0)],
                             scope='fixture shared budget', last_update_utc='fixture-initial')
        self.put(resources.LEDGER, self.original)

    def tearDown(self):
        self.temp.cleanup()

    def file(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def put(self, name, value):
        return self.file(name, resources.encoded(value))

    def get(self, name):
        return resources.read(self.root / name)

    def allow(self, limit=21600, acknowledge=True):
        return resources.allow(self.root, limit, 'Measured CPU throughput requires more local time.', acknowledge)

    def progress(self, elapsed=11000):
        data = self.get(resources.LEDGER)
        data['history'].append(dict(label='evaluate_external', status='complete', elapsed_seconds=elapsed))
        data['consumed_seconds'] += elapsed
        data['last_update_utc'] = 'fixture-later'
        data['last_resource'] = {'peak_rss_bytes': 1}
        self.put(resources.LEDGER, data)
        return data

    def test_original_limit_check(self):
        result = resources.check(self.root, pre_target=True)
        self.assertEqual(result['effective_limit_seconds'], 10800)
        self.assertFalse(result['original_180_minutes_exceeded'])

    def test_allow_conserves_time_history_and_inputs(self):
        before_bytes = (self.root / resources.LEDGER).read_bytes()
        source_bytes = {p: (self.root / p).read_bytes() for p in
                        ['FRESH_RUN.json', 'STUDY_SETUP.json', str(resources.CNN / 'cluster_config.json'),
                         str(resources.CNN / 'external_config.json')]}
        result = self.allow()
        current = self.get(resources.LEDGER)
        self.assertEqual(result['effective_limit_seconds'], 21600)
        self.assertEqual(current['history'], self.original['history'])
        self.assertEqual(current['consumed_seconds'], 100.0)
        self.assertEqual((self.root / resources.DIRECTORY / 'ledger_before.json').read_bytes(), before_bytes)
        self.assertTrue(all((self.root / p).read_bytes() == b for p, b in source_bytes.items()))

    def test_check_after_completion_reports_original_exceeded(self):
        self.allow(); self.progress()
        self.put('outputs/research_v2/research/external_global_model_freeze.json',
                 dict(budget_limit_seconds=21600, budget_consumed_seconds_before_target=100.0))
        self.file(resources.CNN / 'external/predictions/seed.csv', b'fixture prediction\n')
        result = resources.check(self.root)
        self.assertTrue(result['original_180_minutes_exceeded'])
        self.assertEqual(result['consumed_seconds'], 11100)
        with self.assertRaisesRegex(ValueError, 'Target output'):
            resources.check(self.root, pre_target=True)

    def test_explicit_boundary_acknowledgement_required(self):
        with self.assertRaisesRegex(ValueError, 'Explicit'):
            self.allow(acknowledge=False)
        self.assertEqual(self.get(resources.LEDGER), self.original)

    def test_active_writer_refused(self):
        value = copy.deepcopy(self.original); value['active_session'] = {'label': 'select_cluster'}
        self.put(resources.LEDGER, value)
        for operation in (self.allow, lambda: resources.check(self.root)):
            with self.assertRaisesRegex(ValueError, 'Active session'):
                operation()

    def test_malformed_and_missing_active_session_refused(self):
        for invalid in ({}, [], False, 0, '', 'missing-field'):
            value = copy.deepcopy(self.original)
            if invalid == 'missing-field':
                del value['active_session']
            else:
                value['active_session'] = invalid
            self.put(resources.LEDGER, value)
            for operation in (self.allow, lambda: resources.check(self.root)):
                with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'Active session'):
                    operation()

    def test_unknown_original_limit_refused(self):
        value = copy.deepcopy(self.original); value['limit_seconds'] = 15000
        self.put(resources.LEDGER, value)
        with self.assertRaisesRegex(ValueError, 'Unknown original limit'):
            self.allow()
        with self.assertRaisesRegex(ValueError, 'Missing allowance chain'):
            resources.check(self.root)

    def test_invalid_new_limits_refused(self):
        for value in (None, True, 10800, 10799, 21601, 30000):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'finite allowance'):
                self.allow(limit=value)

    def test_duplicate_allowance_refused(self):
        self.allow()
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.allow()

    def test_target_and_seal_refused_before_allowance(self):
        for name in (resources.CNN / 'cluster/predictions/seed.csv',
                     Path('outputs/research_v2/experiments/calibration/external/method_predictions.csv'),
                     Path('outputs/research/experiments/raw_results/mode0_repeat0_method.csv'),
                     Path('outputs/research_v2/research/external_global_model_freeze.json')):
            path = self.file(name, b'fixture')
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Target output'):
                self.allow()
            path.unlink()

    def test_previous_target_execution_refused(self):
        value = copy.deepcopy(self.original); value['history'][0]['label'] = 'evaluate_cluster'
        self.put(resources.LEDGER, value)
        with self.assertRaisesRegex(ValueError, 'Target execution'):
            self.allow()

    def test_nonfinite_and_unaccounted_time_refused(self):
        for value in (float('nan'), -1, 99, 101):
            ledger = copy.deepcopy(self.original); ledger['consumed_seconds'] = value
            self.file(resources.LEDGER, json.dumps(ledger).encode())
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.allow()

    def test_original_exhaustion_is_not_recovered(self):
        value = copy.deepcopy(self.original)
        value['consumed_seconds'] = value['history'][0]['elapsed_seconds'] = 10801
        self.put(resources.LEDGER, value)
        with self.assertRaisesRegex(ValueError, 'already exceeded'):
            self.allow()

    def test_missing_chain_and_contradictory_limit_refused(self):
        self.allow()
        path = self.root / resources.DIRECTORY / 'application.json'
        saved = path.read_bytes(); path.unlink()
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            resources.check(self.root)
        path.write_bytes(saved)
        ledger = self.get(resources.LEDGER); ledger['limit_seconds'] = 21599
        self.put(resources.LEDGER, ledger)
        with self.assertRaisesRegex(ValueError, 'contradictory'):
            resources.check(self.root)

    def test_tampered_receipt_refused(self):
        self.allow()
        name = resources.DIRECTORY / 'allowance.json'; value = self.get(name)
        value['reason'] = 'Altered reason after application'; self.put(name, value)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            resources.check(self.root)

    def test_original_amendment_tampering_refused(self):
        self.allow()
        name = resources.CNN / 'compute_budget_amendment_02.json'; value = self.get(name)
        value['new_limit_seconds'] = 21600; self.put(name, value)
        with self.assertRaisesRegex(ValueError, 'original amendment limits'):
            resources.check(self.root)

    def test_scientific_settings_or_actual_input_tampering_refused(self):
        name = resources.CNN / 'external_config.json'; original = self.get(name)
        altered = copy.deepcopy(original); altered['max_epochs'] = 5; self.put(name, altered)
        with self.assertRaisesRegex(ValueError, 'scientific settings'):
            self.allow()
        self.put(name, original); self.allow()
        self.file(original['input_file'], b'changed fixture inputs')
        with self.assertRaisesRegex(ValueError, 'Input hash mismatch'):
            resources.check(self.root)

    def test_history_reset_refused_even_when_sum_still_matches(self):
        self.allow()
        value = self.get(resources.LEDGER)
        value.update(history=[], consumed_seconds=0)
        self.put(resources.LEDGER, value)
        with self.assertRaisesRegex(ValueError, 'history was reset'):
            resources.check(self.root)

    def test_unexpected_receipt_and_ledger_fields_refused(self):
        self.allow()
        extra = self.file(resources.DIRECTORY / 'second_allowance.json', b'{}')
        with self.assertRaisesRegex(ValueError, 'unexpected allowance chain'):
            resources.check(self.root)
        extra.unlink()
        value = self.get(resources.LEDGER); value['scope'] = 'changed scope'; self.put(resources.LEDGER, value)
        with self.assertRaisesRegex(ValueError, 'Unexpected ledger fields'):
            resources.check(self.root)

    def test_mutation_detected_before_ledger_replace(self):
        original_bindings = resources.input_bindings
        def mutate(root):
            result = original_bindings(root)
            value = self.get(resources.LEDGER); value['last_update_utc'] = 'competing writer'
            self.put(resources.LEDGER, value)
            return result
        with patch.object(resources, 'input_bindings', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'Ledger changed during preparation'):
                self.allow()
        self.assertEqual(self.get(resources.LEDGER)['limit_seconds'], 10800)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            resources.check(self.root)

    def test_later_source_context_is_supported(self):
        self.allow()
        self.put(resources.CNN / 'external/selection_context.json',
                 dict(files_sha256={'scientific_producer.py': resources.sha(self.root / 'scientific_producer.py')}))
        self.assertEqual(resources.check(self.root)['status'], 'resource_chain_valid')

    def test_global_seal_limit_mismatch_refused(self):
        self.allow(); self.progress()
        self.put('outputs/research_v2/research/external_global_model_freeze.json',
                 dict(budget_limit_seconds=10800, budget_consumed_seconds_before_target=100))
        with self.assertRaisesRegex(ValueError, 'Global seal resource fields'):
            resources.check(self.root)

    def test_cli_default_check_does_not_mutate(self):
        before = (self.root / resources.LEDGER).read_bytes()
        process = subprocess.run([sys.executable, str(Path(resources.__file__).resolve()), 'check',
                                  '--project-root', str(self.root), '--pre-target'], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)['status'], 'resource_chain_valid')
        self.assertEqual((self.root / resources.LEDGER).read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
