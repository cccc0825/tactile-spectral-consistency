#!/usr/bin/env python3
"""Guard tests using artificial temporary records, never scientific observations."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import analyze_fresh_results as analysis


def rows_for(correct):
    rows = []
    for specimen in range(12):
        n = 100 if specimen < 8 else 99
        for trial in range(1, n+1):
            rows.append(dict(index=len(rows)+2354, acquisition_id=f'{specimen}_40_{trial}',
                texture=specimen, speed=40, domain='speed',
                prediction=specimen if correct else (specimen+1)%12, correct=int(correct)))
    return rows


class FreshAnalysisGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.a, self.b = rows_for(True), rows_for(False)
        self.target = {r['index']: {'row': str(r['index']), 'class_id': str(r['texture']),
                       'speed': '40', 'trial': r['acquisition_id'].split('_')[-1]}
                       for r in self.a}

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, rows):
        p = self.root / name
        with p.open('w', newline='') as stream:
            w = csv.DictWriter(stream, fieldnames=list(self.a[0])); w.writeheader(); w.writerows(rows)
        return p

    def pair(self, a=None, b=None):
        return analysis.paired_rows(self.write('a.csv', self.a if a is None else a),
                                    self.write('b.csv', self.b if b is None else b), self.target)

    def test_all_positive_results_are_retained(self):
        a, b = self.pair(); rows, summary = analysis.summarize(a, b)
        self.assertEqual(summary['specimen_signs'], {'improved': 12, 'worsened': 0, 'tied': 0})
        self.assertEqual(summary['all12_difference_pp'], 100)
        self.assertEqual(summary['conditional_interval_pp'], [100, 100])
        self.assertTrue(summary['primary_support']); self.assertEqual(len(rows), 12)

    def test_all_negative_results_are_retained(self):
        a, b = self.pair(self.b, self.a); _, summary = analysis.summarize(a, b)
        self.assertEqual(summary['specimen_signs']['worsened'], 12)
        self.assertEqual(summary['conditional_interval_pp'], [-100, -100])
        self.assertFalse(summary['primary_support'])

    def test_equal_results(self):
        a, b = self.pair(self.a, self.a); _, summary = analysis.summarize(a, b)
        self.assertEqual(summary['specimen_signs']['tied'], 12)
        self.assertEqual(summary['conditional_interval_pp'], [0, 0])
        self.assertFalse(summary['primary_support'])

    def test_duplicate_index_rejected(self):
        self.a[0] = dict(self.a[1])
        with self.assertRaisesRegex(ValueError, 'Duplicate or missing'): self.pair()

    def test_missing_episode_rejected(self):
        with self.assertRaisesRegex(ValueError, 'row count'): self.pair(self.a[:-1])

    def test_false_correct_field_rejected(self):
        self.a[0]['correct'] = 0
        with self.assertRaisesRegex(ValueError, 'correct field'): self.pair()

    def test_wrong_eligible_identity_rejected(self):
        self.a[0]['acquisition_id'] = 'different'
        with self.assertRaisesRegex(ValueError, 'identity'): self.pair()

    def test_source_row_rejected(self):
        self.a[0]['index'] = 1
        with self.assertRaisesRegex(ValueError, 'non-target'): self.pair()

    def test_wrong_truth_rejected(self):
        self.a[0]['texture'] = 11
        with self.assertRaisesRegex(ValueError, 'truth/domain'): self.pair()

    def test_reference_file_rejected(self):
        p = self.root / 'reference_release/pred.csv'; p.parent.mkdir(); p.write_text('test only')
        with self.assertRaisesRegex(ValueError, 'reference output'):
            analysis.checked_path(self.root, p)

    def test_output_overwrite_rejected(self):
        with self.assertRaisesRegex(ValueError, 'new directory'):
            analysis.export(self.root, self.root/'run', self.root)

    def test_historical_external_synthetic_run_rejected(self):
        with self.assertRaisesRegex(ValueError, 'inside the new study'):
            analysis.export(self.root, self.root.parent/'historical', self.root/'out')

    def test_axis_does_not_clip_extremes(self):
        lower, upper = analysis.plot_limits([-100, 100, -99, 99])
        self.assertLess(lower, -100); self.assertGreater(upper, 100)

    def test_missing_global_seal_rejected(self):
        (self.root/'STUDY_SETUP.json').write_text('{}')
        with patch.object(analysis, 'verify_study', return_value={'created_utc': '2026-01-01T00:00:00+00:00'}):
            with self.assertRaisesRegex(ValueError, 'Artifact missing'):
                analysis.verify_external(self.root)

    def test_caption_all_three_states(self):
        path = Path(__file__).resolve().parents[1] / 'outputs/research_v2/analysis/make_supplement.py'
        spec = importlib.util.spec_from_file_location('supplement_under_test', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.assertIn('was met', module.primary_caption(True))
        self.assertIn('was not met', module.primary_caption(False))
        self.assertIn('unavailable', module.primary_caption(None))
        self.assertNotIn('did not open', module.primary_caption(True))


if __name__ == '__main__':
    unittest.main()
