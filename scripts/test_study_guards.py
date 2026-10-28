"""Small deterministic safety checks; no data acquisition, model fit or outcome."""
import json
from pathlib import Path
import tempfile
import unittest
from prepare_study import destination_path, safe_path, sha, verify_study, write_new
from accept_source_audits import forbid_target_outputs, validate_checks, bound_path


class StudyGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_reference_target_predictions_are_relocated(self):
        name = 'outputs/research_v2/experiments/calibration/external/external40_corrected_predictions.csv'
        self.assertEqual(destination_path(name), Path('reference_release') / name)

    def test_source_files_keep_original_layout(self):
        name = 'outputs/research_v2/research/seal_external_models.py'
        self.assertEqual(destination_path(name), Path(name))

    def test_metadata_keeps_dependency_path(self):
        name = 'outputs/research/corpus/texture_list.xlsx'
        self.assertEqual(destination_path(name), Path(name))

    def test_manifest_escape_is_rejected(self):
        with self.assertRaises(ValueError):
            safe_path(self.root, '../outside.py')

    def test_absolute_audit_input_escape_is_rejected(self):
        with self.assertRaises(ValueError):
            bound_path(self.root, self.root.parent / 'outside.json')

    def test_existing_record_is_never_overwritten(self):
        path = self.root / 'receipt.json'
        write_new(path, {'fixture': 1})
        with self.assertRaises(FileExistsError):
            write_new(path, {'fixture': 2})
        self.assertEqual(json.loads(path.read_text()), {'fixture': 1})

    def test_source_file_drift_is_rejected(self):
        script = self.root / 'program.py'
        script.write_text('pass\n')
        write_new(self.root / 'STUDY_SETUP.json', {
            'kind': 'new_full_source_study_tree', 'source_root': str(self.root.parent),
            'files_sha256': {'program.py': sha(script)}})
        verify_study(self.root)
        script.write_text('raise RuntimeError\n')
        with self.assertRaises(ValueError):
            verify_study(self.root)

    def test_external_target_output_closes_gate(self):
        forbid_target_outputs(self.root)
        path = self.root / 'outputs/research_v2/experiments/calibration/external/a_predictions.csv'
        path.parent.mkdir(parents=True)
        path.write_text('fixture-only\n')
        with self.assertRaises(ValueError):
            forbid_target_outputs(self.root)

    def test_prior_target_history_closes_gate(self):
        write_new(self.root / 'outputs/research_v2/experiments/cnn/compute_budget.json',
                  {'history': [{'label': 'evaluate_external'}], 'active_session': None})
        with self.assertRaises(ValueError):
            forbid_target_outputs(self.root)

    def test_empty_audit_cannot_pass(self):
        script = self.root / 'audit.py'
        script.write_text('pass\n')
        with self.assertRaises(ValueError):
            validate_checks({'script_sha256': sha(script), 'failed_checks': [], 'checks': [],
                             'check_count': 0}, script)

    def test_failed_check_cannot_pass(self):
        script = self.root / 'audit.py'
        script.write_text('pass\n')
        with self.assertRaises(ValueError):
            validate_checks({'script_sha256': sha(script), 'failed_checks': [],
                             'checks': [{'passed': False}], 'check_count': 1}, script)

    def test_report_for_changed_script_is_rejected(self):
        script = self.root / 'audit.py'
        script.write_text('pass\n')
        with self.assertRaises(ValueError):
            validate_checks({'script_sha256': '0' * 64, 'failed_checks': [],
                             'checks': [{'passed': True}], 'check_count': 1}, script)


if __name__ == '__main__':
    unittest.main()
