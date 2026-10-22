"""Regression tests for 0100 part selection and explicit tentacle opt-in."""
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'tools' / 'udf_0100_run_direct.py'
spec = importlib.util.spec_from_file_location('udf_0100_runner', RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class PartSelectionTests(unittest.TestCase):
    def test_default_all_sentinel(self):
        self.assertEqual(runner.parse_parts('all'), 'all')
        self.assertEqual(runner.parse_parts(' ALL '), 'all')

    def test_named_subset(self):
        self.assertEqual(runner.parse_parts(' body, tentacle_t1 '), ['body', 'tentacle_t1'])
        self.assertEqual(runner.parse_parts('body'), ['body'])

    def test_invalid_selections(self):
        for argument in ['', 'body,', 'body,,tentacle_t1', 'all,body', 'body,body']:
            with self.subTest(argument=argument), self.assertRaises(ValueError):
                runner.parse_parts(argument)

    def test_nonbody_opt_in_needed(self):
        self.assertTrue(runner.needs_nonbody_opt_in('all'))
        self.assertTrue(runner.needs_nonbody_opt_in(['body', 'tentacle_t3']))
        self.assertFalse(runner.needs_nonbody_opt_in(['body']))

    def test_cli_no_implicit_body_only(self):
        base = [sys.executable, str(RUNNER),
                '--sdf-run', 'dummy_sdf', '--udf-run', 'dummy_udf',
                '--out', 'dummy_out', '--dry-run']
        missing_optin = subprocess.run(base, text=True, capture_output=True)
        self.assertNotEqual(missing_optin.returncode, 0)
        self.assertIn('--allow-uncalibrated-nonbody', missing_optin.stderr)
        all_parts = subprocess.run(base + ['--allow-uncalibrated-nonbody'],
                                   text=True, capture_output=True)
        self.assertEqual(all_parts.returncode, 0, all_parts.stderr)
        self.assertIn('query parts: all', all_parts.stdout)
        body = subprocess.run(base + ['--parts', 'body'], text=True, capture_output=True)
        self.assertEqual(body.returncode, 0, body.stderr)
        self.assertIn("query parts: ['body']", body.stdout)


if __name__ == '__main__':
    unittest.main()
