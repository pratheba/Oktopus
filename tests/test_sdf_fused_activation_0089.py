"""Regression tests: 0087 YAML flag must reach the SDF extractor."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('oct22_run_0089', ROOT / 'tools/oct22_run.py')
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


class TestFusionActivation(unittest.TestCase):
    def test_current_sdf_source_forwards_fused_flag(self):
        run.verify_fused_yaml_forwarding(ROOT / 'src/app/agent_3dvec_sdf.py')

    def test_original_0087_whitelist_regression_rejected(self):
        source = (ROOT / 'src/app/agent_3dvec_sdf.py').read_text()
        key = "                        'sdf_fused_exterior_0087',\n"
        self.assertEqual(source.count(key), 1)
        with tempfile.TemporaryDirectory() as folder:
            broken_path = Path(folder) / 'agent_3dvec_sdf.py'
            broken_path.write_text(source.replace(key, '', 1))
            with self.assertRaisesRegex(ValueError, 'NOT forwarded'):
                run.verify_fused_yaml_forwarding(broken_path)

    def test_unrecognized_source_structure_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            broken_path = Path(folder) / 'agent_3dvec_sdf.py'
            broken_path.write_text('def action_part_adapt(): pass\n')
            with self.assertRaisesRegex(ValueError, 'Cannot verify'):
                run.verify_fused_yaml_forwarding(broken_path)


if __name__ == '__main__':
    unittest.main()
