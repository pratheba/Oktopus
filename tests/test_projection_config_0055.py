"""Guard 0055 config forwarding and final-part-only extraction mode.

PYTHONPATH=src:src/ngc python -m unittest discover -s tests -p 'test_projection_config_0055.py' -v
"""
from pathlib import Path
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src' / 'app' / 'agent_3dvec_sdf.py'
YAML_ROOT = ROOT / 'exp' / 'train' / 'manipulation'

class Config0055Tests(unittest.TestCase):
    def test_flag_loaded_then_forwarded_into_puffer_adaptation(self):
        code = SRC.read_text()
        key = "'sdf_projection_refine_fast_0055'"
        profile = "'sdf_projection_profile_split_0055'"
        self.assertGreaterEqual(code.count(key), 3)
        self.assertGreaterEqual(code.count(profile), 3)
        self.assertIn("[sdf_projection_0055]", code)
        self.assertIn('projection_profile_start_0055 = (', code)

    def test_body_and_t1_configs_keep_audit(self):
        for stem, stop in [('body_audit', 1), ('t1_audit', 4)]:
            cfg = yaml.safe_load((YAML_ROOT / (
                'adapt_oktopus_puffer_root_leaves_0055_' + stem + '.yaml')).read_text())
            self.assertEqual(cfg['sdf_puffer_stop_after'], stop)
            self.assertTrue(cfg['sdf_candidate_interval_audit'])
            self.assertTrue(cfg['sdf_projection_refine_fast_0055'])
            self.assertTrue(cfg['sdf_projection_profile_split_0055'])
            self.assertEqual(len(cfg['adaptations']), 15)

    def test_full_config_retains_ownership_but_skips_redundant_meshes(self):
        cfg = yaml.safe_load((YAML_ROOT / (
            'adapt_oktopus_puffer_root_leaves_0055_full_audit.yaml')).read_text())
        self.assertNotIn('sdf_puffer_stop_after', cfg)
        self.assertFalse(cfg['export_individual_meshes'])
        self.assertTrue(cfg['sdf_candidate_interval_audit'])
        self.assertEqual(len(cfg['adaptations']), 15)
        self.assertEqual(cfg['root_leaves']['root'], 'puffer_body_wrap_no_snug')
        self.assertEqual(len(cfg['root_leaves']['leaves']), 7)
        self.assertEqual(sorted(cfg['blend_groups']), ['t1','t2','t4','t5'])
        # Current pipeline still extracts singleton root/leaf parts with
        # rig_single_part and every completed group even when the redundant
        # per-section diagnostic meshes are disabled.
        code = SRC.read_text()
        self.assertIn('if export_individual_meshes or rig_single_part:', code)
        self.assertIn('if root_leaf_plan is not None:', code)
        self.assertIn("context=f'root_leaf_{part_name}'", code)

if __name__ == '__main__':
    unittest.main()
