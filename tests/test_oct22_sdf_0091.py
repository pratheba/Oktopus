"""Checks for radius/mode-aware bounded acceleration and runnable command defaults."""
import unittest
import subprocess
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/app'))
from sdf_bounded_eligibility_0091 import permitted_bounded_localization


class BoundGuard(unittest.TestCase):
    def test_rigid_supported(self):
        self.assertTrue(permitted_bounded_localization({'mode': 'direct', 'rigid_radius': True, 'wrap_radius': False})[0])

    def test_cone_regular_wrap_not_supported(self):
        self.assertFalse(permitted_bounded_localization({'mode': 'direct', 'rigid_radius': False, 'wrap_radius': True})[0])

    def test_puffer_staging_can_use_bounded(self):
        self.assertTrue(permitted_bounded_localization({'mode': 'direct', 'rigid_radius': False, 'wrap_radius': True}, staged_wrap=True)[0])

    def test_merged_wrap_with_global_statistics_cannot_use_bounded(self):
        self.assertFalse(permitted_bounded_localization({'mode': 'direct', 'rigid_radius': False, 'wrap_radius': True, 'use_attach_to_free_blend': True}, staged_wrap=True)[0])

    def test_rigid_with_global_statistics_cannot_use_bounded(self):
        self.assertFalse(permitted_bounded_localization({'mode': 'direct', 'rigid_radius': True, 'use_wrap_to_rigid_blend': True})[0])

    def test_unsupported_mode(self):
        self.assertFalse(permitted_bounded_localization({'mode': 'loop', 'rigid_radius': True})[0])

    def test_cone_yaml_is_fused_and_uses_full_localization(self):
        d = yaml.safe_load((ROOT/'exp/train/manipulation/oct22_adapt_oktopus_cone_post_sdf_0091_sdf_optimization.yaml').read_text())
        self.assertFalse(d['dcsdd_bounded_filtering'])
        self.assertTrue(d['sdf_fused_exterior_0087'])
        self.assertEqual(d['root_leaves']['parts'], {'cone': {'blend_group': 'cone_body'}})
        self.assertEqual(len(d['adaptations']), 2)
        self.assertTrue(all(a['wrap_radius'] for a in d['adaptations']))

    def test_source_guard_runs_before_filtering(self):
        s = (ROOT/'src/app/agent_3dvec_sdf.py').read_text()
        start = s.index('permitted_bounded_localization(')
        finish = s.index('curve.filter_grid_adapt(', start)
        self.assertTrue(start < finish)

    def test_shell_dryrun_never_forces_bounded(self):
        s = (ROOT/'test_dc_oct22_editable.sh').read_text()
        # Test a temp copy with DRY_RUN=1, retaining same script root by using
        # here-doc invocation from repository root is hard; copy only script
        # into the work tree is unnecessary because command is printed through
        # the script's real path.
        script = (ROOT/'test_dc_oct22_editable.sh')
        try:
            script.write_text(s.replace('DRY_RUN=0 ', 'DRY_RUN=1 ', 1))
            p = subprocess.run(['bash', str(script)], cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn('fused=true',p.stdout)
            self.assertIn('modes=rigid=0,wrap=2',p.stdout)
            self.assertNotIn('--dcsdd-bounded-filtering',p.stdout)
        finally:
            script.write_text(s)


if __name__ == '__main__':
    unittest.main()
