"""0056 settings are loaded and forwarded to the actual puffer call site."""
from pathlib import Path
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / 'src/app/agent_3dvec_sdf.py'
LOCALIZE = ROOT / 'src/ngc/pwla_curve/_localize.py'
GEOMETRY = ROOT / 'src/ngc/pwla_curve/_geometry.py'
WRITER = ROOT / 'src/app/sdf_puffer_prediction_stream_0053.py'
YAMLS = ROOT / 'exp/train/manipulation'


class Wiring0056Tests(unittest.TestCase):
    def test_adaptation_and_projection_wiring(self):
        a = AGENT.read_text()
        l = LOCALIZE.read_text()
        g = GEOMETRY.read_text()
        w = WRITER.read_text()
        self.assertGreaterEqual(a.count("'sdf_projection_leafsize_0056'"), 3)
        self.assertIn('projection_leafsize_0056=int(adapt_arg.get(', l)
        self.assertIn('leafsize=leafsize_0056', g)
        self.assertIn('leafsize_0056', g[g.index('cache_key = ('):])
        self.assertIn('sdf_prediction_scratch_dir_0056', w)
        self.assertIn("write(memoryview(a).cast('B'))", w)
        self.assertIn("'sdf_prediction_scratch_dir_0056'", a)

    def test_configs_preserve_body_audit_and_15_adaptations(self):
        for stem, audit in (('body_audit', True),('body_profile', False)):
            cfg = yaml.safe_load((YAMLS / (
                f'adapt_oktopus_puffer_root_leaves_0056_{stem}.yaml')).read_text())
            self.assertEqual(cfg['sdf_puffer_stop_after'],1)
            self.assertEqual(cfg['sdf_candidate_interval_audit'],audit)
            self.assertEqual(cfg['sdf_projection_leafsize_0056'],32)
            self.assertTrue(cfg['sdf_prediction_zero_copy_write_0056'])
            self.assertEqual(len(cfg['adaptations']),15)
            self.assertEqual(cfg['root_leaves']['root'],'puffer_body_wrap_no_snug')
            self.assertTrue(cfg['sdf_projection_refine_fast_0055'])
            self.assertTrue(cfg['sdf_projection_cache_0054'])


if __name__ == '__main__':
    unittest.main()
