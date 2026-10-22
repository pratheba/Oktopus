"""Offline consistency checks; no neural model or CUDA required."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT/'exp/train/manipulation'
spec = importlib.util.spec_from_file_location('runner', ROOT/'tools/oct22_run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

class TestOct22Runner(unittest.TestCase):
    def test_puffer_is_full_fused_0087(self):
        p, d = runner.load_yaml('oct22_adapt_oktopus_puffer_full_sdf_optimization.yaml')
        self.assertTrue(d['sdf_fused_exterior_0087'])
        self.assertEqual(runner.adaptation_modes(d), ('mixed', {'rigid':14, 'wrap':1}))
        self.assertTrue(runner.root_leaf_is_full_puffer(d))
        self.assertTrue(all(a['cut_avatar'] is False for a in d['adaptations']))

    def test_udf_matches_puffer_identifiers(self):
        a= yaml.safe_load((MAN/'oct22_adapt_oktopus_puffer_full_sdf_optimization.yaml').read_text())
        b= yaml.safe_load((MAN/'oct22_adapt_oktopus_puffer_full_udf_optimization.yaml').read_text())
        self.assertEqual([{k:v for k,v in x.items() if not k.startswith('sdf_')} for x in a['adaptations']], [{k:v for k,v in x.items() if not k.startswith('sdf_')} for x in b['adaptations']])
        self.assertEqual(a['root_leaves'],b['root_leaves'])
        self.assertEqual(a['blend_groups'],b['blend_groups'])
        self.assertTrue(b['udf_full_pipeline_0084']['enabled'])
        self.assertTrue(b['udf_cut_curve_local_preview_0081'])

    def test_boots_rigid_only(self):
        _,a=runner.load_yaml('oct22_adapt_oktopus_boots_t7_boot1_only_sdf_optimization.yaml')
        self.assertEqual(runner.adaptation_modes(a)[0],'rigid')
        self.assertFalse(a.get('sdf_fused_exterior_0087', False))

    def test_cone_two_wrap_combined_but_not_0087(self):
        _,a=runner.load_yaml('oct22_adapt_oktopus_cone_post_sdf_sdf_optimization.yaml')
        self.assertEqual(runner.adaptation_modes(a)[0],'wrap')
        self.assertTrue(a['combine_adaptations'])
        self.assertFalse(a.get('sdf_fused_exterior_0087', False))

    def test_ambiguous_radius_mode_is_not_guessable(self):
        self.assertRaisesRegex(ValueError,'ambiguous radius mode',runner.adaptation_modes,{
            'adaptations':[{'target_key':'x','wrap_radius':False,'rigid_radius':False}]})

    def test_cli_dry_run_puffer(self):
        res=subprocess.run([sys.executable,str(ROOT/'tools/oct22_run.py'),'sdf',
                            '--case','oktopus_puffer_full','--require-fused','--dry-run'],
                           capture_output=True,text=True,check=True)
        self.assertIn('true_fused_exterior=True',res.stdout)
        self.assertIn('test_dc.py',res.stdout)
        self.assertIn('oct22_adapt_oktopus_puffer_full_sdf_optimization.yaml',res.stdout)
        self.assertNotIn('test_udf_sdf_cut_stitch.py',res.stdout)

    def test_cli_rejects_nonfused_if_required(self):
        res=subprocess.run([sys.executable,str(ROOT/'tools/oct22_run.py'),'sdf',
                            '--case','oktopus_boots_t1','--require-fused','--dry-run'],
                           capture_output=True,text=True)
        self.assertNotEqual(res.returncode,0)
        self.assertIn('no sdf_fused_exterior_0087',res.stderr)

    def test_udf_dry_run_names_not_fused(self):
        res=subprocess.run([sys.executable,str(ROOT/'tools/oct22_run.py'),'udf-parts',
                            '--case','oktopus_puffer_full','--dry-run'],
                           capture_output=True,text=True,check=True)
        self.assertIn('EIGHT PARTS',res.stdout)
        self.assertIn('test_udf_sdf_cut_stitch.py',res.stdout)

if __name__=='__main__': unittest.main()
