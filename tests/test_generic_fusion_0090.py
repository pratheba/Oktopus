import importlib.util
from pathlib import Path
import unittest
import yaml
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/app'))
from root_leaf_combination import build_root_leaf_plan
spec=importlib.util.spec_from_file_location('oct22',ROOT/'tools/oct22_run.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

class TestGenericSDF0090(unittest.TestCase):
    def example(self,name):
        return yaml.safe_load((ROOT/'exp/train/manipulation'/name).read_text())
    def test_puffer_legacy_unchanged(self):
        f=Path('/mnt/data/oktopus_0089_patchbase/exp/train/manipulation/oct22_adapt_oktopus_puffer_full_sdf_optimization.yaml')
        if not f.exists(): self.skipTest('baseline fixture absent')
        d=yaml.safe_load(f.read_text())
        plan=build_root_leaf_plan(d['adaptations'],d['root_leaves'])
        self.assertEqual(len(plan.part_names),8)
        self.assertEqual(plan.part_names[0],'body')
    def test_single_boot(self):
        d=self.example('oct22_example_oktopus_boots_t7_fused_sdf.yaml')
        p=build_root_leaf_plan(d['adaptations'],d['root_leaves'])
        self.assertEqual(p.part_names,('boot',))
        self.assertEqual(p.item_to_part,{0:'boot'})
    def test_armadillo_selected_mixed_boot(self):
        d=self.example('oct22_example_armadillo_boots2_selected_fused_sdf.yaml')
        self.assertEqual(runner.adaptation_modes(d)[0], 'mixed')
        p=build_root_leaf_plan(d['adaptations'],d['root_leaves'])
        self.assertEqual(p.part_names, ('boot',))
        self.assertEqual(p.group_to_part, {'boot':'boot'})
        self.assertEqual(len(d['adaptations']), 2)
    def test_cone_group(self):
        d=self.example('oct22_example_oktopus_cone_fused_sdf.yaml')
        p=build_root_leaf_plan(d['adaptations'],d['root_leaves'])
        self.assertEqual(p.part_names,('cone',))
        self.assertEqual(p.group_to_part,{'cone_body':'cone'})
    def test_trex4(self):
        d=self.example('oct22_example_trex2_puffer_fused_sdf.yaml')
        p=build_root_leaf_plan(d['adaptations'],d['root_leaves'])
        self.assertEqual(p.part_names,('body','neck','front_r_leg','front_l_leg'))
        self.assertEqual(p.attachments,{'neck':'body','front_r_leg':'body','front_l_leg':'body'})
    def test_trex_dress_selected_separate_parts_global_boolean(self):
        d=self.example('oct22_example_trex_dress_selected_fused_sdf.yaml')
        p=build_root_leaf_plan(d['adaptations'],d['root_leaves'])
        self.assertEqual(p.part_names,('dress_top','dress_bottom'))
        self.assertEqual(p.item_to_part,{0:'dress_top',1:'dress_bottom'})
        self.assertTrue(d['sdf_fused_exterior_0087'])
    def test_mixed_group_with_explicit_assignment(self):
        a=[{'target_key':'x|body','wrap_radius':True,'name':'body'},
           {'target_key':'x|arm','rigid_radius':True,'blend_group':'arm'},
           {'target_key':'x|arm','wrap_radius':True,'blend_group':'arm'}]
        s={'parts':{'body':{'adaptation_name':'body'}, 'arm':{'blend_group':'arm','attach_to':'body'}}}
        p=build_root_leaf_plan(a,s)
        self.assertEqual(p.item_to_part,{0:'body',1:'arm',2:'arm'})
        self.assertEqual(p.group_to_part,{'arm':'arm'})
    def test_missing_assignment(self):
        with self.assertRaisesRegex(ValueError,'unassigned'):
            build_root_leaf_plan([{'target_key':'x|a'}, {'target_key':'x|b'}],{'parts':{'first':{'adaptation_indices':[0]}}})
    def test_wrong_unmarked_multiple(self):
        with self.assertRaisesRegex(ValueError,'shared blend_group'):
            build_root_leaf_plan([{'target_key':'x|a'}, {'target_key':'x|a'}],{'parts':{'a':{'target_key':'x|a'}}})
    def test_bad_cycle(self):
        with self.assertRaisesRegex(ValueError,'Cycle'):
            build_root_leaf_plan([{'target_key':'x|a'}, {'target_key':'x|b'}],{
                'parts':{'a':{'target_key':'x|a','attach_to':'b'},'b':{'target_key':'x|b','attach_to':'a'}}})
    def test_sdf_non_sparse_global_accumulate_enabled(self):
        s=(ROOT/'src/app/agent_3dvec_sdf.py').read_text()
        self.assertIn('fused_exterior_0087 or root_leaf_plan is None',s)
        self.assertIn("'source_parts': list(root_leaf_plan.part_names) if root_leaf_plan is not None else ['combined']",s)
    def test_runner_accepts_generic_rootplan(self):
        # Test runner logic using mocked loaded yaml and a model config available in packaged source.
        import argparse
        d=self.example('oct22_example_trex2_puffer_fused_sdf.yaml')
        args=argparse.Namespace(out='trex_dry',shape='trex_9_v2',resolution=128,
             yaml='oct22_example_trex2_puffer_fused_sdf.yaml',case='trex2_puffer',
             config='config/YOUR_TREX_MODEL_CONFIG.yaml',dry_run=True,
             require_fused=True,stream=False,dcsdd_repo='third_party/dcsdd',
             memmap_dir='logs', model_size='4M', checkpoint='final',outer_iters=5,inner_iters=3)
        with patch.object(runner,'ROOT',ROOT),patch.object(runner,'YAML_DIR',ROOT/'exp/train/manipulation'):
            runner.sdf(args)

if __name__=='__main__': unittest.main()
