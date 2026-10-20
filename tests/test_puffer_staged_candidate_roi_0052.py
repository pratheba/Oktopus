"""0052 staged puffer candidate ROI: exact parity and excluded-voxel audit.

No inference shortcuts: puffer's global rho-quantile support clamp remains
unchanged in the downstream 0051a pipeline.
"""
import contextlib
import io
import unittest
from unittest import mock
import numpy as np

from test_sdf_candidate_interval_0048 import StraightHandle
from utils.mc_utils import MCGrid


KEYS = ('samples_local','coords','angles','radius','rho','rho_n')


def model_rows(pts, keep):
    selected = pts[keep]
    n = len(keep)
    return {
        'samples_local': selected.copy(),
        'coords': np.ascontiguousarray(selected[:,0]),
        'angles': np.ascontiguousarray(selected[:,1]),
        'radius': np.ones((n,2),np.float64),
        'rho': np.linalg.norm(selected[:,1:],axis=1),
        'rho_n': np.linalg.norm(selected[:,1:],axis=1),
    }


def synthetic_localize(pts, arg):
    keep=np.flatnonzero((pts[:,0]>-.26)&(pts[:,0]<-.18)&
                        (pts[:,1]**2+pts[:,2]**2 < .07**2))
    return model_rows(pts,keep),model_rows(pts,keep),keep


class PufferROITests(unittest.TestCase):
    def setUp(self):
        self.grid = MCGrid({'reso':24, 'size':1.2, 'level':0.0})
        self.curve = StraightHandle()
        self.curve.core.localize_samples_adapt=synthetic_localize
        self.args={
            'mode':'direct', 'src_0':.3, 'src_1':.4,
            'use_adapt_control_filter_grid': True,
            'rigid_radius': True, 'wrap_radius': False,
            'dcsdd_bounded_filtering': True,
            'dcsdd_filter_batch_size': 19,
            '_sdf_puffer_minimal_staging': True,
            'adapt_debug_counts':False,
            'sdf_candidate_interval_prune':False,
            'sdf_candidate_interval_audit':False,
        }

    def run_stage(self, arg):
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            acc,av,idx,inside=self.curve.filter_grid_adapt(self.grid,arg)
        self.assertIsNone(inside)
        self.assertEqual(set(acc),set(KEYS))
        self.assertEqual(set(av), set(KEYS) if arg.get('cut_avatar', False) else {'coords'})
        return ({k:np.asarray(v).copy() for k,v in acc.items()},
                {k:np.asarray(v).copy() for k,v in av.items()},
                np.asarray(idx).copy(),captured.getvalue())

    def test_rigid_staged_candidate_parity_with_audit(self):
        full=self.run_stage(self.args)
        clipped=self.run_stage({**self.args,'sdf_candidate_interval_prune':True,
                                'sdf_candidate_interval_audit':True})
        self.assertIn('missing_accepted= 0',clipped[3])
        self.assertIn('candidate_roi=staged_puffer',clipped[3])
        for d1,d2 in zip(full[:2],clipped[:2]):
            for key in d1:np.testing.assert_array_equal(d1[key],d2[key],err_msg=key)
        np.testing.assert_array_equal(full[2],clipped[2])
        self.assertGreater(len(full[2]),0)

    def test_wrap_staged_candidate_parity_with_audit(self):
        cfg={**self.args,'rigid_radius':False,'wrap_radius':True,
             '_sdf_puffer_bounded_wrap_staging':True,
             'cut_avatar':True}
        full=self.run_stage(cfg)
        clipped=self.run_stage({**cfg,'sdf_candidate_interval_prune':True,
                                'sdf_candidate_interval_audit':True})
        self.assertIn('missing_accepted= 0',clipped[3])
        for d1,d2 in zip(full[:2],clipped[:2]):
            for key in d1:np.testing.assert_array_equal(d1[key],d2[key],err_msg=key)
        np.testing.assert_array_equal(full[2],clipped[2])

    def test_audit_rejects_excluded_accepted_samples(self):
        def all_localize(pts,arg):
            keep=np.arange(len(pts))
            return model_rows(pts,keep),model_rows(pts,keep),keep
        self.curve.core.localize_samples_adapt=all_localize
        with self.assertRaisesRegex(ValueError,'UNSAFE'):
            self.run_stage({**self.args,'sdf_candidate_interval_prune':True,
                            'sdf_candidate_interval_audit':True})

    def test_rejects_unauthorized_puffer_staging(self):
        cfg={**self.args,'_sdf_puffer_minimal_staging':False,
             'sdf_candidate_interval_prune':True}
        with self.assertRaisesRegex(ValueError,'bounded direct streaming'):
            self.curve.filter_grid_adapt(self.grid,cfg)

    def test_staged_prune_does_not_disable_other_guards(self):
        cfg={**self.args,'wrap_to_rigid':True,
             'sdf_candidate_interval_prune':True}
        with self.assertRaisesRegex(ValueError,'global-statistic'):
            self.curve.filter_grid_adapt(self.grid,cfg)

    def test_probe_yaml_explicitly_stops_before_final_mesh(self):
        from pathlib import Path
        import yaml
        config_path=Path(__file__).resolve().parents[1]/'exp/train/manipulation/adapt_oktopus_puffer_root_leaves_0052_roi_body_audit.yaml'
        cfg=yaml.safe_load(config_path.read_text())
        self.assertEqual(cfg['sdf_puffer_stop_after'],1)
        self.assertTrue(cfg['sdf_candidate_interval_audit'])
        self.assertTrue(cfg['sdf_candidate_interval_prune'])
        self.assertTrue(cfg['sdf_puffer_minimal_staging'])
        self.assertEqual(len(cfg['adaptations']),15)

if __name__=='__main__': unittest.main()
