"""0035: compact boots staging and exact inference partition compatibility.

PYTHONPATH=src:src/ngc python -m unittest discover -s tests -p 'test_dcsdd_compact_2048.py' -v
"""
import tempfile
import unittest
import numpy as np
import torch

from ngc.curve_handle import CurveHandle
from ngc.handle_utils.mesh_utils import CylindersMesh
from utils.mc_utils import MCGrid
from app.agent_3dvec_base import AgentBase
from test_dcsdd_bounded_filtering import FakeCore, FakeCurve


class CompactCore(FakeCore):
    def localize_samples_adapt(self, pts, adapt_arg):
        acc, av, inside = super().localize_samples_adapt(pts, adapt_arg)
        n = len(inside)
        acc['rho'] = np.abs(acc['coords']) * 0.001
        acc['rho_n'] = np.abs(acc['samples_local'][:, 1])
        acc['angles'] = np.arctan2(acc['samples_local'][:, 2], acc['samples_local'][:, 1])
        acc['frame'] = np.broadcast_to(np.eye(3), (n,3,3)).copy()
        return acc, av, inside


class TestCompactStaging(unittest.TestCase):
    def test_compact_same_inference_fields_and_voxels(self):
        keys = ('samples_local','coords','angles','radius','rho','rho_n')
        for memmap in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                grid = MCGrid({'reso':19,'size':1.2,'level':0.0})
                if memmap:
                    grid.memmap_dir=directory
                cyl=CylindersMesh()
                cyl.cylinder_samples=np.random.default_rng(42).normal(0,.26,(15,12,3))
                curve=FakeCurve(cyl)
                curve.core=CompactCore()
                common={'mode':'direct','src_0':.1,'src_1':.9,
                        'use_adapt_control_filter_grid':True,'rigid_radius':True,
                        'dcsdd_fast_bbox_marking':True,'adapt_debug_counts':False}
                ref_acc, ref_av, ref_idx, ref_inside = curve.filter_grid_adapt(grid, common)
                acc, av, idx, inside = curve.filter_grid_adapt(grid, {
                    **common, 'dcsdd_bounded_filtering':True,
                    'dcsdd_compact_staging':True, 'dcsdd_filter_batch_size':19,
                })
                self.assertIsNone(av)
                self.assertIsNone(inside)
                self.assertEqual(set(acc), set(keys))
                np.testing.assert_array_equal(idx, ref_idx)
                for k in keys:
                    np.testing.assert_array_equal(acc[k],ref_acc[k],err_msg=k)
                if memmap:
                    self.assertTrue(hasattr(grid, '_bounded_scratch'))

    def test_inference_slices_same_batches_as_previous_indexing(self):
        class DummyModel:
            def inference_full(self, curve_data, transform=None):
                coords = torch.from_numpy(np.asarray(curve_data['coords'])).float()
                pos = torch.from_numpy(np.asarray(curve_data['samples_local'])).float()
                vals = coords + pos[:, 0] * 0.7
                return {'dist': vals[:,None], 'dist_base': (vals + 0.7)[:,None],
                        'dist_detail': (vals*0.3)[:,None], 'gate_detail': None}
        base=AgentBase()
        base.model=DummyModel()
        base.device=torch.device('cpu')
        base.feat_dict={'boots':0}
        for n in (8,9,10,11,64,65,100,101,202):
            rng=np.random.default_rng(n)
            with tempfile.TemporaryDirectory() as directory:
                arr=np.memmap(directory+'/coords.bin',dtype='float64',mode='w+',shape=(n,))
                arr[:]=rng.normal(size=n)
                data={'coords':arr,'samples_local':rng.normal(size=(n,3))}
                for size in (7,17,33):
                    original=base._inference_full_vals(data,'boots',batch_size=size)
                    sliced=base._inference_full_vals(data,'boots',batch_size=size,slice_batches=True)
                    self.assertEqual(set(original),set(sliced))
                    for key in original:
                        np.testing.assert_array_equal(original[key],sliced[key],err_msg=f'{n}/{size}/{key}')

if __name__=='__main__':
    unittest.main()
