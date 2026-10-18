"""Patch 0034: broad-phase index and chunked-localization parity.

Run with PYTHONPATH=src:src/ngc python -m unittest discover -s tests -p test_dcsdd_bounded_filtering.py -v
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from ngc.handle_utils.mesh_utils import CylindersMesh
from ngc.handle_utils.bounded_filter import iter_bbox_kidx
from ngc.curve_handle import CurveHandle
from utils.mc_utils import MCGrid


class FakeCore:
    def build_adapt_control_field(self, s_full, **kwargs):
        return {"points": np.zeros((len(s_full), 3), dtype=np.float64),
                "frame": np.broadcast_to(np.eye(3), (len(s_full), 3, 3)).copy(),
                "radius": np.ones((len(s_full), 2)) * 0.23}

    def localize_samples_adapt(self, pts, adapt_arg):
        # Pointwise surrogate for the real localizer. Its result does not
        # depend on the size or boundaries of the candidate batch.
        inside = np.flatnonzero((pts[:, 0] + 0.2*pts[:, 1] - 0.17*pts[:, 2]) > -0.1)
        accepted = pts[inside]
        av = {'samples': accepted,
              'samples_local': accepted * 1.5,
              'coords': accepted[:, 0] + 2.0,
              'radius': np.abs(accepted[:, :2]) + 1.0,
              'frame_mat': np.broadcast_to(np.eye(3), (len(inside), 3, 3)).copy()}
        acc = dict(av)
        acc['coords'] = av['coords'] + 3.0
        acc['samples_local'] = av['samples_local'] + 7.0
        return acc, av, inside


class FakeCurve(CurveHandle):
    def __init__(self, cylinder):
        self.cylinder = cylinder
        self.core = FakeCore()

    def _CurveHandle__gen_cyl_mesh(self, intpl):
        return self.cylinder


class BoundedParityTest(unittest.TestCase):
    def test_exact_broad_candidates(self):
        for res in (5, 12, 32):
            for extend in (False, True):
                for seed in range(15):
                    rng = np.random.default_rng(seed+13*res)
                    grid = MCGrid({'reso':res, 'size':1.2, 'level':0.0})
                    cylinder = CylindersMesh()
                    points = rng.normal(0, .45, (9,16,3))
                    if seed % 3 == 0: points[:, :, 1] += 1.6
                    if seed % 5 == 0: points[:, :, 0] -= 2.0
                    cylinder.cylinder_samples = points
                    _, expected = cylinder.filter_grid(grid, extend_bbox=extend, fast_bbox=True)
                    result = list(iter_bbox_kidx(points,grid,extend_bbox=extend,
                                                 batch_size=13,max_slab_voxels=100))
                    actual = np.concatenate(result) if result else np.empty(0,dtype=np.int64)
                    np.testing.assert_array_equal(actual, expected)
                    self.assertTrue(np.all(actual[:-1] < actual[1:]))

    def test_chunk_localization_same_output_and_order(self):
        for res in (11, 19, 33):
            for memmap in (False, True):
                with tempfile.TemporaryDirectory() as directory:
                    grid = MCGrid({'reso':res, 'size':1.2, 'level':0.0})
                    if memmap: grid.memmap_dir = directory
                    rng=np.random.default_rng(res)
                    cylinder=CylindersMesh()
                    cylinder.cylinder_samples=rng.normal(0,.26,(15,12,3))
                    handle=FakeCurve(cylinder)
                    args={'mode':'direct','src_0':.1,'src_1':.9,
                          'use_adapt_control_filter_grid':True,'rigid_radius':True,
                          'dcsdd_fast_bbox_marking':True,
                          'adapt_debug_counts':False}
                    ref=handle.filter_grid_adapt(grid, args)
                    opt=handle.filter_grid_adapt(grid, {
                        **args, 'dcsdd_bounded_filtering':True,
                        'dcsdd_filter_batch_size':19,
                    })
                    for a,b in zip(ref[0:2],opt[0:2]):
                        self.assertEqual(set(a),set(b))
                        for key in a:
                            np.testing.assert_array_equal(a[key], b[key], err_msg=key)
                    np.testing.assert_array_equal(ref[2],opt[2])
                    np.testing.assert_array_equal(ref[3],opt[3])
                    if memmap:
                        # Keep mappings alive until the owner grid is released.
                        self.assertTrue(hasattr(grid,'_bounded_scratch'))


if __name__ == '__main__': unittest.main()
