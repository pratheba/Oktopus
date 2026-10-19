"""0050a: keep full localization fields in the puffer wrap body's bounded stage.

Run with: PYTHONPATH=src:src/ngc python -m unittest discover -s tests -p 'test_puffer_bounded_wrap_0050a.py' -v
"""
import contextlib
import io
import tempfile
import unittest
from unittest import mock

import numpy as np

from ngc.curve_handle import CurveHandle
from ngc.handle_utils.mesh_utils import CylindersMesh
from utils.mc_utils import MCGrid
from test_sdf_wrap_after_gate_0042 import CurveFixture, wrap_radius_stub
import ngc.pwla_curve._localize as localize_module


class AdaptedCurve(CurveHandle):
    def __init__(self, cylinder):
        self.cylinder = cylinder
        self.core = CurveFixture()

    def _CurveHandle__gen_cyl_mesh(self, intpl):
        return self.cylinder


class PufferWrapStagingTests(unittest.TestCase):
    def setUp(self):
        self.grid = MCGrid({'reso': 15, 'size': 1.2, 'level': 0.0})
        rng = np.random.default_rng(420)
        cylinder = CylindersMesh()
        cylinder.cylinder_samples = rng.normal(0, 0.35, (18, 15, 3))
        self.curve = AdaptedCurve(cylinder)
        self.args = {
            'mode': 'direct', 'src_0': 0.10, 'src_1': 0.90,
            'tgt_0': 0.25, 'tgt_1': 0.70, 'scale': 0.72,
            'rot_deg': 0.0, 'wrap_radius': True,
            'rigid_radius': False, 'use_accessory_support_clamp': True,
            'support_rho_limit': 1.0, 'use_soft_snug': True,
            'use_hard_avatar_clamp': True, 'cut_avatar': True,
            'sdf_localize_after_gate': True, 'dcsdd_fast_bbox_marking': True,
            'adapt_debug_counts': False, 'infer_scale': 2.0,
            'accessory_curve_handle': type('Acc', (), {'core': CurveFixture()})(),
        }

    def test_direct_puffer_wrap_staging_matches_unbounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.grid.memmap_dir = tmp
            with mock.patch.object(localize_module, 'interpolate_wrap_radius1', wrap_radius_stub), \
                 contextlib.redirect_stdout(io.StringIO()):
                reference = self.curve.filter_grid_adapt(self.grid, self.args)
                staged = self.curve.filter_grid_adapt(self.grid, {
                    **self.args,
                    'dcsdd_bounded_filtering': True,
                    '_sdf_puffer_bounded_wrap_staging': True,
                    'dcsdd_filter_batch_size': 43,
                })
            for reference_data, staged_data in zip(reference[:2], staged[:2]):
                self.assertEqual(set(reference_data), set(staged_data))
                for key in reference_data:
                    expected, actual = reference_data[key], staged_data[key]
                    if isinstance(expected, np.ndarray):
                        np.testing.assert_array_equal(expected, actual, err_msg=key)
            np.testing.assert_array_equal(reference[2], staged[2])
            np.testing.assert_array_equal(reference[3], staged[3])
            self.assertTrue(hasattr(self.grid, '_bounded_scratch'))

    def test_unapproved_wrap_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'bounded filtering supports'):
            self.curve.filter_grid_adapt(self.grid, {
                **self.args, 'dcsdd_bounded_filtering': True})

    def test_puffer_opt_in_rejects_callback_and_global_statistics(self):
        base = {**self.args, 'dcsdd_bounded_filtering': True,
                '_sdf_puffer_bounded_wrap_staging': True}
        with self.assertRaisesRegex(ValueError, 'requires staged direct'):
            self.curve.filter_grid_adapt(self.grid, {
                **base, '_dcsdd_stream_callback': lambda *_: None})
        with self.assertRaisesRegex(ValueError, 'global-statistic'):
            self.curve.filter_grid_adapt(self.grid, {
                **base, 'free_anchor_to_release_scale': True})


if __name__ == '__main__':
    unittest.main()
