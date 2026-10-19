"""0045: prove zero-angular wrap and multiplicative snug no-op parity."""
import contextlib
import io
import unittest
from unittest import mock

import numpy as np
from app.sdf_zero_angle_optimization import can_skip_inert_snug
from ngc.pwla_curve import _localize as localize_module
from test_sdf_wrap_after_gate_0042 import CurveFixture, wrap_radius_stub


class ZeroAngularWrapParity(unittest.TestCase):
    def setUp(self):
        self.points = np.random.default_rng(45).normal(size=(1503, 3)) * .18
        self.points[:, 0] = np.linspace(-.45, .45, len(self.points))
        self.base_arg = {
            'src_0': .08, 'src_1': .88, 'tgt_0': .1, 'tgt_1': .75,
            'scale': 1.7, 'rot_deg': 11.0, 'adapt_localize_norm': 1.25,
            'avatar_cylinder_radius_scale': 1.0,
            'avatar_cylinder_radius_add': .05,
            'wrap_radius': True, 'rigid_radius': False,
            'mode': 'direct', 'adapt_debug_counts': False,
            'wrap_angular_follow': 0.0, 'wrap_scale_max': 2.5,
            'sdf_localize_after_gate': True,
            'auto_avatar_snug_field': True,
        }

    def run_localize(self, arg):
        source = CurveFixture()
        tgt = CurveFixture()
        arg = {**arg, 'accessory_curve_handle': type('Accessory', (), {'core': tgt})()}
        with mock.patch.object(localize_module, 'interpolate_wrap_radius1', wrap_radius_stub):
            with contextlib.redirect_stdout(io.StringIO()):
                accessory, avatar, idx = source.localize_samples_adapt(self.points, arg)
        return accessory, avatar, idx

    def assert_same(self, a, b):
        for pair1, pair2 in zip(a, b):
            if isinstance(pair1, dict):
                self.assertEqual(pair1.keys(), pair2.keys())
                for k in pair1:
                    np.testing.assert_array_equal(pair1[k], pair2[k], err_msg=k)
            else:
                np.testing.assert_array_equal(pair1, pair2)

    def test_multiplicative_snug_has_no_effect_at_zero_follow(self):
        without = self.run_localize(self.base_arg)
        with_snug = self.run_localize({
            **self.base_arg, 'avatar_snug_scale_field': object(),
        })
        self.assert_same(without, with_snug)

    def test_skipping_angular_lookup_is_exact(self):
        base = self.run_localize(self.base_arg)
        fast = self.run_localize({
            **self.base_arg, 'sdf_optimize_zero_angular_wrap': True,
        })
        self.assert_same(base, fast)

    def test_angular_interpolator_not_called_in_fast_mode(self):
        source = CurveFixture()
        tgt = CurveFixture()
        arg = {**self.base_arg, 'sdf_optimize_zero_angular_wrap': True,
               'accessory_curve_handle': type('Accessory', (), {'core': tgt})()}
        with mock.patch.object(localize_module, 'interpolate_wrap_radius1',
                               side_effect=AssertionError('unexpected angular interpolation')):
            with contextlib.redirect_stdout(io.StringIO()):
                source.localize_samples_adapt(self.points, arg)

    def test_inert_snug_guard(self):
        self.assertTrue(can_skip_inert_snug(self.base_arg))
        for changes in (
            {'wrap_angular_follow': .01},
            {'snug_mode': 'additive'},
            {'shared_snug_field': 'output.npz'},
            {'avatar_snug_scale_field': object()},
            {'use_attach_to_free_blend': True},
            {'snug_penetration_guard': True},
            {'wrap_debug': True},
            {'auto_avatar_snug_field': False},
            {'rigid_radius': True},
        ):
            with self.subTest(changes=changes):
                self.assertFalse(can_skip_inert_snug({**self.base_arg, **changes}))


if __name__ == '__main__':
    unittest.main()
