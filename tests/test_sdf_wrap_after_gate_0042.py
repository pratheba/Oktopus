"""0042: direct wrap and rigid adaptation parity under after-cylinder-gate localization.

Run: PYTHONPATH=src:src/ngc python -m unittest discover -s tests \
  -p test_sdf_wrap_after_gate_0042.py -v
"""
import contextlib
import io
from unittest import TestCase, mock

import numpy as np
import ngc.pwla_curve._localize as localize_module
from ngc.pwla_curve._localize import _LocalizeMixin


class CurveFixture(_LocalizeMixin):
    def __init__(self, offset=0.0):
        self.offset = float(offset)
        self.wrap_s_bins_full = np.linspace(0., 1., 6)
        self.wrap_theta_bins = np.linspace(-np.pi, np.pi, 16, endpoint=False)
        self.key_wrap_radius_full = np.tile(
            np.linspace(.15, .22, 6)[:, None], (1, 16))
        self.wrap_s_bins = self.wrap_s_bins_full
        self.core = self
        self.train_rows = []

    def update_coords(self):
        pass

    def update_frame(self):
        pass

    def curve_projection_interval(self, samples, s0, s1, *, outside=False, **_projection_flags):
        return np.asarray(samples)[:, 0] + .5

    def curve_projection(self, samples, *, outside=False, **_projection_flags):
        return np.asarray(samples)[:, 0] + .5

    def calc_x_radius(self, ts):
        return .3 + .1 * np.asarray(ts)

    def calc_curve_length(self):
        return 1.0, None

    def map_coords_to_by_arclen(self, coords, _accessory, s0, s1, t0, t1):
        return t0 + (np.asarray(coords) - s0) * (t1 - t0) / (s1 - s0)

    def interpolate(self, ts, *, radius_type='train', points=True,
                    radius=True, frame=True):
        ts = np.asarray(ts)
        result = {}
        if points:
            result['points'] = np.stack((ts - .5 + self.offset,
                                         np.zeros_like(ts), np.zeros_like(ts)), axis=1)
        if frame:
            result['frame'] = np.broadcast_to(np.eye(3), (len(ts), 3, 3)).copy()
        if radius:
            if radius_type == 'train' and not points and not frame:
                self.train_rows.append(len(ts))
            val = .16 if radius_type == 'cylinder' else .18
            result['radius'] = np.stack((val + .025 * ts,
                                          val + .02 * (1. - ts)), axis=1)
        return result

    def interpolate_snug_scale_field(self, _field, coords, theta):
        return .96 + .02 * np.cos(np.asarray(theta))


def wrap_radius_stub(_curve, coords, theta, _wrap, _theta_bins, _s_bins):
    return .16 + .03 * np.asarray(coords) + .01 * np.cos(np.asarray(theta))


class WrapParity(TestCase):
    def check(self, *, wrap, tilt=False, snug=False):
        rng = np.random.default_rng(118)
        points = np.empty((1503, 3), dtype=np.float64)
        points[:, 0] = rng.uniform(-.45, .55, len(points))
        points[:, 1:] = rng.normal(0., .25, (len(points), 2))
        # Include partial source interval and out-of-cylinder samples.
        arg = {
            'src_0': .08, 'src_1': .88, 'tgt_0': .1, 'tgt_1': .75,
            'scale': 1.7, 'rot_deg': 11.0, 'adapt_localize_norm': 1.25,
            'avatar_cylinder_radius_scale': 1.0,
            'avatar_cylinder_radius_add': .05,
            'wrap_radius': wrap, 'rigid_radius': not wrap,
            'adapt_debug_counts': False, 'wrap_angular_follow': 0.0,
            'wrap_scale_max': 2.5,
        }
        if tilt:
            arg.update(tilt_v=-1.5, tilt_v_anchor=.1, tilt_v_end=.7)
        if snug:
            arg['avatar_snug_scale_field'] = object()
        legacy = CurveFixture()
        optimized = CurveFixture()
        arg0 = {**arg, 'accessory_curve_handle': type('Accessory', (), {'core': CurveFixture()})()}
        arg1 = {**arg, 'accessory_curve_handle': type('Accessory', (), {'core': CurveFixture()})(),
                'sdf_localize_after_gate': True}
        with mock.patch.object(localize_module, 'interpolate_wrap_radius1', wrap_radius_stub):
            with contextlib.redirect_stdout(io.StringIO()):
                acc0, av0, idx0 = legacy.localize_samples_adapt(points, arg0)
                acc1, av1, idx1 = optimized.localize_samples_adapt(points, arg1)
        np.testing.assert_array_equal(idx0, idx1)
        self.assertGreater(len(idx0), 0)
        self.assertLess(len(idx0), len(points))
        for label, old, new in [('avatar', av0, av1), ('accessory', acc0, acc1)]:
            self.assertEqual(set(old), set(new), label)
            for key in old:
                if isinstance(old[key], np.ndarray):
                    np.testing.assert_array_equal(old[key], new[key],
                                                  err_msg=f'{label}.{key}')
        self.assertGreater(legacy.train_rows[0], optimized.train_rows[0])

    def test_puffer_wrap(self):
        self.check(wrap=True)

    def test_cone_wrap_with_tilt_and_snug_field(self):
        self.check(wrap=True, tilt=True, snug=True)

    def test_combined_rigid_piece(self):
        self.check(wrap=False)
