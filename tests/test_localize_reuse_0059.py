"""0059 exact reuse of existing projected avatar fields in the adaptation path."""
import contextlib
import io
import unittest
from unittest import mock
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import yaml
from ngc.pwla_curve._geometry import _GeometryMixin
from ngc.pwla_curve._localize import _LocalizeMixin
from ngc.pwla_curve._interpolate import _InterpolateMixin
from ngc.pwla_curve._radius import _RadiusMixin
from ngc.pwla_curve._frames import _FramesMixin
from ngc.pwla_curve import _localize as localize_module


class WorkingCurve(_GeometryMixin, _LocalizeMixin, _InterpolateMixin,
                   _RadiusMixin, _FramesMixin):
    def __init__(self):
        self.key_ts = np.array([0., .18, .35, .64, .83, 1.])
        self.key_points = np.array([
            [-1., .0, .1], [-.6, .4, .0], [-.25, .55, .3],
            [.3, .3, -.25], [.6, -.3, -.3], [1., -.4, .0]])
        self.key_cylinder_radius = np.array([
            [.19, .18], [.2, .24], [.16, .2], [.25, .26],
            [.21, .15], [.15, .22]])
        self.key_train_radius = self.key_cylinder_radius * .8
        # Vary tangent/normal/binormal across the curve; identity frames
        # alone would not exercise repeated Gram-Schmidt interpolation.
        angles = np.array([[0., 0., 0.], [5., -7., 12.], [12., 3., 25.],
                           [-8., 17., 35.], [4., -12., 50.], [10., 5., 70.]])
        self.key_frame = Rotation.from_euler('xyz', angles, degrees=True).as_matrix()
        self.core = self
        self.wrap_s_bins_full = np.linspace(0., 1., 6)
        self.wrap_theta_bins = np.linspace(-np.pi, np.pi, 16, endpoint=False)
        self.key_wrap_radius_full = np.tile(
            np.linspace(.15, .22, 6)[:, None], (1, 16))
        self.wrap_s_bins = self.wrap_s_bins_full

    def update_coords(self):
        pass

    def update_frame(self):
        pass

    def calc_curve_length(self):
        return 1.0, None

    def map_coords_to_by_arclen(self, coords, target, s0, s1, t0, t1):
        return t0 + (np.asarray(coords) - s0) * (t1 - t0) / (s1 - s0)


def wrap_stub(_curve, coords, theta, _wrap, _theta_bins, _s_bins):
    return .16 + .03 * np.asarray(coords) + .01 * np.cos(np.asarray(theta))


class ExactReuse0059(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(59159)
        c = WorkingCurve()
        coords = rng.uniform(.4, .9, 4000)
        pts = c.interpolate(coords, radius=False, frame=False)['points']
        near = pts + rng.normal(0, .10, pts.shape)
        outer = rng.uniform(-2, 2, (5000, 3))
        self.samples = np.concatenate([near, outer])
        self.options = dict(
            norm=1., projection_s0=.4, projection_s1=.9,
            projection_workers_0054=2, projection_cache_0054=True,
            projection_refine_fast_0055=True,
            projection_leafsize_0056=32,
            sdf_localize_after_gate=True, projection_sphere_0058=True)

    def test_first_pass_minimal_values_are_exact(self):
        c = WorkingCurve()
        ref, ids = c.localize_samples(self.samples, **self.options)
        fast, new_ids = c.localize_samples(self.samples,
                     projection_reuse_0059=True, **self.options)
        np.testing.assert_array_equal(ids, new_ids)
        self.assertEqual(set(fast) - {'_proj_points_0059', '_local0_0059'},
                         {'samples', 'coords', 'radius', 'radius_cylinder', 'frame_mat'})
        for k in ('samples', 'coords', 'radius', 'radius_cylinder', 'frame_mat'):
            np.testing.assert_array_equal(ref[k], fast[k], err_msg=k)
        self.assertGreater(len(ids), 0)

    def _run_adapt(self, use_reuse, wrap, scale=1., add=0., norm=1.):
        c = WorkingCurve()
        target = WorkingCurve()
        arg = dict(accessory_curve_handle=type('Accessory', (), {'core': target})(),
                   src_0=.4, src_1=.9, tgt_0=.1, tgt_1=.5,
                   scale=.72, rot_deg=0., wrap_radius=wrap,
                   rigid_radius=not wrap, adapt_debug_counts=False,
                   avatar_cylinder_radius_scale=scale,
                   avatar_cylinder_radius_add=add,
                   adapt_localize_norm=norm,
                   wrap_angular_follow=0.,
                   sdf_localize_after_gate=True,
                   sdf_localize_sphere_0058=True,
                   sdf_localize_reuse_0059=use_reuse,
                   sdf_projection_workers_0054=2,
                   sdf_projection_cache_0054=True,
                   sdf_projection_refine_fast_0055=True,
                   sdf_projection_leafsize_0056=32)
        with mock.patch.object(localize_module, 'interpolate_wrap_radius1', wrap_stub):
            with contextlib.redirect_stdout(io.StringIO()):
                return c.localize_samples_adapt(self.samples, arg)

    def test_full_adaptation_parity_exact(self):
        for wrap in (False, True):
            for scale, add, norm in ((1., 0., 1.), (.8, .05, 1.2)):
                with self.subTest(wrap=wrap, scale=scale, add=add):
                    ref = self._run_adapt(False, wrap, scale, add, norm)
                    result = self._run_adapt(True, wrap, scale, add, norm)
                    np.testing.assert_array_equal(ref[2], result[2])
                    for label, a, b in zip(('accessory', 'avatar'), ref[:2], result[:2]):
                        self.assertEqual(set(a), set(b), label)
                        for k in a:
                            np.testing.assert_array_equal(a[k], b[k],
                                                          err_msg=f'{label}.{k}')

    def test_empty_and_all_outside_parity(self):
        for samples in (np.empty((0, 3), dtype=np.float64),
                        np.full((32, 3), 1000., dtype=np.float64)):
            c = WorkingCurve()
            ref, ids = c.localize_samples(samples, **self.options)
            fast, kept = c.localize_samples(samples,
                projection_reuse_0059=True, **self.options)
            np.testing.assert_array_equal(ids, kept)
            for key in ('samples', 'coords', 'radius', 'radius_cylinder', 'frame_mat'):
                np.testing.assert_array_equal(ref[key], fast[key], err_msg=key)

    def test_unsupported_modes_fail(self):
        c = WorkingCurve()
        for diff in ({'projection_sphere_0058': False},
                     {'sdf_localize_after_gate': False},
                     {'k_project': 2}):
            with self.subTest(diff=diff), self.assertRaisesRegex(ValueError, '0059'):
                c.localize_samples(self.samples[:10], projection_reuse_0059=True,
                                   **{**self.options, **diff})

    def test_config_and_forwarding(self):
        project = Path(__file__).resolve().parents[1]
        for profile in ('audit', 'profile'):
            cfg = yaml.safe_load((project/'exp/train/manipulation'/
                f'adapt_oktopus_puffer_root_leaves_0059_body_{profile}.yaml').read_text())
            self.assertTrue(cfg['sdf_localize_reuse_0059'])
            self.assertEqual(cfg['sdf_localize_reuse_audit_0059'], profile == 'audit')
            self.assertEqual(cfg['sdf_localize_reuse_verified_0059'], profile == 'profile')
            self.assertEqual(cfg['sdf_puffer_stop_after'], 1)
        s = (project/'src/app/agent_3dvec_sdf.py').read_text()
        self.assertEqual(s.count("'sdf_localize_reuse_0059'"), 3)


if __name__ == '__main__':
    unittest.main()
