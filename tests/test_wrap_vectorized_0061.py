"""0061: exact-index, numerically equivalent four-corner wrap interpolation."""
import contextlib
import io
import unittest
from pathlib import Path
import numpy as np
import yaml

from curve_functions._interpolate import interpolate_wrap_radius1
from test_localize_reuse_0059 import WorkingCurve, ExactReuse0059


class WrapVectorized0061Tests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(61)
        self.ts = np.concatenate([
            rng.uniform(-.5, 1.5, 4000),
            np.array([-10., -1., 0., .25, .5, 1., 1.001, 5.])
        ])
        self.theta = np.concatenate([
            rng.uniform(-11 * np.pi, 11 * np.pi, 4000),
            np.array([-np.pi, np.pi, -3 * np.pi, 3 * np.pi,
                      0., 2 * np.pi, np.pi / 2, -np.pi / 2])
        ])
        self.s_bins = np.array([0., .03, .15, .18, .51, .83, 1.])
        self.theta_bins = np.linspace(-np.pi, np.pi, 24, endpoint=False)
        self.wrap = .6 + rng.random((len(self.s_bins), len(self.theta_bins)))

    def run_both(self, ts=None, theta=None, wrap=None, theta_bins=None,
                 s_bins=None):
        obj = type('Curve', (), {})()
        args = (self.ts if ts is None else ts,
                self.theta if theta is None else theta,
                self.wrap if wrap is None else wrap,
                self.theta_bins if theta_bins is None else theta_bins,
                self.s_bins if s_bins is None else s_bins)
        baseline = interpolate_wrap_radius1(obj, *args)
        optimized = interpolate_wrap_radius1(obj, *args,
                         vectorized_0061=True, audit_0061=True)
        np.testing.assert_allclose(optimized, baseline, rtol=1e-12, atol=1e-12,
                                   equal_nan=True)
        return optimized, obj

    def test_nonuniform_s_and_wrapped_angles(self):
        optimized, obj = self.run_both()
        self.assertEqual(obj._wrap_vector_stats_0061['audited_points'], len(self.ts))
        self.assertEqual(obj._wrap_vector_stats_0061['fallback_calls'], 0)
        self.assertEqual(optimized.shape, self.ts.shape)

    def test_small_single_empty_and_edge_cases(self):
        for count in (0, 1, 2, 31):
            with self.subTest(count=count):
                self.run_both(self.ts[:count], self.theta[:count])
        # Sample exactly at the s and angular bins as well as near them.
        ts = np.repeat(self.s_bins, len(self.theta_bins))
        theta = np.tile(self.theta_bins, len(self.s_bins))
        self.run_both(ts, theta)
        self.run_both(np.nextafter(ts, -np.inf),
                      np.nextafter(theta, np.inf))

    def test_supported_nonfinite_queries(self):
        self.run_both(np.array([np.nan, .4, .7, .0]),
                      np.array([0., np.nan, np.inf, -.2]))

    def test_degenerate_longitudinal_bins_fallback(self):
        self.run_both(s_bins=np.array([0., .0, .2, .7, 1., 1.1, 2.]))

    def test_integration_acceptance_and_all_fields(self):
        fixture = ExactReuse0059()
        fixture.setUp()
        def run(fast, audit):
            core = WorkingCurve()
            target = WorkingCurve()
            # Give the wrap actual theta-dependent geometry, to exercise all
            # four gathers. The previous 0059 tests used a constant stub.
            rng = np.random.default_rng(61)
            core.key_wrap_radius_full = .1 + rng.random((6, 16)) * .1
            target.key_wrap_radius_full = .2 + rng.random((6, 16)) * .2
            config = dict(
                accessory_curve_handle=type('Accessory', (), {'core': target})(),
                src_0=.4, src_1=.9, tgt_0=.1, tgt_1=.5,
                scale=.72, rot_deg=0., wrap_radius=True,
                rigid_radius=False, adapt_debug_counts=False,
                avatar_cylinder_radius_scale=.8,
                avatar_cylinder_radius_add=.05,
                adapt_localize_norm=1.2,
                wrap_angular_follow=1.,
                sdf_localize_after_gate=True,
                sdf_localize_sphere_0058=True,
                sdf_localize_reuse_0059=True,
                sdf_vectorize_wrap_0061=fast,
                sdf_vectorize_wrap_audit_0061=audit,
                sdf_localize_profile_0060=True,
                sdf_projection_workers_0054=2,
                sdf_projection_cache_0054=True,
                sdf_projection_refine_fast_0055=True,
                sdf_projection_leafsize_0056=32)
            with contextlib.redirect_stdout(io.StringIO()):
                result = core.localize_samples_adapt(fixture.samples, config)
            return result, core, target
        (reference, _, _), (fast, src, tgt) = run(False, False), run(True, True)
        np.testing.assert_array_equal(reference[2], fast[2])
        for left, right in zip(reference[:2], fast[:2]):
            self.assertEqual(set(left), set(right))
            for key in left:
                np.testing.assert_allclose(left[key], right[key],
                    atol=1e-12, rtol=1e-12, equal_nan=True,
                    err_msg=key)
        self.assertGreater(src._wrap_vector_stats_0061['points'], 0)
        self.assertGreater(tgt._wrap_vector_stats_0061['points'], 0)
        self.assertGreater(src._wrap_vector_stats_0061['audited_points'], 0)

    def test_yaml_and_forwarding(self):
        root = Path(__file__).resolve().parents[1]
        for mode in ('audit', 'profile'):
            cfg = yaml.safe_load((root/'exp/train/manipulation'/
                 f'adapt_oktopus_puffer_root_leaves_0061_body_{mode}.yaml').read_text())
            self.assertTrue(cfg['sdf_vectorize_wrap_0061'])
            self.assertEqual(cfg['sdf_vectorize_wrap_audit_0061'], mode == 'audit')
            self.assertEqual(cfg['sdf_localize_profile_0060'], mode == 'audit')
            self.assertFalse(cfg['sdf_localize_sphere_audit_0058'])
            self.assertFalse(cfg['sdf_localize_reuse_audit_0059'])
            self.assertEqual(cfg['sdf_puffer_stop_after'], 1)
        src = (root/'src/app/agent_3dvec_sdf.py').read_text()
        self.assertEqual(src.count("'sdf_vectorize_wrap_0061'"), 3)
        self.assertEqual(src.count("'sdf_vectorize_wrap_audit_0061'"), 3)


if __name__ == '__main__':
    unittest.main()
