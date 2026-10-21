"""0058 conservative nearest-vertex sphere gate and exact accepted-row parity.

PYTHONPATH=src:src/ngc python -m unittest discover -s tests -p 'test_localize_sphere_0058.py' -v
"""
import unittest
import numpy as np
from ngc.pwla_curve._geometry import _GeometryMixin
from ngc.pwla_curve._localize import _LocalizeMixin
from ngc.pwla_curve._interpolate import _InterpolateMixin
from ngc.pwla_curve._radius import _RadiusMixin
from ngc.pwla_curve._frames import _FramesMixin


class TestCurve(_GeometryMixin, _LocalizeMixin, _InterpolateMixin,
                _RadiusMixin, _FramesMixin):
    def __init__(self):
        self.key_ts = np.array([0., .18, .35, .64, .83, 1.])
        self.key_points = np.array([
            [-1., .0, .1], [-.6, .4, .0], [-.25, .55, .3],
            [.3, .3, -.25], [.6, -.3, -.3], [1., -.4, .0]])
        self.key_cylinder_radius = np.array([
            [.19, .18], [.2, .24], [.16, .2], [.25, .26],
            [.21, .15], [.15, .22]])
        self.key_train_radius = self.key_cylinder_radius * 0.8
        self.key_frame = np.tile(np.eye(3), (len(self.key_ts), 1, 1))


class SphereGate0058Tests(unittest.TestCase):
    def setUp(self):
        self.c = TestCurve()
        rng = np.random.default_rng(58058)
        outer = rng.uniform(-2, 2, (16000, 3))
        coords = rng.uniform(.15, .85, 5000)
        center = self.c.interpolate(coords, radius=False, frame=False)['points']
        near = center + rng.normal(0, .085, (len(center), 3))
        self.samples = np.concatenate([outer, near])
        self.options = dict(norm=1., projection_s0=.15, projection_s1=.85,
                            projection_workers_0054=2,
                            projection_cache_0054=True,
                            projection_refine_fast_0055=True,
                            projection_leafsize_0056=32,
                            sdf_localize_after_gate=True)

    def test_parity_localized_fields_and_original_row_indices(self):
        ref, ids = self.c.localize_samples(self.samples, **self.options)
        actual, kept = self.c.localize_samples(self.samples,
                projection_sphere_0058=True, **self.options)
        self.assertGreater(self.c._projection_sphere_rejected_0058, 0)
        self.assertLess(self.c._projection_sphere_rejected_0058, len(self.samples))
        np.testing.assert_array_equal(ids, kept)
        self.assertEqual(set(ref), set(actual))
        for k in ref:
            np.testing.assert_array_equal(ref[k], actual[k], err_msg=k)

    def test_nontrivial_radius_scale_add_parity(self):
        for scale, add, norm in ((2., .12, 1.), (.5, -.02, .95),
                                  (-1.2, .3, 1.0), (0., 1.5, 1.)):
            opts = {**self.options, 'runtime_cylinder_radius_scale': scale,
                    'runtime_cylinder_radius_add': add, 'norm': norm}
            ref, ids = self.c.localize_samples(self.samples, **opts)
            actual, kept = self.c.localize_samples(self.samples,
                projection_sphere_0058=True, **opts)
            np.testing.assert_array_equal(ids, kept)
            for k in ref:
                np.testing.assert_array_equal(ref[k], actual[k], err_msg=k)

    def test_distance_bound_contains_all_ref_inside(self):
        pts, verts, tree, _, _ = self.c._projection_skeleton_interval(
            .15, .85, 200, cache_0054=True, leafsize_0056=32)
        bound = self.c._projection_sphere_bound_0058(verts, pts, 1., 1., 0.)
        self.assertIsNotNone(bound)
        data, inside = self.c.localize_samples(self.samples, **self.options)
        distances, _ = tree.query(self.samples, workers=2)
        self.assertTrue(np.all(distances[inside] <= bound),
            'Conservative gate would exclude a genuinely accepted point')

    def test_fallback_for_unverified_x_radius(self):
        self.c.calc_x_radius = lambda s: np.full(len(s), .1)
        opts = dict(self.options)
        before, rows = self.c.localize_samples(self.samples, **opts)
        after, newrows = self.c.localize_samples(self.samples,
            projection_sphere_0058=True, **opts)
        np.testing.assert_array_equal(rows, newrows)
        self.assertEqual(self.c._projection_sphere_rejected_0058, 0)
        for k in before:
            np.testing.assert_array_equal(before[k], after[k])

    def test_empty_and_all_surviving_cases(self):
        for sample in (np.zeros((0, 3)),
                       np.array([[.2, .2, .2], [100., 100., 100.]])):
            original, ids = self.c.localize_samples(sample, **self.options)
            result, got = self.c.localize_samples(sample,
                projection_sphere_0058=True, **self.options)
            np.testing.assert_array_equal(ids, got)
            for k in original:
                np.testing.assert_array_equal(original[k], result[k], err_msg=k)

    def test_rejects_unsupported_modes(self):
        for change in ({'k_project': 3}, {'outside': True},
                       {'return_sdf': True}, {'update_curve': True},
                       {'sdf_localize_after_gate': False},
                       {'projection_s0': None}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, '0058'):
                self.c.localize_samples(self.samples[:100],
                    projection_sphere_0058=True, **{**self.options, **change})

    def test_yaml_config_routing(self):
        from pathlib import Path
        import yaml
        project = Path(__file__).resolve().parents[1]
        for profile in ('audit', 'profile'):
            config = yaml.safe_load((project / 'exp/train/manipulation' /
                ('adapt_oktopus_puffer_root_leaves_0058_body_' + profile + '.yaml')).read_text())
            self.assertTrue(config['sdf_localize_sphere_0058'])
            self.assertEqual(config['sdf_puffer_stop_after'], 1)
            self.assertEqual(config['sdf_localize_sphere_audit_0058'], profile == 'audit')
        agent = (project / 'src/app/agent_3dvec_sdf.py').read_text()
        self.assertGreaterEqual(agent.count("'sdf_localize_sphere_0058'"), 3)


if __name__ == '__main__':
    unittest.main()
