"""0055 exact nearest-curve refinement parity, endpoints, caching, loop branches.

PYTHONPATH=src:src/ngc python -m unittest discover -s tests -p 'test_projection_refinement_0055.py' -v
"""
import unittest
import numpy as np
from ngc.pwla_curve._geometry import _GeometryMixin


class _Polyline(_GeometryMixin):
    def __init__(self):
        self.key_ts = np.array([0., .15, .4, .7, 1.], dtype=np.float64)
        self.key_points = np.array([
            [0., 0., 0.], [.2, .1, 0.], [.5, .35, .05],
            [.85, -.15, .15], [1.2, -.25, .2]
        ], dtype=np.float64)
    def interpolate(self, ts, radius=False, frame=False):
        return {'points': np.stack([
            np.interp(ts, self.key_ts, self.key_points[:, j]) for j in range(3)
        ], axis=1)}


class ExactRefinementTests(unittest.TestCase):
    def setUp(self):
        self.c = _Polyline()
        rng = np.random.default_rng(8855)
        self.samples = rng.uniform(-.6, 1.8, size=(8003, 3))

    def test_exact_interval_and_looping_candidates(self):
        for s0, s1 in [(.2, .8), (0., 1.), (.1, .25), (.97, 1.)]:
            for outside in (False, True):
                for k in (1, 3):
                    args = dict(projection_cache_0054=True,
                                projection_workers_0054=2,
                                projection_profile_split_0055=True)
                    if k == 1:
                        baseline = self.c.curve_projection_interval(
                            self.samples, s0, s1, outside=outside, **args)
                        optimized = self.c.curve_projection_interval(
                            self.samples, s0, s1, outside=outside,
                            projection_refine_fast_0055=True, **args)
                    else:
                        baseline = self.c.curve_projection_interval_candidates(
                            self.samples, s0, s1, k, outside=outside, **args)
                        optimized = self.c.curve_projection_interval_candidates(
                            self.samples, s0, s1, k, outside=outside,
                            projection_refine_fast_0055=True, **args)
                    np.testing.assert_array_equal(optimized, baseline)

    def test_exact_assign_ts_endpoints_and_degenerate_edges(self):
        rng = np.random.default_rng(444)
        for nv in (0, 1, 2, 20):
            verts = rng.normal(size=(nv, 3))
            if nv > 3:
                verts[3] = verts[2]
            params = np.linspace(.05, .96, max(1, nv))[:nv]
            vids = rng.integers(0, max(1, nv), size=103, dtype=np.int64)
            for outside in (False, True):
                original = self.c._assign_ts(self.samples[:103], vids, params, verts, outside)
                fast = self.c._assign_ts_fast_0055(self.samples[:103], vids, params, verts, outside)
                np.testing.assert_array_equal(original, fast)

    def test_fast_cache_invalidated_inplace_curve_update(self):
        b = self.c.curve_projection_interval(self.samples, .2, .8,
            projection_refine_fast_0055=True, projection_cache_0054=True)
        cached_verts = self.c._projection_edges_cache_0055[0]
        self.c.key_points[2, 2] += .32
        actual = self.c.curve_projection_interval(self.samples, .2, .8,
            projection_refine_fast_0055=True, projection_cache_0054=True)
        expected = self.c.curve_projection_interval(self.samples, .2, .8,
            projection_cache_0054=False)
        np.testing.assert_array_equal(actual, expected)
        self.assertIsNot(cached_verts, self.c._projection_edges_cache_0055[0])
        self.assertFalse(np.array_equal(b, actual))

    def test_profile_knn_and_refinement(self):
        self.c.curve_projection_interval(self.samples, .2, .8,
            projection_refine_fast_0055=True, projection_profile_split_0055=True)
        self.assertGreater(self.c._projection_knn_seconds_0055, 0)
        self.assertGreater(self.c._projection_refine_seconds_0055, 0)


if __name__ == '__main__':
    unittest.main()
