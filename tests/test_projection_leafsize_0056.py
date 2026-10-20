"""0056 leafsize opt-in checks: near-field exact parity and cache invalidation."""
import unittest
import numpy as np
from ngc.pwla_curve._geometry import _GeometryMixin


class Curve(_GeometryMixin):
    def __init__(self):
        self.key_ts = np.array([0, .2, .4, .7, 1.], dtype=np.float64)
        self.key_points = np.array([[0,0,0],[.18,.20,.03],[.39,.45,.1],
                                    [.74,.30,.25],[1.2,.1,.12]], dtype=np.float64)

    def interpolate(self, ts, radius=False, frame=False):
        return {'points': np.column_stack([
            np.interp(ts, self.key_ts, self.key_points[:, j]) for j in range(3)])}


class Leafsize0056Tests(unittest.TestCase):
    def setUp(self):
        self.curve = Curve()
        rng = np.random.default_rng(20261020)
        self.samples = rng.uniform(-.5, 1.7, (24007, 3))

    def test_leafsize_32_keeps_near_field_projection_values(self):
        # Random samples avoid exact ties between vertices; both KD-tree
        # queries are exact (eps=0). Tie order is not promised across trees.
        for k in (1, 3):
            for outside in (False, True):
                for s0, s1 in ((.2,.9), (.0,1.), (.1,.28)):
                    opts = dict(projection_cache_0054=True,
                                projection_refine_fast_0055=True,
                                projection_workers_0054=2)
                    if k == 1:
                        fn = self.curve.curve_projection_interval
                        args = (self.samples,s0,s1)
                    else:
                        fn = self.curve.curve_projection_interval_candidates
                        args = (self.samples,s0,s1,k)
                    baseline = fn(*args, outside=outside,
                                  projection_leafsize_0056=10, **opts)
                    tuned = fn(*args, outside=outside,
                               projection_leafsize_0056=32, **opts)
                    np.testing.assert_array_equal(baseline, tuned)

    def test_cached_tree_recreated_on_leafsize_or_curve_change(self):
        opts = dict(projection_cache_0054=True, projection_refine_fast_0055=True)
        self.curve.curve_projection_interval(self.samples[:100], .2, .8,
                                             projection_leafsize_0056=10, **opts)
        tree10 = self.curve._projection_interval_cache_0054[1][2]
        self.curve.curve_projection_interval(self.samples[:100], .2, .8,
                                             projection_leafsize_0056=32, **opts)
        tree32 = self.curve._projection_interval_cache_0054[1][2]
        self.assertIsNot(tree10, tree32)
        self.assertEqual(tree32.leafsize,32)
        self.curve.curve_projection_interval(self.samples[:100], .2, .8,
                                             projection_leafsize_0056=32, **opts)
        self.assertIs(tree32, self.curve._projection_interval_cache_0054[1][2])
        self.curve.key_points[1, 2] += .2
        self.curve.curve_projection_interval(self.samples[:100], .2, .8,
                                             projection_leafsize_0056=32, **opts)
        self.assertIsNot(tree32, self.curve._projection_interval_cache_0054[1][2])

    def test_invalid_leafsize_fails_closed(self):
        for bad in (-1, 0, 129):
            with self.assertRaises(ValueError):
                self.curve.curve_projection_interval(self.samples[:10], .1, .9,
                                                     projection_leafsize_0056=bad)


if __name__ == '__main__':
    unittest.main()
