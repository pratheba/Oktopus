"""0054 exact projection parity, one-entry cache invalidation, multithread parity.

Run: PYTHONPATH=src:src/ngc python -m unittest discover -s tests -p 'test_projection_interval_0054.py' -v
"""
import unittest
import numpy as np
from ngc.pwla_curve._geometry import _GeometryMixin


class _SyntheticCurve(_GeometryMixin):
    def __init__(self):
        self.key_ts = np.array([0, .15, .4, .7, 1.], dtype=np.float64)
        self.key_points = np.array([
            [0., 0., 0.], [.2, .1, 0.], [.5, .35, .05],
            [.85, -.15, .15], [1.2, -.25, .2]
        ], dtype=np.float64)
        self.interpolate_calls = 0

    def interpolate(self, ts, radius=False, frame=False):
        self.interpolate_calls += 1
        return {'points': np.stack([
            np.interp(ts, self.key_ts, self.key_points[:, k]) for k in range(3)
        ], axis=1)}


class ProjectionInterval0054(unittest.TestCase):
    def setUp(self):
        self.c = _SyntheticCurve()
        self.points = np.random.default_rng(543).normal(size=(1500, 3)) * .2
        self.points += np.array([.65, .04, .03])

    def test_cache_parity_single_and_multiworker(self):
        baseline = self.c.curve_projection_interval(self.points, .2, .8)
        for nworkers in (1, 2, 4):
            for _ in range(2):
                projected = self.c.curve_projection_interval(
                    self.points, .2, .8,
                    projection_workers_0054=nworkers,
                    projection_cache_0054=True)
                np.testing.assert_array_equal(projected, baseline)

    def test_cache_hit_and_single_entry(self):
        self.c.curve_projection_interval(self.points, .2, .8,
                                         projection_cache_0054=True)
        first = self.c.interpolate_calls
        self.c.curve_projection_interval(self.points, .2, .8,
                                         projection_cache_0054=True)
        self.assertEqual(self.c.interpolate_calls, first)
        self.c.curve_projection_interval(self.points, .1, .8,
                                         projection_cache_0054=True)
        self.assertEqual(self.c.interpolate_calls, first + 1)

    def test_inplace_position_change_invalidates_cache(self):
        self.c.curve_projection_interval(self.points, .2, .8,
                                         projection_cache_0054=True)
        first = self.c.interpolate_calls
        self.c.key_points[2, 1] += .4
        expected = self.c.curve_projection_interval(self.points, .2, .8,
                                                     projection_cache_0054=False)
        actual = self.c.curve_projection_interval(self.points, .2, .8,
                                                   projection_cache_0054=True)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(self.c.interpolate_calls, first + 2)

    def test_inplace_parameter_change_invalidates_cache(self):
        self.c.curve_projection_interval(self.points, .2, .8,
                                         projection_cache_0054=True)
        first = self.c.interpolate_calls
        self.c.key_ts[2] = .42
        expected = self.c.curve_projection_interval(self.points, .2, .8)
        actual = self.c.curve_projection_interval(self.points, .2, .8,
                                                  projection_cache_0054=True)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(self.c.interpolate_calls, first + 2)

    def test_loop_candidate_worker_parity(self):
        baseline = self.c.curve_projection_interval_candidates(self.points, .2, .8, 3)
        for nworkers in (1, 2, 4):
            result = self.c.curve_projection_interval_candidates(
                self.points, .2, .8, 3,
                projection_workers_0054=nworkers,
                projection_cache_0054=True)
            np.testing.assert_array_equal(result, baseline)

    def test_reject_oversubscription(self):
        for invalid in (0, -1, 17):
            with self.assertRaisesRegex(ValueError, 'workers'):
                self.c.curve_projection_interval(self.points, .2, .8,
                                                 projection_workers_0054=invalid)


if __name__ == '__main__':
    unittest.main()
