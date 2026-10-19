"""0041: the optimized SDF localizer must keep the exact cylinder acceptance.

Run: PYTHONPATH=src:src/ngc python -m unittest discover -s tests \
    -p test_sdf_localize_after_gate_0041.py -v
"""
import unittest
import numpy as np

from ngc.pwla_curve._localize import _LocalizeMixin


class ProjectionFixture(_LocalizeMixin):
    def __init__(self, ts):
        self.ts = np.asarray(ts, dtype=np.float64)
        self.train_rows = []

    def curve_projection_interval(self, samples, s0, s1, *, outside=False):
        return self.ts.copy()

    def curve_projection(self, samples, outside=False):
        return self.ts.copy()

    def calc_x_radius(self, ts):
        return .25 + .2 * np.asarray(ts)

    def interpolate(self, ts, *, radius_type='train', points=True, radius=True,
                    frame=True):
        ts = np.asarray(ts, dtype=np.float64)
        if radius_type == 'train' and not points and not frame and radius:
            self.train_rows.append(len(ts))
        result = {}
        if points:
            result['points'] = np.stack((ts - .5, ts * 0, ts * 0), axis=1)
        if frame:
            result['frame'] = np.broadcast_to(np.eye(3), (len(ts), 3, 3)).copy()
        if radius:
            base = .12 if radius_type == 'train' else .15
            result['radius'] = np.stack((base + .05 * ts,
                                          base + .08 * (1. - ts)), axis=1)
        return result


class AfterGateParityTests(unittest.TestCase):
    def _check(self, seed, n, norm):
        rng = np.random.default_rng(seed)
        ts = rng.uniform(-.1, 1.1, n)
        # Explicit points outside projection and inside cylinder; also include
        # many rejected points to exercise actual deferred work.
        points = rng.normal(0., .4, (n, 3))
        points[:, 0] += np.clip(ts, 0., 1.) - .5
        reference = ProjectionFixture(ts)
        optimized = ProjectionFixture(ts)
        base, idx_base = reference.localize_samples(
            points, norm=norm, projection_s0=.0, projection_s1=1.,
            runtime_cylinder_radius_scale=1.1,
            runtime_cylinder_radius_add=.017)
        fast, idx_fast = optimized.localize_samples(
            points, norm=norm, projection_s0=.0, projection_s1=1.,
            runtime_cylinder_radius_scale=1.1,
            runtime_cylinder_radius_add=.017,
            sdf_localize_after_gate=True)
        np.testing.assert_array_equal(idx_fast, idx_base)
        self.assertEqual(set(fast), set(base))
        for key in base:
            np.testing.assert_array_equal(fast[key], base[key], err_msg=key)
        self.assertEqual(reference.train_rows, [np.count_nonzero((ts >= 0) & (ts <= 1))])
        self.assertEqual(optimized.train_rows, [len(idx_base)])

    def test_random_full_and_partial_acceptance(self):
        for seed in range(10):
            for norm in (.8, 1.5, 3.0, 5.0):
                with self.subTest(seed=seed, norm=norm):
                    self._check(seed, n=1031, norm=norm)

    def test_all_rejected(self):
        fixture = ProjectionFixture([.4, .7, 1.2])
        points = np.array([[9., 9., 9.], [9., 8., 9.], [0., 0., 0.]])
        output, inside = fixture.localize_samples(
            points, norm=1., projection_s0=0., projection_s1=1.,
            sdf_localize_after_gate=True)
        self.assertEqual(len(inside), 0)
        self.assertEqual(output['samples_local'].shape, (0, 3))
        self.assertEqual(output['radius'].shape, (0, 2))


if __name__ == '__main__':
    unittest.main()
