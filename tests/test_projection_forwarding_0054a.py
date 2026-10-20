"""0054a checks the entire config-to-interval-projection forwarding chain."""
import pathlib
import unittest
import numpy as np
from ngc.pwla_curve._localize import _LocalizeMixin
from ngc.pwla_curve._geometry import _GeometryMixin


class _StopHere(Exception):
    pass


class _ProjectionProbe(_LocalizeMixin):
    def __init__(self):
        self.calls = []

    def curve_projection_interval(self, *args, **kwargs):
        self.calls.append(('single', args, kwargs))
        raise _StopHere

    def curve_projection_interval_candidates(self, *args, **kwargs):
        self.calls.append(('multiple', args, kwargs))
        raise _StopHere


class _AccessoryCore:
    def update_coords(self):
        pass

    def update_frame(self):
        pass


class _AdapterProbe(_LocalizeMixin):
    def localize_samples(self, *args, **kwargs):
        self.forwarded = kwargs
        raise _StopHere


class _FullCurve(_GeometryMixin, _LocalizeMixin):
    def __init__(self):
        self.key_ts = np.array([0., 0.5, 1.], dtype=np.float64)
        self.key_points = np.array([[0., 0., 0.], [0.5, 0.1, 0.], [1., 0.1, 0.]], dtype=np.float64)

    def interpolate(self, ts, *, radius=False, frame=False, points=True, radius_type='train'):
        ts = np.asarray(ts)
        out = {}
        if points:
            out['points'] = np.stack([
                np.interp(ts, self.key_ts, self.key_points[:, j]) for j in range(3)
            ], axis=1)
        if radius:
            out['radius'] = np.full((len(ts), 2), 0.5, dtype=np.float64)
        if frame:
            out['frame'] = np.broadcast_to(np.eye(3), (len(ts), 3, 3)).copy()
        return out

    def calc_x_radius(self, ts):
        return np.ones_like(ts)


class ProjectionForwarding0054a(unittest.TestCase):
    def test_interval_single_query_receives_opt_in_flags(self):
        c = _ProjectionProbe()
        with self.assertRaises(_StopHere):
            c.localize_samples(
                np.zeros((3, 3)), projection_s0=0.2, projection_s1=0.8,
                projection_workers_0054=2,
                projection_cache_0054=True,
                projection_profile_0054=True)
        kind, args, kwargs = c.calls[0]
        self.assertEqual(kind, 'single')
        self.assertEqual(kwargs['projection_workers_0054'], 2)
        self.assertTrue(kwargs['projection_cache_0054'])
        self.assertIsNotNone(args[0])

    def test_interval_multiple_query_receives_opt_in_flags(self):
        c = _ProjectionProbe()
        with self.assertRaises(_StopHere):
            c.localize_samples(
                np.zeros((3, 3)), projection_s0=0.2, projection_s1=0.8,
                k_project=3,
                projection_workers_0054=4,
                projection_cache_0054=True)
        kind, args, kwargs = c.calls[0]
        self.assertEqual(kind, 'multiple')
        self.assertEqual(args[3], 3)
        self.assertEqual(kwargs['projection_workers_0054'], 4)
        self.assertTrue(kwargs['projection_cache_0054'])

    def test_adaptation_passes_flags_to_localizer(self):
        c = _AdapterProbe()
        acc = type('Accessory', (), {'core': _AccessoryCore()})()
        settings = {
            'accessory_curve_handle': acc,
            'src_0': 0.4, 'src_1': 0.9,
            'tgt_0': 0.1, 'tgt_1': 0.5,
            'sdf_projection_workers_0054': 2,
            'sdf_projection_cache_0054': True,
            'sdf_projection_profile_0054': True,
        }
        with self.assertRaises(_StopHere):
            c.localize_samples_adapt(np.zeros((2,3)), settings)
        self.assertEqual(c.forwarded['projection_workers_0054'], 2)
        self.assertTrue(c.forwarded['projection_cache_0054'])
        self.assertTrue(c.forwarded['projection_profile_0054'])
        self.assertEqual(c.forwarded['projection_s0'], 0.4)
        self.assertEqual(c.forwarded['projection_s1'], 0.9)

    def test_default_flags_are_inert(self):
        c = _AdapterProbe()
        acc = type('Accessory', (), {'core': _AccessoryCore()})()
        settings = {
            'accessory_curve_handle': acc,
            'src_0': 0.4, 'src_1': 0.9,
            'tgt_0': 0.1, 'tgt_1': 0.5,
        }
        with self.assertRaises(_StopHere):
            c.localize_samples_adapt(np.zeros((2,3)), settings)
        self.assertEqual(c.forwarded['projection_workers_0054'], 1)
        self.assertFalse(c.forwarded['projection_cache_0054'])
        self.assertFalse(c.forwarded['projection_profile_0054'])

    def test_actual_localizer_profiles_and_preserves_values(self):
        c = _FullCurve()
        rng = np.random.default_rng(765)
        samples = np.column_stack((rng.uniform(0.1, 0.9, 300),
                                   rng.uniform(-0.1, 0.1, 300),
                                   rng.uniform(-0.1, 0.1, 300)))
        old, old_idx = c.localize_samples(
            samples, projection_s0=0.2, projection_s1=0.8,
            sdf_localize_after_gate=True)
        for _ in range(2):
            new, new_idx = c.localize_samples(
                samples, projection_s0=0.2, projection_s1=0.8,
                projection_workers_0054=2,
                projection_cache_0054=True,
                projection_profile_0054=True,
                sdf_localize_after_gate=True)
            np.testing.assert_array_equal(new_idx, old_idx)
            for field in old:
                np.testing.assert_array_equal(new[field], old[field])
        self.assertEqual(c._projection_profile_calls_0054, 2)
        self.assertGreater(c._projection_profile_seconds_0054, 0.)

    def test_puffer_config_forwards_projection_options(self):
        # Structural regression for the top-level puffer forwarding stage,
        # which was absent in 0054 despite the YAML + extraction allowlist.
        source = pathlib.Path(__file__).resolve().parents[1] / 'src/app/agent_3dvec_sdf.py'
        text = source.read_text()
        region = text.split('                # 0054a: forward projection controls', 1)[1]
        region = region.split('                stream_requested', 1)[0]
        for key in ('sdf_projection_workers_0054',
                    'sdf_projection_cache_0054',
                    'sdf_projection_profile_0054'):
            self.assertIn(repr(key), region)
        self.assertIn('adapt_arg[roi_key] = extraction_config[roi_key]', region)


if __name__ == '__main__':
    unittest.main()
