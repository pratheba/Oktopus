import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'app'))
from seam_optimization import (build_seam_plan, radius_scale_at, radius_scale_uv_at,
                               warp_accessory_queries)


class FakeCore:
    def __init__(self, radius):
        self.radius = radius

    def interpolate(self, s):
        s = np.asarray(s)
        return {'radius': self.radius(s)[:, None]}


class FakeCurve:
    def __init__(self, radius):
        self.core = FakeCore(radius)


def item(name, target, accessory, src0, src1, scale, group):
    return {'name': name, 'target_key': target, 'accessory_key': accessory,
            'src_0': src0, 'src_1': src1, 'tgt_0': 0.1, 'tgt_1': 0.9,
            'scale': scale, 'rigid_radius': True, 'mode': 'direct', 'blend_group': group}


class SeamTests(unittest.TestCase):
    def setUp(self):
        self.curves = {'puffer|left': FakeCurve(lambda s: 1.0 + .05 * s),
                       'puffer|right': FakeCurve(lambda s: 1.0 + .03 * s)}
        self.config = {'t1': {'seam_optimization': {'enabled': True, 'radius_source': 'curve'}},
                       't2': {'seam_optimization': {'enabled': True, 'radius_source': 'curve'}}}
        self.items = [item('a', 'avatar|t1', 'puffer|left', .9, .6, 2.0, 't1'),
                      item('b', 'avatar|t1', 'puffer|left', .62, .32, 2.2, 't1'),
                      item('c', 'avatar|t1', 'puffer|left', .34, .1, 2.2, 't1'),
                      item('d', 'avatar|t2', 'puffer|right', .1, .4, 1.5, 't2'),
                      item('e', 'avatar|t2', 'puffer|right', .38, .68, 2.0, 't2')]

    def test_independent_blend_groups_and_two_seams_on_middle_piece(self):
        plan = build_seam_plan(self.items, self.config, self.curves.__getitem__)
        self.assertEqual(set(plan), set(range(5)))
        self.assertEqual(len(plan[1]), 2)
        self.assertAlmostEqual(plan[0][0].source_s, .61)
        self.assertAlmostEqual(plan[1][0].source_s, .33)
        self.assertAlmostEqual(plan[3][0].source_s, .39)
        self.assertLess(abs(plan[0][0].log_radius_scale), math.log(1.2))

    def test_scales_match_the_joint_at_zero_regularization(self):
        plan = build_seam_plan(self.items, self.config, self.curves.__getitem__)
        s = .61
        a, b = self.items[:2]
        def physical(item, correction):
            t = item['tgt_0'] + (s - item['src_0']) / (item['src_1'] - item['src_0']) * (item['tgt_1'] - item['tgt_0'])
            return float(self.curves[item['accessory_key']].core.interpolate([t])['radius'][0, 0]) / item['scale'] * radius_scale_at([s], [correction])[0]
        self.assertAlmostEqual(physical(a, plan[0][0]), physical(b, plan[1][1]), places=10)

    def test_radius_profile_vanishes_smoothly_at_band_edges(self):
        plan = build_seam_plan(self.items, self.config, self.curves.__getitem__)
        corr = plan[0][0]
        s0 = corr.source_s - corr.half_width
        v = radius_scale_at([s0 - 1e-4, s0, s0 + 1e-4], [corr])
        self.assertAlmostEqual(v[0], 1.0)
        self.assertAlmostEqual(v[1], 1.0)
        self.assertLess(abs(v[2] - 1.0), 1e-4)

    def test_warp_preserves_original_and_changes_only_radial_coordinates(self):
        plan = build_seam_plan(self.items, self.config, self.curves.__getitem__)
        original = {'samples_local': np.array([[.5, 1.0, 2.0], [.7, 3.0, 4.0]]),
                    'rho_n': np.array([1.0, 2.0])}
        avatar = {'coords': np.array([.61, .9])}
        before_local = original['samples_local'].copy()
        before_rho = original['rho_n'].copy()
        adjusted = warp_accessory_queries(original, avatar, plan[0])
        self.assertIs(adjusted, original)
        self.assertTrue(np.array_equal(before_local[:, 0], adjusted['samples_local'][:, 0]))
        self.assertAlmostEqual(adjusted['samples_local'][1, 1], 3.0)
        expected_scale = radius_scale_at(avatar['coords'], plan[0])
        np.testing.assert_allclose(adjusted['samples_local'][:, 1:], before_local[:, 1:] / expected_scale[:, None])
        np.testing.assert_allclose(adjusted['rho_n'], before_rho / expected_scale)

    def test_prevent_accidental_cross_target_group(self):
        items = list(self.items)
        items[1] = dict(items[1], target_key='avatar|t2')
        with self.assertRaisesRegex(ValueError, 'mixes avatar targets'):
            build_seam_plan(items, self.config, self.curves.__getitem__)

    def test_disabled_group_does_not_modify_adaptation(self):
        plan = build_seam_plan(self.items, {'t1': {'seam_optimization': {'enabled': False}}}, self.curves.__getitem__)
        self.assertEqual(plan, {})

    def test_position_fitting_aligns_unequal_surface_centers(self):
        class SurfaceCore:
            def __init__(self):
                angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
                grid_s = np.linspace(0.0, 1.0, 200)
                self.surface_points_owned = np.array([
                    (s, (0.12 * s) + np.cos(theta), (-0.08 * s) + np.sin(theta))
                    for s in grid_s for theta in angles], dtype=float)

            def curve_projection(self, points):
                return points[:, 0]

            def interpolate(self, s, **kwargs):
                s = np.asarray(s, dtype=float).reshape(-1)
                return {'radius': np.ones((len(s), 2)),
                        'points': np.column_stack((s, np.zeros_like(s), np.zeros_like(s))),
                        'frame': np.broadcast_to(np.eye(3), (len(s), 3, 3))}

        class SurfaceCurve:
            def __init__(self):
                self.core = SurfaceCore()

        curve = SurfaceCurve()
        pieces = [item('a', 'avatar|t4', 'puffer|right', .1, .4, 1.5, 't4'),
                  item('b', 'avatar|t4', 'puffer|right', .38, .68, 1.8, 't4')]
        pieces[1]['tgt_0'], pieces[1]['tgt_1'] = .55, .95
        spec = {'t4': {'seam_optimization': {'enabled': True,
                                          'optimize_radius': False,
                                          'optimize_position': True,
                                          'max_position_change': .05}}}
        plan = build_seam_plan(pieces, spec, lambda _: curve)
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[0][0].log_radius_scale, 0.0)
        self.assertNotEqual(plan[0][0].position_uv, (0.0, 0.0))
        self.assertLessEqual(np.linalg.norm(plan[0][0].position_uv), .05 + 1e-12)
        self.assertLessEqual(np.linalg.norm(plan[1][0].position_uv), .05 + 1e-12)
        # When fitted positions are applied to world-space center, mismatch shrinks.
        from seam_optimization import _surface_profile, _position_center
        seam_s = .39
        profile = _surface_profile(curve)
        before = [_position_center(it, curve, profile, seam_s, 1., .025)
                  for it in pieces]
        after = [before[i] + np.asarray(plan[i][0].position_uv) / it['scale']
                 for i, it in enumerate(pieces)]
        self.assertLess(np.linalg.norm(after[0] - after[1]),
                        np.linalg.norm(before[0] - before[1]))

    def test_position_warp_local_and_fades_without_touching_axial_queries(self):
        from seam_optimization import SeamCorrection
        seam = SeamCorrection(.5, .1, 0., 0., (.02, -.04))
        queries = {'samples_local': np.array([[.0, 1., 2.], [.0, 1., 2.], [.0, 1., 2.]]),
                   'radius': np.array([[.5, 2.], [.5, 2.], [.5, 2.]]),
                   'rho_n': np.sqrt(np.array([5., 5., 5.])),
                   'angles': np.arctan2(np.array([2., 2., 2.]), np.array([1., 1., 1.]))}
        original = queries['samples_local'].copy()
        result = warp_accessory_queries(queries, {'coords': np.array([.2, .5, .8])}, [seam])
        np.testing.assert_allclose(result['samples_local'][0], original[0])
        np.testing.assert_allclose(result['samples_local'][2], original[2])
        np.testing.assert_allclose(result['samples_local'][1], [.0, .96, 2.02])
        np.testing.assert_allclose(result['rho_n'][1], np.hypot(.96, 2.02))

    def test_rigid_global_scale_is_inverse_in_physical_radius(self):
        curves = {'puffer|x': FakeCurve(lambda s: np.ones_like(s))}
        pieces = [item('a', 'avatar|t', 'puffer|x', .1, .4, 1.5, 'g'),
                  item('b', 'avatar|t', 'puffer|x', .38, .68, 1.8, 'g')]
        spec = {'g': {'seam_optimization': {'enabled': True,
                                           'radius_source': 'curve',
                                           'match_surface_tangent': False}}}
        plan = build_seam_plan(pieces, spec, curves.__getitem__)
        # Before fitting, scale 1.5 produces the larger physical radius (1/1.5).
        # It must shrink, while scale 1.8 must expand.
        self.assertLess(math.exp(plan[0][0].log_radius_scale), 1.0)
        self.assertGreater(math.exp(plan[1][0].log_radius_scale), 1.0)
        seam_s = plan[0][0].source_s
        ra = (1.0 / 1.5) * radius_scale_at([seam_s], plan[0])[0]
        rb = (1.0 / 1.8) * radius_scale_at([seam_s], plan[1])[0]
        self.assertAlmostEqual(ra, rb, places=10)

    def test_position_tangent_is_fitted_and_warped_locally(self):
        class SurfaceCore:
            def __init__(self):
                angles = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
                grid_s = np.linspace(0.0, 1.0, 260)
                self.surface_points_owned = np.array([
                    (s, 0.35 * s + (1.0 + 0.15 * s) * np.cos(theta),
                     -0.18 * s + (1.0 + 0.15 * s) * np.sin(theta))
                    for s in grid_s for theta in angles], dtype=float)

            def curve_projection(self, points):
                return points[:, 0]

            def interpolate(self, s, **kwargs):
                s = np.asarray(s, dtype=float).reshape(-1)
                return {'radius': (1.0 + .15 * s)[:, None],
                        'points': np.column_stack((s, np.zeros_like(s), np.zeros_like(s))),
                        'frame': np.broadcast_to(np.eye(3), (len(s), 3, 3))}

        class SurfaceCurve:
            def __init__(self):
                self.core = SurfaceCore()

        curve = SurfaceCurve()
        pieces = [item('a', 'avatar|t', 'puffer|x', .1, .4, 1.5, 'g'),
                  item('b', 'avatar|t', 'puffer|x', .38, .68, 1.8, 'g')]
        pieces[0]['tgt_0'], pieces[0]['tgt_1'] = .10, .40
        pieces[1]['tgt_0'], pieces[1]['tgt_1'] = .55, .95
        spec = {'g': {'seam_optimization': {'enabled': True,
                                           'optimize_radius': True,
                                           'optimize_position': True,
                                           'radius_source': 'surface',
                                           'match_surface_tangent': True,
                                           'match_position_tangent': True,
                                           'max_position_change': .2}}}
        plan = build_seam_plan(pieces, spec, lambda _: curve)
        self.assertTrue(np.linalg.norm(plan[0][0].position_slope_uv) > 0.0)
        self.assertTrue(np.linalg.norm(plan[1][0].position_slope_uv) > 0.0)
        self.assertNotEqual(plan[0][0].log_radius_slope, 0.0)

        seam_s = plan[0][0].source_s
        query_s = np.array([seam_s - .01, seam_s, seam_s + .01])
        queries = {'samples_local': np.column_stack((np.zeros(3), np.ones(3), np.zeros(3))),
                   'radius': np.ones((3, 2)),
                   'rho_n': np.ones(3)}
        before = queries['samples_local'].copy()
        out = warp_accessory_queries(queries, {'coords': query_s}, plan[0])
        # C1 positional term makes the two sides of the seam differ; a C0-only
        # translation would be symmetric around the center sample.
        self.assertFalse(np.allclose(out['samples_local'][0, 1:],
                                     out['samples_local'][2, 1:]))
        self.assertTrue(np.array_equal(before[:, 0], out['samples_local'][:, 0]))


    def test_position_enabled_requires_surface_evidence(self):
        config = {'t1': {'seam_optimization': {'enabled': True, 'optimize_position': True}}}
        with self.assertRaisesRegex(ValueError, 'surface_points_owned'):
            build_seam_plan(self.items, config, self.curves.__getitem__)


    def test_anisotropic_surface_fit_reduces_elliptical_profile_mismatch(self):
        class EllipseCore:
            def __init__(self):
                angles = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
                grid_s = np.linspace(0.0, 1.0, 260)
                self.surface_points_owned = np.array([
                    (s,
                     (1.0 + 0.8 * s) * np.cos(theta),
                     (1.1 - 0.45 * s) * np.sin(theta))
                    for s in grid_s for theta in angles
                ], dtype=float)

            def curve_projection(self, points):
                return points[:, 0]

            def interpolate(self, s, **kwargs):
                s = np.asarray(s, dtype=float).reshape(-1)
                return {'radius': np.ones((len(s), 2)),
                        'points': np.column_stack((s, np.zeros_like(s), np.zeros_like(s))),
                        'frame': np.broadcast_to(np.eye(3), (len(s), 3, 3))}

        class EllipseCurve:
            def __init__(self):
                self.core = EllipseCore()

        curve = EllipseCurve()
        pieces = [item('a', 'avatar|t', 'puffer|x', .1, .4, 1.0, 'g'),
                  item('b', 'avatar|t', 'puffer|x', .38, .68, 1.0, 'g')]
        pieces[0]['tgt_0'], pieces[0]['tgt_1'] = .10, .40
        pieces[1]['tgt_0'], pieces[1]['tgt_1'] = .58, .88
        spec = {'g': {'seam_optimization': {
            'enabled': True,
            'optimize_radius': True,
            'radius_source': 'surface',
            'radius_mode': 'anisotropic',
            'max_radius_change': .30,
            'max_anisotropic_change': .25,
            'deformation_weight': 0.0,
            'match_surface_tangent': True,
        }}}
        plan = build_seam_plan(pieces, spec, lambda _: curve)
        self.assertIsNotNone(plan[0][0].log_radius_scale_uv)
        self.assertIsNotNone(plan[1][0].log_radius_scale_uv)
        self.assertNotAlmostEqual(plan[0][0].log_radius_scale_uv[0],
                                  plan[0][0].log_radius_scale_uv[1])
        self.assertNotAlmostEqual(plan[0][0].log_radius_slope_uv[0],
                                  plan[0][0].log_radius_slope_uv[1])

        from seam_optimization import _surface_profile, _surface_log_radius_profile
        profile = _surface_profile(curve)
        seam_s = plan[0][0].source_s
        before_l = _surface_log_radius_profile(pieces[0], profile, seam_s, .025, 24, .85)
        before_r = _surface_log_radius_profile(pieces[1], profile, seam_s, .025, 24, .85)
        after_l = _surface_log_radius_profile(
            pieces[0], profile, seam_s, .025, 24, .85, plan[0][0].log_radius_scale_uv)
        after_r = _surface_log_radius_profile(
            pieces[1], profile, seam_s, .025, 24, .85, plan[1][0].log_radius_scale_uv)
        before = np.sqrt(np.mean((before_l - before_r) ** 2))
        after = np.sqrt(np.mean((after_l - after_r) ** 2))
        self.assertLess(after, 0.25 * before)

    def test_anisotropic_query_warp_scales_uv_independently(self):
        from seam_optimization import SeamCorrection
        seam = SeamCorrection(
            .5, .1, 0.0, 0.0,
            log_radius_scale_uv=(math.log(2.0), math.log(0.5)),
            log_radius_slope_uv=(0.0, 0.0),
        )
        queries = {
            'samples_local': np.array([[.0, 1.0, 2.0]]),
            'rho_n': np.array([np.hypot(1.0, 2.0)]),
            'u_n': np.array([1.0]),
            'v_n': np.array([2.0]),
            'angles': np.array([math.atan2(2.0, 1.0)]),
        }
        out = warp_accessory_queries(queries, {'coords': np.array([.5])}, [seam])
        np.testing.assert_allclose(out['samples_local'][0], [0.0, 0.5, 4.0])
        np.testing.assert_allclose(out['rho_n'][0], np.hypot(.5, 4.0))
        np.testing.assert_allclose(out['u_n'][0], .5)
        np.testing.assert_allclose(out['v_n'][0], 4.0)
        np.testing.assert_allclose(out['angles'][0], math.atan2(4.0, .5))
        np.testing.assert_allclose(radius_scale_uv_at([.5], [seam])[0], [2.0, .5])
        self.assertAlmostEqual(radius_scale_at([.5], [seam])[0], 1.0)



    def test_angular_profile_fit_reduces_lobed_surface_mismatch(self):
        class LobedCore:
            def __init__(self):
                angles = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
                grid_s = np.linspace(0.0, 1.0, 320)
                self.surface_points_owned = np.array([
                    (s,
                     (1.0 + .18 * s * np.sin(2.0 * theta)
                      + .14 * s * np.cos(3.0 * theta)) * np.cos(theta),
                     (1.0 + .18 * s * np.sin(2.0 * theta)
                      + .14 * s * np.cos(3.0 * theta)) * np.sin(theta))
                    for s in grid_s for theta in angles
                ], dtype=float)

            def curve_projection(self, points):
                return points[:, 0]

            def interpolate(self, s, **kwargs):
                s = np.asarray(s, dtype=float).reshape(-1)
                return {'radius': np.ones((len(s), 2)),
                        'points': np.column_stack((s, np.zeros_like(s), np.zeros_like(s))),
                        'frame': np.broadcast_to(np.eye(3), (len(s), 3, 3))}

        class LobedCurve:
            def __init__(self):
                self.core = LobedCore()

        curve = LobedCurve()
        pieces = [item('a', 'avatar|t', 'puffer|x', .1, .4, 1.0, 'g'),
                  item('b', 'avatar|t', 'puffer|x', .38, .68, 1.0, 'g')]
        pieces[0]['tgt_0'], pieces[0]['tgt_1'] = .10, .40
        pieces[1]['tgt_0'], pieces[1]['tgt_1'] = .58, .88
        spec = {'g': {'seam_optimization': {
            'enabled': True,
            'optimize_radius': True,
            'radius_source': 'surface',
            'radius_mode': 'angular',
            'max_radius_change': .30,
            'max_anisotropic_change': .20,
            'max_angular_radius_change': .12,
            'angular_profile_order': 3,
            'deformation_weight': 0.0,
            'angular_deformation_weight': 0.0,
            'match_surface_tangent': True,
        }}}
        plan = build_seam_plan(pieces, spec, lambda _: curve)
        self.assertIsNotNone(plan[0][0].angular_log_radius_coeffs)
        self.assertEqual(len(plan[0][0].angular_log_radius_coeffs), 4)  # k=2,3
        self.assertIsNotNone(plan[0][0].angular_log_radius_slope_coeffs)

        from seam_optimization import _surface_profile, _surface_log_radius_profile
        profile = _surface_profile(curve)
        seam_s = plan[0][0].source_s
        before_l = _surface_log_radius_profile(pieces[0], profile, seam_s, .025, 24, .85)
        before_r = _surface_log_radius_profile(pieces[1], profile, seam_s, .025, 24, .85)
        c0, c1 = plan[0][0], plan[1][0]
        after_l = _surface_log_radius_profile(
            pieces[0], profile, seam_s, .025, 24, .85,
            c0.log_radius_scale_uv, c0.angular_log_radius_coeffs)
        after_r = _surface_log_radius_profile(
            pieces[1], profile, seam_s, .025, 24, .85,
            c1.log_radius_scale_uv, c1.angular_log_radius_coeffs)
        before = np.sqrt(np.mean((before_l - before_r) ** 2))
        after = np.sqrt(np.mean((after_l - after_r) ** 2))
        self.assertLess(after, 0.70 * before)

    def test_angular_query_warp_is_directional_and_local(self):
        from seam_optimization import SeamCorrection, angular_radius_scale_at
        log_amp = math.log(1.1)
        seam = SeamCorrection(
            .5, .1, 0.0, 0.0,
            angular_log_radius_coeffs=(log_amp, 0.0, 0.0, 0.0),
            angular_log_radius_slope_coeffs=(0.0, 0.0, 0.0, 0.0),
        )
        coords = np.array([.2, .5, .5])
        queries = {
            'samples_local': np.array([[.0, 1.0, 0.0],
                                       [.0, 1.0, 0.0],
                                       [.0, 0.0, 1.0]]),
            'radius': np.ones((3, 2)),
            'rho_n': np.ones(3),
            'angles': np.array([0.0, 0.0, np.pi / 2.0]),
        }
        out = warp_accessory_queries(queries, {'coords': coords}, [seam])
        # Outside the correction band nothing changes.
        np.testing.assert_allclose(out['samples_local'][0], [0.0, 1.0, 0.0])
        # k=2: theta=0 expands geometry by 1.1, hence inverse query shrinks.
        np.testing.assert_allclose(out['samples_local'][1], [0.0, 1.0 / 1.1, 0.0])
        # theta=pi/2 has cos(2theta)=-1, so geometry scale is 1/1.1.
        np.testing.assert_allclose(out['samples_local'][2], [0.0, 0.0, 1.1])
        factors = angular_radius_scale_at(
            [.4, .5, .6], [0.0, 0.0, 0.0], [seam])
        np.testing.assert_allclose(factors[[0, 2]], [1.0, 1.0], atol=1e-12)
        self.assertAlmostEqual(factors[1], 1.1)


if __name__ == '__main__':
    unittest.main()
