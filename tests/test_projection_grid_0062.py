"""0062 strict conservative pre-KDTree spatial coarse grid and bounded-query tests.

PYTHONPATH=src:src/ngc python -m unittest discover -s tests -p 'test_projection_grid_0062.py' -v
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
            [-1., 0., .1], [-.6, .4, 0.], [-.25, .55, .3],
            [.3, .3, -.25], [.6, -.3, -.3], [1., -.4, 0.]])
        self.key_cylinder_radius = np.array([
            [.19, .18], [.2, .24], [.16, .2], [.25, .26],
            [.21, .15], [.15, .22]])
        self.key_train_radius = self.key_cylinder_radius * .8
        self.key_frame = np.tile(np.eye(3), (len(self.key_ts), 1, 1))


class TestProjectionGrid0062(unittest.TestCase):
    def setUp(self):
        self.c = TestCurve()
        rng = np.random.default_rng(62062)
        far = rng.uniform(-3, 3, (9000, 3))
        coords = rng.uniform(.15, .85, 5000)
        center = self.c.interpolate(coords, radius=False, frame=False)['points']
        near = center + rng.normal(0, .10, (len(center), 3))
        self.points = np.concatenate([far, near, np.empty((0, 3))])
        self.opts = dict(projection_s0=.15, projection_s1=.85,
                         projection_workers_0054=2, projection_cache_0054=True,
                         projection_refine_fast_0055=True,
                         projection_leafsize_0056=32,
                         sdf_localize_after_gate=True,
                         projection_sphere_0058=True)

    def assertFieldsMatch(self, a, b):
        self.assertEqual(set(a), set(b))
        for k in a:
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)

    def test_localized_fields_and_original_indices_identical(self):
        reference, orig = self.c.localize_samples(self.points, **self.opts)
        for bound in (False, True):
            got, ids = self.c.localize_samples(
                self.points, projection_grid_0062=True,
                projection_bound_query_0062=bound,
                projection_grid_audit_0062=True, **self.opts)
            np.testing.assert_array_equal(orig, ids)
            self.assertFieldsMatch(reference, got)
        self.assertGreater(self.c._projection_grid_considered_0062,
                           self.c._projection_grid_queried_0062)
        self.assertEqual(self.c._projection_grid_audited_rows_0062,
                         len(self.points) * 2)

    def test_accepted_parity_for_different_radius_settings(self):
        for scale, add, norm in ((1., 0., 1.), (.5, -.02, .95),
                                 (-1.2, .3, 1.), (2., .12, 1.)):
            kwargs = dict(norm=norm, runtime_cylinder_radius_scale=scale,
                          runtime_cylinder_radius_add=add)
            a, ia = self.c.localize_samples(self.points, **self.opts, **kwargs)
            b, ib = self.c.localize_samples(
                self.points, projection_grid_0062=True,
                projection_bound_query_0062=True,
                projection_grid_audit_0062=True,
                **self.opts, **kwargs)
            np.testing.assert_array_equal(ia, ib)
            self.assertFieldsMatch(a, b)

    def test_empty_and_all_far(self):
        for samples in (np.zeros((0, 3)),
                        np.array([[100., 0., -70.], [-60., 20., 30.]])):
            a, ia = self.c.localize_samples(samples, **self.opts)
            b, ib = self.c.localize_samples(
                samples, projection_grid_0062=True,
                projection_bound_query_0062=True,
                projection_grid_audit_0062=True, **self.opts)
            np.testing.assert_array_equal(ia, ib)
            self.assertFieldsMatch(a, b)

    def test_cache_reuse_and_mutation_invalidation(self):
        self.c.localize_samples(self.points, projection_grid_0062=True, **self.opts)
        old = self.c._projection_grid_cache_0062
        self.c.localize_samples(self.points, projection_grid_0062=True, **self.opts)
        self.assertIs(old, self.c._projection_grid_cache_0062)
        self.c.key_points[1, 0] += .13
        self.c.localize_samples(self.points, projection_grid_0062=True, **self.opts)
        self.assertIsNot(old, self.c._projection_grid_cache_0062)

    def test_too_many_cells_fallback_is_exact(self):
        verts = self.c._projection_skeleton_interval(.15,.85,200,cache_0054=True,
                                                       leafsize_0056=32)
        _, coords, tree, _, _ = verts
        kept = self.c._projection_grid_filter_0062(self.points, coords, tree, .15,
                                                    cell_ratio=.00001,
                                                    max_cells=100, workers=2)
        self.assertIsNone(kept)
        self.assertGreater(self.c._projection_grid_fallbacks_0062, 0)
        a, ia = self.c.localize_samples(self.points, **self.opts)
        b, ib = self.c.localize_samples(self.points, projection_grid_0062=True,
                              projection_grid_cell_ratio_0062=.00001, **self.opts)
        np.testing.assert_array_equal(ia, ib)
        self.assertFieldsMatch(a, b)

    def test_disabled_or_invalid_modes_fail_closed(self):
        with self.assertRaisesRegex(ValueError, '0062'):
            self.c.localize_samples(self.points, projection_grid_0062=True,
                                    **{**self.opts, 'projection_sphere_0058':False})
        for ratio in (0., -1., 1.1):
            with self.assertRaisesRegex(ValueError, '0062'):
                self.c.localize_samples(self.points,
                      projection_grid_0062=True,
                      projection_grid_cell_ratio_0062=ratio, **self.opts)

    def test_configs_and_forwarding(self):
        from pathlib import Path
        import yaml
        root = Path(__file__).resolve().parents[1]
        for mode in ('audit', 'profile'):
            cfg = yaml.safe_load((root / 'exp/train/manipulation'/
                ('adapt_oktopus_puffer_root_leaves_0062_body_' + mode + '.yaml')).read_text())
            self.assertTrue(cfg['sdf_projection_grid_0062'])
            self.assertFalse(cfg['sdf_projection_bound_query_0062'])
            self.assertEqual(cfg['sdf_projection_grid_audit_0062'], mode=='audit')
            self.assertEqual(cfg['sdf_puffer_stop_after'], 1)
        for mode in ('audit','profile'):
            cfg = yaml.safe_load((root/'exp/train/manipulation'/
                ('adapt_oktopus_puffer_root_leaves_0062_body_bounded_' + mode + '.yaml')).read_text())
            self.assertTrue(cfg['sdf_projection_bound_query_0062'])
            self.assertTrue(cfg['sdf_projection_grid_0062'])
            self.assertEqual(cfg['sdf_projection_grid_audit_0062'], mode == 'audit')
        agent = (root/'src/app/agent_3dvec_sdf.py').read_text()
        self.assertGreaterEqual(agent.count("'sdf_projection_grid_0062'"),3)


if __name__ == '__main__':
    unittest.main()

# This integration test intentionally enters through localize_samples_adapt,
# exercising the 0059 reuse and 0061 vectorized-wrap forwarding layers.
class AdaptationIntegration0062(unittest.TestCase):
    def test_wrap_and_rigid_adaptation_parity(self):
        import contextlib, io
        from unittest import mock
        from tests.test_localize_reuse_0059 import WorkingCurve, wrap_stub
        from ngc.pwla_curve import _localize as localize_module
        rng=np.random.default_rng(620063)
        c=WorkingCurve()
        ts=rng.uniform(.4,.9,2500)
        near=c.interpolate(ts,radius=False,frame=False)['points'] + rng.normal(0,.13,(2500,3))
        pts=np.concatenate([near,rng.uniform(-2,2,(2500,3))])
        for wrap in (False,True):
            kwargs=dict(accessory_curve_handle=type('Accessory',(),{'core':WorkingCurve()})(),
                    src_0=.4,src_1=.9,tgt_0=.1,tgt_1=.5,
                    scale=.72,rot_deg=0.,wrap_radius=wrap,rigid_radius=not wrap,
                    sdf_localize_after_gate=True,sdf_localize_sphere_0058=True,
                    sdf_localize_reuse_0059=True,
                    sdf_projection_workers_0054=2,sdf_projection_cache_0054=True,
                    sdf_projection_refine_fast_0055=True,sdf_projection_leafsize_0056=32)
            def run(grid, bounded):
                curve=WorkingCurve()
                config={**kwargs,'sdf_projection_grid_0062':grid,
                        'sdf_projection_bound_query_0062':bounded,
                        'sdf_projection_grid_audit_0062':grid}
                with mock.patch.object(localize_module,'interpolate_wrap_radius1',wrap_stub):
                    with contextlib.redirect_stdout(io.StringIO()):
                        return curve.localize_samples_adapt(pts,config)
            ref=run(False,False)
            for bound in (False,True):
                got=run(True,bound)
                np.testing.assert_array_equal(ref[2],got[2])
                for x,y in zip(ref[:2],got[:2]):
                    self.assertEqual(set(x),set(y))
                    for k in x:
                        np.testing.assert_array_equal(x[k],y[k],err_msg=k)
