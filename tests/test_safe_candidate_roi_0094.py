"""0094: deterministic rigid/wrap candidate pruning without changing SDF inputs."""
import contextlib
import io
import unittest
import numpy as np

from ngc.curve_handle import CurveHandle
from PWLA_curve_handle import PWLACurve
from ngc.handle_utils.safe_candidate_roi_0094 import certified_source_boxes
from utils.mc_utils import MCGrid


class SafeCandidate0094Tests(unittest.TestCase):
    def setUp(self):
        self.grid = MCGrid({'reso': 32, 'size': 1.2, 'level': 0.0})
        self.core = PWLACurve.__new__(PWLACurve)
        self.core.key_ts = np.linspace(0, 1, 9)
        s = self.core.key_ts
        self.core.key_points = np.column_stack((1.6*s-.8, np.zeros(len(s)), np.zeros(len(s))))
        self.core.key_cylinder_radius = np.tile([.19, .14], (len(s), 1))
        self.core.key_frame = np.tile(np.eye(3), (len(s), 1, 1))
        self.handle = CurveHandle.__new__(CurveHandle)
        self.handle.core = self.core
        profile = self.core.interpolate(np.linspace(0, 1, 200), radius_type='cylinder')
        profile['thetas'] = 2*np.pi*np.linspace(0, 1, 120, endpoint=False)
        self.handle.cyl_mesh = self.handle._CurveHandle__gen_cyl_mesh(profile)
        self.params = dict(mode='direct', src_0=.0, src_1=.10,
                           rigid_radius=True, wrap_radius=False,
                           dcsdd_bounded_filtering=True, dcsdd_filter_batch_size=97,
                           adapt_debug_counts=False, sdf_localize_after_gate=True)

        def localize(pts, arg):
            # Exact cylinder ellipsoid gate for this straight PWLA. Constrain
            # source projection and crop identically to the real direct path.
            a, b = sorted((arg['src_0'], arg['src_1']))
            t = np.clip((pts[:, 0]+.8)/1.6, a, b)
            center_x = 1.6*t-.8
            xyz = np.column_stack((pts[:, 0]-center_x,
                                   pts[:, 1], pts[:, 2]))
            radius = np.array([1, .19, .14])
            keep = np.flatnonzero(np.linalg.norm(xyz/radius, axis=1) <= 1)
            # Return stable model inputs to test exact row/index parity.
            payload = dict(coords=t[keep], samples_local=xyz[keep])
            return payload, payload, keep
        self.core.localize_samples_adapt = localize

    def run_mode(self, mode, optimize):
        outputs = []
        def callback(acc, idx, *extra):
            outputs.append((idx.copy(), {k:v.copy() for k, v in acc.items()}))
        cfg = {**self.params, 'rigid_radius':mode=='rigid',
               'wrap_radius':mode=='wrap',
               'sdf_stream_combined_wrap':mode=='wrap',
               'sdf_candidate_safe_prune_0094':optimize,
               '_dcsdd_stream_callback':callback}
        with contextlib.redirect_stdout(io.StringIO()) as out:
            count, n, _ = self.handle.filter_grid_adapt(self.grid, cfg)
        indices = np.concatenate([i for i, _ in outputs]) if outputs else np.empty(0, dtype=int)
        fields = {name:np.concatenate([f[name] for _, f in outputs])
                  for name in ('coords', 'samples_local')} if outputs else {}
        return count, n, indices, fields, out.getvalue()

    def test_rigid_and_wrap_identical_indices_and_model_inputs(self):
        for mode in ('rigid', 'wrap'):
            with self.subTest(mode=mode):
                baseline = self.run_mode(mode, False)
                optimized = self.run_mode(mode, True)
                self.assertLess(optimized[0], baseline[0])
                self.assertEqual(optimized[1], baseline[1])
                np.testing.assert_array_equal(optimized[2], baseline[2])
                for k in baseline[3]:
                    np.testing.assert_array_equal(optimized[3][k], baseline[3][k])
                self.assertIn('[sdf_candidate_safe_0094] enabled=True', optimized[4])

    def test_exact_audit_passes_and_detects_excluded_accepted(self):
        good = {**self.params, 'sdf_candidate_safe_prune_0094':True,
                'sdf_candidate_safe_audit_0094':True,
                '_dcsdd_stream_callback':lambda *_:None}
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.handle.filter_grid_adapt(self.grid, good)
        self.assertIn('missing_accepted= 0',out.getvalue())
        # Deliberately broken localizer: ALL broad phase indices are accepted.
        # The exact audit must stop before pruning any useful ones.
        orig = self.core.localize_samples_adapt
        try:
            self.core.localize_samples_adapt = lambda pts,arg: (
                {}, {}, np.arange(len(pts)))
            with self.assertRaisesRegex(ValueError,'UNSAFE'):
                self.handle.filter_grid_adapt(self.grid, good)
        finally:
            self.core.localize_samples_adapt = orig

    def test_curved_keyframes_fallback_sphere_still_contains_gate(self):
        # Even a strong turn cannot cause an unsafe fast frame bound.
        from scipy.spatial.transform import Rotation
        frames = np.array([Rotation.from_euler('z', k, degrees=True).as_matrix()
                           for k in np.linspace(0, 140, len(self.core.key_ts))])
        self.core.key_frame = frames
        boxes = certified_source_boxes(self.core, .25, .65, self.params, self.grid)
        self.assertTrue(boxes)
        for i in range(5, 6):
            a = self.core.key_ts[i]
            for u in np.linspace(0, 1, 20):
                s = a + u*(self.core.key_ts[i+1]-a)
                interp = self.core.interpolate(np.array([s]), radius_type='cylinder')
                center = interp['points'][0]
                frame = interp['frame'][0]
                r = np.r_[1., interp['radius'][0]]
                for theta in np.linspace(0, 2*np.pi, 17):
                    local = np.array([0.75*r[0], 0.5*r[1]*np.cos(theta),
                                      0.5*r[2]*np.sin(theta)])
                    world = center + np.linalg.solve(frame, local)
                    self.assertTrue(any(np.all(world>=lo) and np.all(world<=hi)
                                        for lo,hi in boxes))

    def test_uncertified_custom_core_preserves_original(self):
        class Custom:
            pass
        self.assertIsNone(certified_source_boxes(Custom(), .0, .1, self.params, self.grid))
        self.core.key_cylinder_radius[0, 0] = np.nan
        self.assertIsNone(certified_source_boxes(self.core, .0, .1, self.params, self.grid))

    def test_conflicting_gates_or_unbounded_configuration_fails(self):
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            self.handle.filter_grid_adapt(self.grid, {
                **self.params, 'sdf_candidate_safe_prune_0094':True,
                'sdf_candidate_interval_prune':True})
        with self.assertRaisesRegex(ValueError, 'requires bounded direct'):
            self.handle.filter_grid_adapt(self.grid, {
                **self.params, 'sdf_candidate_safe_prune_0094':True,
                'dcsdd_bounded_filtering':False})

    def test_config_is_wired_for_standard_and_streamed_0093(self):
        from pathlib import Path
        root=Path(__file__).resolve().parents[1]
        agent=(root/'src/app/agent_3dvec_sdf.py').read_text()
        self.assertGreaterEqual(agent.count("'sdf_candidate_safe_prune_0094'"), 3)
        cfg=(root/'exp/train/manipulation/oct22_adapt_oktopus_cone_wrap_0093_sdf_optimization.yaml').read_text()
        self.assertIn('sdf_candidate_safe_prune_0094: true',cfg)


if __name__ == '__main__':
    unittest.main()
