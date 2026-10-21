"""0060 timing-only instrumentation, preserving the 0059 accepted model rows."""
import contextlib
import io
from pathlib import Path
import unittest
from unittest import mock
import numpy as np
import yaml

from test_localize_reuse_0059 import WorkingCurve, ExactReuse0059, wrap_stub
from ngc.pwla_curve import _localize as localize_module


class LocalizationProfile0060Tests(unittest.TestCase):
    def setUp(self):
        fixture = ExactReuse0059()
        fixture.setUp()
        self.samples = fixture.samples
        self.options = fixture.options

    def _run(self, wrap, profile):
        core = WorkingCurve()
        accessory = WorkingCurve()
        arg = dict(
            accessory_curve_handle=type('Accessory', (), {'core': accessory})(),
            src_0=.4, src_1=.9, tgt_0=.1, tgt_1=.5,
            scale=.72, rot_deg=0., wrap_radius=wrap,
            rigid_radius=not wrap, adapt_debug_counts=False,
            avatar_cylinder_radius_scale=.8,
            avatar_cylinder_radius_add=.05,
            adapt_localize_norm=1.2,
            wrap_angular_follow=0.,
            sdf_localize_after_gate=True,
            sdf_localize_sphere_0058=True,
            sdf_localize_reuse_0059=True,
            sdf_localize_profile_0060=profile,
            sdf_projection_workers_0054=2,
            sdf_projection_cache_0054=True,
            sdf_projection_refine_fast_0055=True,
            sdf_projection_leafsize_0056=32,
        )
        with mock.patch.object(localize_module, 'interpolate_wrap_radius1', wrap_stub):
            with contextlib.redirect_stdout(io.StringIO()):
                result = core.localize_samples_adapt(self.samples, arg)
        return result, core

    def test_exclusive_timers_do_not_change_values_or_indices(self):
        for wrap in (False, True):
            with self.subTest(wrap=wrap):
                baseline, old_core = self._run(wrap, False)
                profiled, new_core = self._run(wrap, True)
                self.assertFalse(hasattr(old_core, '_localize_stages_0060'))
                np.testing.assert_array_equal(baseline[2], profiled[2])
                for label, original, actual in zip(('accessory', 'avatar'),
                                                   baseline[:2], profiled[:2]):
                    self.assertEqual(set(original), set(actual), label)
                    for key in original:
                        np.testing.assert_array_equal(original[key], actual[key],
                                                      err_msg=label + '.' + key)
                table = new_core._localize_stages_0060
                for stage in (
                        'core.projection', 'core.projection_select',
                        'core.frame_interpolation', 'core.world_to_local',
                        'core.cylinder_radius_interpolation',
                        'core.cylinder_gate', 'core.train_radius_interpolation',
                        'core.output_gather', 'adapt.avatar_localize',
                        'adapt.source_interval_crop', 'adapt.avatar_fields',
                        'adapt.arclength_mapping', 'adapt.wrap_table_preparation',
                        'adapt.accessory_interpolation', 'adapt.radial_mapping',
                        'adapt.normalize_model_coordinates',
                        'adapt.pack_output_fields'):
                    self.assertIn(stage, table)
                    self.assertEqual(table[stage][1], 1)
                    self.assertGreaterEqual(table[stage][0], 0.)
                self.assertGreaterEqual(table['adapt.avatar_localize'][0],
                                        sum(v[0] for k, v in table.items()
                                            if k.startswith('core.')) * .95)

    def test_profiler_empty_input_keeps_exact_output(self):
        core = WorkingCurve()
        options = dict(self.options, projection_reuse_0059=True)
        a, aid = core.localize_samples(np.empty((0, 3)), **options)
        b, bid = core.localize_samples(np.empty((0, 3)),
                                       localize_profile_0060=True, **options)
        np.testing.assert_array_equal(aid, bid)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        self.assertIn('core.output_gather', core._localize_stages_0060)

    def test_bounded_stream_report_without_changing_selected_indices(self):
        from test_sdf_candidate_interval_0048 import StraightHandle
        from utils.mc_utils import MCGrid
        grid = MCGrid({'reso': 24, 'size': 1.2, 'level': 0.0})
        handle = StraightHandle()
        cfg = dict(mode='direct', src_0=.3, src_1=.4,
                   use_adapt_control_filter_grid=True,
                   rigid_radius=True, wrap_radius=False,
                   dcsdd_bounded_filtering=True, dcsdd_filter_batch_size=19,
                   sdf_localize_after_gate=True, adapt_debug_counts=False)
        def collect(enable):
            selected = []
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                count, accepted, _ = handle.filter_grid_adapt(grid, {
                    **cfg, 'sdf_localize_profile_0060': enable,
                    '_dcsdd_stream_callback': lambda acc, idx: selected.extend(idx.tolist())})
            return count, accepted, np.asarray(selected), output.getvalue()
        old_count, old_accepted, old_idx, old_log = collect(False)
        count, accepted, idx, log = collect(True)
        self.assertEqual((count, accepted), (old_count, old_accepted))
        np.testing.assert_array_equal(idx, old_idx)
        self.assertNotIn('[sdf_profile_0060]', old_log)
        self.assertIn('[sdf_profile_0060] scope=stream', log)
        self.assertIn('candidate_iterator_s=', log)
        self.assertIn('batch_p95_s=', log)

    def test_profile_yaml_and_forwarding(self):
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load((root / 'exp/train/manipulation/' /
            'adapt_oktopus_puffer_root_leaves_0060_body_profile.yaml').read_text())
        self.assertTrue(config['sdf_localize_profile_0060'])
        self.assertTrue(config['sdf_localize_reuse_0059'])
        self.assertTrue(config['sdf_localize_sphere_0058'])
        self.assertFalse(config['sdf_localize_sphere_audit_0058'])
        self.assertFalse(config['sdf_localize_reuse_audit_0059'])
        self.assertEqual(config['sdf_puffer_stop_after'], 1)
        agent = (root / 'src/app/agent_3dvec_sdf.py').read_text()
        self.assertEqual(agent.count("'sdf_localize_profile_0060'"), 3)
        handle = (root / 'src/ngc/curve_handle.py').read_text()
        self.assertIn('[sdf_profile_0060_stage]', handle)
        self.assertIn('batch_p95_s=', handle)


if __name__ == '__main__':
    unittest.main()
