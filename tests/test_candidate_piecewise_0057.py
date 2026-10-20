"""0057: bounded subset, fail-closed exact localization audit, and model-row parity."""
import contextlib
import io
import unittest
from unittest import mock
import numpy as np

from test_sdf_candidate_interval_0048 import StraightHandle
from test_puffer_staged_candidate_roi_0052 import model_rows, synthetic_localize
from ngc.handle_utils.bounded_filter import iter_bbox_kidx
from ngc.handle_utils.interval_envelope import (
    localizer_gate_envelope, localizer_piecewise_gate_envelopes,
    PiecewiseGateEnvelopeCandidates,
)
from utils.mc_utils import MCGrid


class Piecewise0057Tests(unittest.TestCase):
    def setUp(self):
        self.grid = MCGrid({'reso': 32, 'size': 1.2, 'level': 0.0})
        self.handle = StraightHandle()
        self.handle.core.localize_samples_adapt = synthetic_localize
        self.params = {
            'mode': 'direct', 'src_0': .3, 'src_1': .4,
            'use_adapt_control_filter_grid': True,
            'rigid_radius': True, 'wrap_radius': False,
            'dcsdd_bounded_filtering': True, 'dcsdd_filter_batch_size': 31,
            '_sdf_puffer_minimal_staging': True,
            'adapt_debug_counts': False,
            'sdf_candidate_interval_prune': True,
            'sdf_candidate_interval_audit': True,
        }

    def run_filtered(self, cfg):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            accessory, avatar, idx, inside = self.handle.filter_grid_adapt(
                self.grid, cfg)
        self.assertIsNone(inside)
        return ({k: np.asarray(v).copy() for k, v in accessory.items()},
                {k: np.asarray(v).copy() for k, v in avatar.items()},
                np.asarray(idx).copy(), output.getvalue())

    def test_0057_parity_with_full_and_single_box(self):
        full = self.run_filtered({**self.params, 'sdf_candidate_interval_prune': False,
                                  'sdf_candidate_interval_audit': False})
        single = self.run_filtered(self.params)
        piecewise = self.run_filtered({**self.params,
                                       'sdf_candidate_piecewise_0057': True,
                                       'sdf_candidate_piecewise_bins_0057': 8})
        self.assertIn('[sdf_candidate_piecewise_0057 audit]', piecewise[3])
        self.assertIn('missing_accepted= 0', piecewise[3])
        for ref in (single, piecewise):
            np.testing.assert_array_equal(ref[2], full[2])
            for orig, got in zip(full[:2], ref[:2]):
                self.assertEqual(set(orig), set(got))
                for key in orig:
                    np.testing.assert_array_equal(orig[key], got[key], err_msg=key)
        self.assertGreater(len(piecewise[2]), 0)

    def test_world_envelopes_include_tangent_endpoint_support(self):
        cfg = self.params
        base_lo, base_hi = localizer_gate_envelope(
            self.handle.core, .3, .4, cfg, self.grid)
        intervals = localizer_piecewise_gate_envelopes(
            self.handle.core, .3, .4, cfg, self.grid, bins=8)
        self.assertEqual(len(intervals), 8)
        self.assertTrue(any(lo[0] < -.5 for lo, _ in intervals))
        self.assertLessEqual(min(lo[0] for lo, _ in intervals), base_lo[0] + .10)
        self.assertGreaterEqual(max(hi[0] for _, hi in intervals), base_hi[0] - .10)

    def test_multi_box_is_sorted_unique_subset_of_original_voxels(self):
        s = np.linspace(0, 1, 45)
        fields = self.handle.core.build_adapt_control_field(s)
        cylinder = self.handle._CurveHandle__gen_cyl_mesh({
            **fields, 'thetas': 2 * np.pi * np.linspace(0, 1, 16, endpoint=False)})
        all_idx = np.concatenate(list(iter_bbox_kidx(
            cylinder.cylinder_samples, self.grid, batch_size=47)))
        gates = localizer_piecewise_gate_envelopes(
            self.handle.core, .3, .4, self.params, self.grid, bins=10)
        chunks = list(PiecewiseGateEnvelopeCandidates(
            cylinder, gates).iter_filter_grid_indices(self.grid, batch_size=19))
        reduced = np.concatenate(chunks)
        self.assertTrue(np.all(reduced[1:] > reduced[:-1]))
        np.testing.assert_array_equal(reduced, all_idx[np.isin(all_idx, reduced)])
        self.assertLessEqual(len(reduced), len(all_idx))

    def test_fail_closed_when_candidate_rejection_is_unsafe(self):
        self.handle.core.localize_samples_adapt = (
            lambda pts, arg: (model_rows(pts, np.arange(len(pts))),
                              model_rows(pts, np.arange(len(pts))),
                              np.arange(len(pts))))
        with self.assertRaisesRegex(ValueError, 'UNSAFE'):
            self.run_filtered({**self.params,
                               'sdf_candidate_piecewise_0057': True})

    def test_opt_in_requires_exact_audit_and_existing_prune(self):
        for key, changes in (
            ('interval pruning', {'sdf_candidate_interval_prune': False}),
            ('audit', {'sdf_candidate_interval_audit': False}),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, '0057'):
                self.run_filtered({**self.params,
                                   'sdf_candidate_piecewise_0057': True,
                                   **changes})
        with self.assertRaisesRegex(ValueError, '0057 interval bins'):
            self.run_filtered({**self.params,
                               'sdf_candidate_piecewise_0057': True,
                               'sdf_candidate_piecewise_bins_0057': 33})


    def test_profile_opt_in_only_after_explicit_verification(self):
        cfg = {**self.params, 'sdf_candidate_piecewise_0057': True,
               'sdf_candidate_interval_audit': False}
        with self.assertRaisesRegex(ValueError, 'previously verified'):
            self.run_filtered(cfg)
        audited = self.run_filtered({**cfg, 'sdf_candidate_interval_audit': True})
        approved = self.run_filtered({**cfg, 'sdf_candidate_piecewise_verified_0057': True})
        np.testing.assert_array_equal(audited[2], approved[2])
        self.assertIn('verified_without_audit= True', approved[3])

    def test_config_forwarded_through_agent(self):
        from pathlib import Path
        import yaml
        folder = Path(__file__).resolve().parents[1]
        cfg = yaml.safe_load((folder / 'exp/train/manipulation/'
            'adapt_oktopus_puffer_root_leaves_0057_body_audit.yaml').read_text())
        self.assertTrue(cfg['sdf_candidate_piecewise_0057'])
        self.assertTrue(cfg['sdf_candidate_interval_audit'])
        self.assertEqual(cfg['sdf_candidate_piecewise_bins_0057'], 12)
        self.assertEqual(cfg['sdf_puffer_stop_after'], 1)
        agent = (folder / 'src/app/agent_3dvec_sdf.py').read_text()
        self.assertGreaterEqual(agent.count("'sdf_candidate_piecewise_0057'"), 3)


if __name__ == '__main__':
    unittest.main()
