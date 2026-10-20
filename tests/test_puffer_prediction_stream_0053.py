"""0053 direct prediction callback: writer parity and whole-section clamp parity."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from app.sdf_puffer_prediction_stream_0053 import (
    PredictionStream0053, needs_avatar_inference, validate_prediction_stream)
from test_puffer_staged_candidate_roi_0052 import StraightHandle, synthetic_localize
from utils.mc_utils import MCGrid


class FakeAgent:
    def __init__(self):
        self.calls = []

    def _inference_full_vals(self, data, name, batch_size=None):
        self.calls.append((name, len(data['coords'])))
        sl = np.asarray(data['samples_local'])
        coords = np.asarray(data['coords'])
        # Mimic a deterministic per-row inference_full, preserving float64.
        sdf = np.asarray(coords * .2 + sl[:, 1] * .12 + sl[:, 2] * .03)
        return {'dist': sdf, 'dist_base': sdf + .014,
                'dist_detail': np.full(len(coords), -.004, np.float64)}


class DirectPredictionTests(unittest.TestCase):
    def setUp(self):
        self.grid = MCGrid({'reso':24, 'size':1.2, 'level':0.0})
        self.curve = StraightHandle()
        self.curve.core.localize_samples_adapt = synthetic_localize
        self.cfg = dict(mode='direct', src_0=.3, src_1=.4,
                        use_adapt_control_filter_grid=True,
                        rigid_radius=True, wrap_radius=False,
                        dcsdd_bounded_filtering=True,
                        dcsdd_filter_batch_size=19,
                        _sdf_puffer_minimal_staging=True,
                        adapt_debug_counts=False,
                        cut_avatar=True, use_accessory_support_clamp=True)

    def _run_stream(self, cfg):
        with tempfile.TemporaryDirectory() as tmp:
            agent = FakeAgent()
            writer = PredictionStream0053(agent, 'puffer', 'avatar', cfg,
                                          None, tmp, 19)
            streaming = dict(cfg, _sdf_puffer_direct_predictions_0053=True,
                             _dcsdd_stream_callback=writer.push)
            with contextlib.redirect_stdout(io.StringIO()):
                count, accepted, _ = self.curve.filter_grid_adapt(self.grid, streaming)
                result = writer.finish()
            result = {k: np.asarray(v).copy() for k,v in result.items()}
            self.assertEqual(len(result['index']), accepted)
            self.assertGreater(count, accepted)
            self.assertGreater(accepted, 0)
            self.assertTrue(agent.calls)
            self.assertLessEqual(max(c[1] for c in agent.calls), 19)
            writer.cleanup()
            return result

    def test_rigid_vs_staged_and_global_calibration_inputs(self):
        with contextlib.redirect_stdout(io.StringIO()):
            acc, av, idx, _ = self.curve.filter_grid_adapt(self.grid, self.cfg)
        data = self._run_stream(self.cfg)
        np.testing.assert_array_equal(idx, data['index'])
        np.testing.assert_array_equal(acc['coords'], data['acc_coords'])
        np.testing.assert_array_equal(acc['samples_local'], data['acc_local'])
        np.testing.assert_array_equal(av['coords'], data['av_coords'])
        pred = FakeAgent()._inference_full_vals(acc,'puffer')['dist']
        np.testing.assert_array_equal(pred, data['acc_dist'])
        # The support quantile requires all accepted rows, not per-chunk quantiles.
        from app.agent_3dvec_base import AgentBase
        a = AgentBase.__new__(AgentBase)
        expected = a.estimate_rho_limit_from_pred(pred, acc)
        replay = a.estimate_rho_limit_from_pred(data['acc_dist'],
                    dict(samples_local=data['acc_local'], coords=data['acc_coords']))
        for x, y in zip(expected, replay):
            np.testing.assert_array_equal(x, y)

    def test_whole_section_support_clamp_and_avatar_boolean_parity(self):
        with contextlib.redirect_stdout(io.StringIO()):
            acc, av, idx, _ = self.curve.filter_grid_adapt(self.grid, self.cfg)
        data = self._run_stream(self.cfg)
        from app.agent_3dvec_base import AgentBase
        agent = AgentBase.__new__(AgentBase)
        model = FakeAgent()
        predicted = model._inference_full_vals(acc, 'puffer')
        avatar = model._inference_full_vals(av, 'avatar')['dist']
        baseline, baseline_mask = agent.clamp_pred_sdf_by_support(
            predicted['dist'], acc, rho_limit=1.0, verbose=False)
        replay, replay_mask = agent.clamp_pred_sdf_by_support(
            data['acc_dist'],
            {'coords': data['acc_coords'], 'samples_local': data['acc_local']},
            rho_limit=1.0, verbose=False)
        np.testing.assert_array_equal(baseline, replay)
        np.testing.assert_array_equal(baseline_mask, replay_mask)
        np.testing.assert_array_equal(avatar, data['av_dist'])
        # Same model output, ownership and avatar boolean inputs => same
        # reconstructed base cut. The final detail formula is unchanged.
        base_full, _ = agent.clamp_pred_sdf_by_support(
            predicted['dist_base'], acc, rho_limit=1.0)
        base_record, _ = agent.clamp_pred_sdf_by_support(
            data['acc_dist_base'],
            {'coords': data['acc_coords'], 'samples_local': data['acc_local']},
            rho_limit=1.0)
        np.testing.assert_array_equal(np.maximum(base_full, .0001 - avatar),
                                      np.maximum(base_record, .0001-data['av_dist']))

    def test_wrap_and_pruning_callback_parity(self):
        cfg = dict(self.cfg, rigid_radius=False, wrap_radius=True,
                   _sdf_puffer_bounded_wrap_staging=True)
        data = self._run_stream(cfg)
        with contextlib.redirect_stdout(io.StringIO()):
            acc, av, idx, _ = self.curve.filter_grid_adapt(self.grid, cfg)
        np.testing.assert_array_equal(data['index'], idx)
        np.testing.assert_array_equal(data['acc_local'], acc['samples_local'])
        np.testing.assert_array_equal(data['av_coords'], av['coords'])

    def test_model_and_avatar_fields_are_not_staged(self):
        result = self._run_stream(self.cfg)
        self.assertEqual(set(result), {'index', 'acc_coords', 'acc_local',
                                     'av_coords','av_angles','acc_dist',
                                     'acc_dist_base','acc_dist_detail','av_dist'})

    def test_rejects_unsupported_full_field_corrections(self):
        for key, value in [('auto_avatar_snug_field', True),
                           ('use_tiled_detail', True),
                           ('carve_k_bulge', .5)]:
            cfg=dict(self.cfg, **{key:value})
            if key=='carve_k_bulge':cfg['final_carve']=True
            with self.assertRaisesRegex(ValueError, 'unsupported options'):
                validate_prediction_stream(cfg)

    def test_no_avatar_prediction_without_cut(self):
        cfg=dict(self.cfg, cut_avatar=False)
        self.assertFalse(needs_avatar_inference(cfg))
        data=self._run_stream(cfg)
        self.assertNotIn('av_dist', data)
        self.assertIn('av_coords', data)

    def test_source_contains_config_forwarding(self):
        text=(Path(__file__).resolve().parents[1]/'src/app/agent_3dvec_sdf.py').read_text()
        self.assertIn("'sdf_puffer_direct_predictions_0053',", text)
        self.assertIn("adapt_arg['_dcsdd_stream_callback'] = prediction_stream_0053.push",text)
        self.assertIn('prediction_stream_0053.cleanup()',text)

if __name__=='__main__':
    unittest.main()
