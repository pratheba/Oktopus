"""0040: automatic count-free streaming, ordered appends and cropped-SDF parity."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import trimesh

from app.dcsdd_stream_one_pass import OnePassBootStreamWriter
from app.dcsdd_stream_boot import INFERENCE_KEYS
from app.agent_3dvec_sdf_dc import AgentSDFDC
from ngc.handle_utils.mesh_utils import CylindersMesh
from test_dcsdd_bounded_filtering import FakeCurve
from test_dcsdd_compact_2048 import CompactCore
from utils.mc_utils import MCGrid


class AppendWriterTests(unittest.TestCase):
    def test_unknown_count_irregular_chunks_preserve_alignment_and_calibration(self):
        rng = np.random.default_rng(4001)
        n = 157
        coords = rng.uniform(.1, .9, size=n)
        sl = rng.normal(0, .2, (n, 3))
        data = {
            'samples_local': sl,
            'coords': coords,
            'angles': rng.normal(size=n),
            'radius': rng.uniform(.1, 1., size=(n, 2)),
            'rho': rng.uniform(.1, 1., size=n),
            'rho_n': rng.uniform(.1, 1., size=n),
        }
        indices = np.arange(200, 200+n, dtype=np.int64) * 3
        calls = []
        def model(batch):
            calls.append(len(batch['coords']))
            v = np.asarray(batch['coords'], dtype=np.float32)
            return {'dist': v - .4, 'dist_base': v * .5 - .4,
                    'dist_detail': v * .2}
        with tempfile.TemporaryDirectory() as d:
            w = OnePassBootStreamWriter(batch_size=19, outdir=d, inference=model)
            start = 0
            for take in (3, 17, 2, 34, 44, 1, 56):
                end = min(n, start+take)
                w.push({k: v[start:end] for k, v in data.items()},
                       indices[start:end])
                start = end
            self.assertEqual(start,n)
            profiles = w.finish()
            w.replay.finish()
            self.assertEqual(calls, [19]*8 + [5])
            self.assertEqual(w.accepted, n)
            self.assertEqual(w.written, n)
            self.assertEqual(set(profiles), {'dist', 'dist_base'})
            expected = model(data)
            for k in expected:
                np.testing.assert_array_equal(w.raw[k], expected[k])
            records = np.concatenate([part for _, _, part in w.replay.iter_chunks(12)])
            np.testing.assert_array_equal(records['index'], indices)
            np.testing.assert_array_equal(records['coord'], coords)
            np.testing.assert_array_equal(records['rho'], np.sqrt(
                sl[:,1]*sl[:,1] + sl[:,2]*sl[:,2]))
            self.assertEqual(w.replay.bytes, 24 * n)
            self.assertEqual(Path(d,'replay_metadata.bin').stat().st_size, 24*n)
            with self.assertRaises(ValueError):
                w.push(data, indices)

    def test_global_support_calibration_is_independent_of_stream_chunks(self):
        from app.dcsdd_stream_boot import GlobalRhoCalibration
        rng = np.random.default_rng(4002)
        n = 1039
        coords = rng.uniform(0., 1., size=n)
        local = rng.normal(0., .15, (n, 3))
        pred = rng.uniform(-.02, .02, size=n)
        ref = GlobalRhoCalibration()
        ref.add(pred, local, coords)
        batched = GlobalRhoCalibration()
        for start in range(0, n, 17):
            stop = min(n, start + 17)
            batched.add(pred[start:stop], local[start:stop], coords[start:stop])
        a = ref.finish()
        b = batched.finish()
        for x,y in zip(a,b):
            np.testing.assert_array_equal(x,y)
        self.assertEqual(ref.count, batched.count)
        self.assertGreater(ref.count, 0)

    def test_empty_stream_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            w = OnePassBootStreamWriter(batch_size=17, outdir=d,
                                        inference=lambda x: None)
            with self.assertRaises(ValueError):
                w.finish()


class OnePassIntegrationTests(unittest.TestCase):
    def test_output_matches_counted_baseline_for_pointwise_model(self):
        rng = np.random.default_rng(4003)
        with tempfile.TemporaryDirectory() as directory:
            grid = MCGrid({'reso': 32, 'size': 1.2, 'level': 0.,
                           'memmap_dir': directory, 'geometry_only': True})
            cylinder = CylindersMesh()
            cylinder.cylinder_samples = rng.normal(0, .18, (12, 12, 3))
            curve = FakeCurve(cylinder)
            curve.core = CompactCore()
            agent = AgentSDFDC()
            agent.device = torch.device('cpu')
            agent.feat_dict = {'boots': 1}

            class Model:
                class detail_model:
                    sigma = .047
                def __init__(self):
                    self.calls = []
                def inference_full(self, data, transform=None):
                    self.calls.append(len(data['coords']))
                    coords = torch.as_tensor(data['coords'], dtype=torch.float32)
                    sl = torch.as_tensor(data['samples_local'], dtype=torch.float32)
                    return {
                        'dist': (coords * .3 + sl[:,0] * .1 - 2.1)[:,None],
                        'dist_base': (sl[:,1] * .2 + sl[:,0] * .1 - 2.4)[:,None],
                        'dist_detail': (sl[:,2] * .025)[:,None],
                        'gate_detail': None,
                    }
            model = Model()
            agent.model = model
            adapt = {
                'mode': 'direct', 'rigid_radius': True,
                'src_0': .1, 'src_1': .9,
                'use_adapt_control_filter_grid': True,
                'dcsdd_fast_bbox_marking': True,
                'dcsdd_bounded_filtering': True,
                'dcsdd_filter_batch_size': 11,
                'use_accessory_support_clamp': True,
                'support_rho_limit': 30., 'support_positive_value': 1.,
                'adapt_debug_counts': False,
            }
            cfg = {'dcsdd_stream_crop_first': True,
                   'dcsdd_stream_replay_metadata': True,
                   'dcsdd_crop_grid': True, 'surface_extraction': 'ours'}
            outputs = {}
            what = ['baseline']
            def save_grid(input_grid, config, **kwargs):
                s, u, empty, side = agent._grid_to_S_U(
                    input_grid, config={'dcsdd_crop_grid': False})
                outputs[what[0]] = (np.array(s).copy(), np.array(u).copy(),
                                    np.array(empty).copy(), side)
                return trimesh.Trimesh(vertices=np.empty((0, 3)),
                                       faces=np.empty((0, 3), dtype=int))
            agent.extract_surface_mesh = save_grid
            calls = []
            wrapped = curve.filter_grid_adapt
            def track(*args, **kwargs):
                result = wrapped(*args, **kwargs)
                calls.append(result[1])
                return result
            curve.filter_grid_adapt = track
            agent._stream_individual_rigid_boot(curve, grid, adapt, 'boots',
                                                cfg, 0, 'direct', 19, directory)
            self.assertEqual(len(calls), 2)
            baseline_calls = model.calls.copy()
            calls.clear()
            model.calls.clear()
            what[0] = 'onepass'
            agent._stream_individual_rigid_boot(
                curve, grid, adapt, 'boots',
                {**cfg, 'dcsdd_stream_one_pass': True},
                0, 'direct', 19, directory)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0], sum(model.calls))
            self.assertEqual(sum(model.calls), sum(baseline_calls))
            self.assertTrue(all(n == 19 for n in model.calls[:-1]))
            a, b = outputs['baseline'], outputs['onepass']
            self.assertEqual(a[3], b[3])
            for old,new in zip(a[:3], b[:3]):
                np.testing.assert_array_equal(old, new)

    def test_one_pass_rejects_count_hint(self):
        with tempfile.TemporaryDirectory() as d:
            agent = AgentSDFDC()
            grid = MCGrid({'reso':4,'size':1.2,'level':0.,'memmap_dir':d, 'geometry_only':True})
            with self.assertRaisesRegex(ValueError,'does not accept'):
                agent._stream_individual_rigid_boot(
                    None,grid,{},'boot',
                    {'dcsdd_stream_one_pass':True,
                     'dcsdd_stream_replay_metadata':True,
                     'dcsdd_stream_accepted_count':1},
                    0,'direct',19,d)

if __name__ == '__main__':
    unittest.main()
