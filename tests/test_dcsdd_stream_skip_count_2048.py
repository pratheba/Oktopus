"""0039: accepted-count hint skips localization count without changing SDF.

Uses the same independent rigid-boot synthetic fixture as 0038.  Inference
partitioning remains np.array_split(total, total//batch + 1).
"""
import tempfile
import unittest

import numpy as np
import torch
import trimesh

from app.agent_3dvec_sdf_dc import AgentSDFDC
from app.dcsdd_stream_boot import inference_partition_sizes
from ngc.handle_utils.mesh_utils import CylindersMesh
from test_dcsdd_bounded_filtering import FakeCurve
from test_dcsdd_compact_2048 import CompactCore
from utils.mc_utils import MCGrid


class TestSkipCounting(unittest.TestCase):
    @staticmethod
    def fixture(directory, resolution=29):
        rng = np.random.default_rng(3901)
        grid = MCGrid({'reso':resolution, 'size':1.2, 'level':0.,
                       'memmap_dir':directory, 'geometry_only':True})
        cylinder = CylindersMesh()
        cylinder.cylinder_samples = rng.normal(0., .18, (12, 12, 3))
        curve = FakeCurve(cylinder)
        curve.core = CompactCore()
        agent = AgentSDFDC()
        agent.device = torch.device('cpu')
        agent.feat_dict = {'boots':1}

        class Model:
            class detail_model:
                sigma = .047
            def __init__(self):
                self.calls = []
            def inference_full(self, data, transform=None):
                self.calls.append(len(data['coords']))
                coords = torch.as_tensor(data['coords'], dtype=torch.float32)
                local = torch.as_tensor(data['samples_local'], dtype=torch.float32)
                return {
                    'dist': (coords * .3 + local[:,0] * .1 - 2.1)[:,None],
                    'dist_base': (local[:,1] * .2 + local[:,0] * .1 - 2.4)[:,None],
                    'dist_detail': (local[:,2] * .025)[:,None],
                    'gate_detail': None,
                }
        model = Model()
        agent.model = model
        adapt = {
            'mode':'direct', 'rigid_radius':True, 'src_0':.1, 'src_1':.9,
            'use_adapt_control_filter_grid':True,
            'dcsdd_fast_bbox_marking':True, 'dcsdd_bounded_filtering':True,
            'dcsdd_filter_batch_size':11, 'use_accessory_support_clamp':True,
            'support_rho_limit':30., 'support_positive_value':1.,
            'adapt_debug_counts':False,
        }
        config = {
            'dcsdd_stream_crop_first':True,
            'dcsdd_stream_replay_metadata':True,
            'dcsdd_crop_grid':True,
            'surface_extraction':'ours',
        }
        return grid, curve, agent, model, adapt, config

    def test_explicit_hint_skips_pass_preserves_partitions_and_exact_cropped_sdf(self):
        with tempfile.TemporaryDirectory() as temp:
            grid, curve, agent, model, adapt, config = self.fixture(temp)
            captured = {}
            def capture(input_grid, cfg, **kwargs):
                s, u, empty, side = agent._grid_to_S_U(
                    input_grid, config={'dcsdd_crop_grid':False})
                captured[mode[0]] = (np.array(s).copy(), np.array(u).copy(),
                                     np.array(empty).copy(), side)
                return trimesh.Trimesh(vertices=np.empty((0,3)),
                                       faces=np.empty((0,3), dtype=int))
            agent.extract_surface_mesh = capture
            call_counts = []
            original = curve.filter_grid_adapt
            def counted(grid_, options):
                result = original(grid_, options)
                call_counts.append(result[1])
                return result
            curve.filter_grid_adapt = counted
            mode = ['legacy']
            agent._stream_individual_rigid_boot(
                curve, grid, adapt, 'boots', config, 0, 'direct', 19, temp)
            self.assertEqual(len(call_counts), 2)
            actual = call_counts[-1]
            self.assertGreater(actual, 0)
            old_model_calls = list(model.calls)
            self.assertEqual(tuple(old_model_calls), inference_partition_sizes(actual, 19))
            call_counts.clear()
            model.calls.clear()
            mode[0] = 'hinted'
            agent._stream_individual_rigid_boot(
                curve, grid, adapt, 'boots',
                {**config, 'dcsdd_stream_accepted_count':actual},
                0, 'direct', 19, temp)
            self.assertEqual(call_counts, [actual])
            self.assertEqual(model.calls, old_model_calls)
            original_arrays = captured['legacy']
            new_arrays = captured['hinted']
            self.assertEqual(original_arrays[3], new_arrays[3])
            for a,b in zip(original_arrays[:3], new_arrays[:3]):
                np.testing.assert_array_equal(a,b)

    def test_wrong_hint_does_not_export_mesh(self):
        with tempfile.TemporaryDirectory() as temp:
            grid, curve, agent, _, adapt, config = self.fixture(temp, resolution=22)
            def forbidden(*args, **kwargs):
                raise AssertionError('Should not mesh with incorrect count')
            agent.extract_surface_mesh = forbidden
            _, total, _ = curve.filter_grid_adapt(
                grid, {**adapt, '_dcsdd_stream_callback':lambda a,i:None})
            with self.assertRaisesRegex((RuntimeError, ValueError, AssertionError),
                                        '(?i)accepted|sample|count|metadata|inferred'):
                agent._stream_individual_rigid_boot(
                    curve, grid, adapt, 'boots',
                    {**config, 'dcsdd_stream_accepted_count':total+1},
                    0, 'direct', 19, temp)

    def test_hint_requires_replay_and_valid_range(self):
        with tempfile.TemporaryDirectory() as temp:
            grid, curve, agent, _, adapt, config = self.fixture(temp, resolution=14)
            with self.assertRaisesRegex(ValueError, 'crop-first replay'):
                agent._stream_individual_rigid_boot(
                    curve, grid, adapt, 'boots',
                    {**config, 'dcsdd_stream_replay_metadata':False,
                     'dcsdd_stream_accepted_count':123},
                    0, 'direct', 19, temp)
            for bad in (0, -1, (grid.reso+1)**3+1):
                with self.subTest(bad=bad):
                    with self.assertRaisesRegex(ValueError, 'voxel range'):
                        agent._stream_individual_rigid_boot(
                            curve, grid, adapt, 'boots',
                            {**config, 'dcsdd_stream_accepted_count':bad},
                            0, 'direct', 19, temp)


if __name__ == '__main__':
    unittest.main()
