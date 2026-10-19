"""0037: exact crop-first composition parity with legacy full-grid cropping."""
import tempfile
import unittest
import numpy as np

from utils.mc_utils import MCGrid
from app.dcsdd_stream_crop import (
    NegativeVoxelBounds, make_crop_grid, write_global_chunk_to_crop,
)
from app.agent_3dvec_sdf_dc import AgentSDFDC


class TestStreamCrop(unittest.TestCase):
    def test_crop_precomposition_matches_full_grid_for_random_negatives(self):
        rng = np.random.default_rng(377)
        adapter = AgentSDFDC()
        for n in (16, 29, 47):
            for trial in range(16):
                size = 1.2
                global_grid = MCGrid({'reso': n, 'size': size, 'level': 0.})
                n1 = n + 1
                full = n1 ** 3
                # Constrain negative voxels to one compact cube while adding
                # evaluated positive samples outside it. This tests the
                # exact-background reconstruction performed by crop-first.
                positions = rng.choice(full, size=min(int(full*.12), 9000), replace=False)
                vals = rng.uniform(0.01, 0.5, size=positions.size)
                # Deliberately include edge-of-domain negatives in some cases.
                starts_cluster = np.array([n//4, n//3, n//4], dtype=np.int64)
                if trial % 5 == 0:
                    starts_cluster = np.array([0, n//3, 0], dtype=np.int64)
                width = max(1, n//10)
                xyz = rng.integers(0, width, size=(50,3)) + starts_cluster
                neg_idx = xyz[:,0] + n1*(xyz[:,1] + n1*xyz[:,2])
                neg_val = -rng.uniform(.001,.1,size=len(neg_idx))
                positions = np.concatenate((positions,neg_idx))
                vals = np.concatenate((vals,neg_val))
                # Collision handling: negative values take priority.
                order=np.argsort(positions,kind='stable')
                positions,vals=positions[order],vals[order]
                uniq,start=np.unique(positions,return_index=True)
                vals=np.minimum.reduceat(vals,start)
                positions=uniq
                global_grid.update_grid(vals, positions, mode='overwrite', mark=True)
                config = {'dcsdd_crop_grid': True}
                expected_s, expected_u, expected_empty, side = adapter._grid_to_S_U(
                    global_grid, config=config)

                bounds = NegativeVoxelBounds(n1)
                # Scramble batch sizes, retain *global* ordering and values.
                for pos in range(0, len(positions), 17):
                    bounds.add(positions[pos:pos+17], vals[pos:pos+17])
                neg_lo, neg_hi = bounds.bounds()
                step = global_grid.grid_config['step']
                pad = int(np.ceil(adapter._dcsdd_band(step, config)/step)) + 2
                starts, got_side = adapter._crop_cube_bounds(neg_lo, neg_hi, n1, pad)
                self.assertEqual(side, got_side)
                grid = make_crop_grid(global_grid, starts, side)
                for pos in range(0, len(positions), 23):
                    write_global_chunk_to_crop(grid, positions[pos:pos+23], vals[pos:pos+23],
                                               n1=n1, starts=starts)
                got_s, got_u, got_empty, got_side2 = adapter._grid_to_S_U(
                    grid, config={'dcsdd_crop_grid': False})
                self.assertEqual(got_side2, side)
                np.testing.assert_array_equal(got_s, expected_s)
                np.testing.assert_array_equal(got_empty, expected_empty)
                np.testing.assert_allclose(got_u, expected_u, rtol=0, atol=1e-14)

    def test_negative_bbox_ignores_nonfinite_and_checks_bounds(self):
        n1=13
        tracker=NegativeVoxelBounds(n1)
        tracker.add(np.array([0, 12, 13, 169, 171]),
                    np.array([np.nan, -.1, np.inf, -.2, .3]))
        lo,hi=tracker.bounds()
        np.testing.assert_array_equal(lo,[0,0,0])
        np.testing.assert_array_equal(hi,[12,0,1])
        self.assertEqual(tracker.count,2)
        with self.assertRaises(ValueError):
            tracker.add([n1**3],[-.1])

    def test_grid_uses_memmap_above_threshold(self):
        with tempfile.TemporaryDirectory() as temp:
            global_grid=MCGrid({'reso':20,'size':1.2,'level':0.,'geometry_only':True,
                                'memmap_dir':temp})
            grid=make_crop_grid(global_grid,np.array([3,4,5]),11,
                                max_ram_samples=1000)
            self.assertIsInstance(grid.val_grid,np.memmap)
            self.assertEqual(grid.grid_config['step'],global_grid.grid_config['step'])
            self.assertEqual(grid.reso,10)

class TestStreamCropIntegration(unittest.TestCase):
    def test_agent_streamed_cropped_field_matches_original_full_field(self):
        import torch
        import trimesh
        from ngc.handle_utils.mesh_utils import CylindersMesh
        from test_dcsdd_bounded_filtering import FakeCurve
        from test_dcsdd_compact_2048 import CompactCore
        rng=np.random.default_rng(1006)
        class Model:
            class detail_model:
                sigma = .047
            def inference_full(self,data,transform=None):
                coords=torch.as_tensor(data['coords'],dtype=torch.float32)
                sl=torch.as_tensor(data['samples_local'],dtype=torch.float32)
                v=coords*.3+sl[:,0]*.1-2.1
                b=sl[:,1]*.2+sl[:,0]*.1-2.4
                detail=sl[:,2]*.025
                return {'dist':v[:,None], 'dist_base':b[:,None],
                        'dist_detail':detail[:,None], 'gate_detail': None}
        for n in (22, 42):
            with tempfile.TemporaryDirectory() as directory:
                global_grid=MCGrid({'reso':n,'size':1.2,'level':0.,
                                    'memmap_dir':directory,'geometry_only':True})
                cyl=CylindersMesh()
                cyl.cylinder_samples=rng.normal(0,.18,(12,12,3))
                curve=FakeCurve(cyl)
                curve.core=CompactCore()
                agent=AgentSDFDC()
                agent.model=Model()
                agent.device=torch.device('cpu')
                agent.feat_dict={'boots':1}
                arg={'mode':'direct','rigid_radius':True,'src_0':.1,'src_1':.9,
                     'use_adapt_control_filter_grid':True,
                     'dcsdd_fast_bbox_marking':True,
                     'dcsdd_bounded_filtering':True,
                     'dcsdd_filter_batch_size':11,
                     'use_accessory_support_clamp':True,
                     'support_rho_limit':30., 'support_positive_value':1.,
                     'adapt_debug_counts':False}
                captured={}
                def collect(input_grid, cfg, **kwargs):
                    captured['grid']=input_grid
                    captured['config']=cfg
                    return trimesh.Trimesh(vertices=np.empty((0,3)),faces=np.empty((0,3),dtype=int))
                agent.extract_surface_mesh=collect
                agent._stream_individual_rigid_boot(
                    curve,global_grid,arg,'boots',
                    {'dcsdd_stream_crop_first':True,'dcsdd_crop_grid':True,'surface_extraction':'ours'},
                    0,'direct',19,directory)
                self.assertFalse(captured['config']['dcsdd_crop_grid'])
                cropped=captured['grid']
                acc,_,idx,_=curve.filter_grid_adapt(
                    global_grid,{**arg,'dcsdd_compact_staging':True})
                vals=agent._inference_full_vals(acc,'boots',batch_size=19,slice_batches=True)
                dist,valid=agent.clamp_pred_sdf_by_support(
                    vals['dist'],acc,rho_limit=30.,positive_value=1.,verbose=False)
                base,_=agent.clamp_pred_sdf_by_support(
                    vals['dist_base'],acc,rho_limit=30.,positive_value=1.,verbose=False)
                detail=np.where(valid,vals['dist_detail'],0.)
                expect=base+np.exp(-(base*base)/(2.*.047**2+1e-12))*detail
                dense=MCGrid({'reso':n,'size':1.2,'level':0.})
                dense.update_grid(expect,idx,mark=True,mode='overwrite')
                expected_s,expected_u,expected_empty,expected_side=agent._grid_to_S_U(
                    dense,config={'dcsdd_crop_grid':True})
                got_s,got_u,got_empty,got_side=agent._grid_to_S_U(
                    cropped,config={'dcsdd_crop_grid':False})
                self.assertEqual(got_side,expected_side)
                np.testing.assert_allclose(got_s,expected_s,rtol=1e-14,atol=1e-14)
                np.testing.assert_array_equal(got_empty,expected_empty)
                np.testing.assert_allclose(got_u,expected_u,rtol=0,atol=1e-14)

if __name__=='__main__':
    unittest.main()
