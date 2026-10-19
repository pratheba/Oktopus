"""0038 replay metadata parity against 0037 relocalization and composition."""
import tempfile
import unittest

import numpy as np

from app.dcsdd_stream_boot import compose_boot_chunk
from app.dcsdd_stream_replay import (
    BootReplayMetadata, REPLAY_DTYPE, compose_replay_chunk)
from utils.mc_utils import MCGrid


class TestReplayMath(unittest.TestCase):
    def test_exact_float64_support_composition(self):
        rng = np.random.default_rng(3801)
        for n in (1, 3, 31, 509, 2049):
            sl = rng.normal(size=(n, 3))
            sl[:,1:] *= rng.choice([.01, .4, 1., 5.], size=(n,1))
            coords = rng.uniform(-.4, 1.4, size=n)
            data = {'samples_local': sl, 'coords': coords}
            # Exactly the same three independently emitted scalar arrays.
            raw = {key: rng.normal(size=n) for key in ('dist', 'dist_base', 'dist_detail')}
            profile = (np.linspace(0.,1.,48),rng.uniform(.1,1.,size=48))
            profiles = {'dist': profile, 'dist_base': (profile[0], profile[1] + .13)}
            old, old_bad, old_rho = compose_boot_chunk(
                raw, data, profiles, sigma=.063, rho_limit=.95, positive_value=1.1)
            rho = np.sqrt(sl[:,1]*sl[:,1] + sl[:,2]*sl[:,2])
            new, new_bad, new_rho = compose_replay_chunk(
                raw, coords, rho, profiles, sigma=.063, rho_limit=.95,
                positive_value=1.1)
            np.testing.assert_array_equal(new, old)
            self.assertEqual(old_bad, new_bad)
            self.assertEqual(old_rho, new_rho)

    def test_replay_writer_preserves_order_indices_and_radial_values(self):
        rng=np.random.default_rng(3802)
        n=1009
        idx=np.sort(rng.choice(3999,size=n,replace=False))
        data={'samples_local':rng.normal(size=(n,3)),
              'coords':rng.uniform(-.2,1.2,size=n)}
        with tempfile.TemporaryDirectory() as d:
            writer=BootReplayMetadata(n=n,outdir=d)
            pos=0
            while pos < n:
                next_pos=min(n,pos+int(rng.integers(1,44)))
                writer.push({k:v[pos:next_pos] for k,v in data.items()},idx[pos:next_pos])
                pos=next_pos
            writer.finish()
            chunks=list(writer.iter_chunks(67))
            actual=np.concatenate([c for _,_,c in chunks])
            np.testing.assert_array_equal(actual['index'],idx)
            np.testing.assert_array_equal(actual['coord'],data['coords'])
            radial=np.sqrt(data['samples_local'][:,1] * data['samples_local'][:,1] +
                           data['samples_local'][:,2] * data['samples_local'][:,2])
            np.testing.assert_array_equal(actual['rho'],radial)
            self.assertEqual(writer.bytes,n*REPLAY_DTYPE.itemsize)
            with self.assertRaises(ValueError):
                writer.push(data,idx)


class TestAgentReplayIntegration(unittest.TestCase):
    def test_replay_matches_0037_and_original_dense_crop(self):
        import torch
        import trimesh
        from app.agent_3dvec_sdf_dc import AgentSDFDC
        from ngc.handle_utils.mesh_utils import CylindersMesh
        from test_dcsdd_bounded_filtering import FakeCurve
        from test_dcsdd_compact_2048 import CompactCore
        rng=np.random.default_rng(3803)
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
                        'dist_detail':detail[:,None], 'gate_detail':None}
        for n in (22, 42):
            with tempfile.TemporaryDirectory() as directory:
                grid=MCGrid({'reso':n,'size':1.2,'level':0.,
                             'memmap_dir':directory,'geometry_only':True})
                cylinder=CylindersMesh()
                cylinder.cylinder_samples=rng.normal(0,.18,(12,12,3))
                curve=FakeCurve(cylinder)
                curve.core=CompactCore()
                agent=AgentSDFDC()
                agent.model=Model()
                agent.device=torch.device('cpu')
                agent.feat_dict={'boots':1}
                arg={'mode':'direct','rigid_radius':True,'src_0':.1,'src_1':.9,
                     'use_adapt_control_filter_grid':True,
                     'dcsdd_fast_bbox_marking':True,'dcsdd_bounded_filtering':True,
                     'dcsdd_filter_batch_size':11,
                     'use_accessory_support_clamp':True,
                     'support_rho_limit':30., 'support_positive_value':1.,
                     'adapt_debug_counts':False}
                outputs={}
                def collect(input_grid,cfg,**kwargs):
                    s,u,empty,side=agent._grid_to_S_U(
                        input_grid,config={'dcsdd_crop_grid':False})
                    outputs[current[0]]=(np.asarray(s).copy(),np.asarray(u).copy(),
                                         np.asarray(empty).copy(),side)
                    return trimesh.Trimesh(vertices=np.empty((0,3)),
                                           faces=np.empty((0,3),dtype=int))
                agent.extract_surface_mesh=collect
                common={'dcsdd_stream_crop_first':True,
                        'dcsdd_crop_grid':True,'surface_extraction':'ours'}
                current=['legacy']
                agent._stream_individual_rigid_boot(
                    curve,grid,arg,'boots',common,0,'direct',19,directory)
                current[0]='replay'
                agent._stream_individual_rigid_boot(
                    curve,grid,arg,'boots',
                    {**common,'dcsdd_stream_replay_metadata':True},
                    0,'direct',19,directory)
                a,b=outputs['legacy'],outputs['replay']
                self.assertEqual(a[3],b[3])
                np.testing.assert_array_equal(a[0],b[0])
                np.testing.assert_array_equal(a[1],b[1])
                np.testing.assert_array_equal(a[2],b[2])

if __name__=='__main__':
    unittest.main()
