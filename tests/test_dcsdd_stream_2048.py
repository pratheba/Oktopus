"""0036 parity tests for exact-global-support, bounded prediction streaming.

PYTHONPATH=src:src/ngc python -m unittest discover -s tests -p 'test_dcsdd_stream_2048.py' -v
"""
import tempfile
import unittest
import numpy as np

from app.dcsdd_stream_boot import (
    GlobalRhoCalibration, clamp_with_profile, compose_boot_chunk,
    inference_partition_sizes, PartitionedPredictionWriter, INFERENCE_KEYS,
)
from app.agent_3dvec_base import AgentBase
from ngc.curve_handle import CurveHandle
from ngc.handle_utils.mesh_utils import CylindersMesh
from utils.mc_utils import MCGrid
from test_dcsdd_bounded_filtering import FakeCurve, FakeCore


class TestStreamRhoAndInference(unittest.TestCase):
    def test_global_calibration_identical_to_existing_quantiles(self):
        rng = np.random.default_rng(1234)
        for n in (3, 50, 1000, 9001):
            data = {
                'samples_local': rng.normal(size=(n,3)),
                'coords': rng.random(n),
            }
            pred = rng.normal(0, .05, n)
            for cutoff in (7, 64, 263):
                c = GlobalRhoCalibration()
                for pos in range(0, n, cutoff):
                    c.add(pred[pos:pos+cutoff],
                          data['samples_local'][pos:pos+cutoff],
                          data['coords'][pos:pos+cutoff])
                expected = AgentBase().estimate_rho_limit_from_pred(pred, data, fallback=1.3)
                actual = c.finish(fallback=1.3)
                np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=0)
                np.testing.assert_allclose(actual[1], expected[1], rtol=1e-14, atol=1e-14)

    def test_exact_full_clamp_recomposition(self):
        rng = np.random.default_rng(2345)
        for n in (150, 1800, 9001):
            data = {
                'samples_local': rng.normal(size=(n,3)),
                'coords': rng.random(n),
            }
            pred = rng.normal(0, .07, n)
            base = rng.normal(0, .06, n)
            detail = rng.normal(0,.01,n)
            radius_limit = 1.0
            sigma = .015
            obj = AgentBase()
            full_pred, valid = obj.clamp_pred_sdf_by_support(
                pred, data, rho_limit=radius_limit, verbose=False)
            full_base, _ = obj.clamp_pred_sdf_by_support(
                base, data, rho_limit=radius_limit, verbose=False)
            ref = full_base + np.exp(
                -(full_base * full_base) / (2*sigma*sigma+1e-12)) * np.where(valid, detail, 0.)
            profiles = {}
            for key, values in (('dist',pred), ('dist_base',base)):
                c=GlobalRhoCalibration()
                for i in range(0,n,97):
                    c.add(values[i:i+97],data['samples_local'][i:i+97],data['coords'][i:i+97])
                profiles[key]=c.finish(fallback=radius_limit)
            got=[]
            for i in range(0,n,101):
                d={k:v[i:i+101] for k,v in data.items()}
                raw={'dist':pred[i:i+101], 'dist_base':base[i:i+101],
                     'dist_detail':detail[i:i+101]}
                out,_,_=compose_boot_chunk(raw,d,profiles,sigma=sigma,rho_limit=radius_limit)
                got.append(out)
            np.testing.assert_allclose(np.concatenate(got),ref,rtol=1e-14,atol=1e-14)

    def test_partitioner_preserves_global_inference_batches(self):
        rng=np.random.default_rng(443)
        for n in (1,7,62,64,65,100,251,262):
            for bs in (8,64,100):
                all_data={
                    'samples_local':rng.normal(size=(n,3)),
                    'coords':rng.uniform(size=n),
                    'angles':rng.normal(size=n),
                    'radius':rng.uniform(size=n),
                    'rho':rng.uniform(size=n),
                    'rho_n':rng.uniform(size=n),
                }
                def inference(data):
                    v=data['coords'] + .7*data['samples_local'][:,0]
                    return {'dist':v.astype('float32'),
                            'dist_base':(v + .07).astype('float32'),
                            'dist_detail':(v*.03).astype('float32')}
                ref=inference(all_data)
                sizes=[]
                with tempfile.TemporaryDirectory() as directory:
                    def observe(chunk):
                        sizes.append(len(chunk['coords']))
                        return inference(chunk)
                    writer=PartitionedPredictionWriter(
                        n=n,batch_size=bs,outdir=directory,inference=observe)
                    pos=0
                    while pos<n:
                        width=int(rng.integers(1,15))
                        writer.push({k:v[pos:pos+width] for k,v in all_data.items()})
                        pos+=width
                    writer.finish(fallback=1.)
                    self.assertEqual(tuple(sizes),inference_partition_sizes(n,bs))
                    self.assertEqual(writer.written,n)
                    for k in ref:
                        np.testing.assert_array_equal(writer.raw[k],ref[k])

    def test_callback_indices_and_values_match_bounded(self):
        rng=np.random.default_rng(900)
        grid=MCGrid({'reso':18,'size':1.2,'level':0.})
        cylinder=CylindersMesh()
        cylinder.cylinder_samples=rng.normal(0,.3,(14,12,3))
        curve=FakeCurve(cylinder)
        common={'mode':'direct','rigid_radius':True,'src_0':.1,'src_1':.9,
                'use_adapt_control_filter_grid':True,
                'dcsdd_fast_bbox_marking':True,
                'dcsdd_bounded_filtering':True,
                'dcsdd_filter_batch_size':11,
                'adapt_debug_counts':False}
        reference,_,ref_idx,_=curve.filter_grid_adapt(grid,{**common,'dcsdd_compact_staging':True})
        kept=[]
        indices=[]
        def callback(acc,kidx):
            kept.append(np.asarray(acc['samples_local']))
            indices.append(np.asarray(kidx))
        result=curve.filter_grid_adapt(grid,{**common,'_dcsdd_stream_callback':callback})
        self.assertEqual(result[1],len(ref_idx))
        np.testing.assert_array_equal(np.concatenate(indices),ref_idx)
        np.testing.assert_array_equal(np.concatenate(kept),reference['samples_local'])

if __name__=='__main__':
    unittest.main()

class TestEndToEndStreaming(unittest.TestCase):
    def test_streamed_grid_matches_existing_unbatched_path(self):
        import torch
        import trimesh
        from app.agent_3dvec_sdf import AgentSDF
        from test_dcsdd_compact_2048 import CompactCore
        rng=np.random.default_rng(666)
        class Model:
            class detail_model:
                sigma = .047
            def inference_full(self,data,transform=None):
                coords=torch.as_tensor(data['coords'],dtype=torch.float32)
                sl=torch.as_tensor(data['samples_local'],dtype=torch.float32)
                v=coords*.3+sl[:,0]*.1-0.11
                b=sl[:,1]*.2+sl[:,0]*.1-.06
                detail=sl[:,2]*.025
                return {'dist':v[:,None], 'dist_base':b[:,None],
                        'dist_detail':detail[:,None], 'gate_detail': None}
        with tempfile.TemporaryDirectory() as directory:
            grid=MCGrid({'reso':18,'size':1.2,'level':0.,
                         'memmap_dir':directory,'geometry_only':True})
            cyl=CylindersMesh()
            cyl.cylinder_samples=rng.normal(0,.3,(12,12,3))
            curve=FakeCurve(cyl)
            curve.core=CompactCore()
            base_arg={'mode':'direct','rigid_radius':True,'src_0':.1,'src_1':.9,
                      'use_adapt_control_filter_grid':True,
                      'dcsdd_fast_bbox_marking':True,
                      'dcsdd_bounded_filtering':True,
                      'dcsdd_filter_batch_size':13,
                      'use_accessory_support_clamp':True,
                      'support_rho_limit':1., 'support_positive_value':1.,
                      'adapt_debug_counts':False}
            agent=AgentSDF()
            agent.model=Model()
            agent.device=torch.device('cpu')
            agent.feat_dict={'boots':1}
            recorded={}
            def collect(input_grid, cfg, **kwargs):
                idx=np.flatnonzero(~input_grid.empty_marks)
                recorded['indices']=idx
                recorded['values']=np.asarray(input_grid.val_grid[idx]).copy()
                return trimesh.Trimesh(vertices=np.empty((0,3)),faces=np.empty((0,3),dtype=int))
            agent.extract_surface_mesh=collect
            agent._stream_individual_rigid_boot(curve,grid,base_arg,'boots',{},0,'direct',19,directory)
            acc,_,idx,_=curve.filter_grid_adapt(grid,{**base_arg,'dcsdd_compact_staging':True})
            vals=agent._inference_full_vals(acc,'boots',batch_size=19,slice_batches=True)
            dist,valid=agent.clamp_pred_sdf_by_support(
                vals['dist'],acc,rho_limit=1.,positive_value=1.,verbose=False)
            base,_=agent.clamp_pred_sdf_by_support(
                vals['dist_base'],acc,rho_limit=1.,positive_value=1.,verbose=False)
            detail=np.where(valid,vals['dist_detail'],0.)
            expected=base+np.exp(-(base*base)/(2.*.047**2+1e-12))*detail
            np.testing.assert_array_equal(recorded['indices'],idx)
            np.testing.assert_allclose(recorded['values'],expected,rtol=1e-14,atol=1e-14)
