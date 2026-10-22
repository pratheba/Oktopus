"""0093: guarded generic wrap acceleration and post-blend Boolean geometry."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.sdf_zero_angle_optimization import can_skip_inert_snug
from app.sdf_stream_combined_wrap import (
    validate_cone_stream, combine_two_sparse_parts,
    merge_sparse_avatar_field, apply_postblend_avatar_boolean,
)
from app.sdf_stream_evaluator import SDFStreamPiece
from post_sdf_seam_blend import PostSDFSeamPair
from root_leaf_combination import build_root_leaf_plan


def three_wrap_items():
    return [{
        'name': f'dress_section_{i}', 'target_key': 'trex|body',
        'accessory_key': 'dress|top', 'blend_group': 'dress_body',
        'mode': 'direct', 'wrap_radius': True, 'rigid_radius': False,
        'wrap_angular_follow': 0.0, 'use_accessory_support_clamp': True,
        'auto_avatar_snug_field': True, 'snug_mode': 'multiplicative',
        'src_0': i * .2, 'src_1': i * .2 + .4,
        'tgt_0': i * .2, 'tgt_1': i * .2 + .4,
    } for i in range(3)]


class GenericWrap0093Tests(unittest.TestCase):
    def setUp(self):
        self.items = three_wrap_items()
        self.seams = {
            1: PostSDFSeamPair('dress_body', 0, 1, .2, .4, .15),
            2: PostSDFSeamPair('dress_body', 1, 2, .4, .6, .15),
        }
        self.cfg = dict(combine_adaptations=True, export_individual_meshes=False,
                        dcsdd_crop_grid=True, sdf_localize_after_gate=True,
                        sdf_optimize_zero_angular_wrap=True)
        self.plan = build_root_leaf_plan(self.items, {
            'output_mode': 'separate_parts',
            'parts': {'dress': {'blend_group': 'dress_body'}}})

    def test_generic_three_wrap_sections_share_one_part(self):
        self.assertEqual(validate_cone_stream(
            self.items, self.cfg, self.seams, {}, self.plan), self.seams)
        self.assertTrue(all(can_skip_inert_snug(x) for x in self.items))

    def test_multi_part_mixed_and_active_snug_rejected(self):
        with self.assertRaisesRegex(ValueError, 'ONE YAML-defined'):
            validate_cone_stream(self.items, self.cfg, self.seams, {},
                SimpleNamespace(part_names=('body', 'dress'),
                                item_to_part={0:'dress',1:'dress',2:'body'}))
        for key, value in [('rigid_radius', True), ('wrap_angular_follow', .3),
                           ('cut_avatar', True), ('snug_mode', 'additive'),
                           ('use_attach_to_free_blend', True)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                broken = [dict(x) for x in self.items]
                broken[1][key] = value
                validate_cone_stream(broken, self.cfg, self.seams, {}, self.plan)

    def test_avatar_alignment_and_boolean_after_seam(self):
        a = np.array([2,3,5])
        b = np.array([3,5,6])
        c = np.array([5,6,8])
        av1 = np.array([.1,.3,.2])
        av2 = np.array([.3,.2,.1])
        av3 = np.array([.2,.1,.3])
        union, values = merge_sparse_avatar_field(a, av1, b, av2)
        union, values = merge_sparse_avatar_field(union, values, c, av3)
        np.testing.assert_array_equal(union, [2,3,5,6,8])
        np.testing.assert_array_equal(values, [.1,.3,.2,.1,.3])
        accessory = np.array([-.02,-.2,-.1,.1,-.4])
        out = apply_postblend_avatar_boolean(union, accessory, union, values,
                                             clearance=.05)
        np.testing.assert_allclose(out, [-.02,-.2,-.1,.1,-.25])
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            merge_sparse_avatar_field(a, av1, b, av2+.02, tolerance=.005)
        with self.assertRaisesRegex(ValueError, 'exactly'):
            apply_postblend_avatar_boolean(union, accessory, a, av1)

    def test_avatar_samples_record_in_single_stream(self):
        with tempfile.TemporaryDirectory() as td:
            def model(x):
                z=np.asarray(x['samples_local'])[:,0]
                return {'dist': -z*.1, 'dist_base': -z*.09,
                        'dist_detail': z*0.0}
            def avatar(x):
                return np.asarray(x['samples_local'])[:,0]*.11
            st = SDFStreamPiece(batch_size=4, outdir=td, inference=model,
                                record_avatar_source=True, avatar_inference=avatar)
            n=12
            data={'coords':np.linspace(.1,.9,n),'samples_local':np.ones((n,3)),
                  'angles':np.zeros(n),'radius':np.ones(n),
                  'rho':np.ones(n),'rho_n':np.ones(n)}
            for start in range(0,n,4):
                sub={k:v[start:start+4] for k,v in data.items()}
                st.push(sub, np.arange(start,start+4), sub)
            st.finish(expected_accepted=n,fallback=1.0)
            np.testing.assert_allclose(st.avatar_sdf,np.full(n,.11))
            np.testing.assert_allclose(st.source_s,data['coords'])

    def test_postblend_does_not_ignore_unsupported_modes(self):
        from pathlib import Path
        text=Path(__file__).resolve().parents[1].joinpath(
            'src/app/agent_3dvec_sdf.py').read_text()
        self.assertIn('0093 post-blend avatar Boolean currently requires',text)
        self.assertIn('sdf_wrap_auto_safe_0093',text)
        self.assertIn('sdf_stream_combined_wrap',text)


if __name__ == '__main__':
    unittest.main()

class GenericWrapStreamingIntegration(unittest.TestCase):
    def test_three_sections_named_part_postblend_boolean_one_crop(self):
        from app.agent_3dvec_sdf_dc import AgentSDFDC
        from utils.mc_utils import MCGrid
        import trimesh
        items=three_wrap_items()
        n1=29
        rng=np.random.default_rng(493)
        xyz=rng.integers(7,23,size=(530,3))
        keys=np.unique(xyz[:,0]+n1*(xyz[:,1]+n1*xyz[:,2]))
        parts=[keys[:245],keys[100:365],keys[230:]]
        def data(idx):
            n=len(idx)
            loc=np.zeros((n,3))
            loc[:,0]=idx/30000.0
            loc[:,1]=0.1
            return {'coords': np.linspace(.1,.7,n), 'samples_local':loc,
                    'angles':np.zeros(n),'radius':np.ones(n),
                    'rho':np.zeros(n),'rho_n':np.ones(n)}
        inp=[data(p) for p in parts]
        class FakeCurve:
            name='body'
            calls=[]
            def filter_grid_adapt(self, grid, arg):
                j=round(float(arg['src_0'])/.2)
                self.calls.append(j)
                idx=parts[j]; values=inp[j]
                for start in range(0,len(idx),47):
                    end=start+47
                    sub={k:v[start:end] for k,v in values.items()}
                    avatar=dict(sub)
                    avatar['coords']=np.linspace(.2,.5,len(idx))[start:end]
                    arg['_dcsdd_stream_callback'](sub, idx[start:end], avatar)
                return 1200,len(idx),0.0
        curve=FakeCurve()
        agent=AgentSDFDC()
        agent.device='cpu'
        agent.feat_dict={'trex|body':0,'dress|top':1}
        agent.curve_from_key=lambda key: object()
        agent.model=SimpleNamespace(detail_model=SimpleNamespace(sigma=.1))
        def predict(batch,key,batch_size=None):
            u=np.asarray(batch['samples_local'])[:,0]
            if key=='trex|body':
                return {'dist': np.where(u>0.48,-.03,.2)}
            return {'dist':np.full(len(u),-.10),
                    'dist_base':np.full(len(u),-.105),
                    'dist_detail':np.zeros(len(u))}
        agent._inference_full_vals=predict
        captured=[]
        def extract(grid,cfg,context):
            captured.append((np.asarray(grid.val_grid).copy(),
                             np.asarray(grid.empty_marks).copy(),
                             np.asarray(grid.crop_start_global),grid.reso+1))
            return trimesh.Trimesh(vertices=np.array([[0,0,0],[1,0,0],
                                    [0,1,0],[0,0,1]],dtype=float),
                                   faces=np.array([[0,1,2],[0,3,1],
                                   [0,2,3],[1,3,2]]), process=False)
        agent.extract_surface_mesh=extract
        raw={'blend_groups':{'dress_body':{'seam_blend':{
            'enabled':True,'mode':'overlap_crossfade',
            'edge_hold_fraction':.15}}}}
        with tempfile.TemporaryDirectory() as td:
            grid=MCGrid({'reso':n1-1,'size':.8,'level':0,
                         'geometry_only':True,'memmap_dir':td})
            cfg={'combine_adaptations':True,'export_individual_meshes':False,
                 'sdf_stream_combined_wrap':True,'dcsdd_crop_grid':True,
                 'sdf_localize_after_gate':True,
                 'sdf_optimize_zero_angular_wrap':True,
                 'dcsdd_stream_replay_dir':td,
                 'dcsdd_filter_batch_size':47,
                 'sdf_postblend_avatar_boolean_0093':{
                     'enabled':True,'clearance':.001},
                 'sdf_fused_exterior_0087':True,
                 'sdf_stream_export_sections_0093':True,
                 'root_leaves':{'output_mode':'separate_parts',
                     'parts':{'dress':{'blend_group':'dress_body'}}},
                 'surface_extraction':'ours','level':0.}
            agent._stream_combined_wrap_cone(
                {'shape':'trex','mc_grid':grid},cfg,items,raw,
                SimpleNamespace(curves=[curve]),td,'trex_adapt',47)
            self.assertEqual(curve.calls,[0,1,2])
            self.assertEqual(len(captured),4)  # 3 sections + 1 blended final
            for j in range(3):
                self.assertTrue((Path(td)/f'individual_adaptations/section_{j:02d}.ply').is_file())
            self.assertTrue((Path(td)/'riggable_parts/dress.ply').is_file())
            self.assertTrue((Path(td)/'riggable_fused_sdf_exterior.ply').is_file())
            self.assertTrue((Path(td)/'root_leaves_ownership.json').is_file())
            self.assertTrue((Path(td)/'riggable_fused_sdf_exterior_report.json').is_file())
            vals, marks, origin, side=captured[-1]
            for idx in np.union1d(np.union1d(parts[0],parts[1]),parts[2]):
                i=int(idx)%n1; j=(int(idx)//n1)%n1; k=int(idx)//(n1*n1)
                ijk=np.array([i,j,k])-origin
                if np.any(ijk<0) or np.any(ijk>=side):
                    continue
                loc=int(ijk[0]+side*(ijk[1]+side*ijk[2]))
                self.assertFalse(marks[loc])
                avatar=(-.03 if idx/30000.0 > .48 else .2)
                self.assertGreaterEqual(vals[loc]+1e-10,.001-avatar)
