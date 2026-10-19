"""0046 sparse cone union, support calibration and strict fail-closed guards."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.dcsdd_stream_boot import GlobalRhoCalibration
from app.dcsdd_stream_crop import (
    NegativeVoxelBounds, make_crop_grid, write_global_chunk_to_crop,
)
from app.sdf_stream_combined_wrap import (
    validate_cone_stream, combine_two_sparse_parts,
    compose_piece_from_replay,
)
from app.dcsdd_stream_one_pass import OnePassBootStreamWriter
from post_sdf_seam_blend import (
    PostSDFSeamPair, overlap_crossfade_sdf,
)
from utils.mc_utils import MCGrid


PAIR = PostSDFSeamPair('cone_body', 0, 1, .2, .3, .15)


class SparseWrapConeTests(unittest.TestCase):
    def test_sparse_seam_matches_dense_group(self):
        rng = np.random.default_rng(46)
        for n in (19, 1000, 6000):
            left_idx = np.sort(rng.choice(n * 5, n, replace=False)).astype(np.int64)
            right_idx = np.sort(rng.choice(n * 5, n, replace=False)).astype(np.int64)
            left = rng.standard_normal(n)
            right = rng.standard_normal(n)
            src = rng.uniform(.1, .4, n)
            indices, values, shared = combine_two_sparse_parts(
                left_idx, left, right_idx, right, src, PAIR)
            group = np.full(n * 5, 10., dtype=np.float64)
            marks = np.ones(n * 5, dtype=bool)
            group[left_idx] = left
            marks[left_idx] = False
            overlap = ~marks[right_idx]
            group[right_idx[overlap]] = overlap_crossfade_sdf(
                group[right_idx[overlap]], right[overlap], src[overlap], PAIR)
            group[right_idx[~overlap]] = right[~overlap]
            marks[right_idx] = False
            np.testing.assert_array_equal(indices, np.flatnonzero(~marks))
            np.testing.assert_array_equal(values, group[indices])
            self.assertEqual(shared, np.count_nonzero(overlap))

    def test_voxel_crop_matches_full_grid_crop(self):
        rng = np.random.default_rng(47)
        full = MCGrid({'reso': 40, 'size': 0.8, 'level': 0})
        n1 = 41
        # Two overlapping sparse fields whose negative region is well inside domain.
        xyz = np.stack(np.meshgrid(np.arange(9, 31), np.arange(8, 33),
                                   np.arange(7, 29), indexing='ij'), axis=-1).reshape(-1, 3)
        k = xyz[:, 0] + n1 * (xyz[:, 1] + n1 * xyz[:, 2])
        left_idx = np.sort(k[::2]); right_idx = np.sort(k[1::2])
        right_idx = np.union1d(right_idx, left_idx[::5])
        left_v = rng.standard_normal(len(left_idx)) * .09
        right_v = rng.standard_normal(len(right_idx)) * .09
        src = rng.uniform(.15, .35, len(right_idx))
        indices, values, _ = combine_two_sparse_parts(
            left_idx, left_v, right_idx, right_v, src, PAIR)
        full.clear_grid()
        full.update_grid(values, indices, mode='overwrite', mark=True)
        bounds = NegativeVoxelBounds(n1)
        bounds.add(indices, values)
        lo, hi = bounds.bounds()
        starts = np.maximum(lo - 2, 0)
        side = int(np.max(np.minimum(hi + 2, n1 - 1) - starts + 1))
        starts = np.minimum(starts, n1 - side)
        crop = make_crop_grid(full, starts, side)
        for start in range(0, len(indices), 999):
            write_global_chunk_to_crop(crop, indices[start:start + 999],
                                       values[start:start + 999], n1=n1, starts=starts)
        x, y, z = starts
        expected_v = np.asarray(full.val_grid).reshape((n1, n1, n1), order='F')[
            x:x+side, y:y+side, z:z+side].ravel(order='F')
        expected_m = np.asarray(full.empty_marks).reshape((n1, n1, n1), order='F')[
            x:x+side, y:y+side, z:z+side].ravel(order='F')
        np.testing.assert_array_equal(crop.val_grid, expected_v)
        np.testing.assert_array_equal(crop.empty_marks, expected_m)

    def test_global_rho_calibration_matches_legacy(self):
        from agent_3dvec_base import AgentBase
        rng = np.random.default_rng(49)
        n = 8013
        coords = rng.uniform(0.001, .999, n)
        local = rng.standard_normal((n, 3))
        pred = rng.standard_normal(n) * .035
        old = AgentBase.estimate_rho_limit_from_pred(
            AgentBase.__new__(AgentBase), pred,
            {'coords': coords, 'samples_local': local}, fallback=1.0)
        cal = GlobalRhoCalibration()
        for start in range(0, n, 237):
            cal.add(pred[start:start+237], local[start:start+237], coords[start:start+237])
        new = cal.finish(fallback=1.0)
        np.testing.assert_array_equal(new[0], old[0])
        np.testing.assert_allclose(new[1], old[1], rtol=0, atol=1e-14)

    def test_guard_rejects_active_snug_and_incompatible_modes(self):
        base = dict(mode='direct', target_key='avatar|body', accessory_key='cone|cone',
                    wrap_radius=True, rigid_radius=False, wrap_angular_follow=0.0,
                    use_accessory_support_clamp=True, auto_avatar_snug_field=True,
                    blend_group='cone_body')
        cfg = dict(combine_adaptations=True, export_individual_meshes=False,
                   dcsdd_crop_grid=True, sdf_localize_after_gate=True,
                   sdf_optimize_zero_angular_wrap=True)
        pair = {1: PAIR}
        self.assertIs(validate_cone_stream([base, base], cfg, pair, {}, None), PAIR)
        for change in ({'wrap_angular_follow': .4}, {'cut_avatar': True},
                       {'use_free_profile_scale': True},
                       {'snug_mode': 'additive'}):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    validate_cone_stream([base, {**base, **change}], cfg, pair, {}, None)
        with self.assertRaises(ValueError):
            validate_cone_stream([base, base], {**cfg, 'export_individual_meshes': True}, pair, {}, None)

    def test_end_to_end_replay_matches_direct_clamp_and_base_detail(self):
        from agent_3dvec_base import AgentBase
        rng = np.random.default_rng(50)
        n = 420
        coords = rng.uniform(.01, .99, n)
        local = rng.normal(0, .6, (n, 3))
        # Minimal complete model input for deterministic pointwise fake inference.
        acc = {'coords': coords, 'samples_local': local,
               'angles': np.zeros(n), 'radius': np.ones(n),
               'rho': np.ones(n), 'rho_n': np.ones(n)}
        def infer(part):
            u = np.asarray(part['samples_local'])[:, 1]
            return {'dist': u * .12, 'dist_base': u * .08,
                    'dist_detail': np.cos(u) * .02}
        with tempfile.TemporaryDirectory() as d:
            w = OnePassBootStreamWriter(batch_size=47, outdir=d, inference=infer)
            for start in range(0, n, 83):
                sub = {k: v[start:start+83] for k, v in acc.items()}
                w.push(sub, np.arange(start, min(start+83, n), dtype=np.int64))
            prof = w.finish(fallback=1.0)
            w.replay.finish()
            final = compose_piece_from_replay(
                w, prof, outdir=d, sigma=.15, rho_limit=1., positive_value=1.,
                chunk_size=57)
            a = AgentBase.__new__(AgentBase)
            out = infer(acc)
            _, valid = a.clamp_pred_sdf_by_support(out['dist'], acc,
                                                  rho_limit=1., positive_value=1.)
            base, _ = a.clamp_pred_sdf_by_support(out['dist_base'], acc,
                                                 rho_limit=1., positive_value=1.)
            expected = base + np.exp(-base**2 / (2*.15**2 + 1e-12)) * np.where(
                valid, out['dist_detail'], 0.)
            np.testing.assert_allclose(final, expected, rtol=0, atol=1e-14)


if __name__ == '__main__':
    unittest.main()

class CombinedWrapIntegration(unittest.TestCase):
    def test_two_sections_only_one_localization_each_and_exact_cropped_field(self):
        from types import SimpleNamespace
        from app.agent_3dvec_sdf_dc import AgentSDFDC
        from app.dcsdd_stream_crop import NegativeVoxelBounds
        from post_sdf_seam_blend import build_post_sdf_seam_plan
        import trimesh
        rng = np.random.default_rng(146)
        n1 = 31
        with tempfile.TemporaryDirectory() as d:
            grid = MCGrid({'reso':n1-1, 'size':.6, 'level':0,
                           'geometry_only':True, 'memmap_dir':d})
            xyz = rng.integers(6, 24, size=(520, 3))
            k = np.unique(xyz[:, 0] + n1 * (xyz[:, 1] + n1 * xyz[:, 2]))
            a, b = k[:380], k[100:]
            def acc_for(idx, phase):
                coord = np.linspace(.15, .55, len(idx))
                sl = np.zeros((len(idx), 3), dtype=np.float64)
                sl[:, 0] = .2
                sl[:, 1] = -.4 + np.linspace(0, .8, len(idx)) + phase * .05
                sl[:, 2] = .02
                return dict(coords=coord, samples_local=sl,
                            angles=np.zeros(len(idx)), radius=np.ones(len(idx)),
                            rho=np.zeros(len(idx)), rho_n=np.ones(len(idx)))
            inputs = [(a, acc_for(a, 0.)), (b, acc_for(b, 1.))]
            class FakeCurve:
                name='body'
                def __init__(self):
                    self.calls=[]
                def filter_grid_adapt(self, g, cfg):
                    src=float(cfg['src_0'])
                    j=0 if src==0. else 1
                    self.calls.append(j)
                    idx, data = inputs[j]
                    for start in range(0, len(idx), 37):
                        end=start+37
                        sub={key:value[start:end] for key,value in data.items()}
                        cfg['_dcsdd_stream_callback'](sub,idx[start:end],
                                                      {'coords': np.linspace(.12, .35, len(idx))[start:end]})
                    return 1700,len(idx),0.
            curve=FakeCurve()
            agent=AgentSDFDC()
            agent.device='cpu'
            agent.feat_dict={'avatar|body':0,'accessory|cone':1}
            agent.curve_from_key=lambda key: object()
            class Model:
                class detail_model: sigma=.12
            agent.model=Model()
            def predict(acc, key, batch_size=None):
                s=acc['samples_local'][:,1]
                return {'dist':s*.2-.13,'dist_base':s*.2-.15,
                        'dist_detail':np.sin(s)*.005}
            agent._inference_full_vals=predict
            captured=[]
            def capture(crop, cfg, context):
                self.assertEqual(context,'combined')
                self.assertFalse(cfg['dcsdd_crop_grid'])
                captured.append((np.asarray(crop.val_grid).copy(),
                                 np.asarray(crop.empty_marks).copy(),
                                 np.array(crop.crop_start_global),int(crop.reso)+1))
                return trimesh.Trimesh(vertices=np.empty((0, 3)),
                                       faces=np.empty((0, 3), dtype=int),process=False)
            agent.extract_surface_mesh=capture
            items=[{'target_key':'avatar|body','accessory_key':'accessory|cone',
                    'mode':'direct','wrap_radius':True,'rigid_radius':False,
                    'wrap_angular_follow':0.,'use_accessory_support_clamp':True,
                    'support_rho_limit':8.,'support_positive_value':1.,
                    'auto_avatar_snug_field':True,'blend_group':'cone_body',
                    'src_0':0.,'src_1':.3,'tgt_0':0.,'tgt_1':.7},
                   {'target_key':'avatar|body','accessory_key':'accessory|cone',
                    'mode':'direct','wrap_radius':True,'rigid_radius':False,
                    'wrap_angular_follow':0.,'use_accessory_support_clamp':True,
                    'support_rho_limit':8.,'support_positive_value':1.,
                    'auto_avatar_snug_field':True,'blend_group':'cone_body',
                    'src_0':.2,'src_1':.55,'tgt_0':.7,'tgt_1':.9}]
            raw={'blend_groups':{'cone_body':{'seam_blend':{
                 'enabled':True,'mode':'overlap_crossfade','edge_hold_fraction':.15}}}}
            cfg={'combine_adaptations':True,'export_individual_meshes':False,
                 'dcsdd_crop_grid':True,'sdf_localize_after_gate':True,
                 'sdf_optimize_zero_angular_wrap':True,
                 'dcsdd_stream_replay_dir':d,'surface_extraction':'ours',
                 'level':0.,'dcsdd_filter_batch_size':37}
            agent._stream_combined_wrap_cone(
                {'shape':'avatar','mc_grid':grid},cfg,items,raw,
                SimpleNamespace(curves=[curve]),d,'avatar_adapt',29)
            self.assertEqual(curve.calls,[0,1])
            self.assertEqual(len(captured),1)
            # Recompute old dense group via identical per-piece predictions,
            # globally calibrated clamp and original post-SDF seam handling.
            old=SimpleNamespace()
            before=np.full(n1**3,10.,dtype=np.float64)
            valid=np.zeros(n1**3,dtype=bool)
            pair=build_post_sdf_seam_plan(items,raw['blend_groups'])[1]
            for j,(idx, acc) in enumerate(inputs):
                out=predict(acc,'accessory|cone')
                from agent_3dvec_base import AgentBase
                ab=AgentBase.__new__(AgentBase)
                _,support=ab.clamp_pred_sdf_by_support(out['dist'],acc,
                    rho_limit=8., positive_value=1.)
                base,_=ab.clamp_pred_sdf_by_support(out['dist_base'],acc,
                    rho_limit=8.,positive_value=1.)
                values=base+np.exp(-base**2/(2*.12**2+1e-12))*np.where(
                    support,out['dist_detail'],0.)
                overlap=valid[idx]
                if j==1:
                    s=np.linspace(.12,.35,len(idx))
                    before[idx[overlap]]=overlap_crossfade_sdf(
                        before[idx[overlap]],values[overlap],s[overlap],pair)
                before[idx[~overlap]]=values[~overlap]
                valid[idx]=True
            bounds=NegativeVoxelBounds(n1)
            bounds.add(np.flatnonzero(valid),before[valid])
            lo,hi=bounds.bounds()
            step=float(grid.grid_config['step'])
            pad=int(np.ceil(agent._dcsdd_band(step,cfg)/step))+2
            starts,side=agent._crop_cube_bounds(lo,hi,n1,pad)
            self.assertEqual(side,captured[0][3])
            np.testing.assert_array_equal(starts,captured[0][2])
            xx,yy,zz=starts
            np.testing.assert_allclose(captured[0][0],before.reshape((n1,)*3,order='F')[
                xx:xx+side,yy:yy+side,zz:zz+side].ravel(order='F'),rtol=0,atol=1e-13)
            np.testing.assert_array_equal(captured[0][1],(~valid).reshape((n1,)*3,order='F')[
                xx:xx+side,yy:yy+side,zz:zz+side].ravel(order='F'))
