"""0071 restores optional per-owner calibration and rejects non-cap cut components."""
import sys
import types
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'app'))
from agent_3dvec_udf_cut import AgentUDFCut
from udf_cut_cap_guard_0071 import (
    calibrate_p95_from_final_faces, endcap_face_evidence, filter_cap_components,
)


class StraightCurve:
    key_ts = np.array([0.0, 1.0])

    def interpolate(self, ts):
        ts = np.asarray(ts)
        return {'points': np.column_stack([np.zeros(len(ts)), np.zeros(len(ts)), ts - 0.5])}


def segmented_cylinder(radius=.2, sections=64, nz=11):
    zs = np.linspace(-.5,.5,nz)
    theta = np.linspace(0, 2*np.pi, sections, endpoint=False)
    coords = np.array([[radius*np.cos(t), radius*np.sin(t), z] for z in zs for t in theta])
    faces=[]
    for k in range(nz-1):
        for j in range(sections):
            a=k*sections+j; b=k*sections+(j+1)%sections
            c=(k+1)*sections+j; d=(k+1)*sections+(j+1)%sections
            faces.extend([[a,b,c],[b,d,c]])
    low_idx=len(coords); high_idx=len(coords)+1
    coords = np.vstack([coords, [0,0,-.5], [0,0,.5]])
    for j in range(sections):
        faces.append([low_idx, (j+1)%sections, j])
        faces.append([high_idx, (nz-1)*sections+j, (nz-1)*sections+(j+1)%sections])
    return trimesh.Trimesh(vertices=coords, faces=np.array(faces), process=False)


def test_legacy_p95_formula_excludes_far_plateau():
    scores = np.r_[np.full(500, 0.006), np.full(500, .014), np.full(100, .1)]
    seed, grow, report = calibrate_p95_from_final_faces(scores, np.ones(len(scores), bool), far_world=.1)
    assert np.isclose(seed, .014 * 1.1)
    assert np.isclose(grow, .014 * .65)
    assert report['informative'] == 1000
    assert not report['saturated']


def test_calibration_abstains_without_informative_support():
    scores = np.r_[np.full(1000, .1), np.full(12, .008)]
    seed, grow, report = calibrate_p95_from_final_faces(scores, np.ones(len(scores), bool), far_world=.1)
    assert seed is None and grow is None
    assert report['saturated']


def test_cap_geometry_classifies_axial_disk_not_sidewall():
    mesh = segmented_cylinder()
    curve = types.SimpleNamespace(core=StraightCurve())
    endpoint, axial, report = endcap_face_evidence(mesh, curve, source_bounds=(0., 1.), end='both')
    normals = np.asarray(mesh.face_normals)
    caps = np.abs(normals[:, 2]) > .95
    walls = np.abs(normals[:, 2]) < .10
    assert np.count_nonzero(endpoint & axial & caps) >= 100
    assert np.count_nonzero(endpoint & axial & walls) == 0
    assert report['cap_faces'] >= 100


def test_low_end_guard_does_not_authorize_tip_cap():
    mesh = segmented_cylinder()
    curve = types.SimpleNamespace(core=StraightCurve())
    near, axial, report = endcap_face_evidence(mesh, curve, source_bounds=(0., 1.), end='low')
    centers = np.asarray(mesh.triangles_center)
    assert np.any(near & axial & (centers[:,2] < -.45))
    assert not np.any(near & axial & (centers[:,2] > .45))
    assert report['end']=='low'


def test_direct_final_keeps_sidewall_and_removes_verified_cap_only():
    mesh = segmented_cylinder()
    curve = types.SimpleNamespace(core=StraightCurve())
    agent = object.__new__(AgentUDFCut)
    normals = np.asarray(mesh.face_normals)
    centroids = np.asarray(mesh.triangles_center)
    caps = (centroids[:, 2] > .45) & (np.abs(normals[:, 2]) > .95)
    sidewall_band = (np.abs(centroids[:, 2]) < .10) & (np.abs(normals[:, 2]) < .10)
    assert caps.any() and sidewall_band.any()

    def fake_query(self, mesh_arg, face_ids, **kwargs):
        values = np.where(caps[face_ids] | sidewall_band[face_ids], .04, .002)
        return values, np.ones(len(values), dtype=bool)

    agent._query_face_scores_for_ids = types.MethodType(fake_query, agent)
    owners = [dict(item_index=1, target_key='t1', accessory_key='l_arm', avatar_curve=curve,
                   adapt_arg={'src_0':0., 'src_1':1.}, seed_world=.02, grow_world=.01)]
    kept, removed, report, _evidence = agent._direct_udf_cut_combined_mesh(
        mesh, owners=owners, config={
            '_udf_cut_cap_guard_active_0071': True,
            'udf_cut_direct_min_informative_faces': 0,
            'udf_cut_direct_min_informative_fraction': 0,
            'udf_cut_combined_direct_min_faces': 8,
            'udf_cut_combined_direct_preserve_existing_boundaries': False,
            'udf_cut_combined_direct_preserve_components': True,
            'udf_cut_direct_endpoint_band_0071': .07,
            'udf_cut_direct_cap_end_0071': 'both',
        }, model_batch_size=100,query_batch_size=100,far_world=.1,ownership_mode='curve_interval')
    removed_centers = np.asarray(removed.triangles_center)
    assert len(removed.faces) > 0
    assert len(removed.faces) <= caps.sum()
    assert np.min(removed_centers[:, 2]) > .45
    assert report['cap_guard_0071'] is not None
    assert any(c['reason'] in ('0071_not_at_curve_end','0071_not_axial_cap') for c in report['cap_guard_0071']['components'])
    assert len(kept.faces) + len(removed.faces) == len(mesh.faces)


def test_direct_final_abstains_on_sidewall_only():
    mesh = segmented_cylinder()
    curve = types.SimpleNamespace(core=StraightCurve())
    agent = object.__new__(AgentUDFCut)
    centers = np.asarray(mesh.triangles_center)
    side = (np.abs(centers[:,2]) < .15) & (np.abs(mesh.face_normals[:,2]) < .1)
    def query(self, mesh_arg, face_ids, **kwargs):
        return np.where(side[face_ids], .04, .002), np.ones(len(face_ids), dtype=bool)
    agent._query_face_scores_for_ids = types.MethodType(query, agent)
    owners = [dict(item_index=1,target_key='t1',accessory_key='x',avatar_curve=curve,
                   adapt_arg={'src_0':0.,'src_1':1.},seed_world=.02,grow_world=.01)]
    kept,removed,report,_ = agent._direct_udf_cut_combined_mesh(
        mesh,owners=owners,config={'_udf_cut_cap_guard_active_0071':True,
                                  'udf_cut_direct_min_informative_faces':0,
                                  'udf_cut_direct_min_informative_fraction':0,
                                  'udf_cut_combined_direct_preserve_existing_boundaries':False},
        model_batch_size=200, query_batch_size=200,far_world=.1,ownership_mode='curve_interval')
    assert len(kept.faces)==len(mesh.faces)
    assert len(removed.faces)==0
    assert report['cap_guard_0071']['accepted_faces']==0
