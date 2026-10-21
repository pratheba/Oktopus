"""0069: final-mesh per-adaptation curve interval ownership (no intermediate meshes)."""
import types
import sys
from pathlib import Path
import numpy as np
import pytest
import trimesh
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'app'))
from udf_cut_final_root_leaves import curve_interval_owner_masks
from agent_3dvec_udf_cut import AgentUDFCut


class LineCore:
    key_ts = np.asarray([0., 1.])
    def interpolate(self, ts):
        return {'points': np.column_stack([ts, np.zeros(len(ts)), np.zeros(len(ts))])}


def fixture_mesh():
    xcoords = [.08, .22, .44, .51, .67, .82, .95]
    vertices = []
    faces = []
    for x in xcoords:
        n = len(vertices)
        vertices.extend([[x-.005,0,0], [x+.005,.005,0], [x+.005,0,.005]])
        faces.append([n, n+1, n+2])
    return trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces), process=False)


def fixture_owners():
    curve = types.SimpleNamespace(core=LineCore())
    return [dict(item_index=i,target_key='t1',accessory_key=str(i),avatar_curve=curve,
                 adapt_arg=dict(src_0=a,src_1=b),seed_world=.02,grow_world=.01)
            for i,(a,b) in enumerate([(.52,.06),(.96,.48)])]


def test_interval_mapped_not_full_mesh():
    mesh = fixture_mesh()
    masks, report = curve_interval_owner_masks(mesh, fixture_owners(), pad=.035)
    assert report['unassigned_faces'] == 0
    assert sum(np.asarray(masks).sum(axis=0) > 1) == 0
    assert masks[0].sum() > 0 and masks[1].sum() > 0
    assert masks[0][0] and not masks[1][0]
    assert masks[1][-1] and not masks[0][-1]
    assert masks[0].sum() < len(mesh.faces)
    assert masks[1].sum() < len(mesh.faces)


def test_interval_unowned_is_not_arbitrarily_assigned():
    mesh = fixture_mesh()
    owners = fixture_owners()
    owners[0]['adapt_arg'] = {'src_0':.1,'src_1':.2}
    owners[1]['adapt_arg'] = {'src_0':.8,'src_1':.9}
    masks, report = curve_interval_owner_masks(mesh, owners, pad=.0)
    assert report['unassigned_faces'] >= 3
    assert not np.any(np.asarray(masks)[:, 3])


def test_interval_invalid_parameters_fail():
    with pytest.raises(ValueError, match='within'):
        curve_interval_owner_masks(fixture_mesh(), fixture_owners(), pad=-.1)
    owners = fixture_owners()
    del owners[0]['adapt_arg']['src_0']
    with pytest.raises(ValueError, match='Missing src_0'):
        curve_interval_owner_masks(fixture_mesh(), owners)


def test_0069_queries_each_owner_only_on_its_curve_region():
    mesh = fixture_mesh()
    agent = object.__new__(AgentUDFCut)
    visited = {}
    def query(self, mesh_arg, face_ids, **kwargs):
        visited[kwargs['accessory_key']] = np.asarray(face_ids).copy()
        return np.full(len(face_ids), .015), np.ones(len(face_ids), dtype=bool)
    agent._query_face_scores_for_ids = types.MethodType(query, agent)
    kept, removed, report, evidence = agent._direct_udf_cut_combined_mesh(
        mesh, owners=fixture_owners(), config={
            'udf_cut_direct_min_informative_faces':0,
            'udf_cut_direct_min_informative_fraction':0,
            'udf_cut_combined_direct_preserve_components':False,
            'udf_cut_combined_direct_preserve_existing_boundaries':False,
        }, model_batch_size=20,query_batch_size=20,far_world=.1,
        ownership_mode='curve_interval')
    assert len(visited)==2
    assert set(visited['0']).isdisjoint(set(visited['1']))
    assert len(visited['0'])+len(visited['1'])==len(mesh.faces)
    assert report['curve_interval_0069']['unassigned_faces']==0
    assert report['ownership_mode']=='curve_interval'
    assert len(kept.faces)==len(mesh.faces)
    assert len(removed.faces)==0


def test_0069_hysteresis_removes_only_high_udf_owned_region():
    mesh = fixture_mesh()
    agent = object.__new__(AgentUDFCut)
    def query(self, mesh_arg, face_ids, **kwargs):
        value = .04 if kwargs['accessory_key'] == '0' else .005
        return np.full(len(face_ids), value), np.ones(len(face_ids), dtype=bool)
    agent._query_face_scores_for_ids = types.MethodType(query, agent)
    kept, removed, report, _ = agent._direct_udf_cut_combined_mesh(
        mesh, owners=fixture_owners(), config={
            'udf_cut_direct_min_informative_faces':0,
            'udf_cut_direct_min_informative_fraction':0,
            'udf_cut_combined_direct_min_faces':1,
            'udf_cut_combined_direct_min_seed_faces':1,
            'udf_cut_combined_direct_min_seed_fraction':0,
            'udf_cut_combined_direct_preserve_components':False,
            'udf_cut_combined_direct_preserve_existing_boundaries':False,
        }, model_batch_size=20, query_batch_size=20, far_world=.1,
        ownership_mode='curve_interval')
    assert len(kept.faces) > 0 and len(removed.faces)>0
    assert np.max(np.asarray(removed.triangles_center)[:,0]) < .67
    assert np.min(np.asarray(kept.triangles_center)[:,0]) >= .51
    assert report['removed_faces'] == len(removed.faces)
