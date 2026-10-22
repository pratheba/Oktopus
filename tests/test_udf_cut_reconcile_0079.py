"""0079: overlap agreement + detached debris, no learned model required."""
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'app'))
from udf_cut_reconcile_0079 import (vote_agreement, keep_largest_edge_component,
                                    _edge_component_labels)


def test_one_owner_remove_and_keep():
    cut, cls = vote_agreement([1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 1, 0])
    np.testing.assert_array_equal(cut, [True, False, False])
    assert cls['missing_evidence'].tolist() == [False, False, True]


def test_two_owners_compete_removal_does_not_override_keep():
    cut, cls = vote_agreement([2, 2, 2], [2, 2, 1], [2, 1, 1], [0, 1, 0])
    np.testing.assert_array_equal(cut, [True, False, False])
    assert cls['keep_conflict'].tolist() == [False, True, False]
    assert cls['missing_evidence'].tolist() == [False, False, True]


def test_unassigned_and_missing_are_not_equivalent_to_remove():
    cut, cls = vote_agreement([0, 2, 2, 1], [0, 1, 2, 1],
                              [0, 1, 0, 1], [0, 0, 2, 0])
    assert cut.tolist() == [False, False, False, True]
    assert cls['unassigned'][0] and cls['missing_evidence'][1]


def test_invalid_votes_refused():
    with pytest.raises(ValueError):
        vote_agreement([1], [1], [1], [1])
    with pytest.raises(ValueError):
        vote_agreement([1], [2], [2], [0])


def test_detached_triangles_removed_by_generic_edge_connectivity():
    mesh = trimesh.creation.box()
    faces = np.asarray(mesh.faces).copy()
    vertices = np.vstack((mesh.vertices, np.array([[2., 2., 2.], [2., 3., 2.], [3., 2., 2.]])))
    merged = trimesh.Trimesh(vertices=vertices,
            faces=np.vstack((faces, np.array([[8, 9, 10]]))), process=False)
    kept, debris, stats = keep_largest_edge_component(merged)
    assert len(kept.faces) == 12 and len(debris.faces) == 1
    assert stats['components_before'] == 2
    assert stats['detached_faces'] == 1
    assert len(mesh.faces) == 12  # no input mutation


def test_face_touching_at_vertex_is_not_edge_connected():
    mesh = trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]],
                          faces=[[0,1,2],[0,3,4]], process=False)
    _, counts = _edge_component_labels(mesh.faces)
    assert len(counts) == 2
    with pytest.raises(ValueError, match='main face component'):
        keep_largest_edge_component(mesh)


def test_no_debris_is_identity():
    mesh = trimesh.creation.icosphere(subdivisions=1)
    result, debris, stats = keep_largest_edge_component(mesh)
    assert len(result.faces) == len(mesh.faces)
    assert not len(debris.faces)
    assert stats['components_before'] == 1


def test_direct_reconcile_two_owners_keep_veto_and_agree():
    """Real cutter integration: the second owner must veto a false sidewall cut."""
    from agent_3dvec_udf_cut import AgentUDFCut
    import types
    mesh = trimesh.creation.icosphere(subdivisions=1)
    agent = object.__new__(AgentUDFCut)
    owners = []
    for i in (1, 2):
        owners.append({'item_index':i, 'target_key':'obj|t1',
                       'accessory_key':f'src|{i}', 'avatar_curve':object(),
                       'adapt_arg':{}, 'seed_world':.02, 'grow_world':.01})
    def query(self, mesh, face_ids, **kwargs):
        z = mesh.triangles_center[face_ids, 2]
        if kwargs['accessory_key'] == 'src|2':
            score = np.full(len(face_ids), .005)  # second owner supports KEEP
        else:
            score = np.where(z > 0, .05, .005)
        return score, np.ones(len(face_ids), dtype=bool)
    agent._query_face_scores_for_ids = types.MethodType(query, agent)
    config = {'udf_cut_reconcile_0079':True,
              'udf_cut_direct_min_informative_faces':0,
              'udf_cut_direct_min_informative_fraction':0,
              'udf_cut_combined_direct_preserve_components':False,
              'udf_cut_combined_direct_preserve_existing_boundaries':False,
              'udf_cut_combined_direct_min_faces':1,
              'udf_cut_combined_direct_min_seed_faces':1}
    kept, removed, report, score = agent._direct_udf_cut_combined_mesh(
        mesh,owners=owners,config=config,model_batch_size=32,
        query_batch_size=32,far_world=.1,ownership_mode='udf_support')
    assert len(removed.faces) == 0
    assert report['reconcile_0079']['classes']['keep_conflict'] > 0
    assert report['reconcile_0079']['classes']['agreed_remove'] == 0
    assert len(kept.faces) == len(mesh.faces)

    def agree_query(self, mesh, face_ids, **kwargs):
        z = mesh.triangles_center[face_ids, 2]
        return np.where(z > 0, .05, .005), np.ones(len(face_ids), dtype=bool)
    agent._query_face_scores_for_ids = types.MethodType(agree_query, agent)
    _kept, removed, report, _score = agent._direct_udf_cut_combined_mesh(
        mesh,owners=owners,config=config,model_batch_size=32,
        query_batch_size=32,far_world=.1,ownership_mode='udf_support')
    assert len(removed.faces) > 0
    assert report['reconcile_0079']['classes']['keep_conflict'] == 0


def test_curve_interval_masks_preserve_overlap_when_requested():
    from udf_cut_final_root_leaves import curve_interval_owner_masks
    class Curve:
        class Core:
            key_ts = np.array([0., 1.])
            def interpolate(self, ts):
                return {'points':np.column_stack((ts,np.zeros_like(ts),np.zeros_like(ts)))}
        core = Core()
    mesh = trimesh.creation.box(extents=[.15,.15,.15])
    mesh.apply_translation([.5,0,0])
    curve=Curve()
    owners=[{'item_index':1,'target_key':'o|t1','avatar_curve':curve,
             'adapt_arg':{'src_0':.1,'src_1':.7}},
            {'item_index':2,'target_key':'o|t1','avatar_curve':curve,
             'adapt_arg':{'src_0':.3,'src_1':.9}}]
    exclusive, e = curve_interval_owner_masks(mesh,owners,pad=.01,samples=128)
    overlap, o = curve_interval_owner_masks(mesh,owners,pad=.01,samples=128,
                                          preserve_overlaps=True)
    assert o['overlap_evaluations'] is True
    assert e['overlap_evaluations'] is False
    assert (exclusive[0] & exclusive[1]).sum() == 0
    assert (overlap[0] & overlap[1]).sum() > 0
    assert sum(map(np.sum, overlap)) > sum(map(np.sum, exclusive))
