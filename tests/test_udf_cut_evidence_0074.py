"""0074: diagnostic-only UDF evidence atlas and anatomical ROI reporting."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'app'))
from udf_cut_evidence_0074 import (
    CATEGORIES, classify_face_evidence, export_evidence, summarise_masks,
)


def _owner():
    return {'item_index': 4,
            'face_ids': np.array([2, 3, 4, 5]),
            'raw_scores': np.array([.07, .015, .025, .05]),
            'raw_valid': np.array([True, True, True, True]),
            'effective_valid': np.array([True, True, True, True]),
            'seed_world': .04, 'grow_world': .02,
            'metric_status': 'ok'}


def test_disjoint_coverage_categories():
    category = classify_face_evidence(
        assignments=[0,1,1,1,1,1,1,1],
        raw_valid=[0,0,1,1,1,1,1,1],
        effective_valid=[0,0,0,1,1,1,1,1],
        fused=[np.nan,np.nan,np.nan,np.nan,.3,1.1,2.4,2.5],
        remove_mask=[False,False,False,False,False,False,False,True],
    )
    assert category.tolist() == list(range(8))
    assert len(CATEGORIES) == 8


def test_inconsistent_counts_raise():
    with pytest.raises(ValueError):
        classify_face_evidence(assignments=[0], raw_valid=[1],
            effective_valid=[1], fused=[2.], remove_mask=[True])


def test_user_defined_regions_are_reported_without_anatomical_inference():
    summary = summarise_masks(
        centroids=np.array([[0,0,0],[1,0,0],[2,0,0], [3,0,0],[4,0,0],[5,0,0]]),
        category=np.array([0,1,2,4,6,7], dtype=np.uint8), owners=[],
        regions=[{'name':'neck', 'center':[4.,0.,0.], 'radius_world':1.1}],
        bins=3,
    )
    assert summary['regions'][0]['total_faces'] == 3
    assert summary['regions'][0]['classes']['removed'] == 1
    assert sum(row['total_faces'] for row in summary['axis_bins']['x']) == 6
    assert 'sleeve' not in str(summary['regions'])


def test_export_preserves_original_mesh_and_per_owner_scores(tmp_path):
    mesh = trimesh.creation.box(extents=(1.,1.,1.))
    n = len(mesh.faces)
    original_faces = np.asarray(mesh.faces).copy()
    owner = {
        **_owner(), 'face_ids': np.array([2,3,4,5]),
        'raw_scores': np.array([.07,.015,.025,.05]),
    }
    assigned = np.zeros(n,dtype=np.int16); assigned[2:6] = 1
    raw = assigned.copy(); effective = assigned.copy()
    fused = np.full(n, np.nan)
    fused[2:6] = [2.5, .5, 1.5, 2.6]
    removed = np.zeros(n,dtype=bool); removed[5] = True
    payload = {
        'assignment_count':assigned,'raw_valid_count':raw,
        'effective_valid_count':effective,'fused':fused,
        'remove_mask':removed, 'pre_topology_mask':removed.copy(),
        'owners':[owner],
    }
    summary = export_evidence(mesh, part='body',output_folder=tmp_path,
            payload=payload,
            regions=[{'name':'neck','center':[0,0,0], 'radius_world':4.}])
    assert len(mesh.faces) == n and np.array_equal(original_faces, mesh.faces)
    folder = tmp_path/'udf_cut_evidence_0074'/'body'
    assert (folder/'removed.ply').exists()
    assert (folder/'seed_not_removed.ply').exists()
    assert summary['all']['classes']['removed'] == 1
    assert summary['regions'][0]['total_faces'] == n
    assert summary['owners'][0]['raw_seed_faces'] == 2
    npz = np.load(folder/'face_evidence.npz')
    np.testing.assert_array_equal(npz['item_04_faces'], [2,3,4,5])
    np.testing.assert_allclose(npz['item_04_raw_scores'], [.07,.015,.025,.05],atol=1e-7)
    assert json.loads((folder/'summary.json').read_text())['all']['total_faces'] == n


def test_invalid_anatomical_roi_fails():
    with pytest.raises(ValueError):
        summarise_masks(centroids=np.zeros((4,3)), category=np.zeros(4,dtype=np.uint8),
            owners=[], regions=[{'name':'neck','center':[0,0,0],'radius_world':0.}])


def test_enabling_evidence_does_not_change_cut_geometry_or_scores():
    from agent_3dvec_udf_cut import AgentUDFCut
    import types
    mesh = trimesh.creation.icosphere(subdivisions=1)
    agent = object.__new__(AgentUDFCut)
    def query(self, _mesh, face_ids, **_kwargs):
        centers = np.asarray(_mesh.triangles_center)[face_ids]
        values = np.where(centers[:, 2] > 0, .05, .005)
        return values, np.ones(len(face_ids), dtype=bool)
    agent._query_face_scores_for_ids = types.MethodType(query, agent)
    owners = [{'item_index':0,'target_key':'o|body','accessory_key':'a|body',
               'avatar_curve':object(), 'adapt_arg':{},
               'seed_world':.02, 'grow_world':.01}]
    config = {'udf_cut_direct_min_informative_faces':0,
              'udf_cut_direct_min_informative_fraction':0,
              'udf_cut_combined_direct_preserve_components':False,
              'udf_cut_combined_direct_preserve_existing_boundaries':False,
              'udf_cut_combined_direct_min_seed_faces':1,
              'udf_cut_combined_direct_min_faces':1}
    a = agent._direct_udf_cut_combined_mesh(
        mesh, owners=owners, config=config,
        model_batch_size=16,query_batch_size=16,far_world=.1,ownership_mode='udf_support')
    b = agent._direct_udf_cut_combined_mesh(
        mesh, owners=owners, config={**config,'udf_cut_evidence_0074':True},
        model_batch_size=16,query_batch_size=16,far_world=.1,ownership_mode='udf_support')
    assert np.array_equal(a[0].faces, b[0].faces)
    assert np.array_equal(a[1].faces, b[1].faces)
    np.testing.assert_array_equal(a[3], b[3])
    assert '_evidence_payload_0074' not in a[2]
    payload = b[2].pop('_evidence_payload_0074')
    categories = classify_face_evidence(
        assignments=payload['assignment_count'],
        raw_valid=payload['raw_valid_count'],
        effective_valid=payload['effective_valid_count'],
        fused=payload['fused'],remove_mask=payload['remove_mask'])
    assert int((categories==7).sum())==len(b[1].faces)
    assert payload['owners'][0]['raw_valid'].sum() == len(mesh.faces)
