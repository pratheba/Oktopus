import json
import types
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "app"))
from agent_3dvec_udf_cut import AgentUDFCut
from udf_cut_final_root_leaves import (
    load_final_root_leaf_meshes,
    export_direct_root_leaf_results,
)
from root_leaf_combination import build_root_leaf_plan


def _plan():
    items = [
        {'name': 'body', 'target_key': 'o|body', 'accessory_key': 'a|body'},
        {'name': 't1_a', 'target_key': 'o|t1', 'accessory_key': 'a|arm', 'blend_group': 't1'},
        {'name': 't1_b', 'target_key': 'o|t1', 'accessory_key': 'a|arm', 'blend_group': 't1'},
        {'name': 't2', 'target_key': 'o|t2', 'accessory_key': 'a|arm'},
    ]
    plan = build_root_leaf_plan(items, {
        'root': 'body',
        'leaves': {'tentacle_t1': {'blend_group': 't1'}, 'tentacle_t2': {'target_key': 'o|t2'}},
    })
    return items, plan


def _fixture(tmp_path):
    items, plan = _plan()
    source = tmp_path / 'sdf' / 'riggable_parts'
    source.mkdir(parents=True)
    meshes = {part: trimesh.creation.icosphere(subdivisions=1) for part in plan.part_names}
    for part, mesh in meshes.items():
        mesh.export(source / (part + '.ply'))
    (source.parent / 'root_leaves_ownership.json').write_text(json.dumps({
        'adaptation_fingerprint': plan.adaptation_fingerprint,
        'part_names': list(plan.part_names),
        'item_to_part': {str(k): v for k, v in sorted(plan.item_to_part.items())},
    }))
    return items, plan, source, meshes


def test_loading_final_parts_does_not_require_individual_meshes(tmp_path):
    _items, plan, source, _meshes = _fixture(tmp_path)
    directory, loaded = load_final_root_leaf_meshes(
        plan=plan, mc_parts_dir=source, root_path=tmp_path, require_manifest=True)
    assert directory == source.resolve()
    assert set(loaded) == set(plan.part_names)


def test_manifest_mismatch_fails_closed(tmp_path):
    _items, plan, source, _meshes = _fixture(tmp_path)
    manifest = source.parent / 'root_leaves_ownership.json'
    data = json.loads(manifest.read_text())
    data['adaptation_fingerprint'] = 'mismatch'
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='fingerprint mismatch'):
        load_final_root_leaf_meshes(
            plan=plan, mc_parts_dir=source, root_path=tmp_path, require_manifest=True)


def test_direct_mode_uses_valid_udf_support_without_individual_refs():
    mesh = trimesh.creation.icosphere(subdivisions=1)
    agent = object.__new__(AgentUDFCut)

    def query(self, mesh_arg, face_ids, **kwargs):
        z = np.asarray(mesh_arg.triangles_center)[face_ids, 2]
        if kwargs['accessory_key'] == 'a':
            return np.full(len(face_ids), .5), (z > 0)
        return np.full(len(face_ids), .1), (z < 0)

    agent._query_face_scores_for_ids = types.MethodType(query, agent)
    owners = [{
        'item_index': i, 'target_key': 'o|t1', 'accessory_key': accessory,
        'avatar_curve': object(), 'adapt_arg': {},
        'seed_world': .4, 'grow_world': .2,
    } for i, accessory in enumerate(('a', 'b'))]
    kept, removed, report, fused = agent._direct_udf_cut_combined_mesh(
        mesh, owners=owners, config={
            'udf_cut_direct_min_informative_faces': 0,
            'udf_cut_combined_direct_min_faces': 1,
            'udf_cut_combined_direct_min_seed_faces': 1,
            'udf_cut_combined_direct_min_seed_fraction': 0,
            'udf_cut_combined_direct_preserve_components': False,
            'udf_cut_combined_direct_preserve_existing_boundaries': False,
        }, model_batch_size=16, query_batch_size=16, far_world=1.0,
        ownership_mode='udf_support',
    )
    assert report['ownership_mode'] == 'udf_support'
    assert report['owned_faces'] > 0
    assert 0 < len(removed.faces) < len(mesh.faces)
    assert np.sum(np.isfinite(fused)) == report["owned_faces"]
    # The low-UDF supported hemisphere is retained.
    assert np.all(np.asarray(removed.triangles_center)[:, 2] > 0)


def test_end_to_end_part_dispatch_stays_inside_owner_part(tmp_path):
    items, plan, source, _ = _fixture(tmp_path)
    agent = object.__new__(AgentUDFCut)
    agent.device = 'cpu'
    agent.feat_dict = {k: i for i, k in enumerate(['o|body', 'o|t1', 'o|t2', 'a|body', 'a|arm'])}
    agent.curve_from_key = lambda key: object()
    queried = []

    def fake_cut(self, mesh, *, owners, **kwargs):
        queried.append([owner['item_index'] for owner in owners])
        empty = trimesh.Trimesh(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=int))
        return mesh, empty, {'removed_faces': 0, 'ownership_mode':kwargs['ownership_mode']}, np.full(len(mesh.faces), np.nan)

    agent._direct_udf_cut_combined_mesh = types.MethodType(fake_cut, agent)
    output = tmp_path / 'output'
    report = agent._cut_final_root_leaf_parts(
        adaptation_items=items, extraction_config={
            'udf_cut_auto_thresholds': {'enabled': False},
            'udf_cut_seed_world':.02, 'udf_cut_grow_world':.01,
        }, root_path=str(tmp_path), output_folder=str(output),
        target_curves={i['target_key']:object() for i in items},
        seam_plan={}, apply_seam_warp=True,
        rig_parts_plan=plan, rig_parts_spec=str(source),
        model_batch_size=32, query_batch_size=32, far_world=.1,
    )
    assert queried == [[0], [1,2], [3]]
    assert set(report['parts']) == set(plan.part_names)
    assert (output/'riggable_after_udf_cut.glb').exists()
    assert (output/'riggable_after_udf_cut_UNSTITCHED_parts.ply').exists()
    assert len(list((output/'riggable_parts_after_udf_cut').glob('*.ply'))) == 3


def test_direct_mode_rejects_unavailable_auto_calibration(tmp_path):
    items, plan, source, _ = _fixture(tmp_path)
    agent = object.__new__(AgentUDFCut)
    with pytest.raises(ValueError, match='requires udf_cut_auto_thresholds.enabled: false'):
        agent._cut_final_root_leaf_parts(
            adaptation_items=items,
            extraction_config={'udf_cut_auto_thresholds': {'enabled': True}},
            root_path=str(tmp_path), output_folder=str(tmp_path/'out'),
            target_curves={}, seam_plan={}, apply_seam_warp=True,
            rig_parts_plan=plan, rig_parts_spec=source,
            model_batch_size=4, query_batch_size=4, far_world=.1,
        )


def test_far_plateau_abstains_instead_of_deleting_final_mesh():
    mesh = trimesh.creation.icosphere(subdivisions=1)
    agent = object.__new__(AgentUDFCut)

    def saturated(self, mesh_arg, face_ids, **kwargs):
        return np.full(len(face_ids), 0.1), np.ones(len(face_ids), dtype=bool)

    agent._query_face_scores_for_ids = types.MethodType(saturated, agent)
    kept, removed, report, _ = agent._direct_udf_cut_combined_mesh(
        mesh,
        owners=[{
            'item_index': 0, 'target_key': 'o|body', 'accessory_key': 'a|body',
            'avatar_curve': object(), 'adapt_arg': {},
            'seed_world': 0.02, 'grow_world': 0.01,
        }],
        config={'udf_cut_direct_min_informative_faces': 2},
        model_batch_size=16, query_batch_size=16, far_world=0.1,
        ownership_mode='udf_support',
    )
    assert len(kept.faces) == len(mesh.faces)
    assert len(removed.faces) == 0
    assert report['unowned_faces'] == len(mesh.faces)
