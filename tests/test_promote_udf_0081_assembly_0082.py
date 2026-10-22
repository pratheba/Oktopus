"""0082 regression tests: identity-preserving promotion, no welding or inference."""
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from tools.promote_udf_0081_assembly import promote

PARTS = ('body', 'tentacle_t1', 'tentacle_t2', 'tentacle_t3',
         'tentacle_t4', 'tentacle_t5', 'tentacle_t6', 'tentacle_t8')
ASSIGNMENT = ['body', 'tentacle_t1', 'tentacle_t1', 'tentacle_t1',
              'tentacle_t2', 'tentacle_t2', 'tentacle_t3',
              'tentacle_t4', 'tentacle_t4', 'tentacle_t4',
              'tentacle_t5', 'tentacle_t5', 'tentacle_t5',
              'tentacle_t6', 'tentacle_t8']


def fixture_runs(tmp_path):
    sdf = tmp_path / 'sdf'
    udf = tmp_path / 'udf'
    (sdf / 'riggable_parts').mkdir(parents=True)
    manifest = {
        'schema_version': 49,
        'part_names': list(PARTS),
        'item_to_part': {str(i): part for i, part in enumerate(ASSIGNMENT)},
        'adaptation_fingerprint': 'test_exact_fingerprint',
    }
    (sdf / 'root_leaves_ownership.json').write_text(json.dumps(manifest))
    for index, part in enumerate(PARTS):
        base = trimesh.creation.icosphere(subdivisions=1, radius=.1)
        base.apply_translation([index * .3, 0, 0])
        base.export(sdf / 'riggable_parts' / (part + '.ply'))
        selected = np.zeros(len(base.faces), dtype=bool)
        selected[:8] = True
        keep = base.copy()
        keep.update_faces(~selected)
        keep.remove_unreferenced_vertices()
        folder = udf / 'udf_cut_curve_local_0081' / part
        folder.mkdir(parents=True)
        keep.export(folder / 'proposed_kept_cleaned.ply')
        (folder / 'summary.json').write_text(json.dumps({
            'mode': 'surface_and_arclength_local_0081',
            'selected_faces': 8, 'faces_after_preview': len(keep.faces),
        }))
        np.savez_compressed(folder / 'locality.npz',
                            selected_faces=selected,
                            candidate_faces=selected,
                            seed_faces=selected,
                            face_centroids_xyz=np.asarray(base.triangles_center, dtype=np.float32))
    return sdf, udf, tmp_path / 'assembled'


def test_full_eight_part_promotion_preserves_names_counts_and_manifest(tmp_path):
    sdf, udf, target = fixture_runs(tmp_path)
    report = promote(sdf_run=sdf, udf_run=udf, out=target)
    assert target.is_dir()
    assert (target / 'riggable_after_udf_cut.glb').is_file()
    assert (target / 'riggable_after_udf_cut_UNSTITCHED_parts.ply').is_file()
    assert (target / 'riggable_after_udf_cut_report.json').is_file()
    owners = json.loads((target / 'root_leaves_ownership.json').read_text())
    assert owners['adaptation_fingerprint'] == 'test_exact_fingerprint'
    assert owners['item_to_part'] == {str(i): p for i, p in enumerate(ASSIGNMENT)}
    assert owners['connected_mesh'] is False
    assert owners['stitched_root_leaf_interfaces'] is False
    assert owners['part_names'] == list(PARTS)
    glb = trimesh.load(target / 'riggable_after_udf_cut.glb', force='scene')
    assert set(glb.geometry) == set(PARTS)
    for part in PARTS:
        assert len(glb.geometry[part].faces) == 72
        actual = trimesh.load(target / 'riggable_parts_after_udf_cut' / (part + '.ply'), process=False)
        assert len(actual.faces) == 72
        assert report['parts'][part]['faces_after'] == 72
        assert report['parts'][part]['promotion_0082']['selected_faces_0081'] == 8
    assert report['promotion_0082']['mesh_welding_performed'] is False
    assert report['glb'] == str(target / 'riggable_after_udf_cut.glb')
    assert not str(udf) in str(target / 'nonexistent')
    assert (sdf / 'riggable_parts' / 'body.ply').exists()
    with pytest.raises(FileExistsError):
        promote(sdf_run=sdf, udf_run=udf, out=target)


def test_refuses_mismatched_evidence_before_writing_any_output(tmp_path):
    sdf, udf, target = fixture_runs(tmp_path)
    path = udf / 'udf_cut_curve_local_0081' / 'tentacle_t4' / 'locality.npz'
    with np.load(path) as data:
        arrays = {name: data[name].copy() for name in data.files}
    arrays['face_centroids_xyz'] += 1.0
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match='different SDF geometry'):
        promote(sdf_run=sdf, udf_run=udf, out=target)
    assert not target.exists()


def test_refuses_changed_preview_vertices(tmp_path):
    sdf, udf, target = fixture_runs(tmp_path)
    path = udf / 'udf_cut_curve_local_0081' / 'tentacle_t2' / 'proposed_kept_cleaned.ply'
    mesh = trimesh.load(path, process=False)
    mesh.vertices[0] += [0.02, 0.0, 0.0]
    mesh.export(path)
    with pytest.raises(ValueError, match='preview vertices differ'):
        promote(sdf_run=sdf, udf_run=udf, out=target)
    assert not target.exists()


def test_missing_preview_not_silently_replaced_by_original(tmp_path):
    sdf, udf, target = fixture_runs(tmp_path)
    (udf / 'udf_cut_curve_local_0081' / 'tentacle_t8' / 'proposed_kept_cleaned.ply').unlink()
    with pytest.raises(FileNotFoundError, match='Required mesh missing'):
        promote(sdf_run=sdf, udf_run=udf, out=target)
    assert not target.exists()


def test_refuses_to_write_inside_inputs(tmp_path):
    sdf, udf, _ = fixture_runs(tmp_path)
    with pytest.raises(ValueError, match='nested inside'):
        promote(sdf_run=sdf, udf_run=udf, out=udf / 'promoted')
