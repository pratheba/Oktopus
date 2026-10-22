"""Synthetic boundary-stitch tests: never label concatenated meshes stitched."""
import numpy as np
import pytest
import trimesh
from tools.stitch_udf_0082_assembly import (
    PARTS, stitch_parts, edge_connected_count, mesh_edges, boundary_loops, preflight,
)


def _root_with_seven_holes():
    # Planar grid sheet. Seven non-adjacent missing cells form independent
    # attachment boundaries, in addition to the sheet's outer rim.
    nx, ny = 16, 4
    verts = [[float(x), float(y), 0.] for y in range(ny + 1) for x in range(nx + 1)]
    def idx(x, y): return y * (nx + 1) + x
    missing = {(x, 1) for x in range(1, 14, 2)}
    faces = []
    for y in range(ny):
        for x in range(nx):
            if (x, y) in missing:
                continue
            a, b, c, d = idx(x, y), idx(x + 1, y), idx(x + 1, y + 1), idx(x, y + 1)
            faces += [(a, b, c), (a, c, d)]
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _leaf(x):
    # A square-section tube, open at the body interface but capped at its far end.
    base = np.array([[x, 1, .004], [x+1, 1, .004],
                     [x+1, 2, .004], [x, 2, .004]], dtype=float)
    verts = np.vstack([base, base + [0, 0, .2]])
    faces = []
    for i in range(4):
        j = (i+1) % 4
        faces.extend([(i, j, i+4), (j, j+4, i+4)])
    faces += [(4, 5, 6), (4, 6, 7)]
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def fixture_meshes():
    result = {'body': _root_with_seven_holes()}
    for i, name in enumerate(PARTS[1:]):
        result[name] = _leaf(1 + 2*i)
    return result


def test_stitches_seven_separate_parts_into_one_face_connected_surface():
    meshes = fixture_meshes()
    assert len(boundary_loops(meshes['body'])[0]) == 8  # outer rim + 7 holes
    result, report = stitch_parts(meshes, max_seam_gap=.02, max_bridge_edge=1.5)
    assert len(report['stitches']) == 7
    assert report['edge_connected_components_after'] == 1
    assert edge_connected_count(len(result.faces), result.faces) == 1
    assert report['boundary_edges_after'] < report['boundary_edges_before']
    assert report['nonmanifold_edges_after'] == 0
    assert report['unmatched_body_boundary_loops'] == 1
    assert len(result.faces) == report['faces_before'] + sum(
        step['bridge_faces'] for step in report['stitches'])


def test_fails_closed_when_leaf_has_no_open_interface():
    meshes = fixture_meshes()
    meshes['tentacle_t1'] = trimesh.creation.icosphere(subdivisions=1)
    with pytest.raises(ValueError, match='cannot match|no usable'):
        stitch_parts(meshes, max_seam_gap=.02, max_bridge_edge=1.5)


def test_fails_closed_when_no_matching_root_opening():
    meshes = fixture_meshes()
    meshes['body'] = trimesh.creation.icosphere(subdivisions=1)
    with pytest.raises(ValueError, match='distinct tentacle attachment loops'):
        stitch_parts(meshes)


def test_fails_closed_when_attachment_gap_exceeds_limit():
    meshes = fixture_meshes()
    meshes['tentacle_t1'].apply_translation([0, 0, 3])
    with pytest.raises(ValueError, match='cannot match'):
        stitch_parts(meshes, max_seam_gap=.02, max_bridge_edge=1.5)


def test_rejects_long_bridges_without_changes():
    meshes = fixture_meshes()
    with pytest.raises(ValueError, match='Stitch bridge edge'):
        stitch_parts(meshes, max_seam_gap=.02, max_bridge_edge=.001)
    assert len(meshes['body'].faces) == 2*(16*4-7)


def test_preflight_shows_matching_open_loops():
    report = preflight(fixture_meshes())
    assert report['ready_for_simple_loop_stitch'] is True
    assert report['parts']['body']['simple_closed_boundary_loops'] == 8
    assert report['parts']['tentacle_t1']['simple_closed_boundary_loops'] == 1
    meshes = fixture_meshes()
    meshes['body'] = trimesh.creation.icosphere(subdivisions=1)
    report = preflight(meshes)
    assert report['ready_for_simple_loop_stitch'] is False
