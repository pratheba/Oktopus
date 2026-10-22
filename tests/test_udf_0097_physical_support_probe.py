"""Synthetic checks for diagnostic-only 0097 physical surface support."""
from pathlib import Path
import sys

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from udf_0097_physical_support_probe import (adjacency_graph_for_window,
    accepted_components, boundary_face_mask, probe_window)


def plane_grid(nx=12, ny=12, z=0.0):
    v = np.array([(float(x), float(y), z) for y in range(ny + 1)
                  for x in range(nx + 1)], dtype=np.float64)
    def vid(x, y):
        return y * (nx + 1) + x
    faces = []
    for y in range(ny):
        for x in range(nx):
            a, b, c, d = vid(x, y), vid(x+1, y), vid(x, y+1), vid(x+1, y+1)
            faces.extend([(a, b, d), (a, d, c)])
    return trimesh.Trimesh(vertices=v, faces=np.asarray(faces), process=False)


def sample_case(mesh):
    n = len(mesh.faces)
    s = np.full(n, 0.895)
    score = np.zeros(n)
    # Four strong-score triangles, separated by weak-score gaps.
    indices = np.array([2 * (6*12+5), 2*(6*12+6),
                        2*(7*12+5), 2*(7*12+6)])
    score[indices] = .1
    return dict(mesh=mesh, arc=s, scores=score, valid=np.ones(n, bool),
                seed=.02, grow=.01, saved_votes=np.zeros(n, bool),
                saved_0081_seed=np.zeros(n, bool),
                boundary=boundary_face_mask(mesh.faces))


def test_radius_zero_preserves_baseline_and_short_radius_connects():
    case = sample_case(plane_grid())
    r = probe_window(case, low=.89, high=.90, radii=(0, .7, 1.5),
                     max_arc_step=.02, min_normal_dot=.5)
    assert r['radii'][0]['extra_supported_faces'] == 0
    assert r['radii'][0]['accepted_components'] == 0
    assert r['radii'][1]['extra_supported_faces'] > 0
    assert r['radii'][2]['supported_faces'] >= r['radii'][1]['supported_faces']
    assert r['radii'][1]['accepted_seed_faces'] >= 0


def test_no_euclidean_jump_to_close_disconnected_surface():
    a = plane_grid()
    b = plane_grid(z=.00001)  # Geometrically very close but not edge-connected.
    both = trimesh.util.concatenate((a, b))
    case = sample_case(both)
    selected, weighted, _ = adjacency_graph_for_window(
        both, case['valid'], case['arc'], max_arc_step=.02,
        min_normal_dot=.5)
    from scipy.sparse.csgraph import dijkstra
    sources = np.flatnonzero(case['scores'][selected] >= .02)
    dist = dijkstra(weighted, indices=sources, min_only=True, limit=2.)
    assert np.all(np.isinf(dist[len(a.faces):]))


def test_invalid_faces_not_crossed():
    mesh = plane_grid()
    case = sample_case(mesh)
    # Mark a vertical face stripe invalid; graph must not create links through it.
    centers = mesh.triangles_center
    case['valid'][(centers[:, 0] > 7) & (centers[:, 0] < 8)] = False
    selected, weighted, _ = adjacency_graph_for_window(
        mesh, case['valid'], case['arc'], max_arc_step=.02,
        min_normal_dot=.5)
    assert len(selected) < len(mesh.faces)
    assert weighted.shape == (len(selected), len(selected))


def test_component_gate_keeps_minimum_faces_and_seeds():
    import scipy.sparse as sparse
    g = sparse.csr_matrix(np.ones((5,5)) - np.eye(5))
    accepted, stats = accepted_components(g, np.ones(5, bool),
        np.ones(5, bool), np.ones(5), np.zeros(5, bool))
    assert not accepted.any()
    assert stats['rejected_counts']['too_few_faces'] == 1


def test_read_case_saved_structure(tmp_path):
    from udf_0097_physical_support_probe import read_case
    import json
    mesh = plane_grid(2, 2)
    source = tmp_path / "body.ply"
    mesh.export(source)
    p = tmp_path / "run" / "_udf_inference_0084" / "oktopus_9_v1"
    (p / "udf_cut_curve_local_0081/body").mkdir(parents=True)
    (p / "udf_cut_evidence_0074/body").mkdir(parents=True)
    (p / "udf_cut_reconcile_0079/body").mkdir(parents=True)
    n = len(mesh.faces)
    report = {"parts": {"body": {"input": str(source),
             "direct_udf": {"owners": [{"item_index": 0,
                 "seed_world": .02, "grow_world": .01}]}}}}
    (p / "riggable_after_udf_cut_report.json").write_text(json.dumps(report))
    np.savez_compressed(p / "udf_cut_curve_local_0081/body/locality.npz",
        face_curve_arclength=np.full(n, .9), seed_faces=np.zeros(n, bool),
        face_centroids_xyz=mesh.triangles_center)
    np.savez_compressed(p / "udf_cut_evidence_0074/body/face_evidence.npz",
        item_00_faces=np.arange(n), item_00_raw_scores=np.full(n, .025),
        item_00_effective_valid=np.ones(n, bool))
    np.savez_compressed(p / "udf_cut_reconcile_0079/body/votes.npz",
        remove_votes=np.ones(n, dtype=np.int64))
    c = read_case(tmp_path / "run", "oktopus_9_v1")
    assert len(c['scores']) == n
    assert np.all(c['valid'])
    assert c['saved_votes'].all()
