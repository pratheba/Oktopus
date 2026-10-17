import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "app"))
from agent_3dvec_udf_cut import AgentUDFCut


def _fast_eval(agent, mesh, scores, usable, *, seed, grow, preserve_boundary=True):
    return agent._evaluate_cut_candidate_fast(
        mesh,
        face_scores=scores,
        usable_faces=usable,
        seed_world=seed,
        grow_world=grow,
        min_faces=3,
        min_seed_faces=1,
        min_seed_fraction=0.03,
        min_area_world2=0.0,
        preserve_existing_boundaries=preserve_boundary,
        reference_tree=None,
        reference_max_distance_world=None,
        adjacency=np.asarray(mesh.face_adjacency, dtype=np.int64),
        face_areas=np.asarray(mesh.area_faces, dtype=np.float64),
        existing_boundary_faces=agent._existing_boundary_faces(mesh),
        centroids=np.asarray(mesh.triangles_center, dtype=np.float64),
        topology=agent._prepare_search_topology(mesh),
    )


def test_fast_candidate_matches_full_cut_and_topology():
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    scores = (centers[:, 2] + 1.0) * 0.5
    usable = np.ones(len(mesh.faces), dtype=bool)

    agent = object.__new__(AgentUDFCut)
    full = agent._cut_mesh(
        mesh,
        face_scores=scores,
        usable_faces=usable,
        seed_world=0.85,
        grow_world=0.70,
        min_faces=3,
        min_seed_faces=1,
        min_seed_fraction=0.03,
        min_area_world2=0.0,
        preserve_existing_boundaries=True,
        reference_points=None,
        reference_max_distance_world=None,
    )
    fast = _fast_eval(agent, mesh, scores, usable, seed=0.85, grow=0.70)

    assert np.array_equal(full[2], fast[0])  # seed mask
    assert np.array_equal(full[3], fast[1])  # grow mask
    assert np.array_equal(full[5], fast[3])  # remove mask
    assert [d.as_dict() for d in full[4]] == [d.as_dict() for d in fast[2]]
    assert agent._topology_stats(full[0]) == fast[4]
    assert agent._topology_stats(full[1]) == fast[5]


def test_fast_candidate_matches_fragmented_open_mesh_case():
    rng = np.random.default_rng(4)
    base = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    keep = np.asarray(base.triangles_center)[:, 2] < 0.75
    mesh = AgentUDFCut._mesh_from_face_mask(base, keep)

    scores = rng.random(len(mesh.faces))
    usable = rng.random(len(mesh.faces)) > 0.03
    agent = object.__new__(AgentUDFCut)

    full = agent._cut_mesh(
        mesh,
        face_scores=scores,
        usable_faces=usable,
        seed_world=0.92,
        grow_world=0.80,
        min_faces=3,
        min_seed_faces=1,
        min_seed_fraction=0.03,
        min_area_world2=0.0,
        preserve_existing_boundaries=True,
        reference_points=None,
        reference_max_distance_world=None,
    )
    fast = _fast_eval(agent, mesh, scores, usable, seed=0.92, grow=0.80)

    assert np.array_equal(full[5], fast[3])
    assert [d.as_dict() for d in full[4]] == [d.as_dict() for d in fast[2]]
    assert agent._topology_stats(full[0]) == fast[4]
    assert agent._topology_stats(full[1]) == fast[5]
