import sys
import types
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "app"))
from agent_3dvec_udf_cut import AgentUDFCut


def _owner(item_index, mesh, accessory_key, seed=0.4, grow=0.2):
    return {
        "item_index": item_index,
        "target_key": f"target-{item_index}",
        "accessory_key": accessory_key,
        "avatar_curve": object(),
        "adapt_arg": {},
        "seed_world": seed,
        "grow_world": grow,
        "ownership_points": np.concatenate(
            [
                np.asarray(mesh.vertices, dtype=np.float64),
                np.asarray(mesh.triangles_center, dtype=np.float64),
            ],
            axis=0,
        ),
    }


def test_normalize_owner_evidence_hits_common_grow_and_seed_levels():
    got = AgentUDFCut._normalize_owner_evidence(
        np.asarray([0.1, 0.2, 0.3, 0.4, 0.5]),
        grow_world=0.2,
        seed_world=0.4,
    )
    np.testing.assert_allclose(got, [0.5, 1.0, 1.5, 2.0, 2.5])


def test_normalize_owner_evidence_equal_seed_grow_preserves_binary_threshold():
    got = AgentUDFCut._normalize_owner_evidence(
        np.asarray([0.1, 0.2, 0.3]),
        grow_world=0.2,
        seed_world=0.2,
    )
    assert got[0] < 1.0
    assert got[1] >= 2.0
    assert got[2] > got[1]


def test_direct_combined_overlap_keeps_face_when_any_owner_supports_it():
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    agent = object.__new__(AgentUDFCut)

    def fake_query(self, mesh_arg, face_ids, **kwargs):
        # owner-a rejects everything; owner-b supports everything.
        value = 0.5 if kwargs["accessory_key"] == "owner-a" else 0.1
        return (
            np.full(len(face_ids), value, dtype=np.float64),
            np.ones(len(face_ids), dtype=bool),
        )

    agent._query_face_scores_for_ids = types.MethodType(fake_query, agent)
    owners = [
        _owner(0, mesh, "owner-a"),
        _owner(1, mesh, "owner-b"),
    ]
    kept, removed, report, fused = agent._direct_udf_cut_combined_mesh(
        mesh,
        owners=owners,
        config={
            "udf_cut_combined_direct_owner_radius_edges": 100.0,
            "udf_cut_combined_direct_owner_fallback_radius_edges": 100.0,
            "udf_cut_combined_direct_min_faces": 1,
            "udf_cut_combined_direct_min_seed_faces": 1,
            "udf_cut_combined_direct_min_seed_fraction": 0.0,
            "udf_cut_combined_direct_preserve_components": True,
        },
        model_batch_size=32,
        query_batch_size=32,
        far_world=0.1,
    )

    assert len(kept.faces) == len(mesh.faces)
    assert len(removed.faces) == 0
    assert report["removed_faces"] == 0
    assert report["owner_count_histogram"]["2"] == len(mesh.faces)
    assert np.nanmax(fused) < 1.0


def test_direct_combined_single_owner_can_remove_directly_from_final_mesh():
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    agent = object.__new__(AgentUDFCut)

    def fake_query(self, mesh_arg, face_ids, **kwargs):
        # One connected hemisphere is strongly unsupported, the other is kept.
        centers = np.asarray(mesh_arg.triangles_center, dtype=np.float64)[face_ids]
        values = np.where(centers[:, 2] > 0.0, 0.5, 0.1)
        return values, np.ones(len(face_ids), dtype=bool)

    agent._query_face_scores_for_ids = types.MethodType(fake_query, agent)
    owners = [_owner(0, mesh, "owner-a")]
    kept, removed, report, fused = agent._direct_udf_cut_combined_mesh(
        mesh,
        owners=owners,
        config={
            "udf_cut_combined_direct_owner_radius_edges": 100.0,
            "udf_cut_combined_direct_owner_fallback_radius_edges": 100.0,
            "udf_cut_combined_direct_min_faces": 1,
            "udf_cut_combined_direct_min_seed_faces": 1,
            "udf_cut_combined_direct_min_seed_fraction": 0.0,
            "udf_cut_combined_direct_preserve_components": True,
        },
        model_batch_size=32,
        query_batch_size=32,
        far_world=0.1,
    )

    assert 0 < len(removed.faces) < len(mesh.faces)
    assert len(kept.faces) + len(removed.faces) == len(mesh.faces)
    assert report["removed_faces"] == len(removed.faces)
    assert report["after_raw"]["components"] == report["before"]["components"]
    assert np.nanmax(fused) >= 2.0
    assert np.nanmin(fused) < 1.0
