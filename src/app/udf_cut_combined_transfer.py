"""Transfer individual UDF-cut regions onto a blended combined SDF mesh.

The combined SDF mesh remains the geometry authority. Individual UDF cuts are
used only as references describing which nearby surface regions should be
removed. This avoids trying to boolean-union or re-sign open UDF-cut meshes.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import trimesh


def _median_edge_length(mesh: trimesh.Trimesh) -> float:
    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    if len(edges) == 0:
        return 0.0
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    lengths = np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)
    positive = lengths[np.isfinite(lengths) & (lengths > 1e-15)]
    return float(np.median(positive)) if len(positive) else 0.0


def _reference_points(meshes: Sequence[trimesh.Trimesh]) -> np.ndarray:
    """Collect vertices and triangle centers from non-empty reference meshes."""
    chunks: List[np.ndarray] = []
    for mesh in meshes:
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            continue
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        centers = np.asarray(mesh.triangles_center, dtype=np.float64)
        if len(vertices):
            chunks.append(vertices)
        if len(centers):
            chunks.append(centers)
    if not chunks:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def _face_components(mask: np.ndarray, adjacency: np.ndarray) -> List[np.ndarray]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    active = np.flatnonzero(mask)
    if active.size == 0:
        return []

    parent = np.arange(mask.size, dtype=np.int64)
    rank = np.zeros(mask.size, dtype=np.int8)

    def find(x: int) -> int:
        x = int(x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for a_raw, b_raw in np.asarray(adjacency, dtype=np.int64):
        a, b = int(a_raw), int(b_raw)
        if mask[a] and mask[b]:
            union(a, b)

    groups: Dict[int, List[int]] = {}
    for face_id in active:
        groups.setdefault(find(int(face_id)), []).append(int(face_id))
    return [np.asarray(ids, dtype=np.int64) for ids in groups.values()]


def _mesh_from_face_mask(mesh: trimesh.Trimesh, mask: np.ndarray) -> trimesh.Trimesh:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(mask) != len(faces):
        raise ValueError(f"Face mask length mismatch: {len(mask)} vs {len(faces)}")
    if not np.any(mask):
        return trimesh.Trimesh(
            vertices=np.zeros((0, 3), dtype=np.float64),
            faces=np.zeros((0, 3), dtype=np.int64),
            process=False,
        )
    out = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64).copy(),
        faces=faces[mask].copy(),
        process=False,
    )
    out.remove_unreferenced_vertices()
    return out


def transfer_udf_cut_to_combined_mesh(
    combined_mesh: trimesh.Trimesh,
    *,
    removed_reference_meshes: Sequence[trimesh.Trimesh],
    kept_reference_meshes: Sequence[trimesh.Trimesh],
    reference_group_ids: Sequence[str] | None = None,
    radius_edge_fraction: float = 2.0,
    seed_advantage_edge_fraction: float = 0.30,
    grow_advantage_edge_fraction: float = 0.00,
    min_faces: int = 8,
    min_seed_faces: int = 2,
    min_seed_fraction: float = 0.05,
    min_valid_samples: int = 2,
    owner_margin_edge_fraction: float = 0.0,
) -> Tuple[trimesh.Trimesh, trimesh.Trimesh, Dict[str, object]]:
    """Transfer cuts conservatively using per-group surface ownership.

    Each combined triangle is probed at its centroid and three edge midpoints.
    When ``reference_group_ids`` are provided, a face is first assigned to the
    nearest *kept* reference group.  Removed-vs-kept distance is then evaluated
    only inside that owner group.  This prevents a removed patch from one
    tentacle from cutting a nearby crossing tentacle.

    ``owner_margin_edge_fraction`` rejects ambiguous faces whose best and
    second-best kept groups are nearly tied.  With one group, or with the
    default margin of zero, this reduces to the original transfer behaviour.
    """
    if not isinstance(combined_mesh, trimesh.Trimesh) or len(combined_mesh.faces) == 0:
        raise ValueError("combined_mesh must be a non-empty triangle mesh")
    if len(removed_reference_meshes) != len(kept_reference_meshes):
        raise ValueError(
            "removed/kept reference mesh count mismatch: "
            f"{len(removed_reference_meshes)} vs {len(kept_reference_meshes)}"
        )

    if reference_group_ids is None:
        group_ids = ["__all__"] * len(kept_reference_meshes)
    else:
        group_ids = [str(v) for v in reference_group_ids]
        if len(group_ids) != len(kept_reference_meshes):
            raise ValueError(
                "reference_group_ids length mismatch: "
                f"{len(group_ids)} vs {len(kept_reference_meshes)}"
            )

    # Preserve first-seen order for stable diagnostics.
    group_order: List[str] = []
    for group_id in group_ids:
        if group_id not in group_order:
            group_order.append(group_id)

    grouped_removed: Dict[str, List[trimesh.Trimesh]] = {
        group_id: [] for group_id in group_order
    }
    grouped_kept: Dict[str, List[trimesh.Trimesh]] = {
        group_id: [] for group_id in group_order
    }
    for group_id, removed_mesh, kept_mesh in zip(
        group_ids, removed_reference_meshes, kept_reference_meshes
    ):
        grouped_removed[group_id].append(removed_mesh)
        grouped_kept[group_id].append(kept_mesh)

    from scipy.spatial import cKDTree

    kept_points_by_group: Dict[str, np.ndarray] = {}
    removed_points_by_group: Dict[str, np.ndarray] = {}
    kept_trees = {}
    removed_trees = {}
    for group_id in group_order:
        kept_points = _reference_points(grouped_kept[group_id])
        removed_points = _reference_points(grouped_removed[group_id])
        if len(kept_points) == 0:
            # A group with no retained surface cannot own a combined face.
            continue
        kept_points_by_group[group_id] = kept_points
        removed_points_by_group[group_id] = removed_points
        kept_trees[group_id] = cKDTree(kept_points)
        if len(removed_points):
            removed_trees[group_id] = cKDTree(removed_points)

    active_groups = [g for g in group_order if g in kept_trees]
    if not active_groups:
        raise ValueError("No kept UDF-cut reference surface points were collected")
    total_removed_points = int(
        sum(len(removed_points_by_group.get(g, ())) for g in active_groups)
    )
    if total_removed_points == 0:
        raise ValueError("No removed UDF-cut reference surface points were collected")

    edge_scale = _median_edge_length(combined_mesh)
    if edge_scale <= 0.0:
        raise ValueError("Could not determine a positive combined-mesh edge scale")

    radius_world = max(0.0, float(radius_edge_fraction)) * edge_scale
    seed_advantage_world = float(seed_advantage_edge_fraction) * edge_scale
    grow_advantage_world = float(grow_advantage_edge_fraction) * edge_scale
    owner_margin_world = max(0.0, float(owner_margin_edge_fraction)) * edge_scale
    if grow_advantage_world > seed_advantage_world:
        raise ValueError(
            "grow advantage must be <= seed advantage: "
            f"{grow_advantage_world} > {seed_advantage_world}"
        )

    vertices = np.asarray(combined_mesh.vertices, dtype=np.float64)
    faces = np.asarray(combined_mesh.faces, dtype=np.int64)
    tri = vertices[faces]
    samples = np.stack(
        [
            tri.mean(axis=1),
            0.5 * (tri[:, 0] + tri[:, 1]),
            0.5 * (tri[:, 1] + tri[:, 2]),
            0.5 * (tri[:, 2] + tri[:, 0]),
        ],
        axis=1,
    )
    flat = samples.reshape(-1, 3)
    n_faces = len(faces)

    # Kept-surface proximity determines which tentacle/body group owns a face.
    kept_distance_by_group: Dict[str, np.ndarray] = {}
    removed_distance_by_group: Dict[str, np.ndarray] = {}
    owner_scores = []
    for group_id in active_groups:
        kept_distance, _ = kept_trees[group_id].query(flat, k=1)
        kept_distance = np.asarray(kept_distance, dtype=np.float64).reshape(-1, 4)
        kept_distance_by_group[group_id] = kept_distance
        owner_scores.append(np.median(kept_distance, axis=1))

        removed_tree = removed_trees.get(group_id)
        if removed_tree is None:
            removed_distance = np.full((n_faces, 4), np.inf, dtype=np.float64)
        else:
            removed_distance, _ = removed_tree.query(flat, k=1)
            removed_distance = np.asarray(
                removed_distance, dtype=np.float64
            ).reshape(-1, 4)
        removed_distance_by_group[group_id] = removed_distance

    owner_scores = np.stack(owner_scores, axis=1)
    owner_index = np.argmin(owner_scores, axis=1)
    if len(active_groups) == 1:
        owner_margin = np.full(n_faces, np.inf, dtype=np.float64)
        owner_unambiguous = np.ones(n_faces, dtype=bool)
    else:
        sorted_scores = np.sort(owner_scores, axis=1)
        owner_margin = sorted_scores[:, 1] - sorted_scores[:, 0]
        owner_unambiguous = owner_margin >= owner_margin_world

    face_advantage = np.full(n_faces, -np.inf, dtype=np.float64)
    face_removed_distance = np.full(n_faces, np.inf, dtype=np.float64)
    candidate = np.zeros(n_faces, dtype=bool)
    owner_labels = np.asarray(
        [active_groups[int(i)] for i in owner_index], dtype=object
    )

    min_samples = max(1, int(min_valid_samples))
    for group_idx, group_id in enumerate(active_groups):
        owner_mask = (owner_index == group_idx) & owner_unambiguous
        if not np.any(owner_mask) or group_id not in removed_trees:
            continue

        removed_distance = removed_distance_by_group[group_id]
        kept_distance = kept_distance_by_group[group_id]
        local = removed_distance <= radius_world
        advantage = kept_distance - removed_distance
        sample_valid_count = np.sum(local, axis=1)
        group_candidate = owner_mask & (sample_valid_count >= min_samples)
        if not np.any(group_candidate):
            continue

        masked_advantage = np.where(local, advantage, np.nan)
        face_advantage[group_candidate] = np.nanmedian(
            masked_advantage[group_candidate], axis=1
        )
        face_removed_distance[group_candidate] = np.min(
            removed_distance[group_candidate], axis=1
        )
        candidate[group_candidate] = True

    seed_mask = (
        candidate
        & (face_removed_distance <= radius_world)
        & (face_advantage >= seed_advantage_world)
    )
    grow_mask = (
        candidate
        & (face_removed_distance <= radius_world)
        & (face_advantage >= grow_advantage_world)
    )

    remove_mask = np.zeros(n_faces, dtype=bool)
    component_reports: List[Dict[str, object]] = []
    adjacency = np.asarray(combined_mesh.face_adjacency, dtype=np.int64)
    for component_id, face_ids in enumerate(_face_components(grow_mask, adjacency)):
        face_count = int(len(face_ids))
        seed_count = int(np.sum(seed_mask[face_ids]))
        seed_fraction = float(seed_count / max(face_count, 1))
        component_owners, component_owner_counts = np.unique(
            owner_labels[face_ids], return_counts=True
        )
        dominant_owner = str(component_owners[np.argmax(component_owner_counts)])
        accepted = True
        reason = "accepted"
        if face_count < int(min_faces):
            accepted, reason = False, "too_few_faces"
        elif seed_count < int(min_seed_faces):
            accepted, reason = False, "too_few_seed_faces"
        elif seed_fraction < float(min_seed_fraction):
            accepted, reason = False, "seed_fraction_too_low"
        if accepted:
            remove_mask[face_ids] = True
        component_reports.append(
            {
                "component_id": int(component_id),
                "owner_group": dominant_owner,
                "face_count": face_count,
                "seed_count": seed_count,
                "seed_fraction": seed_fraction,
                "advantage_min": float(np.min(face_advantage[face_ids])),
                "advantage_median": float(np.median(face_advantage[face_ids])),
                "advantage_max": float(np.max(face_advantage[face_ids])),
                "removed_distance_min": float(np.min(face_removed_distance[face_ids])),
                "removed_distance_median": float(
                    np.median(face_removed_distance[face_ids])
                ),
                "accepted": bool(accepted),
                "reason": reason,
            }
        )

    kept_mesh = _mesh_from_face_mask(combined_mesh, ~remove_mask)
    removed_mesh = _mesh_from_face_mask(combined_mesh, remove_mask)
    owner_counts = {
        group_id: int(np.sum(owner_labels == group_id)) for group_id in active_groups
    }
    report = {
        "median_edge_length": float(edge_scale),
        "radius_edge_fraction": float(radius_edge_fraction),
        "radius_world": float(radius_world),
        "seed_advantage_edge_fraction": float(seed_advantage_edge_fraction),
        "seed_advantage_world": float(seed_advantage_world),
        "grow_advantage_edge_fraction": float(grow_advantage_edge_fraction),
        "grow_advantage_world": float(grow_advantage_world),
        "owner_margin_edge_fraction": float(owner_margin_edge_fraction),
        "owner_margin_world": float(owner_margin_world),
        "reference_groups": active_groups,
        "owner_face_counts": owner_counts,
        "ambiguous_owner_faces": int(np.sum(~owner_unambiguous)),
        "min_valid_samples": int(min_valid_samples),
        "removed_reference_points": total_removed_points,
        "kept_reference_points": int(
            sum(len(kept_points_by_group[g]) for g in active_groups)
        ),
        "candidate_faces": int(np.sum(candidate)),
        "seed_faces": int(np.sum(seed_mask)),
        "grow_faces": int(np.sum(grow_mask)),
        "removed_faces": int(np.sum(remove_mask)),
        "components": component_reports,
    }
    return kept_mesh, removed_mesh, report
