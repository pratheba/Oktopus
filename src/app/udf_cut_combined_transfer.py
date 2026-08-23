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
    radius_edge_fraction: float = 2.0,
    seed_advantage_edge_fraction: float = 0.30,
    grow_advantage_edge_fraction: float = 0.00,
    min_faces: int = 8,
    min_seed_faces: int = 2,
    min_seed_fraction: float = 0.05,
    min_valid_samples: int = 2,
) -> Tuple[trimesh.Trimesh, trimesh.Trimesh, Dict[str, object]]:
    """Remove combined-mesh patches that are closer to removed than kept refs.

    Each combined triangle is probed at its centroid and three edge midpoints.
    A reference-distance advantage is measured as ``d_kept - d_removed``.
    Positive values favour removal. Connected grow components are accepted only
    when they contain enough stronger seed faces, preventing isolated speckles.
    """
    if not isinstance(combined_mesh, trimesh.Trimesh) or len(combined_mesh.faces) == 0:
        raise ValueError("combined_mesh must be a non-empty triangle mesh")

    removed_points = _reference_points(removed_reference_meshes)
    kept_points = _reference_points(kept_reference_meshes)
    if len(removed_points) == 0:
        raise ValueError("No removed UDF-cut reference surface points were collected")
    if len(kept_points) == 0:
        raise ValueError("No kept UDF-cut reference surface points were collected")

    from scipy.spatial import cKDTree

    removed_tree = cKDTree(removed_points)
    kept_tree = cKDTree(kept_points)

    edge_scale = _median_edge_length(combined_mesh)
    if edge_scale <= 0.0:
        raise ValueError("Could not determine a positive combined-mesh edge scale")

    radius_world = max(0.0, float(radius_edge_fraction)) * edge_scale
    seed_advantage_world = float(seed_advantage_edge_fraction) * edge_scale
    grow_advantage_world = float(grow_advantage_edge_fraction) * edge_scale
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
    removed_distance, _ = removed_tree.query(flat, k=1)
    kept_distance, _ = kept_tree.query(flat, k=1)
    removed_distance = np.asarray(removed_distance, dtype=np.float64).reshape(-1, 4)
    kept_distance = np.asarray(kept_distance, dtype=np.float64).reshape(-1, 4)

    local = removed_distance <= radius_world
    advantage = kept_distance - removed_distance
    sample_valid_count = np.sum(local, axis=1)

    masked_advantage = np.where(local, advantage, np.nan)
    face_advantage = np.full(len(faces), -np.inf, dtype=np.float64)
    candidate = sample_valid_count >= max(1, int(min_valid_samples))
    if np.any(candidate):
        face_advantage[candidate] = np.nanmedian(masked_advantage[candidate], axis=1)

    face_removed_distance = np.min(removed_distance, axis=1)
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

    remove_mask = np.zeros(len(faces), dtype=bool)
    component_reports: List[Dict[str, object]] = []
    adjacency = np.asarray(combined_mesh.face_adjacency, dtype=np.int64)
    for component_id, face_ids in enumerate(_face_components(grow_mask, adjacency)):
        face_count = int(len(face_ids))
        seed_count = int(np.sum(seed_mask[face_ids]))
        seed_fraction = float(seed_count / max(face_count, 1))
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
                "face_count": face_count,
                "seed_count": seed_count,
                "seed_fraction": seed_fraction,
                "advantage_min": float(np.min(face_advantage[face_ids])),
                "advantage_median": float(np.median(face_advantage[face_ids])),
                "advantage_max": float(np.max(face_advantage[face_ids])),
                "removed_distance_min": float(np.min(face_removed_distance[face_ids])),
                "removed_distance_median": float(np.median(face_removed_distance[face_ids])),
                "accepted": bool(accepted),
                "reason": reason,
            }
        )

    kept_mesh = _mesh_from_face_mask(combined_mesh, ~remove_mask)
    removed_mesh = _mesh_from_face_mask(combined_mesh, remove_mask)
    report = {
        "median_edge_length": float(edge_scale),
        "radius_edge_fraction": float(radius_edge_fraction),
        "radius_world": float(radius_world),
        "seed_advantage_edge_fraction": float(seed_advantage_edge_fraction),
        "seed_advantage_world": float(seed_advantage_world),
        "grow_advantage_edge_fraction": float(grow_advantage_edge_fraction),
        "grow_advantage_world": float(grow_advantage_world),
        "min_valid_samples": int(min_valid_samples),
        "removed_reference_points": int(len(removed_points)),
        "kept_reference_points": int(len(kept_points)),
        "candidate_faces": int(np.sum(candidate)),
        "seed_faces": int(np.sum(seed_mask)),
        "grow_faces": int(np.sum(grow_mask)),
        "removed_faces": int(np.sum(remove_mask)),
        "components": component_reports,
    }
    return kept_mesh, removed_mesh, report
