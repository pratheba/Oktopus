"""0107: route a fused SDF triangle to its saved SDF part, without UDF evidence.

The fused exterior is remeshed by the SDF Boolean, so part ownership cannot
be copied by face index. Use nearby source triangles from riggable_parts/*.ply,
then select the nearest *surface point* with compatible normal. No old 0092
outputs, UDF scores, or cut decisions are read or used here.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def mesh_from_mask(mesh: trimesh.Trimesh, mask: np.ndarray) -> trimesh.Trimesh:
    result = trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                             faces=np.asarray(mesh.faces)[np.asarray(mask, bool)],
                             process=False)
    result.remove_unreferenced_vertices()
    return result


def route_fused_faces_from_sdf_parts(
    fused: trimesh.Trimesh, parts: Mapping[str, trimesh.Trimesh], *,
    distance_edge_factor: float = 1.75, normal_dot: float = 0.55,
    neighbors: int = 12, batch_size: int = 4096,
    max_unmatched_fraction: float = 0.05,
):
    """Return (part index per fused face, report) with fail-closed checks.

    Part indices follow the insertion order of `parts` (the validated manifest
    part order). Unmatched faces receive -1: the direct UDF cutter will preserve
    them as UNKNOWN rather than guessing an owner.
    """
    if not parts:
        raise ValueError('0107 requires at least one SDF root/leaf part')
    if not (np.isfinite(distance_edge_factor) and 0 < distance_edge_factor <= 10):
        raise ValueError('0107 distance_edge_factor must be in (0, 10]')
    if not (np.isfinite(normal_dot) and 0 < normal_dot <= 1):
        raise ValueError('0107 normal_dot must be in (0, 1]')
    if not (1 <= int(neighbors) <= 128 and 1 <= int(batch_size) <= 131072):
        raise ValueError('0107 invalid query size')
    if not (np.isfinite(max_unmatched_fraction) and 0 <= max_unmatched_fraction < 1):
        raise ValueError('0107 max_unmatched_fraction must be in [0, 1)')
    if len(fused.faces) == 0 or not np.isfinite(fused.vertices).all():
        raise ValueError('0107 requires a finite, nonempty fused SDF mesh')

    all_centers, all_triangles, all_normals, all_labels, source_scales = [], [], [], [], []
    names = list(parts)
    for idx, name in enumerate(names):
        part = parts[name]
        if len(part.faces) == 0 or not np.isfinite(part.vertices).all():
            raise ValueError(f'0107 invalid SDF part mesh: {name}')
        edges = np.asarray(part.edges_unique_length)
        positive = edges[np.isfinite(edges) & (edges > 1e-12)]
        if positive.size == 0:
            raise ValueError(f'0107 degenerate SDF part: {name}')
        source_scales.append(float(np.median(positive)))
        all_centers.append(np.asarray(part.triangles_center))
        all_triangles.append(np.asarray(part.triangles))
        all_normals.append(np.asarray(part.face_normals))
        all_labels.append(np.full(len(part.faces), idx, dtype=np.int32))
    target_edges = np.asarray(fused.edges_unique_length)
    positive = target_edges[np.isfinite(target_edges) & (target_edges > 1e-12)]
    if positive.size == 0:
        raise ValueError('0107 degenerate fused SDF')
    tolerance = float(distance_edge_factor) * max(float(np.median(positive)),
                                                   float(np.median(source_scales)))
    centers = np.concatenate(all_centers)
    triangles = np.concatenate(all_triangles)
    normals = np.concatenate(all_normals)
    labels = np.concatenate(all_labels)
    tree = cKDTree(centers)
    k = min(int(neighbors), len(centers))
    n = len(fused.faces)
    routed = np.full(n, -1, dtype=np.int32)
    # Bounded batches avoid allocating N*k triangle arrays for all fused faces.
    for start in range(0, n, int(batch_size)):
        end = min(start + int(batch_size), n)
        points = np.asarray(fused.triangles_center[start:end])
        cand = np.asarray(tree.query(points, k=k)[1], dtype=np.int64).reshape(-1, k)
        nearest = trimesh.triangles.closest_point(
            triangles[cand.reshape(-1)], np.repeat(points, k, axis=0))
        distances = np.linalg.norm(np.repeat(points, k, axis=0) - nearest,
                                   axis=1).reshape(-1, k)
        targets = np.asarray(fused.face_normals[start:end])
        orientation = np.einsum('ijk,ik->ij', normals[cand], targets)
        acceptable = (orientation >= float(normal_dot)) & (distances <= tolerance)
        valid_distance = np.where(acceptable, distances, np.inf)
        which = np.argmin(valid_distance, axis=1)
        good = np.isfinite(valid_distance[np.arange(end - start), which])
        block = routed[start:end]
        block[good] = labels[cand[np.arange(end - start), which][good]]

    unmatched = int(np.count_nonzero(routed == -1))
    unmatched_fraction = unmatched / n
    report = {
        'method': 'direct_SDF_riggable_parts_aligned_surface_correspondence_0107',
        'source': 'SDF riggable_parts and validated root_leaves_ownership.json',
        'uses_0092_evidence': False,
        'fused_faces': int(n),
        'matched_faces': int(n - unmatched),
        'unmatched_faces': unmatched,
        'unmatched_fraction': float(unmatched_fraction),
        'max_unmatched_fraction': float(max_unmatched_fraction),
        'distance_limit_world': tolerance,
        'normal_alignment_min_dot': float(normal_dot),
        'part_face_counts': {name: int(np.count_nonzero(routed == i))
                             for i, name in enumerate(names)},
        'unmatched_policy': 'retain_as_unknown',
    }
    if unmatched_fraction > float(max_unmatched_fraction):
        raise ValueError(
            f'0107 SDF part routing could only match {n-unmatched}/{n} faces '
            f'({unmatched_fraction:.2%} unmatched; allowed {max_unmatched_fraction:.2%}). '
            'Check SDF fused and riggable_parts geometry; do not reuse 0092 results.'
        )
    return routed, report
