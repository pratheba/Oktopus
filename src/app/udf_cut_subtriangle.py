"""Local sub-triangle clipping for UDF-guided SDF cap removal.

The face-level cutter is retained for robust component selection.  This helper
only refines the accepted component boundary: selected faces and a tiny
conforming neighbourhood are split at the UDF threshold, the high-UDF side is
removed, and everything else remains unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import trimesh


@dataclass(frozen=True)
class SubtriangleCutReport:
    enabled: bool
    threshold_world: float
    expansion_rings: int
    original_vertices: int
    original_faces: int
    accepted_faces: int
    active_faces: int
    expanded_faces: int
    mixed_faces: int
    fully_removed_faces: int
    fully_kept_active_faces: int
    inserted_vertices: int
    kept_faces: int
    removed_faces: int
    unresolved_active_boundary_crossings: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "enabled": bool(self.enabled),
            "threshold_world": float(self.threshold_world),
            "expansion_rings": int(self.expansion_rings),
            "original_vertices": int(self.original_vertices),
            "original_faces": int(self.original_faces),
            "accepted_faces": int(self.accepted_faces),
            "active_faces": int(self.active_faces),
            "expanded_faces": int(self.expanded_faces),
            "mixed_faces": int(self.mixed_faces),
            "fully_removed_faces": int(self.fully_removed_faces),
            "fully_kept_active_faces": int(self.fully_kept_active_faces),
            "inserted_vertices": int(self.inserted_vertices),
            "kept_faces": int(self.kept_faces),
            "removed_faces": int(self.removed_faces),
            "unresolved_active_boundary_crossings": int(
                self.unresolved_active_boundary_crossings
            ),
        }


def _face_adjacency_lists(mesh: trimesh.Trimesh) -> List[List[int]]:
    result: List[List[int]] = [[] for _ in range(len(mesh.faces))]
    for a_raw, b_raw in np.asarray(mesh.face_adjacency, dtype=np.int64):
        a, b = int(a_raw), int(b_raw)
        result[a].append(b)
        result[b].append(a)
    return result


def _edge_incident_faces(faces: np.ndarray) -> Dict[Tuple[int, int], List[int]]:
    result: Dict[Tuple[int, int], List[int]] = {}
    for face_id, tri in enumerate(np.asarray(faces, dtype=np.int64)):
        for a_raw, b_raw in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            a, b = int(a_raw), int(b_raw)
            key = (a, b) if a < b else (b, a)
            result.setdefault(key, []).append(int(face_id))
    return result


def _edge_crosses_threshold(
    value_a: float,
    valid_a: bool,
    value_b: float,
    valid_b: bool,
    threshold: float,
    epsilon: float,
) -> bool:
    if not (bool(valid_a) and bool(valid_b)):
        return False
    da = float(value_a) - float(threshold)
    db = float(value_b) - float(threshold)
    if abs(da) <= float(epsilon) or abs(db) <= float(epsilon):
        return True
    return bool((da < 0.0 and db > 0.0) or (da > 0.0 and db < 0.0))


def _expand_active_faces(
    mesh: trimesh.Trimesh,
    accepted_mask: np.ndarray,
    vertex_values: np.ndarray,
    vertex_valid: np.ndarray,
    *,
    expansion_rings: int,
    threshold: float,
    epsilon: float,
) -> Tuple[np.ndarray, int]:
    """Expand locally and guarantee both sides of every split edge are active."""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    active = np.asarray(accepted_mask, dtype=bool).copy()
    adjacency = _face_adjacency_lists(mesh)

    frontier = np.flatnonzero(active).tolist()
    for _ in range(max(0, int(expansion_rings))):
        next_frontier: List[int] = []
        for face_id in frontier:
            for neighbour in adjacency[int(face_id)]:
                if not active[neighbour]:
                    active[neighbour] = True
                    next_frontier.append(int(neighbour))
        frontier = next_frontier
        if not frontier:
            break

    edge_faces = _edge_incident_faces(faces)
    changed = True
    while changed:
        changed = False
        for (a, b), incident in edge_faces.items():
            if len(incident) < 2:
                continue
            active_count = sum(bool(active[f]) for f in incident)
            if active_count == 0 or active_count == len(incident):
                continue
            if _edge_crosses_threshold(
                vertex_values[a],
                vertex_valid[a],
                vertex_values[b],
                vertex_valid[b],
                threshold,
                epsilon,
            ):
                for face_id in incident:
                    if not active[face_id]:
                        active[face_id] = True
                        changed = True

    unresolved = 0
    for (a, b), incident in edge_faces.items():
        if len(incident) < 2:
            continue
        active_count = sum(bool(active[f]) for f in incident)
        if active_count == 0 or active_count == len(incident):
            continue
        if _edge_crosses_threshold(
            vertex_values[a],
            vertex_valid[a],
            vertex_values[b],
            vertex_valid[b],
            threshold,
            epsilon,
        ):
            unresolved += 1

    return active, int(unresolved)


def _dedupe_polygon(ids: Sequence[int]) -> List[int]:
    cleaned: List[int] = []
    for item in ids:
        item = int(item)
        if not cleaned or cleaned[-1] != item:
            cleaned.append(item)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    return cleaned


def _triangulate_polygon(
    polygon: Sequence[int],
    vertices: Sequence[np.ndarray],
    *,
    min_area_world2: float,
) -> List[Tuple[int, int, int]]:
    ids = _dedupe_polygon(polygon)
    if len(ids) < 3:
        return []

    result: List[Tuple[int, int, int]] = []
    a = int(ids[0])
    for index in range(1, len(ids) - 1):
        b, c = int(ids[index]), int(ids[index + 1])
        if a == b or b == c or c == a:
            continue
        pa = np.asarray(vertices[a], dtype=np.float64)
        pb = np.asarray(vertices[b], dtype=np.float64)
        pc = np.asarray(vertices[c], dtype=np.float64)
        area2 = float(np.linalg.norm(np.cross(pb - pa, pc - pa)))
        if area2 <= 2.0 * float(min_area_world2):
            continue
        result.append((a, b, c))
    return result


def clip_accepted_udf_components(
    mesh: trimesh.Trimesh,
    *,
    accepted_face_mask: np.ndarray,
    vertex_values: np.ndarray,
    vertex_valid: np.ndarray,
    centroid_values: np.ndarray,
    centroid_valid: np.ndarray,
    threshold_world: float,
    expansion_rings: int = 1,
    epsilon: float = 1e-10,
    min_area_world2: float = 1e-14,
) -> Tuple[trimesh.Trimesh, trimesh.Trimesh, Dict[str, object]]:
    """Clip only accepted cap components at a continuous UDF threshold.

    Each active original triangle is split into three centroid triangles.  Each
    of those is clipped into low-UDF (kept) and high-UDF (removed) polygons.
    Intersections on original mesh edges are cached globally, so adjacent faces
    share exactly the same inserted boundary vertex.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    accepted = np.asarray(accepted_face_mask, dtype=bool).reshape(-1)
    vertex_values = np.asarray(vertex_values, dtype=np.float64).reshape(-1)
    vertex_valid = np.asarray(vertex_valid, dtype=bool).reshape(-1)
    centroid_values = np.asarray(centroid_values, dtype=np.float64).reshape(-1)
    centroid_valid = np.asarray(centroid_valid, dtype=bool).reshape(-1)

    if len(accepted) != len(faces):
        raise ValueError(
            f"accepted_face_mask has {len(accepted)} rows for {len(faces)} faces"
        )
    if len(vertex_values) != len(vertices) or len(vertex_valid) != len(vertices):
        raise ValueError("vertex UDF arrays do not match mesh vertices")
    if len(centroid_values) != len(faces) or len(centroid_valid) != len(faces):
        raise ValueError("centroid UDF arrays do not match mesh faces")

    if not np.any(accepted):
        empty = trimesh.Trimesh(
            vertices=np.zeros((0, 3), dtype=np.float64),
            faces=np.zeros((0, 3), dtype=np.int64),
            process=False,
        )
        report = SubtriangleCutReport(
            enabled=True,
            threshold_world=float(threshold_world),
            expansion_rings=int(expansion_rings),
            original_vertices=int(len(vertices)),
            original_faces=int(len(faces)),
            accepted_faces=0,
            active_faces=0,
            expanded_faces=0,
            mixed_faces=0,
            fully_removed_faces=0,
            fully_kept_active_faces=0,
            inserted_vertices=0,
            kept_faces=int(len(faces)),
            removed_faces=0,
            unresolved_active_boundary_crossings=0,
        )
        return mesh.copy(), empty, report.as_dict()

    active, unresolved = _expand_active_faces(
        mesh,
        accepted,
        vertex_values,
        vertex_valid,
        expansion_rings=int(expansion_rings),
        threshold=float(threshold_world),
        epsilon=float(epsilon),
    )

    output_vertices: List[np.ndarray] = [p.copy() for p in vertices]
    scalar_value: Dict[int, float] = {
        int(i): float(vertex_values[i]) for i in range(len(vertices))
    }
    scalar_valid: Dict[int, bool] = {
        int(i): bool(vertex_valid[i]) for i in range(len(vertices))
    }
    centroid_node: Dict[int, int] = {}
    intersection_cache: Dict[Tuple[int, int], int] = {}

    kept_faces: List[Tuple[int, int, int]] = []
    removed_faces: List[Tuple[int, int, int]] = []
    mixed_faces = 0
    fully_removed_faces = 0
    fully_kept_active_faces = 0

    threshold = float(threshold_world)
    epsilon = float(epsilon)

    def get_centroid_node(face_id: int) -> int:
        face_id = int(face_id)
        existing = centroid_node.get(face_id)
        if existing is not None:
            return existing
        tri = faces[face_id]
        point = vertices[tri].mean(axis=0)
        node_id = len(output_vertices)
        output_vertices.append(np.asarray(point, dtype=np.float64))
        value = float(centroid_values[face_id])
        valid = bool(centroid_valid[face_id] and np.isfinite(value))
        if not valid:
            local_values = vertex_values[tri]
            local_valid = vertex_valid[tri] & np.isfinite(local_values)
            if np.any(local_valid):
                value = float(np.mean(local_values[local_valid]))
                valid = True
            else:
                value = threshold - max(epsilon, 1e-12)
                valid = False
        scalar_value[node_id] = value
        scalar_valid[node_id] = valid
        centroid_node[face_id] = node_id
        return node_id

    def node_value(node_id: int) -> float:
        value = float(scalar_value[int(node_id)])
        if scalar_valid.get(int(node_id), False) and np.isfinite(value):
            return value
        # Invalid support samples are kept conservatively.
        return threshold - max(epsilon, 1e-12)

    def intersection_node(a_raw: int, b_raw: int) -> int:
        a, b = int(a_raw), int(b_raw)
        key = (a, b) if a < b else (b, a)
        cached = intersection_cache.get(key)
        if cached is not None:
            return cached

        va, vb = node_value(a), node_value(b)
        denominator = vb - va
        if abs(denominator) <= max(epsilon, 1e-15):
            t = 0.5
        else:
            t = (threshold - va) / denominator
        t = float(np.clip(t, 0.0, 1.0))
        if t <= epsilon:
            intersection_cache[key] = a
            return a
        if t >= 1.0 - epsilon:
            intersection_cache[key] = b
            return b

        point = (1.0 - t) * output_vertices[a] + t * output_vertices[b]
        node_id = len(output_vertices)
        output_vertices.append(np.asarray(point, dtype=np.float64))
        scalar_value[node_id] = threshold
        scalar_valid[node_id] = True
        intersection_cache[key] = node_id
        return node_id

    def clip_polygon(nodes: Sequence[int], keep_low: bool) -> List[int]:
        if not nodes:
            return []
        result: List[int] = []
        previous = int(nodes[-1])
        previous_value = node_value(previous)
        previous_inside = (
            previous_value <= threshold + epsilon
            if keep_low
            else previous_value >= threshold - epsilon
        )

        for current_raw in nodes:
            current = int(current_raw)
            current_value = node_value(current)
            current_inside = (
                current_value <= threshold + epsilon
                if keep_low
                else current_value >= threshold - epsilon
            )
            if current_inside:
                if not previous_inside:
                    result.append(intersection_node(previous, current))
                result.append(current)
            elif previous_inside:
                result.append(intersection_node(previous, current))
            previous = current
            previous_value = current_value
            previous_inside = current_inside
        return _dedupe_polygon(result)

    for face_id, tri_raw in enumerate(faces):
        tri = tuple(int(v) for v in tri_raw)
        if not active[face_id]:
            kept_faces.append(tri)
            continue

        center = get_centroid_node(face_id)
        nodes = [tri[0], tri[1], tri[2], center]
        values = np.asarray([node_value(node) for node in nodes], dtype=np.float64)
        low = values <= threshold + epsilon
        high = values >= threshold - epsilon

        if np.all(low) and not np.any(values > threshold + epsilon):
            kept_faces.append(tri)
            fully_kept_active_faces += 1
            continue
        if np.all(high) and not np.any(values < threshold - epsilon):
            removed_faces.append(tri)
            fully_removed_faces += 1
            continue

        mixed_faces += 1
        subtriangles = (
            (tri[0], tri[1], center),
            (tri[1], tri[2], center),
            (tri[2], tri[0], center),
        )
        for subtriangle in subtriangles:
            low_polygon = clip_polygon(subtriangle, keep_low=True)
            high_polygon = clip_polygon(subtriangle, keep_low=False)
            kept_faces.extend(
                _triangulate_polygon(
                    low_polygon,
                    output_vertices,
                    min_area_world2=float(min_area_world2),
                )
            )
            removed_faces.extend(
                _triangulate_polygon(
                    high_polygon,
                    output_vertices,
                    min_area_world2=float(min_area_world2),
                )
            )

    vertex_array = np.asarray(output_vertices, dtype=np.float64)
    kept_array = (
        np.asarray(kept_faces, dtype=np.int64).reshape(-1, 3)
        if kept_faces
        else np.zeros((0, 3), dtype=np.int64)
    )
    removed_array = (
        np.asarray(removed_faces, dtype=np.int64).reshape(-1, 3)
        if removed_faces
        else np.zeros((0, 3), dtype=np.int64)
    )

    kept_mesh = trimesh.Trimesh(
        vertices=vertex_array.copy(), faces=kept_array, process=False
    )
    removed_mesh = trimesh.Trimesh(
        vertices=vertex_array.copy(), faces=removed_array, process=False
    )
    kept_mesh.remove_unreferenced_vertices()
    removed_mesh.remove_unreferenced_vertices()

    report = SubtriangleCutReport(
        enabled=True,
        threshold_world=threshold,
        expansion_rings=int(expansion_rings),
        original_vertices=int(len(vertices)),
        original_faces=int(len(faces)),
        accepted_faces=int(np.sum(accepted)),
        active_faces=int(np.sum(active)),
        expanded_faces=int(np.sum(active & ~accepted)),
        mixed_faces=int(mixed_faces),
        fully_removed_faces=int(fully_removed_faces),
        fully_kept_active_faces=int(fully_kept_active_faces),
        inserted_vertices=int(len(output_vertices) - len(vertices)),
        kept_faces=int(len(kept_mesh.faces)),
        removed_faces=int(len(removed_mesh.faces)),
        unresolved_active_boundary_crossings=int(unresolved),
    )
    return kept_mesh, removed_mesh, report.as_dict()
