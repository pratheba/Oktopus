"""Conservative cleanup for boundaries created by UDF-guided SDF face cuts.

The cutter deliberately removes complete SDF face patches, so the new opening
follows existing Marching-Cubes edges and can look stair-stepped.  This module
only post-processes that cut result:

* fill closed boundary loops that are small by every configured measure;
* smooth the remaining closed loops with tangential Taubin steps;
* never remesh or smooth the garment interior globally.

Nothing in this module runs unless ``udf_cut_cleanup_boundary`` is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import trimesh


@dataclass(frozen=True)
class BoundaryLoop:
    vertices: np.ndarray
    edge_count: int
    perimeter: float
    span: float
    directed: bool

    def as_dict(self) -> Dict[str, object]:
        return {
            "vertices": int(len(self.vertices)),
            "edge_count": int(self.edge_count),
            "perimeter": float(self.perimeter),
            "span": float(self.span),
            "directed": bool(self.directed),
        }


def _directed_boundary_edges(mesh: trimesh.Trimesh) -> np.ndarray:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) == 0:
        return np.zeros((0, 2), dtype=np.int64)

    directed = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
        axis=0,
    )
    undirected = np.sort(directed, axis=1)
    _unique, inverse, counts = np.unique(
        undirected, axis=0, return_inverse=True, return_counts=True
    )
    return directed[counts[inverse] == 1]


def boundary_loops(mesh: trimesh.Trimesh) -> Tuple[List[BoundaryLoop], List[Dict[str, object]]]:
    """Return directed closed boundary loops and diagnostics for invalid ones."""
    edges = _directed_boundary_edges(mesh)
    if len(edges) == 0:
        return [], []

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    undirected_neighbors: Dict[int, List[int]] = {}
    outgoing: Dict[int, List[int]] = {}
    incoming_count: Dict[int, int] = {}

    for a_raw, b_raw in edges:
        a, b = int(a_raw), int(b_raw)
        undirected_neighbors.setdefault(a, []).append(b)
        undirected_neighbors.setdefault(b, []).append(a)
        outgoing.setdefault(a, []).append(b)
        incoming_count[b] = incoming_count.get(b, 0) + 1
        incoming_count.setdefault(a, incoming_count.get(a, 0))

    unseen = set(undirected_neighbors)
    loops: List[BoundaryLoop] = []
    invalid: List[Dict[str, object]] = []

    while unseen:
        start = min(unseen)
        stack = [start]
        component = []
        unseen.remove(start)
        while stack:
            current = stack.pop()
            component.append(current)
            for nbr in undirected_neighbors.get(current, []):
                if nbr in unseen:
                    unseen.remove(nbr)
                    stack.append(nbr)

        component_set = set(component)
        component_edges = [
            (int(a), int(b))
            for a, b in edges
            if int(a) in component_set and int(b) in component_set
        ]
        degree_ok = all(
            len(undirected_neighbors.get(v, [])) == 2 for v in component
        )
        directed_ok = degree_ok and all(
            len(outgoing.get(v, [])) == 1 and incoming_count.get(v, 0) == 1
            for v in component
        )

        if not directed_ok:
            invalid.append(
                {
                    "vertices": int(len(component)),
                    "edges": int(len(component_edges)),
                    "degree_min": int(
                        min(len(undirected_neighbors.get(v, [])) for v in component)
                    ),
                    "degree_max": int(
                        max(len(undirected_neighbors.get(v, [])) for v in component)
                    ),
                    "reason": "branched_or_inconsistently_oriented_boundary",
                }
            )
            continue

        ordered = []
        current = min(component)
        for _ in range(len(component) + 1):
            if ordered and current == ordered[0]:
                break
            if current in ordered:
                break
            ordered.append(current)
            current = int(outgoing[current][0])

        if current != ordered[0] or len(ordered) != len(component):
            invalid.append(
                {
                    "vertices": int(len(component)),
                    "edges": int(len(component_edges)),
                    "reason": "boundary_did_not_form_one_closed_cycle",
                }
            )
            continue

        ids = np.asarray(ordered, dtype=np.int64)
        points = vertices[ids]
        edge_lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        loops.append(
            BoundaryLoop(
                vertices=ids,
                edge_count=int(len(ids)),
                perimeter=float(np.sum(edge_lengths)),
                span=float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))),
                directed=True,
            )
        )

    loops.sort(key=lambda loop: loop.perimeter, reverse=True)
    return loops, invalid


def boundary_point_cloud(mesh: trimesh.Trimesh) -> trimesh.PointCloud:
    edges = _directed_boundary_edges(mesh)
    if len(edges) == 0:
        points = np.zeros((0, 3), dtype=np.float64)
    else:
        ids = np.unique(edges.reshape(-1))
        points = np.asarray(mesh.vertices, dtype=np.float64)[ids]
    return trimesh.PointCloud(points)


def _is_small_hole(
    loop: BoundaryLoop,
    *,
    max_edges: int,
    max_perimeter_world: float,
    max_span_world: float,
) -> bool:
    # All enabled limits must pass.  This prevents a legitimate narrow opening
    # from being filled merely because one of its measurements is small.
    if max_edges > 0 and loop.edge_count > int(max_edges):
        return False
    if max_perimeter_world > 0.0 and loop.perimeter > float(max_perimeter_world):
        return False
    if max_span_world > 0.0 and loop.span > float(max_span_world):
        return False
    return True


def fill_small_boundary_loops(
    mesh: trimesh.Trimesh,
    *,
    max_edges: int,
    max_perimeter_world: float,
    max_span_world: float,
) -> Tuple[trimesh.Trimesh, List[Dict[str, object]], List[Dict[str, object]]]:
    """Fill only tiny closed loops with a consistently oriented center fan."""
    loops, invalid = boundary_loops(mesh)
    selected = [
        loop
        for loop in loops
        if _is_small_hole(
            loop,
            max_edges=max_edges,
            max_perimeter_world=max_perimeter_world,
            max_span_world=max_span_world,
        )
    ]
    if not selected:
        return mesh.copy(), [], invalid

    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    faces = np.asarray(mesh.faces, dtype=np.int64).copy()
    added_vertices = []
    added_faces = []
    reports: List[Dict[str, object]] = []

    for loop in selected:
        ids = loop.vertices
        points = vertices[ids]
        center_id = len(vertices) + len(added_vertices)
        center = points.mean(axis=0)
        added_vertices.append(center)

        loop_faces = []
        for i, a_raw in enumerate(ids):
            a = int(a_raw)
            b = int(ids[(i + 1) % len(ids)])
            # The existing face traverses the boundary edge a -> b.  The new
            # cap must traverse it in reverse so that the shared edge has two
            # opposite orientations.
            tri = [b, a, center_id]
            pa, pb = vertices[a], vertices[b]
            area2 = np.linalg.norm(np.cross(pa - center, pb - center))
            if area2 > 1e-14:
                loop_faces.append(tri)

        if len(loop_faces) != len(ids):
            # Do not partially fill a loop.  Remove the center staged above.
            added_vertices.pop()
            reports.append(
                {
                    **loop.as_dict(),
                    "filled": False,
                    "reason": "degenerate_center_fan",
                }
            )
            continue

        added_faces.extend(loop_faces)
        reports.append(
            {
                **loop.as_dict(),
                "filled": True,
                "added_faces": int(len(loop_faces)),
            }
        )

    if not added_faces:
        return mesh.copy(), reports, invalid

    out = trimesh.Trimesh(
        vertices=np.vstack([vertices, np.asarray(added_vertices, dtype=np.float64)]),
        faces=np.vstack([faces, np.asarray(added_faces, dtype=np.int64)]),
        process=False,
    )
    return out, reports, invalid


def _area_weighted_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    face_vectors = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals = np.zeros_like(vertices, dtype=np.float64)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_vectors)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-15
    normals[valid] /= lengths[valid, None]
    return normals


def smooth_large_boundary_loops(
    mesh: trimesh.Trimesh,
    *,
    iterations: int,
    lambda_step: float,
    mu_step: float,
    min_edges: int,
    max_step_fraction: float,
    max_total_fraction: float,
) -> Tuple[trimesh.Trimesh, List[Dict[str, object]], List[Dict[str, object]]]:
    """Tangential Taubin smoothing of boundary vertices only.

    Interior vertices and face connectivity are untouched.  Per-step and total
    displacement limits are expressed relative to each loop's median edge
    length, making the defaults independent of mesh resolution.
    """
    loops, invalid = boundary_loops(mesh)
    selected = [loop for loop in loops if loop.edge_count >= int(min_edges)]
    if int(iterations) <= 0 or not selected:
        return mesh.copy(), [], invalid

    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    faces = np.asarray(mesh.faces, dtype=np.int64)
    original = vertices.copy()

    loop_scales: Dict[int, float] = {}
    for loop_index, loop in enumerate(selected):
        points = vertices[loop.vertices]
        lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        positive = lengths[lengths > 1e-15]
        loop_scales[loop_index] = float(np.median(positive)) if len(positive) else 0.0

    for _ in range(int(iterations)):
        for coefficient in (float(lambda_step), float(mu_step)):
            normals = _area_weighted_vertex_normals(vertices, faces)
            updated = vertices.copy()

            for loop_index, loop in enumerate(selected):
                ids = loop.vertices
                scale = loop_scales[loop_index]
                if scale <= 0.0:
                    continue

                current = vertices[ids]
                target = 0.5 * (
                    np.roll(current, 1, axis=0) + np.roll(current, -1, axis=0)
                )
                displacement = target - current
                loop_normals = normals[ids]
                displacement -= (
                    np.sum(displacement * loop_normals, axis=1, keepdims=True)
                    * loop_normals
                )

                # Preserve the loop centroid; remove only high-frequency zigzag.
                displacement -= displacement.mean(axis=0, keepdims=True)
                displacement -= (
                    np.sum(displacement * loop_normals, axis=1, keepdims=True)
                    * loop_normals
                )

                delta = coefficient * displacement
                step_limit = max(0.0, float(max_step_fraction)) * scale
                if step_limit > 0.0:
                    lengths = np.linalg.norm(delta, axis=1)
                    too_large = lengths > step_limit
                    if np.any(too_large):
                        delta[too_large] *= (
                            step_limit / (lengths[too_large, None] + 1e-15)
                        )

                candidate = current + delta
                total_limit = max(0.0, float(max_total_fraction)) * scale
                if total_limit > 0.0:
                    total = candidate - original[ids]
                    total_lengths = np.linalg.norm(total, axis=1)
                    too_large = total_lengths > total_limit
                    if np.any(too_large):
                        total[too_large] *= (
                            total_limit / (total_lengths[too_large, None] + 1e-15)
                        )
                        candidate[too_large] = original[ids][too_large] + total[too_large]

                updated[ids] = candidate

            vertices = updated

    out = trimesh.Trimesh(vertices=vertices, faces=faces.copy(), process=False)
    reports = []
    for loop_index, loop in enumerate(selected):
        displacement = np.linalg.norm(
            vertices[loop.vertices] - original[loop.vertices], axis=1
        )
        reports.append(
            {
                **loop.as_dict(),
                "median_edge_length": float(loop_scales[loop_index]),
                "displacement_min": float(np.min(displacement)),
                "displacement_mean": float(np.mean(displacement)),
                "displacement_max": float(np.max(displacement)),
            }
        )
    return out, reports, invalid


def cleanup_cut_boundary(
    mesh: trimesh.Trimesh,
    *,
    fill_small_holes: bool = True,
    fill_max_edges: int = 24,
    fill_max_perimeter_world: float = 0.08,
    fill_max_span_world: float = 0.04,
    smooth_iterations: int = 8,
    smooth_lambda: float = 0.45,
    smooth_mu: float = -0.47,
    smooth_min_edges: int = 12,
    smooth_max_step_fraction: float = 0.25,
    smooth_max_total_fraction: float = 0.75,
) -> Tuple[trimesh.Trimesh, Dict[str, object]]:
    before_loops, before_invalid = boundary_loops(mesh)
    work = mesh.copy()

    filled: List[Dict[str, object]] = []
    fill_invalid: List[Dict[str, object]] = []
    if bool(fill_small_holes):
        work, filled, fill_invalid = fill_small_boundary_loops(
            work,
            max_edges=int(fill_max_edges),
            max_perimeter_world=float(fill_max_perimeter_world),
            max_span_world=float(fill_max_span_world),
        )

    after_fill_loops, after_fill_invalid = boundary_loops(work)
    work, smoothed, smooth_invalid = smooth_large_boundary_loops(
        work,
        iterations=int(smooth_iterations),
        lambda_step=float(smooth_lambda),
        mu_step=float(smooth_mu),
        min_edges=int(smooth_min_edges),
        max_step_fraction=float(smooth_max_step_fraction),
        max_total_fraction=float(smooth_max_total_fraction),
    )
    after_loops, after_invalid = boundary_loops(work)

    report = {
        "parameters": {
            "fill_small_holes": bool(fill_small_holes),
            "fill_max_edges": int(fill_max_edges),
            "fill_max_perimeter_world": float(fill_max_perimeter_world),
            "fill_max_span_world": float(fill_max_span_world),
            "smooth_iterations": int(smooth_iterations),
            "smooth_lambda": float(smooth_lambda),
            "smooth_mu": float(smooth_mu),
            "smooth_min_edges": int(smooth_min_edges),
            "smooth_max_step_fraction": float(smooth_max_step_fraction),
            "smooth_max_total_fraction": float(smooth_max_total_fraction),
        },
        "boundary_before": [loop.as_dict() for loop in before_loops],
        "invalid_boundary_before": before_invalid,
        "filled_loops": filled,
        "invalid_boundary_during_fill": fill_invalid,
        "boundary_after_fill": [loop.as_dict() for loop in after_fill_loops],
        "invalid_boundary_after_fill": after_fill_invalid,
        "smoothed_loops": smoothed,
        "invalid_boundary_during_smoothing": smooth_invalid,
        "boundary_after": [loop.as_dict() for loop in after_loops],
        "invalid_boundary_after": after_invalid,
    }
    return work, report
