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

from udf_cut_boundary_strip import rebuild_boundary_strips


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



def _incident_face_ids(faces: np.ndarray, vertex_ids: np.ndarray) -> np.ndarray:
    """Faces touching any vertex in ``vertex_ids``."""
    ids = np.asarray(vertex_ids, dtype=np.int64).reshape(-1)
    if ids.size == 0:
        return np.zeros((0,), dtype=np.int64)
    mask = np.isin(np.asarray(faces, dtype=np.int64), ids).any(axis=1)
    return np.flatnonzero(mask).astype(np.int64, copy=False)


def _periodic_spline_targets(
    points: np.ndarray,
    *,
    smoothing_fraction: float,
) -> Tuple[np.ndarray, float]:
    """Fit a periodic cubic spline and sample it uniformly by arclength.

    ``smoothing_fraction`` is dimensionless.  The scipy smoothing budget is
    scaled by the loop median edge length, so the same value behaves similarly
    at 128 and 256 resolution.
    """
    from scipy.interpolate import splprep, splev

    points = np.asarray(points, dtype=np.float64)
    n = int(len(points))
    if n < 4:
        return points.copy(), 0.0

    closed = np.vstack([points, points[0]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    perimeter = float(np.sum(seg))
    positive = seg[seg > 1e-15]
    median_edge = float(np.median(positive)) if positive.size else 0.0
    if perimeter <= 1e-15 or median_edge <= 1e-15:
        return points.copy(), median_edge

    u = np.concatenate([[0.0], np.cumsum(seg)]) / perimeter
    smooth_world = max(0.0, float(smoothing_fraction)) * median_edge
    smoothing_budget = float(len(closed)) * smooth_world * smooth_world
    degree = min(3, n - 1)

    tck, _ = splprep(
        closed.T,
        u=u,
        s=smoothing_budget,
        per=True,
        k=degree,
    )

    dense_count = max(256, 12 * n)
    dense_u = np.linspace(0.0, 1.0, dense_count, endpoint=False)
    dense = np.stack(splev(dense_u, tck), axis=1)
    dense_closed = np.vstack([dense, dense[0]])
    dense_seg = np.linalg.norm(np.diff(dense_closed, axis=0), axis=1)
    dense_cum = np.concatenate([[0.0], np.cumsum(dense_seg)])
    dense_total = float(dense_cum[-1])
    if dense_total <= 1e-15:
        return points.copy(), median_edge

    dense_u_closed = np.concatenate([dense_u, [1.0]])
    target_arc = np.linspace(0.0, dense_total, n, endpoint=False)
    target_u = np.interp(target_arc, dense_cum, dense_u_closed)
    targets = np.stack(splev(target_u, tck), axis=1)

    # Prevent a global translation of the opening.
    targets += points.mean(axis=0, keepdims=True) - targets.mean(
        axis=0, keepdims=True
    )
    return targets, median_edge


def spline_smooth_boundary_loops(
    mesh: trimesh.Trimesh,
    *,
    smoothing_fraction: float,
    blend: float,
    min_edges: int,
    max_total_fraction: float,
    min_area_ratio: float,
    min_normal_dot: float,
    min_backtrack_scale: float,
) -> Tuple[trimesh.Trimesh, List[Dict[str, object]], List[Dict[str, object]]]:
    """Spline-fair only large closed boundary loops.

    The vertex count and face connectivity are unchanged.  Each loop is fit by
    a periodic cubic B-spline, sampled uniformly by arclength, and moved only
    in the local tangent plane.  A local triangle-quality guard backtracks the
    move if adjacent faces would collapse or flip.
    """
    loops, invalid = boundary_loops(mesh)
    selected = [loop for loop in loops if loop.edge_count >= int(min_edges)]
    if not selected:
        return mesh.copy(), [], invalid

    blend = float(np.clip(float(blend), 0.0, 1.0))
    max_total_fraction = max(0.0, float(max_total_fraction))
    min_area_ratio = max(0.0, float(min_area_ratio))
    min_normal_dot = float(np.clip(float(min_normal_dot), -1.0, 1.0))
    min_backtrack_scale = float(
        np.clip(float(min_backtrack_scale), 0.0, 1.0)
    )

    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    faces = np.asarray(mesh.faces, dtype=np.int64).copy()
    reports: List[Dict[str, object]] = []

    for loop in selected:
        ids = np.asarray(loop.vertices, dtype=np.int64)
        current = vertices[ids].copy()
        try:
            targets, median_edge = _periodic_spline_targets(
                current,
                smoothing_fraction=float(smoothing_fraction),
            )
        except Exception as exc:
            reports.append(
                {
                    **loop.as_dict(),
                    "accepted": False,
                    "reason": f"spline_fit_failed: {exc}",
                }
            )
            continue

        if median_edge <= 1e-15:
            reports.append(
                {
                    **loop.as_dict(),
                    "accepted": False,
                    "reason": "zero_median_edge_length",
                }
            )
            continue

        normals = _area_weighted_vertex_normals(vertices, faces)[ids]
        displacement = targets - current
        displacement -= (
            np.sum(displacement * normals, axis=1, keepdims=True) * normals
        )
        displacement -= displacement.mean(axis=0, keepdims=True)
        displacement -= (
            np.sum(displacement * normals, axis=1, keepdims=True) * normals
        )
        displacement *= blend

        total_limit = max_total_fraction * median_edge
        if total_limit > 0.0:
            lengths = np.linalg.norm(displacement, axis=1)
            too_large = lengths > total_limit
            if np.any(too_large):
                displacement[too_large] *= (
                    total_limit / (lengths[too_large, None] + 1e-15)
                )

        incident = _incident_face_ids(faces, ids)
        if incident.size == 0:
            reports.append(
                {
                    **loop.as_dict(),
                    "accepted": False,
                    "reason": "no_incident_faces",
                }
            )
            continue

        tri0 = vertices[faces[incident]]
        cross0 = np.cross(tri0[:, 1] - tri0[:, 0], tri0[:, 2] - tri0[:, 0])
        area0 = np.linalg.norm(cross0, axis=1)
        normal0 = np.zeros_like(cross0)
        valid0 = area0 > 1e-15
        normal0[valid0] = cross0[valid0] / area0[valid0, None]

        accepted_scale = 0.0
        accepted_area_ratio = 0.0
        accepted_normal_dot = -1.0
        scale = 1.0
        while scale + 1e-15 >= min_backtrack_scale:
            candidate_vertices = vertices.copy()
            candidate_vertices[ids] = current + scale * displacement
            tri1 = candidate_vertices[faces[incident]]
            cross1 = np.cross(
                tri1[:, 1] - tri1[:, 0], tri1[:, 2] - tri1[:, 0]
            )
            area1 = np.linalg.norm(cross1, axis=1)
            ratio = area1 / (area0 + 1e-15)
            normal1 = np.zeros_like(cross1)
            valid1 = area1 > 1e-15
            normal1[valid1] = cross1[valid1] / area1[valid1, None]
            dots = np.sum(normal0 * normal1, axis=1)

            area_ratio_min = float(np.min(ratio)) if ratio.size else 0.0
            normal_dot_min = float(np.min(dots)) if dots.size else -1.0
            if (
                np.all(valid1)
                and area_ratio_min >= min_area_ratio
                and normal_dot_min >= min_normal_dot
            ):
                accepted_scale = float(scale)
                accepted_area_ratio = area_ratio_min
                accepted_normal_dot = normal_dot_min
                vertices = candidate_vertices
                break
            scale *= 0.5

        final_displacement = np.linalg.norm(
            vertices[ids] - current, axis=1
        )
        reports.append(
            {
                **loop.as_dict(),
                "accepted": bool(accepted_scale > 0.0),
                "reason": (
                    "accepted" if accepted_scale > 0.0 else "quality_guard_rejected"
                ),
                "smoothing_fraction": float(smoothing_fraction),
                "blend": float(blend),
                "median_edge_length": float(median_edge),
                "accepted_scale": float(accepted_scale),
                "area_ratio_min": float(accepted_area_ratio),
                "normal_dot_min": float(accepted_normal_dot),
                "displacement_min": float(np.min(final_displacement)),
                "displacement_mean": float(np.mean(final_displacement)),
                "displacement_max": float(np.max(final_displacement)),
                "affected_faces": int(incident.size),
            }
        )

    out = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return out, reports, invalid

def cleanup_cut_boundary(
    mesh: trimesh.Trimesh,
    *,
    fill_small_holes: bool = True,
    fill_max_edges: int = 24,
    fill_max_perimeter_world: float = 0.08,
    fill_max_span_world: float = 0.04,
    strip_rebuild_enabled: bool = False,
    strip_rebuild_min_edges: int = 100,
    strip_rebuild_min_perimeter_world: float = 0.15,
    strip_rebuild_max_loops: int = 8,
    strip_rebuild_min_rings: int = 2,
    strip_rebuild_max_rings: int = 6,
    strip_rebuild_spline_smoothing_fraction: float = 0.35,
    strip_rebuild_target_edge_scale: float = 1.0,
    strip_rebuild_max_spline_displacement_fraction: float = 1.5,
    strip_rebuild_min_area_world2: float = 1e-14,
    spline_enabled: bool = False,
    spline_smoothing_fraction: float = 0.35,
    spline_blend: float = 1.0,
    spline_min_edges: int = 20,
    spline_max_total_fraction: float = 1.0,
    spline_min_area_ratio: float = 0.10,
    spline_min_normal_dot: float = 0.0,
    spline_min_backtrack_scale: float = 0.0625,
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

    strip_reports: List[Dict[str, object]] = []
    if bool(strip_rebuild_enabled):
        work, strip_reports = rebuild_boundary_strips(
            work,
            [loop.vertices for loop in after_fill_loops],
            min_edges=int(strip_rebuild_min_edges),
            min_perimeter_world=float(strip_rebuild_min_perimeter_world),
            max_loops=int(strip_rebuild_max_loops),
            min_rings=int(strip_rebuild_min_rings),
            max_rings=int(strip_rebuild_max_rings),
            spline_smoothing_fraction=float(
                strip_rebuild_spline_smoothing_fraction
            ),
            target_edge_scale=float(strip_rebuild_target_edge_scale),
            max_spline_displacement_fraction=float(
                strip_rebuild_max_spline_displacement_fraction
            ),
            min_area_world2=float(strip_rebuild_min_area_world2),
        )
    after_strip_loops, after_strip_invalid = boundary_loops(work)

    spline_reports: List[Dict[str, object]] = []
    spline_invalid: List[Dict[str, object]] = []
    if bool(spline_enabled) and not bool(strip_rebuild_enabled):
        work, spline_reports, spline_invalid = spline_smooth_boundary_loops(
            work,
            smoothing_fraction=float(spline_smoothing_fraction),
            blend=float(spline_blend),
            min_edges=int(spline_min_edges),
            max_total_fraction=float(spline_max_total_fraction),
            min_area_ratio=float(spline_min_area_ratio),
            min_normal_dot=float(spline_min_normal_dot),
            min_backtrack_scale=float(spline_min_backtrack_scale),
        )
    after_spline_loops, after_spline_invalid = boundary_loops(work)

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
            "strip_rebuild_enabled": bool(strip_rebuild_enabled),
            "strip_rebuild_min_edges": int(strip_rebuild_min_edges),
            "strip_rebuild_min_perimeter_world": float(
                strip_rebuild_min_perimeter_world
            ),
            "strip_rebuild_max_loops": int(strip_rebuild_max_loops),
            "strip_rebuild_min_rings": int(strip_rebuild_min_rings),
            "strip_rebuild_max_rings": int(strip_rebuild_max_rings),
            "strip_rebuild_spline_smoothing_fraction": float(
                strip_rebuild_spline_smoothing_fraction
            ),
            "strip_rebuild_target_edge_scale": float(
                strip_rebuild_target_edge_scale
            ),
            "strip_rebuild_max_spline_displacement_fraction": float(
                strip_rebuild_max_spline_displacement_fraction
            ),
            "strip_rebuild_min_area_world2": float(
                strip_rebuild_min_area_world2
            ),
            "spline_enabled": bool(spline_enabled),
            "spline_smoothing_fraction": float(spline_smoothing_fraction),
            "spline_blend": float(spline_blend),
            "spline_min_edges": int(spline_min_edges),
            "spline_max_total_fraction": float(spline_max_total_fraction),
            "spline_min_area_ratio": float(spline_min_area_ratio),
            "spline_min_normal_dot": float(spline_min_normal_dot),
            "spline_min_backtrack_scale": float(spline_min_backtrack_scale),
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
        "strip_rebuild_loops": strip_reports,
        "boundary_after_strip_rebuild": [
            loop.as_dict() for loop in after_strip_loops
        ],
        "invalid_boundary_after_strip_rebuild": after_strip_invalid,
        "spline_skipped_due_to_strip_rebuild": bool(
            spline_enabled and strip_rebuild_enabled
        ),
        "spline_loops": spline_reports,
        "invalid_boundary_during_spline": spline_invalid,
        "boundary_after_spline": [
            loop.as_dict() for loop in after_spline_loops
        ],
        "invalid_boundary_after_spline": after_spline_invalid,
        "smoothed_loops": smoothed,
        "invalid_boundary_during_smoothing": smooth_invalid,
        "boundary_after": [loop.as_dict() for loop in after_loops],
        "invalid_boundary_after": after_invalid,
    }
    return work, report
