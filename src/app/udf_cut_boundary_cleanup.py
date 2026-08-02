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



def _vertex_neighbors(faces: np.ndarray, n_vertices: int) -> List[np.ndarray]:
    """Build deterministic one-ring vertex adjacency."""
    neighbors = [set() for _ in range(int(n_vertices))]
    for a_raw, b_raw, c_raw in np.asarray(faces, dtype=np.int64):
        a, b, c = int(a_raw), int(b_raw), int(c_raw)
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    return [np.asarray(sorted(items), dtype=np.int64) for items in neighbors]


def _uniform_resample_closed_polyline(points: np.ndarray) -> np.ndarray:
    """Return the same number of samples at uniform closed-loop arclength."""
    points = np.asarray(points, dtype=np.float64)
    count = int(len(points))
    if count < 3:
        return points.copy()

    closed = np.vstack([points, points[0]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    positive = lengths > 1e-15
    if int(np.sum(positive)) < 3:
        return points.copy()

    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    total = float(cumulative[-1])
    if total <= 1e-15:
        return points.copy()

    targets = np.linspace(0.0, total, count, endpoint=False)
    segment = np.searchsorted(cumulative, targets, side="right") - 1
    segment = np.clip(segment, 0, count - 1)
    local = (targets - cumulative[segment]) / (lengths[segment] + 1e-15)
    return (
        (1.0 - local[:, None]) * closed[segment]
        + local[:, None] * closed[segment + 1]
    )


def _fair_closed_polyline(
    points: np.ndarray,
    *,
    iterations: int,
    alpha: float,
) -> np.ndarray:
    """Low-pass a closed loop while preserving its centroid."""
    work = np.asarray(points, dtype=np.float64).copy()
    if len(work) < 3 or int(iterations) <= 0:
        return work

    centroid = work.mean(axis=0)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    for _ in range(int(iterations)):
        average = 0.5 * (np.roll(work, 1, axis=0) + np.roll(work, -1, axis=0))
        work = work + alpha * (average - work)
        work += centroid - work.mean(axis=0)
    return work


def _limit_displacement(
    displacement: np.ndarray,
    limit: np.ndarray,
) -> np.ndarray:
    displacement = np.asarray(displacement, dtype=np.float64).copy()
    limit = np.asarray(limit, dtype=np.float64).reshape(-1)
    lengths = np.linalg.norm(displacement, axis=1)
    active = (limit > 0.0) & (lengths > limit)
    if np.any(active):
        displacement[active] *= (
            limit[active, None] / (lengths[active, None] + 1e-15)
        )
    return displacement


def _affected_face_quality_ok(
    original_vertices: np.ndarray,
    candidate_vertices: np.ndarray,
    faces: np.ndarray,
    affected_faces: np.ndarray,
    *,
    min_area_ratio: float,
    min_normal_dot: float,
) -> Tuple[bool, Dict[str, float]]:
    ids = np.asarray(affected_faces, dtype=np.int64)
    if len(ids) == 0:
        return True, {
            "area_ratio_min": 1.0,
            "normal_dot_min": 1.0,
        }

    tri_old = original_vertices[faces[ids]]
    tri_new = candidate_vertices[faces[ids]]
    cross_old = np.cross(
        tri_old[:, 1] - tri_old[:, 0], tri_old[:, 2] - tri_old[:, 0]
    )
    cross_new = np.cross(
        tri_new[:, 1] - tri_new[:, 0], tri_new[:, 2] - tri_new[:, 0]
    )
    area_old2 = np.linalg.norm(cross_old, axis=1)
    area_new2 = np.linalg.norm(cross_new, axis=1)
    ratio = area_new2 / (area_old2 + 1e-15)

    normal_dot = np.sum(cross_old * cross_new, axis=1) / (
        area_old2 * area_new2 + 1e-15
    )
    ratio_min = float(np.min(ratio)) if len(ratio) else 1.0
    dot_min = float(np.min(normal_dot)) if len(normal_dot) else 1.0
    ok = bool(
        np.all(np.isfinite(candidate_vertices))
        and ratio_min >= float(min_area_ratio)
        and dot_min >= float(min_normal_dot)
    )
    return ok, {
        "area_ratio_min": ratio_min,
        "normal_dot_min": dot_min,
    }


def redistribute_boundary_strip(
    mesh: trimesh.Trimesh,
    *,
    min_edges: int,
    ring_count: int,
    curve_smooth_iterations: int,
    curve_smooth_alpha: float,
    harmonic_iterations: int,
    strip_relax_iterations: int,
    strip_relax_step: float,
    max_boundary_displacement_fraction: float,
    max_strip_displacement_fraction: float,
    min_area_ratio: float,
    min_normal_dot: float,
) -> Tuple[trimesh.Trimesh, List[Dict[str, object]], List[Dict[str, object]]]:
    """Uniformly redistribute large boundaries and relax only a narrow strip.

    This is deliberately topology-preserving.  It changes the selected boundary
    vertices and at most ``ring_count`` adjacent vertex rings.  Every other
    vertex and every face index remain bit-for-bit unchanged.
    """
    loops, invalid = boundary_loops(mesh)
    selected = [loop for loop in loops if loop.edge_count >= int(min_edges)]
    ring_count = max(0, int(ring_count))
    if not selected:
        return mesh.copy(), [], invalid

    original = np.asarray(mesh.vertices, dtype=np.float64).copy()
    faces = np.asarray(mesh.faces, dtype=np.int64).copy()
    neighbors = _vertex_neighbors(faces, len(original))
    normals = _area_weighted_vertex_normals(original, faces)

    # Multi-source graph distance from every selected large boundary.
    distance = np.full(len(original), -1, dtype=np.int32)
    boundary_ids = np.unique(
        np.concatenate([loop.vertices for loop in selected], axis=0)
    )
    distance[boundary_ids] = 0
    frontier = boundary_ids.tolist()
    for depth in range(1, ring_count + 1):
        next_frontier = []
        for vertex_id in frontier:
            for neighbor_id in neighbors[int(vertex_id)]:
                neighbor_id = int(neighbor_id)
                if distance[neighbor_id] < 0:
                    distance[neighbor_id] = depth
                    next_frontier.append(neighbor_id)
        frontier = next_frontier
        if not frontier:
            break

    strip_ids = np.flatnonzero((distance >= 0) & (distance <= ring_count))
    strip_mask = np.zeros(len(original), dtype=bool)
    strip_mask[strip_ids] = True
    boundary_mask = distance == 0

    # Local scale per vertex from the original one-ring edge lengths.
    local_scale = np.zeros(len(original), dtype=np.float64)
    for vertex_id in strip_ids:
        nbr = neighbors[int(vertex_id)]
        if len(nbr):
            lengths = np.linalg.norm(original[nbr] - original[int(vertex_id)], axis=1)
            lengths = lengths[lengths > 1e-15]
            if len(lengths):
                local_scale[int(vertex_id)] = float(np.median(lengths))
    positive_scale = local_scale[local_scale > 0.0]
    fallback_scale = float(np.median(positive_scale)) if len(positive_scale) else 1.0
    local_scale[local_scale <= 0.0] = fallback_scale

    displacement = np.zeros_like(original)
    reports: List[Dict[str, object]] = []

    # Smooth each closed curve, resample it uniformly, and map the same ordered
    # vertex IDs to those targets.  No boundary edges are inserted or removed.
    for loop in selected:
        ids = loop.vertices
        points = original[ids]
        edge_lengths = np.linalg.norm(
            np.roll(points, -1, axis=0) - points, axis=1
        )
        positive = edge_lengths[edge_lengths > 1e-15]
        median_edge = float(np.median(positive)) if len(positive) else fallback_scale

        fair = _fair_closed_polyline(
            points,
            iterations=int(curve_smooth_iterations),
            alpha=float(curve_smooth_alpha),
        )
        # Low-pass filtering tends to shrink a closed curve. Restore its
        # original perimeter before arclength redistribution so the intended
        # collar/sleeve opening does not become smaller merely from cleanup.
        fair_lengths = np.linalg.norm(
            np.roll(fair, -1, axis=0) - fair, axis=1
        )
        fair_perimeter = float(np.sum(fair_lengths))
        original_perimeter = float(np.sum(edge_lengths))
        if fair_perimeter > 1e-15 and original_perimeter > 1e-15:
            center = fair.mean(axis=0)
            fair = center + (fair - center) * (
                original_perimeter / fair_perimeter
            )
        target = _uniform_resample_closed_polyline(fair)

        # The target curve is built only from the original boundary itself, so
        # keep the full 3-D displacement. Projecting it into independently
        # estimated vertex tangent planes makes neighboring samples diverge and
        # reintroduces uneven saw-tooth spacing. Preserve the loop centroid
        # instead; the narrow strip propagation below keeps the surface joined.
        delta = target - points
        delta -= delta.mean(axis=0, keepdims=True)
        limits = np.full(
            len(ids),
            max(0.0, float(max_boundary_displacement_fraction)) * median_edge,
            dtype=np.float64,
        )
        delta = _limit_displacement(delta, limits)
        displacement[ids] = delta

        reports.append(
            {
                **loop.as_dict(),
                "median_edge_before": median_edge,
                "edge_min_before": float(np.min(edge_lengths)),
                "edge_max_before": float(np.max(edge_lengths)),
                "boundary_displacement_mean_requested": float(
                    np.mean(np.linalg.norm(delta, axis=1))
                ),
                "boundary_displacement_max_requested": float(
                    np.max(np.linalg.norm(delta, axis=1))
                ),
            }
        )

    # Diffuse the boundary displacement harmonically through only the selected
    # strip. Vertices outside the strip are fixed zero-displacement constraints.
    harmonic_iterations = max(0, int(harmonic_iterations))
    for _ in range(harmonic_iterations):
        updated = displacement.copy()
        interior_ids = np.flatnonzero(strip_mask & ~boundary_mask)
        for vertex_id in interior_ids:
            nbr = neighbors[int(vertex_id)]
            if len(nbr):
                updated[int(vertex_id)] = np.mean(displacement[nbr], axis=0)
        displacement = updated
        displacement[boundary_ids] = displacement[boundary_ids]

    # Enforce ring-dependent movement limits.  The outermost modified ring gets
    # the smallest allowance, so the strip blends continuously into fixed mesh.
    for depth in range(1, ring_count + 1):
        ids = np.flatnonzero(distance == depth)
        if not len(ids):
            continue
        decay = float(ring_count + 1 - depth) / float(ring_count + 1)
        limits = (
            max(0.0, float(max_strip_displacement_fraction))
            * decay
            * local_scale[ids]
        )
        displacement[ids] = _limit_displacement(displacement[ids], limits)

    candidate = original + displacement

    # Tangentially improve triangles in the strip while freezing both the new
    # boundary curve and every vertex outside the strip.
    relax_step = float(np.clip(strip_relax_step, 0.0, 1.0))
    movable = np.flatnonzero(strip_mask & ~boundary_mask)
    for _ in range(max(0, int(strip_relax_iterations))):
        updated = candidate.copy()
        current_normals = _area_weighted_vertex_normals(candidate, faces)
        for vertex_id in movable:
            nbr = neighbors[int(vertex_id)]
            if not len(nbr):
                continue
            target = np.mean(candidate[nbr], axis=0)
            delta = target - candidate[int(vertex_id)]
            normal = current_normals[int(vertex_id)]
            delta -= float(np.dot(delta, normal)) * normal
            step_limit = (
                max(0.0, float(max_strip_displacement_fraction))
                * 0.25
                * local_scale[int(vertex_id)]
            )
            delta = _limit_displacement(
                delta.reshape(1, 3), np.asarray([step_limit])
            )[0]
            updated[int(vertex_id)] += relax_step * delta
        candidate = updated

    changed = np.linalg.norm(candidate - original, axis=1) > 1e-14
    affected_faces = np.flatnonzero(np.any(changed[faces], axis=1))

    # Back off all local movement together until no affected triangle flips or
    # collapses below the configured area ratio.
    accepted_scale = 0.0
    quality: Dict[str, float] = {
        "area_ratio_min": 0.0,
        "normal_dot_min": -1.0,
    }
    accepted_vertices = original.copy()
    for scale in (1.0, 0.75, 0.5, 0.25, 0.125):
        trial = original + scale * (candidate - original)
        ok, trial_quality = _affected_face_quality_ok(
            original,
            trial,
            faces,
            affected_faces,
            min_area_ratio=float(min_area_ratio),
            min_normal_dot=float(min_normal_dot),
        )
        if ok:
            accepted_scale = float(scale)
            quality = trial_quality
            accepted_vertices = trial
            break

    out = trimesh.Trimesh(
        vertices=accepted_vertices,
        faces=faces,
        process=False,
    )

    final_displacement = np.linalg.norm(accepted_vertices - original, axis=1)
    shared = {
        "changed_vertices": int(np.sum(final_displacement > 1e-14)),
        "boundary_vertices": int(np.sum(boundary_mask)),
        "strip_vertices": int(len(strip_ids)),
        "affected_faces": int(len(affected_faces)),
        "ring_count": int(ring_count),
        "accepted_scale": float(accepted_scale),
        **quality,
        "displacement_mean": float(
            np.mean(final_displacement[final_displacement > 1e-14])
        ) if np.any(final_displacement > 1e-14) else 0.0,
        "displacement_max": float(np.max(final_displacement))
        if len(final_displacement)
        else 0.0,
    }
    for item in reports:
        item.update(shared)
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
    redistribute_boundary: bool = False,
    redistribute_min_edges: int = 12,
    redistribute_ring_count: int = 1,
    redistribute_curve_smooth_iterations: int = 6,
    redistribute_curve_smooth_alpha: float = 0.45,
    redistribute_harmonic_iterations: int = 20,
    redistribute_strip_relax_iterations: int = 4,
    redistribute_strip_relax_step: float = 0.25,
    redistribute_max_boundary_displacement_fraction: float = 2.0,
    redistribute_max_strip_displacement_fraction: float = 0.80,
    redistribute_min_area_ratio: float = 0.10,
    redistribute_min_normal_dot: float = 0.0,
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
    redistributed: List[Dict[str, object]] = []
    redistribute_invalid: List[Dict[str, object]] = []
    if bool(redistribute_boundary):
        work, redistributed, redistribute_invalid = redistribute_boundary_strip(
            work,
            min_edges=int(redistribute_min_edges),
            ring_count=int(redistribute_ring_count),
            curve_smooth_iterations=int(
                redistribute_curve_smooth_iterations
            ),
            curve_smooth_alpha=float(redistribute_curve_smooth_alpha),
            harmonic_iterations=int(redistribute_harmonic_iterations),
            strip_relax_iterations=int(redistribute_strip_relax_iterations),
            strip_relax_step=float(redistribute_strip_relax_step),
            max_boundary_displacement_fraction=float(
                redistribute_max_boundary_displacement_fraction
            ),
            max_strip_displacement_fraction=float(
                redistribute_max_strip_displacement_fraction
            ),
            min_area_ratio=float(redistribute_min_area_ratio),
            min_normal_dot=float(redistribute_min_normal_dot),
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
            "redistribute_boundary": bool(redistribute_boundary),
            "redistribute_min_edges": int(redistribute_min_edges),
            "redistribute_ring_count": int(redistribute_ring_count),
            "redistribute_curve_smooth_iterations": int(
                redistribute_curve_smooth_iterations
            ),
            "redistribute_curve_smooth_alpha": float(
                redistribute_curve_smooth_alpha
            ),
            "redistribute_harmonic_iterations": int(
                redistribute_harmonic_iterations
            ),
            "redistribute_strip_relax_iterations": int(
                redistribute_strip_relax_iterations
            ),
            "redistribute_strip_relax_step": float(
                redistribute_strip_relax_step
            ),
            "redistribute_max_boundary_displacement_fraction": float(
                redistribute_max_boundary_displacement_fraction
            ),
            "redistribute_max_strip_displacement_fraction": float(
                redistribute_max_strip_displacement_fraction
            ),
            "redistribute_min_area_ratio": float(
                redistribute_min_area_ratio
            ),
            "redistribute_min_normal_dot": float(
                redistribute_min_normal_dot
            ),
        },
        "boundary_before": [loop.as_dict() for loop in before_loops],
        "invalid_boundary_before": before_invalid,
        "filled_loops": filled,
        "invalid_boundary_during_fill": fill_invalid,
        "boundary_after_fill": [loop.as_dict() for loop in after_fill_loops],
        "invalid_boundary_after_fill": after_fill_invalid,
        "smoothed_loops": smoothed,
        "invalid_boundary_during_smoothing": smooth_invalid,
        "redistributed_loops": redistributed,
        "invalid_boundary_during_redistribution": redistribute_invalid,
        "boundary_after": [loop.as_dict() for loop in after_loops],
        "invalid_boundary_after": after_invalid,
    }
    return work, report
