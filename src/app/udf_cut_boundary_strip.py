"""Local boundary-strip rebuild for UDF-cut SDF meshes.

This module replaces only a narrow face strip adjacent to selected open
boundary loops.  It keeps the outer strip boundary fixed, fits a smooth
periodic spline to the inner opening, resamples that opening at approximately
the surrounding mesh spacing, and stitches the two loops with a conforming
triangle strip.

The operation is intentionally opt-in.  When disabled, the existing cutter and
boundary cleanup paths are unchanged.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import trimesh


def _edge_face_map(faces: np.ndarray) -> Dict[Tuple[int, int], List[int]]:
    result: Dict[Tuple[int, int], List[int]] = {}
    for face_id, tri in enumerate(np.asarray(faces, dtype=np.int64)):
        for a_raw, b_raw in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            a, b = int(a_raw), int(b_raw)
            key = (a, b) if a < b else (b, a)
            result.setdefault(key, []).append(int(face_id))
    return result


def _face_adjacency(edge_faces: Dict[Tuple[int, int], List[int]], n_faces: int) -> List[List[int]]:
    adjacency: List[List[int]] = [[] for _ in range(int(n_faces))]
    for incident in edge_faces.values():
        if len(incident) != 2:
            continue
        a, b = int(incident[0]), int(incident[1])
        adjacency[a].append(b)
        adjacency[b].append(a)
    return adjacency


def _ordered_cycle_from_edges(edges: Sequence[Tuple[int, int]]) -> Tuple[np.ndarray, Dict[str, object]]:
    neighbours: Dict[int, List[int]] = {}
    for a_raw, b_raw in edges:
        a, b = int(a_raw), int(b_raw)
        neighbours.setdefault(a, []).append(b)
        neighbours.setdefault(b, []).append(a)

    if not neighbours:
        return np.zeros((0,), dtype=np.int64), {"reason": "empty_interface"}

    degree_values = [len(items) for items in neighbours.values()]
    if any(value != 2 for value in degree_values):
        return np.zeros((0,), dtype=np.int64), {
            "reason": "interface_not_simple_cycle",
            "vertices": int(len(neighbours)),
            "degree_min": int(min(degree_values)),
            "degree_max": int(max(degree_values)),
        }

    unseen = set(neighbours)
    components: List[List[int]] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack = [start]
        component: List[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbour in neighbours[current]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
        components.append(component)

    if len(components) != 1:
        return np.zeros((0,), dtype=np.int64), {
            "reason": "interface_has_multiple_cycles",
            "components": int(len(components)),
            "component_sizes": [int(len(item)) for item in components],
        }

    start = min(neighbours)
    ordered: List[int] = []
    previous = None
    current = start
    for _ in range(len(neighbours) + 1):
        ordered.append(int(current))
        candidates = neighbours[current]
        if previous is None:
            next_vertex = int(candidates[0])
        else:
            next_vertex = int(candidates[0] if candidates[0] != previous else candidates[1])
        previous, current = current, next_vertex
        if current == start:
            break

    if current != start or len(ordered) != len(neighbours):
        return np.zeros((0,), dtype=np.int64), {
            "reason": "interface_ordering_failed",
            "ordered": int(len(ordered)),
            "vertices": int(len(neighbours)),
        }

    return np.asarray(ordered, dtype=np.int64), {
        "reason": "ok",
        "vertices": int(len(ordered)),
    }


def _loop_edge_keys(loop_ids: np.ndarray) -> List[Tuple[int, int]]:
    ids = np.asarray(loop_ids, dtype=np.int64).reshape(-1)
    result: List[Tuple[int, int]] = []
    for index, a_raw in enumerate(ids):
        a = int(a_raw)
        b = int(ids[(index + 1) % len(ids)])
        result.append((a, b) if a < b else (b, a))
    return result


def _find_strip_and_outer_cycle(
    faces: np.ndarray,
    edge_faces: Dict[Tuple[int, int], List[int]],
    face_adjacency: List[List[int]],
    loop_ids: np.ndarray,
    *,
    min_rings: int,
    max_rings: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    boundary_keys = _loop_edge_keys(loop_ids)
    strip = set()
    for key in boundary_keys:
        incident = edge_faces.get(key, [])
        if len(incident) != 1:
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
                {
                    "reason": "boundary_edge_incidence_not_one",
                    "edge": [int(key[0]), int(key[1])],
                    "incidence": int(len(incident)),
                },
            )
        strip.add(int(incident[0]))

    min_rings = max(1, int(min_rings))
    max_rings = max(min_rings, int(max_rings))
    frontier = set(strip)

    for ring_count in range(1, max_rings + 1):
        if ring_count > 1:
            next_frontier = set()
            for face_id in frontier:
                for neighbour in face_adjacency[int(face_id)]:
                    if neighbour not in strip:
                        strip.add(int(neighbour))
                        next_frontier.add(int(neighbour))
            frontier = next_frontier

        if ring_count < min_rings:
            continue

        interface_edges: List[Tuple[int, int]] = []
        for key, incident in edge_faces.items():
            if len(incident) != 2:
                continue
            inside = int(incident[0] in strip) + int(incident[1] in strip)
            if inside == 1:
                interface_edges.append(key)

        outer_ids, cycle_report = _ordered_cycle_from_edges(interface_edges)
        if len(outer_ids):
            return (
                np.asarray(sorted(strip), dtype=np.int64),
                outer_ids,
                {
                    "reason": "ok",
                    "ring_count": int(ring_count),
                    "strip_faces": int(len(strip)),
                    "outer_edges": int(len(interface_edges)),
                    "outer_vertices": int(len(outer_ids)),
                },
            )

    return (
        np.zeros((0,), dtype=np.int64),
        np.zeros((0,), dtype=np.int64),
        {
            "reason": "no_simple_outer_cycle",
            "min_rings": int(min_rings),
            "max_rings": int(max_rings),
        },
    )


def _closed_polyline_arclength(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    points = np.asarray(points, dtype=np.float64)
    closed = np.vstack([points, points[0]])
    segment = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment)])
    return segment, cumulative, float(cumulative[-1])


def _resample_closed_polyline(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    count = max(3, int(count))
    _segment, cumulative, total = _closed_polyline_arclength(points)
    if total <= 1e-15:
        return np.repeat(points[:1], count, axis=0)

    closed = np.vstack([points, points[0]])
    target = np.linspace(0.0, total, count, endpoint=False)
    result = np.empty((count, 3), dtype=np.float64)
    for axis in range(3):
        result[:, axis] = np.interp(target, cumulative, closed[:, axis])
    return result


def _periodic_spline_resample(
    points: np.ndarray,
    *,
    count: int,
    smoothing_world: float,
) -> np.ndarray:
    from scipy.interpolate import splprep, splev

    points = np.asarray(points, dtype=np.float64)
    count = max(3, int(count))
    if len(points) < 4:
        return _resample_closed_polyline(points, count)

    segment, cumulative, total = _closed_polyline_arclength(points)
    if total <= 1e-15:
        return _resample_closed_polyline(points, count)

    closed = np.vstack([points, points[0]])
    parameter = cumulative / total
    degree = min(3, len(points) - 1)
    smoothing_budget = float(len(closed)) * max(0.0, float(smoothing_world)) ** 2
    tck, _ = splprep(
        closed.T,
        u=parameter,
        s=smoothing_budget,
        per=True,
        k=degree,
    )

    dense_count = max(512, 12 * max(len(points), count))
    dense_parameter = np.linspace(0.0, 1.0, dense_count, endpoint=False)
    dense = np.stack(splev(dense_parameter, tck), axis=1)
    return _resample_closed_polyline(dense, count)


def _area_weighted_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    vectors = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals = np.zeros_like(vertices, dtype=np.float64)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], vectors)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-15
    normals[valid] /= lengths[valid, None]
    return normals


def _clamp_rows(vectors: np.ndarray, maximum: float) -> np.ndarray:
    result = np.asarray(vectors, dtype=np.float64).copy()
    if maximum <= 0.0:
        return result
    lengths = np.linalg.norm(result, axis=1)
    mask = lengths > float(maximum)
    if np.any(mask):
        result[mask] *= float(maximum) / (lengths[mask, None] + 1e-15)
    return result


def _prepare_inner_loop(
    vertices: np.ndarray,
    faces: np.ndarray,
    loop_ids: np.ndarray,
    outer_ids: np.ndarray,
    *,
    smoothing_fraction: float,
    target_edge_scale: float,
    max_displacement_fraction: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    inner_points = vertices[np.asarray(loop_ids, dtype=np.int64)]
    outer_points = vertices[np.asarray(outer_ids, dtype=np.int64)]

    outer_lengths = np.linalg.norm(
        np.roll(outer_points, -1, axis=0) - outer_points,
        axis=1,
    )
    positive_outer = outer_lengths[outer_lengths > 1e-15]
    if not len(positive_outer):
        return np.zeros((0, 3), dtype=np.float64), {"reason": "zero_outer_edge_length"}

    outer_median = float(np.median(positive_outer))
    inner_segment, _inner_cumulative, inner_perimeter = _closed_polyline_arclength(inner_points)
    target_edge = max(1e-12, float(target_edge_scale) * outer_median)
    target_count = int(np.clip(round(inner_perimeter / target_edge), 16, max(16, 2 * len(outer_ids))))

    base = _resample_closed_polyline(inner_points, target_count)
    smooth_world = max(0.0, float(smoothing_fraction)) * outer_median
    spline = _periodic_spline_resample(
        inner_points,
        count=target_count,
        smoothing_world=smooth_world,
    )

    # Project spline displacement to the local surface tangent plane, using the
    # nearest original boundary vertex normal as a stable local frame.
    from scipy.spatial import cKDTree

    normals = _area_weighted_vertex_normals(vertices, faces)
    tree = cKDTree(inner_points)
    _distance, nearest = tree.query(base, k=1)
    local_normals = normals[np.asarray(loop_ids, dtype=np.int64)[nearest]]

    displacement = spline - base
    displacement -= (
        np.sum(displacement * local_normals, axis=1, keepdims=True)
        * local_normals
    )
    displacement -= displacement.mean(axis=0, keepdims=True)
    displacement -= (
        np.sum(displacement * local_normals, axis=1, keepdims=True)
        * local_normals
    )
    displacement = _clamp_rows(
        displacement,
        max(0.0, float(max_displacement_fraction)) * outer_median,
    )
    result = base + displacement

    return result, {
        "reason": "ok",
        "original_inner_vertices": int(len(loop_ids)),
        "resampled_inner_vertices": int(len(result)),
        "inner_perimeter": float(inner_perimeter),
        "outer_median_edge": float(outer_median),
        "target_edge": float(target_edge),
        "smoothing_world": float(smooth_world),
        "spline_displacement_mean": float(np.mean(np.linalg.norm(displacement, axis=1))),
        "spline_displacement_max": float(np.max(np.linalg.norm(displacement, axis=1))),
    }


def _rotate_cycle(ids: np.ndarray, start_index: int, reverse: bool) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    if reverse:
        ids = ids[::-1]
    start_index = int(start_index) % len(ids)
    return np.concatenate([ids[start_index:], ids[:start_index]])


def _align_outer_cycle(
    inner_points: np.ndarray,
    outer_ids: np.ndarray,
    vertices: np.ndarray,
) -> np.ndarray:
    from scipy.spatial import cKDTree

    outer_ids = np.asarray(outer_ids, dtype=np.int64)
    best_ids = outer_ids.copy()
    best_score = float("inf")
    sample_count = min(128, max(16, len(inner_points)))
    inner_sample = _resample_closed_polyline(inner_points, sample_count)

    for reverse in (False, True):
        candidate_ids = outer_ids[::-1] if reverse else outer_ids
        candidate_points = vertices[candidate_ids]
        tree = cKDTree(candidate_points)
        _distance, nearest = tree.query(inner_points[:1], k=1)
        rotated_ids = _rotate_cycle(candidate_ids, int(nearest[0]), False)
        outer_sample = _resample_closed_polyline(vertices[rotated_ids], sample_count)
        score = float(np.mean(np.linalg.norm(inner_sample - outer_sample, axis=1)))
        if score < best_score:
            best_score = score
            best_ids = rotated_ids
    return best_ids


def _loop_parameters(points: np.ndarray) -> np.ndarray:
    segment, cumulative, total = _closed_polyline_arclength(points)
    if total <= 1e-15:
        return np.linspace(0.0, 1.0, len(points) + 1)
    return cumulative / total


def _zipper_faces(inner_ids: np.ndarray, outer_ids: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    inner_ids = np.asarray(inner_ids, dtype=np.int64)
    outer_ids = np.asarray(outer_ids, dtype=np.int64)
    inner_parameter = _loop_parameters(vertices[inner_ids])
    outer_parameter = _loop_parameters(vertices[outer_ids])

    i = 0
    j = 0
    triangles: List[Tuple[int, int, int]] = []
    tolerance = 1e-12
    while i < len(inner_ids) or j < len(outer_ids):
        next_inner = inner_parameter[i + 1] if i < len(inner_ids) else float("inf")
        next_outer = outer_parameter[j + 1] if j < len(outer_ids) else float("inf")
        inner_current = int(inner_ids[i % len(inner_ids)])
        outer_current = int(outer_ids[j % len(outer_ids)])

        if next_inner < next_outer - tolerance:
            inner_next = int(inner_ids[(i + 1) % len(inner_ids)])
            triangles.append((inner_current, inner_next, outer_current))
            i += 1
        elif next_outer < next_inner - tolerance:
            outer_next = int(outer_ids[(j + 1) % len(outer_ids)])
            triangles.append((inner_current, outer_next, outer_current))
            j += 1
        else:
            inner_next = int(inner_ids[(i + 1) % len(inner_ids)])
            outer_next = int(outer_ids[(j + 1) % len(outer_ids)])
            triangles.append((inner_current, inner_next, outer_current))
            triangles.append((inner_next, outer_next, outer_current))
            i += 1
            j += 1

    return np.asarray(triangles, dtype=np.int64).reshape(-1, 3)


def _orient_strip_faces(
    triangles: np.ndarray,
    outer_ids: np.ndarray,
    faces: np.ndarray,
    edge_faces: Dict[Tuple[int, int], List[int]],
    strip_faces: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Orient the new strip opposite the retained face on every outer edge."""
    triangles = np.asarray(triangles, dtype=np.int64).copy()
    outer_ids = np.asarray(outer_ids, dtype=np.int64)
    strip_set = set(int(item) for item in np.asarray(strip_faces, dtype=np.int64))

    retained_direction: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for index, a_raw in enumerate(outer_ids):
        a = int(a_raw)
        b = int(outer_ids[(index + 1) % len(outer_ids)])
        key = (a, b) if a < b else (b, a)
        incident = edge_faces.get(key, [])
        retained = [int(face_id) for face_id in incident if int(face_id) not in strip_set]
        if len(retained) != 1:
            continue
        tri = np.asarray(faces[int(retained[0])], dtype=np.int64)
        direction = None
        for u_raw, v_raw in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            u, v = int(u_raw), int(v_raw)
            if (u == a and v == b) or (u == b and v == a):
                direction = (u, v)
                break
        if direction is not None:
            retained_direction[key] = direction

    def orientation_counts(candidate: np.ndarray) -> Tuple[int, int, int]:
        same = 0
        opposite = 0
        seen = 0
        for tri in np.asarray(candidate, dtype=np.int64):
            for u_raw, v_raw in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                u, v = int(u_raw), int(v_raw)
                key = (u, v) if u < v else (v, u)
                retained = retained_direction.get(key)
                if retained is None:
                    continue
                seen += 1
                if retained == (u, v):
                    same += 1
                else:
                    opposite += 1
        return same, opposite, seen

    same, opposite, seen = orientation_counts(triangles)
    globally_flipped = False
    if same > opposite:
        triangles[:, [1, 2]] = triangles[:, [2, 1]]
        globally_flipped = True
        same, opposite, seen = orientation_counts(triangles)

    return triangles, {
        "outer_edges_checked": int(seen),
        "outer_edges_same_orientation": int(same),
        "outer_edges_opposite_orientation": int(opposite),
        "globally_flipped": bool(globally_flipped),
    }

def _edge_count_stats(faces: np.ndarray) -> Dict[str, int]:
    faces = np.asarray(faces, dtype=np.int64)
    if not len(faces):
        return {"boundary_edges": 0, "nonmanifold_edges": 0}
    edges = np.sort(
        np.concatenate(
            [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
            axis=0,
        ),
        axis=1,
    )
    _unique, counts = np.unique(edges, axis=0, return_counts=True)
    return {
        "boundary_edges": int(np.sum(counts == 1)),
        "nonmanifold_edges": int(np.sum(counts > 2)),
    }


def rebuild_boundary_strips(
    mesh: trimesh.Trimesh,
    loop_vertex_sequences: Sequence[np.ndarray],
    *,
    min_edges: int = 100,
    min_perimeter_world: float = 0.15,
    max_loops: int = 8,
    min_rings: int = 2,
    max_rings: int = 4,
    spline_smoothing_fraction: float = 0.35,
    target_edge_scale: float = 1.0,
    max_spline_displacement_fraction: float = 1.5,
    min_area_world2: float = 1e-14,
) -> Tuple[trimesh.Trimesh, List[Dict[str, object]]]:
    """Rebuild narrow strips around selected simple boundary loops.

    The input ``loop_vertex_sequences`` must contain ordered, closed boundary
    loops from the same mesh.  Loops are processed in the supplied order,
    normally largest perimeter first.  Overlapping strips are rejected rather
    than modifying the same surface region twice.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    faces = np.asarray(mesh.faces, dtype=np.int64).copy()
    edge_faces = _edge_face_map(faces)
    face_adjacency = _face_adjacency(edge_faces, len(faces))

    reports: List[Dict[str, object]] = []
    plans: List[Dict[str, object]] = []
    occupied_faces = set()

    for loop_ids_raw in list(loop_vertex_sequences)[: max(0, int(max_loops))]:
        loop_ids = np.asarray(loop_ids_raw, dtype=np.int64).reshape(-1)
        points = vertices[loop_ids]
        perimeter = float(
            np.sum(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1))
        )
        base_report: Dict[str, object] = {
            "edge_count": int(len(loop_ids)),
            "perimeter": float(perimeter),
            "accepted": False,
        }
        if len(loop_ids) < int(min_edges):
            base_report["reason"] = "too_few_edges"
            reports.append(base_report)
            continue
        if perimeter < float(min_perimeter_world):
            base_report["reason"] = "perimeter_too_small"
            reports.append(base_report)
            continue

        strip_faces, outer_ids, strip_report = _find_strip_and_outer_cycle(
            faces,
            edge_faces,
            face_adjacency,
            loop_ids,
            min_rings=int(min_rings),
            max_rings=int(max_rings),
        )
        base_report.update(strip_report)
        if not len(strip_faces):
            base_report["reason"] = strip_report.get("reason", "strip_failed")
            reports.append(base_report)
            continue

        overlap = occupied_faces.intersection(int(item) for item in strip_faces)
        if overlap:
            base_report["reason"] = "strip_overlaps_previous_loop"
            base_report["overlap_faces"] = int(len(overlap))
            reports.append(base_report)
            continue

        inner_points, inner_report = _prepare_inner_loop(
            vertices,
            faces,
            loop_ids,
            outer_ids,
            smoothing_fraction=float(spline_smoothing_fraction),
            target_edge_scale=float(target_edge_scale),
            max_displacement_fraction=float(max_spline_displacement_fraction),
        )
        base_report.update(inner_report)
        if not len(inner_points):
            base_report["reason"] = inner_report.get("reason", "inner_loop_failed")
            reports.append(base_report)
            continue

        aligned_outer = _align_outer_cycle(inner_points, outer_ids, vertices)
        plans.append(
            {
                "loop_ids": loop_ids,
                "strip_faces": strip_faces,
                "outer_ids": aligned_outer,
                "inner_points": inner_points,
                "report": base_report,
            }
        )
        occupied_faces.update(int(item) for item in strip_faces)

    if not plans:
        return mesh.copy(), reports

    kept_face_mask = np.ones(len(faces), dtype=bool)
    kept_face_mask[np.asarray(sorted(occupied_faces), dtype=np.int64)] = False
    new_vertices: List[np.ndarray] = [item.copy() for item in vertices]
    for plan in plans:
        inner_points = np.asarray(plan["inner_points"], dtype=np.float64)
        inner_start = len(new_vertices)
        new_vertices.extend(point.copy() for point in inner_points)
        inner_ids = np.arange(
            inner_start,
            inner_start + len(inner_points),
            dtype=np.int64,
        )
        outer_ids = np.asarray(plan["outer_ids"], dtype=np.int64)
        working_vertices = np.asarray(new_vertices, dtype=np.float64)
        zipper = _zipper_faces(inner_ids, outer_ids, working_vertices)
        zipper, orientation_report = _orient_strip_faces(
            zipper,
            outer_ids,
            faces,
            edge_faces,
            np.asarray(plan["strip_faces"], dtype=np.int64),
        )

        tri = working_vertices[zipper]
        area2 = np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
            axis=1,
        )
        valid = area2 > 2.0 * float(min_area_world2)
        zipper = zipper[valid]
        if len(zipper) == 0:
            plan["report"]["reason"] = "all_rebuilt_faces_degenerate"
            continue
        if (
            int(orientation_report["outer_edges_checked"]) != len(outer_ids)
            or int(orientation_report["outer_edges_same_orientation"]) != 0
        ):
            plan["report"]["reason"] = "outer_edge_orientation_guard_rejected"
            plan["report"].update(orientation_report)
            continue
        plan["zipper_faces"] = zipper
        plan["report"].update(orientation_report)
        plan["report"].update(
            {
                "accepted": True,
                "reason": "accepted",
                "added_vertices": int(len(inner_points)),
                "added_faces": int(len(zipper)),
                "dropped_degenerate_faces": int(np.sum(~valid)),
            }
        )

    accepted_plans = [plan for plan in plans if plan["report"].get("accepted")]
    if not accepted_plans:
        reports.extend(plan["report"] for plan in plans)
        return mesh.copy(), reports

    # Only delete strips whose replacement was accepted.
    accepted_strip_faces = np.concatenate(
        [np.asarray(plan["strip_faces"], dtype=np.int64) for plan in accepted_plans]
    )
    final_keep = np.ones(len(faces), dtype=bool)
    final_keep[accepted_strip_faces] = False
    final_faces = [faces[final_keep]]
    accepted_face_arrays = [
        np.asarray(plan["zipper_faces"], dtype=np.int64)
        for plan in accepted_plans
    ]
    final_faces.extend(accepted_face_arrays)

    out = trimesh.Trimesh(
        vertices=np.asarray(new_vertices, dtype=np.float64),
        faces=np.vstack(final_faces).astype(np.int64, copy=False),
        process=False,
    )
    out.remove_unreferenced_vertices()

    before_stats = _edge_count_stats(faces)
    after_stats = _edge_count_stats(np.asarray(out.faces, dtype=np.int64))
    if after_stats["nonmanifold_edges"] > before_stats["nonmanifold_edges"]:
        for plan in accepted_plans:
            plan["report"]["accepted"] = False
            plan["report"]["reason"] = "global_nonmanifold_guard_rejected"
        reports.extend(plan["report"] for plan in plans)
        return mesh.copy(), reports

    reports.extend(plan["report"] for plan in plans)
    return out, reports
