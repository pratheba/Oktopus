#!/usr/bin/env python3
"""Close every simple open boundary loop in a triangular mesh.

The script:
  1. loads a triangle mesh,
  2. performs conservative cleanup,
  3. finds all boundary edges (edges used by exactly one face),
  4. orders them into simple loops,
  5. triangulates every loop in a best-fit 2D plane,
  6. maps the triangulation back to the original 3D boundary vertices,
  7. fixes face winding/normals and validates that no boundary edges remain.

Only the input mesh is required. The missing cap geometry is inferred from each
boundary loop; no point cloud or full/reference mesh is used.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from shapely.ops import triangulate


class BoundaryError(RuntimeError):
    """Raised when a mesh boundary cannot be interpreted as simple loops."""


def load_triangle_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)

    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geom
            for geom in loaded.geometry.values()
            if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 0
        ]
        if not meshes:
            raise ValueError(f"No triangle mesh was found in: {path}")
        mesh = trimesh.util.concatenate(meshes)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError(f"Unsupported geometry type: {type(loaded)!r}")

    if len(mesh.faces) == 0:
        raise ValueError("The input contains no triangle faces.")

    # Conservative cleanup: no smoothing, remeshing, or vertex displacement.
    mesh = mesh.copy()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()

    # Boundary orientation is much easier to interpret when adjacent faces have
    # coherent winding. This changes triangle ordering, not vertex positions.
    trimesh.repair.fix_winding(mesh)
    return mesh


def edge_topology(faces: np.ndarray):
    """Return boundary edges, directed boundary edges, and non-manifold edges."""
    faces = np.asarray(faces, dtype=np.int64)

    directed = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    undirected = np.sort(directed, axis=1)

    unique_edges, first, inverse, counts = np.unique(
        undirected,
        axis=0,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )

    boundary_ids = np.flatnonzero(counts == 1)
    nonmanifold_edges = unique_edges[counts > 2]

    # For a boundary edge there is exactly one occurrence, so its direction is
    # the orientation induced by the existing incident face.
    boundary_directed = directed[first[boundary_ids]]
    boundary_undirected = unique_edges[boundary_ids]

    return boundary_undirected, boundary_directed, nonmanifold_edges


def connected_boundary_components(boundary_edges: np.ndarray) -> list[np.ndarray]:
    """Group boundary edges into connected boundary graph components."""
    if len(boundary_edges) == 0:
        return []

    vertex_to_edges: dict[int, list[int]] = defaultdict(list)
    for edge_id, (a, b) in enumerate(boundary_edges):
        vertex_to_edges[int(a)].append(edge_id)
        vertex_to_edges[int(b)].append(edge_id)

    unseen = set(range(len(boundary_edges)))
    components: list[np.ndarray] = []

    while unseen:
        seed = unseen.pop()
        queue = deque([seed])
        component = [seed]

        while queue:
            edge_id = queue.popleft()
            a, b = boundary_edges[edge_id]
            for vertex in (int(a), int(b)):
                for neighbor_edge in vertex_to_edges[vertex]:
                    if neighbor_edge in unseen:
                        unseen.remove(neighbor_edge)
                        queue.append(neighbor_edge)
                        component.append(neighbor_edge)

        components.append(np.asarray(component, dtype=np.int64))

    return components


def order_simple_loop(
    component_edges: np.ndarray,
    directed_boundary_set: set[tuple[int, int]],
) -> np.ndarray:
    """Order one degree-2 boundary component into a consistently directed loop."""
    adjacency: dict[int, list[int]] = defaultdict(list)
    for a, b in component_edges:
        adjacency[int(a)].append(int(b))
        adjacency[int(b)].append(int(a))

    bad_degree = {v: len(nbrs) for v, nbrs in adjacency.items() if len(nbrs) != 2}
    if bad_degree:
        preview = list(bad_degree.items())[:10]
        raise BoundaryError(
            "Boundary component is not a simple loop. "
            f"Boundary vertex degrees (first entries): {preview}. "
            "This usually indicates non-manifold geometry, T-junctions, or an "
            "open boundary chain."
        )

    start = min(adjacency)
    previous = None
    current = start
    loop = [start]

    for _ in range(len(component_edges)):
        neighbors = adjacency[current]
        next_vertex = neighbors[0] if neighbors[0] != previous else neighbors[1]

        if next_vertex == start:
            break

        loop.append(next_vertex)
        previous, current = current, next_vertex
    else:
        raise BoundaryError("Failed to close a boundary component into a loop.")

    if len(loop) != len(component_edges):
        raise BoundaryError(
            "Boundary traversal did not use every edge exactly once: "
            f"vertices={len(loop)}, edges={len(component_edges)}."
        )

    loop_arr = np.asarray(loop, dtype=np.int64)

    # The walk direction above was arbitrary. Align it with the directions of
    # the existing boundary faces, then reverse it for the cap. Adjacent faces
    # on a manifold edge must traverse their shared edge in opposite directions.
    same = 0
    opposite = 0
    for i, a in enumerate(loop_arr):
        b = int(loop_arr[(i + 1) % len(loop_arr)])
        if (int(a), b) in directed_boundary_set:
            same += 1
        if (b, int(a)) in directed_boundary_set:
            opposite += 1

    existing_direction = loop_arr if same >= opposite else loop_arr[::-1]
    cap_direction = existing_direction[::-1]
    return cap_direction


def signed_area_2d(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _normalise(vector: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length < 1.0e-14:
        return None
    return vector / length


def _newell_normal(points_3d: np.ndarray) -> np.ndarray | None:
    """Return a Newell polygon normal for an ordered 3D loop."""
    points_3d = np.asarray(points_3d, dtype=np.float64)
    shifted = np.roll(points_3d, -1, axis=0)
    normal = np.array(
        [
            np.sum((points_3d[:, 1] - shifted[:, 1])
                   * (points_3d[:, 2] + shifted[:, 2])),
            np.sum((points_3d[:, 2] - shifted[:, 2])
                   * (points_3d[:, 0] + shifted[:, 0])),
            np.sum((points_3d[:, 0] - shifted[:, 0])
                   * (points_3d[:, 1] + shifted[:, 1])),
        ],
        dtype=np.float64,
    )
    return _normalise(normal)


def _plane_basis_from_normal(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = _normalise(normal)
    if normal is None:
        raise BoundaryError("Cannot construct a plane from a zero normal.")

    # Choose the coordinate axis least parallel to the normal.
    helpers = np.eye(3, dtype=np.float64)
    helper = helpers[int(np.argmin(np.abs(helpers @ normal)))]

    axis_u = _normalise(np.cross(normal, helper))
    if axis_u is None:
        raise BoundaryError("Failed to construct the first plane axis.")

    axis_v = _normalise(np.cross(normal, axis_u))
    if axis_v is None:
        raise BoundaryError("Failed to construct the second plane axis.")

    return axis_u, axis_v


def _fibonacci_sphere_directions(count: int) -> list[np.ndarray]:
    """Deterministic, approximately uniform unit directions."""
    count = max(1, int(count))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    directions: list[np.ndarray] = []

    for index in range(count):
        z = 1.0 - 2.0 * (index + 0.5) / count
        radius = np.sqrt(max(0.0, 1.0 - z * z))
        phi = golden_angle * index
        directions.append(
            np.array(
                [radius * np.cos(phi), radius * np.sin(phi), z],
                dtype=np.float64,
            )
        )

    return directions


def _project_loop_with_normal(
    points_3d: np.ndarray,
    normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = np.asarray(points_3d, dtype=np.float64).mean(axis=0)
    centered = np.asarray(points_3d, dtype=np.float64) - center
    axis_u, axis_v = _plane_basis_from_normal(normal)
    points_2d = np.column_stack((centered @ axis_u, centered @ axis_v))

    # Make the cap-directed loop counter-clockwise in the projected plane.
    if signed_area_2d(points_2d) < 0.0:
        axis_v = -axis_v
        points_2d[:, 1] *= -1.0

    return points_2d, center, axis_u, axis_v


def project_loop_to_best_fit_plane(points_3d: np.ndarray):
    """Return the traditional PCA projection used by earlier versions."""
    points_3d = np.asarray(points_3d, dtype=np.float64)
    center = points_3d.mean(axis=0)
    centered = points_3d - center

    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if len(singular_values) < 2 or singular_values[1] < 1e-12:
        raise BoundaryError("Boundary loop is degenerate or nearly collinear.")

    axis_u = vh[0]
    axis_v = vh[1]
    points_2d = np.column_stack((centered @ axis_u, centered @ axis_v))

    if signed_area_2d(points_2d) < 0.0:
        axis_v = -axis_v
        points_2d[:, 1] *= -1.0

    return points_2d, center, axis_u, axis_v


def project_loop_to_valid_plane(
    points_3d: np.ndarray,
    search_direction_count: int = 96,
):
    """
    Find a non-self-intersecting orthographic projection of a 3D loop.

    A strongly non-planar boundary can self-intersect in its PCA projection
    even when the 3D boundary itself is a perfectly valid simple loop.  Try
    PCA, Newell, coordinate-axis and deterministic spherical directions, then
    retain the valid projection with the largest projected area.
    """
    points_3d = np.asarray(points_3d, dtype=np.float64)
    if len(points_3d) < 3:
        raise BoundaryError("A boundary loop needs at least three vertices.")

    centered = points_3d - points_3d.mean(axis=0)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if len(singular_values) < 2 or singular_values[1] < 1.0e-12:
        raise BoundaryError("Boundary loop is degenerate or nearly collinear.")

    candidate_normals: list[np.ndarray] = []

    if vh.shape[0] >= 3:
        candidate_normals.append(np.asarray(vh[2], dtype=np.float64))

    newell = _newell_normal(points_3d)
    if newell is not None:
        candidate_normals.append(newell)

    candidate_normals.extend(np.eye(3, dtype=np.float64))
    candidate_normals.extend(
        _fibonacci_sphere_directions(search_direction_count)
    )

    # Remove duplicate planes. n and -n describe the same projection plane.
    unique_normals: list[np.ndarray] = []
    for normal in candidate_normals:
        normal = _normalise(normal)
        if normal is None:
            continue
        if any(abs(float(np.dot(normal, previous))) > 1.0 - 1.0e-8
               for previous in unique_normals):
            continue
        unique_normals.append(normal)

    best = None
    best_score = -np.inf
    attempted = 0

    for normal in unique_normals:
        attempted += 1
        try:
            projection = _project_loop_with_normal(points_3d, normal)
        except BoundaryError:
            continue

        points_2d = projection[0]
        bbox_diag = float(np.linalg.norm(np.ptp(points_2d, axis=0)))
        area_tolerance = max(
            1.0e-16,
            1.0e-14 * max(bbox_diag * bbox_diag, 1.0),
        )

        polygon = Polygon(points_2d)
        if polygon.is_empty:
            continue
        if not polygon.is_valid or not polygon.exterior.is_simple:
            continue
        if float(polygon.area) <= area_tolerance:
            continue

        # Prefer a projection far from edge-on degeneracy.
        score = float(polygon.area)
        if score > best_score:
            best = projection
            best_score = score

    if best is None:
        raise BoundaryError(
            "No non-self-intersecting planar projection was found for this "
            f"non-planar boundary loop after testing {attempted} planes."
        )

    return best


def triangulate_boundary_loop(
    vertices: np.ndarray,
    loop: np.ndarray,
    tolerance_scale: float = 1e-7,
) -> np.ndarray:
    """Triangulate one (possibly concave) simple boundary loop."""
    points_3d = vertices[loop]
    points_2d, _, _, _ = project_loop_to_valid_plane(points_3d)

    polygon = Polygon(points_2d)
    if polygon.is_empty or polygon.area <= 1e-16:
        raise BoundaryError("Projected boundary polygon has zero area.")
    if not polygon.is_valid:
        raise BoundaryError(
            "Projected boundary loop self-intersects or is otherwise invalid. "
            "Automatic capping is unsafe for this loop."
        )

    # Shapely performs a Delaunay triangulation of the loop vertices. Keep only
    # triangles fully covered by the possibly concave polygon.
    candidates = triangulate(polygon)
    kept = [tri for tri in candidates if polygon.covers(tri)]
    if not kept:
        raise BoundaryError("Triangulation produced no triangles inside the loop.")

    tree = cKDTree(points_2d)
    bbox_diag = float(np.linalg.norm(np.ptp(points_2d, axis=0)))
    tolerance = max(1e-12, tolerance_scale * max(bbox_diag, 1.0))

    cap_faces: list[list[int]] = []
    seen: set[tuple[int, int, int]] = set()

    for tri in kept:
        coords = np.asarray(tri.exterior.coords[:-1], dtype=np.float64)
        if coords.shape != (3, 2):
            continue

        distances, local_ids = tree.query(coords, k=1)
        if np.max(distances) > tolerance:
            raise BoundaryError(
                "Triangulation introduced a vertex that cannot be mapped back "
                "to the original boundary loop."
            )

        local_ids = np.asarray(local_ids, dtype=np.int64)
        if len(np.unique(local_ids)) != 3:
            continue

        tri_2d = points_2d[local_ids]
        e1 = tri_2d[1] - tri_2d[0]
        e2 = tri_2d[2] - tri_2d[0]
        area2 = float(e1[0] * e2[1] - e1[1] * e2[0])
        if area2 < 0.0:
            local_ids[[1, 2]] = local_ids[[2, 1]]

        face = [int(loop[i]) for i in local_ids]
        key = tuple(sorted(face))
        if key not in seen:
            seen.add(key)
            cap_faces.append(face)

    if len(cap_faces) != len(loop) - 2:
        # A valid triangulation of a simple polygon with no Steiner points has
        # exactly n-2 triangles. Treat a mismatch as unsafe rather than silently
        # producing a partial cap.
        raise BoundaryError(
            f"Expected {len(loop) - 2} cap triangles but produced "
            f"{len(cap_faces)}."
        )

    return np.asarray(cap_faces, dtype=np.int64)



def _undirected_edge_counts(faces: np.ndarray) -> dict[tuple[int, int], int]:
    """Count how many triangles use each undirected edge."""
    faces = np.asarray(faces, dtype=np.int64)
    edges = np.sort(
        np.concatenate(
            [
                faces[:, [0, 1]],
                faces[:, [1, 2]],
                faces[:, [2, 0]],
            ],
            axis=0,
        ),
        axis=1,
    )
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return {
        (int(edge[0]), int(edge[1])): int(count)
        for edge, count in zip(unique, counts)
    }


def _validate_cap_against_source(
    source_faces: np.ndarray,
    loop: np.ndarray,
    cap_faces: np.ndarray,
) -> list[str]:
    """
    Validate one cap before adding it to the source mesh.

    A valid cap must:
      - contain no face already present in the source,
      - use every source boundary-loop edge exactly once,
      - use every new internal cap edge exactly twice,
      - never reuse a source interior edge.
    """
    source_faces = np.asarray(source_faces, dtype=np.int64)
    cap_faces = np.asarray(cap_faces, dtype=np.int64)
    loop = np.asarray(loop, dtype=np.int64)

    errors: list[str] = []

    source_face_keys = {
        tuple(sorted(map(int, face)))
        for face in source_faces
    }
    duplicate_faces = [
        tuple(map(int, face))
        for face in cap_faces
        if tuple(sorted(map(int, face))) in source_face_keys
    ]
    if duplicate_faces:
        errors.append(
            "cap duplicates source faces "
            f"{duplicate_faces[:10]}"
        )

    source_edge_counts = _undirected_edge_counts(source_faces)
    cap_edge_counts = _undirected_edge_counts(cap_faces)

    loop_edges = np.sort(
        np.column_stack([loop, np.roll(loop, -1)]),
        axis=1,
    )
    loop_edge_set = {
        (int(edge[0]), int(edge[1]))
        for edge in loop_edges
    }

    bad_loop_edges = []
    for edge in sorted(loop_edge_set):
        source_count = source_edge_counts.get(edge, 0)
        cap_count = cap_edge_counts.get(edge, 0)
        if source_count != 1 or cap_count != 1:
            bad_loop_edges.append(
                (edge, source_count, cap_count)
            )
    if bad_loop_edges:
        errors.append(
            "loop edges must have source_count=1 and cap_count=1: "
            f"{bad_loop_edges[:10]}"
        )

    bad_cap_edges = []
    for edge, cap_count in cap_edge_counts.items():
        source_count = source_edge_counts.get(edge, 0)

        if edge in loop_edge_set:
            continue

        # A cap-internal diagonal must be new and shared by two cap faces.
        if source_count != 0 or cap_count != 2:
            bad_cap_edges.append(
                (edge, source_count, cap_count)
            )

    if bad_cap_edges:
        errors.append(
            "cap internal edges must be new and used by exactly two "
            f"cap faces: {bad_cap_edges[:10]}"
        )

    return errors


def _cross2(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Signed twice-area of the 2D triangle (a,b,c)."""
    ab = b - a
    ac = c - a
    return float(ab[0] * ac[1] - ab[1] * ac[0])


def _point_in_or_on_ccw_triangle(
    point: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    tolerance: float,
) -> bool:
    """Return True when point lies inside or on a CCW triangle."""
    c1 = _cross2(a, b, point)
    c2 = _cross2(b, c, point)
    c3 = _cross2(c, a, point)
    return (
        c1 >= -tolerance
        and c2 >= -tolerance
        and c3 >= -tolerance
    )


def triangulate_boundary_loop_ear_clip(
    vertices: np.ndarray,
    loop: np.ndarray,
    source_faces: np.ndarray,
) -> np.ndarray:
    """
    Constrained ear-clipping fallback with backtracking.

    A greedy ear clip can choose an individually valid ear and later leave a
    polygon for which every remaining diagonal conflicts with source topology.
    This implementation explores alternate valid ears and returns the first
    complete triangulation that satisfies all constraints.

    Constraints:
      - no cap face may duplicate a source face,
      - no cap-internal diagonal may reuse a source edge,
      - no ear may contain another active boundary vertex,
      - all triangles preserve the projected CCW orientation.
    """
    loop = np.asarray(loop, dtype=np.int64)
    points_3d = np.asarray(vertices, dtype=np.float64)[loop]
    points_2d, _, _, _ = project_loop_to_valid_plane(points_3d)

    polygon = Polygon(points_2d)
    if polygon.is_empty or polygon.area <= 1e-16:
        raise BoundaryError("Projected boundary polygon has zero area.")
    if not polygon.is_valid:
        raise BoundaryError(
            "Projected boundary loop is invalid; constrained ear clipping "
            "is unsafe."
        )

    source_edge_counts = _undirected_edge_counts(source_faces)
    source_edges = set(source_edge_counts)
    source_face_keys = {
        tuple(sorted(map(int, face)))
        for face in np.asarray(source_faces, dtype=np.int64)
    }

    loop_edges = np.sort(
        np.column_stack([loop, np.roll(loop, -1)]),
        axis=1,
    )
    loop_edge_set = {
        (int(edge[0]), int(edge[1]))
        for edge in loop_edges
    }

    bbox_diag = float(np.linalg.norm(np.ptp(points_2d, axis=0)))
    area_tolerance = max(
        1e-16,
        1e-14 * max(bbox_diag * bbox_diag, 1.0),
    )

    failed_states: set[tuple[int, ...]] = set()
    visited_states = 0
    max_states = max(10000, 500 * len(loop) * len(loop))

    def triangle_is_allowed(
        a_local: int,
        b_local: int,
        c_local: int,
        active: tuple[int, ...],
        check_other_vertices: bool,
    ) -> bool:
        a = points_2d[a_local]
        b = points_2d[b_local]
        c = points_2d[c_local]

        if _cross2(a, b, c) <= area_tolerance:
            return False

        a_vertex = int(loop[a_local])
        b_vertex = int(loop[b_local])
        c_vertex = int(loop[c_local])

        face_key = tuple(sorted((a_vertex, b_vertex, c_vertex)))
        if face_key in source_face_keys:
            return False

        # All triangle edges that are not original loop edges become cap
        # diagonals. They must not already exist in the source mesh.
        for u, v in (
            (a_vertex, b_vertex),
            (b_vertex, c_vertex),
            (c_vertex, a_vertex),
        ):
            edge = tuple(sorted((u, v)))
            if edge in source_edges and edge not in loop_edge_set:
                return False

        if check_other_vertices:
            for other_local in active:
                if other_local in (a_local, b_local, c_local):
                    continue
                if _point_in_or_on_ccw_triangle(
                    points_2d[other_local],
                    a,
                    b,
                    c,
                    area_tolerance,
                ):
                    return False

        return True

    def solve(active: tuple[int, ...]) -> list[list[int]] | None:
        nonlocal visited_states

        if active in failed_states:
            return None

        visited_states += 1
        if visited_states > max_states:
            raise BoundaryError(
                "Constrained ear-clipping backtracking exceeded its search "
                f"limit ({max_states} states) for a {len(loop)}-vertex loop."
            )

        if len(active) == 3:
            a_local, b_local, c_local = active
            if not triangle_is_allowed(
                a_local,
                b_local,
                c_local,
                active,
                check_other_vertices=False,
            ):
                failed_states.add(active)
                return None

            return [[
                int(loop[a_local]),
                int(loop[b_local]),
                int(loop[c_local]),
            ]]

        candidates: list[tuple[float, int, list[int]]] = []

        for position in range(len(active)):
            prev_local = active[(position - 1) % len(active)]
            curr_local = active[position]
            next_local = active[(position + 1) % len(active)]

            if not triangle_is_allowed(
                prev_local,
                curr_local,
                next_local,
                active,
                check_other_vertices=True,
            ):
                continue

            area2 = _cross2(
                points_2d[prev_local],
                points_2d[curr_local],
                points_2d[next_local],
            )
            face = [
                int(loop[prev_local]),
                int(loop[curr_local]),
                int(loop[next_local]),
            ]
            candidates.append((area2, position, face))

        # Smaller ears first tends to preserve larger unconstrained regions,
        # but unlike the previous implementation every alternative can be
        # revisited if the first choice reaches a dead end.
        candidates.sort(key=lambda item: item[0])

        for _, position, face in candidates:
            next_active = active[:position] + active[position + 1:]
            remainder = solve(next_active)
            if remainder is not None:
                return [face] + remainder

        failed_states.add(active)
        return None

    result_list = solve(tuple(range(len(loop))))

    if result_list is None:
        raise BoundaryError(
            "No complete constrained triangulation exists using only the "
            "current boundary vertices without duplicating source faces or "
            "reusing source interior edges. "
            f"Loop vertices: {[int(v) for v in loop.tolist()]}"
        )

    result = np.asarray(result_list, dtype=np.int64)

    if len(result) != len(loop) - 2:
        raise BoundaryError(
            f"Expected {len(loop) - 2} constrained cap triangles but "
            f"produced {len(result)}."
        )

    return result


def triangulate_boundary_loop_center_fan(
    vertices: np.ndarray,
    loop: np.ndarray,
    source_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Last-resort non-planar cap using one new Steiner vertex.

    This fallback does not require a globally valid 2D projection.  Every
    original boundary edge receives exactly one cap triangle and every new
    center-to-boundary edge receives exactly two cap triangles, so the result
    is topologically closed.  It is intended for SDF-oracle closure of highly
    folded/non-planar cut boundaries.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    loop = np.asarray(loop, dtype=np.int64)
    points = vertices[loop]

    edge_vectors = np.roll(points, -1, axis=0) - points
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    scale = float(np.linalg.norm(np.ptp(points, axis=0)))
    if not np.isfinite(scale) or scale < 1.0e-12:
        raise BoundaryError("Cannot fan-cap a degenerate boundary loop.")

    mean_center = points.mean(axis=0)
    median_center = np.median(points, axis=0)

    length_sum = float(edge_lengths.sum())
    if length_sum > 1.0e-14:
        edge_midpoints = 0.5 * (points + np.roll(points, -1, axis=0))
        weighted_center = np.sum(
            edge_midpoints * edge_lengths[:, None],
            axis=0,
        ) / length_sum
    else:
        weighted_center = mean_center

    normals: list[np.ndarray] = []
    newell = _newell_normal(points)
    if newell is not None:
        normals.append(newell)

    centered = points - mean_center
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if len(singular_values) >= 2 and singular_values[1] >= 1.0e-12:
        if vh.shape[0] >= 3:
            pca_normal = _normalise(vh[2])
            if pca_normal is not None:
                normals.append(pca_normal)

    if not normals:
        normals.append(np.array([0.0, 0.0, 1.0], dtype=np.float64))

    base_centers = [mean_center, weighted_center, median_center]
    relative_offsets = [0.0, 1.0e-4, -1.0e-4, 1.0e-3, -1.0e-3,
                        1.0e-2, -1.0e-2, 5.0e-2, -5.0e-2]

    area_tolerance = max(1.0e-18, 1.0e-14 * scale * scale)
    best = None
    best_min_area = -np.inf

    for base_center in base_centers:
        for normal in normals:
            for relative_offset in relative_offsets:
                center = base_center + relative_offset * scale * normal
                center_id = len(vertices)
                cap_faces = np.column_stack(
                    [
                        loop,
                        np.roll(loop, -1),
                        np.full(len(loop), center_id, dtype=np.int64),
                    ]
                ).astype(np.int64)

                tri_a = points
                tri_b = np.roll(points, -1, axis=0)
                doubled_areas = np.linalg.norm(
                    np.cross(tri_b - tri_a, center[None, :] - tri_a),
                    axis=1,
                )
                min_area = float(np.min(doubled_areas))
                if not np.isfinite(min_area) or min_area <= area_tolerance:
                    continue

                errors = _validate_cap_against_source(
                    source_faces,
                    loop,
                    cap_faces,
                )
                if errors:
                    continue

                # Prefer the most numerically stable fan.  Zero-offset
                # candidates are encountered first when equally stable.
                if min_area > best_min_area:
                    best = (center, cap_faces)
                    best_min_area = min_area

    if best is None:
        raise BoundaryError(
            "Failed to construct a non-degenerate center-fan cap for the "
            "non-planar boundary loop."
        )

    center, cap_faces = best
    vertices_out = np.vstack([vertices, center[None, :]])
    return vertices_out, cap_faces


def describe_boundaries(mesh: trimesh.Trimesh):
    boundary, directed, nonmanifold = edge_topology(mesh.faces)
    components = connected_boundary_components(boundary)
    sizes = sorted((len(component) for component in components), reverse=True)
    return boundary, directed, nonmanifold, components, sizes


def close_all_boundaries(mesh: trimesh.Trimesh):
    boundary, directed, nonmanifold, components, sizes = describe_boundaries(mesh)

    if len(nonmanifold) > 0:
        raise BoundaryError(
            f"The mesh contains {len(nonmanifold)} non-manifold edges "
            "(used by more than two faces). Repair those before hole closing."
        )

    if len(boundary) == 0:
        return mesh.copy(), [], np.zeros((0, 3), dtype=np.int64)

    directed_set = {(int(a), int(b)) for a, b in directed}
    loops: list[np.ndarray] = []
    all_cap_faces: list[np.ndarray] = []
    working_vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()

    for component_id, edge_ids in enumerate(components):
        component_edges = boundary[edge_ids]
        loop = order_simple_loop(component_edges, directed_set)

        delaunay_failure = None
        try:
            cap_faces = triangulate_boundary_loop(working_vertices, loop)
            cap_errors = _validate_cap_against_source(
                mesh.faces,
                loop,
                cap_faces,
            )
        except BoundaryError as exc:
            delaunay_failure = str(exc)
            cap_errors = []

        if delaunay_failure is not None or cap_errors:
            if delaunay_failure is not None:
                print(
                    f"  boundary {component_id:02d}: "
                    "Delaunay triangulation failed; "
                    "using constrained ear clipping"
                )
                print(f"    {delaunay_failure}")
            else:
                print(
                    f"  boundary {component_id:02d}: "
                    "Delaunay cap conflicts with source topology; "
                    "using constrained ear clipping"
                )
                for error in cap_errors:
                    print(f"    {error}")

            ear_failure = None
            try:
                cap_faces = triangulate_boundary_loop_ear_clip(
                    working_vertices,
                    loop,
                    mesh.faces,
                )
                cap_errors = _validate_cap_against_source(
                    mesh.faces,
                    loop,
                    cap_faces,
                )
                if cap_errors:
                    raise BoundaryError(
                        "Constrained cap failed topology validation: "
                        + "; ".join(cap_errors)
                    )
            except BoundaryError as exc:
                ear_failure = str(exc)

            if ear_failure is not None:
                print(
                    f"  boundary {component_id:02d}: "
                    "constrained ear clipping failed; "
                    "using non-planar center-fan fallback"
                )
                print(f"    {ear_failure}")

                working_vertices, cap_faces = (
                    triangulate_boundary_loop_center_fan(
                        working_vertices,
                        loop,
                        mesh.faces,
                    )
                )
                cap_errors = _validate_cap_against_source(
                    mesh.faces,
                    loop,
                    cap_faces,
                )
                if cap_errors:
                    raise BoundaryError(
                        "Center-fan cap failed topology validation: "
                        + "; ".join(cap_errors)
                    )

                print(
                    f"    added one Steiner cap vertex at index "
                    f"{len(working_vertices) - 1}"
                )

        loops.append(loop)
        all_cap_faces.append(cap_faces)
        print(
            f"  boundary {component_id:02d}: "
            f"{len(loop)} edges -> {len(cap_faces)} cap triangles"
        )

    cap_faces_all = np.vstack(all_cap_faces)
    closed = trimesh.Trimesh(
        vertices=working_vertices,
        faces=np.vstack(
            (
                np.asarray(mesh.faces, dtype=np.int64),
                np.asarray(cap_faces_all, dtype=np.int64),
            )
        ),
        process=False,
    )

    boundary_after, _, nonmanifold_after, _, _ = describe_boundaries(closed)

    if len(boundary_after) != 0:
        raise BoundaryError(
            "Validated cap insertion still left "
            f"{len(boundary_after)} boundary edges."
        )
    if len(nonmanifold_after) != 0:
        raise BoundaryError(
            "Validated cap insertion produced "
            f"{len(nonmanifold_after)} non-manifold edges."
        )

    trimesh.repair.fix_winding(closed)
    trimesh.repair.fix_normals(closed, multibody=True)

    return closed, loops, cap_faces_all


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Close every simple open boundary loop in a triangle mesh."
    )
    parser.add_argument("input_mesh", type=Path, help="Input PLY/OBJ/STL/etc.")
    parser.add_argument("output_mesh", type=Path, help="Output closed mesh")
    parser.add_argument(
        "--allow-still-open",
        action="store_true",
        help="Save even if validation still finds boundary edges.",
    )
    args = parser.parse_args()

    if not args.input_mesh.exists():
        parser.error(f"Input does not exist: {args.input_mesh}")

    mesh = load_triangle_mesh(args.input_mesh)
    before = describe_boundaries(mesh)
    _, _, nonmanifold_before, components_before, sizes_before = before

    print(f"Input: {args.input_mesh}")
    print(f"  vertices: {len(mesh.vertices)}")
    print(f"  faces: {len(mesh.faces)}")
    print(f"  connected bodies: {mesh.body_count}")
    print(f"  boundary loops/components: {len(components_before)}")
    print(f"  boundary sizes (edges): {sizes_before}")
    print(f"  non-manifold edges: {len(nonmanifold_before)}")
    print(f"  watertight before: {mesh.is_watertight}")

    try:
        closed, loops, cap_faces = close_all_boundaries(mesh)
    except BoundaryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    boundary_after, _, nonmanifold_after, components_after, sizes_after = (
        describe_boundaries(closed)
    )

    print("After closing:")
    print(f"  caps added: {len(loops)}")
    print(f"  cap triangles added: {len(cap_faces)}")
    print(f"  remaining boundary edges: {len(boundary_after)}")
    print(f"  remaining boundary components: {len(components_after)}")
    print(f"  remaining boundary sizes: {sizes_after}")
    print(f"  non-manifold edges: {len(nonmanifold_after)}")
    print(f"  watertight: {closed.is_watertight}")
    print(f"  winding consistent: {closed.is_winding_consistent}")
    print(f"  is volume: {closed.is_volume}")

    valid = (
        len(boundary_after) == 0
        and len(nonmanifold_after) == 0
        and closed.is_watertight
    )

    if not valid and not args.allow_still_open:
        print(
            "ERROR: The result did not pass closure validation; output was not saved. "
            "Use --allow-still-open only for debugging.",
            file=sys.stderr,
        )
        return 3

    args.output_mesh.parent.mkdir(parents=True, exist_ok=True)
    closed.export(args.output_mesh)
    print(f"Saved: {args.output_mesh}")
    return 0 if valid else 4


if __name__ == "__main__":
    raise SystemExit(main())
