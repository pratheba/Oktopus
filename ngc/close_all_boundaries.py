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


def project_loop_to_best_fit_plane(points_3d: np.ndarray):
    """Return 2D coordinates and an orthonormal PCA plane basis."""
    center = points_3d.mean(axis=0)
    centered = points_3d - center

    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if len(singular_values) < 2 or singular_values[1] < 1e-12:
        raise BoundaryError("Boundary loop is degenerate or nearly collinear.")

    axis_u = vh[0]
    axis_v = vh[1]
    points_2d = np.column_stack((centered @ axis_u, centered @ axis_v))

    # Make the cap-directed loop counter-clockwise in the projected plane.
    if signed_area_2d(points_2d) < 0.0:
        axis_v = -axis_v
        points_2d[:, 1] *= -1.0

    return points_2d, center, axis_u, axis_v


def triangulate_boundary_loop(
    vertices: np.ndarray,
    loop: np.ndarray,
    tolerance_scale: float = 1e-7,
) -> np.ndarray:
    """Triangulate one (possibly concave) simple boundary loop."""
    points_3d = vertices[loop]
    points_2d, _, _, _ = project_loop_to_best_fit_plane(points_3d)

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

    for component_id, edge_ids in enumerate(components):
        component_edges = boundary[edge_ids]
        loop = order_simple_loop(component_edges, directed_set)
        cap_faces = triangulate_boundary_loop(mesh.vertices, loop)
        loops.append(loop)
        all_cap_faces.append(cap_faces)
        print(
            f"  boundary {component_id:02d}: "
            f"{len(loop)} edges -> {len(cap_faces)} cap triangles"
        )

    cap_faces_all = np.vstack(all_cap_faces)
    closed = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices).copy(),
        faces=np.vstack((np.asarray(mesh.faces), cap_faces_all)),
        process=False,
    )

    closed.update_faces(closed.nondegenerate_faces())
    closed.update_faces(closed.unique_faces())
    closed.remove_unreferenced_vertices()
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
