"""Topology diagnostics for a true scalar SDF min-union (never claim welding)."""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def fused_surface_report(mesh):
    """Return edge-level topology status without modifying any triangles."""
    n_faces = int(len(mesh.faces))
    if n_faces == 0:
        raise ValueError('Cannot report topology of empty fused SDF mesh')
    edge_occurrences = np.bincount(
        np.asarray(mesh.edges_unique_inverse, dtype=np.int64),
        minlength=len(mesh.edges_unique),
    )
    adjacent = np.asarray(mesh.face_adjacency, dtype=np.int64).reshape(-1, 2)
    if len(adjacent):
        a = np.concatenate((adjacent[:, 0], adjacent[:, 1]))
        b = np.concatenate((adjacent[:, 1], adjacent[:, 0]))
        graph = coo_matrix(
            (np.ones(len(a), dtype=np.int8), (a, b)),
            shape=(n_faces, n_faces),
        ).tocsr()
        num, labels = connected_components(graph, directed=False)
    else:
        num, labels = n_faces, np.arange(n_faces)
    sizes = np.bincount(labels, minlength=num)
    return {
        'faces': n_faces,
        'vertices': int(len(mesh.vertices)),
        'face_components': int(num),
        'largest_face_component': int(sizes.max()),
        'boundary_edges': int(np.count_nonzero(edge_occurrences == 1)),
        'nonmanifold_edges': int(np.count_nonzero(edge_occurrences > 2)),
        'watertight': bool(mesh.is_watertight),
        'bounds': np.asarray(mesh.bounds).tolist(),
        'warning': ('This is the outer surface of the composed signed field. '
                    'It is not necessarily watertight, and UDF openings '
                    'have not yet been cut on this fused mesh.'),
    }
