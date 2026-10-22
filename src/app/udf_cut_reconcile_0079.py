"""Opt-in independent UDF evidence reconciliation and generic debris cleanup.

No anatomical coordinates, pre-recorded masks, old intermediate SDF meshes, or
avatar Boolean are used. Unknown evidence is *never* interpreted as removal.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def vote_agreement(assigned, valid, remove_votes, keep_votes):
    """Remove only where EVERY eligible UDF validly accepted a cut component.

    An owner with a valid score but whose seed/grow component was rejected
    contributes keep evidence. An unknown owner abstains, blocking automatic
    removal; fully unassigned faces are retained as unknown.
    """
    a, v, r, k = [np.asarray(x, dtype=np.int64).reshape(-1)
                  for x in (assigned, valid, remove_votes, keep_votes)]
    if not (len(a) == len(v) == len(r) == len(k)):
        raise ValueError('0079 per-face vote lengths mismatch')
    if (np.any(a < 0) or np.any(v < 0) or np.any(r < 0) or
            np.any(k < 0) or np.any(v > a) or np.any(r + k != v)):
        raise ValueError('0079 inconsistent owner votes')
    unassigned = a == 0
    missing = (a > 0) & (v < a)
    # Keep conflicting proposals distinguishable from missing evidence.
    conflict = (r > 0) & (k > 0)
    # Includes faces with valid UDF but no accepted connected cut component.
    keep = (v > 0) & (k > 0) & ~conflict
    proposed = (a > 0) & (v == a) & (r == a) & (k == 0)
    return proposed, {
        'unassigned': unassigned,
        'missing_evidence': missing,
        'keep_conflict': conflict,
        'kept_by_owner': keep,
        'agreed_remove': proposed,
    }


def _edge_component_labels(faces):
    """Edge-connectivity, including shared nonmanifold edges (not just vertices)."""
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError('0079 expects triangular faces')
    if not len(f):
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int64)
    # Sparse adjacency; identical shared edges map to the same id.
    edges = np.sort(np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]])), axis=1)
    _, inverse = np.unique(edges, axis=0, return_inverse=True)
    face_id = np.tile(np.arange(len(f), dtype=np.int64), 3)
    order = np.argsort(inverse, kind='stable')
    inv = inverse[order]
    ids = face_id[order]
    starts = np.r_[0, np.flatnonzero(np.diff(inv)) + 1]
    sizes = np.diff(np.r_[starts, len(inv)])
    root_index = np.repeat(starts, sizes)
    nonfirst = np.arange(len(inv)) != root_index
    row = ids[nonfirst]
    col = ids[root_index[nonfirst]]
    graph = coo_matrix((np.ones(2 * len(row), dtype=np.uint8),
                        (np.r_[row, col], np.r_[col, row])),
                       shape=(len(f), len(f))).tocsr()
    _, labels = connected_components(graph, directed=False, return_labels=True)
    return labels.astype(np.int32), np.bincount(labels).astype(np.int64)


def keep_largest_edge_component(mesh, *, min_main_fraction=0.85):
    """Remove detached debris only; do not smooth, fill holes, or trim rim ears.

    Refuse to discard substantial parts of a multi-piece mesh. The caller
    should invoke this on one root/leaf part, never on a combined scene.
    """
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError('0079 requires a nonempty mesh')
    faces = np.asarray(mesh.faces, dtype=np.int64)
    labels, counts = _edge_component_labels(faces)
    winner = int(np.argmax(counts))
    keep = labels == winner
    ratio = float(counts[winner] / len(faces))
    if ratio < float(min_main_fraction):
        raise ValueError(f'0079 main face component only {ratio:.1%}; cannot clean safely')
    vertices = np.asarray(mesh.vertices).copy()
    kept = trimesh.Trimesh(vertices=vertices, faces=faces[keep].copy(), process=False)
    detached = trimesh.Trimesh(vertices=vertices.copy(), faces=faces[~keep].copy(), process=False)
    kept.remove_unreferenced_vertices()
    if len(detached.faces):
        detached.remove_unreferenced_vertices()
    return kept, detached, {
        'mode': 'largest_edge_connected_component',
        'components_before': int(len(counts)),
        'largest_faces': int(counts[winner]),
        'main_fraction': ratio,
        'detached_faces': int(np.count_nonzero(~keep)),
        'detached_components': int(len(counts) - 1),
        'boundary_smoothing': False,
    }
