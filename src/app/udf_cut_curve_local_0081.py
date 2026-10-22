"""0081 experimental, mesh- and curve-local UDF expansion on final root/leaf parts.

Never treats Euclidean proximity to a folded-back skeleton as ownership.
The cut begins at independently agreed 0079 seed faces and grows only through
shared mesh edges, with continuous target-curve *arclength* and a bounded
surface walk. Unknown faces are candidates, NOT unconditional removals.

This is a diagnostic geometry preview; anatomical correctness is unverified.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class LocalExpansionSettings:
    max_curve_fraction: float = 0.11  # curve length fraction beyond seed component
    max_neighbor_curve_jump: float = 0.07  # ban shortcuts at folds
    max_surface_walk_fraction: float = 0.40  # of full curve world length
    min_seed_component_faces: int = 8  # smaller seeds remain, but do not grow

    def __post_init__(self):
        for key in ('max_curve_fraction', 'max_neighbor_curve_jump',
                    'max_surface_walk_fraction'):
            value = float(getattr(self, key))
            if not np.isfinite(value) or value <= 0 or value > 1:
                raise ValueError(f'0081 {key} must be in (0, 1]')
        if int(self.min_seed_component_faces) < 1:
            raise ValueError('0081 min_seed_component_faces must be >= 1')


def project_faces_to_curve_arclength(mesh: trimesh.Trimesh, curve: Any,
                                     *, samples: int = 1025):
    """Return normalized cumulative *arc* coordinate per triangle, not XYZ distance.

    For self-near curves, this nearest projection may be ambiguous; the local
    mesh-edge arclength-continuity check below *blocks* large discontinuities.
    """
    if int(samples) < 32 or int(samples) > 8192:
        raise ValueError('0081 curve samples must be 32..8192')
    ts = np.asarray(curve.core.key_ts, dtype=np.float64).reshape(-1)
    if len(ts) < 2 or not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
        raise ValueError('0081 invalid curve key_ts')
    sample_ts = np.linspace(ts[0], ts[-1], int(samples))
    points = np.asarray(curve.core.interpolate(sample_ts)['points'], dtype=np.float64)
    if points.shape != (len(sample_ts), 3) or not np.isfinite(points).all():
        raise ValueError('0081 invalid sampled curve points')
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.r_[0, np.cumsum(lengths)]
    total = float(cumulative[-1])
    if not np.isfinite(total) or total < 1e-8:
        raise ValueError('0081 target curve has zero or invalid arclength')
    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    idx = np.asarray(cKDTree(points).query(centers, k=1)[1], dtype=np.int64)
    best = np.full(len(centers), np.inf)
    arc = np.zeros(len(centers), dtype=np.float64)
    for start in (np.clip(idx - 1, 0, len(points) - 2),
                  np.clip(idx, 0, len(points) - 2)):
        p = points[start]
        v = points[start + 1] - p
        len2 = np.einsum('ij,ij->i', v, v)
        frac = np.divide(np.einsum('ij,ij->i', centers - p, v), len2,
                         out=np.zeros(len(centers)), where=len2 > 1e-20)
        np.clip(frac, 0, 1, out=frac)
        closest = p + frac[:, None] * v
        d2 = np.einsum('ij,ij->i', centers - closest, centers - closest)
        better = d2 < best
        best[better] = d2[better]
        arc[better] = (cumulative[start[better]] + frac[better] * lengths[start[better]]) / total
    return arc, total


def _seed_groups(seeds: np.ndarray, edges: np.ndarray):
    """Connected seed components under shared-face-edge connectivity."""
    ids = np.flatnonzero(seeds)
    if not len(ids):
        return []
    renumber = np.full(len(seeds), -1, dtype=np.int64)
    renumber[ids] = np.arange(len(ids))
    good = seeds[edges[:, 0]] & seeds[edges[:, 1]]
    pairs = renumber[edges[good]]
    rows = np.r_[pairs[:, 0], pairs[:, 1]] if len(pairs) else np.empty(0, dtype=np.int64)
    cols = np.r_[pairs[:, 1], pairs[:, 0]] if len(pairs) else np.empty(0, dtype=np.int64)
    graph = coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)),
                       shape=(len(ids), len(ids))).tocsr()
    n, labels = connected_components(graph, directed=False)
    return [ids[labels == j] for j in range(n)]


def expand_face_mask(mesh: trimesh.Trimesh, *, seed_mask, candidate_mask,
                     curve_s, curve_length_world: float,
                     settings: LocalExpansionSettings = LocalExpansionSettings()):
    """Geodesic flood fill with *per-seed-component* curve-local boundaries.

    Seed triangles remain selected even when tiny, but seed components smaller
    than min_seed_component_faces cannot recruit unsupported neighbors.
    """
    n = len(mesh.faces)
    seeds = np.asarray(seed_mask, dtype=bool).reshape(-1)
    candidates = np.asarray(candidate_mask, dtype=bool).reshape(-1)
    s = np.asarray(curve_s, dtype=np.float64).reshape(-1)
    if not (n == len(seeds) == len(candidates) == len(s)):
        raise ValueError('0081 face-mask / curve-length mismatch')
    if not np.all(np.isfinite(s)) or np.min(s, initial=0) < -1e-8 or np.max(s, initial=0) > 1 + 1e-8:
        raise ValueError('0081 invalid face curve coordinates')
    if not np.all(seeds <= candidates):
        raise ValueError('0081 seed faces must belong to four-class candidates')
    if not np.isfinite(curve_length_world) or curve_length_world <= 0:
        raise ValueError('0081 invalid curve length')
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    centroids = np.asarray(mesh.triangles_center, dtype=np.float64)
    # Unconnected but spatially nearby arms are NEVER graph neighbors.
    neighbor_jump = np.abs(s[adjacency[:, 0]] - s[adjacency[:, 1]]) if len(adjacency) else np.empty(0)
    links = adjacency[neighbor_jump <= settings.max_neighbor_curve_jump]
    # Explicit adjacency lists outperform a sparse repeated neighborhood scan.
    graph = [[] for _ in range(n)]
    for a, b in links:
        graph[int(a)].append(int(b))
        graph[int(b)].append(int(a))
    output = seeds.copy()
    seed_components = _seed_groups(seeds, adjacency)
    geodesic_limit = settings.max_surface_walk_fraction * curve_length_world
    newly_added_by_group = []
    small_groups = 0
    skipped_bad_param = int(np.count_nonzero(neighbor_jump > settings.max_neighbor_curve_jump))
    visited = np.full(n, -1, dtype=np.int32)
    for group_i, group in enumerate(seed_components):
        if len(group) < settings.min_seed_component_faces:
            small_groups += 1
            continue
        # Robust seed span ignores occasional bad self-near projections while
        # keeping all seed triangles in the output unchanged.
        seed_s = s[group]
        low, high = np.quantile(seed_s, [0.02, 0.98])
        low = max(0., low - settings.max_curve_fraction)
        high = min(1., high + settings.max_curve_fraction)
        # Individual seed components are independent; their bands must not
        # be merged just because they touch different sections of a hairpin.
        distances = {}
        heap = []
        for face in group:
            i = int(face)
            distances[i] = 0.
            heap.append((0., i))
        heapq.heapify(heap)
        added = 0
        while heap:
            dist, current = heapq.heappop(heap)
            if dist > distances[current] + 1e-12:
                continue
            if dist > geodesic_limit:
                break
            for nb in graph[current]:
                if not candidates[nb] or not (low <= s[nb] <= high):
                    continue
                cost = float(np.linalg.norm(centroids[nb] - centroids[current]))
                next_dist = dist + cost
                if next_dist > geodesic_limit or next_dist >= distances.get(nb, np.inf):
                    continue
                distances[nb] = next_dist
                heapq.heappush(heap, (next_dist, nb))
                if not output[nb]:
                    output[nb] = True
                    added += 1
        newly_added_by_group.append(added)
    new = output & ~seeds
    blocked = candidates & ~output
    info = {
        'mode': 'surface_and_arclength_local_0081',
        'candidate_faces': int(np.count_nonzero(candidates)),
        'seed_faces': int(np.count_nonzero(seeds)),
        'added_faces': int(np.count_nonzero(new)),
        'blocked_candidates': int(np.count_nonzero(blocked)),
        'selected_faces': int(np.count_nonzero(output)),
        'seed_components': int(len(seed_components)),
        'small_seed_components_not_expanded': int(small_groups),
        'blocked_mesh_links_curve_jump': skipped_bad_param,
        'curve_length_world': float(curve_length_world),
        'max_walk_world': float(geodesic_limit),
        'max_curve_fraction': float(settings.max_curve_fraction),
        'max_neighbor_curve_jump': float(settings.max_neighbor_curve_jump),
        'seed_group_new_faces': newly_added_by_group,
    }
    return output, info
