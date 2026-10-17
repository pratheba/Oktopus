"""Use an adapted UDF as a world-space filter for an existing SDF/MC mesh.

This does *not* mesh the UDF.  The input Marching-Cubes mesh remains the
geometry source.  The adapted UDF is queried at the MC triangle centroid and
three edge midpoints using the same target/accessory handles and the same
adaptation YAML.  Connected high-UDF face patches are removed as likely false
caps across intentional openings.

Because the UDF is evaluated directly at the MC vertices in world space, there
is no independent UDF extraction cube, bbox normalization, or post-hoc scale
alignment.
"""

from __future__ import annotations

import json
import os
import os.path as op
from time import perf_counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import trimesh

from agent_3dvec_udf import AgentUDF
from seam_optimization import build_seam_plan
from udf_cut_boundary_cleanup import (
    boundary_point_cloud,
    cleanup_cut_boundary,
)
from snug_field_io import (
    load_shared_snug_field,
    resolve_shared_snug_field_path,
    snug_field_stats,
)
from udf_cut_subtriangle import clip_accepted_udf_components
from udf_cut_combined_transfer import transfer_udf_cut_to_combined_mesh


@dataclass
class _ComponentDecision:
    component_id: int
    face_count: int
    seed_count: int
    seed_fraction: float
    area: float
    score_min: float
    score_median: float
    score_max: float
    ring_face_count: int
    ring_score_median: Optional[float]
    touches_existing_boundary: bool
    reference_distance: Optional[float]
    accepted: bool
    reason: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "component_id": self.component_id,
            "face_count": self.face_count,
            "seed_count": self.seed_count,
            "seed_fraction": self.seed_fraction,
            "area": self.area,
            "score_min": self.score_min,
            "score_median": self.score_median,
            "score_max": self.score_max,
            "ring_face_count": self.ring_face_count,
            "ring_score_median": self.ring_score_median,
            "touches_existing_boundary": self.touches_existing_boundary,
            "reference_distance": self.reference_distance,
            "accepted": self.accepted,
            "reason": self.reason,
        }


class AgentUDFCut(AgentUDF):
    """Direct adapted-UDF cap removal for an already reconstructed MC mesh."""

    @staticmethod
    def _safe_label(text: str) -> str:
        return str(text).replace("|", "_").replace("/", "_")

    @staticmethod
    def _resolve_path_spec(
        spec: str,
        *,
        root_path: str,
        item_index: int,
        mode: str,
        accessory_key: str,
        target_key: str,
    ) -> str:
        if not spec:
            raise ValueError("A mesh path specification is required.")

        formatted = str(spec).format(
            index=item_index,
            mode=mode,
            accessory=AgentUDFCut._safe_label(accessory_key),
            target=AgentUDFCut._safe_label(target_key),
        )
        path = formatted if op.isabs(formatted) else op.join(root_path, formatted)
        path = op.abspath(op.expanduser(path))
        if not op.isfile(path):
            raise FileNotFoundError(f"Mesh file not found: {path}")
        return path

    @staticmethod
    def _mesh_from_face_mask(mesh: trimesh.Trimesh, mask: np.ndarray) -> trimesh.Trimesh:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if mask.shape[0] != len(mesh.faces):
            raise ValueError(
                f"Face mask length mismatch: {mask.shape[0]} vs {len(mesh.faces)}"
            )
        if not np.any(mask):
            return trimesh.Trimesh(
                vertices=np.zeros((0, 3), dtype=np.float64),
                faces=np.zeros((0, 3), dtype=np.int64),
                process=False,
            )
        out = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices, dtype=np.float64).copy(),
            faces=np.asarray(mesh.faces, dtype=np.int64)[mask].copy(),
            process=False,
        )
        out.remove_unreferenced_vertices()
        return out

    @staticmethod
    def _topology_stats(mesh: trimesh.Trimesh) -> Dict[str, object]:
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if len(faces) == 0:
            return {
                "vertices": int(len(mesh.vertices)),
                "faces": 0,
                "boundary_edges": 0,
                "nonmanifold_edges": 0,
                "components": 0,
            }

        edges = np.sort(
            np.concatenate(
                [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
                axis=0,
            ),
            axis=1,
        )
        _unique_edges, counts = np.unique(edges, axis=0, return_counts=True)

        # Vertex-connected component count.  This works even for non-manifold
        # intermediate meshes and does not require trimesh.split().
        n_vertices = len(mesh.vertices)
        parent = np.arange(n_vertices, dtype=np.int64)
        rank = np.zeros(n_vertices, dtype=np.int8)

        def find(x: int) -> int:
            x = int(x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = int(parent[x])
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

        for tri in faces:
            union(int(tri[0]), int(tri[1]))
            union(int(tri[1]), int(tri[2]))
            union(int(tri[2]), int(tri[0]))

        roots = {find(int(v)) for v in np.unique(faces)}
        return {
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(faces)),
            "boundary_edges": int(np.sum(counts == 1)),
            "nonmanifold_edges": int(np.sum(counts > 2)),
            "components": int(len(roots)),
        }

    @staticmethod
    def _existing_boundary_faces(mesh: trimesh.Trimesh) -> np.ndarray:
        faces = np.asarray(mesh.faces, dtype=np.int64)
        result = np.zeros(len(faces), dtype=bool)
        if len(faces) == 0:
            return result

        face_ids = np.tile(np.arange(len(faces), dtype=np.int64), 3)
        edges = np.concatenate(
            [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
        )
        edges = np.sort(edges, axis=1)
        _, inverse, counts = np.unique(
            edges, axis=0, return_inverse=True, return_counts=True
        )
        result[face_ids[counts[inverse] == 1]] = True
        return result

    @staticmethod
    def _face_components_fast(
        mask: np.ndarray, adjacency: np.ndarray
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Return active face components using SciPy's compiled graph traversal.

        The automatic percentile search evaluates the same large MC mesh many
        times.  The old helper ran a Python union/find loop over every face
        adjacency for every candidate pair, which dominates runtime on 512+
        meshes.  This helper preserves the same face-adjacency notion of a
        component while moving the graph traversal into scipy.sparse.csgraph.

        The returned ``labels`` array is indexed by global face id.  Inactive
        faces are -1; active faces contain a compact component id matching the
        returned component list.
        """
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        active = np.flatnonzero(mask)
        labels_out = np.full(mask.shape[0], -1, dtype=np.int64)
        if active.size == 0:
            return [], labels_out

        adjacency = np.asarray(adjacency, dtype=np.int64)
        if adjacency.size == 0:
            labels_out[active] = np.arange(active.size, dtype=np.int64)
            return [np.asarray([f], dtype=np.int64) for f in active], labels_out

        try:
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components
        except Exception:
            # Keep the historical implementation as a correctness fallback.
            components = AgentUDFCut._face_components(mask, adjacency)
            for component_id, face_ids in enumerate(components):
                labels_out[face_ids] = component_id
            return components, labels_out

        # Work only on active faces so the sparse graph scales with the grow
        # region rather than with the complete mesh.
        local = np.full(mask.shape[0], -1, dtype=np.int64)
        local[active] = np.arange(active.size, dtype=np.int64)
        a = adjacency[:, 0]
        b = adjacency[:, 1]
        keep = mask[a] & mask[b]
        aa = local[a[keep]]
        bb = local[b[keep]]

        rows = np.concatenate([aa, bb])
        cols = np.concatenate([bb, aa])
        data = np.ones(rows.shape[0], dtype=np.uint8)
        graph = coo_matrix(
            (data, (rows, cols)), shape=(active.size, active.size)
        ).tocsr()
        _n_components, local_labels = connected_components(
            graph, directed=False, return_labels=True
        )

        # Group without a Python pass over every active face.  Stable sorting
        # also leaves face ids ascending inside a component, matching the old
        # helper's deterministic traversal closely.
        order = np.argsort(local_labels, kind="stable")
        sorted_faces = active[order]
        sorted_labels = local_labels[order]
        cuts = np.flatnonzero(np.diff(sorted_labels)) + 1
        components = [
            np.asarray(chunk, dtype=np.int64)
            for chunk in np.split(sorted_faces, cuts)
            if len(chunk)
        ]
        components.sort(key=lambda ids: int(ids[0]))
        for component_id, face_ids in enumerate(components):
            labels_out[face_ids] = component_id
        return components, labels_out

    @staticmethod
    def _component_rings_fast(
        grow_mask: np.ndarray,
        component_labels: np.ndarray,
        adjacency: np.ndarray,
        n_components: int,
    ) -> List[np.ndarray]:
        """Compute all one-face component rings in one vectorized pass."""
        rings = [np.zeros((0,), dtype=np.int64) for _ in range(n_components)]
        if n_components == 0:
            return rings

        grow_mask = np.asarray(grow_mask, dtype=bool).reshape(-1)
        adjacency = np.asarray(adjacency, dtype=np.int64)
        if adjacency.size == 0:
            return rings

        a = adjacency[:, 0]
        b = adjacency[:, 1]
        cross = grow_mask[a] ^ grow_mask[b]
        if not np.any(cross):
            return rings

        aa = a[cross]
        bb = b[cross]
        a_inside = grow_mask[aa]
        inside = np.where(a_inside, aa, bb)
        outside = np.where(a_inside, bb, aa)
        comp = component_labels[inside]
        valid = comp >= 0
        if not np.any(valid):
            return rings

        pairs = np.stack([comp[valid], outside[valid]], axis=1)
        pairs = np.unique(pairs, axis=0)
        order = np.argsort(pairs[:, 0], kind="stable")
        pairs = pairs[order]
        comp_ids = pairs[:, 0]
        cuts = np.flatnonzero(np.diff(comp_ids)) + 1
        for chunk in np.split(pairs, cuts):
            if len(chunk):
                rings[int(chunk[0, 0])] = np.asarray(chunk[:, 1], dtype=np.int64)
        return rings

    @staticmethod
    def _prepare_search_topology(mesh: trimesh.Trimesh) -> Dict[str, np.ndarray]:
        """Precompute immutable edge data reused by every threshold pair."""
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if len(faces) == 0:
            return {
                "faces": faces,
                "unique_edges": np.zeros((0, 2), dtype=np.int64),
                "face_edge_ids": np.zeros((0, 3), dtype=np.int64),
                "edge_total_counts": np.zeros((0,), dtype=np.int64),
                "n_vertices": np.asarray([len(mesh.vertices)], dtype=np.int64),
            }

        face_edges = np.stack(
            [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=1
        )
        flat_edges = np.sort(face_edges.reshape(-1, 2), axis=1)
        unique_edges, inverse, counts = np.unique(
            flat_edges, axis=0, return_inverse=True, return_counts=True
        )
        return {
            "faces": faces,
            "unique_edges": np.asarray(unique_edges, dtype=np.int64),
            "face_edge_ids": np.asarray(inverse, dtype=np.int64).reshape(-1, 3),
            "edge_total_counts": np.asarray(counts, dtype=np.int64),
            "n_vertices": np.asarray([len(mesh.vertices)], dtype=np.int64),
        }

    @staticmethod
    def _component_count_from_edge_counts(
        topology: Dict[str, np.ndarray], edge_counts: np.ndarray
    ) -> Tuple[int, int]:
        """Return (used vertex count, vertex-connected component count)."""
        edge_counts = np.asarray(edge_counts, dtype=np.int64).reshape(-1)
        selected_edges = topology["unique_edges"][edge_counts > 0]
        if len(selected_edges) == 0:
            return 0, 0

        used_vertices = np.unique(selected_edges.reshape(-1))
        try:
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components
        except Exception:
            # Rare fallback: a compact union/find over only selected edges.
            parent = np.arange(used_vertices.size, dtype=np.int64)
            rank = np.zeros(used_vertices.size, dtype=np.int8)
            local = np.full(
                int(topology["n_vertices"][0]), -1, dtype=np.int64
            )
            local[used_vertices] = np.arange(used_vertices.size, dtype=np.int64)

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = int(parent[x])
                return x

            def union(a: int, b: int) -> None:
                ra, rb = find(a), find(b)
                if ra == rb:
                    return
                if rank[ra] < rank[rb]:
                    parent[ra] = rb
                elif rank[ra] > rank[rb]:
                    parent[rb] = ra
                else:
                    parent[rb] = ra
                    rank[ra] += 1

            for a, b in local[selected_edges]:
                union(int(a), int(b))
            roots = {find(i) for i in range(used_vertices.size)}
            return int(used_vertices.size), int(len(roots))

        local = np.full(int(topology["n_vertices"][0]), -1, dtype=np.int64)
        local[used_vertices] = np.arange(used_vertices.size, dtype=np.int64)
        local_edges = local[selected_edges]
        rows = np.concatenate([local_edges[:, 0], local_edges[:, 1]])
        cols = np.concatenate([local_edges[:, 1], local_edges[:, 0]])
        data = np.ones(rows.shape[0], dtype=np.uint8)
        graph = coo_matrix(
            (data, (rows, cols)),
            shape=(used_vertices.size, used_vertices.size),
        ).tocsr()
        n_components = connected_components(
            graph, directed=False, return_labels=False
        )
        return int(used_vertices.size), int(n_components)

    @classmethod
    def _topology_stats_from_remove_mask(
        cls,
        topology: Dict[str, np.ndarray],
        remove_mask: np.ndarray,
    ) -> Tuple[Dict[str, object], Dict[str, object]]:
        """Compute kept/removed topology directly from a face mask.

        This avoids constructing two Trimesh objects and calling ``np.unique``
        on all triangle edges for every percentile candidate.
        """
        remove_mask = np.asarray(remove_mask, dtype=bool).reshape(-1)
        face_edge_ids = topology["face_edge_ids"]
        n_edges = len(topology["edge_total_counts"])
        if np.any(remove_mask):
            removed_edge_counts = np.bincount(
                face_edge_ids[remove_mask].reshape(-1), minlength=n_edges
            ).astype(np.int64, copy=False)
        else:
            removed_edge_counts = np.zeros(n_edges, dtype=np.int64)
        kept_edge_counts = topology["edge_total_counts"] - removed_edge_counts

        kept_vertices, kept_components = cls._component_count_from_edge_counts(
            topology, kept_edge_counts
        )
        removed_vertices, removed_components = cls._component_count_from_edge_counts(
            topology, removed_edge_counts
        )

        kept_stats = {
            "vertices": kept_vertices,
            "faces": int(np.sum(~remove_mask)),
            "boundary_edges": int(np.sum(kept_edge_counts == 1)),
            "nonmanifold_edges": int(np.sum(kept_edge_counts > 2)),
            "components": kept_components,
        }
        removed_stats = {
            "vertices": removed_vertices,
            "faces": int(np.sum(remove_mask)),
            "boundary_edges": int(np.sum(removed_edge_counts == 1)),
            "nonmanifold_edges": int(np.sum(removed_edge_counts > 2)),
            "components": removed_components,
        }
        return kept_stats, removed_stats

    @classmethod
    def _evaluate_cut_candidate_fast(
        cls,
        mesh: trimesh.Trimesh,
        *,
        face_scores: np.ndarray,
        usable_faces: np.ndarray,
        seed_world: float,
        grow_world: float,
        min_faces: int,
        min_seed_faces: int,
        min_seed_fraction: float,
        min_area_world2: float,
        preserve_existing_boundaries: bool,
        reference_tree,
        reference_max_distance_world: Optional[float],
        adjacency: np.ndarray,
        face_areas: np.ndarray,
        existing_boundary_faces: np.ndarray,
        centroids: np.ndarray,
        topology: Dict[str, np.ndarray],
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        List[_ComponentDecision],
        np.ndarray,
        Dict[str, object],
        Dict[str, object],
    ]:
        """Evaluate one seed/grow pair without materializing candidate meshes."""
        if grow_world > seed_world:
            raise ValueError(
                f"udf_cut_grow_world ({grow_world}) must be <= "
                f"udf_cut_seed_world ({seed_world})."
            )

        scores = np.asarray(face_scores, dtype=np.float64).reshape(-1)
        usable = np.asarray(usable_faces, dtype=bool).reshape(-1)
        seed_mask = usable & (scores >= float(seed_world))
        grow_mask = usable & (scores >= float(grow_world))

        components, labels = cls._face_components_fast(grow_mask, adjacency)
        rings = cls._component_rings_fast(
            grow_mask, labels, adjacency, len(components)
        )
        remove_mask = np.zeros(len(mesh.faces), dtype=bool)
        decisions: List[_ComponentDecision] = []

        for component_id, face_ids in enumerate(components):
            seed_count = int(np.sum(seed_mask[face_ids]))
            face_count = int(len(face_ids))
            seed_fraction = float(seed_count / max(face_count, 1))
            area = float(np.sum(face_areas[face_ids]))
            comp_scores = scores[face_ids]
            ring = rings[component_id]
            ring_valid = ring[np.isfinite(scores[ring])] if len(ring) else ring
            ring_median = (
                float(np.median(scores[ring_valid])) if len(ring_valid) else None
            )
            touches_boundary = bool(np.any(existing_boundary_faces[face_ids]))

            reference_distance = None
            if reference_tree is not None:
                probe_faces = ring if len(ring) else face_ids
                distances, _ = reference_tree.query(centroids[probe_faces], k=1)
                reference_distance = float(np.min(np.asarray(distances)))

            accepted = True
            reason = "accepted"
            if face_count < int(min_faces):
                accepted, reason = False, "too_few_faces"
            elif seed_count < int(min_seed_faces):
                accepted, reason = False, "too_few_seed_faces"
            elif seed_fraction < float(min_seed_fraction):
                accepted, reason = False, "seed_fraction_too_low"
            elif area < float(min_area_world2):
                accepted, reason = False, "area_too_small"
            elif preserve_existing_boundaries and touches_boundary:
                accepted, reason = False, "touches_existing_boundary"
            elif (
                reference_tree is not None
                and reference_max_distance_world is not None
                and reference_distance is not None
                and reference_distance > float(reference_max_distance_world)
            ):
                accepted, reason = False, "far_from_reference_boundary"

            if accepted:
                remove_mask[face_ids] = True

            decisions.append(
                _ComponentDecision(
                    component_id=component_id,
                    face_count=face_count,
                    seed_count=seed_count,
                    seed_fraction=seed_fraction,
                    area=area,
                    score_min=float(np.min(comp_scores)),
                    score_median=float(np.median(comp_scores)),
                    score_max=float(np.max(comp_scores)),
                    ring_face_count=int(len(ring)),
                    ring_score_median=ring_median,
                    touches_existing_boundary=touches_boundary,
                    reference_distance=reference_distance,
                    accepted=accepted,
                    reason=reason,
                )
            )

        kept_stats, removed_stats = cls._topology_stats_from_remove_mask(
            topology, remove_mask
        )
        return (
            seed_mask,
            grow_mask,
            decisions,
            remove_mask,
            kept_stats,
            removed_stats,
        )

    @staticmethod
    def _face_components(mask: np.ndarray, adjacency: np.ndarray) -> List[np.ndarray]:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        active = np.flatnonzero(mask)
        if active.size == 0:
            return []

        parent = np.arange(mask.shape[0], dtype=np.int64)
        rank = np.zeros(mask.shape[0], dtype=np.int8)

        def find(x: int) -> int:
            x = int(x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = int(parent[x])
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

        for pair in np.asarray(adjacency, dtype=np.int64):
            a, b = int(pair[0]), int(pair[1])
            if mask[a] and mask[b]:
                union(a, b)

        groups: Dict[int, List[int]] = {}
        for face_id in active:
            groups.setdefault(find(int(face_id)), []).append(int(face_id))
        return [np.asarray(ids, dtype=np.int64) for ids in groups.values()]

    @staticmethod
    def _component_ring(
        component_faces: np.ndarray,
        adjacency: np.ndarray,
        n_faces: int,
    ) -> np.ndarray:
        in_component = np.zeros(n_faces, dtype=bool)
        in_component[np.asarray(component_faces, dtype=np.int64)] = True
        ring = []
        for a, b in np.asarray(adjacency, dtype=np.int64):
            a, b = int(a), int(b)
            if in_component[a] and not in_component[b]:
                ring.append(b)
            elif in_component[b] and not in_component[a]:
                ring.append(a)
        if not ring:
            return np.zeros((0,), dtype=np.int64)
        return np.unique(np.asarray(ring, dtype=np.int64))

    @staticmethod
    def _exact_weld_for_boundary(mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray]:
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if len(vertices) == 0 or len(faces) == 0:
            return vertices, faces
        unique_vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
        welded_faces = inverse[faces]
        repeated = (
            (welded_faces[:, 0] == welded_faces[:, 1])
            | (welded_faces[:, 1] == welded_faces[:, 2])
            | (welded_faces[:, 2] == welded_faces[:, 0])
        )
        return unique_vertices, welded_faces[~repeated]

    @classmethod
    def _large_reference_boundary_points(
        cls,
        reference_mesh: trimesh.Trimesh,
        *,
        min_edges: int,
        min_span_world: float,
    ) -> np.ndarray:
        vertices, faces = cls._exact_weld_for_boundary(reference_mesh)
        if len(faces) == 0:
            return np.zeros((0, 3), dtype=np.float64)

        edges = np.sort(
            np.concatenate(
                [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
                axis=0,
            ),
            axis=1,
        )
        unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
        boundary_edges = unique_edges[counts == 1]
        if len(boundary_edges) == 0:
            return np.zeros((0, 3), dtype=np.float64)

        boundary_vertices = np.unique(boundary_edges.reshape(-1))
        parent = {int(v): int(v) for v in boundary_vertices}

        def find(v: int) -> int:
            while parent[v] != v:
                parent[v] = parent[parent[v]]
                v = parent[v]
            return v

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for a, b in boundary_edges:
            union(int(a), int(b))

        edge_groups: Dict[int, List[Tuple[int, int]]] = {}
        for a, b in boundary_edges:
            edge_groups.setdefault(find(int(a)), []).append((int(a), int(b)))

        kept_points = []
        for group_edges in edge_groups.values():
            if len(group_edges) < int(min_edges):
                continue
            ids = np.unique(np.asarray(group_edges, dtype=np.int64).reshape(-1))
            points = vertices[ids]
            span = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
            if span < float(min_span_world):
                continue
            kept_points.append(points)

        if not kept_points:
            return np.zeros((0, 3), dtype=np.float64)
        return np.concatenate(kept_points, axis=0)

    def _query_face_scores(
        self,
        mesh: trimesh.Trimesh,
        *,
        avatar_curve,
        adapt_arg: dict,
        accessory_key: str,
        model_batch_size: int,
        query_batch_size: int,
        far_world: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        tri = vertices[faces]

        samples = np.stack(
            [
                tri.mean(axis=1),
                0.5 * (tri[:, 0] + tri[:, 1]),
                0.5 * (tri[:, 1] + tri[:, 2]),
                0.5 * (tri[:, 2] + tri[:, 0]),
            ],
            axis=1,
        )
        flat = samples.reshape(-1, 3)
        values = np.empty(flat.shape[0], dtype=np.float64)
        valid = np.zeros(flat.shape[0], dtype=bool)

        query_batch_size = max(1, int(query_batch_size))
        for start in range(0, flat.shape[0], query_batch_size):
            end = min(start + query_batch_size, flat.shape[0])
            value_chunk, valid_chunk = self._query_adapted_raw_udf(
                flat[start:end],
                avatar_curve=avatar_curve,
                adapt_arg=adapt_arg,
                accessory_key=accessory_key,
                batch_size=int(model_batch_size),
                far_world=float(far_world),
                return_valid=True,
            )
            values[start:end] = np.asarray(value_chunk, dtype=np.float64)
            valid[start:end] = np.asarray(valid_chunk, dtype=bool)

        values = values.reshape(len(faces), 4)
        valid = valid.reshape(len(faces), 4)
        masked = np.where(valid, values, np.nan)

        valid_counts = valid.sum(axis=1)

        # A face is usable only when its centroid is valid and at least one
        # additional sample is valid.
        candidate = valid[:, 0] & (valid_counts >= 2)

        scores = np.full(len(faces), np.nan, dtype=np.float64)

        # Do not pass all-NaN rows to np.nanmedian.
        if np.any(candidate):
            scores[candidate] = np.nanmedian(masked[candidate], axis=1)

        usable = candidate & np.isfinite(scores)
        scores[~usable] = np.nan
        return scores, usable, values, valid

    def _query_vertex_values(
        self,
        mesh: trimesh.Trimesh,
        *,
        avatar_curve,
        adapt_arg: dict,
        accessory_key: str,
        model_batch_size: int,
        query_batch_size: int,
        far_world: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the adapted UDF once at every MC vertex."""
        points = np.asarray(mesh.vertices, dtype=np.float64)
        values = np.empty(len(points), dtype=np.float64)
        valid = np.zeros(len(points), dtype=bool)

        query_batch_size = max(1, int(query_batch_size))
        for start in range(0, len(points), query_batch_size):
            end = min(start + query_batch_size, len(points))
            value_chunk, valid_chunk = self._query_adapted_raw_udf(
                points[start:end],
                avatar_curve=avatar_curve,
                adapt_arg=adapt_arg,
                accessory_key=accessory_key,
                batch_size=int(model_batch_size),
                far_world=float(far_world),
                return_valid=True,
            )
            values[start:end] = np.asarray(value_chunk, dtype=np.float64)
            valid[start:end] = np.asarray(valid_chunk, dtype=bool)

        valid &= np.isfinite(values)
        values[~valid] = np.nan
        return values, valid

    def _cut_mesh(
        self,
        mesh: trimesh.Trimesh,
        *,
        face_scores: np.ndarray,
        usable_faces: np.ndarray,
        seed_world: float,
        grow_world: float,
        min_faces: int,
        min_seed_faces: int,
        min_seed_fraction: float,
        min_area_world2: float,
        preserve_existing_boundaries: bool,
        reference_points: Optional[np.ndarray],
        reference_max_distance_world: Optional[float],
    ) -> Tuple[
        trimesh.Trimesh,
        trimesh.Trimesh,
        np.ndarray,
        np.ndarray,
        List[_ComponentDecision],
        np.ndarray,
    ]:
        if grow_world > seed_world:
            raise ValueError(
                f"udf_cut_grow_world ({grow_world}) must be <= "
                f"udf_cut_seed_world ({seed_world})."
            )

        scores = np.asarray(face_scores, dtype=np.float64).reshape(-1)
        usable = np.asarray(usable_faces, dtype=bool).reshape(-1)
        seed_mask = usable & (scores >= float(seed_world))
        grow_mask = usable & (scores >= float(grow_world))

        adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
        components = self._face_components(grow_mask, adjacency)
        face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
        existing_boundary_faces = self._existing_boundary_faces(mesh)
        centroids = np.asarray(mesh.triangles_center, dtype=np.float64)

        reference_tree = None
        if reference_points is not None and len(reference_points):
            try:
                from scipy.spatial import cKDTree

                reference_tree = cKDTree(np.asarray(reference_points, dtype=np.float64))
            except Exception as exc:
                print("[udf cut] reference KD-tree unavailable:", exc)

        remove_mask = np.zeros(len(mesh.faces), dtype=bool)
        decisions: List[_ComponentDecision] = []

        for component_id, face_ids in enumerate(components):
            seed_count = int(np.sum(seed_mask[face_ids]))
            face_count = int(len(face_ids))
            seed_fraction = float(seed_count / max(face_count, 1))
            area = float(np.sum(face_areas[face_ids]))
            comp_scores = scores[face_ids]
            ring = self._component_ring(face_ids, adjacency, len(mesh.faces))
            ring_valid = ring[np.isfinite(scores[ring])] if len(ring) else ring
            ring_median = (
                float(np.median(scores[ring_valid])) if len(ring_valid) else None
            )
            touches_boundary = bool(np.any(existing_boundary_faces[face_ids]))

            reference_distance = None
            if reference_tree is not None:
                # The component boundary is the useful comparison target.  If
                # the ring is empty, fall back to the component centroids.
                probe_faces = ring if len(ring) else face_ids
                distances, _ = reference_tree.query(centroids[probe_faces], k=1)
                reference_distance = float(np.min(np.asarray(distances)))

            accepted = True
            reason = "accepted"
            if face_count < int(min_faces):
                accepted, reason = False, "too_few_faces"
            elif seed_count < int(min_seed_faces):
                accepted, reason = False, "too_few_seed_faces"
            elif seed_fraction < float(min_seed_fraction):
                accepted, reason = False, "seed_fraction_too_low"
            elif area < float(min_area_world2):
                accepted, reason = False, "area_too_small"
            elif preserve_existing_boundaries and touches_boundary:
                accepted, reason = False, "touches_existing_boundary"
            elif (
                reference_tree is not None
                and reference_max_distance_world is not None
                and reference_distance is not None
                and reference_distance > float(reference_max_distance_world)
            ):
                accepted, reason = False, "far_from_reference_boundary"

            if accepted:
                remove_mask[face_ids] = True

            decisions.append(
                _ComponentDecision(
                    component_id=component_id,
                    face_count=face_count,
                    seed_count=seed_count,
                    seed_fraction=seed_fraction,
                    area=area,
                    score_min=float(np.min(comp_scores)),
                    score_median=float(np.median(comp_scores)),
                    score_max=float(np.max(comp_scores)),
                    ring_face_count=int(len(ring)),
                    ring_score_median=ring_median,
                    touches_existing_boundary=touches_boundary,
                    reference_distance=reference_distance,
                    accepted=accepted,
                    reason=reason,
                )
            )

        removed = self._mesh_from_face_mask(mesh, remove_mask)
        kept = self._mesh_from_face_mask(mesh, ~remove_mask)
        return kept, removed, seed_mask, grow_mask, decisions, remove_mask

    @staticmethod
    def _boundary_cleanup_kwargs(config: dict) -> Dict[str, object]:
        return {
            "fill_small_holes": bool(config.get("udf_cut_fill_small_holes", True)),
            "fill_max_edges": int(config.get("udf_cut_fill_hole_max_edges", 24)),
            "fill_max_perimeter_world": float(
                config.get("udf_cut_fill_hole_max_perimeter_world", 0.08)
            ),
            "fill_max_span_world": float(
                config.get("udf_cut_fill_hole_max_span_world", 0.04)
            ),
            "strip_rebuild_enabled": bool(
                config.get("udf_cut_boundary_strip_rebuild", False)
            ),
            "strip_rebuild_min_edges": int(
                config.get("udf_cut_boundary_strip_min_edges", 100)
            ),
            "strip_rebuild_min_perimeter_world": float(
                config.get("udf_cut_boundary_strip_min_perimeter_world", 0.15)
            ),
            "strip_rebuild_max_loops": int(
                config.get("udf_cut_boundary_strip_max_loops", 8)
            ),
            "strip_rebuild_min_rings": int(
                config.get("udf_cut_boundary_strip_min_rings", 2)
            ),
            "strip_rebuild_max_rings": int(
                config.get("udf_cut_boundary_strip_max_rings", 6)
            ),
            "strip_rebuild_spline_smoothing_fraction": float(
                config.get(
                    "udf_cut_boundary_strip_spline_smoothing_fraction", 0.35
                )
            ),
            "strip_rebuild_target_edge_scale": float(
                config.get("udf_cut_boundary_strip_target_edge_scale", 1.0)
            ),
            "strip_rebuild_max_spline_displacement_fraction": float(
                config.get(
                    "udf_cut_boundary_strip_max_spline_displacement_fraction",
                    1.5,
                )
            ),
            "strip_rebuild_min_area_world2": float(
                config.get("udf_cut_boundary_strip_min_area_world2", 1e-14)
            ),
            "spline_enabled": bool(config.get("udf_cut_boundary_spline", False)),
            "spline_smoothing_fraction": float(
                config.get("udf_cut_boundary_spline_smoothing_fraction", 0.35)
            ),
            "spline_blend": float(
                config.get("udf_cut_boundary_spline_blend", 1.0)
            ),
            "spline_min_edges": int(
                config.get("udf_cut_boundary_spline_min_edges", 20)
            ),
            "spline_max_total_fraction": float(
                config.get("udf_cut_boundary_spline_max_total_fraction", 1.0)
            ),
            "spline_min_area_ratio": float(
                config.get("udf_cut_boundary_spline_min_area_ratio", 0.10)
            ),
            "spline_min_normal_dot": float(
                config.get("udf_cut_boundary_spline_min_normal_dot", 0.0)
            ),
            "spline_min_backtrack_scale": float(
                config.get("udf_cut_boundary_spline_min_backtrack_scale", 0.0625)
            ),
            "smooth_iterations": int(
                config.get("udf_cut_boundary_smooth_iterations", 8)
            ),
            "smooth_lambda": float(
                config.get("udf_cut_boundary_smooth_lambda", 0.45)
            ),
            "smooth_mu": float(config.get("udf_cut_boundary_smooth_mu", -0.47)),
            "smooth_min_edges": int(
                config.get("udf_cut_boundary_smooth_min_edges", 12)
            ),
            "smooth_max_step_fraction": float(
                config.get("udf_cut_boundary_max_step_fraction", 0.25)
            ),
            "smooth_max_total_fraction": float(
                config.get("udf_cut_boundary_max_total_fraction", 0.75)
            ),
        }

    @torch.no_grad()
    def action_part_adapt_udf_cut(self, arg):
        output_folder = arg["output_folder"]
        shape_name = arg["shape"]
        data_root = arg["data_root"]
        root_path = arg.get("root_path", os.getcwd())

        adaptation_items, extraction_config = self._load_adaptations(arg)
        raw_config = {}
        try:
            import app_utils_3dvec as _utils
            loaded = _utils.load_yaml_file(arg["adapt_file"])
            if isinstance(loaded, dict):
                raw_config = loaded
        except Exception as exc:
            print("[udf cut] warning: could not reload seam config:", exc)

        # Build the same per-blend-group seam corrections used by AgentSDF.
        # The UDF is a classifier/filter for the already adapted SDF surface,
        # so its query coordinates must receive the same radius/position warp.
        seam_plan = build_seam_plan(
            adaptation_items,
            raw_config.get("blend_groups", {}) if isinstance(raw_config, dict) else {},
            self.curve_from_key,
        )
        apply_seam_warp = bool(
            extraction_config.get("udf_cut_apply_seam_warp", True)
        )
        if not apply_seam_warp:
            print(
                "[udf cut seam] query seam warp DISABLED; "
                "using pre-0011 UDF query coordinates"
            )

        handle = self.load_shape_handle(data_root, shape_name, "avatar")
        target_curves = {
            self.encode_key(shape_name, curve.name): curve for curve in handle.curves
        }

        missing = sorted(
            {
                item["target_key"]
                for item in adaptation_items
                if item["target_key"] not in target_curves
            }
        )
        if missing:
            raise KeyError(
                f"Missing target curves: {missing}. Available: {sorted(target_curves)}"
            )

        os.makedirs(output_folder, exist_ok=True)
        model_batch_size = int(
            extraction_config.get(
                "udf_model_batch_size",
                extraction_config.get("udf_batch_size", 32768),
            )
        )
        query_batch_size = int(
            extraction_config.get("udf_cut_query_batch_size", 32768)
        )
        far_world = float(extraction_config.get("udf_far_value", 0.1))

        mc_mesh_spec = extraction_config.get("mc_mesh")
        if not mc_mesh_spec:
            raise ValueError("part_adapt_udf_cut requires --mc-mesh / mc_mesh.")
        reference_spec = extraction_config.get("nsdudf_reference_mesh")

        combined_transfer_enabled = bool(
            extraction_config.get("udf_cut_combined_transfer", False)
        )
        combined_mc_spec = extraction_config.get("udf_cut_combined_mc_mesh")
        if combined_transfer_enabled and not combined_mc_spec:
            raise ValueError(
                "udf_cut_combined_transfer=true requires "
                "udf_cut_combined_mc_mesh at YAML top level."
            )
        combined_removed_references: List[trimesh.Trimesh] = []
        combined_kept_references: List[trimesh.Trimesh] = []
        write_debug_outputs = bool(
            extraction_config.get("udf_cut_write_debug_outputs", True)
        )
        if not write_debug_outputs:
            print(
                "[udf cut] debug PLY/NPZ outputs disabled; "
                "only the final cut mesh (and JSON report) will be written"
            )

        # Optional sweep summaries are collected across all adaptations so one
        # run can tune every piece with a single UDF query per piece.
        all_sweep_summaries = []

        for item_index, item in enumerate(adaptation_items):
            target_key = item["target_key"]
            accessory_key = item["accessory_key"]
            mode = str(item.get("mode", "direct"))
            if mode != "direct":
                raise ValueError(
                    f"AgentUDFCut supports mode='direct' only, got {mode!r}."
                )

            avatar_curve = target_curves[target_key]
            accessory_curve = self.curve_from_key(accessory_key)
            config = self._extraction_config_for_item(extraction_config, item)

            adapt_arg = {
                "mode": "direct",
                "avatar_curve_handle": avatar_curve,
                "accessory_curve_handle": accessory_curve,
                "device": self.device,
                "infer_scale": 2.0,
                "avatar_curve_idx": self.feat_dict[target_key],
                "accessory_curve_idx": self.feat_dict[accessory_key],
            }
            adapt_arg.update(item)
            if apply_seam_warp and item_index in seam_plan:
                adapt_arg["_seam_corrections"] = seam_plan[item_index]
                print(
                    f"[udf cut seam] item={item_index} "
                    f"corrections={len(seam_plan[item_index])}"
                )

            shared_snug_path = None
            shared_snug_spec = adapt_arg.get("shared_snug_field", None)
            if shared_snug_spec:
                shared_snug_path = resolve_shared_snug_field_path(
                    str(shared_snug_spec),
                    root_path=root_path,
                    output_folder=output_folder,
                    item_index=item_index,
                    mode=mode,
                    target_key=target_key,
                    accessory_key=accessory_key,
                )
                shared_snug_field = load_shared_snug_field(shared_snug_path)
                adapt_arg["avatar_snug_scale_field"] = shared_snug_field
                print(
                    "[udf cut shared snug load]",
                    shared_snug_path,
                    snug_field_stats(shared_snug_field),
                )

            adapt_arg["adapt_debug_counts"] = False
            adapt_arg["debug_interval_projection"] = False

            if (
                bool(adapt_arg.get("auto_avatar_snug_field", False))
                and shared_snug_path is None
            ):
                print(
                    "[udf cut warning] auto_avatar_snug_field is enabled in the "
                    "YAML, but this direct world-point filter does not rebuild "
                    "the signed-SDF snug field. Add shared_snug_field to the "
                    "adaptation item, or disable auto_avatar_snug_field."
                )

            # Do not derive a separate field from unsigned distances. The
            # injected avatar_snug_scale_field remains independently active.
            adapt_arg["auto_avatar_snug_field"] = False

            mc_path = self._resolve_path_spec(
                str(mc_mesh_spec),
                root_path=root_path,
                item_index=item_index,
                mode=mode,
                accessory_key=accessory_key,
                target_key=target_key,
            )
            mesh = trimesh.load(mc_path, process=False, force="mesh")
            if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
                raise ValueError(f"MC input is empty or not a triangle mesh: {mc_path}")

            reference_points = None
            reference_path = None
            if reference_spec:
                reference_path = self._resolve_path_spec(
                    str(reference_spec),
                    root_path=root_path,
                    item_index=item_index,
                    mode=mode,
                    accessory_key=accessory_key,
                    target_key=target_key,
                )
                reference_mesh = trimesh.load(
                    reference_path, process=False, force="mesh"
                )
                reference_points = self._large_reference_boundary_points(
                    reference_mesh,
                    min_edges=int(config.get("udf_cut_reference_min_edges", 20)),
                    min_span_world=float(
                        config.get("udf_cut_reference_min_span_world", 0.05)
                    ),
                )
                print(
                    "[udf cut reference]",
                    f"path={reference_path}",
                    f"kept_boundary_points={len(reference_points)}",
                )

            print(
                f"[udf cut {item_index + 1}/{len(adaptation_items)}]",
                f"mc={mc_path}",
                f"target={target_key}",
                f"accessory={accessory_key}",
                "world_space_direct_query=True",
            )

            scores, usable, sample_values, sample_valid = self._query_face_scores(
                mesh,
                avatar_curve=avatar_curve,
                adapt_arg=adapt_arg,
                accessory_key=accessory_key,
                model_batch_size=model_batch_size,
                query_batch_size=query_batch_size,
                far_world=far_world,
            )

            # Diagnose UDF support before threshold selection.  `sample_valid`
            # comes directly from _query_adapted_raw_udf(), where a point is
            # marked valid only when localize_samples_adapt() returns it as an
            # inside point.  This lets us distinguish a localization/support
            # failure from a threshold-search failure.
            valid_counts = np.asarray(sample_valid, dtype=bool).sum(axis=1)
            finite_valid = np.asarray(sample_valid, dtype=bool) & np.isfinite(
                sample_values
            )
            finite_valid_counts = finite_valid.sum(axis=1)
            valid_total = int(np.sum(sample_valid))
            finite_valid_total = int(np.sum(finite_valid))
            centroid_valid = int(np.sum(sample_valid[:, 0]))
            centroid_finite = int(np.sum(finite_valid[:, 0]))
            faces_any_valid = int(np.sum(valid_counts >= 1))
            faces_two_valid = int(np.sum(valid_counts >= 2))
            faces_centroid_plus_one = int(
                np.sum(sample_valid[:, 0] & (valid_counts >= 2))
            )
            usable_count = int(np.sum(usable))

            print(
                "[udf cut validity]",
                f"item={item_index}",
                f"faces={len(mesh.faces)}",
                f"query_points={sample_valid.size}",
                f"valid_total={valid_total}",
                f"finite_valid_total={finite_valid_total}",
                f"centroid_valid={centroid_valid}",
                f"centroid_finite={centroid_finite}",
                f"edge_valid={[int(np.sum(sample_valid[:, i])) for i in (1, 2, 3)]}",
                f"faces_any_valid={faces_any_valid}",
                f"faces_2plus_valid={faces_two_valid}",
                f"faces_centroid_plus_one={faces_centroid_plus_one}",
                f"usable_faces={usable_count}",
                f"src={adapt_arg.get('src_0', None)}->{adapt_arg.get('src_1', None)}",
                f"cylinder_scale={adapt_arg.get('avatar_cylinder_radius_scale', 1.0)}",
                f"localize_norm={adapt_arg.get('adapt_localize_norm', 1.0)}",
            )

            if usable_count == 0:
                if valid_total == 0:
                    no_support_reason = "query_validity_rejected_all_samples"
                elif finite_valid_total == 0:
                    no_support_reason = "all_valid_udf_values_nonfinite"
                elif centroid_valid == 0:
                    no_support_reason = "no_valid_face_centroids"
                elif faces_centroid_plus_one == 0:
                    no_support_reason = "insufficient_valid_samples_per_face"
                else:
                    no_support_reason = "no_usable_finite_face_scores"

                safe_key_diag = self._safe_label(accessory_key)
                diag_prefix = f"{item_index}_{mode}_{safe_key_diag}"
                diag_npz_path = op.join(
                    output_folder,
                    f"{diag_prefix}_udf_cut_no_valid_support.npz",
                )
                diag_report_path = op.join(
                    output_folder,
                    f"{diag_prefix}_udf_cut_no_valid_support.json",
                )
                diag_before_path = op.join(
                    output_folder,
                    f"{diag_prefix}_mc_before_udf_cut.ply",
                )
                diag_after_path = op.join(
                    output_folder,
                    f"{diag_prefix}_mc_after_udf_cut.ply",
                )
                valid_points_path = op.join(
                    output_folder,
                    f"{diag_prefix}_udf_cut_valid_query_points.ply",
                )

                # Reconstruct the same four query points per face so that any
                # valid subset can be inspected directly in a viewer.
                diag_vertices = np.asarray(mesh.vertices, dtype=np.float64)
                diag_faces = np.asarray(mesh.faces, dtype=np.int64)
                diag_tri = diag_vertices[diag_faces]
                diag_samples = np.stack(
                    [
                        diag_tri.mean(axis=1),
                        0.5 * (diag_tri[:, 0] + diag_tri[:, 1]),
                        0.5 * (diag_tri[:, 1] + diag_tri[:, 2]),
                        0.5 * (diag_tri[:, 2] + diag_tri[:, 0]),
                    ],
                    axis=1,
                )
                valid_query_points = diag_samples[np.asarray(sample_valid, dtype=bool)]

                mesh.export(diag_before_path)
                mesh.export(diag_after_path)  # explicit no-cut output
                if len(valid_query_points):
                    trimesh.points.PointCloud(valid_query_points).export(
                        valid_points_path
                    )
                else:
                    valid_points_path = None

                np.savez_compressed(
                    diag_npz_path,
                    face_scores=scores,
                    usable_faces=usable,
                    sample_values=sample_values,
                    sample_valid=sample_valid,
                    valid_counts=valid_counts,
                    finite_valid=finite_valid,
                    finite_valid_counts=finite_valid_counts,
                )

                no_support_report = {
                    "item_index": int(item_index),
                    "target_key": target_key,
                    "accessory_key": accessory_key,
                    "mc_input": mc_path,
                    "reason": no_support_reason,
                    "action": "no_cut_continue",
                    "adaptation": {
                        "src_0": adapt_arg.get("src_0", None),
                        "src_1": adapt_arg.get("src_1", None),
                        "avatar_cylinder_radius_scale": adapt_arg.get(
                            "avatar_cylinder_radius_scale", 1.0
                        ),
                        "avatar_cylinder_radius_add": adapt_arg.get(
                            "avatar_cylinder_radius_add", 0.0
                        ),
                        "adapt_localize_norm": adapt_arg.get(
                            "adapt_localize_norm", 1.0
                        ),
                        "loop_projection_candidates": adapt_arg.get(
                            "loop_projection_candidates", 1
                        ),
                    },
                    "counts": {
                        "faces": int(len(mesh.faces)),
                        "query_points": int(sample_valid.size),
                        "valid_total": valid_total,
                        "finite_valid_total": finite_valid_total,
                        "centroid_valid": centroid_valid,
                        "centroid_finite": centroid_finite,
                        "edge_valid": [
                            int(np.sum(sample_valid[:, i])) for i in (1, 2, 3)
                        ],
                        "faces_any_valid": faces_any_valid,
                        "faces_2plus_valid": faces_two_valid,
                        "faces_centroid_plus_one": faces_centroid_plus_one,
                        "usable_faces": usable_count,
                    },
                    "outputs": {
                        "diagnostic_npz": diag_npz_path,
                        "report": diag_report_path,
                        "before": diag_before_path,
                        "after_no_cut": diag_after_path,
                        "valid_query_points": valid_points_path,
                    },
                }
                with open(diag_report_path, "w", encoding="utf-8") as handle_out:
                    json.dump(no_support_report, handle_out, indent=2)

                # Keep combined-transfer ownership aligned with the adaptation
                # list, but contribute no removed surface for this item.
                if combined_transfer_enabled:
                    empty_removed = self._mesh_from_face_mask(
                        mesh, np.zeros(len(mesh.faces), dtype=bool)
                    )
                    combined_removed_references.append(empty_removed)
                    combined_kept_references.append(mesh.copy())
                    blend_group = str(item.get("blend_group", "")).strip()
                    combined_reference_groups.append(
                        blend_group if blend_group else str(target_key)
                    )

                print(
                    "[udf cut skip]",
                    f"item={item_index}",
                    f"reason={no_support_reason}",
                    "action=no_cut_continue",
                    f"diagnostic={diag_report_path}",
                )
                continue

            finite_scores = scores[np.isfinite(scores)]
            score_percentiles = {}
            if len(finite_scores):
                for q in (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100):
                    score_percentiles[str(q)] = float(np.percentile(finite_scores, q))

            seed_world = float(config.get("udf_cut_seed_world", 0.02))
            grow_world = float(config.get("udf_cut_grow_world", 0.01))
            auto_threshold_report = None

            # Optional per-piece threshold search from this piece's own UDF
            # score distribution.  Seed and grow are calibrated independently:
            # no global multiplier is shared across adaptations.  Candidate
            # cuts reuse the already-computed face scores, so the search is
            # graph/topology work only (no extra neural UDF queries).
            auto_thresholds = config.get("udf_cut_auto_thresholds", False)
            if isinstance(auto_thresholds, dict):
                auto_enabled = bool(auto_thresholds.get("enabled", True))
                auto_spec = auto_thresholds
            else:
                auto_enabled = bool(auto_thresholds)
                auto_spec = {}

            usable_scores = scores[usable & np.isfinite(scores)]
            if auto_enabled:
                if len(usable_scores) == 0:
                    raise RuntimeError(
                        "udf_cut_auto_thresholds needs at least one usable "
                        f"finite face score for item {item_index}"
                    )

                seed_percentiles = [
                    float(v) for v in auto_spec.get(
                        "seed_percentiles", [95, 96, 97, 98, 99]
                    )
                ]
                grow_percentiles = [
                    float(v) for v in auto_spec.get(
                        "grow_percentiles", [80, 85, 90, 92, 94]
                    )
                ]
                for q in seed_percentiles + grow_percentiles:
                    if not (0.0 < q < 100.0):
                        raise ValueError(
                            "udf_cut_auto_thresholds percentiles must be in (0, 100)"
                        )

                min_removed_fraction = float(
                    auto_spec.get("min_removed_fraction", 0.002)
                )
                max_removed_fraction = float(
                    auto_spec.get("max_removed_fraction", 0.15)
                )
                require_topology_preserved = bool(
                    auto_spec.get("require_topology_preserved", True)
                )
                min_positive_ring_fraction = float(
                    auto_spec.get("min_positive_ring_fraction", 0.80)
                )
                min_ring_contrast_world = float(
                    auto_spec.get("min_ring_contrast_world", 0.0)
                )
                boundary_penalty = float(
                    auto_spec.get("boundary_penalty", 2.0)
                )
                save_best_n = max(0, int(auto_spec.get("save_best_n", 3)))

                input_stats = self._topology_stats(mesh)
                input_components = int(input_stats["components"])
                input_boundary_edges = int(input_stats["boundary_edges"])
                candidate_rows = []

                # Everything below is invariant across percentile pairs.  The
                # old implementation rebuilt these arrays/KD-tree, constructed
                # kept+removed Trimesh objects, and reran full topology scans
                # for every candidate.  On a ~380k-face 512 mesh that turns a
                # 5x5 threshold search into millions of Python graph operations.
                # Build the immutable search context once and keep candidates as
                # face masks until a winner needs to be exported.
                search_adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
                search_face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
                search_boundary_faces = self._existing_boundary_faces(mesh)
                search_centroids = np.asarray(mesh.triangles_center, dtype=np.float64)
                search_topology = self._prepare_search_topology(mesh)
                search_reference_tree = None
                if reference_points is not None and len(reference_points):
                    try:
                        from scipy.spatial import cKDTree

                        search_reference_tree = cKDTree(
                            np.asarray(reference_points, dtype=np.float64)
                        )
                    except Exception as exc:
                        print("[udf cut] reference KD-tree unavailable:", exc)

                search_t0 = perf_counter()
                search_evaluated = 0
                print(
                    "[udf cut auto search]",
                    f"item={item_index}",
                    f"grid={len(seed_percentiles)}x{len(grow_percentiles)}",
                    "mode=fast_mask_only",
                )

                seed_values = {
                    q: float(np.percentile(usable_scores, q))
                    for q in seed_percentiles
                }
                grow_values = {
                    q: float(np.percentile(usable_scores, q))
                    for q in grow_percentiles
                }

                for seed_q in seed_percentiles:
                    candidate_seed = seed_values[seed_q]
                    for grow_q in grow_percentiles:
                        candidate_grow = grow_values[grow_q]
                        if candidate_grow > candidate_seed:
                            continue

                        (
                            candidate_seed_mask,
                            candidate_grow_mask,
                            candidate_decisions,
                            candidate_remove_mask,
                            kept_stats,
                            removed_stats,
                        ) = self._evaluate_cut_candidate_fast(
                            mesh,
                            face_scores=scores,
                            usable_faces=usable,
                            seed_world=candidate_seed,
                            grow_world=candidate_grow,
                            min_faces=int(config.get("udf_cut_min_faces", 8)),
                            min_seed_faces=int(
                                config.get("udf_cut_min_seed_faces", 2)
                            ),
                            min_seed_fraction=float(
                                config.get("udf_cut_min_seed_fraction", 0.05)
                            ),
                            min_area_world2=float(
                                config.get("udf_cut_min_area_world2", 0.0)
                            ),
                            preserve_existing_boundaries=bool(
                                config.get(
                                    "udf_cut_preserve_existing_boundaries", True
                                )
                            ),
                            reference_tree=search_reference_tree,
                            reference_max_distance_world=(
                                float(config["udf_cut_reference_max_distance_world"])
                                if config.get(
                                    "udf_cut_reference_max_distance_world", None
                                ) is not None
                                else None
                            ),
                            adjacency=search_adjacency,
                            face_areas=search_face_areas,
                            existing_boundary_faces=search_boundary_faces,
                            centroids=search_centroids,
                            topology=search_topology,
                        )
                        search_evaluated += 1
                        accepted = [
                            decision
                            for decision in candidate_decisions
                            if decision.accepted
                        ]
                        ring_evidence = [
                            decision
                            for decision in accepted
                            if decision.ring_score_median is not None
                        ]
                        contrasts = [
                            float(
                                decision.score_median
                                - float(decision.ring_score_median)
                            )
                            for decision in ring_evidence
                        ]
                        ring_weight = float(
                            sum(decision.face_count for decision in ring_evidence)
                        )
                        positive_ring_weight = float(
                            sum(
                                decision.face_count
                                for decision, contrast in zip(
                                    ring_evidence, contrasts
                                )
                                if contrast > 0.0
                            )
                        )
                        positive_ring_fraction = (
                            positive_ring_weight / ring_weight
                            if ring_weight > 0.0
                            else 0.0
                        )
                        median_ring_contrast = (
                            float(np.median(contrasts)) if contrasts else None
                        )
                        removed_fraction = float(
                            int(removed_stats["faces"]) / max(len(mesh.faces), 1)
                        )
                        topology_preserved = (
                            int(kept_stats["components"]) == input_components
                        )
                        boundary_delta = max(
                            0,
                            int(kept_stats["boundary_edges"])
                            - input_boundary_edges,
                        )
                        boundary_per_removed = float(
                            boundary_delta / max(int(removed_stats["faces"]), 1)
                        )

                        eligible = (
                            len(accepted) > 0
                            and int(removed_stats["faces"]) > 0
                            and removed_fraction >= min_removed_fraction
                            and removed_fraction <= max_removed_fraction
                            and (
                                topology_preserved
                                or not require_topology_preserved
                            )
                            and median_ring_contrast is not None
                            and median_ring_contrast >= min_ring_contrast_world
                            and positive_ring_fraction
                            >= min_positive_ring_fraction
                        )

                        # Prefer strong local UDF evidence first.  Once a cut
                        # has good contrast against its surrounding ring, reward
                        # useful cleanup rather than always choosing the tiniest
                        # possible patch.  Penalize ragged/fragmented boundaries.
                        contrast_scale = max(abs(candidate_seed), 1.0e-8)
                        contrast_score = (
                            float(median_ring_contrast) / contrast_scale
                            if median_ring_contrast is not None
                            else -1.0
                        )
                        removal_reward = float(
                            removed_fraction / max(max_removed_fraction, 1.0e-8)
                        )
                        fragmentation_penalty = float(
                            max(int(removed_stats["components"]) - 1, 0)
                        )
                        objective = (
                            contrast_score
                            + 0.35 * removal_reward
                            - boundary_penalty * boundary_per_removed
                            - 0.05 * fragmentation_penalty
                        )

                        candidate_rows.append(
                            {
                                "seed_percentile": seed_q,
                                "grow_percentile": grow_q,
                                "seed_world": candidate_seed,
                                "grow_world": candidate_grow,
                                "seed_faces": int(np.sum(candidate_seed_mask)),
                                "grow_faces": int(np.sum(candidate_grow_mask)),
                                "removed_faces": int(removed_stats["faces"]),
                                "removed_fraction": removed_fraction,
                                "accepted_components": int(len(accepted)),
                                "after_components": int(kept_stats["components"]),
                                "after_boundary_edges": int(
                                    kept_stats["boundary_edges"]
                                ),
                                "removed_components": int(
                                    removed_stats["components"]
                                ),
                                "topology_preserved": bool(
                                    topology_preserved
                                ),
                                "positive_ring_fraction": float(
                                    positive_ring_fraction
                                ),
                                "median_ring_contrast": median_ring_contrast,
                                "boundary_delta": int(boundary_delta),
                                "objective": float(objective),
                                "eligible": bool(eligible),
                            }
                        )

                print(
                    "[udf cut auto search]",
                    f"item={item_index}",
                    f"evaluated={search_evaluated}",
                    f"elapsed_s={perf_counter() - search_t0:.3f}",
                )

                eligible_rows = [
                    row for row in candidate_rows if row["eligible"]
                ]
                eligible_rows.sort(
                    key=lambda row: (
                        -row["objective"],
                        -row["median_ring_contrast"],
                        -row["removed_fraction"],
                        row["after_boundary_edges"],
                    )
                )
                selected = eligible_rows[0] if eligible_rows else None

                if selected is None:
                    # No candidate has enough local/topological evidence.  Make
                    # the normal cut below an explicit no-op instead of falling
                    # back to one global absolute threshold.
                    score_max = float(np.max(usable_scores))
                    margin = max(1.0e-6, abs(score_max) * 1.0e-6)
                    seed_world = score_max + margin
                    grow_world = seed_world
                    print(
                        "[udf cut auto search]",
                        f"item={item_index}",
                        "selected=none",
                        "action=no_cut",
                    )
                else:
                    seed_world = float(selected["seed_world"])
                    grow_world = float(selected["grow_world"])
                    print(
                        "[udf cut auto search]",
                        f"item={item_index}",
                        f"seed=P{selected['seed_percentile']:g}={seed_world:.6g}",
                        f"grow=P{selected['grow_percentile']:g}={grow_world:.6g}",
                        f"removed={selected['removed_faces']}",
                        f"removed_pct={100.0 * selected['removed_fraction']:.2f}",
                        f"contrast={selected['median_ring_contrast']:.6g}",
                        f"ring_positive={selected['positive_ring_fraction']:.3f}",
                        f"objective={selected['objective']:.6g}",
                    )

                    if save_best_n > 0:
                        safe_key_auto = self._safe_label(accessory_key)
                        auto_prefix = f"{item_index}_{mode}_{safe_key_auto}"
                        for rank, row in enumerate(
                            eligible_rows[:save_best_n], start=1
                        ):
                            (
                                _best_seed_mask,
                                _best_grow_mask,
                                _best_decisions,
                                best_remove_mask,
                                _best_kept_stats,
                                _best_removed_stats,
                            ) = self._evaluate_cut_candidate_fast(
                                mesh,
                                face_scores=scores,
                                usable_faces=usable,
                                seed_world=float(row["seed_world"]),
                                grow_world=float(row["grow_world"]),
                                min_faces=int(
                                    config.get("udf_cut_min_faces", 8)
                                ),
                                min_seed_faces=int(
                                    config.get("udf_cut_min_seed_faces", 2)
                                ),
                                min_seed_fraction=float(
                                    config.get(
                                        "udf_cut_min_seed_fraction", 0.05
                                    )
                                ),
                                min_area_world2=float(
                                    config.get("udf_cut_min_area_world2", 0.0)
                                ),
                                preserve_existing_boundaries=bool(
                                    config.get(
                                        "udf_cut_preserve_existing_boundaries",
                                        True,
                                    )
                                ),
                                reference_tree=search_reference_tree,
                                reference_max_distance_world=(
                                    float(
                                        config[
                                            "udf_cut_reference_max_distance_world"
                                        ]
                                    )
                                    if config.get(
                                        "udf_cut_reference_max_distance_world",
                                        None,
                                    )
                                    is not None
                                    else None
                                ),
                                adjacency=search_adjacency,
                                face_areas=search_face_areas,
                                existing_boundary_faces=search_boundary_faces,
                                centroids=search_centroids,
                                topology=search_topology,
                            )
                            best_kept = self._mesh_from_face_mask(
                                mesh, ~best_remove_mask
                            )
                            best_path = op.join(
                                output_folder,
                                f"{auto_prefix}_auto_best{rank}_"
                                f"seedP{row['seed_percentile']:g}_"
                                f"growP{row['grow_percentile']:g}.ply",
                            )
                            best_kept.export(best_path)
                            row["mesh"] = best_path

                auto_threshold_report = {
                    "enabled": True,
                    "mode": "independent_percentile_search",
                    "seed_percentiles": seed_percentiles,
                    "grow_percentiles": grow_percentiles,
                    "min_removed_fraction": min_removed_fraction,
                    "max_removed_fraction": max_removed_fraction,
                    "require_topology_preserved": require_topology_preserved,
                    "min_positive_ring_fraction": min_positive_ring_fraction,
                    "min_ring_contrast_world": min_ring_contrast_world,
                    "boundary_penalty": boundary_penalty,
                    "selected": dict(selected) if selected is not None else None,
                    "candidates": candidate_rows,
                    "seed_world": seed_world,
                    "grow_world": grow_world,
                }

            # Optional threshold sweep for diagnostics/tuning.  The expensive
            # adapted-UDF query above is performed only once; every candidate
            # reuses the same per-face scores and only reruns the cheap
            # connectivity/component decision.  When configured at YAML top
            # level this applies to every adaptation item.
            sweep_config = config.get("udf_cut_sweep", None)
            if isinstance(sweep_config, dict) and bool(
                sweep_config.get("enabled", True)
            ):
                seed_values = [
                    float(v) for v in sweep_config.get(
                        "seed_values", [seed_world]
                    )
                ]
                grow_values = [
                    float(v) for v in sweep_config.get(
                        "grow_values", [grow_world]
                    )
                ]
                save_all_meshes = bool(sweep_config.get("save_meshes", False))
                save_best_n = max(
                    0, int(sweep_config.get("save_best_n", 3))
                )
                max_removed_fraction = float(
                    sweep_config.get("max_removed_fraction", 0.20)
                )
                require_topology_preserved = bool(
                    sweep_config.get("require_topology_preserved", True)
                )
                input_stats = self._topology_stats(mesh)
                input_components = int(input_stats["components"])
                sweep_rows = []
                safe_key_sweep = self._safe_label(accessory_key)
                sweep_prefix = f"{item_index}_{mode}_{safe_key_sweep}"

                for sweep_seed in seed_values:
                    for sweep_grow in grow_values:
                        if sweep_grow > sweep_seed:
                            continue
                        (
                            sweep_kept,
                            sweep_removed,
                            sweep_seed_mask,
                            sweep_grow_mask,
                            sweep_decisions,
                            _sweep_remove_mask,
                        ) = self._cut_mesh(
                            mesh,
                            face_scores=scores,
                            usable_faces=usable,
                            seed_world=sweep_seed,
                            grow_world=sweep_grow,
                            min_faces=int(config.get("udf_cut_min_faces", 8)),
                            min_seed_faces=int(
                                config.get("udf_cut_min_seed_faces", 2)
                            ),
                            min_seed_fraction=float(
                                config.get("udf_cut_min_seed_fraction", 0.05)
                            ),
                            min_area_world2=float(
                                config.get("udf_cut_min_area_world2", 0.0)
                            ),
                            preserve_existing_boundaries=bool(
                                config.get(
                                    "udf_cut_preserve_existing_boundaries", True
                                )
                            ),
                            reference_points=reference_points,
                            reference_max_distance_world=(
                                float(config["udf_cut_reference_max_distance_world"])
                                if config.get(
                                    "udf_cut_reference_max_distance_world", None
                                ) is not None
                                else None
                            ),
                        )

                        kept_stats = self._topology_stats(sweep_kept)
                        removed_stats = self._topology_stats(sweep_removed)
                        accepted_components = sum(
                            1 for decision in sweep_decisions if decision.accepted
                        )
                        removed_fraction = float(
                            len(sweep_removed.faces) / max(len(mesh.faces), 1)
                        )
                        topology_preserved = (
                            int(kept_stats["components"]) == input_components
                        )
                        eligible = (
                            accepted_components > 0
                            and len(sweep_removed.faces) > 0
                            and removed_fraction <= max_removed_fraction
                            and (
                                topology_preserved
                                or not require_topology_preserved
                            )
                        )
                        row = {
                            "seed_world": sweep_seed,
                            "grow_world": sweep_grow,
                            "seed_faces": int(np.sum(sweep_seed_mask)),
                            "grow_faces": int(np.sum(sweep_grow_mask)),
                            "removed_faces": int(len(sweep_removed.faces)),
                            "removed_fraction": removed_fraction,
                            "accepted_components": int(accepted_components),
                            "after_faces": int(len(sweep_kept.faces)),
                            "after_components": int(kept_stats["components"]),
                            "after_boundary_edges": int(
                                kept_stats["boundary_edges"]
                            ),
                            "removed_components": int(
                                removed_stats["components"]
                            ),
                            "topology_preserved": bool(topology_preserved),
                            "eligible": bool(eligible),
                        }

                        seed_tag = (
                            f"{sweep_seed:.4f}".rstrip("0").rstrip(".")
                            .replace("-", "m")
                            .replace(".", "p")
                        )
                        grow_tag = (
                            f"{sweep_grow:.4f}".rstrip("0").rstrip(".")
                            .replace("-", "m")
                            .replace(".", "p")
                        )
                        if save_all_meshes:
                            candidate_path = op.join(
                                output_folder,
                                f"{sweep_prefix}_sweep_seed{seed_tag}_grow{grow_tag}.ply",
                            )
                            sweep_kept.export(candidate_path)
                            row["mesh"] = candidate_path
                        sweep_rows.append(row)

                # Prefer a topology-preserving, non-empty cut and among those
                # remove as little as possible.  Boundary complexity is the
                # tie-breaker.  This is only a diagnostic recommendation, not
                # a production guarantee that the removed patch is semantic.
                eligible_rows = [row for row in sweep_rows if row["eligible"]]
                eligible_rows.sort(
                    key=lambda row: (
                        row["removed_fraction"],
                        row["after_boundary_edges"],
                        -row["grow_world"],
                        row["seed_world"],
                    )
                )
                recommended = eligible_rows[0] if eligible_rows else None

                # Save only the best few candidates by default.  This avoids
                # hundreds of 512-resolution PLYs when sweeping all items.
                if save_best_n > 0:
                    for rank, row in enumerate(eligible_rows[:save_best_n], 1):
                        (
                            best_kept,
                            _best_removed,
                            _best_seed_mask,
                            _best_grow_mask,
                            _best_decisions,
                            _best_remove_mask,
                        ) = self._cut_mesh(
                            mesh,
                            face_scores=scores,
                            usable_faces=usable,
                            seed_world=float(row["seed_world"]),
                            grow_world=float(row["grow_world"]),
                            min_faces=int(config.get("udf_cut_min_faces", 8)),
                            min_seed_faces=int(
                                config.get("udf_cut_min_seed_faces", 2)
                            ),
                            min_seed_fraction=float(
                                config.get("udf_cut_min_seed_fraction", 0.05)
                            ),
                            min_area_world2=float(
                                config.get("udf_cut_min_area_world2", 0.0)
                            ),
                            preserve_existing_boundaries=bool(
                                config.get(
                                    "udf_cut_preserve_existing_boundaries", True
                                )
                            ),
                            reference_points=reference_points,
                            reference_max_distance_world=(
                                float(config["udf_cut_reference_max_distance_world"])
                                if config.get(
                                    "udf_cut_reference_max_distance_world", None
                                ) is not None
                                else None
                            ),
                        )
                        best_path = op.join(
                            output_folder,
                            f"{sweep_prefix}_sweep_best{rank}_"
                            f"seed{str(row['seed_world']).replace('.', 'p')}_"
                            f"grow{str(row['grow_world']).replace('.', 'p')}.ply",
                        )
                        best_kept.export(best_path)
                        row["best_rank"] = rank
                        row["best_mesh"] = best_path

                # Keep the complete candidate table deterministic and easy to
                # inspect after the ranked candidates have been selected.
                sweep_rows.sort(
                    key=lambda row: (
                        not row["eligible"],
                        row["removed_fraction"],
                        row["after_boundary_edges"],
                        row["seed_world"],
                        row["grow_world"],
                    )
                )
                if recommended is not None:
                    recommended = dict(recommended)

                sweep_report_path = op.join(
                    output_folder, f"{sweep_prefix}_udf_cut_sweep.json"
                )
                sweep_report = {
                    "item_index": item_index,
                    "name": item.get("name"),
                    "target_key": target_key,
                    "accessory_key": accessory_key,
                    "input_faces": int(len(mesh.faces)),
                    "input_topology": input_stats,
                    "score_percentiles": score_percentiles,
                    "selection": {
                        "require_topology_preserved": require_topology_preserved,
                        "max_removed_fraction": max_removed_fraction,
                        "recommended": recommended,
                    },
                    "candidates": sweep_rows,
                }
                with open(sweep_report_path, "w", encoding="utf-8") as handle:
                    json.dump(sweep_report, handle, indent=2)

                summary_row = {
                    "item_index": item_index,
                    "name": item.get("name"),
                    "target_key": target_key,
                    "accessory_key": accessory_key,
                    "input_faces": int(len(mesh.faces)),
                    "input_components": input_components,
                    "score_percentiles": score_percentiles,
                    "recommended": recommended,
                    "report": sweep_report_path,
                }
                all_sweep_summaries.append(summary_row)

                print(
                    "[udf cut sweep]",
                    f"item={item_index}",
                    f"candidates={len(sweep_rows)}",
                    f"eligible={len(eligible_rows)}",
                    f"report={sweep_report_path}",
                )
                if recommended is None:
                    print(
                        "[udf cut sweep recommended]",
                        f"item={item_index}",
                        "NONE",
                    )
                else:
                    print(
                        "[udf cut sweep recommended]",
                        f"item={item_index}",
                        f"seed={recommended['seed_world']:.6g}",
                        f"grow={recommended['grow_world']:.6g}",
                        f"removed={recommended['removed_faces']}",
                        f"removed_pct={100.0 * recommended['removed_fraction']:.2f}",
                        f"components={recommended['after_components']}",
                        f"boundary_edges={recommended['after_boundary_edges']}",
                    )
            preserve_existing_boundaries = bool(
                config.get("udf_cut_preserve_existing_boundaries", True)
            )
            reference_max_distance = config.get(
                "udf_cut_reference_max_distance_world", None
            )
            if reference_max_distance is not None:
                reference_max_distance = float(reference_max_distance)

            (
                kept,
                removed,
                seed_mask,
                grow_mask,
                decisions,
                remove_face_mask,
            ) = self._cut_mesh(
                mesh,
                face_scores=scores,
                usable_faces=usable,
                seed_world=seed_world,
                grow_world=grow_world,
                min_faces=int(config.get("udf_cut_min_faces", 8)),
                min_seed_faces=int(config.get("udf_cut_min_seed_faces", 2)),
                min_seed_fraction=float(
                    config.get("udf_cut_min_seed_fraction", 0.05)
                ),
                min_area_world2=float(
                    config.get("udf_cut_min_area_world2", 0.0)
                ),
                preserve_existing_boundaries=preserve_existing_boundaries,
                reference_points=reference_points,
                reference_max_distance_world=reference_max_distance,
            )

            whole_face_kept = kept.copy()
            subtriangle_enabled = bool(config.get("udf_cut_subtriangle", False))
            subtriangle_report = None
            vertex_values = None
            vertex_valid = None
            if subtriangle_enabled and np.any(remove_face_mask):
                vertex_values, vertex_valid = self._query_vertex_values(
                    mesh,
                    avatar_curve=avatar_curve,
                    adapt_arg=adapt_arg,
                    accessory_key=accessory_key,
                    model_batch_size=model_batch_size,
                    query_batch_size=query_batch_size,
                    far_world=far_world,
                )
                subtriangle_threshold = config.get(
                    "udf_cut_subtriangle_threshold_world", None
                )
                if subtriangle_threshold is None:
                    subtriangle_threshold = grow_world
                subtriangle_threshold = float(subtriangle_threshold)
                scalar_smooth_enabled = bool(
                    config.get(
                        "udf_cut_subtriangle_scalar_smooth", False
                    )
                )
                kept, removed, subtriangle_report = clip_accepted_udf_components(
                    mesh,
                    accepted_face_mask=remove_face_mask,
                    vertex_values=vertex_values,
                    vertex_valid=vertex_valid,
                    centroid_values=sample_values[:, 0],
                    centroid_valid=sample_valid[:, 0],
                    threshold_world=subtriangle_threshold,
                    expansion_rings=int(
                        config.get("udf_cut_subtriangle_expansion_rings", 1)
                    ),
                    epsilon=float(
                        config.get("udf_cut_subtriangle_epsilon", 1e-10)
                    ),
                    min_area_world2=float(
                        config.get(
                            "udf_cut_subtriangle_min_area_world2", 1e-14
                        )
                    ),
                    scalar_smooth_rings=(
                        int(
                            config.get(
                                "udf_cut_subtriangle_scalar_smooth_rings", 2
                            )
                        )
                        if scalar_smooth_enabled
                        else 0
                    ),
                    scalar_smooth_iterations=(
                        int(
                            config.get(
                                "udf_cut_subtriangle_scalar_smooth_iterations",
                                3,
                            )
                        )
                        if scalar_smooth_enabled
                        else 0
                    ),
                    scalar_smooth_alpha=float(
                        config.get(
                            "udf_cut_subtriangle_scalar_smooth_alpha", 0.35
                        )
                    ),
                )
                print(
                    "[udf cut subtriangle]",
                    f"threshold={subtriangle_threshold}",
                    f"accepted_faces={subtriangle_report['accepted_faces']}",
                    f"active_faces={subtriangle_report['active_faces']}",
                    f"mixed_faces={subtriangle_report['mixed_faces']}",
                    f"inserted_vertices={subtriangle_report['inserted_vertices']}",
                    f"unresolved_crossings={subtriangle_report['unresolved_active_boundary_crossings']}",
                    f"scalar_smooth={subtriangle_report['scalar_smoothing_enabled']}",
                    f"scalar_changed={subtriangle_report['scalar_smooth_changed_vertices']}",
                    f"scalar_flips={subtriangle_report['scalar_smooth_threshold_flips']}",
                    f"scalar_max_delta={subtriangle_report['scalar_smooth_max_abs_delta']:.6g}",
                )

            kept_raw = kept.copy()
            cleanup_enabled = bool(
                config.get("udf_cut_cleanup_boundary", False)
            )
            cleanup_report = None
            boundary_before_cloud = None
            boundary_after_cloud = None
            if cleanup_enabled:
                boundary_before_cloud = boundary_point_cloud(kept_raw)
                kept, cleanup_report = cleanup_cut_boundary(
                    kept_raw,
                    **self._boundary_cleanup_kwargs(config),
                )
                boundary_after_cloud = boundary_point_cloud(kept)
                print(
                    "[udf cut boundary cleanup]",
                    f"filled={sum(1 for item in cleanup_report['filled_loops'] if item.get('filled'))}",
                    f"strip_rebuilt={sum(1 for item in cleanup_report['strip_rebuild_loops'] if item.get('accepted'))}",
                    f"splined={sum(1 for item in cleanup_report['spline_loops'] if item.get('accepted'))}",
                    f"smoothed={len(cleanup_report['smoothed_loops'])}",
                    f"boundary_loops_before={len(cleanup_report['boundary_before'])}",
                    f"boundary_loops_after={len(cleanup_report['boundary_after'])}",
                )

            if combined_transfer_enabled:
                combined_removed_references.append(removed.copy())
                combined_kept_references.append(kept_raw.copy())

            safe_key = self._safe_label(accessory_key)
            prefix = f"{item_index}_{mode}_{safe_key}"
            before_path = op.join(output_folder, f"{prefix}_mc_before_udf_cut.ply")
            seed_path = op.join(output_folder, f"{prefix}_udf_cut_seed_faces.ply")
            grow_path = op.join(output_folder, f"{prefix}_udf_cut_grow_faces.ply")
            removed_path = op.join(output_folder, f"{prefix}_udf_cut_removed_caps.ply")
            after_path = op.join(output_folder, f"{prefix}_mc_after_udf_cut.ply")
            whole_face_after_path = (
                op.join(
                    output_folder,
                    f"{prefix}_mc_after_udf_cut_whole_face.ply",
                )
                if subtriangle_enabled
                else None
            )
            raw_after_path = (
                op.join(output_folder, f"{prefix}_mc_after_udf_cut_raw.ply")
                if cleanup_enabled
                else None
            )
            boundary_before_path = (
                op.join(output_folder, f"{prefix}_udf_cut_boundary_before.ply")
                if cleanup_enabled
                else None
            )
            boundary_after_path = (
                op.join(output_folder, f"{prefix}_udf_cut_boundary_after.ply")
                if cleanup_enabled
                else None
            )
            npz_path = op.join(output_folder, f"{prefix}_udf_cut_face_scores.npz")
            report_path = op.join(output_folder, f"{prefix}_udf_cut_report.json")

            if write_debug_outputs:
                mesh.export(before_path)
                self._mesh_from_face_mask(mesh, seed_mask).export(seed_path)
                self._mesh_from_face_mask(mesh, grow_mask).export(grow_path)
                removed.export(removed_path)
                if subtriangle_enabled:
                    whole_face_kept.export(whole_face_after_path)
                if cleanup_enabled:
                    kept_raw.export(raw_after_path)
                    if len(boundary_before_cloud.vertices):
                        boundary_before_cloud.export(boundary_before_path)
                    else:
                        boundary_before_path = None
                    if len(boundary_after_cloud.vertices):
                        boundary_after_cloud.export(boundary_after_path)
                    else:
                        boundary_after_path = None
                kept.export(after_path)
                np.savez_compressed(
                    npz_path,
                    face_scores=scores,
                    usable_faces=usable,
                    sample_values=sample_values,
                    sample_valid=sample_valid,
                    seed_mask=seed_mask,
                    grow_mask=grow_mask,
                    remove_face_mask=remove_face_mask,
                    vertex_values=(
                        vertex_values
                        if vertex_values is not None
                        else np.zeros((0,), dtype=np.float64)
                    ),
                    vertex_valid=(
                        vertex_valid
                        if vertex_valid is not None
                        else np.zeros((0,), dtype=bool)
                    ),
                )
            else:
                before_path = None
                seed_path = None
                grow_path = None
                removed_path = None
                whole_face_after_path = None
                raw_after_path = None
                boundary_before_path = None
                boundary_after_path = None
                after_path = None
                npz_path = None

            report = {
                "mc_input": mc_path,
                "shared_snug_field": shared_snug_path,
                "nsdudf_reference": reference_path,
                "target_key": target_key,
                "accessory_key": accessory_key,
                "parameters": {
                    "seed_world": seed_world,
                    "grow_world": grow_world,
                    "min_faces": int(config.get("udf_cut_min_faces", 8)),
                    "min_seed_faces": int(
                        config.get("udf_cut_min_seed_faces", 2)
                    ),
                    "min_seed_fraction": float(
                        config.get("udf_cut_min_seed_fraction", 0.05)
                    ),
                    "min_area_world2": float(
                        config.get("udf_cut_min_area_world2", 0.0)
                    ),
                    "preserve_existing_boundaries": preserve_existing_boundaries,
                    "reference_max_distance_world": reference_max_distance,
                    "auto_thresholds": auto_threshold_report,
                    "subtriangle": subtriangle_enabled,
                    "subtriangle_threshold_world": (
                        None
                        if subtriangle_report is None
                        else subtriangle_report["threshold_world"]
                    ),
                    "subtriangle_expansion_rings": int(
                        config.get("udf_cut_subtriangle_expansion_rings", 1)
                    ),
                    "subtriangle_scalar_smooth": bool(
                        config.get(
                            "udf_cut_subtriangle_scalar_smooth", False
                        )
                    ),
                    "subtriangle_scalar_smooth_rings": int(
                        config.get(
                            "udf_cut_subtriangle_scalar_smooth_rings", 2
                        )
                    ),
                    "subtriangle_scalar_smooth_iterations": int(
                        config.get(
                            "udf_cut_subtriangle_scalar_smooth_iterations", 3
                        )
                    ),
                    "subtriangle_scalar_smooth_alpha": float(
                        config.get(
                            "udf_cut_subtriangle_scalar_smooth_alpha", 0.35
                        )
                    ),
                },
                "face_query": {
                    "faces": int(len(mesh.faces)),
                    "usable_faces": int(np.sum(usable)),
                    "seed_faces": int(np.sum(seed_mask)),
                    "grow_faces": int(np.sum(grow_mask)),
                    "score_percentiles": score_percentiles,
                },
                "before": self._topology_stats(mesh),
                "removed": self._topology_stats(removed),
                "whole_face_after": self._topology_stats(whole_face_kept),
                "after_raw": self._topology_stats(kept_raw),
                "after": self._topology_stats(kept),
                "subtriangle_cut": subtriangle_report,
                "boundary_cleanup": cleanup_report,
                "components": [decision.as_dict() for decision in decisions],
                "outputs": {
                    "before": before_path,
                    "seed_faces": seed_path,
                    "grow_faces": grow_path,
                    "removed_caps": removed_path,
                    "whole_face_after": whole_face_after_path,
                    "after_raw": raw_after_path,
                    "boundary_before": boundary_before_path,
                    "boundary_after": boundary_after_path,
                    "after": after_path,
                    "face_scores": npz_path,
                },
            }
            with open(report_path, "w", encoding="utf-8") as handle_out:
                json.dump(report, handle_out, indent=2)

            print(
                "[udf cut result]",
                f"usable={int(np.sum(usable))}/{len(mesh.faces)}",
                f"seed={int(np.sum(seed_mask))}",
                f"grow={int(np.sum(grow_mask))}",
                f"removed={len(removed.faces)}",
                f"before={report['before']}",
                f"after={report['after']}",
            )
            if after_path is not None:
                print("[udf cut saved]", after_path)

        if all_sweep_summaries:
            all_sweep_report_path = op.join(
                output_folder, "udf_cut_sweep_all.json"
            )
            with open(all_sweep_report_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "items": all_sweep_summaries,
                        "count": len(all_sweep_summaries),
                    },
                    handle,
                    indent=2,
                )
            print(
                "[udf cut sweep all]",
                f"items={len(all_sweep_summaries)}",
                f"report={all_sweep_report_path}",
            )

        if combined_transfer_enabled:
            first_item = adaptation_items[0]
            combined_path = self._resolve_path_spec(
                str(combined_mc_spec),
                root_path=root_path,
                item_index=0,
                mode=str(first_item.get("mode", "direct")),
                accessory_key=str(first_item["accessory_key"]),
                target_key=str(first_item["target_key"]),
            )
            combined_mesh = trimesh.load(
                combined_path, process=False, force="mesh"
            )
            if (
                not isinstance(combined_mesh, trimesh.Trimesh)
                or len(combined_mesh.faces) == 0
            ):
                raise ValueError(
                    f"Combined MC input is empty or not a triangle mesh: {combined_path}"
                )

            print(
                "[udf cut combined transfer]",
                f"combined_mc={combined_path}",
                f"removed_refs={len(combined_removed_references)}",
                f"kept_refs={len(combined_kept_references)}",
            )
            combined_raw, combined_removed, combined_transfer_report = (
                transfer_udf_cut_to_combined_mesh(
                    combined_mesh,
                    removed_reference_meshes=combined_removed_references,
                    kept_reference_meshes=combined_kept_references,
                    radius_edge_fraction=float(
                        extraction_config.get(
                            "udf_cut_combined_transfer_radius_edges", 2.0
                        )
                    ),
                    seed_advantage_edge_fraction=float(
                        extraction_config.get(
                            "udf_cut_combined_transfer_seed_advantage_edges",
                            0.30,
                        )
                    ),
                    grow_advantage_edge_fraction=float(
                        extraction_config.get(
                            "udf_cut_combined_transfer_grow_advantage_edges",
                            0.0,
                        )
                    ),
                    min_faces=int(
                        extraction_config.get(
                            "udf_cut_combined_transfer_min_faces", 8
                        )
                    ),
                    min_seed_faces=int(
                        extraction_config.get(
                            "udf_cut_combined_transfer_min_seed_faces", 2
                        )
                    ),
                    min_seed_fraction=float(
                        extraction_config.get(
                            "udf_cut_combined_transfer_min_seed_fraction", 0.05
                        )
                    ),
                    min_valid_samples=int(
                        extraction_config.get(
                            "udf_cut_combined_transfer_min_valid_samples", 2
                        )
                    ),
                )
            )

            combined_cleanup_report = None
            combined_final = combined_raw
            if bool(extraction_config.get("udf_cut_cleanup_boundary", False)):
                combined_final, combined_cleanup_report = cleanup_cut_boundary(
                    combined_raw,
                    **self._boundary_cleanup_kwargs(extraction_config),
                )

            combined_before_path = op.join(
                output_folder, "combined_mc_before_udf_transfer.ply"
            )
            combined_removed_path = op.join(
                output_folder, "combined_udf_transfer_removed.ply"
            )
            combined_raw_path = op.join(
                output_folder, "combined_mc_after_udf_cut_raw.ply"
            )
            combined_after_path = op.join(
                output_folder, "combined_mc_after_udf_cut.ply"
            )
            combined_report_path = op.join(
                output_folder, "combined_udf_cut_report.json"
            )

            if write_debug_outputs:
                combined_mesh.export(combined_before_path)
                combined_removed.export(combined_removed_path)
                combined_raw.export(combined_raw_path)
            else:
                combined_before_path = None
                combined_removed_path = None
                combined_raw_path = None
            # The final adaptation is always written.
            combined_final.export(combined_after_path)
            with open(combined_report_path, "w", encoding="utf-8") as handle_out:
                json.dump(
                    {
                        "combined_mc_input": combined_path,
                        "before": self._topology_stats(combined_mesh),
                        "after_raw": self._topology_stats(combined_raw),
                        "after": self._topology_stats(combined_final),
                        "transfer": combined_transfer_report,
                        "boundary_cleanup": combined_cleanup_report,
                        "outputs": {
                            "before": combined_before_path,
                            "removed": combined_removed_path,
                            "after_raw": combined_raw_path,
                            "after": combined_after_path,
                        },
                    },
                    handle_out,
                    indent=2,
                )

            print(
                "[udf cut combined result]",
                f"removed_faces={len(combined_removed.faces)}",
                f"before={self._topology_stats(combined_mesh)}",
                f"after={self._topology_stats(combined_final)}",
            )
            print("[udf cut combined saved]", combined_after_path)
