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
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import trimesh

from agent_3dvec_udf import AgentUDF
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

    @torch.no_grad()
    def action_part_adapt_udf_cut(self, arg):
        output_folder = arg["output_folder"]
        shape_name = arg["shape"]
        data_root = arg["data_root"]
        root_path = arg.get("root_path", os.getcwd())

        adaptation_items, extraction_config = self._load_adaptations(arg)
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

            finite_scores = scores[np.isfinite(scores)]
            score_percentiles = {}
            if len(finite_scores):
                for q in (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100):
                    score_percentiles[str(q)] = float(np.percentile(finite_scores, q))

            seed_world = float(config.get("udf_cut_seed_world", 0.02))
            grow_world = float(config.get("udf_cut_grow_world", 0.01))
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
                    fill_small_holes=bool(
                        config.get("udf_cut_fill_small_holes", True)
                    ),
                    fill_max_edges=int(
                        config.get("udf_cut_fill_hole_max_edges", 24)
                    ),
                    fill_max_perimeter_world=float(
                        config.get(
                            "udf_cut_fill_hole_max_perimeter_world", 0.08
                        )
                    ),
                    fill_max_span_world=float(
                        config.get("udf_cut_fill_hole_max_span_world", 0.04)
                    ),
                    smooth_iterations=int(
                        config.get("udf_cut_boundary_smooth_iterations", 8)
                    ),
                    smooth_lambda=float(
                        config.get("udf_cut_boundary_smooth_lambda", 0.45)
                    ),
                    smooth_mu=float(
                        config.get("udf_cut_boundary_smooth_mu", -0.47)
                    ),
                    smooth_min_edges=int(
                        config.get("udf_cut_boundary_smooth_min_edges", 12)
                    ),
                    smooth_max_step_fraction=float(
                        config.get(
                            "udf_cut_boundary_max_step_fraction", 0.25
                        )
                    ),
                    smooth_max_total_fraction=float(
                        config.get(
                            "udf_cut_boundary_max_total_fraction", 0.75
                        )
                    ),
                )
                boundary_after_cloud = boundary_point_cloud(kept)
                print(
                    "[udf cut boundary cleanup]",
                    f"filled={sum(1 for item in cleanup_report['filled_loops'] if item.get('filled'))}",
                    f"smoothed={len(cleanup_report['smoothed_loops'])}",
                    f"boundary_loops_before={len(cleanup_report['boundary_before'])}",
                    f"boundary_loops_after={len(cleanup_report['boundary_after'])}",
                )

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
            print("[udf cut saved]", after_path)
