#!/usr/bin/env python3
"""
Open-mesh UNSIGNED-distance utilities for UDF preprocessing.

This is the UDF analogue of ``sdf_closure_utils_v2.py``.  Unlike the SDF path,
the source meshes are NOT closed and NO artificial caps are added: a UDF is
unsigned, so watertightness / inside-outside is irrelevant.  Every UDF target
is simply the Euclidean distance from a query point to the ORIGINAL open
per-part triangle mesh.

Public API (matches process_data_3dvec_udf_keep_cylinder.py):
    prepare_open_meshes(...)          -> (global_full, global_base,
                                          full_assets_by_curve,
                                          base_assets_by_curve)
    PerCurveOpenUDFEvaluator(assets_by_curve, surface_tolerance=, truncation=)
        .evaluate(data, curve_names=, chunk_size=) -> np.ndarray (N,) float32
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

try:
    import igl
    _HAS_IGL = True
except Exception:  # pragma: no cover - igl optional, trimesh fallback used
    _HAS_IGL = False


# --------------------------------------------------------------------------- #
# Assets
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OpenMeshAsset:
    """A source mesh used *as-is* (open, no closing, no caps)."""
    source_path: str
    mesh_path: str            # == source_path; kept for API symmetry with SDF
    source_face_count: int
    source_vertex_count: int
    watertight: bool


# --------------------------------------------------------------------------- #
# Path helpers (copied verbatim from the SDF module; representation-agnostic)
# --------------------------------------------------------------------------- #
def safe_name(value: Any) -> str:
    text = str(value)
    return "".join(
        c if c.isalnum() or c in ("-", "_", ".") else "_"
        for c in text
    )


def resolve_item_path(path_spec: str, item_path: str | Path, item_name: str) -> Path:
    item_path = Path(item_path)
    formatted = str(path_spec).format(item_path=str(item_path), name=item_name)
    path = Path(formatted).expanduser()
    if not path.is_absolute():
        path = item_path / path
    return path.resolve()


def find_part_mesh(parts_dir, segment_id, segment_name, pattern, part_kind) -> Path:
    parts_dir = Path(parts_dir)
    sanitized = safe_name(segment_name)
    relative = pattern.format(id=int(segment_id), name=sanitized, raw_name=segment_name)
    exact = parts_dir / relative
    if exact.exists():
        return exact.resolve()

    matches = sorted(parts_dir.glob(f"{int(segment_id)}_*.ply"))
    if len(matches) == 1:
        print(f"[part path fallback] kind={part_kind} id={segment_id} "
              f"name={segment_name!r} -> {matches[0]}")
        return matches[0].resolve()
    if not parts_dir.exists():
        raise FileNotFoundError(f"{part_kind} part directory does not exist: {parts_dir}")
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No {part_kind} mesh found for segment id={segment_id}, "
            f"name={segment_name!r}.\nExpected: {exact}\n"
            f"Fallback glob: {parts_dir / f'{segment_id}_*.ply'}")
    raise RuntimeError(
        f"Ambiguous {part_kind} meshes for segment id={segment_id}: "
        f"{[str(p) for p in matches]}")


def _load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load a triangle mesh WITHOUT any processing/repair (positions preserved)."""
    mesh = trimesh.load(str(path), process=False, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
    return mesh


def open_mesh_asset(source_path: str | Path) -> OpenMeshAsset:
    source_path = Path(source_path).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Input mesh does not exist: {source_path}")
    mesh = _load_mesh(source_path)
    return OpenMeshAsset(
        source_path=str(source_path),
        mesh_path=str(source_path),
        source_face_count=int(len(mesh.faces)),
        source_vertex_count=int(len(mesh.vertices)),
        watertight=bool(mesh.is_watertight),
    )


# --------------------------------------------------------------------------- #
# prepare_open_meshes: resolve paths, NO closing, NO caps.
# --------------------------------------------------------------------------- #
def prepare_open_meshes(
    *,
    item_path: str | Path,
    item_name: str,
    handle,
    global_full_mesh_spec: str,
    global_base_mesh_spec: str,
    full_parts_dir_spec: str,
    base_parts_dir_spec: str,
    full_part_pattern: str,
    base_part_pattern: str,
) -> tuple[OpenMeshAsset, OpenMeshAsset,
           dict[str, OpenMeshAsset], dict[str, OpenMeshAsset]]:
    """
    Resolve the global full/base meshes and every per-part full/base mesh from
    the same CLI specs the SDF path uses, but return them as OPEN assets:
    nothing is closed and no caps are generated.  Per-part open meshes are the
    UDF oracles.
    """
    item_path = Path(item_path)

    global_full_source = resolve_item_path(global_full_mesh_spec, item_path, item_name)
    global_base_source = resolve_item_path(global_base_mesh_spec, item_path, item_name)
    full_parts_dir = resolve_item_path(full_parts_dir_spec, item_path, item_name)
    base_parts_dir = resolve_item_path(base_parts_dir_spec, item_path, item_name)

    global_full_asset = open_mesh_asset(global_full_source)
    global_base_asset = open_mesh_asset(global_base_source)

    full_assets: dict[str, OpenMeshAsset] = {}
    base_assets: dict[str, OpenMeshAsset] = {}

    for curve_id, curve in enumerate(handle.curves):
        curve_name = str(curve.name)
        segment_id = int(getattr(curve, "idx", curve_id))

        full_source = find_part_mesh(
            full_parts_dir, segment_id, curve_name, full_part_pattern, "full")
        base_source = find_part_mesh(
            base_parts_dir, segment_id, curve_name, base_part_pattern, "base")

        full_assets[curve_name] = open_mesh_asset(full_source)
        base_assets[curve_name] = open_mesh_asset(base_source)

    return global_full_asset, global_base_asset, full_assets, base_assets


# --------------------------------------------------------------------------- #
# Unsigned-distance oracle
# --------------------------------------------------------------------------- #
class OpenPartUDFOracle:
    """
    Unsigned-distance evaluator for one OPEN part mesh (triangle soup).

    UDF convention:
        0    = on the surface
        > 0  = distance to the nearest triangle (always non-negative)

    `surface_tolerance` : distances <= this are stored as exactly 0.
    `truncation`        : distances are clamped to at most this value
                          (set None or <= 0 to disable).
    """

    def __init__(self, mesh_path, surface_tolerance: float = 1.0e-6,
                 truncation: float | None = 0.1):
        self.mesh_path = str(Path(mesh_path).resolve())

        self.surface_tolerance = float(surface_tolerance)
        if self.surface_tolerance < 0.0:
            raise ValueError(
                f"surface_tolerance must be non-negative, got {self.surface_tolerance}")

        self.truncation = None if truncation is None else float(truncation)
        if self.truncation is not None and self.truncation <= 0.0:
            self.truncation = None

        mesh = _load_mesh(self.mesh_path)
        self.vertices = np.asarray(mesh.vertices, dtype=np.float64)
        self.faces = np.asarray(mesh.faces, dtype=np.int64)
        if len(self.faces) == 0:
            raise ValueError(f"UDF oracle mesh has no faces: {self.mesh_path}")

        # trimesh fallback when igl is not present.
        self._pq = None if _HAS_IGL else trimesh.proximity.ProximityQuery(mesh)

    def _distance(self, query_points: np.ndarray) -> np.ndarray:
        if _HAS_IGL:
            sqr_d, _, _ = igl.point_mesh_squared_distance(
                query_points, self.vertices, self.faces)
            return np.sqrt(np.maximum(np.asarray(sqr_d, dtype=np.float64), 0.0))
        # trimesh: on_surface returns (closest_points, distances, triangle_ids)
        _, distances, _ = self._pq.on_surface(query_points)
        return np.asarray(distances, dtype=np.float64)

    def evaluate(self, samples: np.ndarray, chunk_size: int = 200_000) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[1] != 3:
            raise ValueError(f"samples must have shape (N,3), got {samples.shape}")
        if int(chunk_size) <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        output = np.empty(len(samples), dtype=np.float32)
        for start in range(0, len(samples), int(chunk_size)):
            stop = min(start + int(chunk_size), len(samples))
            distance = self._distance(samples[start:stop])

            if self.surface_tolerance > 0.0:
                distance[distance <= self.surface_tolerance] = 0.0
            if self.truncation is not None:
                distance = np.minimum(distance, self.truncation)

            if not np.isfinite(distance).all():
                raise RuntimeError(
                    f"Non-finite UDF values from {self.mesh_path} "
                    f"for samples [{start}:{stop}]")

            output[start:stop] = distance.astype(np.float32, copy=False)

        return output


class PerCurveOpenUDFEvaluator:
    """Select the open part mesh from data['curve_idx'] and evaluate the UDF."""

    def __init__(self, assets_by_curve: dict[str, OpenMeshAsset],
                 surface_tolerance: float = 1.0e-6,
                 truncation: float | None = 0.1):
        self.oracles = {
            curve_name: OpenPartUDFOracle(
                asset.mesh_path,
                surface_tolerance=surface_tolerance,
                truncation=truncation,
            )
            for curve_name, asset in assets_by_curve.items()
        }

    def evaluate(self, data: dict, curve_names: list[str],
                 chunk_size: int = 200_000) -> np.ndarray:
        samples = np.asarray(data["samples"], dtype=np.float64)
        curve_idx = np.asarray(data["curve_idx"], dtype=np.int64)
        if len(samples) != len(curve_idx):
            raise ValueError(
                f"samples and curve_idx lengths differ: {len(samples)} vs {len(curve_idx)}")

        output = np.empty(len(samples), dtype=np.float32)
        assigned = np.zeros(len(samples), dtype=bool)

        for curve_id, curve_name in enumerate(curve_names):
            mask = curve_idx == curve_id
            if not np.any(mask):
                continue
            if curve_name not in self.oracles:
                raise KeyError(f"No open UDF oracle for curve {curve_name!r}")
            output[mask] = self.oracles[curve_name].evaluate(
                samples[mask], chunk_size=chunk_size)
            assigned[mask] = True

        if not np.all(assigned):
            missing = np.unique(curve_idx[~assigned])
            raise RuntimeError(
                "Some samples do not map to an open part oracle. "
                f"curve_idx values: {missing.tolist()}")

        return output
