#!/usr/bin/env python3
"""
Close global and per-part meshes once, cache both the closed mesh and the
artificial cap-only mesh, and evaluate negative-inside signed distances against
closed per-part meshes.

The cap-only mesh is used by process_data_3dvec_closed_v2.py to add explicit
on-surface and perturbed supervision at every artificial closure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from close_all_boundaries import (
    BoundaryError,
    close_all_boundaries,
    describe_boundaries,
    load_triangle_mesh,
)


@dataclass(frozen=True)
class ClosedMeshAsset:
    source_path: str
    closed_path: str
    cap_path: str | None
    source_face_count: int
    boundary_loop_count: int
    boundary_loop_sizes: tuple[int, ...]
    cap_face_count: int
    cap_area: float
    watertight: bool
    winding_consistent: bool
    is_volume: bool


def safe_name(value: Any) -> str:
    text = str(value)
    return "".join(
        character
        if character.isalnum() or character in ("-", "_", ".")
        else "_"
        for character in text
    )


def resolve_item_path(
    path_spec: str,
    item_path: str | Path,
    item_name: str,
) -> Path:
    """
    Resolve a CLI path option.

    Supported forms:
      absolute:
          /data/puffer/full_parts

      relative to each item:
          parts/full

      formatted:
          {item_path}/parts/full
          /shared/{name}/parts/full
    """
    item_path = Path(item_path)

    formatted = str(path_spec).format(
        item_path=str(item_path),
        name=item_name,
    )
    path = Path(formatted).expanduser()

    if not path.is_absolute():
        path = item_path / path

    return path.resolve()


def find_part_mesh(
    parts_dir: str | Path,
    segment_id: int,
    segment_name: str,
    pattern: str,
    part_kind: str,
) -> Path:
    """
    Resolve one part mesh.

    pattern may use:
        {id}
        {name}      sanitized segment name
        {raw_name}  original segment name

    Exact example:
        {id}_{name}.ply

    If the exact name is absent, a conservative fallback searches:
        <id>_*.ply
    and succeeds only when exactly one file matches.
    """
    parts_dir = Path(parts_dir)
    sanitized = safe_name(segment_name)

    relative = pattern.format(
        id=int(segment_id),
        name=sanitized,
        raw_name=segment_name,
    )
    exact = parts_dir / relative

    if exact.exists():
        return exact.resolve()

    matches = sorted(parts_dir.glob(f"{int(segment_id)}_*.ply"))

    if len(matches) == 1:
        print(
            f"[part path fallback] kind={part_kind} "
            f"id={segment_id} name={segment_name!r} -> {matches[0]}"
        )
        return matches[0].resolve()

    if not parts_dir.exists():
        raise FileNotFoundError(
            f"{part_kind} part directory does not exist: {parts_dir}"
        )

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No {part_kind} mesh found for segment "
            f"id={segment_id}, name={segment_name!r}.\n"
            f"Expected: {exact}\n"
            f"Fallback glob: {parts_dir / f'{segment_id}_*.ply'}"
        )

    raise RuntimeError(
        f"Ambiguous {part_kind} meshes for segment id={segment_id}: "
        f"{[str(path) for path in matches]}"
    )


def close_mesh_cached(
    source_path: str | Path,
    closed_path: str | Path,
    reclose: bool = False,
) -> ClosedMeshAsset:
    """
    Close every simple open boundary loop in source_path.

    The cache contains:
      - the validated watertight closed mesh,
      - a cap-only mesh containing exactly the artificial faces,
      - a JSON report with source/cap counts and paths.

    The cap-only mesh is required for explicit cap supervision later.
    """
    source_path = Path(source_path).resolve()
    closed_path = Path(closed_path).resolve()
    cap_path = closed_path.with_name(closed_path.stem + "_caps_only.ply")
    report_path = closed_path.with_suffix(
        closed_path.suffix + ".closure.json"
    )

    if not source_path.exists():
        raise FileNotFoundError(f"Input mesh does not exist: {source_path}")

    def asset_from_report(report: dict) -> ClosedMeshAsset:
        cap_count = int(report.get("cap_face_count", 0))
        reported_cap_path = report.get("cap_path", None)
        resolved_cap_path = None
        if cap_count > 0:
            resolved_cap_path = str(
                Path(reported_cap_path).resolve()
                if reported_cap_path
                else cap_path
            )

        return ClosedMeshAsset(
            source_path=str(source_path),
            closed_path=str(closed_path),
            cap_path=resolved_cap_path,
            source_face_count=int(report.get("source_faces", 0)),
            boundary_loop_count=int(report.get("boundary_loop_count", 0)),
            boundary_loop_sizes=tuple(
                int(value)
                for value in report.get("boundary_loop_sizes", [])
            ),
            cap_face_count=cap_count,
            cap_area=float(report.get("cap_area", 0.0)),
            watertight=True,
            winding_consistent=True,
            is_volume=True,
        )

    if closed_path.exists() and report_path.exists() and not reclose:
        report = json.loads(report_path.read_text())
        source_stat = source_path.stat()

        cache_matches_source = (
            str(Path(report.get("source_path", "")).resolve())
            == str(source_path)
            and int(report.get("source_size_bytes", -1))
            == int(source_stat.st_size)
            and int(report.get("source_mtime_ns", -1))
            == int(source_stat.st_mtime_ns)
        )

        if not cache_matches_source:
            print(
                "[closure cache stale] source changed; rebuilding:",
                source_path,
            )
        else:
            mesh = load_triangle_mesh(closed_path)
            boundary, _, nonmanifold, _, _ = describe_boundaries(mesh)

            if len(boundary) != 0:
                raise RuntimeError(
                    f"Cached mesh still has {len(boundary)} boundary edges: "
                    f"{closed_path}"
                )
            if len(nonmanifold) != 0:
                raise RuntimeError(
                    f"Cached mesh has {len(nonmanifold)} non-manifold edges: "
                    f"{closed_path}"
                )
            if not mesh.is_watertight:
                raise RuntimeError(
                    f"Cached mesh is not watertight: {closed_path}"
                )
            if not mesh.is_winding_consistent:
                raise RuntimeError(
                    f"Cached mesh has inconsistent winding: {closed_path}"
                )
            if not mesh.is_volume:
                raise RuntimeError(
                    f"Cached mesh is not a valid oriented volume: {closed_path}"
                )

            cap_count = int(report.get("cap_face_count", 0))
            reported_cap_path = report.get("cap_path", None)
            cached_cap_path = (
                Path(reported_cap_path).resolve()
                if reported_cap_path
                else cap_path
            )

            if cap_count > 0 and not cached_cap_path.exists():
                print(
                    "[closure cache incomplete] cap-only mesh missing; rebuilding:",
                    cached_cap_path,
                )
            else:
                return asset_from_report(report)

    mesh = load_triangle_mesh(source_path)

    (
        boundary_before,
        _,
        nonmanifold_before,
        components_before,
        sizes_before,
    ) = describe_boundaries(mesh)

    if len(nonmanifold_before) != 0:
        raise BoundaryError(
            f"{source_path} contains {len(nonmanifold_before)} "
            "non-manifold edges. The automatic loop closer handles simple "
            "boundary loops, not non-manifold junctions."
        )

    if len(boundary_before) == 0:
        closed = mesh.copy()
        loops = []
        cap_faces = np.zeros((0, 3), dtype=np.int64)
        print("[mesh already closed]", source_path)
    else:
        closed, loops, cap_faces = close_all_boundaries(mesh)

    (
        boundary_after,
        _,
        nonmanifold_after,
        _,
        _,
    ) = describe_boundaries(closed)

    if len(boundary_after) != 0:
        raise RuntimeError(
            f"Closure failed for {source_path}: "
            f"{len(boundary_after)} boundary edges remain."
        )

    if len(nonmanifold_after) != 0:
        raise RuntimeError(
            f"Closure produced {len(nonmanifold_after)} non-manifold edges "
            f"for {source_path}."
        )

    trimesh.repair.fix_winding(closed)
    trimesh.repair.fix_normals(closed, multibody=True)

    if not closed.is_watertight:
        raise RuntimeError(f"Closed result is not watertight: {source_path}")
    if not closed.is_winding_consistent:
        raise RuntimeError(
            f"Closed result has inconsistent winding: {source_path}"
        )
    if not closed.is_volume:
        raise RuntimeError(
            f"Closed result does not define a valid oriented volume: "
            f"{source_path}"
        )

    closed_path.parent.mkdir(parents=True, exist_ok=True)
    closed.export(closed_path)

    cap_area = 0.0
    cap_path_value: str | None = None

    if len(cap_faces) > 0:
        # Use the CLOSED vertex array.  Robust non-planar capping may add
        # one Steiner vertex that is referenced only by cap faces.
        cap_mesh = trimesh.Trimesh(
            vertices=np.asarray(closed.vertices, dtype=np.float64).copy(),
            faces=np.asarray(cap_faces, dtype=np.int64).copy(),
            process=False,
        )
        cap_mesh.remove_unreferenced_vertices()

        if len(cap_mesh.faces) == 0 or float(cap_mesh.area) <= 0.0:
            raise RuntimeError(
                f"Artificial cap mesh is empty or has zero area: {source_path}"
            )

        cap_path.parent.mkdir(parents=True, exist_ok=True)
        cap_mesh.export(cap_path)
        cap_area = float(cap_mesh.area)
        cap_path_value = str(cap_path)
    else:
        if cap_path.exists():
            cap_path.unlink()

    source_stat = source_path.stat()

    report = {
        "source_path": str(source_path),
        "source_size_bytes": int(source_stat.st_size),
        "source_mtime_ns": int(source_stat.st_mtime_ns),
        "closed_path": str(closed_path),
        "cap_path": cap_path_value,
        "source_vertices": int(len(mesh.vertices)),
        "source_faces": int(len(mesh.faces)),
        "source_watertight": bool(mesh.is_watertight),
        "boundary_edge_count_before": int(len(boundary_before)),
        "boundary_loop_count": int(len(components_before)),
        "boundary_loop_sizes": [int(value) for value in sizes_before],
        "cap_face_count": int(len(cap_faces)),
        "cap_area": cap_area,
        "closed_vertices": int(len(closed.vertices)),
        "closed_faces": int(len(closed.faces)),
        "boundary_edge_count_after": int(len(boundary_after)),
        "nonmanifold_edge_count_after": int(len(nonmanifold_after)),
        "watertight": bool(closed.is_watertight),
        "winding_consistent": bool(closed.is_winding_consistent),
        "is_volume": bool(closed.is_volume),
        "volume": float(closed.volume),
    }
    report_path.write_text(json.dumps(report, indent=2))

    print(
        "[closed mesh]",
        f"source={source_path}",
        f"loops={len(components_before)}",
        f"cap_faces={len(cap_faces)}",
        f"cap_area={cap_area:.8g}",
        f"output={closed_path}",
    )
    if cap_path_value is not None:
        print("[cap-only mesh]", cap_path_value)

    return ClosedMeshAsset(
        source_path=str(source_path),
        closed_path=str(closed_path),
        cap_path=cap_path_value,
        source_face_count=int(len(mesh.faces)),
        boundary_loop_count=int(len(components_before)),
        boundary_loop_sizes=tuple(int(value) for value in sizes_before),
        cap_face_count=int(len(cap_faces)),
        cap_area=cap_area,
        watertight=True,
        winding_consistent=True,
        is_volume=True,
    )

def prepare_closed_meshes(
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
    closed_mesh_dir_spec: str,
    reclose: bool = False,
) -> tuple[
    ClosedMeshAsset,
    ClosedMeshAsset,
    dict[str, ClosedMeshAsset],
    dict[str, ClosedMeshAsset],
]:
    """
    Resolve all source paths from CLI options, close every source mesh, save
    the results, and return only closed-mesh assets.

    Global meshes are closed and saved for validation/debugging.
    Per-part closed meshes are the SDF oracles.
    """
    item_path = Path(item_path)

    global_full_source = resolve_item_path(
        global_full_mesh_spec,
        item_path,
        item_name,
    )
    global_base_source = resolve_item_path(
        global_base_mesh_spec,
        item_path,
        item_name,
    )
    full_parts_dir = resolve_item_path(
        full_parts_dir_spec,
        item_path,
        item_name,
    )
    base_parts_dir = resolve_item_path(
        base_parts_dir_spec,
        item_path,
        item_name,
    )
    closed_root = resolve_item_path(
        closed_mesh_dir_spec,
        item_path,
        item_name,
    )

    global_full_asset = close_mesh_cached(
        global_full_source,
        closed_root / "global" / "mesh_full_closed.ply",
        reclose=reclose,
    )
    global_base_asset = close_mesh_cached(
        global_base_source,
        closed_root / "global" / "mesh_base_closed.ply",
        reclose=reclose,
    )

    full_assets: dict[str, ClosedMeshAsset] = {}
    base_assets: dict[str, ClosedMeshAsset] = {}

    for curve_id, curve in enumerate(handle.curves):
        curve_name = str(curve.name)
        segment_id = int(getattr(curve, "idx", curve_id))
        output_stem = f"{segment_id:03d}_{safe_name(curve_name)}"

        full_source = find_part_mesh(
            full_parts_dir,
            segment_id=segment_id,
            segment_name=curve_name,
            pattern=full_part_pattern,
            part_kind="full",
        )
        base_source = find_part_mesh(
            base_parts_dir,
            segment_id=segment_id,
            segment_name=curve_name,
            pattern=base_part_pattern,
            part_kind="base",
        )

        full_assets[curve_name] = close_mesh_cached(
            full_source,
            closed_root
            / "parts_full"
            / f"{output_stem}_closed.ply",
            reclose=reclose,
        )
        base_assets[curve_name] = close_mesh_cached(
            base_source,
            closed_root
            / "parts_base"
            / f"{output_stem}_closed.ply",
            reclose=reclose,
        )

    return (
        global_full_asset,
        global_base_asset,
        full_assets,
        base_assets,
    )


class ClosedPartSDFOracle:
    """
    Signed-distance evaluator for one validated CLOSED part mesh.

    Output convention used by the N-GC training and inference pipeline:
        negative = inside
        zero     = surface
        positive = outside

    Trimesh returns the opposite sign, so every query is negated here.

    `surface_tolerance` is an absolute distance in mesh-coordinate units.
    Values with |SDF| <= surface_tolerance are written as exactly zero.
    """

    def __init__(
        self,
        mesh_path: str | Path,
        surface_tolerance: float = 1.0e-6,
    ):
        self.mesh_path = str(Path(mesh_path).resolve())
        self.surface_tolerance = float(surface_tolerance)

        if self.surface_tolerance < 0.0:
            raise ValueError(
                "surface_tolerance must be non-negative, got "
                f"{self.surface_tolerance}"
            )

        self.mesh = load_triangle_mesh(Path(self.mesh_path))

        # Repair triangle ordering only; vertex positions are unchanged.
        trimesh.repair.fix_winding(self.mesh)
        trimesh.repair.fix_normals(self.mesh, multibody=True)

        if not self.mesh.is_watertight:
            raise ValueError(
                f"SDF oracle mesh is not watertight: {self.mesh_path}"
            )
        if not self.mesh.is_winding_consistent:
            raise ValueError(
                f"SDF oracle winding is inconsistent: {self.mesh_path}"
            )
        if not self.mesh.is_volume:
            raise ValueError(
                f"SDF oracle is not a valid oriented volume: "
                f"{self.mesh_path}"
            )

        # Reused for all chunks queried against this part mesh.
        self.query = trimesh.proximity.ProximityQuery(self.mesh)

    def evaluate(
        self,
        samples: np.ndarray,
        chunk_size: int = 200_000,
    ) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float64)

        if samples.ndim != 2 or samples.shape[1] != 3:
            raise ValueError(
                f"samples must have shape (N,3), got {samples.shape}"
            )
        if int(chunk_size) <= 0:
            raise ValueError(
                f"chunk_size must be positive, got {chunk_size}"
            )

        output = np.empty(len(samples), dtype=np.float32)

        for start in range(0, len(samples), int(chunk_size)):
            stop = min(start + int(chunk_size), len(samples))
            query_points = samples[start:stop]

            # Trimesh returns positive-inside / negative-outside.
            raw_signed_distance = np.asarray(
                self.query.signed_distance(query_points),
                dtype=np.float64,
            )

            # Convert once, at the oracle boundary, to the convention used by
            # the rest of the codebase: negative-inside / positive-outside.
            signed_distance = -raw_signed_distance

            # Put the tolerance exactly here:
            # after SDF evaluation and before casting/storing the chunk.
            if self.surface_tolerance > 0.0:
                signed_distance[
                    np.abs(signed_distance) <= self.surface_tolerance
                ] = 0.0

            if not np.isfinite(signed_distance).all():
                raise RuntimeError(
                    f"Non-finite SDF values returned by "
                    f"{self.mesh_path} for samples [{start}:{stop}]"
                )

            output[start:stop] = signed_distance.astype(
                np.float32,
                copy=False,
            )

        return output


class PerCurveClosedSDFEvaluator:
    """
    Select the closed part mesh from data['curve_idx'] and evaluate SDF.
    """

    def __init__(
        self,
        assets_by_curve: dict[str, ClosedMeshAsset],
        surface_tolerance: float = 1.0e-6,
    ):
        self.oracles = {
            curve_name: ClosedPartSDFOracle(
                asset.closed_path,
                surface_tolerance=surface_tolerance,
            )
            for curve_name, asset in assets_by_curve.items()
        }

    def evaluate(
        self,
        data: dict,
        curve_names: list[str],
        chunk_size: int = 200_000,
    ) -> np.ndarray:
        samples = np.asarray(data["samples"], dtype=np.float64)
        curve_idx = np.asarray(data["curve_idx"], dtype=np.int64)

        if len(samples) != len(curve_idx):
            raise ValueError(
                "samples and curve_idx lengths differ: "
                f"{len(samples)} vs {len(curve_idx)}"
            )

        output = np.empty(len(samples), dtype=np.float32)
        assigned = np.zeros(len(samples), dtype=bool)

        for curve_id, curve_name in enumerate(curve_names):
            mask = curve_idx == curve_id
            if not np.any(mask):
                continue

            if curve_name not in self.oracles:
                raise KeyError(
                    f"No closed SDF oracle for curve {curve_name!r}"
                )

            output[mask] = self.oracles[curve_name].evaluate(
                samples[mask],
                chunk_size=chunk_size,
            )
            assigned[mask] = True

        if not np.all(assigned):
            missing_ids = np.unique(curve_idx[~assigned])
            raise RuntimeError(
                "Some samples do not map to a closed part oracle. "
                f"curve_idx values: {missing_ids.tolist()}"
            )

        return output
