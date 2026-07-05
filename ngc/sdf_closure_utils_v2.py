#!/usr/bin/env python3
"""
Close global and per-part meshes once, cache the closed meshes, and evaluate
inside-negative SDF values only against those closed per-part meshes.

This module does NOT add cap points to training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import pymeshlab as ml

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
    boundary_loop_count: int
    boundary_loop_sizes: tuple[int, ...]
    cap_face_count: int
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

    The closed mesh and a JSON validation report are cached. If the cached
    result exists and reclose=False, it is validated and reused.
    """
    source_path = Path(source_path).resolve()
    closed_path = Path(closed_path).resolve()
    report_path = closed_path.with_suffix(
        closed_path.suffix + ".closure.json"
    )

    if not source_path.exists():
        raise FileNotFoundError(f"Input mesh does not exist: {source_path}")

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
            boundary, _, nonmanifold, components, sizes = describe_boundaries(mesh)

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

            return ClosedMeshAsset(
                source_path=str(source_path),
                closed_path=str(closed_path),
                boundary_loop_count=int(
                    report.get("boundary_loop_count", 0)
                ),
                boundary_loop_sizes=tuple(
                    int(value)
                    for value in report.get("boundary_loop_sizes", [])
                ),
                cap_face_count=int(report.get("cap_face_count", 0)),
                watertight=True,
                winding_consistent=True,
                is_volume=True,
            )

    if closed_path.exists() and report_path.exists() and not reclose:
        # Reaching here means the cache was stale and must be regenerated.
        pass

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

    closed, loops, cap_faces = close_all_boundaries(mesh)

    (
        boundary_after,
        _,
        nonmanifold_after,
        components_after,
        sizes_after,
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

    # Re-run normal repair before the final volume checks.
    trimesh.repair.fix_winding(closed)
    trimesh.repair.fix_normals(closed, multibody=True)

    if not closed.is_watertight:
        raise RuntimeError(
            f"Closed result is not watertight: {source_path}"
        )
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

    source_stat = source_path.stat()

    report = {
        "source_path": str(source_path),
        "source_size_bytes": int(source_stat.st_size),
        "source_mtime_ns": int(source_stat.st_mtime_ns),
        "closed_path": str(closed_path),
        "source_vertices": int(len(mesh.vertices)),
        "source_faces": int(len(mesh.faces)),
        "source_watertight": bool(mesh.is_watertight),
        "boundary_edge_count_before": int(len(boundary_before)),
        "boundary_loop_count": int(len(components_before)),
        "boundary_loop_sizes": [int(value) for value in sizes_before],
        "cap_face_count": int(len(cap_faces)),
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
        f"output={closed_path}",
    )

    return ClosedMeshAsset(
        source_path=str(source_path),
        closed_path=str(closed_path),
        boundary_loop_count=int(len(components_before)),
        boundary_loop_sizes=tuple(int(value) for value in sizes_before),
        cap_face_count=int(len(cap_faces)),
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
    print('item_path = ', item_path)
    print('item_name = ', item_name)

    global_full_source = resolve_item_path(
        global_full_mesh_spec,
        item_path,
        item_name,
    )
    print(global_full_source)
    global_base_source = resolve_item_path(
        global_base_mesh_spec,
        item_path,
        item_name,
    )
    print(global_base_source)
    full_parts_dir = resolve_item_path(
        full_parts_dir_spec,
        item_path,
        item_name,
    )
    print(full_parts_dir)
    base_parts_dir = resolve_item_path(
        base_parts_dir_spec,
        item_path,
        item_name,
    )
    print(base_parts_dir)
    closed_root = resolve_item_path(
        closed_mesh_dir_spec,
        item_path,
        item_name,
    )
    print(closed_root)

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
    SDF evaluator for one validated closed part mesh.

    This intentionally uses the same PyMeshLab distance filter as the original
    process_data_3dvec.py. The only semantic change is that the reference mesh
    is now the validated CLOSED part mesh.

    The returned sign convention is therefore preserved from the existing
    preprocessing pipeline.
    """

    def __init__(self, mesh_path: str | Path):
        self.mesh_path = str(Path(mesh_path).resolve())
        self.mesh = load_triangle_mesh(Path(self.mesh_path))

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
                f"SDF oracle is not a valid volume: {self.mesh_path}"
            )

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

        output = np.empty(len(samples), dtype=np.float32)

        for start in range(0, len(samples), int(chunk_size)):
            stop = min(start + int(chunk_size), len(samples))
            query_points = samples[start:stop]

            # Reproduce the original meshlab_SDF_eval behavior:
            #   mesh 0 = closed reference triangle mesh
            #   mesh 1 = query point cloud and current mesh
            mesh_set = ml.MeshSet()
            mesh_set.load_new_mesh(self.mesh_path)
            mesh_set.add_mesh(ml.Mesh(query_points), "query_points")

            mesh_set.compute_scalar_by_distance_from_another_mesh_per_vertex()

            values = np.asarray(
                mesh_set.current_mesh().vertex_scalar_array(),
                dtype=np.float32,
            )

            if len(values) != len(query_points):
                raise RuntimeError(
                    f"PyMeshLab returned {len(values)} values for "
                    f"{len(query_points)} queries against {self.mesh_path}"
                )

            output[start:stop] = values

        if not np.isfinite(output).all():
            raise RuntimeError(
                f"Non-finite SDF values returned by {self.mesh_path}"
            )

        return output


class PerCurveClosedSDFEvaluator:
    """
    Select the closed part mesh from data['curve_idx'] and evaluate SDF.
    """

    def __init__(self, assets_by_curve: dict[str, ClosedMeshAsset]):
        self.oracles = {
            curve_name: ClosedPartSDFOracle(asset.closed_path)
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
