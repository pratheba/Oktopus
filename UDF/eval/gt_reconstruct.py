#!/usr/bin/env python3
"""
Ground-truth UDF -> mesh reconstruction test (DualMeshUDF).

Validates the surface-extraction path INDEPENDENTLY of any trained network:
build an analytic unsigned-distance field directly from a known input mesh, run
DualMeshUDF's extract_mesh on it, and compare the reconstruction to the input.

Two modes:
  1. Whole mesh:   --mesh <mesh.ply>            -> one reconstruction
  2. Per-curve:    --parts_dir <full_parts/>    -> one reconstruction per part
     (recommended for this pipeline: the model is per-curve, so reconstructing
      each open part mesh on its own is the decisive check.)

The UDF and its gradient come from libigl's exact point-to-mesh distance:
    d(p)      = || p - closest_point_on_mesh(p) ||
    grad d(p) = (p - closest_point) / d          (unit, points away from surface)

DualMeshUDF extracts inside the cube [-1, 1]^3, so each mesh is normalized into
that cube first and the result is mapped back for comparison.

Usage
-----
    # per-curve (each part in the directory reconstructed separately):
    python UDF/eval/gt_reconstruct.py --parts_dir <item>/full_parts --out_dir recon_parts --max_depth 7

    # whole mesh:
    python UDF/eval/gt_reconstruct.py --mesh <item>/mesh.ply --out recon.ply --max_depth 7

Requires: DualMeshUDF (pip install the vendored copy), libigl, trimesh, numpy.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import trimesh

try:
    import igl
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "libigl is required (pip install libigl). Import failed: %r" % exc)

try:
    from DualMeshUDF import extract_mesh
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "DualMeshUDF is required. Install the vendored copy under "
        "third_party/DualMesh-UDF (see third_party/README_DualMeshUDF.md). "
        "Import failed: %r" % exc)


def load_mesh(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), process=False, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])
    return mesh


def normalize_to_cube(V: np.ndarray, pad: float = 0.9):
    """Scale/translate vertices into [-pad, pad]^3. Returns (V_norm, center, scale)."""
    lo = V.min(axis=0)
    hi = V.max(axis=0)
    center = 0.5 * (lo + hi)
    extent = float((hi - lo).max())
    scale = (2.0 * pad) / max(extent, 1e-12)
    V_norm = (V - center) * scale
    return V_norm, center, scale


def make_udf_funcs(V: np.ndarray, F: np.ndarray):
    """Return (udf_func, udf_grad_func) matching DualMeshUDF's expected API."""
    Vd = np.asarray(V, dtype=np.float64)
    Fi = np.asarray(F, dtype=np.int64)

    def _dist_and_closest(pts: np.ndarray):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        sqr_d, _, C = igl.point_mesh_squared_distance(pts, Vd, Fi)
        d = np.sqrt(np.maximum(np.asarray(sqr_d, dtype=np.float64), 0.0))
        return pts, d, np.asarray(C, dtype=np.float64)

    def udf_func(pts):
        _, d, _ = _dist_and_closest(pts)
        return d.reshape(-1, 1).astype(np.float32)

    def udf_grad_func(pts):
        pts, d, C = _dist_and_closest(pts)
        g = pts - C
        n = np.linalg.norm(g, axis=1, keepdims=True)
        n[n == 0] = 1.0
        g = g / n
        return d.reshape(-1, 1).astype(np.float32), g.astype(np.float32)

    return udf_func, udf_grad_func


def reconstruct_one(mesh_path, out_path, max_depth=7, batch_size=150000,
                    pad=0.9, no_normalize=False, label=""):
    """Reconstruct a single mesh's GT UDF via DualMeshUDF. Returns recon-to-input
    distance stats (mean, p95, max) in original units, or None on failure."""
    tag = f"[{label}] " if label else ""
    mesh = load_mesh(mesh_path)
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)
    if len(F) == 0:
        print(f"{tag}SKIP (no faces): {mesh_path}")
        return None
    print(f"{tag}input V={V.shape} F={F.shape} watertight={mesh.is_watertight}")

    if no_normalize:
        Vn, center, scale = V, np.zeros(3), 1.0
    else:
        Vn, center, scale = normalize_to_cube(V, pad=pad)

    udf_func, udf_grad_func = make_udf_funcs(Vn, F)

    v, f = extract_mesh(udf_func, udf_grad_func, batch_size=batch_size, max_depth=max_depth)
    v = np.asarray(v, dtype=np.float64)
    f = np.asarray(f, dtype=np.int64)
    if len(v) == 0 or len(f) == 0:
        print(f"{tag}WARNING: extraction produced an empty mesh")
        return None

    v_orig = (v / scale + center) if not no_normalize else v
    recon = trimesh.Trimesh(vertices=v_orig, faces=f, process=False)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    recon.export(out_path)

    sqr_d, _, _ = igl.point_mesh_squared_distance(v_orig, V, F)
    d = np.sqrt(np.maximum(sqr_d, 0.0))
    stats = (float(d.mean()), float(np.quantile(d, 0.95)), float(d.max()))
    print(f"{tag}recon V={v.shape} F={f.shape} -> {out_path}")
    print(f"{tag}recon->input surface  mean={stats[0]:.6g} p95={stats[1]:.6g} max={stats[2]:.6g}")
    return stats


def main():
    ap = argparse.ArgumentParser(description="GT UDF -> DualMeshUDF reconstruction test")
    ap.add_argument("--mesh", default=None, help="single input mesh (whole-mesh mode)")
    ap.add_argument("--out", default="udf_gt_recon.ply", help="output for --mesh mode")
    ap.add_argument("--parts_dir", default=None,
                    help="directory of per-part meshes (per-curve mode)")
    ap.add_argument("--pattern", default="*.ply", help="glob for --parts_dir")
    ap.add_argument("--out_dir", default="udf_gt_recon_parts",
                    help="output directory for per-curve mode")
    ap.add_argument("--max_depth", type=int, default=7, help="octree depth (7 ~= 128^3)")
    ap.add_argument("--batch_size", type=int, default=150000)
    ap.add_argument("--pad", type=float, default=0.9, help="fit each mesh into [-pad,pad]^3")
    ap.add_argument("--no_normalize", action="store_true",
                    help="assume mesh already lives in [-1,1]^3")
    args = ap.parse_args()

    if not args.parts_dir and not args.mesh:
        raise SystemExit("Provide either --parts_dir (per-curve) or --mesh (whole).")

    if args.parts_dir:
        part_paths = sorted(glob.glob(os.path.join(args.parts_dir, args.pattern)))
        if not part_paths:
            raise SystemExit(f"No meshes matched {args.pattern} in {args.parts_dir}")
        print(f"[per-curve] {len(part_paths)} part meshes from {args.parts_dir}\n")
        summary = []
        for p in part_paths:
            stem = Path(p).stem
            out_path = os.path.join(args.out_dir, f"recon_{stem}.ply")
            stats = reconstruct_one(
                p, out_path, max_depth=args.max_depth, batch_size=args.batch_size,
                pad=args.pad, no_normalize=args.no_normalize, label=stem)
            summary.append((stem, stats))
            print()
        print("==== per-curve summary (recon->input surface distance) ====")
        for stem, stats in summary:
            if stats is None:
                print(f"  {stem:40s}  FAILED/empty")
            else:
                print(f"  {stem:40s}  mean={stats[0]:.6g}  p95={stats[1]:.6g}  max={stats[2]:.6g}")
    else:
        reconstruct_one(
            args.mesh, args.out, max_depth=args.max_depth, batch_size=args.batch_size,
            pad=args.pad, no_normalize=args.no_normalize)


if __name__ == "__main__":
    main()
