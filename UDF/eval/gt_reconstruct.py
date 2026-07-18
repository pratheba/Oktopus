#!/usr/bin/env python3
"""
Ground-truth UDF -> mesh reconstruction test (DualMeshUDF).

Purpose
-------
Validate the surface-extraction path (DualMeshUDF) INDEPENDENTLY of any trained
network: build an *analytic* unsigned-distance field directly from a known
input mesh, run DualMeshUDF's ``extract_mesh`` on it, and compare the
reconstruction back to the input.  If this looks good, the extraction pipeline
is sound and any later problems are the network's, not the extractor's.

The UDF and its gradient come from libigl's exact point-to-mesh distance:
    d(p)      = || p - closest_point_on_mesh(p) ||
    grad d(p) = (p - closest_point) / d          (unit, points away from surface)

DualMeshUDF extracts inside the cube [-1, 1]^3, so the input mesh is normalized
into that cube first (and the result is mapped back for comparison/report).

Usage
-----
    python UDF/eval/gt_reconstruct.py --mesh /path/to/mesh.ply \
        --out   /path/to/recon.ply \
        --max_depth 7                 # 7 -> ~128^3

Requires: DualMeshUDF (pip install the vendored copy), libigl, trimesh, numpy.
"""

from __future__ import annotations

import argparse
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


def main():
    ap = argparse.ArgumentParser(description="GT UDF -> DualMeshUDF reconstruction test")
    ap.add_argument("--mesh", required=True, help="input mesh (open or closed)")
    ap.add_argument("--out", default="udf_gt_recon.ply", help="output reconstructed mesh")
    ap.add_argument("--max_depth", type=int, default=7, help="octree depth (7 ~= 128^3)")
    ap.add_argument("--batch_size", type=int, default=150000)
    ap.add_argument("--pad", type=float, default=0.9, help="fit mesh into [-pad,pad]^3")
    ap.add_argument("--no_normalize", action="store_true",
                    help="assume mesh already lives in [-1,1]^3")
    args = ap.parse_args()

    mesh = load_mesh(args.mesh)
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)
    print(f"[input] V={V.shape} F={F.shape} watertight={mesh.is_watertight}")

    if args.no_normalize:
        Vn, center, scale = V, np.zeros(3), 1.0
    else:
        Vn, center, scale = normalize_to_cube(V, pad=args.pad)
        print(f"[normalize] center={center} scale={scale:.6g} -> [-{args.pad},{args.pad}]^3")

    udf_func, udf_grad_func = make_udf_funcs(Vn, F)

    print(f"[extract] running DualMeshUDF (max_depth={args.max_depth}) ...")
    v, f = extract_mesh(udf_func, udf_grad_func,
                        batch_size=args.batch_size, max_depth=args.max_depth)
    print(f"[recon] V={np.asarray(v).shape} F={np.asarray(f).shape}")

    v = np.asarray(v, dtype=np.float64)
    f = np.asarray(f, dtype=np.int64)

    # map reconstruction back to the ORIGINAL mesh frame for fair comparison
    if not args.no_normalize:
        v_orig = v / scale + center
    else:
        v_orig = v

    recon = trimesh.Trimesh(vertices=v_orig, faces=f, process=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    recon.export(args.out)
    print(f"[saved] {args.out}")

    # simple one-sided chamfer (recon vertices -> input surface) in original units
    try:
        sqr_d, _, _ = igl.point_mesh_squared_distance(v_orig, V, F)
        d = np.sqrt(np.maximum(sqr_d, 0.0))
        print(f"[recon->input surface] mean={d.mean():.6g} p95={np.quantile(d,0.95):.6g} "
              f"max={d.max():.6g}")
    except Exception as exc:
        print("[chamfer] skipped:", exc)


if __name__ == "__main__":
    main()
