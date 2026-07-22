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


def _configurable_extract(udf_func, udf_grad_func, *, batch_size, max_depth,
                          reliable, sample_threshold, sampling_depth,
                          subdivide_threshold=None, projection_threshold=None):
    """Configurable DualMeshUDF octree loop (reliable / sample_threshold /
    sampling_depth exposed), matching AgentUDF._dualmeshudf_extract so GT and
    learned fields go through the SAME extractor. Logs projection stats."""
    import numpy as _np
    import igl as _igl
    from DualMeshUDF_core import Octree, triangulate_faces
    from DualMeshUDF.extract_mesh import query_udf, query_udf_and_grad
    if sample_threshold is None:
        sample_threshold = min(0.25 * reliable, 0.005)
    subdivide_threshold = reliable if subdivide_threshold is None else float(subdivide_threshold)
    projection_threshold = reliable if projection_threshold is None else float(projection_threshold)
    octree = Octree(max_depth=int(max_depth),
                    min_corner=_np.array([[-1.], [-1.], [-1.]]),
                    max_corner=_np.array([[1.], [1.], [1.]]),
                    sampling_depth=int(sampling_depth))
    cur = 0
    while cur <= int(max_depth):
        cen = octree.centroids_of_new_nodes().astype(_np.float32)
        cu, cg = query_udf_and_grad(udf_grad_func, cen, batch_size)
        octree.adaptive_subdivide(cu, cg, subdivide_threshold)
        cur += 1
    gi, gc = octree.get_samples_of_new_nodes()
    gu, gg = query_udf_and_grad(udf_grad_func, gc.astype(_np.float32), batch_size)
    octree.set_new_grid_data(gi, gu, gg)
    idx, proj = octree.get_projections_for_checking_validity()
    pu = _np.asarray(query_udf(udf_func, proj, batch_size)).reshape(-1)
    pv = pu < projection_threshold
    pct = _np.percentile(pu, [0, 1, 5, 25, 50, 95]).tolist() if pu.size else []
    print("[dmudf loop] reliable=", reliable, "subdivide=", subdivide_threshold,
          "projection=", projection_threshold, "sample_threshold=", sample_threshold,
          "sampling_depth=", int(sampling_depth), "n_proj=", int(pu.size),
          "n_valid=", int(pv.sum()),
          "valid_pct=", round(100.0 * float(pv.sum()) / max(pu.size, 1), 2),
          "pu_pct[0,1,5,25,50,95]=", [round(float(x), 6) for x in pct])
    octree.set_grid_validity(idx, pv)
    octree.batch_solve(float(sample_threshold), 1.0, 1.0, 0.15, 0.08)
    octree.generate_mesh()
    print("[dmudf loop] pre-triangulate mesh_v=", len(octree.mesh_v),
          "mesh_f=", len(octree.mesh_f))
    tri = triangulate_faces(octree.mesh_v, octree.mesh_f,
                            octree.v_type, octree.mesh_v_dir)
    try:
        v, _, _, f = _igl.remove_duplicate_vertices(
            _np.ascontiguousarray(octree.mesh_v, dtype=_np.float64),
            _np.ascontiguousarray(tri, dtype=_np.int64), 1e-7)
        v, f, _, _ = _igl.remove_unreferenced(
            _np.ascontiguousarray(v, dtype=_np.float64),
            _np.ascontiguousarray(f, dtype=_np.int64))
    except TypeError:
        v = _np.asarray(octree.mesh_v, dtype=_np.float64)
        f = _np.asarray(tri, dtype=_np.int64)
    return _np.asarray(v, dtype=_np.float64), _np.asarray(f, dtype=_np.int64)


def _quality_stats(v, f):
    import numpy as _np
    import collections as _col
    v = _np.asarray(v, dtype=_np.float64); f = _np.asarray(f, dtype=_np.int64)
    nv, nf = int(len(v)), int(len(f))
    if nf == 0:
        return {"V": nv, "F": 0, "boundary": 0, "nonmanifold": 0,
                "largest_face_pct": 0.0, "degenerate": 0}
    ec = _col.Counter()
    for tri in f:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            ec[(int(min(a, b)), int(max(a, b)))] += 1
    boundary = sum(1 for c in ec.values() if c == 1)
    nonman = sum(1 for c in ec.values() if c > 2)
    deg_idx = (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])
    e1 = v[f[:, 1]] - v[f[:, 0]]; e2 = v[f[:, 2]] - v[f[:, 0]]
    area = 0.5 * _np.linalg.norm(_np.cross(e1, e2), axis=1)
    degen = int((deg_idx | (area < 1e-12)).sum())
    par = list(range(nv))
    def _find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for tri in f:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2])):
            ra, rb = _find(int(a)), _find(int(b))
            if ra != rb:
                par[ra] = rb
    roots = _col.Counter(_find(int(t[0])) for t in f)
    largest = 100.0 * max(roots.values()) / nf if roots else 0.0
    return {"V": nv, "F": nf, "boundary": boundary, "nonmanifold": nonman,
            "largest_face_pct": round(largest, 2), "degenerate": degen}


def clean_mesh(mesh):
    """Merge near-duplicate vertices and drop degenerate/duplicate faces.
    Returns (faces_before, faces_after). Version-tolerant across trimesh."""
    before = len(mesh.faces)
    for step in (
        lambda: mesh.merge_vertices(),
        lambda: mesh.update_faces(mesh.nondegenerate_faces(height=1e-8)),
        lambda: mesh.update_faces(mesh.unique_faces()),
        lambda: mesh.remove_unreferenced_vertices(),
    ):
        try:
            step()
        except Exception:
            pass
    return before, len(mesh.faces)


def reconstruct_one(mesh_path, out_path, max_depth=7, batch_size=150000,
                    pad=0.9, no_normalize=False, label="", clean=False,
                    reliable=0.002, sample_threshold=None, sampling_depth=1,
                    subdivide_threshold=None, projection_threshold=None):
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

    v, f = _configurable_extract(
        udf_func, udf_grad_func, batch_size=batch_size, max_depth=max_depth,
        reliable=reliable, sample_threshold=sample_threshold,
        sampling_depth=sampling_depth, subdivide_threshold=subdivide_threshold,
        projection_threshold=projection_threshold)
    print(f"{tag}[dmudf quality]", _quality_stats(v, f))
    v = np.asarray(v, dtype=np.float64)
    f = np.asarray(f, dtype=np.int64)
    if len(v) == 0 or len(f) == 0:
        print(f"{tag}WARNING: extraction produced an empty mesh")
        return None

    v_orig = (v / scale + center) if not no_normalize else v
    recon = trimesh.Trimesh(vertices=v_orig, faces=f, process=False)
    if clean:
        nb, na = clean_mesh(recon)
        print(f"{tag}cleaned faces {nb} -> {na} ({nb - na} degenerate/duplicate removed)")
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
    ap.add_argument("--clean", action="store_true",
                    help="merge close vertices + drop degenerate/duplicate faces before export")
    ap.add_argument("--reliable", type=float, default=0.002,
                    help="DualMeshUDF reliability threshold (stock 0.002)")
    ap.add_argument("--sample_threshold", type=float, default=None,
                    help="batch-solve threshold (default min(0.25*reliable,0.005))")
    ap.add_argument("--sampling_depth", type=int, default=1,
                    help="per-cell sampling depth (1->27 pts/cell, 2->125)")
    ap.add_argument("--subdivide_threshold", type=float, default=None,
                    help="octree adaptive_subdivide threshold (default = reliable)")
    ap.add_argument("--projection_threshold", type=float, default=None,
                    help="grid-validity projection threshold (default = reliable)")
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
                pad=args.pad, no_normalize=args.no_normalize, label=stem, clean=args.clean,
                reliable=args.reliable, sample_threshold=args.sample_threshold,
                sampling_depth=args.sampling_depth,
                subdivide_threshold=args.subdivide_threshold,
                projection_threshold=args.projection_threshold)
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
            pad=args.pad, no_normalize=args.no_normalize, clean=args.clean,
            reliable=args.reliable, sample_threshold=args.sample_threshold,
            sampling_depth=args.sampling_depth,
            subdivide_threshold=args.subdivide_threshold,
            projection_threshold=args.projection_threshold)


if __name__ == "__main__":
    main()
