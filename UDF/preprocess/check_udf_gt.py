#!/usr/bin/env python3
"""
Sanity-check generated UDF training samples before training.

Expected input: the pickle produced by process_data_3dvec_udf_keep_cylinder.py,
usually something like:
    <item>/all_data/udf_samples.pkl

It checks:
  - UDF keys exist
  - full/base UDF are non-negative
  - on-surface full samples have full UDF near zero
  - on-surface base samples have base UDF near zero
  - perturbed and space sample distributions look sane
  - optional PLY exports colored by UDF value
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np


def qstats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {k: float("nan") for k in ["min", "p01", "p05", "p50", "p95", "p99", "max", "mean"]}
    return {
        "min": float(np.min(x)),
        "p01": float(np.quantile(x, 0.01)),
        "p05": float(np.quantile(x, 0.05)),
        "p50": float(np.quantile(x, 0.50)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
    }


def fmt_stats(stats: dict[str, float]) -> str:
    order = ["min", "p01", "p05", "p50", "p95", "p99", "max", "mean"]
    return " ".join(f"{k}={stats[k]:.6g}" for k in order)


def write_point_ply(path: Path, points: np.ndarray, values: np.ndarray, vmin: float = 0.0, vmax: float | None = None):
    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = np.isfinite(points).all(axis=1) & np.isfinite(values)
    points = points[valid]
    values = values[valid]

    if vmax is None:
        vmax = float(np.quantile(values, 0.99)) if len(values) else 1.0
    vmax = max(float(vmax), float(vmin) + 1e-12)

    t = np.clip((values - float(vmin)) / (vmax - float(vmin)), 0.0, 1.0)
    # Simple blue -> green -> red map.
    r = np.where(t < 0.5, 0.0, 2.0 * (t - 0.5))
    g = np.where(t < 0.5, 2.0 * t, 2.0 * (1.0 - t))
    b = np.where(t < 0.5, 1.0 - 2.0 * t, 0.0)
    rgb = np.clip(np.stack([r, g, b], axis=1) * 255.0, 0, 255).astype(np.uint8)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, rgb):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def maybe_subsample(n: int, max_points: int, seed: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_points, replace=False))


def check_group(name: str, data: dict, eps: float):
    print(f"\n=== {name} ===")
    print("keys:", sorted(data.keys()))

    n = len(data.get("samples", []))
    print("count:", n)
    for arr_name in ["samples", "samples_local"]:
        if arr_name in data:
            print(f"{arr_name}.shape:", np.asarray(data[arr_name]).shape)

    for field in ["udf", "udf_base", "udf_res", "sdf", "sdf_base", "sdf_res"]:
        if field not in data:
            continue
        x = np.asarray(data[field], dtype=np.float64).reshape(-1)
        print(f"{field}: {fmt_stats(qstats(x))}")
        if field in ["udf", "udf_base"]:
            neg = int(np.sum(x < -eps))
            print(f"  negative(<-{eps:g}) count: {neg}")

    if name == "on_surface" and "sample_origin" in data:
        origin = np.asarray(data["sample_origin"], dtype=np.int64)
        if "udf" in data:
            m = origin == 0
            if np.any(m):
                x = np.asarray(data["udf"])[m]
                print("origin=0 full-surface, udf should be near 0:", fmt_stats(qstats(x)))
                print(f"  abs>{eps:g}:", int(np.sum(np.abs(x) > eps)), "/", int(np.sum(m)))
        if "udf_base" in data:
            m = origin == 1
            if np.any(m):
                x = np.asarray(data["udf_base"])[m]
                print("origin=1 base-surface, udf_base should be near 0:", fmt_stats(qstats(x)))
                print(f"  abs>{eps:g}:", int(np.sum(np.abs(x) > eps)), "/", int(np.sum(m)))

    if name == "pert_surface" and "perturb_sigma" in data and "udf" in data:
        sigma = np.asarray(data["perturb_sigma"], dtype=np.float64).reshape(-1)
        udf = np.asarray(data["udf"], dtype=np.float64).reshape(-1)
        if len(sigma) == len(udf) and len(udf) > 10:
            ratio = udf / (sigma + 1e-12)
            print("udf / perturb_sigma:", fmt_stats(qstats(ratio)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl", type=Path, help="Path to all_data/udf_samples.pkl")
    ap.add_argument("--eps", type=float, default=1e-5, help="near-zero tolerance")
    ap.add_argument("--export_ply", action="store_true")
    ap.add_argument("--out_dir", type=Path, default=Path("udf_gt_debug"))
    ap.add_argument("--max_points", type=int, default=200_000)
    ap.add_argument("--color_vmax", type=float, default=None, help="fixed color max; default p99 per export")
    args = ap.parse_args()

    with args.pkl.open("rb") as f:
        all_data = pickle.load(f)

    print("loaded:", args.pkl)
    print("top-level keys:", sorted(all_data.keys()))

    for group in ["on_surface", "pert_surface", "space"]:
        if group not in all_data:
            print(f"\nMISSING group: {group}")
            continue
        check_group(group, all_data[group], eps=args.eps)

    if args.export_ply:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for group in ["on_surface", "pert_surface", "space"]:
            if group not in all_data:
                continue
            data = all_data[group]
            if "samples" not in data:
                continue
            pts = np.asarray(data["samples"], dtype=np.float64)
            idx = maybe_subsample(len(pts), args.max_points, seed=17)
            for field in ["udf", "udf_base", "udf_res"]:
                if field not in data:
                    continue
                vals = np.asarray(data[field], dtype=np.float64).reshape(-1)
                out = args.out_dir / f"{group}_{field}.ply"
                write_point_ply(out, pts[idx], vals[idx], vmax=args.color_vmax)
                print("wrote", out)


if __name__ == "__main__":
    main()
