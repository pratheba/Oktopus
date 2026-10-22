#!/usr/bin/env python3
"""Offline 0092 fused-SDF UDF evidence projection (no model re-run or stitching).

Run from repo root using the already successful 0089 SDF and 0091 UDF runs:
  python tools/cut_fused_sdf_0092.py \
    --sdf-shape inference/4M/puffer_oct22_fused_sdf_0089_r128/oktopus_9_v1 \
    --udf-shape inference/4M/puffer_oct22_udf_0091_r128/_udf_inference_0084/oktopus_9_v1 \
    --out inference/4M/puffer_oct22_udf_0092_offline_r128/oktopus_9_v1
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'app'))
from udf_cut_fused_exterior_0092 import generate_fused_cut


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdf-shape', type=Path, required=True)
    p.add_argument('--udf-shape', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--distance-edge-factor', type=float, default=1.75)
    p.add_argument('--normal-dot', type=float, default=.55)
    p.add_argument('--no-zero-seed-caps', action='store_true')
    args = p.parse_args()
    if args.out.exists():
        p.error(f'Output already exists: {args.out}; choose a new destination')
    report = generate_fused_cut(sdf_shape=args.sdf_shape, udf_shape=args.udf_shape,
          out=args.out, distance_edge_factor=args.distance_edge_factor,
          normal_dot=args.normal_dot, allow_zero_seed_caps=not args.no_zero_seed_caps)
    print(json.dumps({k: report[k] for k in ('status','faces_removed','faces_after','output_ply')}, indent=2))


if __name__ == '__main__':
    main()
