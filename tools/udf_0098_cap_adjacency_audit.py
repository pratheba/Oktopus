#!/usr/bin/env python3
"""0098: Read-only audit of edges leaving strong UDF faces.

Explains why 0097's surface-distance dilation adds no support. Counts edges
from high-scoring faces into (a) valid weak-score faces, (b) invalid faces,
(c) faces outside the diagnostic arclength window, and (d) faces excluded by
normal/arclength or physical-distance limits.

Uses original source meshes and saved raw face evidence; never runs UDF,
changes masks, or exports replacement geometry.
Requires tools/udf_0097_physical_support_probe.py from the 0097 patch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from udf_0097_physical_support_probe import read_case


def audit_window(case: dict, low: float, high: float, *, radius: float,
                 max_arc_step: float = .02, min_normal_dot: float = .5) -> dict:
    mesh = case['mesh']
    arc = np.asarray(case['arc'], dtype=np.float64)
    score = np.asarray(case['scores'], dtype=np.float64)
    valid = np.asarray(case['valid'], dtype=bool)
    strong = valid & (score >= case['seed'])
    grow = valid & (score >= case['grow'])
    region = (arc >= low) & (arc < high)
    anchor = strong & region
    n = len(arc)
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    a, b = adjacency[:, 0], adjacency[:, 1]
    ia = anchor[a]
    ib = anchor[b]
    # One directed row for every edge leaving a strong in-window face.
    source = np.concatenate([a[ia], b[ib]])
    neighbor = np.concatenate([b[ia], a[ib]])
    within = region[neighbor]
    dest_valid = valid[neighbor]
    dest_grow = grow[neighbor]
    dest_strong = strong[neighbor]
    weak = dest_valid & ~dest_grow
    coords = np.asarray(mesh.triangles_center)
    normals = np.asarray(mesh.face_normals)
    step = np.linalg.norm(coords[source] - coords[neighbor], axis=1)
    arcstep = np.abs(arc[source] - arc[neighbor])
    dot = np.einsum('ij,ij->i', normals[source], normals[neighbor])
    arc_ok = arcstep <= max_arc_step
    normal_ok = dot >= min_normal_dot
    near = step <= radius
    permissible = dest_valid & within & arc_ok & normal_ok
    weak_edges = weak & within
    eligible_weak = weak_edges & arc_ok & normal_ok
    direct_radius_weak = eligible_weak & near

    # Use per-anchor degrees to distinguish disconnected topology from
    # continuous strong-score patches and missing score support.
    degree = np.bincount(source, minlength=n)[anchor]
    possible_degree = np.bincount(source[permissible], minlength=n)[anchor]
    weak_degree = np.bincount(source[direct_radius_weak], minlength=n)[anchor]
    neighbors_crossing_invalid = ~dest_valid
    neighbors_outside_window = dest_valid & ~within
    neighbors_past_cap = neighbors_outside_window & (arc[neighbor] >= high)

    def count(mask) -> int:
        return int(np.count_nonzero(mask))

    def quantiles(x):
        return [round(float(v), 7) for v in np.quantile(x, [0, .25, .5, .75, .95, 1])] if len(x) else []

    return {
        'arc_window': [low, high],
        'anchor_strong_faces': count(anchor),
        'in_window_valid_faces': count(region & valid),
        'in_window_grow_faces': count(region & grow),
        'in_window_below_grow_faces': count(region & valid & ~grow),
        'total_mesh_adjacency_edges_leaving_strong': len(source),
        'strong_faces_with_zero_mesh_edge_neighbors': count(degree == 0),
        'strong_faces_with_zero_eligible_valid_neighbors': count(possible_degree == 0),
        'strong_faces_with_weak_valid_neighbors_within_radius': count(weak_degree > 0),
        'strong_faces_with_no_weak_valid_neighbors_within_radius': count(weak_degree == 0),
        'edges_to_other_strong_faces': count(dest_strong),
        'edges_to_grow_but_not_strong': count(dest_grow & ~dest_strong),
        'edges_to_below_grow_valid_faces': count(weak),
        'edges_to_invalid_or_missing_raw_evidence': count(neighbors_crossing_invalid),
        'edges_to_valid_faces_outside_window': count(neighbors_outside_window),
        'edges_to_valid_faces_beyond_high_boundary': count(neighbors_past_cap),
        'below_grow_edges_within_window': count(weak_edges),
        'below_grow_edges_after_arc_guard': count(weak_edges & arc_ok),
        'below_grow_edges_after_normal_guard': count(weak_edges & arc_ok & normal_ok),
        'below_grow_edges_within_radius': count(direct_radius_weak),
        'all_edge_steps_quantiles_world': quantiles(step),
        'below_grow_edge_steps_quantiles_world': quantiles(step[eligible_weak]),
        'strong_mesh_degree_quantiles': quantiles(degree),
        'settings': {'radius_world': radius, 'max_arc_step': max_arc_step,
                     'min_normal_dot': min_normal_dot},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--old', type=Path, required=True)
    ap.add_argument('--new', type=Path, required=True)
    ap.add_argument('--shape', default='oktopus_9_v1')
    ap.add_argument('--radius', type=float, default=.006)
    ap.add_argument('--max-arc-step', type=float, default=.02)
    ap.add_argument('--min-normal-dot', type=float, default=.5)
    ap.add_argument('--out', type=Path, default=Path('/tmp/puffer_0098_cap_adjacency.json'))
    a = ap.parse_args()
    if not (np.isfinite(a.radius) and a.radius > 0 and
            np.isfinite(a.max_arc_step) and a.max_arc_step > 0 and
            -1 <= a.min_normal_dot <= 1):
        ap.error('Invalid edge-filter settings')
    output = {}
    for key, path in [('old', a.old), ('new', a.new)]:
        case = read_case(path, a.shape)
        d = {}
        for region, low, high in [('cap', .885, .900), ('control', .400, .450)]:
            d[region] = audit_window(case, low, high, radius=a.radius,
                       max_arc_step=a.max_arc_step, min_normal_dot=a.min_normal_dot)
        output[key] = d
        print(f'\n{key.upper()}: {path}')
        for region in ['cap', 'control']:
            r = d[region]
            print(f" {region}: strong={r['anchor_strong_faces']:,} "
                  f"strong_zero_mesh_degree={r['strong_faces_with_zero_mesh_edge_neighbors']:,} "
                  f"strong_zero_eligible_neighbors={r['strong_faces_with_zero_eligible_valid_neighbors']:,}")
            print(f"   adjacent valid below-grow edges={r['edges_to_below_grow_valid_faces']:,} "
                  f"within-window={r['below_grow_edges_within_window']:,} "
                  f"after-arc={r['below_grow_edges_after_arc_guard']:,} "
                  f"after-normal={r['below_grow_edges_after_normal_guard']:,} "
                  f"within-radius={r['below_grow_edges_within_radius']:,}")
            print(f"   strong faces with weak neighbor <=radius={r['strong_faces_with_weak_valid_neighbors_within_radius']:,} "
                  f"edges to invalid={r['edges_to_invalid_or_missing_raw_evidence']:,} "
                  f"edges to valid outside-window={r['edges_to_valid_faces_outside_window']:,}")
            print(f"   eligible weak edge distance quantiles={r['below_grow_edge_steps_quantiles_world']}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(output, indent=2))
    print(f'\nSaved {a.out}; read-only, no masks changed.')


if __name__ == '__main__':
    main()
