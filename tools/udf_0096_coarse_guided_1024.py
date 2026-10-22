#!/usr/bin/env python3
"""Offline cross-resolution UDF cut GUIDANCE experiment (not production 0081).

Transfer a previously visually validated coarse 0081 cut region to a finer
SDF part only where geometry AND curve arclength agree and fine 0081 already
marks a face as a candidate. Keep 0079/0081 evidence intact. The only staged
change is fine locality.npz:selected_faces (plus its blocked_faces complement).

The saved SDF meshes/UDF runs are never modified. Run 0092 separately on the
staged evidence after inspecting the proposal. No model inference is done.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial import cKDTree
import trimesh


def load_evidence(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def require_aligned(mesh, data, label):
    required = ('selected_faces', 'candidate_faces', 'seed_faces',
                'face_curve_arclength', 'face_centroids_xyz')
    for key in required:
        if key not in data or len(data[key]) != len(mesh.faces):
            raise ValueError(f'{label}: missing or mismatched {key}')
    center = np.asarray(mesh.triangles_center, dtype=np.float64)
    saved = np.asarray(data['face_centroids_xyz'], dtype=np.float64)
    if saved.shape != center.shape:
        raise ValueError(f'{label}: bad face-centroid shape')
    error = float(np.max(np.linalg.norm(center - saved, axis=1)))
    if error > 5e-6:
        raise ValueError(f'{label}: SDF mesh does not match saved UDF face order: {error:.6g}')
    return error


def transfer_guidance(coarse, fine, old, new, *, arc_min, arc_max,
                      normal_dot=.75, max_arc_delta=.015,
                      distance_edge_factor=1.75, neighbors=12, batch_size=16384):
    """Choose coarse nearest *actual triangle*, not only nearest centroid.

    Read-only. Returns proposed additional fine faces and per-face matched
    proximity evidence; only candidate, high-arc, orientation-compatible faces
    may be added. No coordinate-distance-only transfer at folded skeletons.
    """
    if not (0 <= arc_min < arc_max <= 1):
        raise ValueError('invalid arc interval')
    if not (0 <= normal_dot <= 1 and max_arc_delta > 0 and
            distance_edge_factor > 0 and neighbors > 0 and batch_size > 0):
        raise ValueError('invalid matching parameters')
    require_aligned(coarse, old, 'coarse')
    require_aligned(fine, new, 'fine')
    oldsel = np.asarray(old['selected_faces'], dtype=bool)
    newsel = np.asarray(new['selected_faces'], dtype=bool)
    candidate = np.asarray(new['candidate_faces'], dtype=bool)
    oldarc = np.asarray(old['face_curve_arclength'], dtype=float)
    newarc = np.asarray(new['face_curve_arclength'], dtype=float)
    if not np.all(oldsel <= np.asarray(old['candidate_faces'], dtype=bool)):
        raise ValueError('coarse cut lies outside coarse candidates')
    if not np.all(newsel <= candidate):
        raise ValueError('fine cut lies outside fine candidates')

    old_centers = np.asarray(coarse.triangles_center, dtype=float)
    old_normals = np.asarray(coarse.face_normals, dtype=float)
    new_normals = np.asarray(fine.face_normals, dtype=float)
    edges = np.asarray(coarse.edges_unique_length, dtype=float)
    edges = edges[np.isfinite(edges) & (edges > 1e-12)]
    if len(edges) == 0:
        raise ValueError('coarse mesh has no nonzero edges')
    median_edge = float(np.median(edges))
    max_dist = float(distance_edge_factor * median_edge)
    tree = cKDTree(old_centers)
    triangles = np.asarray(coarse.triangles, dtype=float)
    targets = np.asarray(fine.triangles_center, dtype=float)
    eligible = candidate & (~newsel) & (newarc >= arc_min) & (newarc < arc_max)
    matched = np.zeros(len(fine.faces), dtype=bool)
    suggested = np.zeros(len(fine.faces), dtype=bool)
    dist_best = np.full(len(fine.faces), np.inf, dtype=np.float32)
    old_face = np.full(len(fine.faces), -1, dtype=np.int64)
    k = min(int(neighbors), len(coarse.faces))
    for offset in range(0, len(fine.faces), batch_size):
        stop = min(len(fine.faces), offset + batch_size)
        face_ids = np.flatnonzero(eligible[offset:stop]) + offset
        if not len(face_ids):
            continue
        pts = targets[face_ids]
        neighbors_ix = np.asarray(tree.query(pts, k=k)[1], dtype=np.int64).reshape(len(face_ids), k)
        repeated_pts = np.repeat(pts, k, axis=0)
        nearest = trimesh.triangles.closest_point(triangles[neighbors_ix.reshape(-1)], repeated_pts)
        d = np.linalg.norm(nearest - repeated_pts, axis=1).reshape(-1, k)
        norm = np.einsum('ijk,ik->ij', old_normals[neighbors_ix], new_normals[face_ids])
        arc_err = np.abs(oldarc[neighbors_ix] - newarc[face_ids, None])
        # Only a geometrically and arclength-compatible match is eligible.
        good = ((d <= max_dist) & (norm >= normal_dot) &
                (arc_err <= max_arc_delta))
        choice = np.argmin(np.where(good, d, np.inf), axis=1)
        distances = d[np.arange(len(face_ids)), choice]
        ok = good[np.arange(len(face_ids)), choice]
        picked = neighbors_ix[np.arange(len(face_ids)), choice]
        valid_ids = face_ids[ok]
        matched[valid_ids] = True
        dist_best[valid_ids] = distances[ok].astype(np.float32)
        old_face[valid_ids] = picked[ok]
        suggested[valid_ids] = oldsel[picked[ok]]
    stats = {
        'coarse_faces': int(len(coarse.faces)), 'fine_faces': int(len(fine.faces)),
        'coarse_cut_faces': int(oldsel.sum()), 'fine_cut_faces_before': int(newsel.sum()),
        'arc_min': float(arc_min), 'arc_max': float(arc_max),
        'coarse_median_edge_world': median_edge, 'max_transfer_distance_world': max_dist,
        'normal_dot_min': float(normal_dot), 'max_curve_arclength_difference': float(max_arc_delta),
        'eligible_fine_candidate_faces': int(eligible.sum()),
        'geometry_arc_matched_eligible': int(matched.sum()),
        'matched_pct': float(100 * matched.sum()/max(1,eligible.sum())),
        'added_coarse_guided_faces': int(suggested.sum()),
        'total_fine_cut_faces_after': int(np.count_nonzero(newsel | suggested)),
        'high_arc_newly_removed': int(np.count_nonzero(suggested & (newarc >= .89) & (newarc < .95))),
    }
    return suggested, matched, dist_best, old_face, stats


def mesh_mask_export(mesh, selection, path):
    ids = np.asarray(selection, bool)
    if np.any(ids):
        part = trimesh.Trimesh(vertices=np.asarray(mesh.vertices).copy(),
                               faces=np.asarray(mesh.faces)[ids].copy(), process=False)
        part.remove_unreferenced_vertices()
        part.export(str(path))


def run(args):
    oldroot = Path(args.old_udf_shape)
    newroot = Path(args.new_udf_shape)
    oldnpz = oldroot/'udf_cut_curve_local_0081'/args.part/'locality.npz'
    newnpz = newroot/'udf_cut_curve_local_0081'/args.part/'locality.npz'
    oldmeshfile = Path(args.old_mesh)
    newmeshfile = Path(args.new_sdf_shape)/'riggable_parts'/f'{args.part}.ply'
    for path in (oldnpz, newnpz, oldmeshfile, newmeshfile):
        if not path.is_file():
            raise FileNotFoundError(path)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f'Output exists: {out} (never overwrite an experiment)')
    coarse = trimesh.load(oldmeshfile, force='mesh', process=False)
    fine = trimesh.load(newmeshfile, force='mesh', process=False)
    old = load_evidence(oldnpz)
    new = load_evidence(newnpz)
    proposed, matched, distance, oldface, stats = transfer_guidance(
        coarse, fine, old, new,
        arc_min=args.arc_min, arc_max=args.arc_max,
        normal_dot=args.normal_dot, max_arc_delta=args.max_arc_delta,
        distance_edge_factor=args.distance_edge_factor,
        neighbors=args.neighbors, batch_size=args.batch_size,
    )
    # Preflight counts are exported even if the transfer cannot be trusted.
    stats.update({'old_udf_shape':str(oldroot),'new_udf_shape':str(newroot),
                  'old_sdf_mesh':str(oldmeshfile),'new_sdf_mesh':str(newmeshfile),
                  'experimental_only': True,
                  'method':'coarse 0081 selected-mask guidance intersect fine 0081 candidates; surface+normal+curve matching'})
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)
    mesh_mask_export(fine, proposed, out/'newly_guided_faces.ply')
    new_sel = np.asarray(new['selected_faces'], dtype=bool)
    mesh_mask_export(fine, new_sel | proposed, out/'all_guided_removed_faces.ply')
    mesh_mask_export(fine, ~(new_sel | proposed), out/'guided_kept_uncleaned.ply')
    np.savez_compressed(out/'transfer_audit.npz',
                        newly_guided=proposed, geometry_matched=matched,
                        source_coarse_face=oldface, source_distance_world=distance)
    if stats['matched_pct'] < args.min_coverage_pct or not stats['added_coarse_guided_faces']:
        print('NOT STAGED: insufficient matched coverage or zero additional cuts. Inspect report.', flush=True)
        return 2
    # Generate an isolated parallel copy of JUST the 0081 locality evidence.
    # 0092 consumes that folder and original SDF parts; no source changes.
    dst = out/'staged_udf_shape'/'udf_cut_curve_local_0081'
    src = newroot/'udf_cut_curve_local_0081'
    if not src.is_dir():
        raise FileNotFoundError(src)
    for part_dir in sorted(src.iterdir()):
        if not part_dir.is_dir():
            continue
        f = part_dir/'locality.npz'
        if not f.is_file():
            raise FileNotFoundError(f)
        target = dst/part_dir.name/'locality.npz'
        target.parent.mkdir(parents=True, exist_ok=True)
        if part_dir.name == args.part:
            arrays = dict(new)
            arrays['selected_faces'] = new_sel | proposed
            arrays['blocked_faces'] = np.asarray(new['candidate_faces'], bool) & ~arrays['selected_faces']
            np.savez_compressed(target, **arrays)
        else:
            shutil.copy2(f, target)
    (out/'staged_udf_shape'/'EXPERIMENTAL_GUIDANCE_README.txt').write_text(
        'EXPERIMENTAL cross-resolution mask: NOT learned fine-resolution 0079/0081 evidence. '
        'Use only for manually inspected 0092 offline trial; preserve original runs.\n')
    print('STAGED:', out/'staged_udf_shape', flush=True)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-udf-shape', required=True, help='512 run /oktopus_9_v1 with 0081 evidence')
    p.add_argument('--new-udf-shape', required=True, help='1024 run /_udf_inference_0084/oktopus_9_v1')
    p.add_argument('--old-mesh', required=True, help='exact original 512 SDF riggable_parts/body.ply')
    p.add_argument('--new-sdf-shape', required=True, help='1024 SDF run /oktopus_9_v1')
    p.add_argument('--out', required=True, help='NEW output directory; refuse overwrite')
    p.add_argument('--part', default='body')
    p.add_argument('--arc-min', type=float, default=.85)
    p.add_argument('--arc-max', type=float, default=.95)
    p.add_argument('--normal-dot', type=float, default=.75)
    p.add_argument('--max-arc-delta', type=float, default=.015)
    p.add_argument('--distance-edge-factor', type=float, default=1.75)
    p.add_argument('--min-coverage-pct', type=float, default=75.)
    p.add_argument('--neighbors', type=int, default=12)
    p.add_argument('--batch-size', type=int, default=16384)
    return run(p.parse_args())

if __name__ == '__main__':
    raise SystemExit(main())
