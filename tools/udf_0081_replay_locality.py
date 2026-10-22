#!/usr/bin/env python3
"""Offline parameter sweep for 0081 locality on saved UDF evidence (NO model call).

Use the original 0067 mesh and 0081 locality.npz from the SAME UDF run.
Outputs alternate eight-part proposals, never modifies original SDF meshes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'app'))
from udf_cut_curve_local_0081 import LocalExpansionSettings, expand_face_mask
from udf_cut_reconcile_0079 import keep_largest_edge_component

PARTS = ('body', 'tentacle_t1', 'tentacle_t2', 'tentacle_t3',
         'tentacle_t4', 'tentacle_t5', 'tentacle_t6', 'tentacle_t8')


def preview_one(mesh_path: Path, evidence_path: Path, output: Path, settings):
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f'Invalid triangular mesh: {mesh_path}')
    with np.load(evidence_path) as z:
        s = np.asarray(z['face_curve_arclength'], dtype=np.float64)
        total = float(z['curve_length_world'])
        seed = np.asarray(z['seed_faces'], dtype=bool)
        candidate = np.asarray(z['candidate_faces'], dtype=bool)
        centroids = np.asarray(z['face_centroids_xyz'], dtype=np.float64)
    actual = np.asarray(mesh.triangles_center, dtype=np.float64)
    if centroids.shape != actual.shape or len(seed) != len(mesh.faces):
        raise ValueError(f'0081 face count/centroid shape mismatch for {mesh_path}')
    err = float(np.max(np.linalg.norm(centroids - actual, axis=1)))
    if err > 5e-6:
        raise ValueError(f'0081 source mesh is not aligned with evidence: error={err}')
    cut, report = expand_face_mask(mesh, seed_mask=seed, candidate_mask=candidate,
        curve_s=s, curve_length_world=total, settings=settings)
    if np.all(cut):
        raise ValueError('0081 preview removes the entire mesh; refusing')
    kept = trimesh.Trimesh(vertices=np.asarray(mesh.vertices).copy(),
             faces=np.asarray(mesh.faces)[~cut].copy(), process=False)
    kept.remove_unreferenced_vertices()
    try:
        cleaned, debris, cleanup = keep_largest_edge_component(kept)
    except ValueError as e:
        cleaned, debris = kept, None
        cleanup = {'skipped':True,'reason':str(e)}
    output.mkdir(parents=True, exist_ok=True)
    cleaned.export(output / 'proposed_kept_cleaned.ply')
    if debris is not None and len(debris.faces):
        debris.export(output/'detached_fragments.ply')
    extra = candidate & ~cut
    if extra.any():
        blocked = trimesh.Trimesh(vertices=np.asarray(mesh.vertices).copy(),
             faces=np.asarray(mesh.faces)[extra].copy(), process=False)
        blocked.remove_unreferenced_vertices()
        blocked.export(output/'blocked_candidate_faces.ply')
    report['cleanup'] = cleanup
    report['source_mesh'] = str(mesh_path)
    report['centroid_alignment_error'] = err
    report['faces_after_cleanup'] = int(len(cleaned.faces))
    (output/'summary.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mesh-dir', type=Path, required=True,
                    help='0067/.../riggable_parts')
    ap.add_argument('--run-dir', type=Path, required=True,
                    help='0081/.../oktopus_9_v1 directory')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--curve-band', type=float, default=.11)
    ap.add_argument('--curve-step', type=float, default=.07)
    ap.add_argument('--surface-walk', type=float, default=.40)
    ap.add_argument('--min-seed-faces', type=int, default=8)
    args = ap.parse_args()
    settings = LocalExpansionSettings(max_curve_fraction=args.curve_band,
          max_neighbor_curve_jump=args.curve_step,
          max_surface_walk_fraction=args.surface_walk,
          min_seed_component_faces=args.min_seed_faces)
    reports = {}
    for part in PARTS:
        original = args.mesh_dir / (part+'.ply')
        saved = args.run_dir/'udf_cut_curve_local_0081'/part/'locality.npz'
        if not original.is_file() or not saved.is_file():
            raise FileNotFoundError(f'{part}: expected {original} and {saved}')
        report = preview_one(original,saved,args.out/part,settings)
        reports[part] = report
        print('[udf cut curve replay 0081]',f'part={part}',
              f'seeds={report["seed_faces"]}',f'added={report["added_faces"]}',
              f'blocked={report["blocked_candidates"]}',
              f'kept={report["faces_after_cleanup"]}',flush=True)
    (args.out/'all_parts_summary.json').write_text(json.dumps(reports,indent=2))


if __name__=='__main__':
    main()
