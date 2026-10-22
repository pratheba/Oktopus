#!/usr/bin/env python3
"""Explain 0079 body-component accept/reject decisions from saved 0074/0079/0081 files.

Read only. Never evaluates a model, modifies an SDF mesh, or writes masks.
Requires source riggable_parts/body.ply and saved UDF evidence and reports.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import sys
import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def find_base(run: Path, shape: str):
    candidates = [run / shape, run / '_udf_inference_0084' / shape, run]
    for base in candidates:
        if (base / 'udf_cut_evidence_0074' / 'body' / 'face_evidence.npz').is_file() and (base / 'udf_cut_curve_local_0081' / 'body' / 'locality.npz').is_file():
            return base
    raise FileNotFoundError(f'No saved 0074/0081 body evidence found under {run}')


def reject_reason(*, faces, seeds, area, boundary, min_faces=8, min_seed_faces=2,
                  min_seed_fraction=.10, min_area=0., preserve_boundaries=True):
    # Must match the test order in AgentUDFCut._evaluate_cut_candidate_fast.
    if faces < min_faces:
        return 'too_few_faces'
    if seeds < min_seed_faces:
        return 'too_few_seed_faces'
    if seeds / max(1, faces) < min_seed_fraction:
        return 'seed_fraction_too_low'
    if area < min_area:
        return 'area_too_small'
    if preserve_boundaries and boundary:
        return 'touches_existing_boundary'
    return 'accepted'


def existing_boundary_faces(faces):
    """Exactly the edge-incidence rule used by the current cutter."""
    f = np.asarray(faces, dtype=np.int64)
    result = np.zeros(len(f), dtype=bool)
    fid = np.tile(np.arange(len(f), dtype=np.int64), 3)
    edges = np.concatenate((f[:, [0,1]], f[:,[1,2]], f[:,[2,0]]), axis=0)
    edges = np.sort(edges, axis=1)
    _, inv, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    result[fid[counts[inv] == 1]] = True
    return result


def component_decisions(mesh, scores, valid, seed, grow, curve_s,
                        min_faces=8, min_seed_faces=2, min_seed_fraction=.10,
                        min_area=0., preserve_boundaries=True):
    """Compute exactly the per-owner grow connectivity and decision gates for audit."""
    n = len(mesh.faces)
    if not all(len(x) == n for x in (scores, valid, curve_s)):
        raise ValueError('Input arrays do not align to mesh faces')
    growmask = valid & (scores >= grow)
    seedmask = valid & (scores >= seed)
    act = np.flatnonzero(growmask)
    if not len(act):
        return [], np.zeros(n, dtype=bool), {'active_grow': 0, 'active_seed': int(seedmask.sum())}
    old_to_compact = np.full(n, -1, dtype=np.int64)
    old_to_compact[act] = np.arange(len(act))
    adj = np.asarray(mesh.face_adjacency, dtype=np.int64)
    good = growmask[adj[:,0]] & growmask[adj[:,1]]
    links = old_to_compact[adj[good]]
    row = np.r_[links[:,0], links[:,1]]
    col = np.r_[links[:,1], links[:,0]]
    graph = coo_matrix((np.ones(len(row), dtype=np.uint8), (row, col)),
                       shape=(len(act), len(act))).tocsr()
    num, lab = connected_components(graph, directed=False)
    boundary = existing_boundary_faces(mesh.faces)
    counts = np.bincount(lab, minlength=num)
    s_count = np.bincount(lab, weights=seedmask[act].astype(np.uint8), minlength=num).astype(np.int64)
    area = np.bincount(lab, weights=np.asarray(mesh.area_faces)[act], minlength=num)
    bcnt = np.bincount(lab, weights=boundary[act].astype(np.uint8), minlength=num).astype(np.int64)
    arc = ((curve_s[act] >= .89) & (curve_s[act] < .90))
    arc_seed = arc & seedmask[act]
    arc_grow = np.bincount(lab, weights=arc.astype(np.uint8), minlength=num).astype(np.int64)
    arc_scnt = np.bincount(lab, weights=arc_seed.astype(np.uint8), minlength=num).astype(np.int64)
    decide = []
    predicted = np.zeros(n, dtype=bool)
    for i in range(num):
        reason = reject_reason(faces=int(counts[i]), seeds=int(s_count[i]),
                              area=float(area[i]), boundary=bool(bcnt[i]),
                              min_faces=min_faces, min_seed_faces=min_seed_faces,
                              min_seed_fraction=min_seed_fraction, min_area=min_area,
                              preserve_boundaries=preserve_boundaries)
        if reason == 'accepted':
            predicted[act[lab == i]] = True
        if arc_grow[i] or arc_scnt[i]:
            decide.append({'component_id': int(i), 'grow_faces': int(counts[i]),
                           'seed_faces': int(s_count[i]),
                           'seed_fraction': float(s_count[i]/max(1, counts[i])),
                           'area_world2': float(area[i]),
                           'boundary_faces': int(bcnt[i]),
                           'cap_grow_faces': int(arc_grow[i]),
                           'cap_seed_faces': int(arc_scnt[i]),
                           'reason': reason})
    decide.sort(key=lambda x:(-x['cap_seed_faces'], -x['cap_grow_faces']))
    return decide, predicted, {'active_grow': int(growmask.sum()),
                                'active_seed': int(seedmask.sum()),
                                'components_total': int(num),
                                'boundary_faces_total': int(boundary.sum())}


def run_one(run: Path, shape: str, override_mesh: Path | None, args):
    base = find_base(run, shape)
    report_path = base / 'riggable_after_udf_cut_report.json'
    if not report_path.is_file():
        raise FileNotFoundError(f'Missing per-part saved report: {report_path}; use saved complete run.')
    report = json.loads(report_path.read_text())
    part_report = report['parts']['body']
    source = override_mesh or Path(part_report['input'])
    if not source.is_file():
        raise FileNotFoundError(f'Saved input body mesh unavailable: {source}. Specify --{args.label}-mesh PATH')
    mesh = trimesh.load(source, force='mesh', process=False)
    with np.load(base / 'udf_cut_curve_local_0081' / 'body' / 'locality.npz') as z:
        s = np.asarray(z['face_curve_arclength'], dtype=np.float64)
        stored_seed = np.asarray(z['seed_faces'], dtype=bool)
        centroids = np.asarray(z['face_centroids_xyz'], dtype=np.float64)
    if len(mesh.faces) != len(s) or len(centroids) != len(s):
        raise ValueError(f'Face count not aligned with saved locality: {source}')
    centroid_error = np.max(np.linalg.norm(np.asarray(mesh.triangles_center) - centroids, axis=1))
    if centroid_error > 5e-6:
        raise ValueError(f'Source mesh does not match saved evidence (max centroid error={centroid_error})')
    with np.load(base / 'udf_cut_evidence_0074' / 'body' / 'face_evidence.npz') as z:
        prefix = 'item_00'  # Body owns one adaptation; refuse mixed-owner cases.
        names = [k for k in z.files if k.endswith('_faces') and k.startswith('item_')]
        if names != [prefix+'_faces']:
            raise ValueError(f'Expected one body UDF owner item_00; found: {names}')
        ids = np.asarray(z[prefix+'_faces'], dtype=np.int64)
        values = np.asarray(z[prefix+'_raw_scores'], dtype=np.float64)
        effective = np.asarray(z[prefix+'_effective_valid'], dtype=bool)
    with np.load(base / 'udf_cut_reconcile_0079' / 'body' / 'votes.npz') as z:
        existing_votes = np.asarray(z['remove_votes'], dtype=np.int64)
        agreed = np.asarray(z['class_agreed_remove'], dtype=bool)
    if len(existing_votes) != len(mesh.faces):
        raise ValueError('Vote arrays not aligned with mesh')
    owner = part_report['direct_udf']['owners']
    if len(owner) != 1 or owner[0]['item_index'] != 0:
        raise ValueError('Expected one body owner in report')
    seed, grow = float(owner[0]['seed_world']), float(owner[0]['grow_world'])
    face_scores = np.full(len(mesh.faces), np.nan)
    valid = np.zeros(len(mesh.faces), dtype=bool)
    face_scores[ids] = values
    valid[ids] = effective
    decisions, predicted, stats = component_decisions(mesh,face_scores,valid,seed,grow,s,
            min_faces=args.min_faces, min_seed_faces=args.min_seed_faces,
            min_seed_fraction=args.min_seed_fraction, min_area=args.min_area,
            preserve_boundaries=not args.allow_boundaries)
    mismatched = int(np.count_nonzero(predicted != (existing_votes > 0)))
    cap = (s>=.89)&(s<.90)
    ret = {'run': str(run), 'source': str(source), 'report': str(report_path),
           'face_count': int(len(mesh.faces)), 'seed_world': seed, 'grow_world': grow,
           'source_mesh_centroid_error': float(centroid_error),
           'stored_0081_cap_seeds': int(np.count_nonzero(stored_seed & cap)),
           'stored_0079_cap_agreed': int(np.count_nonzero(agreed & cap)),
           'computed_0079_cap_proposed': int(np.count_nonzero(predicted & cap)),
           '0079_replay_vote_disagreements_all_faces': mismatched,
           'stats':stats, 'cap_components':decisions}
    return ret


def pretty(label, r):
    print(f'\n{label}: {r["run"]}')
    print(f'  mesh faces={r["face_count"]:,}, thresholds seed/grow={r["seed_world"]:.6g}/{r["grow_world"]:.6g}')
    print(f'  0079 replay disagreements against saved remove_votes: {r["0079_replay_vote_disagreements_all_faces"]:,}')
    print(f'  s=[.89,.90) stored 0079 agreed={r["stored_0079_cap_agreed"]:,}, computed proposals={r["computed_0079_cap_proposed"]:,}, 0081 seeds={r["stored_0081_cap_seeds"]:,}')
    rows = r['cap_components']
    totals = {}
    for row in rows:
        why = row['reason']
        stats = totals.setdefault(why,{'components':0,'cap_seed_faces':0,'cap_grow_faces':0})
        stats['components']+=1
        stats['cap_seed_faces']+=row['cap_seed_faces']
        stats['cap_grow_faces']+=row['cap_grow_faces']
    print('  cap-region components by decision:',totals)
    print('  leading components (ranked by high-arc seed then grow):')
    for row in rows[:12]:
        print(f'    id={row["component_id"]:<5} cap_seed/grow={row["cap_seed_faces"]}/{row["cap_grow_faces"]} '
              f'whole_seed/grow={row["seed_faces"]}/{row["grow_faces"]} '
              f'fraction={row["seed_fraction"]:.3f} boundary_faces={row["boundary_faces"]} reason={row["reason"]}')
    if r['0079_replay_vote_disagreements_all_faces']:
        print('  WARNING: replay disagrees with saved votes; do not infer reason until mesh/config validated')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old',type=Path,required=True)
    p.add_argument('--new',type=Path,required=True)
    p.add_argument('--shape',default='oktopus_9_v1')
    p.add_argument('--old-mesh',type=Path)
    p.add_argument('--new-mesh',type=Path)
    p.add_argument('--min-faces',type=int,default=8)
    p.add_argument('--min-seed-faces',type=int,default=2)
    p.add_argument('--min-seed-fraction',type=float,default=.10)
    p.add_argument('--min-area',type=float,default=0.)
    p.add_argument('--allow-boundaries',action='store_true',help='Analysis only: alternate gate; do NOT use to change cutting')
    p.add_argument('--json',type=Path)
    a=p.parse_args()
    a.label='old'
    old=run_one(a.old,a.shape,a.old_mesh,a)
    a.label='new'
    new=run_one(a.new,a.shape,a.new_mesh,a)
    print('0079 CONNECTED COMPONENT REJECTION AUDIT (READ ONLY)')
    print('Settings assumed from 0081/0092 YAML: min_faces=8, min_seed_faces=2, min_seed_fraction=.10, preserve_existing_boundaries=True.')
    pretty('WORKING',old)
    pretty('FAILING',new)
    if a.json:
        a.json.write_text(json.dumps({'old':old,'new':new},indent=2))
        print('\nSaved',a.json)

if __name__=='__main__':
    main()
