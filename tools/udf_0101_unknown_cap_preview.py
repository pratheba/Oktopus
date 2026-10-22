#!/usr/bin/env python3
"""Offline *preview* of a cap-like UNKNOWN region in a 0100 fused-UDF result.

Never equates all invalid UDF samples with absent geometry. Candidate unknown
components must be large, edge-connected, approximately planar, predominantly
outside valid UDF query support, and touch independently UDF-confirmed REMOVED
faces. This is an exploratory visualization, NOT a production classification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def cap_candidates(*, face_adjacency, centers, part, unknown, removed,
                   invalid, min_faces=128, min_fraction=0.30,
                   min_invalid_fraction=0.60, max_planarity=0.075):
    """Return conservative candidate face mask and per-component audit.

    Planarity = sqrt(smallest / largest centroid covariance eigenvalue).
    Neither face count nor PCA confirms that a region is an artificial cap;
    the outputs are for manual inspection only.
    """
    centers = np.asarray(centers, dtype=np.float64)
    n = centers.shape[0]
    part, unknown, removed, invalid = (np.asarray(a, bool).reshape(-1)
                                       for a in (part, unknown, removed, invalid))
    if any(len(a) != n for a in (part, unknown, removed, invalid)):
        raise ValueError('Masks do not match mesh face count')
    if np.any(unknown & removed):
        raise ValueError('Invalid overlapping unknown/remove masks')
    if not (0 <= min_fraction <= 1 and 0 <= min_invalid_fraction <= 1 and 0 <= max_planarity <= 1):
        raise ValueError('Invalid thresholds')
    edges = np.asarray(face_adjacency, dtype=np.int64).reshape(-1, 2)
    if edges.size and (edges.min() < 0 or edges.max() >= n):
        raise ValueError('Face adjacency contains invalid indices')
    mask = part & unknown
    ids = np.flatnonzero(mask)
    out = np.zeros(n, bool)
    if ids.size == 0:
        return out, []
    local = np.full(n, -1, np.int64)
    local[ids] = np.arange(len(ids), dtype=np.int64)
    active_edge = mask[edges[:, 0]] & mask[edges[:, 1]] if edges.size else np.zeros(0, bool)
    uv = local[edges[active_edge]] if edges.size else np.empty((0, 2), np.int64)
    graph = coo_matrix((np.ones(2 * len(uv), dtype=np.uint8),
                        (np.concatenate((uv[:, 0], uv[:, 1])),
                         np.concatenate((uv[:, 1], uv[:, 0])))),
                       shape=(len(ids), len(ids))).tocsr()
    count, labels = connected_components(graph, directed=False)
    # Count confirmed-removal contacts by connected unknown component.
    contact = np.zeros(count, dtype=np.int64)
    supported_contact = np.zeros(count, dtype=np.int64)
    if edges.size:
        for a, b in ((edges[:, 0], edges[:, 1]), (edges[:, 1], edges[:, 0])):
            border = mask[a] & part[b] & ~mask[b]
            touched = local[a[border]]
            nb = b[border]
            if len(touched):
                lab = labels[touched]
                contact += np.bincount(lab[removed[nb]], minlength=count)
                supported_contact += np.bincount(lab[~removed[nb]], minlength=count)
    reports = []
    # Avoid O(number_of_components * number_of_faces) scans. A noisy 1024
    # field can contain many tiny islands; inspect only the largest 24,
    # and always include any component that could pass both size gates.
    sizes = np.bincount(labels, minlength=count)
    order = np.argsort(labels, kind="stable")
    offsets = np.r_[0, np.cumsum(sizes)]
    ranked = np.argsort(-sizes, kind="stable")
    interesting = [i for rank, i in enumerate(ranked)
                   if rank < 24 or (sizes[i] >= int(min_faces) and
                                    sizes[i] >= min_fraction * len(ids))]
    for i in interesting:
        members = ids[order[offsets[i]:offsets[i + 1]]]
        fraction = float(len(members) / len(ids))
        inv_fraction = float(np.mean(invalid[members]))
        # Limit PCA to 20k deterministic samples without altering component membership.
        sample = members[np.linspace(0, len(members) - 1,
                                     num=min(20000, len(members)), dtype=int)]
        cloud = centers[sample]
        if len(cloud) >= 3:
            centered = cloud - cloud.mean(axis=0)
            eig = np.linalg.eigvalsh((centered.T @ centered) / len(cloud))
            planarity = float(np.sqrt(max(0, eig[0]) / max(eig[-1], 1e-18)))
        else:
            planarity = 1.0
        reasons = []
        if len(members) < int(min_faces): reasons.append('too_few_faces')
        if fraction < float(min_fraction): reasons.append('small_fraction_of_unknown')
        if inv_fraction < float(min_invalid_fraction): reasons.append('mostly_valid_or_ambiguous')
        if planarity > float(max_planarity): reasons.append('not_planar')
        if contact[i] == 0: reasons.append('no_confirmed_removed_neighbor')
        chosen = not reasons
        if chosen:
            out[members] = True
        reports.append({
            'component': int(i), 'faces': int(len(members)),
            'fraction_of_part_unknown': round(fraction, 5),
            'invalid_fraction': round(inv_fraction, 5),
            'planarity_ratio': round(planarity, 5),
            'edges_to_confirmed_removed': int(contact[i]),
            'edges_to_other_retained': int(supported_contact[i]),
            'preview_eligible': chosen, 'reasons': reasons,
        })
    return out, sorted(reports, key=lambda x: x['faces'], reverse=True)


def mesh_mask(mesh, mask):
    result = trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                             faces=np.asarray(mesh.faces)[mask], process=False)
    result.remove_unreferenced_vertices()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--direct', required=True,
                   help='Path to 0100 udf_direct_fused_0100 output directory')
    p.add_argument('--part', default='body')
    p.add_argument('--out', required=True, help='New empty output directory')
    p.add_argument('--min-faces', type=int, default=128)
    p.add_argument('--min-fraction', type=float, default=.30)
    p.add_argument('--min-invalid-fraction', type=float, default=.60)
    p.add_argument('--max-planarity', type=float, default=.075)
    args = p.parse_args()
    source = Path(args.direct).resolve()
    dest = Path(args.out).resolve()
    if dest.exists():
        p.error('Output exists; choose a new --out')
    report = json.loads((source / 'report.json').read_text())
    sdf_mesh = Path(report['source_fused_sdf']).resolve()
    manifest = json.loads((sdf_mesh.parent / 'root_leaves_ownership.json').read_text())
    names = list(manifest['part_names'])
    if args.part not in names:
        p.error(f'Part {args.part!r} missing from SDF manifest {names!r}')
    if args.part not in report.get('parts', {}):
        p.error(f'Part {args.part!r} was not queried in this 0100 run')
    mesh = trimesh.load(sdf_mesh, force='mesh', process=False)
    with np.load(source / 'consistency.npz') as evidence:
        arrays = {k: evidence[k].copy() for k in
                  ('assigned', 'valid', 'keep', 'remove', 'unknown', 'part_index')}
    n = len(mesh.faces)
    if any(len(a) != n for a in arrays.values()):
        p.error('0100 masks do not align with SDF mesh')
    if np.any(arrays['part_index'] >= len(names)):
        p.error('Part indices do not match SDF manifest')
    is_part = arrays['part_index'] == names.index(args.part)
    unknown = np.asarray(arrays['unknown'], bool)
    removed = np.asarray(arrays['remove'], bool)
    invalid = arrays['valid'] < arrays['assigned']
    selected, components = cap_candidates(
        face_adjacency=np.asarray(mesh.face_adjacency),
        centers=np.asarray(mesh.triangles_center), part=is_part,
        unknown=unknown, removed=removed, invalid=invalid,
        min_faces=args.min_faces, min_fraction=args.min_fraction,
        min_invalid_fraction=args.min_invalid_fraction,
        max_planarity=args.max_planarity,
    )
    # This is intentionally a preview, never a replacement for original 0100.
    dest.mkdir(parents=True)
    if selected.any():
        mesh_mask(mesh, selected).export(dest / f'{args.part}_unknown_cap_candidate.ply')
        mesh_mask(mesh, ~(removed | selected)).export(dest / 'fused_kept_cap_preview.ply')
    preview_report = {
        'status': 'EXPERIMENTAL_ONLY_NOT_PRODUCTION',
        'source_0100': str(source), 'part': args.part,
        'part_unknown_faces': int(np.sum(is_part & unknown)),
        'part_missing_valid_faces': int(np.sum(is_part & unknown & invalid)),
        'new_unknown_cap_candidates': int(selected.sum()),
        'confirmed_removed_faces_unchanged': int(removed.sum()),
        'components': components[:25],
        'safety': ('Only unknown, connected, planar components touching UDF-confirmed '
                   'removed faces are proposed. This is not a proof of surface absence; '
                   'inspect any preview before changing a production cutter.'),
    }
    (dest / 'report.json').write_text(json.dumps(preview_report, indent=2))
    print(json.dumps({k:v for k,v in preview_report.items() if k != 'components'}, indent=2))
    print('Top component audits:', json.dumps(components[:5], indent=2))


if __name__ == '__main__':
    main()
