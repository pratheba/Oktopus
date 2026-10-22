#!/usr/bin/env python3
"""0103: offline, experimental *mesh-connected* UNKNOWN removal preview.

Never uses Euclidean nearest-neighbor proximity to propagate a cut.  It
classifies entire edge-connected UNKNOWN components *within the same routed
part*, retaining all 0100 UDF-SUPPORTED faces.  Components must have mostly
invalid UDF evidence and their external boundary must predominantly touch
independently UDF-CONFIRMED REMOVE faces.  Planarity is not required.

These geometric gates are not proof of UDF surface absence.  Outputs are
DIAGNOSTIC ONLY. No source meshes, saved UDF evidence, or production cutter are
changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def choose_connected_unknown(*, face_adjacency, face_area, part, unknown,
                             removed, supported, invalid, min_area_fraction=0.005,
                             min_invalid_fraction=0.60,
                             min_removed_contact_fraction=0.80,
                             min_removed_contact_edges=4):
    """Select coherent unknown components using only *true mesh edge adjacency*.

    Component eligibility is based on physical area ratio and boundary evidence,
    not absolute face counts (which change 4x between 512 and 1024).
    `part` scopes the graph to the currently routed part. `supported` is a hard
    veto: no face classed as supported is ever selected.
    """
    part, unknown, removed, supported, invalid = (
        np.asarray(arr, dtype=bool).reshape(-1)
        for arr in (part, unknown, removed, supported, invalid)
    )
    area = np.asarray(face_area, dtype=np.float64).reshape(-1)
    n = len(part)
    if not all(len(arr) == n for arr in (unknown, removed, supported, invalid, area)):
        raise ValueError('0103 face masks / area length mismatch')
    if not np.all(np.isfinite(area)) or np.any(area < 0):
        raise ValueError('0103 invalid face area')
    if np.any(unknown & (removed | supported)) or np.any(removed & supported):
        raise ValueError('0103 overlapping UDF categories; refusing preview')
    if not (0 <= min_area_fraction <= 1 and
            0 <= min_invalid_fraction <= 1 and
            0 <= min_removed_contact_fraction <= 1 and
            int(min_removed_contact_edges) >= 1):
        raise ValueError('0103 invalid preview thresholds')
    edges = np.asarray(face_adjacency, dtype=np.int64).reshape(-1, 2)
    if len(edges) and (edges.min() < 0 or edges.max() >= n):
        raise ValueError('0103 invalid mesh face adjacency')
    active = part & unknown
    ids = np.flatnonzero(active)
    out = np.zeros(n, dtype=bool)
    if ids.size == 0:
        return out, []
    part_area = float(area[part].sum())
    if part_area <= 0:
        raise ValueError('0103 part has no positive surface area')
    local = np.full(n, -1, dtype=np.int32)
    local[ids] = np.arange(len(ids), dtype=np.int32)
    both_unknown = active[edges[:, 0]] & active[edges[:, 1]] if len(edges) else np.zeros(0, bool)
    uv = local[edges[both_unknown]]
    mat = coo_matrix((np.ones(2 * len(uv), dtype=np.uint8),
                      (np.r_[uv[:, 0], uv[:, 1]], np.r_[uv[:, 1], uv[:, 0]])),
                     shape=(len(ids), len(ids))).tocsr()
    component_count, lab = connected_components(mat, directed=False)
    sizes = np.bincount(lab, minlength=component_count)
    areas = np.bincount(lab, weights=area[ids], minlength=component_count)
    bad = np.bincount(lab, weights=area[ids] * invalid[ids], minlength=component_count)
    removed_contacts = np.zeros(component_count, dtype=np.int64)
    supported_contacts = np.zeros(component_count, dtype=np.int64)
    other_unknown_contacts = np.zeros(component_count, dtype=np.int64)
    other_part_contacts = np.zeros(component_count, dtype=np.int64)
    # Each undirected edge can contribute from either orientation, never a
    # centroid-to-centroid proximity lookup.  All contacts must stay in part.
    for a, b in ((edges[:, 0], edges[:, 1]), (edges[:, 1], edges[:, 0])):
        border = active[a] & ~active[b]
        if not np.any(border):
            continue
        labels = lab[local[a[border]]]
        neighbors = b[border]
        in_part = part[neighbors]
        removed_contacts += np.bincount(labels[removed[neighbors] & in_part], minlength=component_count)
        supported_contacts += np.bincount(labels[supported[neighbors] & in_part], minlength=component_count)
        other_unknown_contacts += np.bincount(labels[unknown[neighbors] & in_part], minlength=component_count)
        other_part_contacts += np.bincount(labels[~in_part], minlength=component_count)
    ext = removed_contacts + supported_contacts + other_part_contacts
    frac = np.divide(removed_contacts, ext, out=np.zeros(component_count, float), where=ext > 0)
    ratio = np.divide(bad, areas, out=np.zeros(component_count, float), where=areas > 0)
    area_fraction = areas / part_area
    eligible = (area_fraction >= min_area_fraction) & (ratio >= min_invalid_fraction) & (
        removed_contacts >= min_removed_contact_edges) & (frac >= min_removed_contact_fraction)
    # A very large 1024 fused mesh can contain thousands of tiny islands. Audit
    # the largest 32 plus every selected region, rather than making a dict for
    # every triangle-sized component.
    ranked = np.argsort(-sizes, kind='stable')
    audit_ids = np.unique(np.r_[ranked[:32], np.flatnonzero(eligible)])
    candidates = []
    for i in audit_ids:
        reasons = []
        if area_fraction[i] < min_area_fraction:
            reasons.append('too_small_by_physical_area')
        if ratio[i] < min_invalid_fraction:
            reasons.append('insufficient_invalid_udf_evidence')
        if removed_contacts[i] < min_removed_contact_edges:
            reasons.append('few_confirmed_removed_contacts')
        if frac[i] < min_removed_contact_fraction:
            reasons.append('boundary_touches_supported_surface')
        accepted = bool(eligible[i])
        candidates.append({
            'component': int(i), 'faces': int(sizes[i]),
            'area_fraction_of_part': round(float(area_fraction[i]), 6),
            'invalid_area_fraction': round(float(ratio[i]), 6),
            'edges_to_confirmed_removed': int(removed_contacts[i]),
            'edges_to_supported': int(supported_contacts[i]),
            'edges_to_other_unknown': int(other_unknown_contacts[i]),
            'edges_to_other_part': int(other_part_contacts[i]),
            'removed_boundary_fraction': round(float(frac[i]), 6),
            'preview_eligible': bool(accepted), 'reasons': reasons,
        })
    out[ids] = eligible[lab]
    # Defensive veto in addition to graph construction.
    if np.any(out & (supported | removed | ~part)):
        raise RuntimeError('0103 selected a protected face')
    return out, sorted(candidates, key=lambda r: r['faces'], reverse=True)


def mesh_mask(mesh, mask):
    result = trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                             faces=np.asarray(mesh.faces)[mask], process=False)
    result.remove_unreferenced_vertices()
    return result


def keep_largest_edge_component(mesh):
    """Return (largest_component_mesh, dropped_debris_mesh, stats)."""
    if len(mesh.faces) == 0:
        empty = mesh.copy()
        stats = {
            'components_before': 0,
            'largest_faces': 0,
            'kept_fraction': 0.0,
            'dropped_faces': 0,
            'dropped_components': 0,
        }
        return empty, empty.copy(), stats
    comps = mesh.split(only_watertight=False, adjacency=None)
    if len(comps) <= 1:
        empty = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64), process=False)
        stats = {
            'components_before': int(len(comps)),
            'largest_faces': int(len(mesh.faces)),
            'kept_fraction': 1.0,
            'dropped_faces': 0,
            'dropped_components': 0,
        }
        return mesh.copy(), empty, stats
    sizes = np.array([len(c.faces) for c in comps], dtype=np.int64)
    keep_i = int(np.argmax(sizes))
    kept = comps[keep_i].copy()
    debris_parts = [c for i, c in enumerate(comps) if i != keep_i and len(c.faces) > 0]
    if debris_parts:
        debris = trimesh.util.concatenate(debris_parts)
        dropped_faces = int(sum(len(c.faces) for c in debris_parts))
    else:
        debris = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64), process=False)
        dropped_faces = 0
    total = int(sizes.sum())
    stats = {
        'components_before': int(len(comps)),
        'largest_faces': int(sizes[keep_i]),
        'kept_fraction': float(sizes[keep_i] / total) if total else 0.0,
        'dropped_faces': dropped_faces,
        'dropped_components': int(len(comps) - 1),
    }
    return kept, debris, stats


def run(args):
    source = Path(args.direct).resolve()
    dest = Path(args.out).resolve()
    if dest.exists():
        raise ValueError(f'0103 refusing to overwrite output: {dest}')
    report = json.loads((source / 'report.json').read_text(encoding='utf-8'))
    if report.get('status') != 'EXPERIMENTAL_DIRECT_FUSED_UDF_0100':
        raise ValueError('0103 requires original 0100 direct-UDF report')
    fused = Path(report['source_fused_sdf']).resolve()
    manifest = json.loads((fused.parent / 'root_leaves_ownership.json').read_text())
    names = list(manifest['part_names'])
    selected_parts = names if args.parts == 'all' else [p.strip() for p in args.parts.split(',')]
    if not selected_parts or len(set(selected_parts)) != len(selected_parts) or any(
            not p or p not in names for p in selected_parts):
        raise ValueError(f'0103 invalid --parts {args.parts}; expected subset of {names}')
    if any(p not in report.get('parts', {}) for p in selected_parts):
        raise ValueError('0103 requested part not queried in the original direct-UDF run')
    if any(p != 'body' for p in selected_parts) and not args.allow_uncalibrated_nonbody:
        raise ValueError('0103 non-body UDF thresholds uncalibrated; requires --allow-uncalibrated-nonbody')
    mesh = trimesh.load(str(fused), force='mesh', process=False)
    with np.load(source / 'consistency.npz') as z:
        data = {k: np.asarray(z[k]).copy() for k in
                ('assigned', 'valid', 'keep', 'remove', 'unknown', 'part_index')}
    n = len(mesh.faces)
    if any(len(v) != n for v in data.values()) or n != report.get('source_face_count'):
        raise ValueError('0103 direct masks do not match fused mesh face count')
    part_index = data['part_index']
    if np.any((part_index < -1) | (part_index >= len(names))):
        raise ValueError('0103 invalid part ownership')
    unknown = data['unknown'].astype(bool)
    removed = data['remove'].astype(bool)
    supported = data['keep'].astype(bool)
    invalid = data['valid'] < data['assigned']
    adjacency = np.asarray(mesh.face_adjacency)
    areas = np.asarray(mesh.area_faces)
    candidate = np.zeros(n, dtype=bool)
    stats = {}
    for part in selected_parts:
        is_part = part_index == names.index(part)
        part_proposal, audit = choose_connected_unknown(
            face_adjacency=adjacency, face_area=areas,
            part=is_part, unknown=unknown, removed=removed,
            supported=supported, invalid=invalid,
            min_area_fraction=args.min_area_fraction,
            min_invalid_fraction=args.min_invalid_fraction,
            min_removed_contact_fraction=args.min_removed_contact_fraction,
            min_removed_contact_edges=args.min_removed_contact_edges,
        )
        candidate |= part_proposal
        stats[part] = {
            'new_unknown_connected_faces': int(part_proposal.sum()),
            'existing_confirmed_removed_faces': int((is_part & removed).sum()),
            'part_unknown_faces': int((is_part & unknown).sum()),
            'parts_of_unknown': audit[:30],
        }
        print(f'[0103 {part}] remove={stats[part]["existing_confirmed_removed_faces"]}'
              f' unknown={stats[part]["part_unknown_faces"]}'
              f' added_connected={stats[part]["new_unknown_connected_faces"]}', flush=True)
    if np.any(candidate & (supported | removed)):
        raise RuntimeError('0103 would delete known-supported UDF faces')
    if np.all(removed | candidate):
        raise RuntimeError('0103 would remove the entire fused exterior')
    dest.mkdir(parents=True)
    if candidate.any():
        mesh_mask(mesh, candidate).export(dest / 'fused_newly_selected_unknown.ply')
        for part in selected_parts:
            pmask = candidate & (part_index == names.index(part))
            if pmask.any():
                mesh_mask(mesh, pmask).export(dest / f'{part}_newly_selected_unknown.ply')
    kept_mesh = mesh_mask(mesh, ~(removed | candidate))
    kept_mesh.export(dest / 'fused_kept_connected_unknown_preview.ply')
    largest_mesh, debris_mesh, cleanup = keep_largest_edge_component(kept_mesh)
    largest_mesh.export(dest / 'fused_kept_connected_unknown_preview_largest_component.ply')
    if cleanup['dropped_faces'] > 0:
        debris_mesh.export(dest / 'fused_kept_connected_unknown_preview_debris.ply')
    np.savez_compressed(dest / 'selection_0103.npz', confirmed_removed=removed,
                        newly_selected_unknown=candidate,
                        preview_removed=removed | candidate,
                        preview_kept=~(removed | candidate))
    output = {
        'status': 'EXPERIMENTAL_CONNECTED_UNKNOWN_PREVIEW_0103',
        'source_0100': str(source), 'parts': stats,
        'confirmed_removed_0100': int(removed.sum()),
        'additional_unknown_proposed': int(candidate.sum()),
        'total_preview_removed': int((removed | candidate).sum()),
        'known_supported_faces_deleted': int(np.count_nonzero(supported & (removed | candidate))),
        'criteria': {
            'connectivity': 'shared_mesh_edges_within_same_part; no Euclidean hopping',
            'min_area_fraction_of_part': args.min_area_fraction,
            'min_invalid_area_fraction': args.min_invalid_fraction,
            'min_removed_boundary_fraction': args.min_removed_contact_fraction,
            'min_removed_contact_edges': args.min_removed_contact_edges,
        },
        'cleanup_largest_component': cleanup,
        'limitations': [
            'Model-invalid is not proof of absent surface; new cuts are unverified proposals.',
            'Direct UDF thresholds for non-body parts are uncalibrated.',
            'A Boolean-fused actual mesh bridge can connect surfaces; this preview does not certify manifoldness or geometry.',
            'No complete source/UDF inference rerun; no production code changed.',
        ],
    }
    (dest / 'report.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k != 'parts'}, indent=2))
    for name, p in stats.items():
        print(f'[0103 {name}] largest components: {json.dumps(p["parts_of_unknown"][:3])}')
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--direct', required=True, help='Existing 0100 or 0102 direct-UDF output directory')
    p.add_argument('--out', required=True, help='New, nonexistent output directory')
    p.add_argument('--parts', default='body', help='Comma-separated parts or all (default body)')
    p.add_argument('--allow-uncalibrated-nonbody', action='store_true')
    p.add_argument('--min-area-fraction', type=float, default=.005)
    p.add_argument('--min-invalid-fraction', type=float, default=.60)
    p.add_argument('--min-removed-contact-fraction', type=float, default=.80)
    p.add_argument('--min-removed-contact-edges', type=int, default=4)
    args = p.parse_args()
    run(args)


if __name__ == '__main__':
    main()
