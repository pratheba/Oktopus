"""0092: cut the true 0087 fused SDF surface using *saved* 0081 UDF evidence.

This is a conservative evidence projection, NOT a second learned-UDF inference
on the fused triangles. Spatial proximity alone is never a cut instruction:
match source triangles via closest-point distance AND face orientation, then
project only selected source cut faces. Unknown matches retain fused faces.

A separately audited end-cap fallback handles zero accepted UDF seeds, but
only for coherent, planar, curve-terminal candidate components. Every such
component and unmatched region is reported; never call this a watertightness
or manifold guarantee.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from udf_cut_reconcile_0079 import _edge_component_labels, keep_largest_edge_component


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), force='mesh', process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError(f'0092 invalid triangular mesh: {path}')
    return mesh


def mesh_from_mask(mesh: trimesh.Trimesh, mask: np.ndarray) -> trimesh.Trimesh:
    result = trimesh.Trimesh(vertices=np.asarray(mesh.vertices).copy(),
                             faces=np.asarray(mesh.faces)[mask].copy(), process=False)
    result.remove_unreferenced_vertices()
    return result


def choose_zero_seed_endcaps(mesh: trimesh.Trimesh, seeds: np.ndarray,
                            candidates: np.ndarray, arc: np.ndarray,
                            *, min_faces=8, end_band=.12, axial_dot=.72,
                            normal_coherence=.72, max_arc_width=.14):
    """Conservative, auditable optional cap classifier for zero-seed parts.

    The curve endpoint is defined by the *observed* arc span of the original
    mesh, not by an assumed global skeleton endpoint at s=0 or s=1. A valid
    cap must be an entire edge-connected candidate patch, have consistent
    normals, be approximately perpendicular to the local curve, and occupy
    a narrow terminal arclength band. Hairpin-adjacent sidewalls fail either
    the normal or arc coherence gate rather than being cut by XYZ proximity.
    """
    n = len(mesh.faces)
    seed = np.asarray(seeds, bool).reshape(-1)
    cand = np.asarray(candidates, bool).reshape(-1)
    s = np.asarray(arc, float).reshape(-1)
    if len(seed) != n or len(cand) != n or len(s) != n or not np.isfinite(s).all():
        raise ValueError('0092 zero-seed mask / arc mismatch')
    mask = np.zeros(n, bool)
    report = []
    if seed.any() or not cand.any():
        return mask, report
    indices = np.flatnonzero(cand)
    sub = np.asarray(mesh.faces)[indices]
    labels, counts = _edge_component_labels(sub)
    # Curved-skeleton direction estimated using the centroids of nearby
    # arclength bands, not the global PCA direction (which fails on hairpins).
    centroids = np.asarray(mesh.triangles_center, float)
    norms = np.asarray(mesh.face_normals, float)
    areas = np.asarray(mesh.area_faces, float)
    low_s, high_s = np.quantile(s, [.02, .98])
    span = max(1e-6, high_s - low_s)
    for i, num in enumerate(counts):
        ids = indices[labels == i]
        piece = s[ids]
        score = {'component': int(i), 'faces': int(num), 'accepted': False}
        report.append(score)
        if len(ids) < int(min_faces):
            score['reason'] = 'too_few_faces'; continue
        med = float(np.median(piece))
        terminal = 'low' if med <= low_s + end_band * span else (
                   'high' if med >= high_s - end_band * span else None)
        if terminal is None:
            score['reason'] = 'not_curve_terminal'; continue
        if np.quantile(piece, .95) - np.quantile(piece, .05) > max_arc_width * span:
            score['reason'] = 'too_wide_in_arclength'; continue
        # Estimate local centerline direction from two adjacent narrow s bins.
        if terminal == 'low':
            a = (s >= low_s) & (s <= low_s + .12 * span)
            b = (s >= low_s + .17 * span) & (s <= low_s + .34 * span)
        else:
            a = (s >= high_s - .12 * span) & (s <= high_s)
            b = (s >= high_s - .34 * span) & (s <= high_s - .17 * span)
        if a.sum() < 3 or b.sum() < 3:
            score['reason'] = 'insufficient_local_centerline'; continue
        tangent = centroids[b].mean(axis=0) - centroids[a].mean(axis=0)
        tlen = np.linalg.norm(tangent)
        if tlen < 1e-8:
            score['reason'] = 'ambiguous_local_centerline'; continue
        tangent /= tlen
        weights = areas[ids]
        if weights.sum() < 1e-10:
            score['reason'] = 'zero_area'; continue
        combined_normal = np.average(norms[ids], axis=0, weights=weights)
        coherence = float(np.linalg.norm(combined_normal))
        axial = float(np.average(np.abs(norms[ids] @ tangent), weights=weights))
        score.update(terminal=terminal, axial_alignment=round(axial, 4),
                     normal_coherence=round(coherence, 4),
                     arc_span=round(float((np.max(piece) - np.min(piece)) / span), 4))
        if coherence < normal_coherence or axial < axial_dot:
            score['reason'] = 'not_coherent_axial_cap'; continue
        mask[ids] = True
        score.update(accepted=True, reason='coherent_terminal_cap')
    return mask, report


def transfer_cut_to_fused(fused: trimesh.Trimesh, sources: Mapping[str, tuple],
                          *, distance_edge_factor=1.75, normal_dot=.55,
                          neighbors=12, batch_size=4096):
    """Map source cut decisions to the nearest *aligned source triangle*.

    Uses closest points on triangles (not nearest center distance alone).
    Preserving unmatched areas is intentional; importantly, two source
    surfaces crossing near one another cannot each independently delete the
    fused shell simply because they approach the same XYZ coordinates.
    """
    if not sources:
        raise ValueError('0092 at least one source part is required')
    if not (0 < float(normal_dot) <= 1 and 0 < float(distance_edge_factor) <= 10):
        raise ValueError('0092 invalid transfer tolerances')
    centers, triangles, normals, selected, labels, lengths = [], [], [], [], [], []
    names = list(sources)
    for p, name in enumerate(names):
        mesh, mask = sources[name]
        mask = np.asarray(mask, bool)
        if len(mask) != len(mesh.faces):
            raise ValueError(f'0092 {name} selection does not match face count')
        centers.append(np.asarray(mesh.triangles_center))
        triangles.append(np.asarray(mesh.triangles))
        normals.append(np.asarray(mesh.face_normals))
        selected.append(mask)
        labels.append(np.full(len(mask), p, dtype=np.int16))
        e = np.asarray(mesh.edges_unique_length)
        lengths.append(float(np.median(e[e > 1e-12])))
    centers = np.concatenate(centers)
    triangles = np.concatenate(triangles)
    normals = np.concatenate(normals)
    selected = np.concatenate(selected)
    labels = np.concatenate(labels)
    target_edge = np.asarray(fused.edges_unique_length)
    target_scale = float(np.median(target_edge[target_edge > 1e-12]))
    # Strict tolerance measured in local edge lengths. A match far from an
    # underlying SDF part is not enough evidence to modify the fused surface.
    limit = float(distance_edge_factor) * max(target_scale, float(np.median(lengths)))
    tree = cKDTree(centers)
    n = len(fused.faces)
    picked = np.full(n, -1, dtype=np.int32)
    distance = np.full(n, np.inf, dtype=np.float32)
    keep_mask = np.zeros(n, bool)
    normal_ok = np.zeros(n, bool)
    accepted = np.zeros(n, bool)
    count_by_part = {k: 0 for k in names}
    k = min(int(neighbors), len(centers))
    for start in range(0, n, int(batch_size)):
        end = min(n, start + int(batch_size))
        pts = np.asarray(fused.triangles_center[start:end])
        candidates = np.asarray(tree.query(pts, k=k)[1], dtype=np.int64).reshape(-1, k)
        # Exact nearest point for k plausible source triangles.
        mesh_pts = np.repeat(pts, k, axis=0)
        closest = trimesh.triangles.closest_point(triangles[candidates.reshape(-1)], mesh_pts)
        dists = np.linalg.norm(mesh_pts - closest, axis=1).reshape(-1, k)
        target_norms = np.asarray(fused.face_normals[start:end])
        dots = np.einsum('ijk,ik->ij', normals[candidates], target_norms)
        # Signed normals protect against matching backside/internal faces.
        oriented = dots >= float(normal_dot)
        valid = oriented & (dists <= limit)
        best_scores = np.where(valid, dists, np.inf)
        choice = np.argmin(best_scores, axis=1)
        best_dist = best_scores[np.arange(end-start), choice]
        matched = np.isfinite(best_dist)
        refs = candidates[np.arange(end-start), choice]
        out_index = np.arange(start, end)[matched]
        ids = refs[matched]
        picked[out_index] = labels[ids]
        distance[out_index] = best_dist[matched]
        normal_ok[out_index] = True
        accepted[out_index] = selected[ids]
        keep_mask[out_index] = ~selected[ids]
        for j, name in enumerate(names):
            count_by_part[name] += int(np.count_nonzero(accepted[out_index] & (labels[ids] == j)))
    # A cut face must have a trusted mapping. Leave all unmatched faces intact.
    report = {
        'fused_faces': n, 'matched_faces': int((picked >= 0).sum()),
        'unmatched_faces': int((picked < 0).sum()),
        'proposed_remove_faces': int(accepted.sum()),
        'matched_kept_faces': int(keep_mask.sum()),
        'distance_limit_world': limit, 'distance_edge_factor': float(distance_edge_factor),
        'normal_alignment_min_dot': float(normal_dot),
        'removed_votes_by_part': count_by_part,
    }
    return accepted, picked, distance, report


def generate_fused_cut(*, sdf_shape: Path, udf_shape: Path, out: Path,
                       distance_edge_factor=1.75, normal_dot=.55,
                       allow_zero_seed_caps=True, require_nonempty_cut=True):
    """Use saved per-part 0081 UDF decisions to cut the genuine fused SDF PLY.

    Never uses the 0084 seam-stitch function; the SDF union is already built.
    Read-only with respect to input SDF and saved UDF evidence directories.
    """
    sdf_shape = Path(sdf_shape).resolve()
    udf_shape = Path(udf_shape).resolve()
    out = Path(out).resolve()
    fuse = sdf_shape / 'riggable_fused_sdf_exterior.ply'
    if not fuse.is_file():
        raise FileNotFoundError(f'0092 requires true 0087 fused SDF: {fuse}')
    parts_dir = sdf_shape / 'riggable_parts'
    local_dir = udf_shape / 'udf_cut_curve_local_0081'
    if not parts_dir.is_dir() or not local_dir.is_dir():
        raise FileNotFoundError('0092 requires original SDF riggable_parts and 0081 UDF locality evidence')
    manifest = sdf_shape / 'root_leaves_ownership.json'
    if not manifest.is_file():
        raise FileNotFoundError(f'0092 requires SDF ownership manifest: {manifest}')
    report_path = sdf_shape / 'riggable_fused_sdf_exterior_report.json'
    if not report_path.is_file():
        raise FileNotFoundError(f'0092 requires 0087 fusion report: {report_path}')
    # Use the real part set from the manifest, not a fixed tentacle list.
    manifest_data = json.loads(manifest.read_text(encoding='utf8'))
    part_names = list(manifest_data.get('part_names') or manifest_data.get('parts', {}))
    if not part_names:
        part_names = sorted(p.stem for p in parts_dir.glob('*.ply'))
    if not part_names:
        raise ValueError('0092 found no SDF parts')
    fused = load_mesh(fuse)
    sources = {}
    per_part = {}
    for name in part_names:
        src = load_mesh(parts_dir / f'{name}.ply')
        file = local_dir / name / 'locality.npz'
        if not file.is_file():
            raise FileNotFoundError(f'0092 missing UDF locality evidence: {file}')
        with np.load(file) as z:
            seeds = np.asarray(z['seed_faces'], bool)
            candidate = np.asarray(z['candidate_faces'], bool)
            selected = np.asarray(z['selected_faces'], bool)
            arc = np.asarray(z['face_curve_arclength'], float)
            saved_centers = np.asarray(z['face_centroids_xyz'], float)
        if len(src.faces) != len(selected) or len(selected) != len(seeds):
            raise ValueError(f'0092 face count mismatch for {name}')
        actual_centers = np.asarray(src.triangles_center)
        if saved_centers.shape != actual_centers.shape or np.max(np.linalg.norm(saved_centers - actual_centers, axis=1)) > 1e-5:
            raise ValueError(f'0092 UDF and SDF source geometry do not match for {name}; refusing transfer')
        extra, cap_report = choose_zero_seed_endcaps(src, seeds, candidate, arc) if allow_zero_seed_caps else (np.zeros(len(src.faces), bool), [])
        selected = selected | extra
        sources[name] = (src, selected)
        per_part[name] = {'source_faces': len(src.faces), 'udf_selected': int(np.count_nonzero(selected & ~extra)),
                          'zero_seed_cap_added': int(extra.sum()), 'zero_seed_audit': cap_report,
                          'source_selected_total': int(selected.sum())}
    cut_mask, owner, distances, transfer_report = transfer_cut_to_fused(
        fused, sources, distance_edge_factor=distance_edge_factor, normal_dot=normal_dot)
    if not np.any(cut_mask) and require_nonempty_cut:
        raise RuntimeError('0092 found zero matched UDF cut faces on the fused SDF exterior; refusing empty result')
    if np.all(cut_mask):
        raise RuntimeError('0092 would remove the entire fused exterior')
    before = len(fused.faces)
    candidate_result = mesh_from_mask(fused, ~cut_mask)
    # A global fused exterior can be legitimately disconnected; cleanup must
    # not throw away independent major pieces. Only prune when clearly safe.
    try:
        cleaned, fragments, cleanup = keep_largest_edge_component(candidate_result, min_main_fraction=.98)
    except ValueError as exc:
        cleaned = candidate_result
        fragments = None
        cleanup = {'skipped': True, 'reason': str(exc)}
    out.mkdir(parents=True, exist_ok=True)
    name = out / 'riggable_fused_sdf_exterior_after_udf_cut.ply'
    cleaned.export(name)
    if np.any(cut_mask):
        mesh_from_mask(fused, cut_mask).export(out/'fused_udf_removed_faces.ply')
    if fragments is not None and len(fragments.faces):
        fragments.export(out/'fused_udf_detached_fragments.ply')
    np.savez_compressed(out/'fused_udf_transfer_evidence_0092.npz',
                        cut_mask=cut_mask, source_part_index=owner,
                        correspondence_distance=distances)
    unresolved_zero_seed = [name for name, row in per_part.items()
                            if row['udf_selected'] == 0 and
                            (row['zero_seed_cap_added'] == 0 or
                             transfer_report['removed_votes_by_part'].get(name, 0) == 0)]
    report = {'status': 'FUSED_UDF_EVIDENCE_TRANSFER_0092',
              'method': 'project_existing_per_part_0081_UDF_decisions_onto_fused_SDF',
              'not_direct_fused_model_inference': True,
              'not_triangle_stitching': True,
              'sdf_fused_input': str(fuse), 'udf_evidence_dir': str(local_dir),
              'output_ply': str(name), 'faces_before': before,
              'faces_removed': int(cut_mask.sum()), 'faces_after': int(len(cleaned.faces)),
              'is_watertight': bool(cleaned.is_watertight),
              'is_winding_consistent': bool(cleaned.is_winding_consistent),
              'transfer': transfer_report, 'parts': per_part, 'cleanup': cleanup,
              'unresolved_zero_seed_parts': unresolved_zero_seed,
              'unmatched_policy': 'preserve',
              'warnings': ['No claim of manifold/watertight output: check geometry and unmatched faces.',
                           'Zero-seed endcap classification is heuristic and needs visual confirmation.',
                           'Fused surface is cut by UDF evidence TRANSFER, not a new UDF model query.']}
    if unresolved_zero_seed:
        report['warnings'].append('Zero-seed parts still unresolved: ' + ', '.join(unresolved_zero_seed))
    if transfer_report['unmatched_faces'] > len(fused.faces) * .25:
        report['warnings'].append('More than 25% of the fused mesh has no aligned part-face evidence; do not treat this as a complete UDF cut.')
    (out/'riggable_fused_sdf_exterior_after_udf_cut_report.json').write_text(
        json.dumps(report, indent=2), encoding='utf8')
    print('[udf fused 0092] cut',cut_mask.sum(),'/',before,'faces',
          'matched',transfer_report['matched_faces'],
          'remaining',len(cleaned.faces),'output',name,flush=True)
    return report
