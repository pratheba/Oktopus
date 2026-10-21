"""Read-only diagnostics for direct-final UDF cuts, with no additional UDF queries.

The categories describe *evidence on the final mesh*, not anatomical labels.
A region is labelled neck/sleeve/bottom only if the user supplies its coordinates.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


CATEGORIES = (
    'not_assigned',          # no curve interval associated with the face
    'assigned_no_raw_udf',  # localized to an owner, no usable raw sample
    'raw_valid_abstained',  # query worked; metric/plateau calibration abstained
    'effective_no_fused',  # supported by owner(s) but no fused score
    'below_grow',
    'grow_only',
    'seed_not_removed',
    'removed',
)


def classify_face_evidence(*, assignments, raw_valid, effective_valid,
                           fused, remove_mask):
    """Mutually exclusive categorical audit (one category for every face)."""
    assignments = np.asarray(assignments, dtype=np.int64)
    raw_valid = np.asarray(raw_valid, dtype=np.int64)
    effective_valid = np.asarray(effective_valid, dtype=np.int64)
    fused = np.asarray(fused, dtype=float)
    remove_mask = np.asarray(remove_mask, dtype=bool)
    n = len(assignments)
    if any(len(a) != n for a in (raw_valid, effective_valid, fused, remove_mask)):
        raise ValueError('0074 diagnostic arrays have inconsistent lengths')
    if np.any(assignments < raw_valid) or np.any(raw_valid < effective_valid):
        raise ValueError('0074 owner evidence counts are inconsistent')
    category = np.full(n, 3, dtype=np.uint8)
    category[assignments == 0] = 0
    category[(assignments > 0) & (raw_valid == 0)] = 1
    category[(raw_valid > 0) & (effective_valid == 0)] = 2
    usable = (effective_valid > 0) & np.isfinite(fused)
    category[usable & (fused < 1.)] = 4
    category[usable & (fused >= 1.) & (fused < 2.)] = 5
    category[usable & (fused >= 2.)] = 6
    category[remove_mask] = 7
    return category


def summarise_masks(*, centroids, category, owners, regions=(), bins=10):
    """Report evidence for supplied center/radius regions and coordinate bins.

    Axis bins are purely spatial diagnostics: never call them anatomical regions.
    """
    centroids = np.asarray(centroids, dtype=np.float64)
    category = np.asarray(category, dtype=np.uint8)
    if centroids.ndim != 2 or centroids.shape[1] != 3 or len(centroids) != len(category):
        raise ValueError('0074 centroids must be N x 3 and match category length')
    if bins < 1 or bins > 100:
        raise ValueError('0074 bins must be between 1 and 100')

    def tally(sel):
        return {'total_faces': int(sel.sum()),
                'classes': {name: int(np.count_nonzero(sel & (category == i)))
                            for i, name in enumerate(CATEGORIES)}}

    summary = {'category_meanings': list(CATEGORIES),
               'all': tally(np.ones(len(category), dtype=bool)),
               'owners': [], 'regions': [], 'axis_bins': {}}
    for owner in owners:
        ids = np.asarray(owner['face_ids'], dtype=np.int64)
        raw = np.asarray(owner['raw_scores'], dtype=np.float64)
        valid = np.asarray(owner['raw_valid'], dtype=bool)
        effective = np.asarray(owner['effective_valid'], dtype=bool)
        if ids.ndim != 1 or np.any((ids < 0) | (ids >= len(category))):
            raise ValueError('0074 invalid owner face IDs')
        if not (len(ids) == len(raw) == len(valid) == len(effective)):
            raise ValueError('0074 owner score dimensions mismatch')
        # Separate owners can overlap. Each is independently accounted here.
        summary['owners'].append({
            'item_index': int(owner['item_index']),
            'assigned_faces': int(len(ids)),
            'raw_valid_faces': int(valid.sum()),
            'effective_valid_faces': int(effective.sum()),
            'raw_seed_faces': int(np.count_nonzero(effective & (raw >= owner['seed_world']))),
            'raw_grow_faces': int(np.count_nonzero(effective & (raw >= owner['grow_world']))),
            'seed_world': float(owner['seed_world']),
            'grow_world': float(owner['grow_world']),
            'metric_status': owner.get('metric_status'),
            'spatial_category_counts': tally(np.isin(np.arange(len(category)), ids)),
        })
    for region in regions:
        if not isinstance(region, dict) or set(('name', 'center', 'radius_world')) - set(region):
            raise ValueError('0074 regions need name, center, radius_world')
        center = np.asarray(region['center'], dtype=np.float64)
        radius = float(region['radius_world'])
        name = region['name']
        if not isinstance(name, str) or not name or center.shape != (3,) or not np.all(np.isfinite(center)) or not np.isfinite(radius) or radius <= 0:
            raise ValueError('0074 invalid diagnostic region')
        inside = np.sum((centroids - center) ** 2, axis=1) <= radius**2
        summary['regions'].append({'name': name, 'center': center.tolist(),
                                   'radius_world': radius, **tally(inside)})
    for a, axis in enumerate('xyz'):
        if len(centroids):
            lower, upper = float(np.min(centroids[:, a])), float(np.max(centroids[:, a]))
        else:
            lower, upper = 0., 0.
        if upper > lower:
            index = np.minimum(bins - 1, ((centroids[:, a] - lower) / (upper - lower) * bins).astype(int))
        else:
            index = np.zeros(len(category), dtype=int)
        edges = np.linspace(lower, upper, bins + 1)
        summary['axis_bins'][axis] = [
            {'bin': j, 'low': float(edges[j]), 'high': float(edges[j + 1]), **tally(index == j)}
            for j in range(bins)
        ]
    return summary


def export_evidence(mesh, *, part, output_folder, payload, regions=()):
    """Persist classified faces, raw per-owner UDF scores and an atlas of PLY patches.

    Does NOT change or re-export the input mesh, and never calls the UDF model.
    """
    from_path = Path(output_folder) / 'udf_cut_evidence_0074' / part
    from_path.mkdir(parents=True, exist_ok=True)
    n = len(mesh.faces)
    category = classify_face_evidence(
        assignments=payload['assignment_count'], raw_valid=payload['raw_valid_count'],
        effective_valid=payload['effective_valid_count'], fused=payload['fused'],
        remove_mask=payload['remove_mask'])
    if len(category) != n:
        raise ValueError('0074 evidence does not align with original mesh faces')
    centroids = np.asarray(mesh.triangles_center, dtype=np.float64)
    summary = summarise_masks(centroids=centroids, category=category,
                              owners=payload['owners'], regions=regions)
    compact = {
        'face_category': category,
        'face_centroids_xyz': centroids.astype(np.float32),
        'assigned_owners_count': np.asarray(payload['assignment_count'], dtype=np.uint8),
        'raw_valid_owners_count': np.asarray(payload['raw_valid_count'], dtype=np.uint8),
        'effective_valid_owners_count': np.asarray(payload['effective_valid_count'], dtype=np.uint8),
        'normalized_fused_score': np.asarray(payload['fused'], dtype=np.float32),
        'pre_topology_cut_mask': np.asarray(payload['pre_topology_mask'], dtype=np.uint8),
        'final_remove_mask': np.asarray(payload['remove_mask'], dtype=np.uint8),
    }
    for o in payload['owners']:
        stem = f"item_{int(o['item_index']):02d}"
        compact[stem + '_faces'] = np.asarray(o['face_ids'], dtype=np.int32)
        compact[stem + '_raw_scores'] = np.asarray(o['raw_scores'], dtype=np.float32)
        compact[stem + '_raw_valid'] = np.asarray(o['raw_valid'], dtype=np.uint8)
        compact[stem + '_effective_valid'] = np.asarray(o['effective_valid'], dtype=np.uint8)
    np.savez_compressed(from_path / 'face_evidence.npz', **compact)
    (from_path / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    from trimesh import Trimesh
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    exported = {}
    for idx, name in enumerate(CATEGORIES):
        indices = np.flatnonzero(category == idx)
        if len(indices) == 0:
            continue
        # Preserve coordinates; remove unreferenced vertices for compact PLYs.
        fragment = Trimesh(vertices=vertices, faces=faces[indices], process=False)
        fragment.remove_unreferenced_vertices()
        path = from_path / f'{name}.ply'
        fragment.export(str(path))
        exported[name] = int(len(indices))
    summary['exports'] = {'npz': str(from_path / 'face_evidence.npz'),
                          'json': str(from_path / 'summary.json'),
                          'face_plys': exported}
    # Re-save with file destinations.
    (from_path / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary
