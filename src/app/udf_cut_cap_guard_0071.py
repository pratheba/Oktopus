"""Conservative end-cap validation for experimental direct-final UDF cuts.

A high UDF value on a blended side wall is not evidence that the wall should
be removed. This guard rejects entire UDF grow-components unless their faces
are primarily close to a target curve endpoint AND orientated across the
curve's axis. It does not generate new cuts or claim parity with per-item
removed/kept reference transfer.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def calibrate_p95_from_final_faces(
    scores, usable, *, far_world,
    percentile=95.0, seed_multiplier=1.10, grow_multiplier=0.65,
    far_band_world=0.002, min_calibration_fraction=0.05,
    min_calibration_faces=128,
):
    """Reproduce the legacy P95/seed-grow *formula* on final-mesh faces.

    The reference population differs from the original individual meshes.
    Returns None thresholds if the informative population is insufficient.
    """
    if not (0 < float(percentile) < 100):
        raise ValueError('0071 percentile must be in (0, 100)')
    if not (0 < float(grow_multiplier) < float(seed_multiplier)):
        raise ValueError('0071 requires 0 < grow_multiplier < seed_multiplier')
    if float(far_band_world) < 0 or not (0 <= min_calibration_fraction <= 1):
        raise ValueError('Invalid 0071 calibration limits')
    if int(min_calibration_faces) < 0:
        raise ValueError('Invalid 0071 minimum calibration faces')
    values = np.asarray(scores, dtype=np.float64)
    supported = np.asarray(usable, dtype=bool) & np.isfinite(values)
    valid = values[supported]
    usable_count = int(valid.size)
    informative = valid[valid < float(far_world) - float(far_band_world)]
    required = max(int(min_calibration_faces), int(np.ceil(min_calibration_fraction * usable_count)))
    saturated = (usable_count == 0 or informative.size < required)
    report = {
        'usable': usable_count,
        'informative': int(informative.size),
        'required': required,
        'saturated': bool(saturated),
        'far_fraction': float(np.mean(valid >= float(far_world) - float(far_band_world))) if usable_count else 1.0,
    }
    if saturated:
        return None, None, report
    reference = float(np.percentile(informative, percentile))
    seed = reference * float(seed_multiplier)
    grow = reference * float(grow_multiplier)
    report.update({'percentile': float(percentile), 'reference': reference, 'seed': seed, 'grow': grow})
    return seed, grow, report


def endcap_face_evidence(mesh, avatar_curve, *, source_bounds, endpoint_band=0.07,
                         min_normal_dot=0.7, curve_samples=1025, end='low'):
    """Return whether faces lie in an axial end-cap at either source bound.

    Nearest sampled centerline positions are used only for a spatial and
    orientation guard, never to create a UDF cut. The curve coordinates and
    source bounds are in the same parameterization as 0069 ownership.
    """
    lo, hi = map(float, source_bounds)
    if not np.isfinite(lo + hi) or not lo < hi:
        raise ValueError('Invalid endcap source bounds')
    if not (0 < float(endpoint_band) < 0.5 * (hi - lo)):
        raise ValueError('0071 endpoint_band must be positive and less than half the source span')
    if not 0 <= float(min_normal_dot) <= 1:
        raise ValueError('Invalid 0071 normal alignment')
    if not (32 <= int(curve_samples) <= 8192):
        raise ValueError('Invalid 0071 curve_samples')
    if end not in ('low', 'high', 'both'):
        raise ValueError('Invalid 0071 cap end; expected low/high/both')

    core = avatar_curve.core
    key_ts = np.asarray(core.key_ts, dtype=np.float64).reshape(-1)
    ts = np.linspace(key_ts[0], key_ts[-1], int(curve_samples))
    points = np.asarray(core.interpolate(ts)['points'], dtype=np.float64)
    if points.shape != (len(ts), 3) or not np.all(np.isfinite(points)):
        raise ValueError('Invalid 0071 centerline samples')
    # Use an axial tangent from central finite differences; interpolated face
    # normals may be unoriented because the input SDF mesh is nonwatertight.
    tangents = np.gradient(points, axis=0)
    tangent_length = np.linalg.norm(tangents, axis=1)
    tangents /= np.maximum(tangent_length[:, None], 1e-15)
    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    _, idx = cKDTree(points).query(centers, k=1)
    idx = np.asarray(idx, dtype=np.int64)
    face_param = ts[idx]
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    alignment = np.abs(np.einsum('ij,ij->i', normals, tangents[idx]))
    low_end = np.abs(face_param - lo) <= float(endpoint_band)
    high_end = np.abs(face_param - hi) <= float(endpoint_band)
    in_endpoint = low_end if end == 'low' else high_end if end == 'high' else (low_end | high_end)
    axial = alignment >= float(min_normal_dot)
    return in_endpoint, axial, {'bounds': [lo, hi], 'band': float(endpoint_band),
                                'alignment': float(min_normal_dot), 'end': end,
                                'end_faces': int(in_endpoint.sum()),
                                'axial_faces': int(axial.sum()),
                                'cap_faces': int(np.count_nonzero(in_endpoint & axial))}


def filter_cap_components(mesh, remove_mask, grow_mask, decisions, *,
                          adjacency, in_endpoint, axial,
                          component_func, min_end_fraction=0.8,
                          min_axial_fraction=0.60, min_cap_faces=12):
    """Reject a whole accepted grow-component if it is not a plausible cap."""
    if not (0 <= min_end_fraction <= 1 and 0 <= min_axial_fraction <= 1):
        raise ValueError('Invalid 0071 component fractions')
    if int(min_cap_faces) < 1:
        raise ValueError('Invalid 0071 min_cap_faces')
    output = np.asarray(remove_mask, dtype=bool).copy()
    components, _labels = component_func(np.asarray(grow_mask, dtype=bool), adjacency)
    diagnostics = []
    for i, ids in enumerate(components):
        if not decisions[i].accepted:
            continue
        end_fraction = float(np.mean(in_endpoint[ids]))
        axial_fraction = float(np.mean(axial[ids]))
        cap_faces = int(np.count_nonzero(in_endpoint[ids] & axial[ids]))
        reason = None
        if end_fraction < min_end_fraction:
            reason = '0071_not_at_curve_end'
        elif axial_fraction < min_axial_fraction:
            reason = '0071_not_axial_cap'
        elif cap_faces < int(min_cap_faces):
            reason = '0071_insufficient_cap_faces'
        if reason:
            output[ids] = False
            decisions[i].accepted = False
            decisions[i].reason = reason
        diagnostics.append({'component': i, 'faces': int(len(ids)),
                            'end_fraction': end_fraction,
                            'axial_fraction': axial_fraction,
                            'cap_faces': cap_faces,
                            'action': 'reject' if reason else 'keep_cut',
                            'reason': reason or 'plausible_cap'})
    return output, diagnostics
