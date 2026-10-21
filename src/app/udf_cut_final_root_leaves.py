"""Direct learned-UDF cutting of final, separate root/leaf SDF meshes.

Unlike per-item reference transfer, this path does not require intermediate
MC/DCSDD meshes.  Each final mesh is evaluated only by adaptations assigned
by the manifest-validated root/leaf plan; meshes are NEVER Boolean-unioned.
"""
from __future__ import annotations

import json
from pathlib import Path

import trimesh

from app.root_leaf_combination import export_unstitched_parts_ply


def load_final_root_leaf_meshes(*, plan, mc_parts_dir, root_path, require_manifest=True):
    directory = Path(mc_parts_dir).expanduser()
    if not directory.is_absolute():
        directory = Path(root_path) / directory
    directory = directory.resolve()
    missing = [part for part in plan.part_names if not (directory / f'{part}.ply').is_file()]
    if missing:
        raise FileNotFoundError('Missing final root/leaf mesh(s) ' + str(missing) + ' in ' + str(directory))

    manifest_path = directory.parent / 'root_leaves_ownership.json'
    if not manifest_path.is_file():
        if require_manifest:
            raise FileNotFoundError('Missing SDF ownership manifest: ' + str(manifest_path))
    else:
        stored = json.loads(manifest_path.read_text(encoding='utf-8'))
        expected = {
            'adaptation_fingerprint': plan.adaptation_fingerprint,
            'part_names': list(plan.part_names),
            'item_to_part': {str(k): v for k, v in sorted(plan.item_to_part.items())},
        }
        for key, value in expected.items():
            if stored.get(key) != value:
                raise ValueError(f'UDF/SDF final root/leaf {key} mismatch; use exactly the SDF adaptation list')

    result = {}
    for part in plan.part_names:
        mesh = trimesh.load(str(directory / f'{part}.ply'), process=False, force='mesh')
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            raise ValueError(f'Empty/non-triangle final root/leaf mesh: {directory / (part + ".ply")}')
        result[part] = mesh
    return directory, result


def export_direct_root_leaf_results(*, plan, source_dir, meshes, per_part_reports, output_folder):
    destination = Path(output_folder)
    out_dir = destination / 'riggable_parts_after_udf_cut'
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()
    report = {
        'mode': 'direct_udf_final_root_leaves',
        'input_dir': str(source_dir),
        'part_names': list(plan.part_names),
        'leaf_leaf_boolean': False,
        'connected_mesh': False,
        'attachment_stitching_done': False,
        'parts': {},
    }
    for part in plan.part_names:
        mesh = meshes[part]
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            raise RuntimeError(f'No faces remain in final UDF-cut part: {part}')
        part_path = out_dir / f'{part}.ply'
        mesh.export(part_path)
        scene.add_geometry(mesh, geom_name=part, node_name=part)
        report['parts'][part] = {
            **per_part_reports[part],
            'output': str(part_path),
        }
    one_file = export_unstitched_parts_ply(
        meshes, destination / 'riggable_after_udf_cut_UNSTITCHED_parts.ply', plan
    )
    report['single_file_unstitched'] = str(one_file)
    scene_path = destination / 'riggable_after_udf_cut.glb'
    scene.export(scene_path, file_type='glb')
    report['glb'] = str(scene_path)
    report_path = destination / 'riggable_after_udf_cut_report.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('[udf cut final root_leaves] saved', scene_path,
          'parts=', len(meshes), 'leaf_leaf_boolean=False', flush=True)
    return report


def curve_interval_owner_masks(mesh, owners, *, pad=0.035, samples=1025):
    """Assign each final face to one adaptation along its target curve.

    No source/intermediate mesh is needed.  A face is owned by the adaptation
    whose source interval contains its projected target-curve parameter.  At
    overlaps it belongs to the interval whose center is closest, normalised
    by interval width.  Faces outside every padded interval remain unowned.

    This is *not* exact legacy reference transfer; it restores localized UDF
    queries rather than all-owner score fusion and must be validated at 512.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    if not owners:
        raise ValueError('Curve-interval UDF ownership requires owners')
    pad = float(pad)
    if not np.isfinite(pad) or not (0 <= pad <= 0.25):
        raise ValueError('udf_cut_direct_interval_pad_0069 must be within [0, 0.25]')
    samples = int(samples)
    if not (32 <= samples <= 8192):
        raise ValueError('udf_cut_direct_curve_samples_0069 must be within [32, 8192]')

    target_keys = {str(owner['target_key']) for owner in owners}
    if len(target_keys) != 1:
        raise ValueError('Curve-interval UDF ownership requires one target curve per final part')
    curve = owners[0]['avatar_curve']
    core = curve.core
    ts = np.asarray(core.key_ts, dtype=np.float64).reshape(-1)
    if len(ts) < 2 or not np.all(np.isfinite(ts)) or not np.all(np.diff(ts) > 0):
        raise ValueError('Invalid target curve parameterization for UDF ownership')

    sample_ts = np.linspace(float(ts[0]), float(ts[-1]), samples)
    curve_points = np.asarray(core.interpolate(sample_ts)['points'], dtype=np.float64)
    if curve_points.shape != (samples, 3) or not np.all(np.isfinite(curve_points)):
        raise ValueError('Target curve interpolation failed for UDF ownership')

    face_centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    nearest_idx = np.asarray(cKDTree(curve_points).query(face_centers, k=1)[1], dtype=np.int64)
    # Refine the nearest sampled point against its two adjacent segments.
    # This remains lightweight: there is no UDF evaluation in ownership.
    param = np.empty(len(face_centers), dtype=np.float64)
    best_dist2 = np.full(len(face_centers), np.inf, dtype=np.float64)
    for start_idx in (np.clip(nearest_idx - 1, 0, samples - 2),
                      np.clip(nearest_idx, 0, samples - 2)):
        p0 = curve_points[start_idx]
        delta = curve_points[start_idx + 1] - p0
        length2 = np.einsum('ij,ij->i', delta, delta)
        fraction = np.divide(np.einsum('ij,ij->i', face_centers - p0, delta),
                             length2, out=np.zeros(len(face_centers)), where=length2 > 1e-20)
        fraction = np.clip(fraction, 0.0, 1.0)
        projected = p0 + fraction[:, None] * delta
        d2 = np.einsum('ij,ij->i', face_centers - projected, face_centers - projected)
        better = d2 < best_dist2
        param[better] = sample_ts[start_idx[better]] + fraction[better] * (
            sample_ts[start_idx[better] + 1] - sample_ts[start_idx[better]])
        best_dist2[better] = d2[better]

    intervals = []
    priority = np.full((len(owners), len(face_centers)), np.inf, dtype=np.float64)
    for i, owner in enumerate(owners):
        adapt = owner['adapt_arg']
        if 'src_0' not in adapt or 'src_1' not in adapt:
            raise ValueError(f"Missing src_0/src_1 for UDF owner {owner['item_index']}")
        a, b = float(adapt['src_0']), float(adapt['src_1'])
        if not np.isfinite(a + b) or abs(a - b) < 1e-9:
            raise ValueError(f"Invalid source interval for UDF owner {owner['item_index']}")
        lo, hi = sorted((a, b))
        lo = max(float(ts[0]), lo - pad)
        hi = min(float(ts[-1]), hi + pad)
        width = max(hi - lo, 1e-9)
        center = 0.5 * (lo + hi)
        eligible = (param >= lo) & (param <= hi)
        priority[i, eligible] = np.abs(param[eligible] - center) / width
        intervals.append((lo, hi))

    eligible_count = np.sum(np.isfinite(priority), axis=0)
    preferred = np.argmin(priority, axis=0)
    masks = [(preferred == i) & (eligible_count > 0) for i in range(len(owners))]
    summary = {
        'mode': 'curve_interval',
        'assigned_faces': int(np.count_nonzero(eligible_count)),
        'unassigned_faces': int(np.count_nonzero(eligible_count == 0)),
        'overlap_faces': int(np.count_nonzero(eligible_count > 1)),
        'interval_pad': pad,
        'curve_samples': samples,
        'owner_assigned': {str(o['item_index']): int(mask.sum()) for o, mask in zip(owners, masks)},
        'owner_intervals': {str(o['item_index']): list(bounds) for o, bounds in zip(owners, intervals)},
    }
    return masks, summary
