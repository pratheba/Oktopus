"""0046 opt-in, one-pass sparse SDF composition for a two-piece wrap cone.

Do not reuse the independent rigid-boot fast path for an arbitrary wrap or
blend group: composition must retain the post-SDF overlap crossfade and the
GLOBAL support-profile calibration for each individual piece.
"""
from pathlib import Path
import numpy as np

from app.dcsdd_stream_replay import compose_replay_chunk
from post_sdf_seam_blend import overlap_crossfade_sdf


def validate_cone_stream(items, cfg, post_seams, query_seams, root_leaf_plan):
    """Validate a generic *single-part* multi-section direct wrap stream.

    The original name is retained for existing callers/tests. Eligibility is
    geometric, never based on cone/boot/dress/puffer identities. Unsupported
    blend graphs, active snug fields or per-adaptation Boolean cuts fail closed.
    """
    if len(items) < 2 or not bool(cfg.get('combine_adaptations', False)):
        raise ValueError('Wrap streaming requires 2+ combined adaptations')
    if root_leaf_plan is not None:
        if (len(root_leaf_plan.part_names) != 1
                or any(root_leaf_plan.item_to_part.get(i) != root_leaf_plan.part_names[0]
                       for i in range(len(items)))):
            raise ValueError('Wrap streaming only supports ONE YAML-defined output part; use the standard pipeline for multiple parts')
    if bool(cfg.get('export_individual_meshes', False)) and not bool(cfg.get('sdf_stream_export_sections_0093', False)):
        raise ValueError('Wrap streaming: use sdf_stream_export_sections_0093 for separate section PLYs')
    if not (cfg.get('dcsdd_crop_grid', False)
            and cfg.get('sdf_localize_after_gate', False)
            and cfg.get('sdf_optimize_zero_angular_wrap', False)):
        raise ValueError('Wrap streaming requires crop-grid, localize-after-gate and guarded zero-angular optimization')
    if any(cfg.get(k, False) for k in ('dcsdd_stream_inference', 'dcsdd_stream_one_pass',
                                       'dcsdd_stream_crop_first', 'dcsdd_stream_replay_metadata')):
        raise ValueError('Wrap streaming uses its own callback/replay; disable independent boot streaming flags')
    if query_seams:
        raise ValueError('Wrap streaming does not support query-coordinate seam warps')
    groups = {str(item.get('blend_group')) for item in items}
    if len(groups) != 1 or next(iter(groups)) in ('None', 'none', ''):
        raise ValueError('Wrap sections must share one named blend group')
    group = next(iter(groups))
    if set(post_seams) != set(range(1, len(items))) or any(
            post_seams[j].group != group for j in range(1, len(items))):
        raise ValueError('Wrap streaming requires a contiguous post-SDF seam chain for this group')
    if len({item['target_key'] for item in items}) != 1:
        raise ValueError('Wrap streaming requires all sections on the same target curve')
    for i, item in enumerate(items):
        if item.get('mode', 'direct') != 'direct' or not item.get('wrap_radius', False) or item.get('rigid_radius', False):
            raise ValueError(f'Wrap section {i}: requires direct wrap_radius without rigid_radius')
        if float(item.get('wrap_angular_follow', 1.0)) != 0.0:
            raise ValueError(f'Wrap section {i}: nonzero angular follow uses standard pipeline')
        if not item.get('use_accessory_support_clamp', False):
            raise ValueError(f'Wrap section {i}: requires accessory support clamp')
        if str(item.get('snug_mode', 'multiplicative')).lower() != 'multiplicative':
            raise ValueError(f'Wrap section {i}: additive snug uses standard pipeline')
        if item.get('auto_avatar_snug_field', False):
            from app.sdf_zero_angle_optimization import can_skip_inert_snug
            if not can_skip_inert_snug(item):
                raise ValueError(f'Wrap section {i}: auto snug affects geometry')
        forbidden = ('cut_avatar', 'use_tiled_detail', 'detail_avatar_gate',
                     'final_carve', 'use_adaptive_shell_thinning',
                     'use_attach_to_free_blend', 'wrap_to_rigid',
                     'wrap_blend_rigid', 'unified_wrap', 'use_free_profile_scale',
                     'use_adapt_train_radius')
        for key in forbidden:
            if item.get(key, False):
                raise ValueError(f'Wrap section {i}: unsupported {key}; use post-blend Boolean YAML instead of cut_avatar')
        if str(item.get('accessory_offset_mode', 'none')).lower() not in ('none', 'off', 'false'):
            raise ValueError(f'Wrap section {i}: unsupported accessory offset mode')
        if float(item.get('accessory_offset', 0.0)):
            raise ValueError(f'Wrap section {i}: unsupported accessory offset')
        if item.get('avatar_snug_scale_field') is not None:
            raise ValueError(f'Wrap section {i}: external snug field unsupported')
    return post_seams[1] if len(items) == 2 else post_seams


def merge_sparse_avatar_field(left_idx, left_sdf, right_idx, right_sdf, *, tolerance=0.005):
    """Combine sampled avatar SDF on the same target, verifying overlap parity."""
    a = np.asarray(left_idx, dtype=np.int64).reshape(-1)
    b = np.asarray(right_idx, dtype=np.int64).reshape(-1)
    av = np.asarray(left_sdf, dtype=np.float64).reshape(-1)
    bv = np.asarray(right_sdf, dtype=np.float64).reshape(-1)
    if len(a) != len(av) or len(b) != len(bv):
        raise ValueError('Avatar SDF indices/values mismatch')
    if (np.any(a[1:] <= a[:-1]) or np.any(b[1:] <= b[:-1])
            or np.any(a < 0) or np.any(b < 0)
            or not np.all(np.isfinite(av)) or not np.all(np.isfinite(bv))):
        raise ValueError('Invalid avatar SDF index/value stream')
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Avatar SDF tolerance must be finite and nonnegative')
    indices = np.union1d(a, b)
    values = np.empty(len(indices), dtype=np.float64)
    if len(a):
        values[np.searchsorted(indices, a)] = av
    if len(b):
        positions = np.searchsorted(indices, b)
        i = np.searchsorted(a, b)
        overlap = np.zeros(len(b), dtype=bool)
        valid = i < len(a)
        overlap[valid] = (a[i[valid]] == b[valid])
        if np.any(overlap):
            previous = values[positions[overlap]]
            current = bv[overlap]
            mismatch = np.abs(previous - current)
            if np.any(mismatch > tolerance):
                raise ValueError('Post-blend avatar SDF disagrees between overlapping wrap sections: '
                                 f'max_difference={float(mismatch.max()):.6g} tolerance={tolerance}')
            values[positions[overlap]] = np.minimum(previous, current)
        values[positions[~overlap]] = bv[~overlap]
    return indices, values


def apply_postblend_avatar_boolean(indices, blended_sdf, avatar_indices,
                                   avatar_sdf, *, clearance=0.0):
    """Scalar SDF difference: max(blended accessory, clearance - avatar)."""
    if not np.isfinite(clearance) or clearance < 0:
        raise ValueError('Boolean clearance must be nonnegative and finite')
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    avatar_indices = np.asarray(avatar_indices, dtype=np.int64).reshape(-1)
    values = np.asarray(blended_sdf, dtype=np.float64).reshape(-1)
    av = np.asarray(avatar_sdf, dtype=np.float64).reshape(-1)
    if (len(indices) != len(values) or len(avatar_indices) != len(av)
            or not np.array_equal(indices, avatar_indices)):
        raise ValueError('Avatar support must cover exactly the blended voxels')
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(av)):
        raise ValueError('Nonfinite Boolean operands')
    return np.maximum(values, clearance - av)


def combine_two_sparse_parts(left_idx, left_values, right_idx, right_values,
                             right_source_s, seam):
    """Exact counterpart of the dense group accumulator's assignment rules.

    New-only samples take their own value. Shared voxels are crossfaded using
    the *second piece's* avatar s parameter, matching the legacy code.
    The result contains ONLY evaluated voxels in global x-fast index order.
    """
    a = np.asarray(left_idx, dtype=np.int64).reshape(-1)
    b = np.asarray(right_idx, dtype=np.int64).reshape(-1)
    va = np.asarray(left_values).reshape(-1)
    vb = np.asarray(right_values).reshape(-1)
    source = np.asarray(right_source_s).reshape(-1)
    if len(a) != len(va) or len(b) != len(vb) or len(b) != len(source):
        raise ValueError('Sparse voxel/value/source-s array length mismatch')
    if a.size and (np.any(a[1:] <= a[:-1]) or a[0] < 0):
        raise ValueError('Left voxel indices must be strictly increasing')
    if b.size and (np.any(b[1:] <= b[:-1]) or b[0] < 0):
        raise ValueError('Right voxel indices must be strictly increasing')
    indices = np.union1d(a, b)
    values = np.empty(len(indices), dtype=np.result_type(va.dtype, vb.dtype))
    if len(a):
        il = np.searchsorted(indices, a)
        values[il] = va
    if len(b):
        ir = np.searchsorted(indices, b)
        ia = np.searchsorted(a, b)
        overlap = (ia < len(a))
        if len(a):
            overlap &= a[np.minimum(ia, len(a) - 1)] == b
        else:
            overlap[:] = False
        values[ir[~overlap]] = vb[~overlap]
        if np.any(overlap):
            values[ir[overlap]] = overlap_crossfade_sdf(
                values[ir[overlap]], vb[overlap], source[overlap], seam)
    return indices, values, int(np.count_nonzero(overlap)) if len(b) else 0


def compose_piece_from_replay(writer, profiles, *, outdir, sigma, rho_limit=1.,
                              positive_value=1., chunk_size=262144, label=''):
    """Replay after global calibration, never localize a second time."""
    replay = writer.replay
    n = int(writer.accepted)
    path = Path(outdir) / 'final_sdf.bin'
    result = np.memmap(path, dtype=np.float64, mode='w+', shape=(n,))
    bad_sum = 0
    min_rho, max_rho = np.inf, -np.inf
    for start, end, records in replay.iter_chunks(chunk_size):
        raw = {key: value[start:end] for key, value in writer.raw.items()}
        vals, bad, (rho_min, rho_max) = compose_replay_chunk(
            raw, records['coord'], records['rho'], profiles,
            rho_limit=rho_limit, positive_value=positive_value, sigma=sigma)
        result[start:end] = vals
        bad_sum += bad
        min_rho = min(min_rho, rho_min)
        max_rho = max(max_rho, rho_max)
    result.flush()
    print(f'[sdf_stream_combined_wrap] {label} clamped_bad={bad_sum}/{n} '
          f'rho=[{min_rho:.3f},{max_rho:.3f}]', flush=True)
    return result
