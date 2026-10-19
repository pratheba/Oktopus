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
    """Fail closed for options not faithfully handled by the sparse path."""
    if len(items) != 2 or not bool(cfg.get('combine_adaptations', False)):
        raise ValueError('0046 cone streaming requires exactly two combined adaptations')
    if root_leaf_plan is not None:
        raise ValueError('0046 cone streaming does not support root_leaves yet')
    if bool(cfg.get('export_individual_meshes', True)):
        raise ValueError('0046 cone streaming requires export_individual_meshes: false')
    if not (cfg.get('dcsdd_crop_grid', False)
            and cfg.get('sdf_localize_after_gate', False)
            and cfg.get('sdf_optimize_zero_angular_wrap', False)):
        raise ValueError('0046 requires crop-grid, localization-after-gate and zero-angular-wrap optimization')
    if cfg.get('dcsdd_stream_inference', False) or cfg.get('dcsdd_stream_one_pass', False):
        raise ValueError('Use --sdf-stream-combined-wrap alone; do not enable rigid-boot streaming flags')
    if cfg.get('dcsdd_stream_crop_first', False) or cfg.get('dcsdd_stream_replay_metadata', False):
        raise ValueError('0046 combined mode handles replay and cropping internally')
    if query_seams:
        raise ValueError('0046 cone streaming does not support query-coordinate seam warps')
    if len(post_seams) != 1 or 1 not in post_seams:
        raise ValueError('0046 requires the cone post-SDF seam between items 0 and 1')
    groups = {str(item.get('blend_group')) for item in items}
    if len(groups) != 1 or post_seams[1].group not in groups:
        raise ValueError('Both cone adaptations must belong to the same seam group')
    if items[0]['target_key'] != items[1]['target_key']:
        raise ValueError('0046 cannot combine different target curves')
    for i, item in enumerate(items):
        if item.get('mode', 'direct') != 'direct' or not item.get('wrap_radius', False) or item.get('rigid_radius', False):
            raise ValueError(f'0046 item {i}: only direct wrap_radius is supported')
        if float(item.get('wrap_angular_follow', 1.0)) != 0.0:
            raise ValueError(f'0046 item {i}: requires wrap_angular_follow: 0')
        if not item.get('use_accessory_support_clamp', False):
            raise ValueError(f'0046 item {i}: requires accessory support clamp')
        if str(item.get('snug_mode', 'multiplicative')).lower() != 'multiplicative':
            raise ValueError(f'0046 item {i}: additive snug needs the legacy path')
        if item.get('auto_avatar_snug_field', False):
            from app.sdf_zero_angle_optimization import can_skip_inert_snug
            if not can_skip_inert_snug(item):
                raise ValueError(f'0046 item {i}: auto snug has an active effect')
        prohibited = ('cut_avatar', 'use_tiled_detail', 'detail_avatar_gate',
                      'final_carve', 'use_adaptive_shell_thinning',
                      'use_attach_to_free_blend', 'wrap_to_rigid',
                      'wrap_blend_rigid', 'unified_wrap', 'use_free_profile_scale',
                      'use_adapt_train_radius')
        for key in prohibited:
            if item.get(key, False):
                raise ValueError(f'0046 item {i}: unsupported feature {key}')
        if str(item.get('accessory_offset_mode', 'none')).lower() not in ('none', 'off', 'false'):
            raise ValueError(f'0046 item {i}: unsupported accessory_offset_mode')
        if float(item.get('accessory_offset', 0.0)):
            raise ValueError(f'0046 item {i}: nonzero accessory offset')
        if item.get('avatar_snug_scale_field') is not None:
            raise ValueError(f'0046 item {i}: externally supplied snug field')
    return post_seams[1]


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
