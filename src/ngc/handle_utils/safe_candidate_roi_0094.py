"""0094: certified broad-phase AABB union for the stock PWLA localizer.

This *only* clips the old cylinder-box candidate union, not the localizer or
SDF values. The stock PWLA center and radius interpolation are linear between
key_ts, and calc_x_radius is exactly 1.0. Per-key-interval boxes enclose its
ellipsoid gate, including the tangent/end-cap radius. They are independent of
rigid_radius/wrap_radius (which act downstream of the avatar gate).

Fail open to the old broad phase when these assumptions cannot be verified.
An optional 0048 excluded-voxel audit remains available for independent checks.
"""
import numpy as np


def certified_source_boxes(core, src_0, src_1, adapt_arg, mc_grid):
    """Return conservative [(lower, upper)] per source key interval or None.

    Uses a coarse analytic bound on the Gram-Schmidt frame interpolation:
    - T is normalized linear interpolation of its key vectors;
    - N is projected perpendicular to T, then normalized;
    - B = normalized cross(T, N).
    If the keys rotate too much, use the orientation-free enclosing sphere.
    No sampling of the continuous frame is used as a proof of containment.
    """
    from PWLA_curve_handle import PWLACurve
    if (type(core) is not PWLACurve
            or 'interpolate' in vars(core)
            or 'calc_x_radius' in vars(core)
            or '_interp_frames' in vars(core)):
        return None
    try:
        ts = np.asarray(core.key_ts, dtype=np.float64)
        pts = np.asarray(core.key_points, dtype=np.float64)
        radii = np.asarray(core.key_cylinder_radius, dtype=np.float64)
        frames = np.asarray(core.key_frame, dtype=np.float64)
        a, b = sorted((float(src_0), float(src_1)))
        norm = float(adapt_arg.get('adapt_localize_norm', 1.0))
        scale = float(adapt_arg.get('avatar_cylinder_radius_scale', 1.0))
        add = float(adapt_arg.get('avatar_cylinder_radius_add', 0.0))
        step = float(mc_grid.grid_config['step'])
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    if (ts.ndim != 1 or len(ts) < 2 or pts.shape != (len(ts), 3)
            or radii.shape != (len(ts), 2) or frames.shape != (len(ts), 3, 3)
            or not all(np.all(np.isfinite(x)) for x in (ts, pts, radii, frames))
            or not np.all(np.diff(ts) > 0)
            or not (ts[0] <= a < b <= ts[-1])
            or not np.isfinite([a, b, norm, scale, add, step]).all()
            or norm <= 0 or step <= 0 or scale < 0):
        return None
    # The interpolator's input frames must be proper near-orthonormal bases.
    # If not, do not assume a Euclidean ellipsoid bound for the localizer.
    gram = frames @ np.transpose(frames, (0, 2, 1))
    if not np.allclose(gram, np.eye(3), atol=1e-5, rtol=0):
        return None
    boxes = []
    borders = np.unique(np.concatenate(([a, b], ts[(ts > a) & (ts < b)])))
    for s0, s1 in zip(borders[:-1], borders[1:]):
        centers = np.column_stack([
            np.interp([s0, s1], ts, pts[:, j]) for j in range(3)])
        radius_ends = np.column_stack([
            np.interp([s0, s1], ts, radii[:, j]) for j in range(2)])
        radius_ends = radius_ends * scale + add
        if np.any(radius_ends <= 0) or not np.isfinite(radius_ends).all():
            return None
        # Exact x_radius=1 for the stock PWLACurve; other two radius
        # components and centers are linear and attain extrema at endpoints.
        rmax = np.r_[1.0, radius_ends.max(axis=0)]
        # Choose the two key frames bracketing this source subinterval.
        key_idx = min(int(np.searchsorted(ts, (s0 + s1) / 2.0,
                                          side='right')) - 1, len(ts) - 2)
        f0, f1 = frames[key_idx:key_idx+2]
        delta = float(np.max(np.linalg.norm(f1-f0, axis=1)))
        # Bound the deviations of the orthonormalized frame rows from f0.
        # When this would be loose/ill-conditioned, use the guaranteed sphere.
        if delta < 0.055:
            t_err = 2*delta / (1-delta)
            n_raw_err = delta + (1+delta)*(delta+t_err)
            n_err = 2*n_raw_err/(1-n_raw_err) if n_raw_err < 1 else np.inf
            b_err = t_err + n_err + 1e-8
            errors = np.array([t_err, n_err, b_err])
            row_abs = np.minimum(1.0, np.abs(f0) + errors[:, None])
            # Orthonormal world-to-local frames imply inverse=transpose.
            # Extra 5% covers the 1e-12 normalization stabilizer in _interp_frames.
            extents = 1.05 * norm * np.sqrt(np.sum(
                (row_abs * rmax[:, None])**2, axis=0))
        else:
            extents = np.full(3, 1.05 * norm * float(rmax.max()))
        # AABB-to-voxel index conversion also adds two cells in bounded_filter.
        extents += 2.0*step + 1e-9
        boxes.append((centers.min(axis=0)-extents,
                      centers.max(axis=0)+extents))
    return boxes if boxes else None
