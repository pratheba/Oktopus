"""Source-interval gate envelope for audited broad-phase reduction.

The exact localizer evaluates an ellipsoid with three radii, including an
along-tangent radius at the interval endpoints. Merely cropping the broad-
phase cylinder to the source interval can discard accepted voxels.

This module clips the ORIGINAL full-cylinder bounding-box candidate union to
an expanded world-space AABB of the localizer's acceptance ellipsoids.
The continuous parameter envelope is sampled numerically, so an audit of
excluded candidates remains mandatory for each configuration before a
non-audited production run.
"""
import numpy as np


def localizer_gate_envelope(core, src_0, src_1, adapt_arg, mc_grid,
                            *, n_samples=401):
    """Return (lower_world, upper_world) of the sampled interval gate envelope.

    The center, frame, cylinder radii and along-tangent radius are taken from
    exactly the corresponding operations in localize_samples(). Its acceptance
    gate is ||diag(1/radii) @ frame @ (point-center)||_2 <= norm.
    """
    lo, hi = sorted((float(src_0), float(src_1)))
    if not (0 <= lo < hi <= 1) or n_samples < 3:
        raise ValueError('Cannot build valid 0048 interval gate envelope')
    s = np.linspace(lo, hi, n_samples, dtype=np.float64)
    key_ts = np.asarray(getattr(core, 'key_ts', []), dtype=np.float64)
    if key_ts.size:
        s = np.unique(np.concatenate((s, key_ts[(key_ts >= lo) & (key_ts <= hi)])))
    info = core.interpolate(s, radius_type='cylinder', radius=False, frame=True)
    centers = np.asarray(info['points'], dtype=np.float64)
    frames = np.asarray(info['frame'], dtype=np.float64)
    cylinder_radii = np.asarray(core.interpolate(
        s, points=False, frame=False, radius=True, radius_type='cylinder')['radius'],
        dtype=np.float64)
    runtime_scale = float(adapt_arg.get('avatar_cylinder_radius_scale', 1.0))
    runtime_add = float(adapt_arg.get('avatar_cylinder_radius_add', 0.0))
    cylinder_radii = cylinder_radii * runtime_scale + runtime_add
    x_radius = np.asarray(core.calc_x_radius(s), dtype=np.float64).reshape(-1)
    radii = np.column_stack((x_radius, cylinder_radii))
    norm = float(adapt_arg.get('adapt_localize_norm', 1.0))
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError('Invalid localizer norm for 0048 interval envelope')
    if (centers.shape != (len(s), 3) or frames.shape != (len(s), 3, 3)
            or radii.shape != (len(s), 3) or not np.all(np.isfinite(centers))
            or not np.all(np.isfinite(frames)) or not np.all(np.isfinite(radii))
            or np.any(radii <= 0)):
        raise ValueError('Invalid source-interval geometry for 0048 envelope')
    # Unlike a sliced cylinder, this includes the along-tangent endpoint cap.
    # The frame may not be exactly orthonormal, so invert it instead of using
    # its transpose to compute an ellipsoid's WORLD-space coordinate extents.
    inv_frame = np.linalg.inv(frames)
    ellipsoid_extent = norm * np.linalg.norm(inv_frame * radii[:, None, :], axis=2)
    lower_sample = centers - ellipsoid_extent
    upper_sample = centers + ellipsoid_extent
    # Numerical sampling rather than a symbolic PCHIP interval bound. Add a
    # small conservative guard from adjacent sample variation and grid spacing.
    # The exact excluded-candidate audit is still the correctness authority.
    variation = max(
        float(np.max(np.abs(np.diff(lower_sample, axis=0)))),
        float(np.max(np.abs(np.diff(upper_sample, axis=0)))),
    )
    step = float(mc_grid.grid_config['step'])
    guard = 2 * variation + 2 * step
    return lower_sample.min(axis=0) - guard, upper_sample.max(axis=0) + guard


class GateEnvelopeCandidates:
    """Original cylinder's bbox-union candidates intersected with gate AABB."""

    def __init__(self, original_cylinder, lower, upper):
        self.original_cylinder = original_cylinder
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)

    def iter_filter_grid_indices(self, mc_grid, *, batch_size=262144,
                                 extend_bbox=False):
        from .bounded_filter import iter_bbox_kidx
        return iter_bbox_kidx(
            self.original_cylinder.cylinder_samples, mc_grid,
            extend_bbox=extend_bbox, batch_size=batch_size,
            clip_world_aabb=(self.lower, self.upper))
