"""Group-local surface seam fitting for rigid-radius accessory adaptations.

The fit is deliberately low-cost: it uses the accessory curve plus a bounded
sample of accessory-owned surface points and modifies model queries before
inference.  It does not deform extracted meshes and does not allocate another
full reconstruction grid.

For each adjoining pair in a blend group we fit both C0 and C1 behavior:

* transverse center position and center tangent,
* outer cross-section scale and longitudinal taper tangent.

The surface-profile fit is still constrained to one isotropic radial scale per
section.  If two sections have genuinely different non-circular profiles, the
remaining profile residual is reported rather than hidden by a larger blend.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class SeamCorrection:
    source_s: float
    half_width: float
    log_radius_scale: float
    log_radius_slope: float
    # Additional physical translation in accessory-local [U, V] units.
    position_uv: tuple = (0.0, 0.0)
    # d(tloc[U,V]) / d(source_s), also in accessory-local physical units.
    position_slope_uv: tuple = (0.0, 0.0)


def _positive_radius(curve, s):
    """Representative cross-sectional radius from a sampled accessory curve."""
    info = curve.core.interpolate(np.array([s], dtype=np.float64))
    r = np.asarray(info["radius"], dtype=np.float64).reshape(-1)
    r = r[np.isfinite(r) & (r > 0.0)]
    if r.size == 0:
        raise ValueError(f"Invalid accessory curve radius at s={s}")
    return float(np.exp(np.mean(np.log(r))))


def _mapped_s(item, source_s):
    src0, src1 = float(item["src_0"]), float(item["src_1"])
    if abs(src1 - src0) <= 1e-9:
        raise ValueError(f"Zero-length source interval for {item.get('name', item['target_key'])}")
    u = (source_s - src0) / (src1 - src0)
    return float(item["tgt_0"]) + u * (float(item["tgt_1"]) - float(item["tgt_0"]))


def _log_radius(item, curve, source_s):
    """Log physical output-radius proxy for the rigid-radius mapping.

    _localize.py uses ``rho_acc = rho_avatar * global_scale``.  Therefore a
    model-space accessory radius R appears on the avatar with physical radius
    R / global_scale (before this seam correction).  This inverse relationship
    is easy to accidentally reverse when fitting two adaptations.
    """
    scale = float(item.get("scale", 1.0))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Expected positive finite scale, got {scale}")
    return math.log(_positive_radius(curve, _mapped_s(item, source_s))) - math.log(scale)


def _radius_slope(item, curve, source_s, step):
    lo, hi = sorted((float(item["src_0"]), float(item["src_1"])))
    left, right = max(lo, source_s - step), min(hi, source_s + step)
    if right - left <= 1e-9:
        return 0.0
    return (_log_radius(item, curve, right) - _log_radius(item, curve, left)) / (right - left)


def _surface_profile(curve):
    """Project a bounded sample of accessory-owned surface points once."""
    core = curve.core
    points = getattr(core, "surface_points_owned", None)
    if points is None:
        raise ValueError("Surface seam fitting needs accessory surface_points_owned")
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 32:
        raise ValueError("Surface seam fitting needs at least 32 owned XYZ surface points")
    # Subsample BEFORE converting to float64 to bound temporary memory.
    points = np.asarray(
        points[::max(1, math.ceil(len(points) / 20000))], dtype=np.float64
    )
    s = np.asarray(core.curve_projection(points), dtype=np.float64).reshape(-1)
    valid = np.isfinite(s) & (s >= 0.0) & (s <= 1.0)
    points, s = points[valid], s[valid]
    if len(s) < 32:
        raise ValueError("Insufficient valid owned surface points for seam fitting")
    support = core.interpolate(s, radius=False, frame=True)
    local = np.einsum("nij,nj->ni", support["frame"], points - support["points"])
    return s, local[:, 1:3]


def _cross_section_points(profile, target_s, half_window):
    """Return a robust local UV ring and its center near one accessory section."""
    s, uv = profile
    distance = np.abs(s - target_s)
    selected = distance <= half_window
    if np.count_nonzero(selected) < 32:
        selected = distance <= min(0.08, 2.5 * half_window)
    if np.count_nonzero(selected) < 32:
        raise ValueError(
            f"Not enough surface points near accessory s={target_s:.4f}; "
            "cannot estimate seam geometry reliably"
        )
    ring_uv = uv[selected]
    center = 0.5 * (
        np.quantile(ring_uv, 0.10, axis=0) + np.quantile(ring_uv, 0.90, axis=0)
    )
    if not np.all(np.isfinite(center)):
        raise ValueError("Non-finite accessory surface center")
    return ring_uv, center


def _cross_section_center(profile, target_s, half_window):
    return _cross_section_points(profile, target_s, half_window)[1]


def _rotate_uv(vector, degrees):
    value = np.asarray(vector, dtype=np.float64)
    theta = math.radians(degrees)
    c, si = math.cos(theta), math.sin(theta)
    out = np.empty_like(value, dtype=np.float64)
    out[..., 0] = c * value[..., 0] - si * value[..., 1]
    out[..., 1] = si * value[..., 0] + c * value[..., 1]
    return out


def _manual_tloc_uv(item):
    value = np.asarray(
        item.get("tloc", item.get("translate_local", (0, 0, 0))), dtype=np.float64
    )
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"Invalid tloc in {item.get('name', item.get('target_key'))}")
    return value[1:3]


def _surface_log_radius_profile(
    item,
    profile,
    source_s,
    half_window,
    n_bins,
    radius_quantile,
):
    """Outer surface profile in the common avatar-local transverse frame.

    The returned vector is log radius in angular bins.  Rotation is applied
    before binning so two differently rotated adaptations are compared in the
    same frame.  Rigid-radius ``scale`` is inverse in physical output space.
    """
    target_s = _mapped_s(item, source_s)
    ring_uv, center = _cross_section_points(profile, target_s, half_window)
    centered = ring_uv - center
    scale = float(item.get("scale", 1.0))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Surface radius fitting needs positive finite scale")
    common = _rotate_uv(centered / scale, -float(item.get("rot_deg", 0.0)))
    radius = np.linalg.norm(common, axis=1)
    angle = np.mod(np.arctan2(common[:, 1], common[:, 0]), 2.0 * np.pi)
    valid = np.isfinite(radius) & (radius > 1e-12) & np.isfinite(angle)
    radius, angle = radius[valid], angle[valid]
    if len(radius) < 32:
        raise ValueError("Insufficient finite surface radii for seam fitting")

    bin_index = np.minimum((angle / (2.0 * np.pi) * n_bins).astype(int), n_bins - 1)
    values = np.full(n_bins, np.nan, dtype=np.float64)
    for index in range(n_bins):
        samples = radius[bin_index == index]
        if len(samples):
            values[index] = np.quantile(samples, radius_quantile)
    present = np.isfinite(values)
    if np.count_nonzero(present) < max(6, n_bins // 3):
        raise ValueError("Surface ring does not cover enough angular bins")
    # Circularly interpolate sparse bins rather than changing angular alignment.
    x = np.flatnonzero(present).astype(np.float64)
    y = values[present]
    query = np.arange(n_bins, dtype=np.float64)
    values = np.interp(query, np.concatenate((x - n_bins, x, x + n_bins)),
                       np.concatenate((y, y, y)))
    return np.log(np.maximum(values, 1e-12))


def _surface_radius_slope(
    item,
    profile,
    source_s,
    step,
    half_window,
    n_bins,
    radius_quantile,
):
    lo, hi = sorted((float(item["src_0"]), float(item["src_1"])))
    left, right = max(lo, source_s - step), min(hi, source_s + step)
    if right - left <= 1e-9:
        return np.zeros(n_bins, dtype=np.float64)
    profile_l = _surface_log_radius_profile(
        item, profile, left, half_window, n_bins, radius_quantile
    )
    profile_r = _surface_log_radius_profile(
        item, profile, right, half_window, n_bins, radius_quantile
    )
    return (profile_r - profile_l) / (right - left)


def _position_center(item, curve, profile, source_s, radius_factor, half_window):
    """Approximate fitted cross-section center in the common avatar UV frame."""
    target_s = _mapped_s(item, source_s)
    center = _cross_section_center(profile, target_s, half_window)
    # localize_samples_adapt optionally subtracts a surface-center field before
    # applying tloc. Match that convention when it is enabled in the YAML.
    model_center = np.zeros(2)
    if bool(item.get("use_tgt_runtime_uv_center", False)):
        field = curve.core.build_runtime_uv_center_field(
            n_bins=int(item.get("uv_center_n_bins", 64)),
            source=item.get("uv_center_source", "owned"),
            min_count=int(item.get("uv_center_min_count", 20)),
            smooth_s=float(item.get("uv_center_smooth_s", 2.0)),
            robust=item.get("uv_center_robust", "median"),
        )
        if field is None:
            raise ValueError("Unable to estimate the enabled runtime UV center field")
        model_center = curve.core.interpolate_uv_center_field(field, [target_s])[0]
    scale = float(item.get("scale", 1.0))
    # Existing radius warp divides both model-local U and V by radius_factor.
    # A model-space center C therefore appears at radius_factor*C in the
    # original local coordinate system. The tloc sign follows _localize.py.
    effective_center = (
        radius_factor * center - model_center + _manual_tloc_uv(item)
    ) / scale
    return _rotate_uv(effective_center, -float(item.get("rot_deg", 0.0)))


def _smooth_fade(source_s, seam_s, half_width):
    x = (np.asarray(source_s, dtype=np.float64) - seam_s) / half_width
    t = np.minimum(np.abs(x), 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


def _single_radius_scale(source_s, seam_s, half_width, value, slope):
    ds = float(source_s) - seam_s
    fade = float(_smooth_fade([source_s], seam_s, half_width)[0])
    return math.exp((value + slope * ds) * fade)


def _position_slope(
    item,
    curve,
    profile,
    source_s,
    value_correction,
    slope_correction,
    correction_width,
    half_window,
    step,
):
    """Finite-difference cross-section center tangent in source coordinates."""
    lo, hi = sorted((float(item["src_0"]), float(item["src_1"])))
    left, right = max(lo, source_s - step), min(hi, source_s + step)
    if right - left <= 1e-9:
        return np.zeros(2, dtype=np.float64)
    factor_l = _single_radius_scale(
        left, source_s, correction_width, value_correction, slope_correction
    )
    factor_r = _single_radius_scale(
        right, source_s, correction_width, value_correction, slope_correction
    )
    center_l = _position_center(item, curve, profile, left, factor_l, half_window)
    center_r = _position_center(item, curve, profile, right, factor_r, half_window)
    return (center_r - center_l) / (right - left)


def _bound_translation(uv, max_change):
    uv = np.asarray(uv, dtype=np.float64)
    length = float(np.linalg.norm(uv))
    if length > max_change:
        return uv * (max_change / length)
    return uv


def _bound_radius_slope(value, slope, half_width, lower, upper):
    """Conservatively keep value+slope*ds inside the configured radius limit."""
    if half_width <= 1e-12:
        return 0.0
    available = max(0.0, min(upper - value, value - lower))
    max_abs = available / half_width
    return float(np.clip(slope, -max_abs, max_abs))


def build_seam_plan(adaptations, group_settings, curve_lookup):
    """Return correction lists indexed by expanded adaptation index.

    Each enabled group is isolated; no source-curve state is mutated or reused.
    Only touching/overlapping intervals of the same avatar target are paired.
    """
    if not isinstance(group_settings, dict):
        raise TypeError("blend_groups must be a mapping")
    enabled = {
        str(name): spec["seam_optimization"]
        for name, spec in group_settings.items()
        if isinstance(spec, dict)
        and isinstance(spec.get("seam_optimization"), dict)
        and spec["seam_optimization"].get("enabled", False)
    }
    by_group = {}
    for index, item in enumerate(adaptations):
        group = str(item.get("blend_group", "none"))
        if group in enabled:
            by_group.setdefault(group, []).append((index, item))
    missing = set(enabled) - set(by_group)
    if missing:
        raise ValueError(f"Configured seam blend groups without adaptations: {sorted(missing)}")

    result = {}
    for group_name, spec in enabled.items():
        do_radius = bool(spec.get("optimize_radius", True))
        do_position = bool(spec.get("optimize_position", False))
        match_radius_tangent = bool(spec.get("match_surface_tangent", True))
        match_position_tangent = bool(spec.get("match_position_tangent", True))
        if not do_radius and not do_position:
            continue

        max_change = float(spec.get("max_radius_change", 0.20))
        max_position = float(spec.get("max_position_change", 0.05))
        transition = float(spec.get("transition_fraction", 0.20))
        sample_window = float(spec.get("position_sample_window", 0.025))
        surface_window = float(spec.get("surface_sample_window", sample_window))
        tangent_step = float(spec.get("tangent_sample_step", 0.005))
        surface_bins = int(spec.get("surface_profile_bins", 24))
        radius_quantile = float(spec.get("surface_radius_quantile", 0.85))
        radius_source = str(spec.get("radius_source", "surface")).lower()
        max_gap = float(spec.get("max_gap", 0.0))

        if not (
            0.0 < max_change < 1.0
            and 0.0 < transition <= 0.5
            and max_gap >= 0.0
        ):
            raise ValueError(f"Invalid seam_optimization parameters in {group_name}")
        if do_position and not (0.0 < max_position and 0.0 < sample_window <= 0.08):
            raise ValueError(f"Invalid positional optimization settings in {group_name}")
        if radius_source not in {"surface", "curve"}:
            raise ValueError(f"radius_source must be 'surface' or 'curve' in {group_name}")
        if not (0.0 < surface_window <= 0.08 and tangent_step > 0.0):
            raise ValueError(f"Invalid seam sampling settings in {group_name}")
        if surface_bins < 8 or not (0.5 <= radius_quantile < 1.0):
            raise ValueError(f"Invalid surface profile settings in {group_name}")

        group_items = by_group[group_name]
        if len(group_items) < 2:
            raise ValueError(
                f"Blend group {group_name!r} needs at least two pieces for seam optimization"
            )
        targets = {item["target_key"] for _, item in group_items}
        if len(targets) != 1:
            raise ValueError(
                f"Blend group {group_name!r} mixes avatar targets: {sorted(targets)}"
            )
        for _, item in group_items:
            if item.get("mode", "direct") != "direct" or not bool(
                item.get("rigid_radius", False)
            ):
                raise ValueError(
                    f"Group {group_name!r} supports direct rigid_radius pieces only"
                )
            lo, hi = sorted((float(item["src_0"]), float(item["src_1"])))
            if not (0.0 <= lo < hi <= 1.0):
                raise ValueError(
                    f"Invalid source interval in group {group_name!r}: {(lo, hi)}"
                )
            scale = float(item.get("scale", 1.0))
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(
                    f"Seam fitting needs positive finite scale in {group_name}"
                )

        ordered = sorted(
            group_items,
            key=lambda p: min(float(p[1]["src_0"]), float(p[1]["src_1"])),
        )
        surface_profiles = {}  # bounded projections only; never SDF grids
        paired = 0

        for (left_idx, left), (right_idx, right) in zip(ordered, ordered[1:]):
            left_hi = max(float(left["src_0"]), float(left["src_1"]))
            right_lo = min(float(right["src_0"]), float(right["src_1"]))
            overlap = left_hi - right_lo
            if overlap < -max_gap:
                raise ValueError(
                    f"Group {group_name!r}: gap between {left.get('name')} and "
                    f"{right.get('name')}; position fitting cannot extend cropped intervals"
                )
            if overlap < 0.0:
                raise ValueError(f"Group {group_name!r}: source intervals do not overlap")

            seam_s = 0.5 * (left_hi + right_lo)
            curve_l = curve_lookup(left["accessory_key"])
            curve_r = curve_lookup(right["accessory_key"])
            width_l = transition * abs(float(left["src_1"]) - float(left["src_0"]))
            width_r = transition * abs(float(right["src_1"]) - float(right["src_0"]))
            step_l = min(tangent_step, max(width_l / 5.0, 1e-4))
            step_r = min(tangent_step, max(width_r / 5.0, 1e-4))

            need_surface = do_position or (do_radius and radius_source == "surface")
            surface_radius_available = radius_source == "surface"
            if need_surface:
                try:
                    for item, curve in ((left, curve_l), (right, curve_r)):
                        key = item["accessory_key"]
                        if key not in surface_profiles:
                            surface_profiles[key] = _surface_profile(curve)
                except ValueError:
                    if do_position:
                        raise
                    surface_radius_available = False
                    print(
                        f"[seam {group_name}] WARNING surface radius evidence unavailable; "
                        "falling back to curve-radius fitting"
                    )

            corr_l = corr_r = slope_corr_l = slope_corr_r = 0.0
            if do_radius:
                lower, upper = math.log(1.0 - max_change), math.log(1.0 + max_change)
                if surface_radius_available:
                    profile_l = surface_profiles[left["accessory_key"]]
                    profile_r = surface_profiles[right["accessory_key"]]
                    log_l_profile = _surface_log_radius_profile(
                        left,
                        profile_l,
                        seam_s,
                        surface_window,
                        surface_bins,
                        radius_quantile,
                    )
                    log_r_profile = _surface_log_radius_profile(
                        right,
                        profile_r,
                        seam_s,
                        surface_window,
                        surface_bins,
                        radius_quantile,
                    )
                    delta = 0.5 * float(np.median(log_r_profile - log_l_profile))
                    corr_l = min(upper, max(lower, delta))
                    corr_r = min(upper, max(lower, -delta))

                    if match_radius_tangent:
                        slope_l_profile = _surface_radius_slope(
                            left,
                            profile_l,
                            seam_s,
                            step_l,
                            surface_window,
                            surface_bins,
                            radius_quantile,
                        )
                        slope_r_profile = _surface_radius_slope(
                            right,
                            profile_r,
                            seam_s,
                            step_r,
                            surface_window,
                            surface_bins,
                            radius_quantile,
                        )
                        slope_delta = 0.5 * float(
                            np.median(slope_r_profile - slope_l_profile)
                        )
                        slope_corr_l = _bound_radius_slope(
                            corr_l, slope_delta, width_l, lower, upper
                        )
                        slope_corr_r = _bound_radius_slope(
                            corr_r, -slope_delta, width_r, lower, upper
                        )

                    residual_profile = (
                        log_l_profile + corr_l - log_r_profile - corr_r
                    )
                    residual_rms = float(np.sqrt(np.mean(residual_profile ** 2)))
                    if residual_rms > 0.025:
                        print(
                            f"[seam {group_name}] WARNING residual surface-profile mismatch "
                            f"{100.0 * math.expm1(residual_rms):.1f}% after isotropic radius fit"
                        )
                else:
                    log_l = _log_radius(left, curve_l, seam_s)
                    log_r = _log_radius(right, curve_r, seam_s)
                    delta = 0.5 * (log_r - log_l)
                    corr_l = min(upper, max(lower, delta))
                    corr_r = min(upper, max(lower, -delta))
                    residual = abs((log_l + corr_l) - (log_r + corr_r))
                    if residual > 0.025:
                        print(
                            f"[seam {group_name}] WARNING residual radius mismatch "
                            f"{100.0 * math.expm1(residual):.1f}% exceeds allowed correction"
                        )
                    if match_radius_tangent:
                        slope_l = _radius_slope(left, curve_l, seam_s, step_l)
                        slope_r = _radius_slope(right, curve_r, seam_s, step_r)
                        slope_delta = 0.5 * (slope_r - slope_l)
                        slope_corr_l = _bound_radius_slope(
                            corr_l, slope_delta, width_l, lower, upper
                        )
                        slope_corr_r = _bound_radius_slope(
                            corr_r, -slope_delta, width_r, lower, upper
                        )

            pos_l = pos_r = np.zeros(2, dtype=np.float64)
            pos_slope_l = pos_slope_r = np.zeros(2, dtype=np.float64)
            if do_position:
                profile_l = surface_profiles[left["accessory_key"]]
                profile_r = surface_profiles[right["accessory_key"]]
                center_l = _position_center(
                    left, curve_l, profile_l, seam_s, math.exp(corr_l), sample_window
                )
                center_r = _position_center(
                    right, curve_r, profile_r, seam_s, math.exp(corr_r), sample_window
                )
                # Minimum-norm C0 center fit: symmetric displacement in the
                # common avatar transverse plane, converted to each part's
                # physical accessory-local tloc frame.
                gap = center_r - center_l
                pos_l = _bound_translation(
                    _rotate_uv(
                        0.5 * gap * float(left.get("scale", 1.0)),
                        float(left.get("rot_deg", 0.0)),
                    ),
                    max_position,
                )
                pos_r = _bound_translation(
                    _rotate_uv(
                        -0.5 * gap * float(right.get("scale", 1.0)),
                        float(right.get("rot_deg", 0.0)),
                    ),
                    max_position,
                )

                if match_position_tangent:
                    tangent_l = _position_slope(
                        left,
                        curve_l,
                        profile_l,
                        seam_s,
                        corr_l,
                        slope_corr_l,
                        width_l,
                        sample_window,
                        step_l,
                    )
                    tangent_r = _position_slope(
                        right,
                        curve_r,
                        profile_r,
                        seam_s,
                        corr_r,
                        slope_corr_r,
                        width_r,
                        sample_window,
                        step_r,
                    )
                    tangent_gap = tangent_r - tangent_l
                    raw_l = _rotate_uv(
                        0.5 * tangent_gap * float(left.get("scale", 1.0)),
                        float(left.get("rot_deg", 0.0)),
                    )
                    raw_r = _rotate_uv(
                        -0.5 * tangent_gap * float(right.get("scale", 1.0)),
                        float(right.get("rot_deg", 0.0)),
                    )
                    pos_slope_l = _bound_translation(
                        raw_l, max_position / max(width_l, 1e-9)
                    )
                    pos_slope_r = _bound_translation(
                        raw_r, max_position / max(width_r, 1e-9)
                    )

                print(
                    f"[seam {group_name}] position {left.get('name', left_idx)} / "
                    f"{right.get('name', right_idx)} local tloc(U,V) "
                    f"{pos_l.round(5).tolist()} / {pos_r.round(5).tolist()}"
                )
                if match_position_tangent:
                    print(
                        f"[seam {group_name}] position tangent d(tloc)/dsrc "
                        f"{pos_slope_l.round(5).tolist()} / "
                        f"{pos_slope_r.round(5).tolist()}"
                    )

            result.setdefault(left_idx, []).append(
                SeamCorrection(
                    seam_s,
                    width_l,
                    corr_l,
                    slope_corr_l,
                    tuple(pos_l),
                    tuple(pos_slope_l),
                )
            )
            result.setdefault(right_idx, []).append(
                SeamCorrection(
                    seam_s,
                    width_r,
                    corr_r,
                    slope_corr_r,
                    tuple(pos_r),
                    tuple(pos_slope_r),
                )
            )
            paired += 1
            if do_radius:
                source_label = "surface" if surface_radius_available else "curve"
                print(
                    f"[seam {group_name}] {left.get('name', left_idx)} + "
                    f"{right.get('name', right_idx)} at src={seam_s:.4f}: "
                    f"radius scale {math.exp(corr_l):.4f} / {math.exp(corr_r):.4f}; "
                    f"dlogR/dsrc correction {slope_corr_l:.4f} / "
                    f"{slope_corr_r:.4f} ({source_label})"
                )

        if paired == 0:
            raise ValueError(f"Group {group_name!r}: no adjoining sections found")
    return result


def radius_scale_at(source_coords, corrections):
    """C1 local radius profile: identity and zero derivative at band edges."""
    s = np.asarray(source_coords, dtype=np.float64).reshape(-1)
    log_scale = np.zeros_like(s)
    for seam in corrections:
        fade = _smooth_fade(s, seam.source_s, seam.half_width)
        log_scale += (
            seam.log_radius_scale
            + seam.log_radius_slope * (s - seam.source_s)
        ) * fade
    return np.exp(log_scale)


def warp_accessory_queries(accessory_data, avatar_data, corrections):
    """Apply local C1 translation, then the established radius query warp.

    Translation is in accessory-local physical U/V units, like tloc[1:3].  Its
    value and first derivative fade to zero at the correction-band edges.
    Processing is chunked so seam fitting never allocates another full 512^3
    working array.
    """
    if not corrections:
        return accessory_data
    source_s = np.asarray(avatar_data["coords"]).reshape(-1)
    local = np.asarray(accessory_data["samples_local"])
    if len(source_s) != len(local):
        raise ValueError("Seam scale/query point count mismatch")
    if not local.flags.writeable:
        local = local.copy()
        accessory_data["samples_local"] = local

    position_enabled = any(
        np.any(seam.position_uv) or np.any(seam.position_slope_uv)
        for seam in corrections
    )
    if position_enabled:
        radius = np.asarray(accessory_data["radius"])
        if radius.shape != (len(local), 2):
            raise ValueError("Position fitting needs per-point accessory U/V radii")

    chunk_size = 262144
    for start in range(0, len(local), chunk_size):
        stop = min(start + chunk_size, len(local))
        s = source_s[start:stop]
        piece = local[start:stop]
        if position_enabled:
            radii = radius[start:stop]
            if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
                raise ValueError("Invalid accessory radius for seam position fitting")
            for seam in corrections:
                base = np.asarray(seam.position_uv, dtype=np.float64)
                slope = np.asarray(seam.position_slope_uv, dtype=np.float64)
                if not np.any(base) and not np.any(slope):
                    continue
                active = np.abs(s - seam.source_s) < seam.half_width
                if not np.any(active):
                    continue
                ds = s[active] - seam.source_s
                fade = _smooth_fade(s[active], seam.source_s, seam.half_width)
                shift = (base[None, :] + ds[:, None] * slope[None, :]) * fade[:, None]
                # Matching _localize.py: moving geometry by +tloc means
                # subtracting that displacement from its inference queries.
                piece[active, 1] -= shift[:, 0] / radii[active, 0]
                piece[active, 2] -= shift[:, 1] / radii[active, 1]

        scale = radius_scale_at(s, corrections)
        np.divide(piece[:, 1], scale, out=piece[:, 1], casting="unsafe")
        np.divide(piece[:, 2], scale, out=piece[:, 2], casting="unsafe")
        if position_enabled:
            # Translation changes the polar support proxy as well as query UV.
            if "rho_n" in accessory_data:
                rho_n = np.asarray(accessory_data["rho_n"])[start:stop]
                np.hypot(piece[:, 1], piece[:, 2], out=rho_n)
            for key, col in (("u_n", 1), ("v_n", 2)):
                if key in accessory_data:
                    np.asarray(accessory_data[key])[start:stop] = piece[:, col]
            if "angles" in accessory_data:
                target = np.asarray(accessory_data["angles"])[start:stop]
                np.arctan2(piece[:, 2], piece[:, 1], out=target)
        else:
            # Preserve exactly the pre-existing radius-only support behavior.
            for key in ("rho_n", "u_n", "v_n"):
                if key in accessory_data:
                    proxy = np.asarray(accessory_data[key])
                    view = proxy[start:stop]
                    np.divide(
                        view,
                        scale.reshape(view.shape),
                        out=view,
                        casting="unsafe",
                    )
    return accessory_data
