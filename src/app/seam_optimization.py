"""Group-local radius and transverse-position seam fitting.

The low-cost fit uses accessory skeleton radii and owned surface points as
proxies for the neural SDF cross-section. It modifies model queries before
inference, rather than deforming extracted meshes or allocating extra grids.
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
    scale = float(item.get("scale", 1.0))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Expected positive finite scale, got {scale}")
    return math.log(scale) + math.log(_positive_radius(curve, _mapped_s(item, source_s)))


def _radius_slope(item, curve, source_s, step):
    lo, hi = sorted((float(item["src_0"]), float(item["src_1"])))
    left, right = max(lo, source_s - step), min(hi, source_s + step)
    if right - left <= 1e-9:
        return 0.0
    return (_log_radius(item, curve, right) - _log_radius(item, curve, left)) / (right - left)


def _surface_profile(curve):
    """Project a bounded sample of accessory-owned surface points once.

    Do not use mesh vertices or fitted SDF centers: this is a cheap geometric
    proxy that can be computed before the expensive 512^3 reconstruction.
    """
    core = curve.core
    points = getattr(core, "surface_points_owned", None)
    if points is None:
        raise ValueError("Position fitting needs accessory surface_points_owned")
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 32:
        raise ValueError("Position fitting needs at least 32 owned XYZ surface points")
    # Subsample BEFORE converting to float64 to bound temporary memory.
    points = np.asarray(points[::max(1, math.ceil(len(points) / 20000))],
                        dtype=np.float64)
    s = np.asarray(core.curve_projection(points), dtype=np.float64).reshape(-1)
    valid = np.isfinite(s) & (s >= 0.0) & (s <= 1.0)
    points, s = points[valid], s[valid]
    if len(s) < 32:
        raise ValueError("Insufficient valid owned surface points for position fitting")
    support = core.interpolate(s, radius=False, frame=True)
    local = np.einsum("nij,nj->ni", support["frame"], points - support["points"])
    return s, local[:, 1:3]


def _cross_section_center(profile, target_s, half_window):
    """Robust midpoint of opposite surface extents around one source section."""
    s, uv = profile
    distance = np.abs(s - target_s)
    # Widen modestly if a thin puffer section has few sampled surface points.
    selected = distance <= half_window
    if np.count_nonzero(selected) < 32:
        selected = distance <= min(0.08, 2.5 * half_window)
    if np.count_nonzero(selected) < 32:
        raise ValueError(f"Not enough surface points near accessory s={target_s:.4f}; "
                         "cannot estimate a position correction reliably")
    ring_uv = uv[selected]
    center = 0.5 * (np.quantile(ring_uv, 0.10, axis=0) +
                    np.quantile(ring_uv, 0.90, axis=0))
    if not np.all(np.isfinite(center)):
        raise ValueError("Non-finite accessory surface center")
    return center


def _rotate_uv(vector, degrees):
    theta = math.radians(degrees)
    c, si = math.cos(theta), math.sin(theta)
    return np.array([c * vector[0] - si * vector[1],
                     si * vector[0] + c * vector[1]], dtype=np.float64)


def _manual_tloc_uv(item):
    value = np.asarray(item.get("tloc", item.get("translate_local", (0, 0, 0))),
                       dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"Invalid tloc in {item.get('name', item.get('target_key'))}")
    return value[1:3]


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
    effective_center = (radius_factor * center - model_center +
                        _manual_tloc_uv(item)) / scale
    return _rotate_uv(effective_center, -float(item.get("rot_deg", 0.0)))


def _bound_translation(uv, max_change):
    length = float(np.linalg.norm(uv))
    if length > max_change:
        return uv * (max_change / length)
    return uv


def build_seam_plan(adaptations, group_settings, curve_lookup):
    """Return correction lists indexed by expanded adaptation index.

    Each enabled group is isolated; no source-curve state is mutated or reused.
    Only touching/overlapping intervals of the same avatar target are paired.
    """
    if not isinstance(group_settings, dict):
        raise TypeError("blend_groups must be a mapping")
    enabled = {str(name): spec["seam_optimization"] for name, spec in group_settings.items()
               if isinstance(spec, dict) and isinstance(spec.get("seam_optimization"), dict)
               and spec["seam_optimization"].get("enabled", False)}
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
        if not do_radius and not do_position:
            continue
        max_change = float(spec.get("max_radius_change", 0.20))
        max_position = float(spec.get("max_position_change", 0.05))
        transition = float(spec.get("transition_fraction", 0.20))
        sample_window = float(spec.get("position_sample_window", 0.025))
        max_gap = float(spec.get("max_gap", 0.0))
        if not (0.0 < max_change < 1.0 and 0.0 < transition <= 0.5 and max_gap >= 0.0):
            raise ValueError(f"Invalid seam_optimization parameters in {group_name}")
        if do_position and not (0.0 < max_position and 0.0 < sample_window <= 0.08):
            raise ValueError(f"Invalid positional optimization settings in {group_name}")
        group_items = by_group[group_name]
        if len(group_items) < 2:
            raise ValueError(f"Blend group {group_name!r} needs at least two pieces for seam optimization")
        targets = {item["target_key"] for _, item in group_items}
        if len(targets) != 1:
            raise ValueError(f"Blend group {group_name!r} mixes avatar targets: {sorted(targets)}")
        for _, item in group_items:
            if item.get("mode", "direct") != "direct" or not bool(item.get("rigid_radius", False)):
                raise ValueError(f"Group {group_name!r} supports direct rigid_radius pieces only")
            lo, hi = sorted((float(item["src_0"]), float(item["src_1"])))
            if not (0.0 <= lo < hi <= 1.0):
                raise ValueError(f"Invalid source interval in group {group_name!r}: {(lo, hi)}")
            if do_position and (not np.isfinite(float(item.get("scale", 1.0))) or
                                float(item.get("scale", 1.0)) <= 0.0):
                raise ValueError(f"Position fitting needs positive finite scale in {group_name}")
        ordered = sorted(group_items, key=lambda p: min(float(p[1]["src_0"]), float(p[1]["src_1"])))
        surface_profiles = {}  # cache only bounded source-point projections, never SDF grids
        paired = 0
        for (left_idx, left), (right_idx, right) in zip(ordered, ordered[1:]):
            left_hi = max(float(left["src_0"]), float(left["src_1"]))
            right_lo = min(float(right["src_0"]), float(right["src_1"]))
            overlap = left_hi - right_lo
            if overlap < -max_gap:
                raise ValueError(f"Group {group_name!r}: gap between {left.get('name')} and "
                                 f"{right.get('name')}; position fitting cannot extend cropped intervals")
            if overlap < 0.0:
                raise ValueError(f"Group {group_name!r}: source intervals do not overlap")
            seam_s = 0.5 * (left_hi + right_lo)
            curve_l, curve_r = curve_lookup(left["accessory_key"]), curve_lookup(right["accessory_key"])
            width_l = transition * abs(float(left["src_1"]) - float(left["src_0"]))
            width_r = transition * abs(float(right["src_1"]) - float(right["src_0"]))
            corr_l = corr_r = slope_corr_l = slope_corr_r = 0.0
            if do_radius:
                log_l = _log_radius(left, curve_l, seam_s)
                log_r = _log_radius(right, curve_r, seam_s)
                delta = 0.5 * (log_r - log_l)
                lower, upper = math.log(1.0 - max_change), math.log(1.0 + max_change)
                corr_l = min(upper, max(lower, delta))
                corr_r = min(upper, max(lower, -delta))
                residual = abs((log_l + corr_l) - (log_r + corr_r))
                if residual > 0.025:
                    print(f"[seam {group_name}] WARNING residual radius mismatch "
                          f"{100.0 * math.expm1(residual):.1f}% exceeds allowed correction")
                slope_l = _radius_slope(left, curve_l, seam_s, min(width_l / 5.0, 0.005))
                slope_r = _radius_slope(right, curve_r, seam_s, min(width_r / 5.0, 0.005))
                target_slope = 0.5 * (slope_l + slope_r)
                slope_corr_l, slope_corr_r = target_slope - slope_l, target_slope - slope_r
            pos_l = pos_r = np.zeros(2, dtype=np.float64)
            if do_position:
                for item, curve in ((left, curve_l), (right, curve_r)):
                    key = item["accessory_key"]
                    if key not in surface_profiles:
                        surface_profiles[key] = _surface_profile(curve)
                center_l = _position_center(
                    left, curve_l, surface_profiles[left["accessory_key"]],
                    seam_s, math.exp(corr_l), sample_window)
                center_r = _position_center(
                    right, curve_r, surface_profiles[right["accessory_key"]],
                    seam_s, math.exp(corr_r), sample_window)
                # Minimum-norm joint center fit: symmetric displacement in the
                # common avatar transverse plane, converted to each part's
                # physical accessory-local tloc frame, with a per-piece cap.
                gap = center_r - center_l
                pos_l = _bound_translation(
                    _rotate_uv(0.5 * gap * float(left.get("scale", 1.0)),
                               float(left.get("rot_deg", 0.0))), max_position)
                pos_r = _bound_translation(
                    _rotate_uv(-0.5 * gap * float(right.get("scale", 1.0)),
                               float(right.get("rot_deg", 0.0))), max_position)
                print(f"[seam {group_name}] position {left.get('name', left_idx)} / "
                      f"{right.get('name', right_idx)} local tloc(U,V) "
                      f"{pos_l.round(5).tolist()} / {pos_r.round(5).tolist()}")
            result.setdefault(left_idx, []).append(SeamCorrection(
                seam_s, width_l, corr_l, slope_corr_l, tuple(pos_l)))
            result.setdefault(right_idx, []).append(SeamCorrection(
                seam_s, width_r, corr_r, slope_corr_r, tuple(pos_r)))
            paired += 1
            if do_radius:
                print(f"[seam {group_name}] {left.get('name', left_idx)} + "
                      f"{right.get('name', right_idx)} at src={seam_s:.4f}: "
                      f"radius scale {math.exp(corr_l):.4f} / {math.exp(corr_r):.4f}")
        if paired == 0:
            raise ValueError(f"Group {group_name!r}: no adjoining sections found")
    return result


def radius_scale_at(source_coords, corrections):
    """C1 local profile: identity and zero derivative at band edges."""
    s = np.asarray(source_coords, dtype=np.float64).reshape(-1)
    log_scale = np.zeros_like(s)
    for seam in corrections:
        x = (s - seam.source_s) / seam.half_width
        t = np.minimum(np.abs(x), 1.0)
        fade = 1.0 - t * t * (3.0 - 2.0 * t)
        log_scale += (seam.log_radius_scale +
                      seam.log_radius_slope * (s - seam.source_s)) * fade
    return np.exp(log_scale)


def warp_accessory_queries(accessory_data, avatar_data, corrections):
    """Apply local inverse translation, then the established radius query warp.

    Translation is in accessory-local physical U/V units, like tloc[1:3]; it
    fades to zero outside the seam. Process bounded chunks at high resolution
    so position fitting never allocates another full 512^3 working array.
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
    position_enabled = any(np.any(seam.position_uv) for seam in corrections)
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
                du, dv = seam.position_uv
                if du == 0.0 and dv == 0.0:
                    continue
                active = np.abs(s - seam.source_s) < seam.half_width
                if not np.any(active):
                    continue
                t = np.abs(s[active] - seam.source_s) / seam.half_width
                fade = 1.0 - t * t * (3.0 - 2.0 * t)
                # Matching _localize.py: moving geometry by +tloc means
                # subtracting that displacement from its inference queries.
                piece[active, 1] -= du * fade / radii[active, 0]
                piece[active, 2] -= dv * fade / radii[active, 1]
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
                    np.divide(view, scale.reshape(view.shape), out=view, casting="unsafe")
    return accessory_data
