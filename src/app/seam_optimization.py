"""Group-local surface seam fitting for rigid-radius accessory adaptations.

The fit is deliberately low-cost: it uses the accessory curve plus a bounded
sample of accessory-owned surface points and modifies model queries before
inference.  It does not deform extracted meshes and does not allocate another
full reconstruction grid.

For each adjoining pair in a blend group we fit both C0 and C1 behavior:

* transverse center position and center tangent,
* outer cross-section scale and longitudinal taper tangent.

Surface-profile fitting supports isotropic scale, anisotropic U/V scale, and
an optional low-order angular residual.  The angular mode first performs the U/V
fit, then fits mean/center-preserving Fourier harmonics around the cross-section.
All value and longitudinal-derivative corrections fade C1-smoothly back to
identity away from each seam.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import least_squares


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
    # Optional anisotropic radius correction in accessory-local [U, V] axes.
    # None means: use the scalar log_radius_* values for both axes.
    log_radius_scale_uv: tuple = None
    log_radius_slope_uv: tuple = None
    # Optional mean-zero Fourier log-radius correction in accessory-local angle.
    # Coefficients are [cos(theta), sin(theta), cos(2theta), sin(2theta), ...].
    angular_log_radius_coeffs: tuple = None
    angular_log_radius_slope_coeffs: tuple = None


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


def _wrap_reference_radius(curve, s):
    """Median directional wrap radius at one curve coordinate.

    This mirrors the reference-radius term used by ``localize_samples_adapt``
    for wrap-radius mapping.  The full-resolution wrap field is preferred so
    seam planning observes the same geometry as the runtime mapper.
    """
    core = curve.core
    wrap = getattr(core, "key_wrap_radius_full", None)
    s_bins = getattr(core, "wrap_s_bins_full", None)
    if wrap is None or s_bins is None:
        wrap = getattr(core, "key_wrap_radius", None)
        s_bins = getattr(core, "wrap_s_bins", None)
    if wrap is None or s_bins is None:
        # Defensive fallback for old assets without a directional wrap field.
        return _positive_radius(curve, s)
    wrap = np.asarray(wrap, dtype=np.float64)
    s_bins = np.asarray(s_bins, dtype=np.float64).reshape(-1)
    if wrap.ndim != 2 or wrap.shape[0] != len(s_bins) or len(s_bins) < 2:
        raise ValueError("Invalid wrap-radius field for seam optimization")
    reference = np.nanmedian(wrap, axis=1)
    if not np.all(np.isfinite(reference)) or np.any(reference <= 0.0):
        raise ValueError("Non-positive wrap-radius reference for seam optimization")
    return float(np.interp(float(s), s_bins, reference))


def _radial_query_scale(item, accessory_curve, source_s):
    """Physical accessory-radius / avatar-radius scale before seam correction.

    Rigid mode uses the YAML global scale directly.  For wrap mode we support
    the exact angle-independent case (``wrap_angular_follow: 0``), which is the
    cone configuration: ``scale * target_wrap_ref / avatar_wrap_ref``.

    Angle-following wrap is deliberately rejected for now because its inverse
    map is directional and must be fitted per angular sample rather than with a
    single scalar conversion.
    """
    scale = float(item.get("scale", 1.0))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Expected positive finite scale, got {scale}")

    if bool(item.get("wrap_radius", False)):
        alpha = float(item.get("wrap_angular_follow", 1.0))
        if abs(alpha) > 1e-12:
            raise ValueError(
                "Wrap seam optimization currently requires wrap_angular_follow: 0.0"
            )
        if item.get("wrap_adapt_n_keypoints", None) is not None:
            raise ValueError(
                "Wrap seam optimization does not yet mirror wrap_adapt_n_keypoints; "
                "leave it unset"
            )
        if bool(item.get("use_attach_to_free_blend", item.get("use_wrap_to_rigid_blend", False))):
            raise ValueError(
                "Wrap seam optimization does not yet support attach-to-free radial blending"
            )
        avatar_curve = item.get("_seam_target_curve")
        if avatar_curve is None:
            raise ValueError("Wrap seam optimization is missing the avatar target curve")
        target_s = _mapped_s(item, source_s)
        src_ref = _wrap_reference_radius(avatar_curve, source_s)
        tgt_ref = _wrap_reference_radius(accessory_curve, target_s)
        value = scale * tgt_ref / max(src_ref, 1e-12)
        if "wrap_scale_min" in item or "wrap_scale_max" in item:
            value = float(np.clip(
                value,
                float(item.get("wrap_scale_min", 0.25)),
                float(item.get("wrap_scale_max", 4.0)),
            ))
        return value

    if bool(item.get("rigid_radius", False)):
        return scale

    raise ValueError("Seam optimization needs rigid_radius or wrap_radius mapping")


def _tilt_geometry_uv(item, accessory_curve, target_s):
    """Geometry-space U/V offset induced by localize_samples_adapt tilt.

    Runtime tilt subtracts this value from the model query, therefore the
    inferred geometry moves by the positive value returned here.
    """
    shift = np.zeros(2, dtype=np.float64)
    if item.get("tilt_u", None) is None and item.get("tilt_v", None) is None:
        return shift
    curve_len, _ = accessory_curve.core.calc_curve_length()
    curve_len = float(curve_len)
    for axis, key in enumerate(("tilt_u", "tilt_v")):
        if item.get(key, None) is None:
            continue
        anchor = float(item.get(f"{key}_anchor", item["tgt_1"]))
        end = float(item.get(f"{key}_end", item["tgt_0"]))
        direction = float(np.sign(end - anchor))
        if abs(direction) < 1e-12:
            continue
        forward = max(direction * (float(target_s) - anchor) * curve_len, 0.0)
        shift[axis] = math.tan(math.radians(float(item[key]))) * forward
    return shift


def _log_radius(item, curve, source_s):
    """Log physical output-radius proxy for the rigid-radius mapping.

    _localize.py uses ``rho_acc = rho_avatar * global_scale``.  Therefore a
    model-space accessory radius R appears on the avatar with physical radius
    R / global_scale (before this seam correction).  This inverse relationship
    is easy to accidentally reverse when fitting two adaptations.
    """
    query_scale = _radial_query_scale(item, curve, source_s)
    return (
        math.log(_positive_radius(curve, _mapped_s(item, source_s)))
        - math.log(query_scale)
    )


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




def _angular_basis(theta, max_harmonic):
    """Mean/center-preserving Fourier basis for harmonics 2..max_harmonic.

    Harmonic 0 is already handled by radius scale and harmonic 1 behaves like
    a first-order center translation, which is handled by the position fit.
    """
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    cols = []
    for harmonic in range(2, int(max_harmonic) + 1):
        cols.append(np.cos(harmonic * theta))
        cols.append(np.sin(harmonic * theta))
    if not cols:
        return np.zeros((len(theta), 0), dtype=np.float64)
    return np.column_stack(cols)


def _angular_log_field(theta, coeffs):
    """Evaluate a mean-zero angular log-radius field."""
    coeffs = np.asarray(coeffs, dtype=np.float64).reshape(-1)
    if coeffs.size == 0:
        return np.zeros(np.asarray(theta).shape, dtype=np.float64)
    if coeffs.size % 2:
        raise ValueError("Angular seam coefficients must contain cos/sin pairs")
    # Two coefficients per harmonic, starting from harmonic 2.
    basis = _angular_basis(theta, coeffs.size // 2 + 1)
    return basis @ coeffs


def _bound_angular_field(coeffs, max_abs_value, n_probe=128):
    """Scale Fourier coefficients so the sampled field stays within a bound."""
    coeffs = np.asarray(coeffs, dtype=np.float64).reshape(-1).copy()
    if coeffs.size == 0:
        return coeffs
    limit = float(max_abs_value)
    theta = np.linspace(-np.pi, np.pi, int(n_probe), endpoint=False)
    peak = float(np.max(np.abs(_angular_log_field(theta, coeffs))))
    if peak > limit and peak > 1e-12:
        coeffs *= limit / peak
    return coeffs


def _bound_angular_coeffs(coeffs, max_change, n_probe=128):
    """Scale Fourier coefficients so pointwise radial change stays bounded."""
    return _bound_angular_field(
        coeffs, math.log(1.0 + float(max_change)), n_probe=n_probe
    )


def _manual_tloc_uv(item):
    value = np.asarray(
        item.get("tloc", item.get("translate_local", (0, 0, 0))), dtype=np.float64
    )
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"Invalid tloc in {item.get('name', item.get('target_key'))}")
    return value[1:3]


def _angular_log_radius_profile(common_uv, n_bins, radius_quantile):
    """Robust outer radial envelope of a centered UV ring in a common frame."""
    common = np.asarray(common_uv, dtype=np.float64)
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
    values = np.interp(
        query,
        np.concatenate((x - n_bins, x, x + n_bins)),
        np.concatenate((y, y, y)),
    )
    return np.log(np.maximum(values, 1e-12))


def _surface_ring_local(item, profile, source_s, half_window):
    """Centered output ring in accessory-local axes before rot_deg alignment."""
    target_s = _mapped_s(item, source_s)
    ring_uv, center = _cross_section_points(profile, target_s, half_window)
    accessory_curve = item.get("_seam_accessory_curve")
    if accessory_curve is None:
        # Backward compatibility for direct helper tests of rigid-radius pieces.
        query_scale = float(item.get("scale", 1.0))
    else:
        query_scale = _radial_query_scale(item, accessory_curve, source_s)
    if not np.isfinite(query_scale) or query_scale <= 0.0:
        raise ValueError("Surface radius fitting needs positive radial query scale")
    return (ring_uv - center) / query_scale


def _surface_log_radius_profile(
    item,
    profile,
    source_s,
    half_window,
    n_bins,
    radius_quantile,
    log_scale_uv=None,
    angular_coeffs=None,
):
    """Outer surface profile in the common avatar-local transverse frame.

    ``log_scale_uv`` is an optional geometry scale in the accessory's own
    physical U/V axes.  ``angular_coeffs`` adds a low-order mean-zero radial
    field in the *post-U/V-scale accessory-local angle*.  The runtime inverse
    query warp uses the same ordering: undo angular radial scale, then U/V scale.
    """
    local = _surface_ring_local(item, profile, source_s, half_window)
    if log_scale_uv is not None:
        logs = np.asarray(log_scale_uv, dtype=np.float64).reshape(2)
        local = local * np.exp(logs)[None, :]
    if angular_coeffs is not None:
        theta_local = np.arctan2(local[:, 1], local[:, 0])
        angular_log = _angular_log_field(theta_local, angular_coeffs)
        local = local * np.exp(angular_log)[:, None]
    common = _rotate_uv(local, -float(item.get("rot_deg", 0.0)))
    return _angular_log_radius_profile(common, n_bins, radius_quantile)


def _surface_radius_slope(
    item,
    profile,
    source_s,
    step,
    half_window,
    n_bins,
    radius_quantile,
    seam_s=None,
    correction_width=None,
    log_scale_uv=None,
    log_slope_uv=None,
    angular_coeffs=None,
    angular_slope_coeffs=None,
):
    """Finite-difference d(log radial profile)/d(source_s).

    Optional U/V and angular corrections are evaluated with the same C1 fade
    used by the runtime inverse query warp, so tangent fitting targets the
    geometry that will actually be inferred.
    """
    lo, hi = sorted((float(item["src_0"]), float(item["src_1"])))
    left, right = max(lo, source_s - step), min(hi, source_s + step)
    if right - left <= 1e-9:
        return np.zeros(n_bins, dtype=np.float64)

    def values_at(s):
        uv_value = None
        if log_scale_uv is not None:
            base = np.asarray(log_scale_uv, dtype=np.float64).reshape(2)
            slope = np.zeros(2, dtype=np.float64)
            if log_slope_uv is not None:
                slope = np.asarray(log_slope_uv, dtype=np.float64).reshape(2)
            if seam_s is None or correction_width is None:
                uv_value = base + slope * (s - source_s)
            else:
                fade = float(_smooth_fade([s], seam_s, correction_width)[0])
                uv_value = (base + slope * (s - seam_s)) * fade

        angular_value = None
        if angular_coeffs is not None:
            base = np.asarray(angular_coeffs, dtype=np.float64).reshape(-1)
            slope = np.zeros_like(base)
            if angular_slope_coeffs is not None:
                slope = np.asarray(angular_slope_coeffs, dtype=np.float64).reshape(base.shape)
            if seam_s is None or correction_width is None:
                angular_value = base + slope * (s - source_s)
            else:
                fade = float(_smooth_fade([s], seam_s, correction_width)[0])
                angular_value = (base + slope * (s - seam_s)) * fade
        return uv_value, angular_value

    uv_l, angular_l = values_at(left)
    uv_r, angular_r = values_at(right)
    profile_l = _surface_log_radius_profile(
        item, profile, left, half_window, n_bins, radius_quantile, uv_l, angular_l
    )
    profile_r = _surface_log_radius_profile(
        item, profile, right, half_window, n_bins, radius_quantile, uv_r, angular_r
    )
    return (profile_r - profile_l) / (right - left)

def _anisotropy_excess(log_uv, max_anisotropic_change):
    """Amount by which per-axis deformation exceeds the configured shape limit."""
    if max_anisotropic_change is None:
        return 0.0
    max_dev = math.log(1.0 + float(max_anisotropic_change))
    logs = np.asarray(log_uv, dtype=np.float64).reshape(2)
    half_difference = 0.5 * abs(float(logs[0] - logs[1]))
    return max(0.0, half_difference - max_dev)


def _fit_anisotropic_radius_pair(
    left,
    right,
    profile_l,
    profile_r,
    seam_s,
    surface_window,
    surface_bins,
    radius_quantile,
    lower,
    upper,
    max_anisotropic_change,
    deformation_weight,
):
    """Fit two local axis scales per side against the actual surface envelope."""
    baseline_l = _surface_log_radius_profile(
        left, profile_l, seam_s, surface_window, surface_bins, radius_quantile
    )
    baseline_r = _surface_log_radius_profile(
        right, profile_r, seam_s, surface_window, surface_bins, radius_quantile
    )
    # The old isotropic solution is a strong initial point and keeps the new
    # optimizer stable when the cross sections are already close to circular.
    delta = 0.5 * float(np.median(baseline_r - baseline_l))
    iso_l = float(np.clip(delta, lower, upper))
    iso_r = float(np.clip(-delta, lower, upper))
    x0 = np.array([iso_l, iso_l, iso_r, iso_r], dtype=np.float64)
    reg = math.sqrt(max(0.0, deformation_weight))

    def residual(x):
        lp = _surface_log_radius_profile(
            left, profile_l, seam_s, surface_window, surface_bins,
            radius_quantile, x[:2]
        )
        rp = _surface_log_radius_profile(
            right, profile_r, seam_s, surface_window, surface_bins,
            radius_quantile, x[2:]
        )
        parts = [lp - rp]
        if reg > 0.0:
            parts.append(reg * x)
        # Hard axis bounds already cap total scale.  This extra residual keeps
        # the fit from solving folds/noise with an unnecessarily elliptical warp.
        excess = np.array([
            _anisotropy_excess(x[:2], max_anisotropic_change),
            _anisotropy_excess(x[2:], max_anisotropic_change),
        ])
        parts.append(10.0 * excess)
        return np.concatenate(parts)

    fit = least_squares(
        residual,
        x0,
        bounds=(np.full(4, lower), np.full(4, upper)),
        max_nfev=60,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
    )
    x = np.asarray(fit.x, dtype=np.float64)
    # Enforce the anisotropy limit deterministically after optimization while
    # preserving each section's geometric-mean scale as much as possible.
    max_dev = math.log(1.0 + float(max_anisotropic_change))
    for offset in (0, 2):
        pair = x[offset:offset + 2]
        mean = 0.5 * float(pair[0] + pair[1])
        diff = float(np.clip(0.5 * (pair[0] - pair[1]), -max_dev, max_dev))
        pair[:] = np.clip([mean + diff, mean - diff], lower, upper)

    final_l = _surface_log_radius_profile(
        left, profile_l, seam_s, surface_window, surface_bins,
        radius_quantile, x[:2]
    )
    final_r = _surface_log_radius_profile(
        right, profile_r, seam_s, surface_window, surface_bins,
        radius_quantile, x[2:]
    )
    before_rms = float(np.sqrt(np.mean((baseline_l - baseline_r) ** 2)))
    after_rms = float(np.sqrt(np.mean((final_l - final_r) ** 2)))
    return x[:2].copy(), x[2:].copy(), before_rms, after_rms


def _fit_anisotropic_tangent_pair(
    left,
    right,
    profile_l,
    profile_r,
    seam_s,
    width_l,
    width_r,
    step_l,
    step_r,
    surface_window,
    surface_bins,
    radius_quantile,
    base_l,
    base_r,
    lower,
    upper,
    deformation_weight,
):
    """Fit d(log U,V scale)/dsrc so the surface taper is C1 at the seam."""
    base_l = np.asarray(base_l, dtype=np.float64).reshape(2)
    base_r = np.asarray(base_r, dtype=np.float64).reshape(2)

    def slope_limits(base, width):
        available = np.minimum(upper - base, base - lower)
        return np.maximum(available / max(width, 1e-9), 0.0)

    limits = np.concatenate((slope_limits(base_l, width_l),
                             slope_limits(base_r, width_r)))
    if np.all(limits <= 1e-12):
        return np.zeros(2), np.zeros(2)

    slope_l0 = _surface_radius_slope(
        left, profile_l, seam_s, step_l, surface_window, surface_bins,
        radius_quantile, seam_s, width_l, base_l, np.zeros(2)
    )
    slope_r0 = _surface_radius_slope(
        right, profile_r, seam_s, step_r, surface_window, surface_bins,
        radius_quantile, seam_s, width_r, base_r, np.zeros(2)
    )
    delta = 0.5 * float(np.median(slope_r0 - slope_l0))
    x0 = np.array([delta, delta, -delta, -delta], dtype=np.float64)
    x0 = np.clip(x0, -limits, limits)
    safe_limits = np.maximum(limits, 1e-12)
    reg = math.sqrt(max(0.0, deformation_weight))

    def residual(x):
        sl = _surface_radius_slope(
            left, profile_l, seam_s, step_l, surface_window, surface_bins,
            radius_quantile, seam_s, width_l, base_l, x[:2]
        )
        sr = _surface_radius_slope(
            right, profile_r, seam_s, step_r, surface_window, surface_bins,
            radius_quantile, seam_s, width_r, base_r, x[2:]
        )
        parts = [sl - sr]
        if reg > 0.0:
            # Scale by a representative seam step so the penalty is in roughly
            # the same log-radius units as the profile residual.
            parts.append(reg * x * max(step_l, step_r))
        return np.concatenate(parts)

    fit = least_squares(
        residual,
        x0,
        bounds=(-safe_limits, safe_limits),
        max_nfev=60,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
    )
    x = np.asarray(fit.x, dtype=np.float64)
    x[limits <= 1e-12] = 0.0
    return x[:2].copy(), x[2:].copy()


def _fit_angular_radius_pair(
    left,
    right,
    profile_l,
    profile_r,
    seam_s,
    surface_window,
    surface_bins,
    radius_quantile,
    base_uv_l,
    base_uv_r,
    order,
    max_angular_change,
    deformation_weight,
):
    """Fit a low-order angular radial residual on top of U/V scaling.

    The Fourier field has no constant term, so global radius stays controlled by
    the existing isotropic/U-V fit.  Harmonics 1..order only refine circumferential
    profile mismatch.
    """
    n_coeff = 2 * max(0, int(order) - 1)
    if n_coeff <= 0:
        return np.zeros(0), np.zeros(0), 0.0, 0.0
    baseline_l = _surface_log_radius_profile(
        left, profile_l, seam_s, surface_window, surface_bins,
        radius_quantile, base_uv_l
    )
    baseline_r = _surface_log_radius_profile(
        right, profile_r, seam_s, surface_window, surface_bins,
        radius_quantile, base_uv_r
    )
    before_rms = float(np.sqrt(np.mean((baseline_l - baseline_r) ** 2)))
    x0 = np.zeros(2 * n_coeff, dtype=np.float64)
    max_log = math.log(1.0 + float(max_angular_change))
    probe_theta = np.linspace(-np.pi, np.pi, max(64, 4 * surface_bins), endpoint=False)
    probe_basis = _angular_basis(probe_theta, order)
    reg = math.sqrt(max(0.0, deformation_weight))

    def residual(x):
        cl = x[:n_coeff]
        cr = x[n_coeff:]
        lp = _surface_log_radius_profile(
            left, profile_l, seam_s, surface_window, surface_bins,
            radius_quantile, base_uv_l, cl
        )
        rp = _surface_log_radius_profile(
            right, profile_r, seam_s, surface_window, surface_bins,
            radius_quantile, base_uv_r, cr
        )
        parts = [lp - rp]
        if reg > 0.0:
            parts.append(reg * x)
        # Penalize pointwise radial changes beyond the configured bound.
        field_l = probe_basis @ cl
        field_r = probe_basis @ cr
        excess = np.concatenate((
            np.maximum(np.abs(field_l) - max_log, 0.0),
            np.maximum(np.abs(field_r) - max_log, 0.0),
        ))
        parts.append(10.0 * excess)
        return np.concatenate(parts)

    fit = least_squares(
        residual,
        x0,
        bounds=(np.full_like(x0, -max_log), np.full_like(x0, max_log)),
        max_nfev=80,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
    )
    x = np.asarray(fit.x, dtype=np.float64)
    coeff_l = _bound_angular_coeffs(x[:n_coeff], max_angular_change)
    coeff_r = _bound_angular_coeffs(x[n_coeff:], max_angular_change)
    final_l = _surface_log_radius_profile(
        left, profile_l, seam_s, surface_window, surface_bins,
        radius_quantile, base_uv_l, coeff_l
    )
    final_r = _surface_log_radius_profile(
        right, profile_r, seam_s, surface_window, surface_bins,
        radius_quantile, base_uv_r, coeff_r
    )
    after_rms = float(np.sqrt(np.mean((final_l - final_r) ** 2)))
    return coeff_l, coeff_r, before_rms, after_rms


def _fit_angular_tangent_pair(
    left,
    right,
    profile_l,
    profile_r,
    seam_s,
    width_l,
    width_r,
    step_l,
    step_r,
    surface_window,
    surface_bins,
    radius_quantile,
    base_uv_l,
    base_uv_r,
    slope_uv_l,
    slope_uv_r,
    angular_l,
    angular_r,
    max_angular_tangent_change,
    deformation_weight,
):
    """Fit d(Fourier log-radius)/dsrc so angular surface taper is C1."""
    angular_l = np.asarray(angular_l, dtype=np.float64).reshape(-1)
    angular_r = np.asarray(angular_r, dtype=np.float64).reshape(-1)
    if angular_l.shape != angular_r.shape:
        raise ValueError("Angular seam coefficient size mismatch")
    n_coeff = angular_l.size
    if n_coeff == 0:
        return np.zeros(0), np.zeros(0)

    x0 = np.zeros(2 * n_coeff, dtype=np.float64)
    limit = float(max_angular_tangent_change)
    reg = math.sqrt(max(0.0, deformation_weight))

    def residual(x):
        sl = _surface_radius_slope(
            left, profile_l, seam_s, step_l, surface_window, surface_bins,
            radius_quantile, seam_s, width_l, base_uv_l, slope_uv_l,
            angular_l, x[:n_coeff]
        )
        sr = _surface_radius_slope(
            right, profile_r, seam_s, step_r, surface_window, surface_bins,
            radius_quantile, seam_s, width_r, base_uv_r, slope_uv_r,
            angular_r, x[n_coeff:]
        )
        parts = [sl - sr]
        if reg > 0.0:
            parts.append(reg * x * max(step_l, step_r))
        return np.concatenate(parts)

    fit = least_squares(
        residual,
        x0,
        bounds=(np.full_like(x0, -limit), np.full_like(x0, limit)),
        max_nfev=80,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
    )
    x = np.asarray(fit.x, dtype=np.float64)
    slope_l = _bound_angular_field(x[:n_coeff], limit)
    slope_r = _bound_angular_field(x[n_coeff:], limit)
    return slope_l, slope_r

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
    factor = np.asarray(radius_factor, dtype=np.float64)
    if factor.ndim == 0:
        factor = np.repeat(float(factor), 2)
    factor = factor.reshape(2)
    query_scale = _radial_query_scale(item, curve, source_s)
    target_s = _mapped_s(item, source_s)
    tilt_uv = _tilt_geometry_uv(item, curve, target_s)
    # The inverse query warp expands geometry by factor[U,V] in accessory-local
    # physical axes. Manual tloc and one-sided tilt are query translations, so
    # they become positive geometry offsets before mapping back to avatar space.
    effective_center = (
        factor * center - model_center + _manual_tloc_uv(item) + tilt_uv
    ) / query_scale
    return _rotate_uv(effective_center, -float(item.get("rot_deg", 0.0)))


def _smooth_fade(source_s, seam_s, half_width):
    x = (np.asarray(source_s, dtype=np.float64) - seam_s) / half_width
    t = np.minimum(np.abs(x), 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


def _single_radius_scale(source_s, seam_s, half_width, value, slope):
    ds = float(source_s) - seam_s
    fade = float(_smooth_fade([source_s], seam_s, half_width)[0])
    return math.exp((value + slope * ds) * fade)


def _single_radius_scale_uv(source_s, seam_s, half_width, value, slope):
    value = np.asarray(value, dtype=np.float64)
    slope = np.asarray(slope, dtype=np.float64)
    if value.ndim == 0:
        value = np.repeat(float(value), 2)
    if slope.ndim == 0:
        slope = np.repeat(float(slope), 2)
    ds = float(source_s) - seam_s
    fade = float(_smooth_fade([source_s], seam_s, half_width)[0])
    return np.exp((value.reshape(2) + slope.reshape(2) * ds) * fade)


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
    factor_l = _single_radius_scale_uv(
        left, source_s, correction_width, value_correction, slope_correction
    )
    factor_r = _single_radius_scale_uv(
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
        radius_mode = str(spec.get("radius_mode", "isotropic")).lower()
        max_anisotropic_change = float(spec.get("max_anisotropic_change", 0.15))
        deformation_weight = float(spec.get("deformation_weight", 0.01))
        angular_profile_order = int(spec.get("angular_profile_order", 3))
        max_angular_change = float(spec.get("max_angular_radius_change", 0.08))
        angular_deformation_weight = float(
            spec.get("angular_deformation_weight", max(deformation_weight, 0.02))
        )
        max_angular_tangent_change = float(
            spec.get("max_angular_tangent_change", 1.0)
        )
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
        if radius_mode not in {"isotropic", "anisotropic", "angular"}:
            raise ValueError(
                f"radius_mode must be 'isotropic', 'anisotropic', or 'angular' in {group_name}"
            )
        if radius_mode in {"anisotropic", "angular"} and radius_source != "surface":
            raise ValueError(
                f"{radius_mode} radius fitting needs radius_source: surface in {group_name}"
            )
        if not (0.0 <= max_anisotropic_change < 1.0 and deformation_weight >= 0.0):
            raise ValueError(f"Invalid anisotropic seam settings in {group_name}")
        if radius_mode == "angular":
            if not (
                2 <= angular_profile_order <= 6
                and 0.0 < max_angular_change < 1.0
                and angular_deformation_weight >= 0.0
                and max_angular_tangent_change > 0.0
            ):
                raise ValueError(f"Invalid angular seam settings in {group_name}")
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
        prepared_items = []
        for index, item in group_items:
            if item.get("mode", "direct") != "direct":
                raise ValueError(
                    f"Group {group_name!r} supports direct adaptations only"
                )
            rigid = bool(item.get("rigid_radius", False))
            wrap = bool(item.get("wrap_radius", False))
            if rigid == wrap:
                raise ValueError(
                    f"Group {group_name!r} needs exactly one of rigid_radius/wrap_radius"
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
            fit_item = dict(item)
            fit_item["_seam_accessory_curve"] = curve_lookup(item["accessory_key"])
            if wrap:
                fit_item["_seam_target_curve"] = curve_lookup(item["target_key"])
            # Validate wrap restrictions up front so a long reconstruction does
            # not fail only after the first adaptation has been inferred.
            _radial_query_scale(
                fit_item, fit_item["_seam_accessory_curve"], 0.5 * (lo + hi)
            )
            prepared_items.append((index, fit_item))

        ordered = sorted(
            prepared_items,
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
            curve_l = left["_seam_accessory_curve"]
            curve_r = right["_seam_accessory_curve"]
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
            corr_uv_l = corr_uv_r = None
            slope_uv_l = slope_uv_r = None
            anisotropic_residual = None
            anisotropic_before = None
            angular_l = angular_r = None
            angular_slope_l = angular_slope_r = None
            angular_residual = angular_before = None
            if do_radius:
                lower, upper = math.log(1.0 - max_change), math.log(1.0 + max_change)
                if surface_radius_available:
                    profile_l = surface_profiles[left["accessory_key"]]
                    profile_r = surface_profiles[right["accessory_key"]]
                    if radius_mode in {"anisotropic", "angular"}:
                        corr_uv_l, corr_uv_r, anisotropic_before, anisotropic_residual = (
                            _fit_anisotropic_radius_pair(
                                left,
                                right,
                                profile_l,
                                profile_r,
                                seam_s,
                                surface_window,
                                surface_bins,
                                radius_quantile,
                                lower,
                                upper,
                                max_anisotropic_change,
                                deformation_weight,
                            )
                        )
                        if match_radius_tangent:
                            slope_uv_l, slope_uv_r = _fit_anisotropic_tangent_pair(
                                left,
                                right,
                                profile_l,
                                profile_r,
                                seam_s,
                                width_l,
                                width_r,
                                step_l,
                                step_r,
                                surface_window,
                                surface_bins,
                                radius_quantile,
                                corr_uv_l,
                                corr_uv_r,
                                lower,
                                upper,
                                deformation_weight,
                            )
                        else:
                            slope_uv_l = np.zeros(2, dtype=np.float64)
                            slope_uv_r = np.zeros(2, dtype=np.float64)

                        if radius_mode == "angular":
                            angular_l, angular_r, angular_before, angular_residual = (
                                _fit_angular_radius_pair(
                                    left,
                                    right,
                                    profile_l,
                                    profile_r,
                                    seam_s,
                                    surface_window,
                                    surface_bins,
                                    radius_quantile,
                                    corr_uv_l,
                                    corr_uv_r,
                                    angular_profile_order,
                                    max_angular_change,
                                    angular_deformation_weight,
                                )
                            )
                            if match_radius_tangent:
                                angular_slope_l, angular_slope_r = (
                                    _fit_angular_tangent_pair(
                                        left,
                                        right,
                                        profile_l,
                                        profile_r,
                                        seam_s,
                                        width_l,
                                        width_r,
                                        step_l,
                                        step_r,
                                        surface_window,
                                        surface_bins,
                                        radius_quantile,
                                        corr_uv_l,
                                        corr_uv_r,
                                        slope_uv_l,
                                        slope_uv_r,
                                        angular_l,
                                        angular_r,
                                        max_angular_tangent_change,
                                        angular_deformation_weight,
                                    )
                                )
                            else:
                                angular_slope_l = np.zeros_like(angular_l)
                                angular_slope_r = np.zeros_like(angular_r)

                        # Keep scalar summaries for backward-compatible diagnostics
                        # and callers that inspect SeamCorrection directly.
                        corr_l = float(np.mean(corr_uv_l))
                        corr_r = float(np.mean(corr_uv_r))
                        slope_corr_l = float(np.mean(slope_uv_l))
                        slope_corr_r = float(np.mean(slope_uv_r))
                    else:
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

            radius_value_l = corr_uv_l if corr_uv_l is not None else corr_l
            radius_value_r = corr_uv_r if corr_uv_r is not None else corr_r
            radius_slope_l = slope_uv_l if slope_uv_l is not None else slope_corr_l
            radius_slope_r = slope_uv_r if slope_uv_r is not None else slope_corr_r
            radius_factor_l = np.exp(radius_value_l)
            radius_factor_r = np.exp(radius_value_r)

            pos_l = pos_r = np.zeros(2, dtype=np.float64)
            pos_slope_l = pos_slope_r = np.zeros(2, dtype=np.float64)
            if do_position:
                profile_l = surface_profiles[left["accessory_key"]]
                profile_r = surface_profiles[right["accessory_key"]]
                center_l = _position_center(
                    left, curve_l, profile_l, seam_s, radius_factor_l, sample_window
                )
                center_r = _position_center(
                    right, curve_r, profile_r, seam_s, radius_factor_r, sample_window
                )
                # Minimum-norm C0 center fit: symmetric displacement in the
                # common avatar transverse plane, converted to each part's
                # physical accessory-local tloc frame.
                gap = center_r - center_l
                query_scale_l = _radial_query_scale(left, curve_l, seam_s)
                query_scale_r = _radial_query_scale(right, curve_r, seam_s)
                pos_l = _bound_translation(
                    _rotate_uv(
                        0.5 * gap * query_scale_l,
                        float(left.get("rot_deg", 0.0)),
                    ),
                    max_position,
                )
                pos_r = _bound_translation(
                    _rotate_uv(
                        -0.5 * gap * query_scale_r,
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
                        radius_value_l,
                        radius_slope_l,
                        width_l,
                        sample_window,
                        step_l,
                    )
                    tangent_r = _position_slope(
                        right,
                        curve_r,
                        profile_r,
                        seam_s,
                        radius_value_r,
                        radius_slope_r,
                        width_r,
                        sample_window,
                        step_r,
                    )
                    tangent_gap = tangent_r - tangent_l
                    raw_l = _rotate_uv(
                        0.5 * tangent_gap * query_scale_l,
                        float(left.get("rot_deg", 0.0)),
                    )
                    raw_r = _rotate_uv(
                        -0.5 * tangent_gap * query_scale_r,
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
                    None if corr_uv_l is None else tuple(np.asarray(corr_uv_l, dtype=float)),
                    None if slope_uv_l is None else tuple(np.asarray(slope_uv_l, dtype=float)),
                    None if angular_l is None else tuple(np.asarray(angular_l, dtype=float)),
                    None if angular_slope_l is None else tuple(np.asarray(angular_slope_l, dtype=float)),
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
                    None if corr_uv_r is None else tuple(np.asarray(corr_uv_r, dtype=float)),
                    None if slope_uv_r is None else tuple(np.asarray(slope_uv_r, dtype=float)),
                    None if angular_r is None else tuple(np.asarray(angular_r, dtype=float)),
                    None if angular_slope_r is None else tuple(np.asarray(angular_slope_r, dtype=float)),
                )
            )
            paired += 1
            if do_radius:
                source_label = "surface" if surface_radius_available else "curve"
                if corr_uv_l is not None:
                    scale_l = np.exp(corr_uv_l)
                    scale_r = np.exp(corr_uv_r)
                    print(
                        f"[seam {group_name}] {left.get('name', left_idx)} + "
                        f"{right.get('name', right_idx)} at src={seam_s:.4f}: "
                        f"radius scale U,V {scale_l.round(4).tolist()} / "
                        f"{scale_r.round(4).tolist()}; dlogR(U,V)/dsrc "
                        f"{np.asarray(slope_uv_l).round(4).tolist()} / "
                        f"{np.asarray(slope_uv_r).round(4).tolist()} ({source_label}, anisotropic)"
                    )
                    if anisotropic_residual is not None:
                        before_pct = 100.0 * math.expm1(anisotropic_before)
                        after_pct = 100.0 * math.expm1(anisotropic_residual)
                        label = "WARNING " if anisotropic_residual > 0.025 else ""
                        print(
                            f"[seam {group_name}] {label}surface-profile mismatch "
                            f"{before_pct:.1f}% -> {after_pct:.1f}% after anisotropic radius fit"
                        )
                    if angular_l is not None:
                        print(
                            f"[seam {group_name}] angular profile harmonics 2.."
                            f"{angular_profile_order} coeffs "
                            f"{np.asarray(angular_l).round(4).tolist()} / "
                            f"{np.asarray(angular_r).round(4).tolist()}; "
                            f"dcoeff/dsrc {np.asarray(angular_slope_l).round(4).tolist()} / "
                            f"{np.asarray(angular_slope_r).round(4).tolist()}"
                        )
                        before_pct = 100.0 * math.expm1(angular_before)
                        after_pct = 100.0 * math.expm1(angular_residual)
                        label = "WARNING " if angular_residual > 0.025 else ""
                        print(
                            f"[seam {group_name}] {label}surface-profile mismatch "
                            f"{before_pct:.1f}% -> {after_pct:.1f}% after angular profile fit"
                        )
                else:
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


def _seam_radius_logs_uv(seam):
    if seam.log_radius_scale_uv is None:
        value = np.repeat(float(seam.log_radius_scale), 2)
    else:
        value = np.asarray(seam.log_radius_scale_uv, dtype=np.float64).reshape(2)
    if seam.log_radius_slope_uv is None:
        slope = np.repeat(float(seam.log_radius_slope), 2)
    else:
        slope = np.asarray(seam.log_radius_slope_uv, dtype=np.float64).reshape(2)
    return value, slope


def radius_scale_uv_at(source_coords, corrections):
    """Per-axis C1 geometry scale in accessory-local physical U/V axes."""
    s = np.asarray(source_coords, dtype=np.float64).reshape(-1)
    log_scale = np.zeros((len(s), 2), dtype=np.float64)
    for seam in corrections:
        value, slope = _seam_radius_logs_uv(seam)
        fade = _smooth_fade(s, seam.source_s, seam.half_width)
        ds = s - seam.source_s
        log_scale += (value[None, :] + ds[:, None] * slope[None, :]) * fade[:, None]
    return np.exp(log_scale)


def radius_scale_at(source_coords, corrections):
    """Backward-compatible scalar scale (geometric mean of fitted U/V scales)."""
    uv = radius_scale_uv_at(source_coords, corrections)
    return np.sqrt(uv[:, 0] * uv[:, 1])

def angular_radius_scale_at(source_coords, physical_angles, corrections):
    """C1 angular radial scale evaluated in accessory-local physical angle."""
    s = np.asarray(source_coords, dtype=np.float64).reshape(-1)
    theta = np.asarray(physical_angles, dtype=np.float64).reshape(-1)
    if s.shape != theta.shape:
        raise ValueError("Angular seam source/angle count mismatch")
    log_scale = np.zeros_like(s)
    for seam in corrections:
        if seam.angular_log_radius_coeffs is None:
            continue
        value = np.asarray(seam.angular_log_radius_coeffs, dtype=np.float64).reshape(-1)
        if seam.angular_log_radius_slope_coeffs is None:
            slope = np.zeros_like(value)
        else:
            slope = np.asarray(
                seam.angular_log_radius_slope_coeffs, dtype=np.float64
            ).reshape(value.shape)
        basis = _angular_basis(theta, value.size // 2 + 1)
        ds = s - seam.source_s
        fade = _smooth_fade(s, seam.source_s, seam.half_width)
        coeffs = value[None, :] + ds[:, None] * slope[None, :]
        log_scale += np.sum(basis * coeffs, axis=1) * fade
    return np.exp(log_scale)


def warp_accessory_queries(accessory_data, avatar_data, corrections):
    """Apply local translation and profile warps before model inference.

    Inverse-warp order matches the fitted forward geometry:
      translation -> angular radial scale -> anisotropic U/V scale.
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
    angular_enabled = any(
        seam.angular_log_radius_coeffs is not None
        for seam in corrections
    )
    if position_enabled or angular_enabled:
        radius = np.asarray(accessory_data["radius"])
        if radius.shape != (len(local), 2):
            raise ValueError(
                "Position/angular seam fitting needs per-point accessory U/V radii"
            )

    chunk_size = 262144
    for start in range(0, len(local), chunk_size):
        stop = min(start + chunk_size, len(local))
        s = source_s[start:stop]
        piece = local[start:stop]
        if position_enabled or angular_enabled:
            radii = radius[start:stop]
            if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
                raise ValueError("Invalid accessory radius for seam fitting")
        if position_enabled:
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

        if angular_enabled:
            # The angular field is defined in accessory-local *physical* angle
            # after translation and before the U/V anisotropic inverse warp.
            physical_angle = np.arctan2(
                piece[:, 2] * radii[:, 1],
                piece[:, 1] * radii[:, 0],
            )
            angular_scale = angular_radius_scale_at(s, physical_angle, corrections)
            np.divide(piece[:, 1], angular_scale, out=piece[:, 1], casting="unsafe")
            np.divide(piece[:, 2], angular_scale, out=piece[:, 2], casting="unsafe")

        scale_uv = radius_scale_uv_at(s, corrections)
        np.divide(piece[:, 1], scale_uv[:, 0], out=piece[:, 1], casting="unsafe")
        np.divide(piece[:, 2], scale_uv[:, 1], out=piece[:, 2], casting="unsafe")

        anisotropic_enabled = any(
            seam.log_radius_scale_uv is not None or seam.log_radius_slope_uv is not None
            for seam in corrections
        )
        if position_enabled or anisotropic_enabled or angular_enabled:
            # Translation and anisotropic scaling both change rho and angle, so
            # rebuild support proxies from the warped model-local queries.
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
            # Preserve exactly the pre-existing isotropic radius-only behavior.
            scale = scale_uv[:, 0]
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
