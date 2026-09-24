"""Lightweight, group-local radial seam fitting for direct rigid-radius adaptations.

The fit uses accessory skeleton radius as a *proxy* for actual inferred mesh
radius. It modifies inference query coordinates, not SDF output values. This
module deliberately has no dependency on a DCSDD grid or GPU.
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
        if not bool(spec.get("optimize_radius", True)):
            continue
        if spec.get("optimize_position", False):
            raise ValueError("optimize_position is not implemented; keep it false")
        max_change = float(spec.get("max_radius_change", 0.20))
        transition = float(spec.get("transition_fraction", 0.20))
        max_gap = float(spec.get("max_gap", 0.0))
        if not (0.0 < max_change < 1.0 and 0.0 < transition <= 0.5 and max_gap >= 0.0):
            raise ValueError(f"Invalid seam_optimization parameters in {group_name}")
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
        ordered = sorted(group_items, key=lambda p: min(float(p[1]["src_0"]), float(p[1]["src_1"])))
        paired = 0
        for (left_idx, left), (right_idx, right) in zip(ordered, ordered[1:]):
            left_hi = max(float(left["src_0"]), float(left["src_1"]))
            right_lo = min(float(right["src_0"]), float(right["src_1"]))
            overlap = left_hi - right_lo
            # Nonoverlapping pieces must be positioned first; changing radius cannot fill a gap.
            if overlap < -max_gap:
                raise ValueError(f"Group {group_name!r}: gap between {left.get('name')} and "
                                 f"{right.get('name')}; radius-only fitting cannot join them")
            if overlap < 0.0:
                raise ValueError(f"Group {group_name!r}: source intervals do not overlap")
            # Join halfway across the shared interval, so both are sampled at
            # the same avatar arc parameter, regardless of source direction.
            seam_s = 0.5 * (left_hi + right_lo)
            curve_l, curve_r = curve_lookup(left["accessory_key"]), curve_lookup(right["accessory_key"])
            log_l = _log_radius(left, curve_l, seam_s)
            log_r = _log_radius(right, curve_r, seam_s)
            # Closed-form minimum-norm log-scale fit under equal final radii.
            # Symmetric correction avoids shrinking only one piece.
            delta = 0.5 * (log_r - log_l)
            lower, upper = math.log(1.0 - max_change), math.log(1.0 + max_change)
            corr_l = min(upper, max(lower, delta))
            corr_r = min(upper, max(lower, -delta))
            residual = abs((log_l + corr_l) - (log_r + corr_r))
            if residual > 0.025:
                print(f"[seam {group_name}] WARNING residual radius mismatch "
                      f"{100.0 * math.expm1(residual):.1f}% exceeds allowed correction")
            width_l = transition * abs(float(left["src_1"]) - float(left["src_0"]))
            width_r = transition * abs(float(right["src_1"]) - float(right["src_0"]))
            slope_l = _radius_slope(left, curve_l, seam_s, min(width_l / 5.0, 0.005))
            slope_r = _radius_slope(right, curve_r, seam_s, min(width_r / 5.0, 0.005))
            target_slope = 0.5 * (slope_l + slope_r)
            result.setdefault(left_idx, []).append(SeamCorrection(
                seam_s, width_l, corr_l, target_slope - slope_l))
            result.setdefault(right_idx, []).append(SeamCorrection(
                seam_s, width_r, corr_r, target_slope - slope_r))
            paired += 1
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
    """Apply inverse radial query warp before SDF inference.

    A larger fitted sleeve corresponds to *smaller* queried model-local
    radial coordinates. Global XYZ samples and grid indexing remain intact.
    """
    if not corrections:
        return accessory_data
    source_s = np.asarray(avatar_data["coords"], dtype=np.float64).reshape(-1)
    scale = radius_scale_at(source_s, corrections)
    if scale.shape[0] != accessory_data["samples_local"].shape[0]:
        raise ValueError("Seam scale/query point count mismatch")
    # filter_grid_adapt creates this query array for the current adaptation.
    # Mutate it in place to avoid retaining another full N x 3 copy at r=512.
    local = np.asarray(accessory_data["samples_local"])
    if not local.flags.writeable:
        local = local.copy()
        accessory_data["samples_local"] = local
    np.divide(local[:, 1], scale, out=local[:, 1], casting="unsafe")
    np.divide(local[:, 2], scale, out=local[:, 2], casting="unsafe")
    # Keep the support clamp's radial proxy consistent with the warped query.
    for key in ("rho_n", "u_n", "v_n"):
        if key in accessory_data:
            a = np.asarray(accessory_data[key])
            if a.size == scale.size:
                if not a.flags.writeable:
                    a = a.copy()
                    accessory_data[key] = a
                np.divide(a, scale.reshape(a.shape), out=a, casting="unsafe")
    return accessory_data
