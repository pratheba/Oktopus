"""Post-inference seam blending for already-adapted SDF pieces.

This module intentionally does *not* modify adaptation query coordinates.
It is useful for wrap-radius constructions where the original mapping/tilt
must remain unchanged and only the implicit fields should transition inside
an explicit source-curve overlap.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PostSDFSeamPair:
    group: str
    left_index: int
    right_index: int
    source_lo: float
    source_hi: float
    edge_hold_fraction: float


def _source_interval(item):
    a = float(item["src_0"])
    b = float(item["src_1"])
    return min(a, b), max(a, b)


def build_post_sdf_seam_plan(adaptations, group_settings):
    """Build an execution-order plan for seam-only implicit crossfades.

    A ``blend_groups.<name>.seam_blend`` block opts a group into this mode.
    The adaptations themselves are not geometrically warped.  Instead, when
    the later piece in source-curve order is accumulated, its SDF is smoothly
    crossfaded with the already accumulated previous piece across their
    explicit ``src`` overlap.

    To keep the combination streaming and memory bounded, seam-blend pieces
    must appear in YAML in increasing source-interval order.
    """
    if not isinstance(group_settings, dict):
        return {}

    enabled = {}
    for name, group_spec in group_settings.items():
        if not isinstance(group_spec, dict):
            continue
        spec = group_spec.get("seam_blend")
        if isinstance(spec, dict) and bool(spec.get("enabled", False)):
            enabled[str(name)] = spec

    if not enabled:
        return {}

    by_group = {}
    for index, item in enumerate(adaptations):
        group = str(item.get("blend_group", "none"))
        if group in enabled:
            by_group.setdefault(group, []).append((index, item))

    missing = set(enabled) - set(by_group)
    if missing:
        raise ValueError(
            f"Configured seam_blend groups without adaptations: {sorted(missing)}"
        )

    plan = {}
    for group_name, spec in enabled.items():
        mode = str(spec.get("mode", "overlap_crossfade")).lower()
        if mode not in ("overlap_crossfade", "overlap_sdf"):
            raise ValueError(
                f"Group {group_name!r}: unknown seam_blend mode {mode!r}; "
                "expected 'overlap_crossfade'"
            )
        edge = float(spec.get("edge_hold_fraction", 0.15))
        if not (0.0 <= edge < 0.5):
            raise ValueError(
                f"Group {group_name!r}: edge_hold_fraction must be in [0, 0.5)"
            )

        group_items = by_group[group_name]
        if len(group_items) < 2:
            raise ValueError(
                f"Group {group_name!r}: seam_blend needs at least two adaptations"
            )

        targets = {item["target_key"] for _, item in group_items}
        if len(targets) != 1:
            raise ValueError(
                f"Group {group_name!r}: seam_blend mixes avatar targets: "
                f"{sorted(targets)}"
            )

        # Streaming combination needs the left/earlier interval to have been
        # accumulated before the right/later interval arrives.
        yaml_order = [index for index, _ in group_items]
        spatial_order = [
            index for index, _ in sorted(
                group_items,
                key=lambda pair: (
                    _source_interval(pair[1])[0],
                    _source_interval(pair[1])[1],
                ),
            )
        ]
        if yaml_order != spatial_order:
            raise ValueError(
                f"Group {group_name!r}: seam_blend adaptations must be listed "
                "in increasing source-curve order for streaming combination; "
                f"got indices {yaml_order}, expected {spatial_order}"
            )

        ordered = sorted(
            group_items,
            key=lambda pair: (
                _source_interval(pair[1])[0],
                _source_interval(pair[1])[1],
            ),
        )
        for (left_idx, left), (right_idx, right) in zip(ordered, ordered[1:]):
            left_lo, left_hi = _source_interval(left)
            right_lo, right_hi = _source_interval(right)
            lo = max(left_lo, right_lo)
            hi = min(left_hi, right_hi)
            if not hi > lo:
                raise ValueError(
                    f"Group {group_name!r}: seam_blend requires a positive src "
                    f"overlap between {left.get('name', left_idx)!r} and "
                    f"{right.get('name', right_idx)!r}; got [{lo}, {hi}]"
                )
            if right_idx in plan:
                raise ValueError(
                    f"Adaptation index {right_idx} belongs to multiple post-SDF seams"
                )
            plan[right_idx] = PostSDFSeamPair(
                group=group_name,
                left_index=left_idx,
                right_index=right_idx,
                source_lo=lo,
                source_hi=hi,
                edge_hold_fraction=edge,
            )

    return plan


def overlap_crossfade_weights(source_s, pair):
    """C1 smooth left->right weights across a source-curve overlap.

    ``edge_hold_fraction`` preserves an exact copy of the left field near the
    lower overlap boundary and the right field near the upper boundary.  The
    middle transition is cubic smoothstep, so both endpoints have zero slope.
    """
    s = np.asarray(source_s, dtype=np.float64).reshape(-1)
    width = float(pair.source_hi - pair.source_lo)
    if width <= 0.0:
        raise ValueError("Post-SDF seam overlap must have positive width")

    x = (s - pair.source_lo) / width
    edge = float(pair.edge_hold_fraction)
    if edge > 0.0:
        x = (x - edge) / (1.0 - 2.0 * edge)
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def overlap_crossfade_sdf(left_sdf, right_sdf, source_s, pair):
    """Interpolate two already-generated implicit fields only at their seam.

    The returned field equals the left field at the low end of the overlap,
    equals the right field at the high end, and changes with zero derivative at
    both held edges.  No accessory query coordinates or wrap/tilt parameters
    are modified.
    """
    left = np.asarray(left_sdf)
    right = np.asarray(right_sdf)
    if left.shape != right.shape:
        raise ValueError(
            f"Post-SDF seam field shape mismatch: {left.shape} vs {right.shape}"
        )
    w = overlap_crossfade_weights(source_s, pair)
    if w.size != left.size:
        raise ValueError(
            f"Post-SDF seam coordinate/value mismatch: {w.size} vs {left.size}"
        )
    return left * (1.0 - w) + right * w
