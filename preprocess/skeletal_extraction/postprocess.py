#!/usr/bin/env python3
import argparse
import json
import os
import math
from copy import deepcopy
import numpy as np


def seg_lengths(poly):
    if len(poly) < 2:
        return np.zeros(1), 0.0
    ds = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    return s, float(s[-1])


def interp_poly(poly, t):
    s, total = seg_lengths(poly)
    q = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0) * max(total, 1e-12)
    if total <= 1e-12:
        return np.repeat(poly[:1], len(q), axis=0)

    out = np.zeros((len(q), 3), dtype=np.float64)
    for i, qi in enumerate(q):
        idx = np.searchsorted(s, qi, side="right") - 1
        idx = max(0, min(idx, len(poly) - 2))
        a = (qi - s[idx]) / max(s[idx + 1] - s[idx], 1e-12)
        out[i] = (1 - a) * poly[idx] + a * poly[idx + 1]
    return out


def tangent_at(poly, at_start=True):
    if len(poly) < 2:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    v = poly[1] - poly[0] if at_start else poly[-1] - poly[-2]
    return v / max(np.linalg.norm(v), 1e-12)


def nearest_polyline_projection(poly, pts):
    pts = np.asarray(pts, dtype=np.float64)
    if len(poly) < 2 or len(pts) == 0:
        return (
            np.zeros((len(pts), 3), dtype=np.float64),
            np.zeros(len(pts), dtype=np.float64),
            np.zeros(len(pts), dtype=int),
            np.full(len(pts), np.inf, dtype=np.float64),
        )

    seg = np.diff(poly, axis=0)
    seglen2 = np.sum(seg * seg, axis=1)
    cs, total = seg_lengths(poly)

    best_d2 = np.full(len(pts), np.inf, dtype=np.float64)
    best_s = np.zeros(len(pts), dtype=np.float64)
    best_idx = np.zeros(len(pts), dtype=int)
    best_p = np.zeros((len(pts), 3), dtype=np.float64)

    for i in range(len(seg)):
        a = poly[i]
        v = seg[i]
        denom = max(seglen2[i], 1e-12)
        w = pts - a[None, :]
        t = np.clip(np.sum(w * v[None, :], axis=1) / denom, 0.0, 1.0)
        proj = a[None, :] + t[:, None] * v[None, :]
        d2 = np.sum((pts - proj) ** 2, axis=1)
        m = d2 < best_d2
        best_d2[m] = d2[m]
        best_p[m] = proj[m]
        best_s[m] = cs[i] + t[m] * math.sqrt(seglen2[i])
        best_idx[m] = i

    return best_p, best_s / max(total, 1e-12), best_idx, np.sqrt(best_d2)

def radii_from_support_local_frame(
    keypoints,
    frame_t,
    frame_u,
    frame_v,
    support_points,
    q_train=0.95,
    q_wrap=0.98,
    q_cyl=1.0,
    cyl_margin=0.05,
    cyl_relative_margin=0.2,
    slab_half_width=None
):
    keypoints = np.asarray(keypoints, dtype=np.float64)
    frame_t = np.asarray(frame_t, dtype=np.float64)
    frame_u = np.asarray(frame_u, dtype=np.float64)
    frame_v = np.asarray(frame_v, dtype=np.float64)
    support_points = np.asarray(support_points, dtype=np.float64)

    K = len(keypoints)
    if len(support_points) == 0 or K == 0:
        z = np.full(K, 0.01, dtype=np.float64)
        return z, z.copy(), z.copy()

    # default slab size from local spacing
    if slab_half_width is None:
        if K >= 2:
            ds = np.linalg.norm(np.diff(keypoints, axis=0), axis=1)
            slab_half_width = float(max(0.5 * np.median(ds), 1e-3))
        else:
            slab_half_width = 0.01

    r_train = np.full(K, np.nan, dtype=np.float64)
    r_wrap = np.full(K, np.nan, dtype=np.float64)
    r_cyl = np.full(K, np.nan, dtype=np.float64)

    for i in range(K):
        p = keypoints[i]
        t = frame_t[i]
        u = frame_u[i]
        v = frame_v[i]

        rel = support_points - p[None, :]

        # local coordinates in current keypoint frame
        w = rel @ t
        uu = rel @ u
        vv = rel @ v

        # take nearby points in a slab around this keypoint
        m = np.abs(w) <= slab_half_width
        if not np.any(m):
            continue

        rho = np.sqrt(uu[m] ** 2 + vv[m] ** 2)
        rho = rho[np.isfinite(rho)]
        if len(rho) == 0:
            continue

        r_train[i] = np.quantile(rho, q_train)
        r_wrap[i] = np.quantile(rho, q_wrap)

        base_cyl = np.quantile(rho, q_cyl)
        r_cyl[i] = base_cyl * (1.0 + cyl_relative_margin) + cyl_margin

    def fill_nan(r, fallback=0.01):
        good = np.isfinite(r)
        x = np.arange(len(r))
        if np.any(good):
            return np.interp(x, x[good], r[good])
        return np.full(len(r), fallback, dtype=np.float64)

    r_train = fill_nan(r_train, fallback=0.01)
    r_wrap = fill_nan(r_wrap, fallback=0.01)
    r_cyl = fill_nan(r_cyl, fallback=0.02)

    # keep ordering sensible
    r_wrap = np.maximum(r_wrap, r_train)
    r_cyl = np.maximum(r_cyl, r_wrap)

    return r_train, r_wrap, r_cyl

def radii_from_support(
    keypoints,
    support_points,
    q_train=0.95,
    q_wrap=0.97,
    q_cyl=1.0,
    cyl_margin=0.1,
    cyl_relative_margin=0.1,
):
    keypoints = np.asarray(keypoints, dtype=np.float64)
    support_points = np.asarray(support_points, dtype=np.float64)

    if len(support_points) == 0:
        z = np.full(len(keypoints), 0.01, dtype=np.float64)
        return z, z.copy(), z.copy()

    _, sp, _, dist = nearest_polyline_projection(keypoints, support_points)

    bins = np.linspace(0.0, 1.0, len(keypoints) + 1)
    bid = np.clip(np.digitize(sp, bins) - 1, 0, len(keypoints) - 1)

    r_train = np.full(len(keypoints), np.nan, dtype=np.float64)
    r_wrap = np.full(len(keypoints), np.nan, dtype=np.float64)
    r_cyl = np.full(len(keypoints), np.nan, dtype=np.float64)

    for i in range(len(keypoints)):
        m = bid == i
        if not np.any(m):
            continue

        d = dist[m]
        d = d[np.isfinite(d)]
        if len(d) == 0:
            continue

        r_train[i] = np.quantile(d, q_train)
        r_wrap[i] = np.quantile(d, q_wrap)

        base_cyl = np.quantile(d, q_cyl)
        r_cyl[i] = base_cyl * (1.0 + cyl_relative_margin) + cyl_margin

    def fill_nan(r, fallback=0.01):
        good = np.isfinite(r)
        x = np.arange(len(r))
        if np.any(good):
            return np.interp(x, x[good], r[good])
        return np.full(len(r), fallback, dtype=np.float64)

    r_train = fill_nan(r_train, fallback=0.01)
    r_wrap = fill_nan(r_wrap, fallback=0.01)
    r_cyl = fill_nan(r_cyl, fallback=0.02)

    # enforce monotonic safety relationship
    r_wrap = np.maximum(r_wrap, r_train)
    r_cyl = np.maximum(r_cyl, r_wrap)

    return r_train, r_wrap, r_cyl

def straighten_curve_line(poly):
    n = len(poly)
    if n < 2:
        return poly.copy()
    a = poly[0]
    b = poly[-1]
    ts = np.linspace(0.0, 1.0, n)[:, None]
    return (1.0 - ts) * a[None, :] + ts * b[None, :]


def straighten_curve_blend(poly, alpha=0.5):
    line = straighten_curve_line(poly)
    return (1.0 - float(alpha)) * poly + float(alpha) * line

def maybe_straighten_curve(poly, mode=None, alpha=0.5):
    if mode is None:
        return poly
    mode = str(mode).lower()

    if mode in ("line", "straight_line"):
        return straighten_curve_line(poly)

    if mode in ("blend", "straight"):
        return straighten_curve_blend(poly, alpha=alpha)

    return poly

def smooth_poly(poly, window=5, iters=1):
    out = poly.copy()
    r = max(1, window // 2)
    for _ in range(max(1, iters)):
        new = out.copy()
        for i in range(1, len(out) - 1):
            lo, hi = max(0, i - r), min(len(out), i + r + 1)
            new[i] = out[lo:hi].mean(axis=0)
        out = new
    return out


def compute_tangents(poly):
    poly = np.asarray(poly, dtype=np.float64)
    K = len(poly)
    T = np.zeros((K, 3), dtype=np.float64)

    if K == 1:
        T[0] = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        return T

    for i in range(K):
        if i == 0:
            d = poly[1] - poly[0]
        elif i == K - 1:
            d = poly[-1] - poly[-2]
        else:
            d = poly[i + 1] - poly[i - 1]

        n = np.linalg.norm(d)
        T[i] = d / max(n, 1e-12)

    return T


def orthogonal_vector(v):
    v = np.asarray(v, dtype=np.float64)
    if abs(v[0]) < 0.9:
        a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        a = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    u = a - np.dot(a, v) * v
    return u / max(np.linalg.norm(u), 1e-12)


def compute_parallel_transport_frames(poly):
    poly = np.asarray(poly, dtype=np.float64)
    T = compute_tangents(poly)
    K = len(poly)

    U = np.zeros((K, 3), dtype=np.float64)
    V = np.zeros((K, 3), dtype=np.float64)

    U[0] = orthogonal_vector(T[0])
    V[0] = np.cross(T[0], U[0])
    V[0] /= max(np.linalg.norm(V[0]), 1e-12)
    U[0] = np.cross(V[0], T[0])
    U[0] /= max(np.linalg.norm(U[0]), 1e-12)

    for i in range(1, K):
        t_prev = T[i - 1]
        t_cur = T[i]

        axis = np.cross(t_prev, t_cur)
        axis_n = np.linalg.norm(axis)
        dotv = np.clip(np.dot(t_prev, t_cur), -1.0, 1.0)

        if axis_n < 1e-10:
            U[i] = U[i - 1]
        else:
            axis = axis / axis_n
            angle = np.arccos(dotv)

            u_prev = U[i - 1]
            U[i] = (
                u_prev * np.cos(angle)
                + np.cross(axis, u_prev) * np.sin(angle)
                + axis * np.dot(axis, u_prev) * (1.0 - np.cos(angle))
            )

        U[i] = U[i] - np.dot(U[i], t_cur) * t_cur
        U[i] /= max(np.linalg.norm(U[i]), 1e-12)

        V[i] = np.cross(t_cur, U[i])
        V[i] /= max(np.linalg.norm(V[i]), 1e-12)

        U[i] = np.cross(V[i], t_cur)
        U[i] /= max(np.linalg.norm(U[i]), 1e-12)

    frames = np.stack([T, U, V], axis=1)  # (K,3,3)
    return T, U, V, frames

def crop_poly(poly, s0, s1):
    n = max(2, int(round((len(poly) - 1) * (float(s1) - float(s0)))) + 1)
    return interp_poly(poly, np.linspace(float(s0), float(s1), n))


def split_poly(poly, fracs):
    cuts = [0.0] + sorted([float(x) for x in fracs if 0.0 < float(x) < 1.0]) + [1.0]
    parts = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        n = max(2, int(round((len(poly) - 1) * (b - a))) + 1)
        parts.append(interp_poly(poly, np.linspace(a, b, n)))
    return parts


def merge_polys_old(polys):
    if len(polys) == 1:
        return polys[0].copy()

    used = [False] * len(polys)
    used[0] = True
    order = [0]

    while len(order) < len(polys):
        cur = polys[order[-1]]
        end = cur[-1]
        best = None
        bestd = 1e18
        bestrev = False

        for i, p in enumerate(polys):
            if used[i]:
                continue
            d1 = np.linalg.norm(end - p[0])
            d2 = np.linalg.norm(end - p[-1])
            if d1 < bestd:
                best, bestd, bestrev = i, d1, False
            if d2 < bestd:
                best, bestd, bestrev = i, d2, True

        used[best] = True
        if bestrev:
            polys[best] = polys[best][::-1]
        order.append(best)

    merged = polys[order[0]].copy()
    for idx in order[1:]:
        p = polys[idx]
        if np.linalg.norm(merged[-1] - p[0]) > np.linalg.norm(merged[-1] - p[-1]):
            p = p[::-1]
        merged = np.vstack([merged, p[1:]])
    return merged


def merge_polys(polys, seam_window=4):
    if len(polys) == 1:
        return polys[0].copy()

    used = [False] * len(polys)
    used[0] = True
    order = [0]

    while len(order) < len(polys):
        cur = polys[order[-1]]
        end = cur[-1]
        best = None
        bestd = 1e18
        bestrev = False

        for i, p in enumerate(polys):
            if used[i]:
                continue
            d1 = np.linalg.norm(end - p[0])
            d2 = np.linalg.norm(end - p[-1])
            if d1 < bestd:
                best, bestd, bestrev = i, d1, False
            if d2 < bestd:
                best, bestd, bestrev = i, d2, True

        used[best] = True
        if bestrev:
            polys[best] = polys[best][::-1]
        order.append(best)

    merged = polys[order[0]].copy()

    for idx in order[1:]:
        p = polys[idx]
        if np.linalg.norm(merged[-1] - p[0]) > np.linalg.norm(merged[-1] - p[-1]):
            p = p[::-1]

        # local seam smoothing
        w = min(seam_window, len(merged) - 1, len(p) - 1)
        if w >= 2:
            left = merged[-(w + 1):].copy()
            right = p[:(w + 1)].copy()

            a = left[0]
            b = right[-1]

            # straight transition between outer seam anchors
            bridge = np.array([
                (1.0 - t) * a + t * b
                for t in np.linspace(0.0, 1.0, len(left))
            ])

            # blend existing seam neighborhoods toward bridge
            alpha = np.linspace(0.0, 1.0, len(left))[:, None]
            left_new = (1.0 - 0.5 * alpha) * left + (0.5 * alpha) * bridge
            right_new = (0.5 * (1.0 - alpha)) * bridge + (1.0 - 0.5 * (1.0 - alpha)) * right

            merged[-(w + 1):] = left_new
            p[:(w + 1)] = right_new

        merged = np.vstack([merged, p[1:]])

    return merged

def local_support_scale(points, default=0.02):
    if len(points) < 3:
        return default
    c = points.mean(axis=0)
    r = np.linalg.norm(points - c[None, :], axis=1)
    return float(max(default, np.quantile(r, 0.8)))


def bury_depth(endpoint, tangent, support_points):
    if len(support_points) == 0:
        return 0.0
    vals = np.dot(support_points - endpoint[None, :], tangent)
    pos = vals[vals > 0]
    return float(np.quantile(pos, 0.8)) if len(pos) else 0.0

def endpoint_spacing(poly, at_start=True, k=3):
    if len(poly) < 2:
        return 0.01

    if at_start:
        part = poly[:min(len(poly), k + 1)]
    else:
        part = poly[max(0, len(poly) - (k + 1)):]

    ds = np.linalg.norm(np.diff(part, axis=0), axis=1)
    if len(ds) == 0:
        return 0.01

    return float(np.median(ds))


def extend_curve(poly, support_points, extend_start=True, extend_end=True,
                 extend_radius_alpha=2.5, extend_burial_alpha=1.0,
                 extend_min=0.01, target_spacing=0.01):
    pieces = [poly]

    if extend_start:
        endpoint = poly[0]
        tan = -tangent_at(poly, True)
        _, sp, _, _ = nearest_polyline_projection(poly, support_points) if len(support_points) else (None, np.zeros(0), None, None)
        local = support_points[sp < 0.15] if len(support_points) else np.zeros((0, 3), dtype=np.float64)
        ext_len = max(
            extend_min,
            extend_radius_alpha * local_support_scale(local, target_spacing),
            extend_burial_alpha * bury_depth(endpoint, tan, local),
        )
        local_spacing = endpoint_spacing(poly, at_start=True, k=3)
        spacing = min(float(target_spacing), local_spacing)
        n = max(3, int(math.ceil(ext_len / max(spacing, 1e-4))))
        pts = np.array(
            [endpoint + tan * ext_len * ((i + 1) / n) for i in range(n)],
            dtype=np.float64
        )[::-1]
        #n = max(1, int(math.ceil(ext_len / max(target_spacing, 1e-4))))
        #pts = np.array([endpoint + tan * ext_len * ((i + 1) / n) for i in range(n)], dtype=np.float64)[::-1]
        pieces = [pts, poly]

    if extend_end:
        base = np.vstack(pieces)
        endpoint = base[-1]
        tan = tangent_at(base, False)
        _, sp, _, _ = nearest_polyline_projection(base, support_points) if len(support_points) else (None, np.zeros(0), None, None)
        local = support_points[sp > 0.85] if len(support_points) else np.zeros((0, 3), dtype=np.float64)
        ext_len = max(
            extend_min,
            extend_radius_alpha * local_support_scale(local, target_spacing),
            extend_burial_alpha * bury_depth(endpoint, tan, local),
        )
        local_spacing = endpoint_spacing(base, at_start=False, k=3)
        spacing = min(float(target_spacing), local_spacing)
        n = max(3, int(math.ceil(ext_len / max(spacing, 1e-4))))
        pts = np.array(
            [endpoint + tan * ext_len * ((i + 1) / n) for i in range(n)],
            dtype=np.float64
        )
        #n = max(1, int(math.ceil(ext_len / max(target_spacing, 1e-4))))
        #pts = np.array([endpoint + tan * ext_len * ((i + 1) / n) for i in range(n)], dtype=np.float64)
        pieces = [base, pts]

    return np.vstack(pieces)


def reassign(seg):
    #seg["keypoints"], dominant_axis = canonicalize_curve_order(seg["keypoints"])
    #seg["metadata"]["dominant_axis"] = dominant_axis
    pts = np.asarray(seg.get("surface_points_all", np.zeros((0, 3))), dtype=np.float64)
    key = np.asarray(seg["keypoints"], dtype=np.float64)
    T, U, V, frames = compute_parallel_transport_frames(key)

    _, point_s, point_key_ids, _ = nearest_polyline_projection(key, pts)
    r_train, r_wrap, r_cyl = radii_from_support(key, pts)
    #r_train, r_wrap, r_cyl = radii_from_support_local_frame(key, T, U, V, pts)

    seg["point_s"] = point_s
    seg["point_key_ids"] = point_key_ids
    seg["radius_train"] = r_train
    seg["radius_wrap"] = r_wrap
    seg["radius_cylinder"] = r_cyl
    seg["surface_points_owned"] = pts.copy()
    seg["surface_points_shared"] = np.zeros((0, 3), dtype=np.float64)
    seg["frame_t"] = T
    seg["frame_u"] = U
    seg["frame_v"] = V
    seg["frames"] = frames

    meta = dict(seg.get("metadata", {}))
    meta["n_total_keypoints"] = int(len(key))
    meta["has_local_frames"] = True
    seg["metadata"] = meta
    return seg


def canonicalize_curve_order(poly):
    p0 = np.asarray(poly[0], dtype=np.float64)
    p1 = np.asarray(poly[-1], dtype=np.float64)
    d = p1 - p0
    ax = np.abs(d)

    if ax[0] >= ax[1] and ax[0] >= ax[2]:
        # X-dominant: left -> right, smaller x is start
        k0 = (p0[0], -p0[1], -p0[2])
        k1 = (p1[0], -p1[1], -p1[2])
        flipped = k1 < k0
        return (poly[::-1] if flipped else poly), "x", flipped

    elif ax[1] >= ax[0] and ax[1] >= ax[2]:
        # Y-dominant: top -> bottom, larger y is start
        k0 = (-p0[1], p0[0], -p0[2])
        k1 = (-p1[1], p1[0], -p1[2])
        flipped = k1 < k0
        return (poly[::-1] if flipped else poly), "y", flipped

    else:
        # Z-dominant: front -> back, larger z is start
        k0 = (-p0[2], -p0[1], p0[0])
        k1 = (-p1[2], -p1[1], p1[0])
        flipped = k1 < k0
        return (poly[::-1] if flipped else poly), "z", flipped


def orthogonal_vector(v):
    v = np.asarray(v, dtype=np.float64)
    if abs(v[0]) < 0.9:
        a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        a = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = a - np.dot(a, v) * v
    return u / max(np.linalg.norm(u), 1e-12)


def centroid_align(poly, support_points, scale=0.6, max_shift_frac=0.35, smooth_win=5):
    if len(poly) < 2 or len(support_points) == 0:
        return poly.copy()

    _, s_norm, _, _ = nearest_polyline_projection(poly, support_points)
    s, total = seg_lengths(poly)
    t = s / max(total, 1e-12)
    shifts = np.zeros_like(poly)

    for i in range(len(poly)):
        m = (s_norm >= max(0.0, t[i] - 0.06)) & (s_norm <= min(1.0, t[i] + 0.06))
        if m.sum() < 5:
            continue

        c = support_points[m].mean(axis=0)
        delta = c - poly[i]

        tan = poly[min(i + 1, len(poly) - 1)] - poly[max(i - 1, 0)]
        tan = tan / max(np.linalg.norm(tan), 1e-12)

        # NB-plane shift only
        delta = delta - np.dot(delta, tan) * tan

        local = support_points[m]
        if len(local) >= 3:
            lc = local.mean(axis=0)
            lr = np.linalg.norm(local - lc[None, :], axis=1)
            local_scale = float(max(0.02, np.quantile(lr, 0.8)))
        else:
            local_scale = 0.02

        mx = max_shift_frac * local_scale
        n = np.linalg.norm(delta)
        if n > mx:
            delta = delta / n * mx

        shifts[i] = scale * delta

    return poly + smooth_poly(shifts, window=smooth_win, iters=1)


def recenter_and_reassign(seg, center_shift_scale=0.6, center_max_shift_frac=0.35, center_smooth_win=5, center_iters=5):
    pts = np.asarray(seg.get("surface_points_all", np.zeros((0, 3))), dtype=np.float64)
    key = np.asarray(seg["keypoints"], dtype=np.float64)

    key, dominant_axis, flipped = canonicalize_curve_order(key)
    for _ in range(max(1, int(center_iters))):
        key = centroid_align(
            key,
            pts,
            scale=center_shift_scale,
            max_shift_frac=center_max_shift_frac,
            smooth_win=center_smooth_win,
        )
    key, dominant_axis, flipped2 = canonicalize_curve_order(key)

    seg["keypoints"] = key
    seg = reassign(seg)

    meta = dict(seg.get("metadata", {}))
    meta["dominant_axis"] = dominant_axis
    meta["canonical_order_flipped"] = bool(flipped or flipped2)
    meta["ordering_rule"] = "x:left_to_right / y:top_to_bottom / z:front_to_back"
    meta["recentered_after_edit"] = True
    meta["center_iters"] = int(center_iters)
    seg["metadata"] = meta
    return seg

def load_segments(path):
    arr = np.load(path, allow_pickle=True)["segments"]
    #print(arr[0]['id'])
    #exit()
    out = []
    for s in arr:
        out.append(dict(s.item() if hasattr(s, "item") and not isinstance(s, dict) else s))
    #print(out[0].keys())
    #exit()
    return out


def save_segments(path, out_dir, segs):
    np.savez_compressed(path, segments=np.array(segs, dtype=object))
    os.makedirs(out_dir, exist_ok=True)

    summary = []
    for s in segs:
        fp = os.path.join(out_dir, f"segment_{int(s['id'])}.npz")
        np.savez_compressed(fp, segment=np.array(s, dtype=object))
        summary.append(
            {
                "id": int(s["id"]),
                "name": s.get("name", ""),
                "n_keypoints": int(len(s["keypoints"])),
                "n_surface_all": int(len(s.get("surface_points_all", []))),
                "file": fp,
            }
        )

    json.dump({"segments": summary}, open(path.replace(".npz", "_summary.json"), "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npz", required=True)
    ap.add_argument("--ops_json", required=True)
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    by_id = {int(s["id"]): deepcopy(s) for s in load_segments(args.in_npz)}
    print(by_id.keys())
    ops = json.load(open(args.ops_json))


    for sp in ops.get("split", []):
        sid = int(sp["id"])
        seg = by_id.pop(sid)

        if "at_fracs" in sp:
            fracs = sp["at_fracs"]
        elif "at_frac" in sp:
            fracs = [sp["at_frac"]]
        else:
            fracs = [float(sp["at_index"]) / max(len(seg["keypoints"]) - 1, 1)]

        parts = split_poly(np.asarray(seg["keypoints"], dtype=np.float64), fracs)
        new_ids = list(map(int, sp["new_ids"]))
        names = sp.get("names", [f"{seg.get('name', sid)}_{i}" for i in range(len(parts))])

        pts = np.asarray(seg.get("surface_points_all", np.zeros((0, 3))), dtype=np.float64)
        child_d = [nearest_polyline_projection(p, pts)[3] for p in parts] if len(pts) else [np.zeros(0) for _ in parts]
        dmat = np.stack(child_d, axis=1) if len(pts) else np.zeros((0, len(parts)))
        amin = np.argmin(dmat, axis=1) if len(pts) else np.zeros(0, dtype=int)

        for i, p in enumerate(parts):
            child = deepcopy(seg)
            child["id"] = new_ids[i]
            child["name"] = names[i]
            child["keypoints"] = p
            child["surface_points_all"] = pts[amin == i]

            do_recenter = bool(sp.get("recenter", False))
            if do_recenter:
                child = recenter_and_reassign(
                    child,
                    center_shift_scale=float(sp.get("center_shift_scale", 0.6)),
                    center_max_shift_frac=float(sp.get("center_max_shift_frac", 0.35)),
                    center_smooth_win=int(sp.get("center_smooth_win", 5)),
                    center_iters=int(sp.get("center_iters", 3)),
                )
            else:
                child = reassign(child)

            by_id[child["id"]] = child

    for cb in ops.get("combine", []):
        ids = list(map(int, cb["ids"]))
        base = deepcopy(by_id[ids[0]])
        polys = [np.asarray(by_id[i]["keypoints"], dtype=np.float64) for i in ids]
        pts = np.concatenate(
            [np.asarray(by_id[i].get("surface_points_all", np.zeros((0, 3))), dtype=np.float64) for i in ids],
            axis=0,
        ) if ids else np.zeros((0, 3), dtype=np.float64)

        for i in ids:
            by_id.pop(i, None)

        base["id"] = int(cb["new_id"])
        base["name"] = cb.get("name", f"merged_{base['id']}")

        merged = merge_polys(polys)
        merged = maybe_straighten_curve(
            merged,
            mode=cb.get("straighten_mode", None),
            alpha=float(cb.get("straighten_alpha", 0.5)),
        )

        base["keypoints"] = merged
        base["surface_points_all"] = pts

        do_recenter = bool(cb.get("recenter", False))
        if do_recenter:
            base = recenter_and_reassign(
                base,
                center_shift_scale=float(cb.get("center_shift_scale", 0.6)),
                center_max_shift_frac=float(cb.get("center_max_shift_frac", 0.35)),
                center_smooth_win=int(cb.get("center_smooth_win", 5)),
            )
        else:
            base= reassign(base)
        by_id[base["id"]] = base

    for cs in ops.get("combine_surface_only", []):
        keep_id = int(cs["keep_id"])
        absorb_ids = list(map(int, cs.get("absorb_ids", [])))
        new_id = int(cs.get("new_id", keep_id))

        if keep_id not in by_id:
            raise KeyError(f"keep_id {keep_id} not found")

        keep_seg = deepcopy(by_id[keep_id])
        keep_seg["id"] = new_id
        

        point_sets = [
            np.asarray(keep_seg.get("surface_points_all", np.zeros((0, 3))), dtype=np.float64)
        ]

        for aid in absorb_ids:
            if aid not in by_id:
                raise KeyError(f"absorb_id {aid} not found")
            point_sets.append(
                np.asarray(by_id[aid].get("surface_points_all", np.zeros((0, 3))), dtype=np.float64)
            )

        merged_pts = np.concatenate(point_sets, axis=0) if point_sets else np.zeros((0, 3), dtype=np.float64)

        if len(merged_pts):
            merged_pts = np.unique(np.round(merged_pts, decimals=8), axis=0)

        keep_seg["surface_points_all"] = merged_pts
        keep_seg["name"] = cs.get("name", keep_seg.get("name", f"segment_{keep_id}"))

        keep_seg["keypoints"] = maybe_straighten_curve(
            np.asarray(keep_seg["keypoints"], dtype=np.float64),
            mode=cs.get("straighten_mode", None),
            alpha=float(cs.get("straighten_alpha", 0.5)),
        )

        
        do_recenter = bool(cs.get("recenter", False))
        if do_recenter:
            keep_seg = recenter_and_reassign(
                keep_seg,
                center_shift_scale=float(cs.get("center_shift_scale", 0.6)),
                center_max_shift_frac=float(cs.get("center_max_shift_frac", 0.35)),
                center_smooth_win=int(cs.get("center_smooth_win", 5)),
            )
        else:
            keep_seg= reassign(keep_seg)
        by_id[keep_id] = keep_seg

        for aid in absorb_ids:
            if aid != keep_id:
                by_id.pop(aid, None)

    for sm in ops.get("smooth", []):
        sid = int(sm["id"])
        seg = by_id[sid]
        seg["keypoints"] = smooth_poly(
            np.asarray(seg["keypoints"], dtype=np.float64),
            int(sm.get("window", 5)),
            int(sm.get("iterations", 1)),
        )
        by_id[sid] = reassign(seg)

    for cr in ops.get("crop", []):
        sid = int(cr["id"])
        seg = by_id[sid]
        seg["keypoints"] = crop_poly(
            np.asarray(seg["keypoints"], dtype=np.float64),
            cr.get("start_frac", 0.0),
            cr.get("end_frac", 1.0),
        )

        do_recenter = bool(cr.get("recenter", False))
        if do_recenter:
            seg = recenter_and_reassign(
                seg,
                center_shift_scale=float(cr.get("center_shift_scale", 0.6)),
                center_max_shift_frac=float(cr.get("center_max_shift_frac", 0.35)),
                center_smooth_win=int(cr.get("center_smooth_win", 5)),
                center_iters=int(cr.get("center_iters", 3)),
            )
        else:
            seg = reassign(seg)

        by_id[sid] = seg

    for ex in ops.get("extend", []):
        sid = int(ex["id"])
        seg = by_id[sid]
        seg["keypoints"] = extend_curve(
            np.asarray(seg["keypoints"], dtype=np.float64),
            np.asarray(seg.get("surface_points_all", np.zeros((0, 3))), dtype=np.float64),
            extend_start=bool(ex.get("extend_start", False)),
            extend_end=bool(ex.get("extend_end", True)),
            extend_radius_alpha=float(ex.get("extend_radius_alpha", 2.5)),
            extend_burial_alpha=float(ex.get("extend_burial_alpha", 1.0)),
            extend_min=float(ex.get("extend_min", 0.01)),
            target_spacing=float(ex.get("target_spacing", 0.01)),
        )
        by_id[sid] = reassign(seg)

    for rc in ops.get("recenter", []):
        sid = int(rc["id"])
        if sid not in by_id:
            continue

        seg = by_id[sid]
        seg = recenter_and_reassign(
            seg,
            center_shift_scale=float(rc.get("center_shift_scale", 0.6)),
            center_max_shift_frac=float(rc.get("center_max_shift_frac", 0.35)),
            center_smooth_win=int(rc.get("center_smooth_win", 5)),
            center_iters=int(rc.get("center_iters", 3)),
        )
        by_id[sid] = seg

    if "retain_ids" in ops:
        keep = set(map(int, ops["retain_ids"]))
        by_id = {k: v for k, v in by_id.items() if k in keep}

    # final rename/reindex pass on surviving segments only
    if "rename_final" in ops:
        renamed = {}
        for item in ops["rename_final"]:
            old_id = int(item["old_id"])
            if old_id not in by_id:
                continue

            seg = deepcopy(by_id[old_id])
            new_id = int(item.get("new_id", old_id))
            new_name = item.get("new_name", seg.get("name", f"segment_{new_id}"))

            seg["id"] = new_id
            seg["name"] = new_name

            renamed[new_id] = seg

        # keep only untouched survivors + renamed survivors
        untouched = {
            k: v for k, v in by_id.items()
            if k not in {int(x["old_id"]) for x in ops["rename_final"]}
        }

        # merge back
        by_id = {**untouched, **renamed}

    # optional automatic sequential renumbering of final survivors
    if ops.get("renumber_final", False):
        final_ids = sorted(by_id.keys())
        remapped = {}
        name_map = ops.get("rename_final_names", {})

        for new_id, old_id in enumerate(final_ids):
            seg = deepcopy(by_id[old_id])
            seg["id"] = new_id
            seg["name"] = name_map.get(str(old_id), seg.get("name", f"segment_{new_id}"))
            remapped[new_id] = seg

        by_id = remapped


    save_segments(
        args.out_npz,
        args.out_dir,
        sorted(by_id.values(), key=lambda s: int(s["id"])),
    )



if __name__ == "__main__":
    main()
