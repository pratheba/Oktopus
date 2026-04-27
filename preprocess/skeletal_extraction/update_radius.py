#!/usr/bin/env python3
import argparse
import os
import json
import numpy as np
from copy import deepcopy
import trimesh
from scipy.ndimage import gaussian_filter1d


def load_segments(path):
    arr = np.load(path, allow_pickle=True)["segments"]
    out = []
    for s in arr:
        out.append(dict(s.item() if hasattr(s, "item") and not isinstance(s, dict) else s))
    return out


def save_segments(path, out_dir, segs):
    np.savez_compressed(path, segments=np.array(segs, dtype=object))
    os.makedirs(out_dir, exist_ok=True)

    summary = []
    for s in segs:
        fp = os.path.join(out_dir, f"segment_{int(s['id'])}.npz")
        np.savez_compressed(fp, segment=np.array(s, dtype=object))
        summary.append({
            "id": int(s["id"]),
            "name": s.get("name", ""),
            "n_keypoints": int(len(s["keypoints"])),
            "n_surface_all": int(len(s.get("surface_points_all", []))),
            "file": fp,
        })

    with open(path.replace(".npz", "_summary.json"), "w") as f:
        json.dump({"segments": summary}, f, indent=2)


def compute_tangents(poly):
    poly = np.asarray(poly, dtype=np.float64)
    K = len(poly)
    T = np.zeros((K, 3), dtype=np.float64)

    if K == 1:
        T[0] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return T

    for i in range(K):
        if i == 0:
            d = poly[1] - poly[0]
        elif i == K - 1:
            d = poly[-1] - poly[-2]
        else:
            d = poly[i + 1] - poly[i - 1]

        T[i] = d / (np.linalg.norm(d) + 1e-12)

    return T


def orthogonal_vector(v):
    if abs(v[0]) < 0.9:
        a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        a = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    u = a - np.dot(a, v) * v
    return u / (np.linalg.norm(u) + 1e-12)


def compute_parallel_transport_frames(poly):
    poly = np.asarray(poly, dtype=np.float64)
    T = compute_tangents(poly)
    K = len(poly)

    U = np.zeros((K, 3), dtype=np.float64)
    V = np.zeros((K, 3), dtype=np.float64)

    U[0] = orthogonal_vector(T[0])
    V[0] = np.cross(T[0], U[0])
    V[0] /= np.linalg.norm(V[0]) + 1e-12
    U[0] = np.cross(V[0], T[0])
    U[0] /= np.linalg.norm(U[0]) + 1e-12

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
        U[i] /= np.linalg.norm(U[i]) + 1e-12

        V[i] = np.cross(t_cur, U[i])
        V[i] /= np.linalg.norm(V[i]) + 1e-12

        U[i] = np.cross(V[i], t_cur)
        U[i] /= np.linalg.norm(U[i]) + 1e-12

    frames = np.stack([T, U, V], axis=1)
    return T, U, V, frames


def nearest_polyline_projection(poly, pts):
    pts = np.asarray(pts, dtype=np.float64)
    poly = np.asarray(poly, dtype=np.float64)

    if len(poly) < 2 or len(pts) == 0:
        return (
            np.zeros((len(pts), 3), dtype=np.float64),
            np.zeros(len(pts), dtype=np.float64),
            np.zeros(len(pts), dtype=np.int64),
            np.full(len(pts), np.inf, dtype=np.float64),
        )

    seg = np.diff(poly, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    seglen2 = np.sum(seg * seg, axis=1)
    cs = np.concatenate([[0.0], np.cumsum(seglen)])
    total = cs[-1]

    best_d2 = np.full(len(pts), np.inf, dtype=np.float64)
    best_s = np.zeros(len(pts), dtype=np.float64)
    best_idx = np.zeros(len(pts), dtype=np.int64)
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
        best_s[m] = cs[i] + t[m] * seglen[i]
        best_idx[m] = i

    return best_p, best_s / (total + 1e-12), best_idx, np.sqrt(best_d2)


def fill_invalid_1d(x, fallback=0.01):
    x = np.asarray(x, dtype=np.float64)
    good = np.isfinite(x)

    if np.any(good):
        ids = np.arange(len(x))
        return np.interp(ids, ids[good], x[good])

    return np.full(len(x), fallback, dtype=np.float64)


def fill_invalid_periodic_theta(row):
    row = np.asarray(row, dtype=np.float64).copy()
    good = np.isfinite(row)
    N = len(row)

    if np.all(good):
        return row

    if not np.any(good):
        return row

    ids = np.arange(N)
    ids_ext = np.concatenate([ids[good] - N, ids[good], ids[good] + N])
    vals_ext = np.concatenate([row[good], row[good], row[good]])
    return np.interp(ids, ids_ext, vals_ext)


def smooth_periodic_theta(arr, sigma):
    if sigma <= 0:
        return arr

    n_theta = arr.shape[1]
    ext = np.concatenate([arr, arr, arr], axis=1)
    ext = gaussian_filter1d(ext, sigma=sigma, axis=1)
    return ext[:, n_theta:2 * n_theta]


def interp_key_field(key_s, field, s_query):
    field = np.asarray(field, dtype=np.float64)
    out = np.zeros((len(s_query), field.shape[1]), dtype=np.float64)
    for j in range(field.shape[1]):
        out[:, j] = np.interp(s_query, key_s, field[:, j])
    return out


def orthonormalize_frames(T, U, V):
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)

    U = U - np.sum(U * T, axis=1, keepdims=True) * T
    U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)

    V = np.cross(T, U)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)

    U = np.cross(V, T)
    U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)

    return T, U, V


def localize_owned_points(key, T, U, V, pts):
    proj, s, seg_ids, dist = nearest_polyline_projection(key, pts)

    key_s = np.linspace(0.0, 1.0, len(key))

    Tq = interp_key_field(key_s, T, s)
    Uq = interp_key_field(key_s, U, s)
    Vq = interp_key_field(key_s, V, s)

    Tq, Uq, Vq = orthonormalize_frames(Tq, Uq, Vq)

    rel = pts - proj

    w = np.sum(rel * Tq, axis=1)
    u = np.sum(rel * Uq, axis=1)
    v = np.sum(rel * Vq, axis=1)

    # bin assignment for statistics
    right = np.clip(np.searchsorted(key_s, s), 0, len(key) - 1)
    left = np.clip(right - 1, 0, len(key) - 1)

    choose_left = np.abs(s - key_s[left]) < np.abs(s - key_s[right])
    key_ids = right.copy()
    key_ids[choose_left] = left[choose_left]
    #key_ids = np.clip(np.digitize(s, key_s) - 1, 0, len(key) - 1)

    return s, key_ids, w, u, v




def compute_train_cylinder_radius(
    key,
    T,
    U,
    V,
    pts,
    q_train=0.95,
    q_cylinder=0.995,
    min_count=8,
    cyl_margin=0.025,
    cyl_relative_margin=0.15,
    smooth_s=1.0,
):
    K = len(key)

    if len(pts) == 0:
        train = np.full((K, 2), 0.01, dtype=np.float64)
        cylinder = np.full((K, 2), 0.03, dtype=np.float64)
        return train, cylinder

    s, key_ids, w, u, v = localize_owned_points(key, T, U, V, pts)

    ry_train = np.full(K, np.nan, dtype=np.float64)
    rz_train = np.full(K, np.nan, dtype=np.float64)
    ry_cyl = np.full(K, np.nan, dtype=np.float64)
    rz_cyl = np.full(K, np.nan, dtype=np.float64)

    for i in range(K):
        m = key_ids == i
        if np.sum(m) < min_count:
            continue

        au = np.abs(u[m])
        av = np.abs(v[m])

        ry_train[i] = np.quantile(au, q_train)
        rz_train[i] = np.quantile(av, q_train)
        rho = np.sqrt(u[m]**2 + v[m]**2)

        r = np.quantile(rho, q_cylinder)
        r = r * (1.0 + cyl_relative_margin) + cyl_margin

        ry_cyl[i] = r
        rz_cyl[i] = r

        #ry_cyl[i] = np.quantile(au, q_cylinder) * (1.0 + cyl_relative_margin) + cyl_margin
        #rz_cyl[i] = np.quantile(av, q_cylinder) * (1.0 + cyl_relative_margin) + cyl_margin

    ry_train = fill_invalid_1d(ry_train, fallback=0.01)
    rz_train = fill_invalid_1d(rz_train, fallback=0.01)
    ry_cyl = fill_invalid_1d(ry_cyl, fallback=0.03)
    rz_cyl = fill_invalid_1d(rz_cyl, fallback=0.03)

    if smooth_s > 0:
        ry_train = gaussian_filter1d(ry_train, sigma=smooth_s)
        rz_train = gaussian_filter1d(rz_train, sigma=smooth_s)
        ry_cyl = gaussian_filter1d(ry_cyl, sigma=smooth_s)
        rz_cyl = gaussian_filter1d(rz_cyl, sigma=smooth_s)

    train = np.stack([ry_train, rz_train], axis=1)
    cylinder = np.stack([ry_cyl, rz_cyl], axis=1)

    cylinder = np.maximum(cylinder, train + cyl_margin)
    return train, cylinder


def enforce_wrap_coverage(
    wrap,
    key_ids,
    theta_ids,
    rho,
    margin=0.01,
    relative_margin=0.03,
):
    wrap = wrap.copy()
    K, T = wrap.shape

    for i in range(K):
        for j in range(T):
            m = (key_ids == i) & (theta_ids == j)
            if np.sum(m) == 0:
                continue

            r_need = np.max(rho[m])
            r_need = r_need * (1.0 + relative_margin) + margin

            if np.isfinite(r_need):
                wrap[i, j] = max(wrap[i, j], r_need)

    return wrap

def compute_directional_wrap_radius(
    key,
    T,
    U,
    V,
    pts,
    n_theta_bins=48,
    quantile=0.99,
    min_count=5,
    smooth_s=1.0,
    smooth_theta=1.0,
    fallback_radius=None,
    wrap_margin=0.01,
    wrap_relative_margin=0.03,
    wrap_w_factor=1.5
):
    K = len(key)
    theta_bins = np.linspace(-np.pi, np.pi, n_theta_bins, endpoint=False)
    wrap_s_bins = np.linspace(0.0, 1.0, K)

    if len(pts) == 0:
        if fallback_radius is None:
            wrap = np.full((K, n_theta_bins), 0.01, dtype=np.float64)
        else:
            rr = np.mean(fallback_radius, axis=1)
            wrap = np.repeat(rr[:, None], n_theta_bins, axis=1)
        return {
            "key_wrap_radius": wrap,
            "wrap_s_bins": wrap_s_bins,
            "wrap_theta_bins": theta_bins,
            "wrap_radius_max": np.max(wrap, axis=1),
        }

    s, key_ids, w, u, v = localize_owned_points(key, T, U, V, pts)
    rho = np.sqrt(u * u + v * v)
    theta = np.arctan2(v, u)

    # reject points that are not close to the local cross-section
    K = len(key)
    w_limit = wrap_w_factor / K   # try 1.0/K to 2.0/K

    valid = np.abs(w) <= w_limit

    s = s[valid]
    key_ids = key_ids[valid]
    w = w[valid]
    u = u[valid]
    v = v[valid]
    rho = rho[valid]
    theta = theta[valid]

    theta_edges = np.linspace(-np.pi, np.pi, n_theta_bins + 1)
    theta_ids = np.clip(np.digitize(theta, theta_edges) - 1, 0, n_theta_bins - 1)

    wrap = np.full((K, n_theta_bins), np.nan, dtype=np.float64)
    counts = np.zeros((K, n_theta_bins), dtype=np.int32)

    for i in range(K):
        mi = key_ids == i
        if not np.any(mi):
            continue

        for j in range(n_theta_bins):
            m = mi & (theta_ids == j)
            counts[i, j] = int(np.sum(m))
            if counts[i, j] < min_count:
                continue

            #wrap[i, j] = np.quantile(rho[m], quantile)
            wrap[i, j] = np.max(rho[m]) * (1.0 + wrap_relative_margin) + wrap_margin

    # fill missing theta bins per section
    for i in range(K):
        if np.any(np.isfinite(wrap[i])):
            wrap[i] = fill_invalid_periodic_theta(wrap[i])

    # fill missing s bins per theta
    for j in range(n_theta_bins):
        wrap[:, j] = fill_invalid_1d(wrap[:, j], fallback=np.nan)

    # if still all nan, fallback to train radius
    if not np.all(np.isfinite(wrap)):
        if fallback_radius is not None:
            rr = np.mean(fallback_radius, axis=1)
            fallback = np.repeat(rr[:, None], n_theta_bins, axis=1)
        else:
            fallback = np.full((K, n_theta_bins), 0.01, dtype=np.float64)

        wrap = np.where(np.isfinite(wrap), wrap, fallback)

    wrap_raw_interp = wrap.copy()
    if smooth_theta > 0:
        wrap = smooth_periodic_theta(wrap, sigma=smooth_theta)

    if smooth_s > 0:
        wrap = gaussian_filter1d(wrap, sigma=smooth_s, axis=0)

#    wrap_base = wrap.copy()
#
#    wrap_covered = enforce_wrap_coverage(
#        wrap,
#        key_ids=key_ids,
#        theta_ids=theta_ids,
#        rho=rho,
#        margin=wrap_margin,
#        relative_margin=wrap_relative_margin,
#    )
#    delta = 0.1 + wrap_covered - wrap_base
#    delta[delta < 1e-8] = 0.0
#
#    if smooth_theta > 0:
#        delta = smooth_periodic_theta(delta, sigma=smooth_theta)
#
#    if smooth_s > 0:
#        delta = gaussian_filter1d(delta, sigma=smooth_s, axis=0)

    #wrap = wrap_base + delta
    #wrap_raw_coverage = wrap.copy()
    wrap = np.maximum(wrap, 0.90 * wrap_raw_interp)
    wrap = enforce_wrap_coverage(
        wrap,
        key_ids=key_ids,
        theta_ids=theta_ids,
        rho=rho,
        margin=wrap_margin,
        relative_margin=wrap_relative_margin,
    )

    #wrap = np.maximum(wrap, 1e-5)

    return {
        "key_wrap_radius": wrap,
        "wrap_s_bins": wrap_s_bins,
        "wrap_theta_bins": theta_bins,
        "wrap_radius_max": np.max(wrap, axis=1),
        "wrap_counts": counts,
    }


def update_segment_radius(seg, args):

    seg = deepcopy(seg)

    key = np.asarray(seg["keypoints"], dtype=np.float64)


    # Use owned points if present. This is the important part.
    if args.use_owned and "surface_points_owned" in seg:
        pts = np.asarray(seg["surface_points_owned"], dtype=np.float64)
    else:
        pts = np.asarray(seg.get("surface_points_all", np.zeros((0, 3))), dtype=np.float64)

    if args.recenter:
        key = recenter_keypoints_from_owned_points(
            key,
            pts,
            n_iters=args.recenter_iters,
            window_frac=args.recenter_window_frac,
            shift_alpha=args.recenter_alpha,
            max_shift_frac=args.recenter_max_shift_frac,
            smooth_sigma=args.recenter_smooth_sigma,
        )

        seg["keypoints"] = key

    T, U, V, frames = compute_parallel_transport_frames(key)
    _, point_s, point_key_ids, _ = nearest_polyline_projection(key, pts)

    train, cylinder = compute_train_cylinder_radius(
        key, T, U, V, pts,
        q_train=args.q_train,
        q_cylinder=args.q_cylinder,
        min_count=args.min_count,
        cyl_margin=args.cyl_margin,
        cyl_relative_margin=args.cyl_relative_margin,
        smooth_s=args.smooth_s,
    )

    wrap = compute_directional_wrap_radius(
        key, T, U, V, pts,
        n_theta_bins=args.n_theta_bins,
        quantile=args.q_wrap,
        min_count=args.wrap_min_count,
        smooth_s=args.wrap_smooth_s,
        smooth_theta=args.wrap_smooth_theta,
        fallback_radius=train,
        wrap_margin=args.wrap_margin,
        wrap_relative_margin=args.wrap_relative_margin,
        wrap_w_factor=args.wrap_w_factor,
    )

    if args.cylinder_from_wrap:
        wrap_r = wrap["key_wrap_radius"]
        wrap_max = np.max(wrap_r, axis=1)
        cyl_r = wrap_max * (1.0 + args.cyl_relative_margin) + args.cyl_margin
        cylinder = np.stack([cyl_r, cyl_r], axis=1)

    #if args.derive_train_from_wrap:
    #    wrap_r = wrap["key_wrap_radius"]
    #    theta = wrap["wrap_theta_bins"][None, :]

     #   train_y = np.max(np.abs(wrap_r * np.cos(theta)), axis=1)
     #   train_z = np.max(np.abs(wrap_r * np.sin(theta)), axis=1)

      #  train = np.stack([train_y, train_z], axis=1)
    if args.derive_train_from_wrap:
        wrap_r = wrap["key_wrap_radius"]

        r = np.max(wrap_r, axis=1)

        train = np.stack([r, r], axis=1)

        train *= 1.05

    seg["point_s"] = point_s
    seg["point_key_ids"] = point_key_ids

    seg["radius_train"] = train
    seg["radius_cylinder"] = cylinder

    # Real directional wrap.
    seg["key_wrap_radius"] = wrap["key_wrap_radius"]
    seg["wrap_s_bins"] = wrap["wrap_s_bins"]
    seg["wrap_theta_bins"] = wrap["wrap_theta_bins"]
    seg["wrap_radius_max"] = wrap["wrap_radius_max"]
    seg["wrap_counts"] = wrap["wrap_counts"]

    # Backward compatibility: scalar max wrap.
    seg["radius_wrap"] = wrap["wrap_radius_max"]

    seg["frame_t"] = T
    seg["frame_u"] = U
    seg["frame_v"] = V
    seg["frames"] = frames

    meta = dict(seg.get("metadata", {}))
    meta["radius_updated_by"] = "update_radius.py"
    meta["radius_source_points"] = "surface_points_owned" if args.use_owned and "surface_points_owned" in seg else "surface_points_all"
    meta["radius_train_shape"] = list(train.shape)
    meta["radius_cylinder_shape"] = list(cylinder.shape)
    meta["key_wrap_radius_shape"] = list(wrap["key_wrap_radius"].shape)
    meta["n_theta_bins"] = int(args.n_theta_bins)
    seg["metadata"] = meta

    print(
        f"[radius] id={seg.get('id')} name={seg.get('name','')} "
        f"K={len(key)} pts={len(pts)} "
        f"train={train.shape} cyl={cylinder.shape} wrap={wrap['key_wrap_radius'].shape}"
    )

    return seg

def recenter_keypoints_from_owned_points(
    key,
    pts,
    n_iters=5,
    window_frac=0.05,
    shift_alpha=0.6,
    max_shift_frac=0.35,
    smooth_sigma=1.5,
):
    key = np.asarray(key, dtype=np.float64).copy()
    pts = np.asarray(pts, dtype=np.float64)

    if len(pts) == 0 or len(key) < 2:
        return key

    for _ in range(n_iters):
        T, U, V, frames = compute_parallel_transport_frames(key)
        _, s_pts, _, _ = nearest_polyline_projection(key, pts)

        K = len(key)
        key_s = np.linspace(0.0, 1.0, K)
        shifts = np.zeros_like(key)

        for i, s0 in enumerate(key_s):
            m = np.abs(s_pts - s0) <= window_frac
            if np.sum(m) < 10:
                continue

            local_pts = pts[m]
            centroid = np.mean(local_pts, axis=0)
            delta = centroid - key[i]

            # remove tangent component: only move in cross-section plane
            delta = delta - np.dot(delta, T[i]) * T[i]

            # limit shift using local point spread
            rel = local_pts - centroid[None, :]
            spread = np.quantile(np.linalg.norm(rel, axis=1), 0.8)
            max_shift = max_shift_frac * max(spread, 1e-4)

            n = np.linalg.norm(delta)
            if n > max_shift:
                delta = delta / (n + 1e-12) * max_shift

            shifts[i] = shift_alpha * delta

        if smooth_sigma > 0:
            shifts = gaussian_filter1d(shifts, sigma=smooth_sigma, axis=0)

        # keep endpoints less aggressive
        if K > 2:
            weights = np.ones(K)
            weights[0] = 0.25
            weights[-1] = 0.25
            shifts *= weights[:, None]

        key = key + shifts

    return key


def make_polyline(points, closed=False, color=(255, 255, 255, 255), tube_radius=0.0015):
    meshes = []
    pts0 = points
    pts1 = np.roll(points, -1, axis=0) if closed else points[1:]

    if not closed:
        pts0 = points[:-1]

    for a, b in zip(pts0, pts1):
        d = b - a
        L = np.linalg.norm(d)
        if L < 1e-10:
            continue

        cyl = trimesh.creation.cylinder(radius=tube_radius, height=L, sections=8)
        z = np.array([0.0, 0.0, 1.0])
        direction = d / L

        v = np.cross(z, direction)
        c = np.dot(z, direction)

        if np.linalg.norm(v) < 1e-12:
            R = np.eye(3) if c > 0 else np.diag([1, -1, -1])
        else:
            vx = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0],
            ])
            R = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))

        Tm = np.eye(4)
        Tm[:3, :3] = R
        Tm[:3, 3] = 0.5 * (a + b)

        cyl.apply_transform(Tm)
        cyl.visual.vertex_colors = np.tile(np.array(color, dtype=np.uint8), (len(cyl.vertices), 1))
        meshes.append(cyl)

    return trimesh.util.concatenate(meshes) if meshes else None


def ellipse_points(C, U, V, ru, rv, n=72):
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return (
        C[None, :]
        + (ru * np.cos(th))[:, None] * U[None, :]
        + (rv * np.sin(th))[:, None] * V[None, :]
    )


def wrap_points(C, U, V, r_theta, theta_bins):
    return (
        C[None, :]
        + (r_theta * np.cos(theta_bins))[:, None] * U[None, :]
        + (r_theta * np.sin(theta_bins))[:, None] * V[None, :]
    )

def export_colored_ply(points, colors, out_path):
    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.uint8)

    pc = trimesh.points.PointCloud(points, colors=colors)
    pc.export(out_path)
    print("[ply]", out_path)

def visualize_segment_radii_ply(seg, out_dir, stride=1, n_circle=96):
    os.makedirs(out_dir, exist_ok=True)

    key = np.asarray(seg["keypoints"], dtype=np.float64)
    U = np.asarray(seg["frame_u"], dtype=np.float64)
    V = np.asarray(seg["frame_v"], dtype=np.float64)

    train = np.asarray(seg["radius_train"], dtype=np.float64)
    cyl = np.asarray(seg["radius_cylinder"], dtype=np.float64)
    wrap = np.asarray(seg["key_wrap_radius"], dtype=np.float64)
    theta_bins = np.asarray(seg["wrap_theta_bins"], dtype=np.float64)

    sid = int(seg["id"])
    name = str(seg.get("name", f"segment_{sid}")).replace("/", "_")

    # owned points = gray
    pts = np.asarray(seg.get("surface_points_owned", seg.get("surface_points_all", [])), dtype=np.float64)
    if len(pts) > 0:
        colors = np.tile(np.array([180, 180, 180, 90], dtype=np.uint8), (len(pts), 1))
        export_colored_ply(pts, colors, os.path.join(out_dir, f"{sid}_{name}_points.ply"))

    th = np.linspace(0.0, 2.0 * np.pi, n_circle, endpoint=False)

    train_pts = []
    cyl_pts = []
    wrap_pts = []
    center_pts = []

    for i in range(0, len(key), max(1, stride)):
        C = key[i]
        center_pts.append(C)

        # train ellipse
        ru, rv = train[i]
        p = C[None, :] + (ru * np.cos(th))[:, None] * U[i][None, :] + (rv * np.sin(th))[:, None] * V[i][None, :]
        train_pts.append(p)

        # cylinder ellipse
        ru, rv = cyl[i]
        p = C[None, :] + (ru * np.cos(th))[:, None] * U[i][None, :] + (rv * np.sin(th))[:, None] * V[i][None, :]
        cyl_pts.append(p)

        # directional wrap
        if wrap.ndim == 2 and wrap.shape[1] == len(theta_bins):
            r = wrap[i]
            p = C[None, :] + (r * np.cos(theta_bins))[:, None] * U[i][None, :] + (r * np.sin(theta_bins))[:, None] * V[i][None, :]
            wrap_pts.append(p)

    def save_ring_points(rings, color, suffix):
        if len(rings) == 0:
            return
        P = np.vstack(rings)
        P = P[np.isfinite(P).all(axis=1)]
        C = np.tile(np.array(color, dtype=np.uint8), (len(P), 1))
        export_colored_ply(P, C, os.path.join(out_dir, f"{sid}_{name}_{suffix}.ply"))

    save_ring_points(train_pts, [0, 255, 0, 255], "train_radius_green")
    save_ring_points(cyl_pts, [0, 120, 255, 255], "cylinder_radius_blue")
    save_ring_points(wrap_pts, [255, 40, 40, 255], "wrap_radius_red")

    if len(center_pts) > 0:
        P = np.asarray(center_pts)
        C = np.tile(np.array([255, 0, 255, 255], dtype=np.uint8), (len(P), 1))
        export_colored_ply(P, C, os.path.join(out_dir, f"{sid}_{name}_keypoints_magenta.ply"))


def visualize_segment_radii(
    seg,
    out_path,
    stride=1,
    n_circle=72,
    show_points=True,
):
    key = np.asarray(seg["keypoints"], dtype=np.float64)
    T = np.asarray(seg["frame_t"], dtype=np.float64)
    U = np.asarray(seg["frame_u"], dtype=np.float64)
    V = np.asarray(seg["frame_v"], dtype=np.float64)

    train = np.asarray(seg["radius_train"], dtype=np.float64)
    cylinder = np.asarray(seg["radius_cylinder"], dtype=np.float64)

    wrap = np.asarray(seg.get("key_wrap_radius", np.zeros((0, 0))), dtype=np.float64)
    theta_bins = np.asarray(seg.get("wrap_theta_bins", np.zeros(0)), dtype=np.float64)

    scene = trimesh.Scene()

    # owned points
    if show_points:
        pts = np.asarray(seg.get("surface_points_owned", seg.get("surface_points_all", [])), dtype=np.float64)
        if len(pts) > 0:
            pc = trimesh.points.PointCloud(
                pts,
                colors=np.tile(np.array([180, 180, 180, 60], dtype=np.uint8), (len(pts), 1))
            )
            scene.add_geometry(pc)

    # centerline
    centerline = make_polyline(key, closed=False, color=(255, 255, 255, 255), tube_radius=0.0015)
    if centerline is not None:
        scene.add_geometry(centerline)

    for i in range(0, len(key), max(1, stride)):
        C = key[i]

        # train = green
        if train.ndim == 2 and train.shape[0] == len(key):
            ru, rv = train[i]
            if np.isfinite(ru) and np.isfinite(rv) and ru > 0 and rv > 0:
                pts_train = ellipse_points(C, U[i], V[i], ru, rv, n=n_circle)
                mesh = make_polyline(pts_train, closed=True, color=(0, 255, 0, 255), tube_radius=0.0012)
                if mesh is not None:
                    scene.add_geometry(mesh)

        # cylinder = blue
        if cylinder.ndim == 2 and cylinder.shape[0] == len(key):
            ru, rv = cylinder[i]
            if np.isfinite(ru) and np.isfinite(rv) and ru > 0 and rv > 0:
                pts_cyl = ellipse_points(C, U[i], V[i], ru, rv, n=n_circle)
                mesh = make_polyline(pts_cyl, closed=True, color=(0, 120, 255, 255), tube_radius=0.0012)
                if mesh is not None:
                    scene.add_geometry(mesh)

        # wrap = red
        if wrap.ndim == 2 and wrap.shape[0] == len(key) and len(theta_bins) == wrap.shape[1]:
            r = wrap[i]
            if np.all(np.isfinite(r)) and np.max(r) > 0:
                pts_wrap = wrap_points(C, U[i], V[i], r, theta_bins)
                mesh = make_polyline(pts_wrap, closed=True, color=(255, 40, 40, 255), tube_radius=0.0014)
                if mesh is not None:
                    scene.add_geometry(mesh)

    scene.export(out_path)
    print("[viz]", out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_npz", required=True)
    p.add_argument("--out_npz", required=True)
    p.add_argument("--out_dir", required=True)

    p.add_argument("--use_owned", action="store_true", help="Use surface_points_owned instead of surface_points_all when available.")

    p.add_argument("--q_train", type=float, default=0.95)
    p.add_argument("--q_cylinder", type=float, default=0.995)
    p.add_argument("--q_wrap", type=float, default=0.999)

    p.add_argument("--min_count", type=int, default=8)
    p.add_argument("--wrap_min_count", type=int, default=5)

    p.add_argument("--n_theta_bins", type=int, default=48)

    p.add_argument("--cyl_margin", type=float, default=0.1)
    p.add_argument("--cyl_relative_margin", type=float, default=0.25)
    p.add_argument("--cylinder_from_wrap", action="store_true")
    p.add_argument("--derive_train_from_wrap", action="store_true")

    p.add_argument("--smooth_s", type=float, default=1.0)
    p.add_argument("--wrap_smooth_s", type=float, default=1.0)
    p.add_argument("--wrap_smooth_theta", type=float, default=1.0)
    p.add_argument("--wrap_margin", type=float, default=0.01)
    p.add_argument("--wrap_relative_margin", type=float, default=0.03)
    p.add_argument("--wrap_w_factor", type=float, default=1.5)
    p.add_argument("--viz", action="store_true")
    p.add_argument("--viz_ply", action="store_true")
    p.add_argument("--viz_dir", default=None)
    p.add_argument("--viz_stride", type=int, default=1)
    p.add_argument("--recenter", action="store_true")
    p.add_argument("--recenter_iters", type=int, default=5)
    p.add_argument("--recenter_window_frac", type=float, default=0.05)
    p.add_argument("--recenter_alpha", type=float, default=0.6)
    p.add_argument("--recenter_max_shift_frac", type=float, default=0.35)
    p.add_argument("--recenter_smooth_sigma", type=float, default=1.5)

    args = p.parse_args()

    segs = load_segments(args.in_npz)
    out = [update_segment_radius(seg, args) for seg in segs]
    if args.viz_ply:
        viz_dir = args.viz_dir or os.path.join(args.out_dir, "radius_viz_ply")
        os.makedirs(viz_dir, exist_ok=True)

        for seg in out:
            visualize_segment_radii_ply(
                seg,
                viz_dir,
                stride=args.viz_stride,
                n_circle=96,
            )

    if args.viz:
        viz_dir = args.viz_dir or os.path.join(args.out_dir, "radius_viz")
        os.makedirs(viz_dir, exist_ok=True)

        for seg in out:
            sid = int(seg["id"])
            name = str(seg.get("name", f"segment_{sid}")).replace("/", "_")
            visualize_segment_radii(
                seg,
                os.path.join(viz_dir, f"{sid}_{name}_radii.glb"),
                stride=args.viz_stride,
                show_points=True,
            )

    save_segments(
        args.out_npz,
        args.out_dir,
        sorted(out, key=lambda s: int(s["id"]))
    )

    print("[done]", args.out_npz)


if __name__ == "__main__":
    main()
