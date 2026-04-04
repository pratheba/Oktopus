#!/usr/bin/env python3
import argparse, json, math
import numpy as np


def load_polyline_edges(path):
    edges = []
    pts = []
    for line in open(path):
        vals = line.strip().split()
        if not vals or vals[0] != "2" or len(vals) < 7:
            continue
        p0 = np.array(list(map(float, vals[1:4])), dtype=np.float64)
        p1 = np.array(list(map(float, vals[4:7])), dtype=np.float64)
        edges.append((p0, p1))
        pts += [p0, p1]
    return edges, (np.array(pts, dtype=np.float64) if pts else np.zeros((0, 3), dtype=np.float64))


def build_graph(edges, tol=1e-6):
    nodes = []
    adj = {}

    def get_idx(p):
        for i, q in enumerate(nodes):
            if np.linalg.norm(p - q) <= tol:
                return i
        nodes.append(p.copy())
        adj[len(nodes) - 1] = []
        return len(nodes) - 1

    for p0, p1 in edges:
        i, j = get_idx(p0), get_idx(p1)
        if j not in adj[i]:
            adj[i].append(j)
        if i not in adj[j]:
            adj[j].append(i)
    return np.array(nodes), adj

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

def extract_segments(nodes, adj):
    deg = {i: len(v) for i, v in adj.items()}
    visited = set()
    segs = []
    starts = [i for i, d in deg.items() if d != 2] or ([0] if len(nodes) else [])

    for s in starts:
        for nb in adj[s]:
            e = tuple(sorted((s, nb)))
            if e in visited:
                continue
            path = [s, nb]
            visited.add(e)
            prev, cur = s, nb
            while deg[cur] == 2:
                nxts = [x for x in adj[cur] if x != prev]
                if not nxts:
                    break
                nxt = nxts[0]
                e = tuple(sorted((cur, nxt)))
                if e in visited:
                    break
                path.append(nxt)
                visited.add(e)
                prev, cur = cur, nxt
            segs.append(nodes[path])
    return segs


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


def assign_corr_points_to_segments(segments, corr_points, shared_margin=0.005):
    if len(corr_points) == 0:
        return [
            dict(
                owned=np.zeros((0, 3), dtype=np.float64),
                shared=np.zeros((0, 3), dtype=np.float64),
                all=np.zeros((0, 3), dtype=np.float64),
            )
            for _ in segments
        ]

    dists = [nearest_polyline_projection(seg, corr_points)[3] for seg in segments]
    dmat = np.stack(dists, axis=1)
    amin = np.argmin(dmat, axis=1)
    dmin = dmat[np.arange(len(corr_points)), amin]

    out = []
    for j in range(len(segments)):
        owned = amin == j
        shared = (np.abs(dmat[:, j] - dmin) <= shared_margin) & (~owned)
        out.append(
            dict(
                owned=corr_points[owned],
                shared=corr_points[shared],
                all=corr_points[owned | shared],
            )
        )
    return out


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


def load_correspondence_surface_points(path):
    surface_pts = []

    for line in open(path):
        vals = line.strip().split()
        if not vals or vals[0] != "2" or len(vals) < 7:
            continue

        # correspondence segment: p0 -> p1
        # p0 = skeleton-side point
        # p1 = surface point
        p1 = np.array(list(map(float, vals[4:7])), dtype=np.float64)
        surface_pts.append(p1)

    if not surface_pts:
        return np.zeros((0, 3), dtype=np.float64)

    return np.array(surface_pts, dtype=np.float64)

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
        delta = delta - np.dot(delta, tan) * tan

        mx = max_shift_frac * local_support_scale(support_points[m], 0.02)
        n = np.linalg.norm(delta)
        if n > mx:
            delta = delta / n * mx
        shifts[i] = scale * delta

    return poly + smooth_poly(shifts, window=smooth_win, iters=1)


def adaptive_params(poly, support_points, n_interior=36, endpoint_boost=2.0):
    s, total = seg_lengths(poly)
    t = s / max(total, 1e-12)
    w = np.ones_like(t)

    w += endpoint_boost * np.exp(-t / 0.1) + endpoint_boost * np.exp(-(1 - t) / 0.1)

    if len(poly) >= 3:
        d = np.diff(poly, axis=0)
        dn = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        curv = np.zeros(len(poly))
        curv[1:-1] = np.linalg.norm(dn[1:] - dn[:-1], axis=1)
        w += 2.0 * np.clip(curv, 0.0, 1.0)

    if len(support_points):
        _, sp, _, dist = nearest_polyline_projection(poly, support_points)
        bins = np.linspace(0, 1, 17)
        bid = np.clip(np.digitize(sp, bins) - 1, 0, len(bins) - 2)
        per = np.full(len(bins) - 1, np.nan)
        for i in range(len(per)):
            m = bid == i
            if np.any(m):
                per[i] = np.quantile(dist[m], 0.7)
        good = np.isfinite(per)
        if np.any(good):
            x = 0.5 * (bins[:-1] + bins[1:])
            w += 0.5 / np.maximum(np.interp(t, x[good], per[good]), 1e-3)

    ds = np.diff(s)
    wm = 0.5 * (w[:-1] + w[1:])
    ws = np.concatenate([[0.0], np.cumsum(ds * wm)])
    ws /= max(ws[-1], 1e-12)

    return np.interp(np.linspace(0, 1, n_interior), ws, t)


def extend_dense(poly, support_points, extend_radius_alpha=2.5, extend_burial_alpha=1.0, extend_min=0.01, target_spacing=0.01):
    poly = poly.copy()
    exts = []

    for at_start in [True, False]:
        endpoint = poly[0] if at_start else poly[-1]
        tan = -tangent_at(poly, True) if at_start else tangent_at(poly, False)

        _, sp, _, _ = nearest_polyline_projection(poly, support_points) if len(support_points) else (None, np.zeros(0), None, None)
        m = sp < 0.15 if at_start else sp > 0.85
        local = support_points[m] if len(support_points) else np.zeros((0, 3), dtype=np.float64)

        ext_len = max(
            extend_min,
            extend_radius_alpha * local_support_scale(local, target_spacing),
            extend_burial_alpha * bury_depth(endpoint, tan, local),
        )

        n = max(1, int(math.ceil(ext_len / max(target_spacing, 1e-4))))
        pts = np.array([endpoint + tan * ext_len * ((i + 1) / n) for i in range(n)], dtype=np.float64)
        exts.append(pts[::-1] if at_start else pts)

    return np.vstack([exts[0], poly, exts[1]])


def radii_from_support(keypoints, support_points):
    if len(support_points) == 0:
        z = np.full(len(keypoints), 0.01, dtype=np.float64)
        return z, z.copy(), z.copy()

    _, sp, _, dist = nearest_polyline_projection(keypoints, support_points)
    bins = np.linspace(0, 1, len(keypoints) + 1)
    bid = np.clip(np.digitize(sp, bins) - 1, 0, len(keypoints) - 1)

    r = np.full(len(keypoints), np.nan, dtype=np.float64)
    for i in range(len(keypoints)):
        m = bid == i
        if np.any(m):
            r[i] = np.quantile(dist[m], 0.85)

    good = np.isfinite(r)
    x = np.arange(len(r))
    r = np.interp(x, x[good], r[good]) if np.any(good) else np.full(len(r), 0.01, dtype=np.float64)
    return r, r.copy(), 1.1 * r


def canonicalize_curve_order(poly):
    p0 = poly[0]
    p1 = poly[-1]
    d = p1 - p0
    ax = np.abs(d)

    # dominant axis
    if ax[0] >= ax[1] and ax[0] >= ax[2]:
        # X-dominant: left -> right, so smaller x should be start
        k0 = (p0[0], -p0[1], -p0[2])
        k1 = (p1[0], -p1[1], -p1[2])
        if k1 < k0:
            return poly[::-1], "x"
        return poly, "x"

    elif ax[1] >= ax[0] and ax[1] >= ax[2]:
        # Y-dominant: top -> bottom, so larger y should be start
        k0 = (-p0[1], p0[0], -p0[2])
        k1 = (-p1[1], p1[0], -p1[2])
        if k1 < k0:
            return poly[::-1], "y"
        return poly, "y"

    else:
        # Z-dominant: front -> back, so larger z should be start
        k0 = (-p0[2], -p0[1], p0[0])
        k1 = (-p1[2], -p1[1], p1[0])
        if k1 < k0:
            return poly[::-1], "z"
        return poly, "z"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skel_file", required=True)
    ap.add_argument("--corr_file", required=True)
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--out_json", required=True)

    ap.add_argument("--n_keypoints", type=int, default=36)
    ap.add_argument("--center_shift_scale", type=float, default=0.6)
    ap.add_argument("--center_max_shift_frac", type=float, default=0.35)
    ap.add_argument("--center_smooth_win", type=int, default=5)

    ap.add_argument("--extend_radius_alpha", type=float, default=2.5)
    ap.add_argument("--extend_burial_alpha", type=float, default=1.0)
    ap.add_argument("--extend_min", type=float, default=0.01)
    ap.add_argument("--target_spacing", type=float, default=0.01)

    args = ap.parse_args()

    #edges, _ = load_polyline_edges(args.skel_file)
    #_, corr_pts = load_polyline_edges(args.corr_file)
    edges, _ = load_polyline_edges(args.skel_file)
    corr_pts = load_correspondence_surface_points(args.corr_file)

    nodes, adj = build_graph(edges)
    segments = extract_segments(nodes, adj)
    assigns = assign_corr_points_to_segments(segments, corr_pts)

    out = []
    summ = []

    for i, seg in enumerate(segments):
        supp = assigns[i]["all"]
        centered = centroid_align(
            seg,
            supp,
            args.center_shift_scale,
            args.center_max_shift_frac,
            args.center_smooth_win,
        )
        params = adaptive_params(centered, supp, args.n_keypoints)
        interior = interp_poly(centered, params)

        keypoints = extend_dense(
            interior,
            supp,
            args.extend_radius_alpha,
            args.extend_burial_alpha,
            args.extend_min,
            args.target_spacing,
        )

        keypoints, dominant_axis = canonicalize_curve_order(keypoints)

        T, U, V, frames = compute_parallel_transport_frames(keypoints)

        _, point_s, point_key_ids, _ = nearest_polyline_projection(keypoints, supp)
        r_train, r_wrap, r_cyl = radii_from_support(keypoints, supp)

        rec = {
            "id": i,
            "name": f"segment_{i}",
            "polyline_raw": seg,
            "polyline_centered": centered,
            "keypoints": keypoints,
            "frame_t": T,
            "frame_u": U,
            "frame_v": V,
            "frames": frames,
            "surface_points_owned": assigns[i]["owned"],
            "surface_points_shared": assigns[i]["shared"],
            "surface_points_all": supp,
            "point_s": point_s,
            "point_key_ids": point_key_ids,
            "radius_train": r_train,
            "radius_wrap": r_wrap,
            "radius_cylinder": r_cyl,
            "metadata": {
                "n_interior_requested": args.n_keypoints,
                "n_total_keypoints": int(len(keypoints)),
                "adaptive_sampling": True,
                "dense_extension": True,
                "dominant_axis": dominant_axis,
                "ordering_rule": "x:left_to_right / y:top_to_bottom / z:front_to_back"
            },
        }
        out.append(rec)
        summ.append(
            {
                "id": i,
                "name": rec["name"],
                "n_keypoints": int(len(keypoints)),
                "n_surface_all": int(len(supp)),
            }
        )

    np.savez_compressed(args.out_npz, segments=np.array(out, dtype=object))
    json.dump({"segments": summ, "out_npz": args.out_npz}, open(args.out_json, "w"), indent=2)


if __name__ == "__main__":
    main()
