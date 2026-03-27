import torch
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from typing import NamedTuple

class Bin(NamedTuple):
    edge: np.ndarray
    center: np.ndarray
    ids: np.ndarray

def get_bins(points, n_bins, istheta=False):
    if istheta:
        bin_edge = np.linspace(-np.pi, np.pi, n_bins + 1)
    else:
        bin_edge = np.linspace(0.0, 1.0, n_bins + 1)
    bin_center = 0.5 * (bin_edge[:-1] + bin_edge[1:])
    bin_ids = np.clip(np.digitize(points, bin_edge) - 1, 0, n_bins-1)
    return Bin(bin_edge, bin_center, bin_ids) 

def rigid_rotate_curve_and_frames(curve_pts, frames, anchor_point, axis, angle_rad):
    """
    curve_pts: (N,3)
    frames:    (N,3,3), columns [T,N,B]
    anchor_point: (3,)
    axis: (3,) world-space rotation axis
    angle_rad: scalar

    Returns:
        curve_rot:  (N,3)
        frames_rot: (N,3,3)
    """
    R = axis_angle_to_matrix(axis, angle_rad)

    curve_rot = (curve_pts - anchor_point[None, :]) @ R.T + anchor_point[None, :]
    frames_rot = np.matmul(R[None, :, :], frames)   # rotate T,N,B in world

    return curve_rot, frames_rot

def _normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n

def axis_angle_to_matrix(axis, angle_rad):
    """
    axis: (3,)
    returns R: (3,3)
    """
    a = _normalize(np.asarray(axis, dtype=np.float64))
    x, y, z = a
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    C = 1.0 - c

    R = np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C  ],
    ], dtype=np.float64)
    return R

def align_and_twist_local_offsets(w, u, v, frame_src, frame_tgt, delta_theta=0.0):
    """
    frame_src, frame_tgt: (N,3,3), columns [T,N,B]
    delta_theta: scalar or (N,) extra twist around tangent T
    """
    x_local = np.stack([w, u, v], axis=1)[:, :, None]   # (N,3,1)

    # Full frame alignment: source local -> target local
    R_align = np.matmul(np.transpose(frame_tgt, (0, 2, 1)), frame_src)   # (N,3,3)
    x_aligned = np.matmul(R_align, x_local)[:, :, 0]   # (N,3)

    w2 = x_aligned[:, 0]
    u2 = x_aligned[:, 1]
    v2 = x_aligned[:, 2]

    # Optional extra twist around tangent T
    if np.isscalar(delta_theta):
        c = np.cos(delta_theta)
        s = np.sin(delta_theta)
    else:
        c = np.cos(delta_theta)
        s = np.sin(delta_theta)

    u3 = c * u2 - s * v2
    v3 = s * u2 + c * v2

    return w2, u3, v3

def rotate_local_offsets_between_frames(w, u, v, frame_src, frame_tgt):
    """
    frame_src, frame_tgt: (N,3,3), columns [T, N, B]
    w,u,v: (N,)

    Returns rotated local coords in target frame:
        [w_new, u_new, v_new]
    """
    x_local = np.stack([w, u, v], axis=1)[:, :, None]   # (N,3,1)

    # target_local = F_tgt^T @ F_src @ source_local
    R = np.matmul(np.transpose(frame_tgt, (0, 2, 1)), frame_src)   # (N,3,3)
    x_new = np.matmul(R, x_local)[:, :, 0]   # (N,3)

    return x_new[:, 0], x_new[:, 1], x_new[:, 2]


def twist_in_nb_plane(u, v, delta_theta):
    """
    Rotate in target N-B plane around tangent T.
    """
    c = np.cos(delta_theta)
    s = np.sin(delta_theta)
    u2 = c * u - s * v
    v2 = s * u + c * v
    return u2, v2


def smoothstep01(x):
    return x * x * (3 - 2*x)

def make_detail_mask(t, t0, t1, eps=0.03):
    if t.ndim > 0 and t.shape[-1] == 1:
        t = t[..., 0]

    up = np.clip((t - (t0 - eps)) / (2*eps), 0.0, 1.0)
    dn = np.clip(((t1 + eps) - t) / (2*eps), 0.0, 1.0)
    return smoothstep01(up) * smoothstep01(dn)

def seam_fade(phi, eps_seam=0.06):
    """
    phi: (N,) in [0,1)  (wrapped phase)
    returns w_seam: (N,) ~0 near seam (phi~0 or ~1), ~1 away from seam
    """
    dist = np.minimum(phi, 1.0 - phi)             # 0 at seam
    x = np.clip(dist / eps_seam, 0.0, 1.0)        # 0..1
    return smoothstep01(x)

import numpy as np

def _normalize(v):
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

def _enforce_right_handed(T, N, B):
    """
    Make (T,N,B) right-handed and orthonormal-ish:
      B := normalize(T x N), choose sign to match input B
      N := normalize(B x T)
    """
    B_fix = _normalize(np.cross(T, N))
    # keep sign consistent with provided B
    sgn = np.sign(np.sum(B_fix * B, axis=1, keepdims=True) + 1e-12)
    B_fix = B_fix * sgn
    N_fix = _normalize(np.cross(B_fix, T))
    return N_fix, B_fix

def rotate_uv_avatar_to_acc(u_a, v_a, Na, Ba, Nb, Bb, Ta=None, Tb=None):
    """
    Robust version:
    - optionally enforces right-handedness if Ta/Tb given
    - projects the 2x2 map to nearest SO(2) to avoid reflections / drift
    """

    # If tangents are available, fix handedness (HIGHLY recommended)
    if Ta is not None:
        Na, Ba = _enforce_right_handed(Ta, Na, Ba)
    if Tb is not None:
        Nb, Bb = _enforce_right_handed(Tb, Nb, Bb)

    # Build 2x2 change-of-basis Q: [u_b v_b]^T = Q [u_a v_a]^T
    q00 = np.einsum("ij,ij->i", Nb, Na)
    q01 = np.einsum("ij,ij->i", Nb, Ba)
    q10 = np.einsum("ij,ij->i", Bb, Na)
    q11 = np.einsum("ij,ij->i", Bb, Ba)
    Q = np.stack([np.stack([q00, q01], axis=1),
                  np.stack([q10, q11], axis=1)], axis=1)  # (N,2,2)

    # Project to nearest rotation (SO(2)) per-sample: Q <- U V^T, det=+1
    U, _, Vt = np.linalg.svd(Q)
    Qr = U @ Vt
    det = np.linalg.det(Qr)
    bad = det < 0
    if np.any(bad):
        U2 = U.copy()
        U2[bad, :, 1] *= -1
        Qr[bad] = U2[bad] @ Vt[bad]

    uv = np.stack([u_a, v_a], axis=1)
    uv2 = np.einsum("nij,nj->ni", Qr, uv)
    return uv2[:, 0], uv2[:, 1]



def rotate_wuv_avatar_to_acc(w_a, u_a, v_a, Ta, Na, Ba, Tb, Nb, Bb, project_SO3=False):
    """
    Convert (w,u,v) offsets expressed in avatar (Ta,Na,Ba) basis
    into offsets expressed in accessory (Tb,Nb,Bb) basis.

    w_a,u_a,v_a: (N,)
    Ta,Na,Ba,Tb,Nb,Bb: (N,3)
    """

    # 3x3 change of basis: R_ij = <e_i^acc, e_j^avatar>
    # rows are acc basis, cols are avatar basis
    r00 = np.einsum("ij,ij->i", Tb, Ta); r01 = np.einsum("ij,ij->i", Tb, Na); r02 = np.einsum("ij,ij->i", Tb, Ba)
    r10 = np.einsum("ij,ij->i", Nb, Ta); r11 = np.einsum("ij,ij->i", Nb, Na); r12 = np.einsum("ij,ij->i", Nb, Ba)
    r20 = np.einsum("ij,ij->i", Bb, Ta); r21 = np.einsum("ij,ij->i", Bb, Na); r22 = np.einsum("ij,ij->i", Bb, Ba)

    R = np.stack([np.stack([r00,r01,r02],1),
                  np.stack([r10,r11,r12],1),
                  np.stack([r20,r21,r22],1)],1)  # (N,3,3)

    # Optional: project to nearest proper rotation (SO(3)) to kill reflections/drift
    if project_SO3:
        U, _, Vt = np.linalg.svd(R)
        Rr = U @ Vt
        det = np.linalg.det(Rr)
        bad = det < 0
        if np.any(bad):
            U2 = U.copy()
            U2[bad,:,2] *= -1
            Rr[bad] = U2[bad] @ Vt[bad]
        R = Rr

    wuv = np.stack([w_a, u_a, v_a], axis=1)            # (N,3)
    wuv2 = np.einsum("nij,nj->ni", R, wuv)            # (N,3)
    return wuv2[:,0], wuv2[:,1], wuv2[:,2]

def rotate_uv_avatar_to_acc1(u_a, v_a, Na, Ba, Nb, Bb):
    """
    Convert cross-section coords (u,v) expressed in avatar (Na,Ba) basis
    into coords expressed in accessory (Nb,Bb) basis.

    u_a, v_a: (N,) physical offsets in avatar NB coordinates
    Na, Ba:   (N,3) avatar frame vectors
    Nb, Bb:   (N,3) accessory frame vectors

    returns:
      u_b, v_b: (N,) physical offsets in boot NB coordinates
    """
    # 2x2 matrix entries: A_ij = <boot_basis_i, avatar_basis_j>
    a11 = np.einsum("ij,ij->i", Nb, Na)  # <Nb, Na>
    a12 = np.einsum("ij,ij->i", Nb, Ba)  # <Nb, Ba>
    a21 = np.einsum("ij,ij->i", Bb, Na)  # <Bb, Na>
    a22 = np.einsum("ij,ij->i", Bb, Ba)  # <Bb, Ba>

    u_b = a11 * u_a + a12 * v_a
    v_b = a21 * u_a + a22 * v_a
    return u_b, v_b


def maybe_flip_coords(coords, flip=False):
    return 1.0 - coords if flip else coords

def maybe_swap_nb(F, swap=False):
    # F: (M,3,3) rows=[T,N,B]
    if not swap:
        return F
    F2 = F.copy()
    F2[:,1,:], F2[:,2,:] = F2[:,2,:].copy(), F2[:,1,:].copy()
    return F2

def estimate_delta(Na, Ba, Nb):
    # all (N,3)
    x = np.einsum("ij,ij->i", Nb, Na)  # <Nb, Na>
    y = np.einsum("ij,ij->i", Nb, Ba)  # <Nb, Ba>
    delta = np.arctan2(y, x)            # (N,)
    print(delta.min(), delta.max(), delta.mean())
    return delta


def compute_side_imbalance(verts, curve_pts, T, N, B, slab_half_width=0.01, min_points=30):
    verts = np.asarray(verts, dtype=np.float64)
    curve_pts = np.asarray(curve_pts, dtype=np.float64)
    T = normalize(np.asarray(T, dtype=np.float64))
    N = normalize(np.asarray(N, dtype=np.float64))
    B = normalize(np.asarray(B, dtype=np.float64))

    S = curve_pts.shape[0]
    out = {
        "mean_u_pos": np.full(S, np.nan),
        "mean_u_neg": np.full(S, np.nan),
        "mean_v_pos": np.full(S, np.nan),
        "mean_v_neg": np.full(S, np.nan),
    }

    for i in range(S):
        c = curve_pts[i]
        t = T[i]
        n = N[i]
        b = B[i]

        d = verts - c[None, :]
        depth = d @ t
        mask = np.abs(depth) <= slab_half_width
        if np.sum(mask) < min_points:
            continue

        d_sel = d[mask]
        u = d_sel @ n
        v = d_sel @ b

        u_pos = np.abs(u[u > 0])
        u_neg = np.abs(u[u < 0])
        v_pos = np.abs(v[v > 0])
        v_neg = np.abs(v[v < 0])

        if len(u_pos) > 0: out["mean_u_pos"][i] = u_pos.mean()
        if len(u_neg) > 0: out["mean_u_neg"][i] = u_neg.mean()
        if len(v_pos) > 0: out["mean_v_pos"][i] = v_pos.mean()
        if len(v_neg) > 0: out["mean_v_neg"][i] = v_neg.mean()

    return out

def compute_local_centering_stats(samples_local0, s_vals, n_bins=24, min_count=50):
    """
    samples_local0: (N,3) local coords before radius normalization
    s_vals:         (N,)  projected curve coordinate in [0,1]
    """
    u = samples_local0[:, 1]
    v = samples_local0[:, 2]

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_ids = np.clip(np.digitize(s_vals, edges) - 1, 0, n_bins - 1)

    offset_uv = np.full((n_bins, 2), np.nan)
    offset_rel = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    per_bin_points = []

    for b in range(n_bins):
        m = (bin_ids == b)
        counts[b] = np.sum(m)

        if counts[b] < min_count:
            per_bin_points.append(None)
            continue

        uv = np.stack([u[m], v[m]], axis=-1)
        mu = np.median(uv, axis=0) #uv.mean(axis=0)
        r = np.linalg.norm(uv, axis=-1)
        r_quant = np.quantile(r, 0.9) #r.mean()
        rel = np.linalg.norm(mu) / (r_quant + 1e-12)
       
        if rel < 0.35:
            offset_uv[b] = mu
            offset_rel[b] = rel # np.linalg.norm(mu) / (r_mean + 1e-12)
            per_bin_points.append(uv)
        else:
            per_bin_points.append(None)

    valid_mask = (~np.isnan(offset_rel))  #& (counts >= min_count) & (offset_rel < 0.35))

    if np.any(valid_mask):
        good = np.where(valid_mask)[0]
        bad = np.where(~valid_mask)[0]

    if len(good) == 1:
        offset_uv[bad] = offset_uv[good[0]]
        offset_rel[bad] = offset_rel[good[0]]
    elif len(good) > 1:
        offset_uv[bad, 0] = np.interp(bad, good, offset_uv[good, 0])
        offset_uv[bad, 1] = np.interp(bad, good, offset_uv[good, 1])

        # optional: interpolate rel too
        offset_rel[bad] = np.interp(bad, good, offset_rel[good])

    return {
        "offset_uv": offset_uv,
        "offset_rel": offset_rel,
        "counts": counts,
        "valid_mask": valid_mask,
        "edges": edges,
        "centers": centers,
        "per_bin_points": per_bin_points,
    }


def plot_centroid_path_with_origin(stats):
    valid = stats["valid_mask"]
    uv = stats["offset_uv"][valid]

    plt.figure(figsize=(6, 6))
    plt.scatter([0.0], [0.0], s=120, marker='x', label='original center')
    plt.plot(uv[:, 0], uv[:, 1], '-o', markersize=3, label='centered centers')
    plt.axhline(0.0)
    plt.axvline(0.0)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("mean u")
    plt.ylabel("mean v")
    plt.title("Original center vs drifted centers across bins")
    plt.legend()
    plt.savefig("centroid_with_orig.jpg")
    #plt.show()

def plot_local_centering_stats(stats):
    x = np.arange(len(stats["offset_rel"]))
    valid = stats["valid_mask"]
    uv = stats["offset_uv"]

    plt.figure(figsize=(10, 4))
    plt.plot(x[valid], stats["offset_rel"][valid])
    plt.xlabel("bin index")
    plt.ylabel("relative offset")
    plt.title("Centering mismatch along curve")
    plt.grid(True)
    plt.savefig("center_mismatch.jpg")
    #plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(x[valid], uv[valid, 0], label="mean u")
    plt.plot(x[valid], uv[valid, 1], label="mean v")
    plt.axhline(0.0)
    plt.xlabel("bin index")
    plt.ylabel("centroid drift")
    plt.title("Centroid drift in local frame")
    plt.grid(True)
    plt.legend()
    plt.savefig("center_drift.jpg")
    #plt.show()


def plot_local_bins(stats, bins):
    for b in bins:
        uv = stats["per_bin_points"][b]
        if uv is None:
            continue

        mu = stats["offset_uv"][b]

        plt.figure(figsize=(5, 5))
        plt.scatter(uv[:, 0], uv[:, 1], s=5, alpha=0.3)
        plt.scatter([0.0], [0.0], s=80, marker='x', label='curve center')
        plt.scatter([mu[0]], [mu[1]], s=80, label='section centroid')
        plt.axhline(0.0)
        plt.axvline(0.0)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f"Local cross-section bin {b}")
        plt.xlabel("u")
        plt.ylabel("v")
        plt.legend()
        plt.savefig(str(b)+"bin.jpg")


def plot_local_bins_with_drift(stats, bins):
    for b in bins:
        uv = stats["per_bin_points"][b]
        if uv is None:
            continue

        mu = stats["offset_uv"][b]

        plt.figure(figsize=(5, 5))
        plt.scatter(uv[:, 0], uv[:, 1], s=5, alpha=0.25, label="local samples")
        plt.scatter([0.0], [0.0], s=120, marker='x', label='original center')
        plt.scatter([mu[0]], [mu[1]], s=90, label='drifted center')
        plt.plot([0.0, mu[0]], [0.0, mu[1]], linewidth=2)

        plt.axhline(0.0)
        plt.axvline(0.0)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel("u")
        plt.ylabel("v")
        plt.title(f"Bin {b}: local cross-section and center drift")
        plt.legend()
        plt.grid(True)
        plt.savefig(str(b)+"local_cross_section.jpg")


def plot_local_bins_with_drift_clean(stats, bins, marker_size_center=220, marker_size_centroid=180):
    for b in bins:
        uv = stats["per_bin_points"][b]
        if uv is None:
            continue

        mu = stats["offset_uv"][b]

        fig, ax = plt.subplots(figsize=(6, 6))

        ax.scatter(uv[:, 0], uv[:, 1], s=8, alpha=0.25, zorder=1)
        ax.plot([0.0, mu[0]], [0.0, mu[1]], linewidth=2.5, zorder=3)

        ax.scatter([0.0], [0.0], s=marker_size_center, marker='x', linewidths=3, zorder=5)
        ax.scatter([mu[0]], [mu[1]], s=marker_size_centroid, zorder=6)

        ax.annotate("curve center", (0.0, 0.0), xytext=(8, 8), textcoords="offset points")
        ax.annotate("centroid", (mu[0], mu[1]), xytext=(8, 8), textcoords="offset points")

        ax.axhline(0.0, linewidth=1.2, zorder=0)
        ax.axvline(0.0, linewidth=1.2, zorder=0)

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_title(f"Local cross-section bin {b}")
        ax.grid(True, alpha=0.25)

        plt.tight_layout()
        plt.savefig(str(b)+"local_cross_section.jpg")
        #plt.show()



def plot_centroid_offsets_from_origin(stats, bins=None, annotate=False):
    """
    Plot original local center (always at origin) and the drifted center
    for each bin, with a segment from origin -> drifted center.

    stats must contain:
        stats["offset_uv"]   : (B,2)
        stats["valid_mask"]  : (B,)
    """
    uv = np.asarray(stats["offset_uv"])
    valid = np.asarray(stats["valid_mask"])

    if bins is None:
        idx = np.where(valid)[0]
    else:
        bins = np.asarray(bins)
        idx = bins[valid[bins]]

    plt.figure(figsize=(7, 7))

    # original center once
    plt.scatter([0.0], [0.0], s=140, marker='x', label='original center')

    # draw one segment per bin: origin -> drifted center
    for i in idx:
        mu = uv[i]
        plt.plot([0.0, mu[0]], [0.0, mu[1]], alpha=0.6)
        plt.scatter(mu[0], mu[1], s=35)

        if annotate:
            plt.text(mu[0], mu[1], str(i), fontsize=8)

    plt.axhline(0.0)
    plt.axvline(0.0)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("mean u")
    plt.ylabel("mean v")
    plt.title("Original local center vs drifted center per bin")
    plt.legend()
    plt.grid(True)
    plt.savefig("local_drifted.jpg")
    #plt.show()

def plot_centered_curve_local_projections(stats, use_bin_centers=True):
    """
    Plot original local center curve (always zero) and shifted/centered curve
    as functions of curve position.

    stats must contain:
        offset_uv    : (B,2)
        valid_mask   : (B,)
        edges        : (B+1,)   optional
    """
    uv = np.asarray(stats["offset_uv"])
    valid = np.asarray(stats["valid_mask"])

    B = len(uv)

    if use_bin_centers and "edges" in stats:
        edges = np.asarray(stats["edges"])
        s = 0.5 * (edges[:-1] + edges[1:])
    else:
        s = np.linspace(0.0, 1.0, B)

    u_shift = uv[:, 0]
    v_shift = uv[:, 1]

    u_orig = np.zeros_like(s)
    v_orig = np.zeros_like(s)

    # mask invalid bins for shifted curve
    u_shift_plot = np.where(valid, u_shift, np.nan)
    v_shift_plot = np.where(valid, v_shift, np.nan)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(s, u_orig, '--', linewidth=2, label='original center (u=0)')
    axes[0].plot(s, u_shift_plot, linewidth=2, label='centered curve (u)')
    axes[0].set_ylabel("u")
    axes[0].set_title("Local N-direction projection: original vs centered")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(s, v_orig, '--', linewidth=2, label='original center (v=0)')
    axes[1].plot(s, v_shift_plot, linewidth=2, label='centered curve (v)')
    axes[1].set_xlabel("curve position s")
    axes[1].set_ylabel("v")
    axes[1].set_title("Local B-direction projection: original vs centered")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("orginal_centered_projection.jpg")
    #plt.show()


def show_curve_correction(points, C, C_new):
    scene = trimesh.Scene()

    scene.add_geometry(trimesh.points.PointCloud(points))

    scene.add_geometry(
        trimesh.load_path(C.reshape(-1,1,3))
    )

    scene.add_geometry(
        trimesh.load_path(C_new.reshape(-1,1,3))
    )

    segs = np.stack([C, C_new], axis=1)
    scene.add_geometry(trimesh.load_path(segs))
    
    scene.show()

def resample_curve_to_key_ts(s_dense, C_dense, key_ts):
    """
    s_dense : (M,)
    C_dense : (M,3)
    key_ts  : (K,)

    returns:
        C_key : (K,3)
    """
    C_key = np.zeros((len(key_ts), 3), dtype=np.float64)
    for d in range(3):
        C_key[:, d] = np.interp(key_ts, s_dense, C_dense[:, d])
    return C_key

def compute_centered_curve_world(curve_core, stats, alpha=1.0):
    """
    Convert local centroid offsets into world-space curve corrections.
    """

    uv = stats["offset_uv"]
    valid = stats["valid_mask"]

    edges = stats["edges"]
    s = stats["centers"] # 0.5 * (edges[:-1] + edges[1:])

    # interpolate curve frame
    intpl = curve_core.interpolate(s)

    C = intpl["points"]      # original curve (B,3)
    frame = intpl["frame"]   # world->local rotation

    # convert to local axes in world
    T = frame[:,0,:] #axes[:,:,0]
    N = frame[:,1,:] #axes[:,:,1]
    B = frame[:,2,:] #axes[:,:,2]

    C_new = C.copy()

    for i in range(len(s)):
        mu_u, mu_v = uv[i]
        if np.isnan(mu_u) or np.isnan(mu_v):
            continue
        #if not valid[i]:
        #    continue

        delta = mu_u * N[i] + mu_v * B[i]
        C_new[i] = C[i] + alpha* delta

    #C_key_new = resample_curve_to_key_ts(s, C_new, curve_core.key_ts)

    return C, C_new #C_key_new


def export_curve_points_as_ply(C_old, C_new, out_path="curve_compare_points.ply"):
    C_old = np.asarray(C_old, dtype=np.float64)
    C_new = np.asarray(C_new, dtype=np.float64)

    pts = np.vstack([C_old, C_new])

    colors_old = np.tile(np.array([[255, 0, 0, 255]], dtype=np.uint8), (len(C_old), 1))
    colors_new = np.tile(np.array([[0, 255, 0, 255]], dtype=np.uint8), (len(C_new), 1))
    colors = np.vstack([colors_old, colors_new])

    pc = trimesh.points.PointCloud(vertices=pts, colors=colors)
    pc.export(out_path)
    print(f"saved: {out_path}")

def export_shape_and_curves_as_ply(points, C_old, C_new, out_path="shape_and_curves.ply"):
    points = np.asarray(points, dtype=np.float64)
    C_old = np.asarray(C_old, dtype=np.float64)
    C_new = np.asarray(C_new, dtype=np.float64)

    pts = np.vstack([points, C_old, C_new])

    colors_pts = np.tile(np.array([[140, 140, 140, 60]], dtype=np.uint8), (len(points), 1))
    colors_old = np.tile(np.array([[255, 0, 0, 255]], dtype=np.uint8), (len(C_old), 1))
    colors_new = np.tile(np.array([[0, 255, 0, 255]], dtype=np.uint8), (len(C_new), 1))

    colors = np.vstack([colors_pts, colors_old, colors_new])

    pc = trimesh.points.PointCloud(vertices=pts, colors=colors)
    pc.export(out_path)
    print(f"saved: {out_path}")


def fill_invalid_bins(arr, valid):
    arr = arr.copy()
    idx_valid = np.where(valid)[0]
    if len(idx_valid) == 0:
        return arr
    idx_all = np.arange(len(arr))
    for i in idx_all[~valid]:
        j = idx_valid[np.argmin(np.abs(idx_valid - i))]
        arr[i] = arr[j]
    return arr


def fill_invalid_theta(vals, valid_mask):
    vals = vals.copy()
    n = len(vals)
    if n == 0:
        return vals
    if not np.any(valid_mask):
        return np.zeros_like(vals)

    idx = np.arange(n)
    valid_idx = idx[valid_mask]
    valid_vals = vals[valid_mask]

    idx_ext = np.concatenate([valid_idx - n, valid_idx, valid_idx + n])
    vals_ext = np.concatenate([valid_vals, valid_vals, valid_vals])

    vals_filled = np.interp(idx, idx_ext, vals_ext)
    return vals_filled


def interp_periodic_1d(x, xp, fp, period=2*np.pi):
    """
    Periodic 1D interpolation.
    x: (...,)
    xp: (K,) sorted, assumed within one period
    fp: (K,)
    """
    x = np.asarray(x)
    xp = np.asarray(xp)
    fp = np.asarray(fp)

    x0 = xp[0]
    x_wrap = ((x - x0) % period) + x0

    xp_ext = np.concatenate([xp - period, xp, xp + period])
    fp_ext = np.concatenate([fp, fp, fp])

    return np.interp(x_wrap, xp_ext, fp_ext)



def visualize_keyframes_with_ellipses_trimesh(
    curve_core,
    surface_points,
    key_ts=None,
    neighborhood_half_width=0.1,
    show_all_surface=True,
    key_sphere_radius=0.004,
    frame_scale=0.015,
    ellipse_samples=64,
    ellipse_stride=1,
    name='keyframe_with_ellipse'
):
    """
    Visualize keyframes, frame axes, radius bars, and cross-section ellipses.

    curve_core must provide:
        key_ts
        interpolate(ts) -> {'points','frame','radius'}
        calc_x_radius(ts)
        curve_projection(points)
    """
    if key_ts is None:
        key_ts = curve_core.key_ts

    key_ts = np.asarray(key_ts, dtype=np.float64)
    surface_points = np.asarray(surface_points, dtype=np.float64)

    intpl = curve_core.interpolate(key_ts)
    C = intpl["points"]          # (K,3)
    frame = intpl["frame"]       # world->local
    yz_radius = intpl["radius"]  # (K,2)

    axes = np.transpose(frame, (0, 2, 1))
    T = axes[:, :, 0]
    N = axes[:, :, 1]
    B = axes[:, :, 2]

    rw = curve_core.calc_x_radius(key_ts)
    ru = yz_radius[:, 0]
    rv = yz_radius[:, 1]

    scene = trimesh.Scene()

    if show_all_surface:
        #print(surface_points.shape)
        scene.add_geometry(_make_point_cloud(surface_points, color=(140, 140, 140, 45)))

    # local neighborhoods around each keyframe
    s_proj = curve_core.curve_projection(surface_points)
    valid = (s_proj >= 0.0) & (s_proj <= 1.0)
    pts_valid = surface_points[valid]
    s_valid = s_proj[valid]


    for i, s0 in enumerate(key_ts):
        Ci = C[i]
        Fi = frame[i]          # world->local at this keyframe

        local_i = np.einsum('ij,nj->ni', Fi, (surface_points - Ci[None, :]))
        w_i = local_i[:, 0]

        m = np.abs(w_i) <= neighborhood_half_width
        pts_local = surface_points[m]

        if len(pts_local) > 0:
            scene.add_geometry(_make_point_cloud(pts_local, color=(255, 200, 0, 70)))


#    for i, s0 in enumerate(key_ts):
#        m = np.abs(s_valid - s0) <= neighborhood_half_width
#        pts_local = pts_valid[m]
#        if len(pts_local) > 0:
#            scene.add_geometry(_make_point_cloud(pts_local, color=(255, 200, 0, 70)))

    # key centers
    kp = _make_spheres(C, radius=key_sphere_radius, color=(255, 0, 255, 255))
    if kp is not None:
        scene.add_geometry(kp)

    # frame axis direction markers
    scene.add_geometry(_make_segments(C, C + frame_scale * T, color=(255, 120, 120, 255)))
    scene.add_geometry(_make_segments(C, C + frame_scale * N, color=(120, 255, 120, 255)))
    scene.add_geometry(_make_segments(C, C + frame_scale * B, color=(120, 120, 255, 255)))

    # full radius bars
    scene.add_geometry(_make_segments(C - rw[:, None] * T, C + rw[:, None] * T, color=(255, 0, 0, 255)))
    scene.add_geometry(_make_segments(C - ru[:, None] * N, C + ru[:, None] * N, color=(0, 255, 0, 255)))
    scene.add_geometry(_make_segments(C - rv[:, None] * B, C + rv[:, None] * B, color=(0, 0, 255, 255)))

    # ellipses in cross-section plane
    for i in range(0, len(key_ts), ellipse_stride):
        epts = _ellipse_points_world(C[i], N[i], B[i], ru[i], rv[i], n=ellipse_samples)
        scene.add_geometry(_make_polyline(epts, closed=True, color=(0, 255, 255, 255)))
    #scene.show()
    print("scene", flush=True)
    print(name)
    #exit()
    scene.export(name+".glb")
    #scene.dump(concatenate=True).remove_infinite_values().export(name+"_ellipse.ply")
    scene.dump(concatenate=True).export(name+"_ellipse.ply")

#    visualize_single_keyframe_rho_spikes(
#    curve_core=curve_core,
#    surface_points=surface_points,
#    key_index=5,
#    slab_half_width=0.002,
#    name=name
#    )

    visualize_all_keyframes_rho_spikes(
        curve_core=curve_core,
        surface_points=surface_points,
        slab_half_width=0.002,
        max_spikes_per_key=80,
        spike_samples=10,
        name=name
    )
#    visualize_section_radius_uv(
#        curve_core=curve_core,
#        surface_points=surface_points,
#        key_index=0,
#        slab_half_width=0.0015,
#        n_angle_bins=72,
#        quantile=0.98,
#        name=name
#    )


def visualize_single_keyframe_rho_spikes(
    curve_core,
    surface_points,
    key_index,
    slab_half_width=0.003,
    ellipse_samples=64,
    name="rho_debug"
):
    key_ts = np.asarray(curve_core.key_ts, dtype=np.float64)
    s0 = key_ts[key_index:key_index+1]

    intpl = curve_core.interpolate(s0)
    C = intpl["points"][0]
    frame = intpl["frame"][0]      # world -> local
    yz_radius = intpl["radius"][0]

    axes = frame.T
    T = axes[:, 0]
    N = axes[:, 1]
    B = axes[:, 2]

    ru = yz_radius[0]
    rv = yz_radius[1]

    pts = np.asarray(surface_points, dtype=np.float64)

    # project all points into THIS keyframe's local frame
    local = np.einsum('ij,nj->ni', frame, (pts - C[None, :]))
    w = local[:, 0]
    u = local[:, 1]
    v = local[:, 2]

    # thin slab for actual cross-section
    mask = np.abs(w) <= slab_half_width
    pts_slice = pts[mask]
    local_slice = local[mask]

    if len(pts_slice) == 0:
        print("No points in slab.")
        return

    u_slice = local_slice[:, 1]
    v_slice = local_slice[:, 2]
    rho_slice = np.sqrt(u_slice**2 + v_slice**2)

    # ellipse points in world
    theta = np.linspace(0.0, 2.0 * np.pi, ellipse_samples, endpoint=False)
    epts = (
        C[None, :]
        + (ru * np.cos(theta))[:, None] * N[None, :]
        + (rv * np.sin(theta))[:, None] * B[None, :]
    )

    scene = trimesh.Scene()

    # surface slice
    colors = np.tile(np.array([[180, 180, 180, 100]], dtype=np.uint8), (len(pts_slice), 1))
    scene.add_geometry(trimesh.points.PointCloud(pts_slice, colors=colors))

    # center
    sph = trimesh.creation.icosphere(subdivisions=1, radius=0.003)
    sph.apply_translation(C)
    sph.visual.vertex_colors = np.tile(np.array([[255, 0, 255, 255]], dtype=np.uint8), (len(sph.vertices), 1))
    scene.add_geometry(sph)

    # ellipse as point cloud
    ecolors = np.tile(np.array([[255, 0, 0, 255]], dtype=np.uint8), (len(epts), 1))
    scene.add_geometry(trimesh.points.PointCloud(epts, colors=ecolors))

    # rho spikes: center -> each slice point
    segs = np.stack([np.repeat(C[None, :], len(pts_slice), axis=0), pts_slice], axis=1)
    path = trimesh.load_path(segs)
    path.colors = np.tile(np.array([[0, 255, 0, 120]], dtype=np.uint8), (len(path.entities), 1))
    scene.add_geometry(path)

    # optional frame axes
    axis_len = max(ru, rv) * 1.2
    axes_segs = np.array([
        [C, C + axis_len * T],
        [C, C + axis_len * N],
        [C, C + axis_len * B],
    ])
    axis_path = trimesh.load_path(axes_segs)
    axis_colors = np.array([
        [255, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
    ], dtype=np.uint8)
    axis_path.colors = axis_colors[:len(axis_path.entities)]
    scene.add_geometry(axis_path)
    all_pts = []
    all_cols = []

    # slice points
    if len(pts_slice) > 0:
        all_pts.append(pts_slice)
        all_cols.append(np.tile(np.array([[180, 180, 180, 100]], dtype=np.uint8), (len(pts_slice), 1)))

    # ellipse points
    if len(epts) > 0:
        epts_valid = epts[np.isfinite(epts).all(axis=1)]
        all_pts.append(epts_valid)
        all_cols.append(np.tile(np.array([[255, 0, 0, 255]], dtype=np.uint8), (len(epts_valid), 1)))

    # center point
    if np.isfinite(C).all():
        all_pts.append(C[None, :])
        all_cols.append(np.array([[255, 0, 255, 255]], dtype=np.uint8))
    # rho spikes as sampled points
    p0 = np.repeat(C[None, :], len(pts_slice), axis=0)
    p1 = pts_slice

    t = np.linspace(0, 1, 10)[:, None, None]
    spike_pts = ((1 - t) * p0[None, :, :] + t * p1[None, :, :]).reshape(-1, 3)

    all_pts.append(spike_pts)
    all_cols.append(np.tile(np.array([[0,255,0,255]], dtype=np.uint8), (len(spike_pts),1)))

    pts_export = np.vstack(all_pts)
    cols_export = np.vstack(all_cols)

    pc = trimesh.points.PointCloud(pts_export, colors=cols_export)
    pc.export(name + "_rho.ply")
    print("saved", name + "_rho.ply")

    #scene.export(name + ".glb")
    #scene.dump(concatenate=True).export(name+"_rho.ply")
    #print("saved", name + ".glb")
    print("rho min/max/mean:", rho_slice.min(), rho_slice.max(), rho_slice.mean())
    print("ru, rv:", ru, rv)


def sample_segments_as_points(p0, p1, n=12):
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)

    valid = np.isfinite(p0).all(axis=1) & np.isfinite(p1).all(axis=1)
    p0 = p0[valid]
    p1 = p1[valid]

    if len(p0) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    t = np.linspace(0.0, 1.0, n)[:, None, None]
    pts = (1.0 - t) * p0[None, :, :] + t * p1[None, :, :]
    return pts.reshape(-1, 3)


def get_radius_with_eps(radius, eps):
    if np.isscalar(eps):
        radius *= (1.0 + eps)
    else:
        radius += eps
    return radius


def remove_duplicate_consecutive_points(points, eps=1e-10):
    if len(points) <= 1:
        return points

    d = np.linalg.norm(points[1:] - points[:-1], axis=1)
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = d > eps

    pts = points[keep]
    if len(pts) < 2:
        return points[[0, -1]]
    return pts


def find_supported_s_interval(s_vals, n_bins=64, min_count=10, margin=0.01):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    hist, _ = np.histogram(s_vals, bins=edges)

    thresh = max(min_count, int(0.05 * hist.max()))
    valid = hist >= thresh

    if not np.any(valid):
        return 0.0, 1.0, hist, edges

    ids = np.where(valid)[0]
    s_min = max(0.0, edges[ids[0]] - margin)
    s_max = min(1.0, edges[ids[-1] + 1] + margin)
    return s_min, s_max, hist, edges


def prune_curve_points_by_s_interval(curve_points, s_min, s_max, n_out):
    s_old = np.linspace(0.0, 1.0, len(curve_points))

    keep = (s_old >= s_min) & (s_old <= s_max)
    s_keep = s_old[keep]
    p_keep = curve_points[keep]

    if len(p_keep) < 2:
        return curve_points

    s_new = np.linspace(s_min, s_max, n_out)
    x = np.interp(s_new, s_keep, p_keep[:, 0])
    y = np.interp(s_new, s_keep, p_keep[:, 1])
    z = np.interp(s_new, s_keep, p_keep[:, 2])
    return np.stack([x, y, z], axis=-1)
