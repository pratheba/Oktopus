import torch
import numpy as np
import trimesh
import matplotlib.pyplot as plt

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




def compute_local_centering_stats(samples_local0, s_vals, n_bins=100, min_count=20):
    """
    samples_local0: (N,3) local coords before radius normalization
    s_vals:         (N,)  projected curve coordinate in [0,1]
    """
    u = samples_local0[:, 1]
    v = samples_local0[:, 2]

    edges = np.linspace(0.0, 1.0, n_bins + 1)
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
        mu = uv.mean(axis=0)
        r = np.linalg.norm(uv, axis=-1)
        r_mean = r.mean()

        offset_uv[b] = mu
        offset_rel[b] = np.linalg.norm(mu) / (r_mean + 1e-12)
        per_bin_points.append(uv)

    valid_mask = ~np.isnan(offset_rel)

    return {
        "offset_uv": offset_uv,
        "offset_rel": offset_rel,
        "counts": counts,
        "valid_mask": valid_mask,
        "edges": edges,
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

def compute_centered_curve_world(curve_core, stats):
    """
    Convert local centroid offsets into world-space curve corrections.
    """

    uv = stats["offset_uv"]
    valid = stats["valid_mask"]

    edges = stats["edges"]
    s = 0.5 * (edges[:-1] + edges[1:])

    # interpolate curve frame
    intpl = curve_core.interpolate(s)

    C = intpl["points"]      # original curve (B,3)
    frame = intpl["frame"]   # world->local rotation

    # convert to local axes in world
    axes = frame.transpose(0,2,1)
    T = axes[:,:,0]
    N = axes[:,:,1]
    B = axes[:,:,2]

    C_new = C.copy()

    for i in range(len(s)):
        if not valid[i]:
            continue

        mu_u, mu_v = uv[i]

        delta = mu_u * N[i] + mu_v * B[i]

        C_new[i] = C[i] + delta

    return C, C_new


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
