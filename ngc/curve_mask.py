import torch
import numpy as np

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
