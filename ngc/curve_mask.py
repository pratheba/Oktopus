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


def rotate_uv_avatar_to_acc(u_a, v_a, Na, Ba, Nb, Bb):
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
