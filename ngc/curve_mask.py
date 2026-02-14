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
