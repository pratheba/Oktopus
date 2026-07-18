import numpy as np


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)


def _skew(v: np.ndarray) -> np.ndarray:
    """
    v: (..., 3)
    returns: (..., 3, 3)
    """
    z = np.zeros_like(v[..., 0])
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    return np.stack([
        np.stack([z,   -vz,  vy], axis=-1),
        np.stack([vz,   z,  -vx], axis=-1),
        np.stack([-vy,  vx,  z ], axis=-1),
    ], axis=-2)


def _rodrigues_from_a_to_b(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Minimal rotation taking unit vector a -> unit vector b.

    a, b: (..., 3), need not already be normalized
    returns: (..., 3, 3)
    """
    a = _normalize(a, eps)
    b = _normalize(b, eps)

    v = np.cross(a, b)                         # rotation axis * sin(theta)
    c = np.sum(a * b, axis=-1, keepdims=True) # cos(theta)
    s = np.linalg.norm(v, axis=-1, keepdims=True)

    I = np.broadcast_to(np.eye(3), a.shape[:-1] + (3, 3)).copy()
    Vx = _skew(v)

    # Generic Rodrigues formula:
    # R = I + [v]_x + [v]_x^2 * (1-c)/s^2
    s2 = np.clip(s * s, eps, None)
    R = I + Vx + (Vx @ Vx) * ((1.0 - c)[..., None] / s2[..., None])

    # Handle nearly parallel
    parallel = (s[..., 0] < 1e-8) & (c[..., 0] > 0.0)
    if np.any(parallel):
        R[parallel] = np.eye(3)

    # Handle nearly anti-parallel
    antipar = (s[..., 0] < 1e-8) & (c[..., 0] < 0.0)
    if np.any(antipar):
        aa = a[antipar]

        # Choose a stable axis orthogonal to a
        # pick basis axis least aligned with a
        basis = np.zeros_like(aa)
        idx = np.argmin(np.abs(aa), axis=-1)
        basis[np.arange(len(aa)), idx] = 1.0

        axis = np.cross(aa, basis)
        axis = _normalize(axis, eps)

        # 180 deg rotation: R = -I + 2 uu^T
        uuT = axis[..., :, None] * axis[..., None, :]
        R180 = -np.eye(3)[None, :, :] + 2.0 * uuT
        R[antipar] = R180

    return R


def _orthonormalize_frame(frame: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    frame: (N, 3, 3), row-wise convention [T; N; B]
    returns orthonormalized row-wise frame
    """
    T = _normalize(frame[:, 0, :], eps)

    # Gram-Schmidt N against T
    N = frame[:, 1, :] - np.sum(frame[:, 1, :] * T, axis=1, keepdims=True) * T
    N = _normalize(N, eps)

    # Rebuild B to guarantee right-handedness and orthogonality
    B = np.cross(T, N)
    B = _normalize(B, eps)

    # Recompute N once more for numerical cleanliness
    N = np.cross(B, T)
    N = _normalize(N, eps)

    return np.stack([T, N, B], axis=1)


def transfer_frame_orientation(
    old_frame: np.ndarray,
    old_tangent: np.ndarray,
    new_tangent: np.ndarray,
    enforce_continuity: bool = True,
    orthonormalize: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Transfer old frame orientation onto a new tangent field by minimal rotation.

    Assumes row-wise frame convention:
        old_frame[i, 0] = T_old[i]
        old_frame[i, 1] = N_old[i]
        old_frame[i, 2] = B_old[i]

    Parameters
    ----------
    old_frame : (N, 3, 3)
        Source frame field, row-wise [T; N; B].
    old_tangent : (N, 3)
        Source tangents.
    new_tangent : (N, 3)
        Target tangents.
    enforce_continuity : bool
        If True, flips N/B signs to keep neighboring sections phase-consistent.
    orthonormalize : bool
        If True, does a final Gram-Schmidt cleanup.
    eps : float
        Numerical epsilon.

    Returns
    -------
    new_frame : (N, 3, 3)
        Transferred frame, row-wise [T_new; N_new; B_new].
    """
    old_frame = np.asarray(old_frame, dtype=np.float64)
    old_tangent = np.asarray(old_tangent, dtype=np.float64)
    new_tangent = np.asarray(new_tangent, dtype=np.float64)

    assert old_frame.ndim == 3 and old_frame.shape[1:] == (3, 3), old_frame.shape
    assert old_tangent.shape == new_tangent.shape, (old_tangent.shape, new_tangent.shape)
    assert old_tangent.shape[0] == old_frame.shape[0], (old_tangent.shape, old_frame.shape)

    old_tangent = _normalize(old_tangent, eps)
    new_tangent = _normalize(new_tangent, eps)

    # Minimal rotations T_old -> T_new
    R = _rodrigues_from_a_to_b(old_tangent, new_tangent, eps=eps)  # (N,3,3)

    # Rotate old N/B by the same minimal rotation
    old_N = old_frame[:, 1, :]
    old_B = old_frame[:, 2, :]

    new_N = np.einsum('nij,nj->ni', R, old_N)
    new_B = np.einsum('nij,nj->ni', R, old_B)

    new_frame = np.stack([new_tangent, new_N, new_B], axis=1)

    if orthonormalize:
        new_frame = _orthonormalize_frame(new_frame, eps=eps)

    if enforce_continuity:
        T = new_frame[:, 0, :].copy()
        N = new_frame[:, 1, :].copy()
        B = new_frame[:, 2, :].copy()

        for i in range(1, len(T)):
            # Keep phase consistent with previous section
            if np.dot(N[i], N[i - 1]) < 0.0:
                N[i] *= -1.0
                B[i] *= -1.0

        new_frame = np.stack([T, N, B], axis=1)

        if orthonormalize:
            new_frame = _orthonormalize_frame(new_frame, eps=eps)

    return new_frame
