import numpy as np
import trimesh
import os

# ----------------------------
# Basic vector utils
# ----------------------------
def normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + eps)

def project_onto_plane(v, n):
    v = np.asarray(v, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    return v - np.sum(v * n, axis=-1, keepdims=True) * n

# ----------------------------
# Cubic Bezier
# ----------------------------
def bezier_cubic(P0, P1, P2, P3, t):
    """t: (K,) in [0,1] -> (K,3)"""
    t = np.asarray(t, dtype=np.float64)
    omt = 1.0 - t
    omt2 = omt * omt
    t2 = t * t
    b0 = omt2 * omt
    b1 = 3.0 * omt2 * t
    b2 = 3.0 * omt * t2
    b3 = t2 * t
    return (b0[:, None] * P0 +
            b1[:, None] * P1 +
            b2[:, None] * P2 +
            b3[:, None] * P3)

# ----------------------------
# Rotation-minimizing frame (parallel transport)
# ----------------------------
def parallel_transport_frames(points, init_normal=None):
    """
    points: (K,3) polyline
    Returns frame: (K,3,3) with columns [T,N,B]
    """
    points = np.asarray(points, dtype=np.float64)
    K = points.shape[0]
    assert K >= 2 and points.shape[1] == 3

    seg = points[1:] - points[:-1]     # (K-1,3)
    T = np.zeros((K, 3), dtype=np.float64)
    T[:-1] = normalize(seg)
    T[-1] = T[-2]

    T0 = T[0]

    if init_normal is None:
        axes = np.eye(3, dtype=np.float64)
        dots = np.abs(axes @ T0)
        a = axes[np.argmin(dots)]
        n0 = np.cross(T0, a)
        n0 = normalize(n0)
    else:
        n0 = np.asarray(init_normal, dtype=np.float64)
        n0 = project_onto_plane(n0, T0)
        n0 = normalize(n0)

    N = np.zeros((K, 3), dtype=np.float64)
    B = np.zeros((K, 3), dtype=np.float64)
    N[0] = n0
    B[0] = np.cross(T0, N[0])

    for i in range(1, K):
        t_prev = T[i - 1]
        t_cur  = T[i]

        v = np.cross(t_prev, t_cur)
        s = np.linalg.norm(v)
        c = float(np.dot(t_prev, t_cur))

        if s < 1e-10:
            n = N[i - 1].copy()
        else:
            k = v / s
            n_prev = N[i - 1]
            # Rodrigues rotation
            n = (n_prev * c +
                 np.cross(k, n_prev) * s +
                 k * (np.dot(k, n_prev)) * (1.0 - c))

        n = project_onto_plane(n, t_cur)
        n = normalize(n)
        b = np.cross(t_cur, n)

        N[i] = n
        B[i] = b

    frame = np.stack([T, N, B], axis=-1)  # (K,3,3), columns
    return frame

# ----------------------------
# Non-uniform monotonic ts in [0,1]
# ----------------------------
def make_nonuniform_ts(K, bias=1.35):
    u = np.linspace(0.0, 1.0, K, dtype=np.float64)
    ts = u**bias
    ts = (ts - ts[0]) / (ts[-1] - ts[0] + 1e-12)
    ts[0], ts[-1] = 0.0, 1.0
    return ts

# ----------------------------
# Tentacle radius profile (y,z)
# ----------------------------
def tentacle_radius_profile(ts, base_y=0.18, base_z=0.14, tip_min=0.02, wobble=0.10, seed=0):
    rng = np.random.default_rng(seed)
    ts = np.asarray(ts, dtype=np.float64)

    taper = (1.0 - ts)**1.8
    y = tip_min + base_y * taper
    z = tip_min + base_z * taper

    f1 = rng.uniform(1.0, 2.5)
    f2 = rng.uniform(2.0, 4.0)
    ph1 = rng.uniform(0.0, 2*np.pi)
    ph2 = rng.uniform(0.0, 2*np.pi)

    wob = 1.0 + wobble * (0.6*np.sin(2*np.pi*f1*ts + ph1) + 0.4*np.sin(2*np.pi*f2*ts + ph2))
    wob = np.clip(wob, 0.75, 1.35)

    y *= wob
    z *= wob * rng.uniform(0.95, 1.05)

    x = np.ones_like(y)  # placeholder
    return np.stack([x, y, z], axis=-1)

# ----------------------------
# Local->World ring transform (rigid)
# ----------------------------
def make_ring_transform(i, num_tentacles, ring_radius=0.25, center=(0.0, 0.0, 0.0)):
    """
    Returns R,t such that:
      world = (R @ local) + t
    We rotate canonical +X to outward direction at angle i.
    """
    center = np.asarray(center, dtype=np.float64)
    ang = 2*np.pi * (i / num_tentacles)

    t = center + ring_radius * np.array([np.cos(ang), np.sin(ang), 0.0], dtype=np.float64)

    outward = np.array([np.cos(ang), np.sin(ang), 0.0], dtype=np.float64)
    outward = normalize(outward)

    ex = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    v = np.cross(ex, outward)
    s = np.linalg.norm(v)
    c = float(np.dot(ex, outward))

    if s < 1e-10:
        # either identical (+X) or opposite (-X)
        if c > 0:
            R = np.eye(3, dtype=np.float64)
        else:
            R = np.diag([-1.0, -1.0, 1.0]).astype(np.float64)
    else:
        k = v / s
        K = np.array([[0, -k[2], k[1]],
                      [k[2], 0, -k[0]],
                      [-k[1], k[0], 0]], dtype=np.float64)
        # Rodrigues: R = I + K*s + K^2*(1-c)
        R = np.eye(3, dtype=np.float64) + K * s + (K @ K) * (1.0 - c)

    return R, t

def apply_rigid(points, frame, R, t):
    """
    points: (K,3)
    frame:  (K,3,3) columns [T,N,B]
    R: (3,3), t: (3,)
    """
    points = np.asarray(points, dtype=np.float64)
    frame = np.asarray(frame, dtype=np.float64)
    Pw = (points @ R.T) + t[None, :]
    # rotate frame columns by R
    Fw = np.einsum('ij,kjl->kil', R, frame)
    return Pw, Fw

# ----------------------------
# Main generator: returns BOTH local & world skeletons
# ----------------------------
def create_octopus_tentacle_skeletons_local_world(
    num_tentacles=8,
    K=200,
    ring_radius=0.25,
    ring_center=(0.0, 0.0, 0.0),
    tentacle_length=2.0,
    curl_strength=0.6,
    lift=0.15,
    nonuniform_bias=1.35,
    seed=42,
):
    """
    Returns list[dict] with keys:
      key_ts: (K,)
      key_points_local: (K,3)
      key_frame_local:  (K,3,3) columns [T,N,B]
      key_points: (K,3) world
      key_frame:  (K,3,3) world
      radius: (K,3) [x,y,z]
      world_R: (3,3)
      world_t: (3,)
      bezier_ctrl_local: (4,3)
    """
    rng = np.random.default_rng(seed)

    key_ts = make_nonuniform_ts(K, bias=nonuniform_bias)

    # canonical directions for LOCAL tentacle
    ex = np.array([1.0, 0.0, 0.0], dtype=np.float64)  # forward
    ey = np.array([0.0, 1.0, 0.0], dtype=np.float64)  # side curl
    ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # up

    skels = []
    for i in range(num_tentacles):
        # randomize per tentacle
        L = tentacle_length * rng.uniform(0.85, 1.20)
        curl = curl_strength * rng.uniform(0.75, 1.25)
        lift_i = lift * rng.uniform(0.6, 1.6)

        # LOCAL cubic Bezier control points:
        # root at origin, grow along +X, curl in +Y, lift in +Z
        P0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # initial & final direction vectors (local)
        dir0 = normalize(ex + 0.25 * ez)
        dir3 = normalize(0.65 * ex + curl * ey + 0.20 * ez)

        P3 = P0 + L * (0.95 * ex + (0.22 * curl) * ey + lift_i * ez)

        P1 = P0 + (0.35 * L) * dir0 + rng.uniform(-0.04, 0.04, size=3)
        P2 = P3 - (0.35 * L) * dir3 + rng.uniform(-0.04, 0.04, size=3)

        pts_local = bezier_cubic(P0, P1, P2, P3, key_ts)   # (K,3)
        frame_local = parallel_transport_frames(pts_local) # (K,3,3)

        base_y = rng.uniform(0.14, 0.22)
        base_z = rng.uniform(0.10, 0.18)
        rad = tentacle_radius_profile(
            key_ts,
            base_y=base_y,
            base_z=base_z,
            tip_min=rng.uniform(0.015, 0.03),
            wobble=rng.uniform(0.06, 0.14),
            seed=int(seed * 100 + i),
        )

        # World placement on ring
        R, t = make_ring_transform(i, num_tentacles, ring_radius=ring_radius, center=ring_center)
        pts_world, frame_world = apply_rigid(pts_local, frame_local, R, t)

        skels.append({
            "key_ts": key_ts.copy(),
            "key_points_local": pts_local,
            "key_frame_local": frame_local,
            "key_points": pts_world,
            "key_frame": frame_world,
            "radius": rad,
            "world_R": R,
            "world_t": t,
            "bezier_ctrl_local": np.stack([P0, P1, P2, P3], axis=0),
        })

    return skels



def visualize_tentacles(skeletons, n_circle=24):
    meshes = []
    for i,sk in enumerate(skeletons):
        print(sk["radius"])
        #mesh = tube_from_skeleton(sk["key_points"], sk["key_frame"], sk["radius"], n_circle=n_circle)
        mesh = tube_from_skeleton(sk["key_points_local"], sk["key_frame_local"], sk["radius"], n_circle=n_circle)
        path = os.path.join('.', f"tentacle_{i:02d}.ply")
        mesh.export(path)
        meshes.append(mesh)
    #scene = trimesh.Scene(meshes)
    #scene.show()

    combined = trimesh.util.concatenate(meshes)  # meshes: list of Trimesh
    combined.export("octopus_tentacles.ply")
    #scene.export('octopus_tentacles.ply')


def tube_from_skeleton(key_points, key_frame, radius, n_circle=24):
    """
    key_points: [K,3]
    key_frame:  [K,3,3] columns [T,N,B]
    radius:     [K,3]   [x, y, z] we use y,z as ellipse radii in N,B
    """
    K = key_points.shape[0]
    angles = np.linspace(0, 2*np.pi, n_circle, endpoint=False)

    # circle points in (N,B) plane -> ellipse radii (ry, rz)
    cos_a = np.cos(angles)[None, :]  # [1,C]
    sin_a = np.sin(angles)[None, :]  # [1,C]

    verts = []
    for i in range(K):
        c = key_points[i]               # [3]
        T = key_frame[i,:,0]
        N = key_frame[i,:,1]
        B = key_frame[i,:,2]

        ry = radius[i,1]
        rz = radius[i,2]

        ring = c[None, :] + (ry * cos_a.T) * N[None, :] + (rz * sin_a.T) * B[None, :]
        # ring: [C,3]
        verts.append(ring)

    verts = np.vstack(verts)  # [K*C, 3]

    # faces: connect ring i to ring i+1
    faces = []
    C = n_circle
    for i in range(K - 1):
        base0 = i * C
        base1 = (i + 1) * C
        for j in range(C):
            jn = (j + 1) % C
            v00 = base0 + j
            v01 = base0 + jn
            v10 = base1 + j
            v11 = base1 + jn
            # two triangles per quad
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

    faces = np.asarray(faces, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return mesh

if __name__ == '__main__':
    skeletons = create_octopus_tentacle_skeletons_local_world(K=200, num_tentacles=8)
    visualize_tentacles(skeletons, n_circle=24)
