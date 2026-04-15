import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

try:
    import trimesh
except ImportError:
    trimesh = None


def safe_norm(x, axis=None, keepdims=False, eps=1e-12):
    return np.sqrt(np.sum(x * x, axis=axis, keepdims=keepdims) + eps)


def normalize(x, axis=-1, eps=1e-12):
    return x / safe_norm(x, axis=axis, keepdims=True, eps=eps)


def skew(v):
    x, y, z = v
    return np.array([
        [0.0, -z,   y],
        [z,    0.0, -x],
        [-y,   x,   0.0]
    ], dtype=np.float64)


def minimal_rotation_matrix(a, b, eps=1e-12):
    a = normalize(np.asarray(a, dtype=np.float64)[None])[0]
    b = normalize(np.asarray(b, dtype=np.float64)[None])[0]

    v = np.cross(a, b)
    c = np.clip(np.dot(a, b), -1.0, 1.0)
    s = np.linalg.norm(v)

    if s < eps:
        if c > 0:
            return np.eye(3)
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(a[0]) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = normalize(np.cross(a, tmp)[None])[0]
        K = skew(axis)
        return np.eye(3) + 2.0 * (K @ K)

    axis = v / s
    theta = np.arccos(c)
    K = skew(axis)
    R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    return R


def compute_rmf(points):
    points = np.asarray(points, dtype=np.float64)
    K = len(points)

    if K == 1:
        T = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
        N = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
        B = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
        return T, N, B

    T = np.zeros((K, 3), dtype=np.float64)
    T[0] = points[1] - points[0]
    T[-1] = points[-1] - points[-2]
    if K > 2:
        T[1:-1] = points[2:] - points[:-2]
    T = normalize(T)

    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(ref, T[0])) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    N0 = ref - np.dot(ref, T[0]) * T[0]
    N0 = normalize(N0[None])[0]
    B0 = normalize(np.cross(T[0], N0)[None])[0]

    N = np.zeros((K, 3), dtype=np.float64)
    B = np.zeros((K, 3), dtype=np.float64)
    N[0] = N0
    B[0] = B0

    for i in range(1, K):
        R = minimal_rotation_matrix(T[i - 1], T[i])
        N[i] = normalize((R @ N[i - 1])[None])[0]
        B[i] = normalize(np.cross(T[i], N[i])[None])[0]

    return T, N, B


def set_axes_equal(ax, points):
    points = np.asarray(points)
    if len(points) == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def circle3d(center, N, B, r, n=48):
    th = np.linspace(0.0, 2.0 * np.pi, n)
    pts = center[None] + r * (
        np.cos(th)[:, None] * N[None] + np.sin(th)[:, None] * B[None]
    )
    return pts


def plot_mesh_wire(ax, vertices, faces, stride=8, alpha=0.12, linewidth=0.2):
    if vertices is None or faces is None:
        return
    for tri in faces[::max(1, stride)]:
        pts = vertices[tri]
        cyc = np.vstack([pts, pts[:1]])
        ax.plot(cyc[:, 0], cyc[:, 1], cyc[:, 2], alpha=alpha, linewidth=linewidth)


def load_npz_dict(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    out = {}
    for k in data.files:
        out[k] = data[k]
    return out


def get_segment_fields(data, sid):
    """
    Assumes keys are saved like:
      part_000_wrap_curve
      part_000_wrap_radius
      ...
    Adjust here if your exact export naming differs.
    """
    #prefix = f"segment_{sid:03d}_"
    prefix = f"segment_{sid:00d}_"

    def grab(name, required=True):
        key = prefix + name
        if key not in data:
            if required:
                raise KeyError(f"Missing key: {key}")
            return None
        return data[key]

    fields = {
        "wrap_curve": grab("polyline_wrap", required=False),
        "center_curve": grab("polyline_center", required=False),
        "keypoints": grab("keypoints"),
        "key_ts": grab("key_ts", required=False),
        "train_radius": grab("key_radius_train", required=False),
        "wrap_radius": grab("key_radius_wrap"),
        "cylinder_radius": grab("cylinder_radius"),
        "surface_points_owned": grab("surface_points_owned", required=False),
        "surface_points_candidate": grab("surface_points_candidate", required=False),
    }
    return fields


def plot_radius_profiles(key_t, train_radius, wrap_radius, cylinder_radius, out_path):
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)

    if key_t is None:
        n = len(wrap_radius)
        key_t = np.linspace(0.0, 1.0, n)

    if train_radius is not None:
        ax.plot(key_t, train_radius, linewidth=2, label="train_radius")
    ax.plot(key_t, wrap_radius, linewidth=2, label="wrap_radius")
    ax.plot(key_t, cylinder_radius, linewidth=2, label="cylinder_radius")

    ax.set_xlabel("s")
    ax.set_ylabel("radius")
    ax.set_title("Radius profiles")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_radius_overlay(curve, radius, surface_points, mesh_vertices, mesh_faces, out_path, title, every=5):
    T, N, B = compute_rmf(curve)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    if mesh_vertices is not None and mesh_faces is not None:
        plot_mesh_wire(ax, mesh_vertices, mesh_faces)

    if surface_points is not None and len(surface_points):
        ax.scatter(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2], s=1)

    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], linewidth=2)

    idxs = np.arange(0, len(curve), max(1, every))
    for i in idxs:
        ring = circle3d(curve[i], N[i], B[i], radius[i], n=48)
        ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], linewidth=1)

    pts = [curve]
    if surface_points is not None and len(surface_points):
        pts.append(surface_points)
    if mesh_vertices is not None:
        pts.append(mesh_vertices)
    set_axes_equal(ax, np.vstack(pts))

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def build_cylinder_surface(curve, radius, n_theta=24):
    T, N, B = compute_rmf(curve)
    th = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    surf = np.zeros((len(curve), n_theta, 3), dtype=np.float64)
    for i in range(len(curve)):
        surf[i] = curve[i][None] + radius[i] * (
            np.cos(th)[:, None] * N[i][None] + np.sin(th)[:, None] * B[i][None]
        )
    return surf


def plot_cylinder_overlay(curve, cylinder_radius, surface_points, mesh_vertices, mesh_faces, out_path):
    surf = build_cylinder_surface(curve, cylinder_radius, n_theta=24)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    if mesh_vertices is not None and mesh_faces is not None:
        plot_mesh_wire(ax, mesh_vertices, mesh_faces)

    if surface_points is not None and len(surface_points):
        ax.scatter(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2], s=1)

    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], linewidth=2)

    for j in range(surf.shape[1]):
        pts = surf[:, j, :]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=0.8)

    for i in range(0, surf.shape[0], 6):
        ring = np.vstack([surf[i], surf[i, :1]])
        ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], linewidth=0.8)

    pts = [curve, surf.reshape(-1, 3)]
    if surface_points is not None and len(surface_points):
        pts.append(surface_points)
    if mesh_vertices is not None:
        pts.append(mesh_vertices)
    set_axes_equal(ax, np.vstack(pts))

    ax.set_title("Cylinder radius overlay")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True)
    parser.add_argument("--segment_id", type=int, required=True)
    parser.add_argument("--mesh_file", type=str, default=None)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data = load_npz_dict(args.npz)
    fields = get_segment_fields(data, args.segment_id)

    mesh_vertices = None
    mesh_faces = None
    if args.mesh_file is not None:
        if trimesh is None:
            raise ImportError("trimesh is required to load mesh_file")
        mesh = trimesh.load(args.mesh_file, process=False)
        if hasattr(mesh, "vertices"):
            mesh_vertices = np.asarray(mesh.vertices)
        if hasattr(mesh, "faces"):
            mesh_faces = np.asarray(mesh.faces)

    curve = fields["wrap_curve"]
    key_t = fields["key_ts"]
    train_radius = fields["train_radius"]
    wrap_radius = fields["wrap_radius"]
    cylinder_radius = fields["cylinder_radius"]
    surface_points = fields["surface_points_owned"]

    plot_radius_profiles(
        key_t=key_t,
        train_radius=train_radius,
        wrap_radius=wrap_radius,
        cylinder_radius=cylinder_radius,
        out_path=os.path.join(args.out_dir, f"segment_{args.segment_id:03d}_radius_profiles.png"),
    )

    if train_radius is not None:
        plot_radius_overlay(
            curve=curve,
            radius=train_radius,
            surface_points=surface_points,
            mesh_vertices=mesh_vertices,
            mesh_faces=mesh_faces,
            out_path=os.path.join(args.out_dir, f"segment_{args.segment_id:03d}_train_radius.png"),
            title="Train radius rings",
        )

    plot_radius_overlay(
        curve=curve,
        radius=wrap_radius,
        surface_points=surface_points,
        mesh_vertices=mesh_vertices,
        mesh_faces=mesh_faces,
        out_path=os.path.join(args.out_dir, f"segment_{args.segment_id:03d}_wrap_radius.png"),
        title="Wrap radius rings",
    )

    plot_cylinder_overlay(
        curve=curve,
        cylinder_radius=cylinder_radius,
        surface_points=surface_points,
        mesh_vertices=mesh_vertices,
        mesh_faces=mesh_faces,
        out_path=os.path.join(args.out_dir, f"segment_{args.segment_id:03d}_cylinder_radius.png"),
    )

    summary = {
        "segment_id": args.segment_id,
        "has_train_radius": train_radius is not None,
        "n_curve_points": int(len(curve)),
        "n_surface_points": 0 if surface_points is None else int(len(surface_points)),
    }
    with open(os.path.join(args.out_dir, f"segment_{args.segment_id:03d}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
