import os
import argparse
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def load_segments(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return data["segments"]


def extract_boundary_loops(mesh):
    edges = mesh.edges_sorted
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]

    graph = {}
    for a, b in boundary_edges:
        graph.setdefault(int(a), []).append(int(b))
        graph.setdefault(int(b), []).append(int(a))

    loops = []
    used = set()

    for a, b in boundary_edges:
        a, b = int(a), int(b)
        e = tuple(sorted((a, b)))
        if e in used:
            continue

        loop = [a]
        prev, cur = -1, a

        while True:
            nbrs = graph.get(cur, [])
            nxts = [n for n in nbrs if n != prev]

            if not nxts:
                break

            nxt = nxts[0]
            e = tuple(sorted((cur, nxt)))

            if e in used:
                break

            used.add(e)
            loop.append(nxt)

            prev, cur = cur, nxt

            if cur == a:
                break

        if len(loop) >= 3:
            loops.append(np.asarray(loop, dtype=np.int64))

    return loops


def cap_loop(mesh, loop):
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)

    pts = V[loop]
    center = pts.mean(axis=0)
    center_id = len(V)

    new_V = np.vstack([V, center[None]])
    new_faces = []

    for i in range(len(loop)):
        a = loop[i]
        b = loop[(i + 1) % len(loop)]
        new_faces.append([a, b, center_id])

    new_F = np.vstack([F, np.asarray(new_faces, dtype=np.int64)])

    return trimesh.Trimesh(vertices=new_V, faces=new_F, process=False)


def cap_boundary_loops(mesh, min_loop_len=8, max_loops=None):
    loops = extract_boundary_loops(mesh)
    loops = sorted(loops, key=lambda x: len(x), reverse=True)

    capped = mesh.copy()

    capped_count = 0
    for loop in loops:
        if len(loop) < min_loop_len:
            continue

        if max_loops is not None and capped_count >= max_loops:
            break

        capped = cap_loop(capped, loop)
        capped_count += 1

    return capped, loops, capped_count


def extract_segment_mesh_from_points(
    original_mesh,
    segment_points,
    distance_threshold=None,
    face_mode="all",
):
    V = np.asarray(original_mesh.vertices)
    F = np.asarray(original_mesh.faces)

    tree = cKDTree(V)
    dists, vertex_ids = tree.query(segment_points, k=1)

    if distance_threshold is not None:
        vertex_ids = vertex_ids[dists <= distance_threshold]

    vertex_ids = np.unique(vertex_ids)

    keep_vertex = np.zeros(len(V), dtype=bool)
    keep_vertex[vertex_ids] = True

    if face_mode == "all":
        keep_face = keep_vertex[F].all(axis=1)
    elif face_mode == "any":
        keep_face = keep_vertex[F].any(axis=1)
    else:
        raise ValueError("face_mode must be 'all' or 'any'")

    F_old = F[keep_face]

    if len(F_old) == 0:
        return None, vertex_ids

    used_old = np.unique(F_old.reshape(-1))
    old_to_new = -np.ones(len(V), dtype=np.int64)
    old_to_new[used_old] = np.arange(len(used_old))

    V_seg = V[used_old]
    F_seg = old_to_new[F_old]

    seg_mesh = trimesh.Trimesh(vertices=V_seg, faces=F_seg, process=False)
    seg_mesh.remove_unreferenced_vertices()

    return seg_mesh, vertex_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, help="Original full closed mesh .ply/.obj")
    parser.add_argument("--npz", required=True, help="Postprocess segments npz")
    parser.add_argument("--out_dir", required=True, help="Output folder")
    parser.add_argument("--point_key", default="surface_points_all")
    parser.add_argument("--distance_threshold", type=float, default=None)
    parser.add_argument("--face_mode", default="all", choices=["all", "any"])
    parser.add_argument("--min_loop_len", type=int, default=8)
    parser.add_argument("--max_loops", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    original_mesh = trimesh.load(args.mesh, process=False)
    segments = load_segments(args.npz)

    print("Loaded mesh:")
    print("  vertices:", len(original_mesh.vertices))
    print("  faces:", len(original_mesh.faces))
    print("Loaded segments:", len(segments))

    for i, seg in enumerate(segments):
        seg_id = seg.get("id", i)
        seg_name = seg.get("name", f"segment_{seg_id}")

        if args.point_key not in seg:
            print(f"[skip] segment {i}: no key {args.point_key}")
            continue

        pts = np.asarray(seg[args.point_key], dtype=np.float64)

        seg_mesh, matched_vertex_ids = extract_segment_mesh_from_points(
            original_mesh,
            pts,
            distance_threshold=args.distance_threshold,
            face_mode=args.face_mode,
        )

        if seg_mesh is None:
            print(f"[skip] segment {i}: no faces extracted")
            continue

        raw_path = os.path.join(args.out_dir, f"{seg_id}_{seg_name}_open.ply")
        closed_path = os.path.join(args.out_dir, f"{seg_id}_{seg_name}_closed.ply")

        seg_mesh.export(raw_path)

        closed_mesh, loops, capped_count = cap_boundary_loops(
            seg_mesh,
            min_loop_len=args.min_loop_len,
            max_loops=args.max_loops,
        )

        closed_mesh.export(closed_path)

        print(f"[segment {i}] id={seg_id}, name={seg_name}")
        print("  input surface points:", len(pts))
        print("  matched vertices:", len(matched_vertex_ids))
        print("  open mesh vertices/faces:", len(seg_mesh.vertices), len(seg_mesh.faces))
        print("  boundary loops:", len(loops))
        print("  capped loops:", capped_count)
        print("  saved:", closed_path)


if __name__ == "__main__":
    main()