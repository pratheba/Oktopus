import numpy as np
import trimesh

def extract_segment_mesh(original_mesh_path, segment_vertex_ids, output_path):
    mesh = trimesh.load(original_mesh_path, process=False)

    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)

    segment_vertex_ids = np.asarray(segment_vertex_ids, dtype=np.int64)
    keep_vertex = np.zeros(len(V), dtype=bool)
    keep_vertex[segment_vertex_ids] = True

    # keep faces fully inside the segment
    keep_face = keep_vertex[F].all(axis=1)
    F_seg_old = F[keep_face]

    # remap old vertex ids → compact new ids
    used_old_ids = np.unique(F_seg_old.reshape(-1))
    old_to_new = -np.ones(len(V), dtype=np.int64)
    old_to_new[used_old_ids] = np.arange(len(used_old_ids))

    V_seg = V[used_old_ids]
    F_seg = old_to_new[F_seg_old]

    seg_mesh = trimesh.Trimesh(vertices=V_seg, faces=F_seg, process=False)
    seg_mesh.export(output_path)

    return seg_mesh


def extract_boundary_loops(mesh):
    edges = mesh.edges_sorted
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]

    graph = {}
    for a, b in boundary_edges:
        graph.setdefault(a, []).append(b)
        graph.setdefault(b, []).append(a)

    loops = []
    used = set()

    for a, b in boundary_edges:
        e = tuple(sorted((a, b)))
        if e in used:
            continue

        loop = [a]
        prev, cur = -1, a

        while True:
            nbrs = graph[cur]
            nxts = [n for n in nbrs if n != prev]

            if len(nxts) == 0:
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

        if len(loop) > 3:
            loops.append(np.array(loop, dtype=np.int64))

    return loops


def cap_boundary_loop(mesh, loop):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    loop_pts = vertices[loop]
    center = loop_pts.mean(axis=0)

    center_id = len(vertices)
    new_vertices = np.vstack([vertices, center[None]])

    new_faces = []

    for i in range(len(loop)):
        a = loop[i]
        b = loop[(i + 1) % len(loop)]
        new_faces.append([a, b, center_id])

    new_faces = np.asarray(new_faces, dtype=np.int64)
    capped_faces = np.vstack([faces, new_faces])

    return trimesh.Trimesh(
        vertices=new_vertices,
        faces=capped_faces,
        process=False
    )


def cap_all_boundary_loops(mesh, min_loop_len=8):
    loops = extract_boundary_loops(mesh)

    capped = mesh.copy()

    # cap larger loops first
    loops = sorted(loops, key=lambda x: len(x), reverse=True)

    for loop in loops:
        if len(loop) < min_loop_len:
            continue
        capped = cap_boundary_loop(capped, loop)

    return capped, loops


if __name__ == '__main__':
    segment_mesh = extract_segment_mesh(original_mesh_path, segment_vertex_ids, output_path)
    mesh = trimesh.load("shirt_body_segment.ply", process=False)

    closed_mesh, loops = cap_all_boundary_loops(mesh)

    print("boundary loops:", len(loops))
    for i, loop in enumerate(loops):
        pts = mesh.vertices[loop]
        print(i, "num verts:", len(loop), "center:", pts.mean(axis=0))

    closed_mesh.export("shirt_body_segment_capped.ply")