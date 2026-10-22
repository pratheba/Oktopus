"""Conservative boundary-loop stitching of 0082 UDF-cut root/leaf parts.

Never silently export concatenated shells as stitched. The operation preserves
all input triangles, introduces only attachment strips, and fails closed unless
it joins the body and seven leaves into ONE face/edge-connected mesh.

This is not Boolean union, intersection trimming, or watertight repair; it needs
compatible exposed attachment loops and may refuse real meshes that require
those additional operations. All inputs and existing 0082 outputs stay intact.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import trimesh

PARTS = ('body', 'tentacle_t1', 'tentacle_t2', 'tentacle_t3',
         'tentacle_t4', 'tentacle_t5', 'tentacle_t6', 'tentacle_t8')


def mesh_edges(faces):
    f = np.asarray(faces, dtype=np.int64)
    if len(f) == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int32)
    edges = np.sort(f[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return unique, counts


def edge_connected_count(n_faces, faces):
    """Count face components through shared mesh edges (not touching vertices)."""
    if not n_faces:
        return 0
    faces = np.asarray(faces, dtype=np.int64)
    all_edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    owners = np.repeat(np.arange(n_faces, dtype=np.int64), 3)
    _, inverse = np.unique(all_edges, return_inverse=True, axis=0)
    order = np.argsort(inverse, kind='stable')
    group = inverse[order]
    sorted_owner = owners[order]
    same_edge = np.flatnonzero(group[:-1] == group[1:])
    graph = coo_matrix((np.ones(len(same_edge), dtype=np.int8),
                        (sorted_owner[same_edge], sorted_owner[same_edge + 1])),
                       shape=(n_faces, n_faces)).tocsr()
    return int(connected_components(graph, directed=False, return_labels=False))


def boundary_loops(mesh):
    """Return simple boundary cycles and counts of ambiguous open/branched edges.

    A boundary component is a cycle iff every vertex has degree 2 within that
    component. We intentionally do not sew branched/nonmanifold boundaries.
    """
    edge, counts = mesh_edges(mesh.faces)
    borders = edge[counts == 1]
    neighbors = {}
    for u, v in borders:
        u, v = int(u), int(v)
        neighbors.setdefault(u, []).append(v)
        neighbors.setdefault(v, []).append(u)
    seen = set()
    cycles = []
    rejected_edges = 0
    for start in neighbors:
        if start in seen:
            continue
        stack = [start]
        connected = set()
        while stack:
            u = stack.pop()
            if u in connected:
                continue
            connected.add(u)
            stack.extend(neighbors[u])
        seen.update(connected)
        if any(len(neighbors[u]) != 2 for u in connected):
            rejected_edges += sum(len(neighbors[u]) for u in connected) // 2
            continue
        ordered = [start]
        previous, current = -1, start
        while True:
            nxt = neighbors[current][0]
            if nxt == previous:
                nxt = neighbors[current][1]
            if nxt == start:
                break
            if nxt in ordered:
                raise ValueError('Boundary component closed early; ambiguous loop')
            ordered.append(nxt)
            previous, current = current, nxt
        if len(ordered) >= 3 and len(ordered) == len(connected):
            cycles.append(np.asarray(ordered, dtype=np.int64))
        else:
            rejected_edges += len(connected)
    return cycles, rejected_edges


def _loop_points(vertices, loop):
    return np.asarray(vertices, dtype=np.float64)[loop]


def _nearest_loop_match(root_mesh, root_loop, leaf_mesh, leaf_loop):
    a = _loop_points(root_mesh.vertices, root_loop)
    b = _loop_points(leaf_mesh.vertices, leaf_loop)
    ab = cKDTree(b).query(a, k=1)[0]
    ba = cKDTree(a).query(b, k=1)[0]
    return {
        'median_distance': float((np.median(ab) + np.median(ba)) / 2),
        'p95_distance': float(max(np.percentile(ab, 95), np.percentile(ba, 95))),
        'max_nearest_distance': float(max(np.max(ab), np.max(ba))),
        'root_loop_vertices': len(a), 'leaf_loop_vertices': len(b),
    }


def _cyclic_align(a, b):
    """Rotate/reverse two boundary cycles into a nearby compatible zipper."""
    xyz_a = np.asarray(a, dtype=np.float64)
    xyz_b = np.asarray(b, dtype=np.float64)
    _, nearest = cKDTree(xyz_b).query(xyz_a, k=1)
    paired_dist = np.linalg.norm(xyz_a - xyz_b[nearest], axis=1)
    options = np.argsort(paired_dist)[:min(24, len(a))]
    best = None
    for ai in options:
        bi = int(nearest[ai])
        rotated_a = np.roll(np.arange(len(a), dtype=np.int64), -int(ai))
        for flip in (False, True):
            order_b = np.arange(len(b), dtype=np.int64)
            if flip:
                order_b = order_b[::-1]
            bix = int(np.where(order_b == bi)[0][0])
            rotated_b = np.roll(order_b, -bix)
            sa = xyz_a[rotated_a[np.floor(np.arange(32)*len(a)/32).astype(int)]]
            sb = xyz_b[rotated_b[np.floor(np.arange(32)*len(b)/32).astype(int)]]
            error = float(np.median(np.linalg.norm(sa - sb, axis=1)))
            if best is None or error < best[0]:
                best = (error, rotated_a, rotated_b)
    return best[1], best[2]


def zipper_strip(vertices, root_loop, leaf_loop, *, max_bridge_edge):
    """Build triangles bridging matched simple loops; no input faces changed."""
    xyz_a = _loop_points(vertices, root_loop)
    xyz_b = _loop_points(vertices, leaf_loop)
    order_a, order_b = _cyclic_align(xyz_a, xyz_b)
    a = root_loop[order_a]
    b = leaf_loop[order_b]
    # Use normalized path arclength for loop-compatible strip triangulation.
    def marks(idx):
        pts = vertices[idx]
        lengths = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
        perimeter = float(lengths.sum())
        if perimeter <= 0:
            raise ValueError('Zero length attachment boundary')
        return np.concatenate([[0.0], np.cumsum(lengths) / perimeter])
    t_a, t_b = marks(a), marks(b)
    triangles = []
    i = j = 0
    n, m = len(a), len(b)
    while i < n or j < m:
        if j == m or (i < n and t_a[i+1] <= t_b[j+1]):
            triangles.append([int(a[i % n]), int(a[(i + 1) % n]), int(b[j % m])])
            i += 1
        else:
            triangles.append([int(a[i % n]), int(b[(j+1) % m]), int(b[j % m])])
            j += 1
    tri = np.asarray(triangles, dtype=np.int64)
    edges = vertices[tri[:, [0, 1, 2, 0]]]
    lengths = np.linalg.norm(edges[:, 1:] - edges[:, :-1], axis=2)
    length_max = float(lengths.max(initial=0))
    if length_max > max_bridge_edge:
        raise ValueError(f'Stitch bridge edge {length_max:.5g} exceeds '
                         f'--max-bridge-edge {max_bridge_edge:.5g}; refusing long cross-surface triangles')
    cross = np.cross(vertices[tri[:, 1]] - vertices[tri[:, 0]],
                     vertices[tri[:, 2]] - vertices[tri[:, 0]])
    area2 = np.linalg.norm(cross, axis=1)
    if float(area2.min(initial=np.inf)) < 1e-14:
        raise ValueError('Stitch produced degenerate triangles; attachment rings may overlap')
    return tri, {'bridge_faces': len(tri), 'max_bridge_edge': length_max,
                 'min_twice_area': float(area2.min())}


def _load_part(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing 0082 part: {path}')
    mesh = trimesh.load(path, process=False, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f'Expected nonempty triangle PLY: {path}')
    return mesh


def stitch_parts(meshes, *, max_seam_gap=0.025, max_bridge_edge=0.06,
                 max_perimeter_ratio=1.65):
    """Try to stitch 7 open leaf roots to distinct body boundary loops.

    Fail closed if any corresponding loop is missing or ambiguous. Nothing is
    written until this finishes and validates one edge-connected component.
    """
    if set(meshes) != set(PARTS):
        raise ValueError(f'Expected exactly {PARTS}; got {list(meshes)}')
    if max_seam_gap <= 0 or max_bridge_edge <= 0 or max_perimeter_ratio < 1:
        raise ValueError('Stitch tolerances must be positive; perimeter ratio >=1')
    vertices, faces, loops = [], [], {}
    offset = 0
    input_nonmanifold = 0
    input_boundary = 0
    for name in PARTS:
        mesh = meshes[name]
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            raise ValueError(f'Empty part: {name}')
        vertex = np.asarray(mesh.vertices, dtype=np.float64)
        face = np.asarray(mesh.faces, dtype=np.int64)
        edge, count = mesh_edges(face)
        input_nonmanifold += int(np.count_nonzero(count > 2))
        input_boundary += int(np.count_nonzero(count == 1))
        local_loops, rejected = boundary_loops(mesh)
        loops[name] = {'loops': [loop + offset for loop in local_loops],
                       'rejected_boundary_edges': rejected,
                       'all_boundary_edges': int(np.count_nonzero(count == 1))}
        vertices.append(vertex)
        faces.append(face + offset)
        offset += len(vertex)
    vertices = np.concatenate(vertices, axis=0)
    source_faces = np.concatenate(faces, axis=0)
    # Need one coherent surface per part (not 7 hidden debris surfaces).
    for name in PARTS:
        count = edge_connected_count(len(meshes[name].faces), meshes[name].faces)
        if count != 1:
            raise ValueError(f'{name} has {count} pre-existing edge-connected components; '
                             'clean components before stitching')
    root_mesh = trimesh.Trimesh(vertices=vertices, faces=source_faces, process=False)
    body_loops = loops['body']['loops']
    if len(body_loops) < 7:
        raise ValueError(f'Body has only {len(body_loops)} simple boundary loops '
                         f'({loops["body"]["rejected_boundary_edges"]} branched edges), '
                         'but 7 distinct tentacle attachment loops are required. '
                         'Needs interface trimming/repair, not vertex concatenation.')
    used = set()
    strips, stitches = [], []
    for name in PARTS[1:]:
        leaf_loops = loops[name]['loops']
        if not leaf_loops:
            raise ValueError(f'{name} has no usable simple attachment boundary loop')
        matches = []
        for li, leaf_loop in enumerate(leaf_loops):
            lp = _loop_points(vertices, leaf_loop)
            lp_length = float(np.linalg.norm(np.roll(lp,-1,axis=0)-lp,axis=1).sum())
            for ri, root_loop in enumerate(body_loops):
                if ri in used:
                    continue
                rp = _loop_points(vertices, root_loop)
                rp_length = float(np.linalg.norm(np.roll(rp,-1,axis=0)-rp,axis=1).sum())
                perim_ratio = max(lp_length / rp_length, rp_length / lp_length)
                if perim_ratio > max_perimeter_ratio:
                    continue
                metric = _nearest_loop_match(root_mesh, root_loop, root_mesh, leaf_loop)
                if metric['median_distance'] > max_seam_gap or metric['p95_distance'] > 2*max_seam_gap:
                    continue
                matches.append((metric['median_distance'] + 0.2*metric['p95_distance'],
                                ri, li, metric, perim_ratio))
        if not matches:
            raise ValueError(f'{name}: cannot match an attachment loop to a distinct body loop '
                             f'within --max-seam-gap {max_seam_gap}; '
                             'inspect body/leaf boundary loops, consider interface trimming')
        matches.sort(key=lambda row: row[0])
        _, ri, li, metric, perim = matches[0]
        strip, strip_report = zipper_strip(vertices, body_loops[ri], leaf_loops[li],
                                           max_bridge_edge=max_bridge_edge)
        strips.append(strip)
        used.add(ri)
        stitches.append({'leaf': name, 'body_loop': ri, 'leaf_loop': li,
                         'perimeter_ratio': perim, **metric, **strip_report})
    welded_faces = np.concatenate([source_faces, *strips])
    output = trimesh.Trimesh(vertices=vertices, faces=welded_faces, process=False)
    edges, counts = mesh_edges(output.faces)
    nonmanifold = int(np.count_nonzero(counts > 2))
    boundary = int(np.count_nonzero(counts == 1))
    connected = edge_connected_count(len(welded_faces), welded_faces)
    if nonmanifold > input_nonmanifold:
        raise ValueError('Stitch introduced additional nonmanifold edges '
                         f'({input_nonmanifold} -> {nonmanifold}); refusing export')
    if boundary != input_boundary - sum(s['root_loop_vertices']+s['leaf_loop_vertices'] for s in stitches):
        raise ValueError('Stitch produced unexpected boundary edges; refusing export')
    if connected != 1:
        raise ValueError(f'Stitched candidate remains {connected} edge-connected components; '
                         'not exporting misleading STITCHED filename')
    return output, {
        'stitched_root_leaf_interfaces': True,
        'connected_mesh': True,
        'watertight': bool(output.is_watertight),
        'winding_consistent': bool(output.is_winding_consistent),
        'parts': list(PARTS),
        'stitches': stitches,
        'faces_before': len(source_faces),
        'faces_after': len(output.faces),
        'nonmanifold_edges_before': input_nonmanifold,
        'nonmanifold_edges_after': nonmanifold,
        'boundary_edges_before': input_boundary,
        'boundary_edges_after': boundary,
        'edge_connected_components_after': connected,
        'unmatched_body_boundary_loops': len(body_loops) - len(used),
        'unsupported_branched_boundary_edges': {
            name: entry['rejected_boundary_edges'] for name, entry in loops.items()
        },
        'warning': 'Attachment strip sewn; remaining garment openings remain open; '
                   'self-intersections are not exhaustively audited.',
    }


def preflight(meshes):
    """Report whether the input has simple loops suitable for safe sewing."""
    info = {'parts': {}, 'minimum_distinct_body_loops_for_stitch': 7}
    for name in PARTS:
        mesh = meshes[name]
        simple, branched_edges = boundary_loops(mesh)
        edges, count = mesh_edges(mesh.faces)
        info['parts'][name] = {
            'faces': int(len(mesh.faces)),
            'boundary_edges': int(np.count_nonzero(count == 1)),
            'nonmanifold_edges': int(np.count_nonzero(count > 2)),
            'simple_closed_boundary_loops': int(len(simple)),
            'simple_loop_vertex_counts': [int(len(loop)) for loop in simple],
            'branched_or_open_boundary_edges': int(branched_edges),
        }
    info['ready_for_simple_loop_stitch'] = (
        info['parts']['body']['simple_closed_boundary_loops'] >= 7 and
        all(info['parts'][p]['simple_closed_boundary_loops'] >= 1 for p in PARTS[1:])
    )
    info['note'] = ('Simple-loop readiness is necessary but not sufficient: '
                    'seam distance, matching and manifold checks still apply.')
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assembly', type=Path, required=True,
                        help='0082/riggable root with riggable_parts_after_udf_cut/')
    parser.add_argument('--max-seam-gap', type=float, default=0.025)
    parser.add_argument('--max-bridge-edge', type=float, default=0.06)
    parser.add_argument('--max-perimeter-ratio', type=float, default=1.65)
    parser.add_argument('--inspect-only', action='store_true',
                        help='Write a boundary-loop diagnostic JSON without stitching')
    args = parser.parse_args()
    root = args.assembly.resolve()
    part_dir = root / 'riggable_parts_after_udf_cut'
    meshes = {name: _load_part(part_dir / f'{name}.ply') for name in PARTS}
    if args.inspect_only:
        report = preflight(meshes)
        path = root / 'riggable_after_udf_cut_STITCH_preflight.json'
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('[udf stitch 0083] preflight:', json.dumps(report, indent=2), flush=True)
        print('[udf stitch 0083] saved', path, flush=True)
        return
    result, report = stitch_parts(meshes, max_seam_gap=args.max_seam_gap,
                                  max_bridge_edge=args.max_bridge_edge,
                                  max_perimeter_ratio=args.max_perimeter_ratio)
    target = root / 'riggable_after_udf_cut_STITCHED.ply'
    report_path = root / 'riggable_after_udf_cut_STITCHED_report.json'
    if target.exists() or report_path.exists():
        raise FileExistsError('STITCHED PLY/report already exists; refusing overwrite')
    # Write only after validation. The GLB and eight individual meshes remain untouched.
    result.export(target)
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('[udf stitch 0083] saved', target, 'seams=', len(report['stitches']),
          'connected=', report['edge_connected_components_after'],
          'watertight=', report['watertight'], flush=True)


if __name__ == '__main__':
    main()
