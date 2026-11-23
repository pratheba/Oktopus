#!/usr/bin/env python3
"""Safely triangulate mixed-face binary PLY files produced by Blender.

Intended for Oktopus closed_sdf_meshes/parts_full and parts_base.
Reads the declared variable-length face lists directly, preserving all
vertices and existing triangular faces; only n-gon faces are triangulated.

Requires numpy, shapely and trimesh (already used by Oktopus).
Usage:
    python fix_closed_ply_ngons.py Pack3MDataset/trex_9_v1/closed_sdf_meshes
    python fix_closed_ply_ngons.py ... --apply

Dry-run by default. --apply makes .ply.pre_ngon_fix backups and atomically
replaces files only after Trimesh can read them and validates the result.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import os
import re
import shutil
import struct
import tempfile

import numpy as np
from shapely.geometry import Polygon, LineString, Point
from shapely.validation import explain_validity
import trimesh


def read_blender_binary_ply(path: Path):
    b = path.read_bytes()
    marker = b'end_header\n'
    k = b.find(marker)
    if k == -1:
        raise ValueError('Missing end_header newline')
    start = k + len(marker)
    header = b[:start].decode('ascii')
    lines = header.splitlines()
    if not lines or lines[0] != 'ply' or 'format binary_little_endian 1.0' not in lines:
        raise ValueError('Unsupported PLY format (expected binary_little_endian 1.0)')
    match = re.search(r'(?m)^element vertex (\d+)$', header)
    fmatch = re.search(r'(?m)^element face (\d+)$', header)
    if not match or not fmatch:
        raise ValueError('Missing vertex/face element counts')
    nv, nf = int(match.group(1)), int(fmatch.group(1))
    vert_spec = header.split('element vertex ', 1)[1].split('element face ', 1)[0].splitlines()[1:]
    face_spec = header.split('element face ', 1)[1].split('end_header', 1)[0].splitlines()[1:]
    if vert_spec != ['property float x', 'property float y', 'property float z']:
        raise ValueError(f'Unsupported vertex properties: {vert_spec}')
    if face_spec != ['property list uchar uint vertex_indices']:
        raise ValueError(f'Unsupported face properties: {face_spec}')
    vbytes = nv * 12
    if len(b) < start + vbytes:
        raise ValueError('Truncated vertex records')
    xyz = np.frombuffer(b, dtype='<f4', count=nv * 3, offset=start).reshape(-1, 3)
    offset = start + vbytes
    faces = []
    for i in range(nf):
        if offset >= len(b):
            raise ValueError(f'Truncated at face {i}')
        n = b[offset]
        end = offset + 1 + n * 4
        if n < 3 or end > len(b):
            raise ValueError(f'Invalid/truncated face {i}, vertices={n}')
        face = np.frombuffer(b, dtype='<u4', count=n, offset=offset+1).copy()
        if np.any(face >= nv):
            raise ValueError(f'Face {i} references vertex outside range')
        faces.append(face)
        offset = end
    if offset != len(b):
        raise ValueError(f'Unexpected trailing data: {len(b)-offset} bytes')
    return b[:start], b[start:start+vbytes], xyz, faces


def polygon_triangles(indices, vertices, forbidden_edges=None):
    """Constrained polygon triangulation avoiding edges already in source mesh.

    Dynamic programming finds a valid noncrossing diagonal set instead of
    greedy ear clipping, which can dead-end when certain chords are forbidden.
    Preserve original polygon winding and vertex positions.
    """
    forbidden_edges = forbidden_edges or set()
    pts3 = np.asarray(vertices[np.asarray(indices)], dtype=np.float64)
    n = len(indices)
    center = pts3.mean(axis=0)
    _, _, vh = np.linalg.svd(pts3 - center, full_matrices=False)
    pts = (pts3 - center) @ vh[:2].T
    shape = Polygon(pts)
    if not shape.is_valid or shape.area <= 1e-16:
        raise ValueError(f'Cap n-gon has invalid 2D projection: {explain_validity(shape)}')
    signed = float(np.sum(pts[:, 0]*np.roll(pts[:, 1], -1) -
                          np.roll(pts[:, 0], -1)*pts[:, 1]))
    direction = 1 if signed > 0 else -1
    span = float(np.ptp(pts, axis=0).max())
    eps_cross = max(span*span*1e-13, 1e-16)

    # Polygon boundary edges are already present in the original surface.
    # All other diagonals must stay inside the polygon AND must not reproduce
    # an already-existing edge of the surrounding triangle mesh.
    valid = [[False]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1,n):
            if j == i+1 or (i==0 and j==n-1):
                valid[i][j] = True
                continue
            e = tuple(sorted((int(indices[i]),int(indices[j]))))
            if e in forbidden_edges:
                continue
            valid[i][j] = shape.covers(LineString([tuple(pts[i]),tuple(pts[j])]))

    # dp[i][j] = vertex k joining triangle (i,k,j) between polygon-chain
    # endpoints i and j. None means impossible, -1 for a single boundary edge.
    dp = [[None]*n for _ in range(n)]
    for i in range(n-1):
        dp[i][i+1] = -1
    for width in range(2,n):
        for i in range(n-width):
            j=i+width
            if not valid[i][j]:
                continue
            for k in range(i+1,j):
                if dp[i][k] is None or dp[k][j] is None:
                    continue
                a,b,c=pts[[i,k,j]]
                area2 = float((b[0]-a[0])*(c[1]-a[1]) -
                              (b[1]-a[1])*(c[0]-a[0]))*direction
                if area2 <= eps_cross:
                    continue
                if not shape.covers(Polygon([a,b,c])):
                    continue
                dp[i][j]=k
                break
    if dp[0][n-1] is None:
        raise ValueError(f'No manifold-preserving triangulation of {n}-gon '
                         'without reusing existing mesh edges')
    tris=[]
    def collect(i,j):
        if j<=i+1: return
        k=dp[i][j]
        tris.append([int(indices[i]),int(indices[k]),int(indices[j])])
        collect(i,k)
        collect(k,j)
    collect(0,n-1)
    if len(tris)!=n-2:
        raise AssertionError('Triangulation count mismatch')
    return tris


def split_cap_chords_that_reuse_surface_edges(cap_tris, vertices, old_edges):
    """Insert Steiner points on cap diagonals coinciding with old mesh edges.

    Subdividing only the *cap side* resolves an edge-incidence artifact of
    triangulating Blender n-gons; all original triangles stay untouched.
    """
    cap_tris = [list(f) for f in cap_tris]
    vertices_new = [np.array(v, dtype=np.float32) for v in vertices]
    replaced_count = 0
    while True:
        cap_edge_uses = {}
        for fi,face in enumerate(cap_tris):
            for j in range(3):
                e = tuple(sorted((int(face[j]),int(face[(j+1)%3]))))
                cap_edge_uses.setdefault(e, []).append(fi)
        conflicts=[(e,face_ids) for e,face_ids in cap_edge_uses.items()
                   if e in old_edges and len(face_ids)==2]
        if not conflicts:
            break
        e,face_ids=conflicts[0]
        new_index=len(vertices_new)
        vertices_new.append((vertices_new[e[0]]+vertices_new[e[1]])*0.5)
        # Split both cap triangles incident to this diagonal, preserving winding.
        for fi in face_ids:
            f=cap_tris[fi]
            j=next(j for j in range(3)
                   if tuple(sorted((int(f[j]), int(f[(j+1)%3]))))==e)
            a,b,c=f[j],f[(j+1)%3],f[(j+2)%3]
            cap_tris[fi]=[a,new_index,c]
            cap_tris.append([new_index,b,c])
        replaced_count+=1
        if replaced_count>500:
            raise ValueError('More than 500 chord refinements required')
    return cap_tris, vertices_new, replaced_count


def convert(path: Path, apply: bool):
    header, vertex_bytes, vertices, faces = read_blender_binary_ply(path)
    counts = Counter(map(len, faces))
    if max(counts) == 3:
        print(f'ALREADY TRIANGLES {path}: {len(faces)} faces', flush=True)
        return
    original_edges = set()
    for face in faces:
        if len(face)==3:
            for i in range(3):
                original_edges.add(tuple(sorted((int(face[i]), int(face[(i+1)%3])))))
    all_verts = [np.array(v, dtype=np.float32) for v in vertices]
    out = []
    refinements=0
    for face in faces:
        if len(face) == 3:
            out.append([int(i) for i in face])
        else:
            # DP polygon triangulation finds a valid noncrossing partition.
            # Some cap diagonals may duplicate existing mesh edges; split them.
            tris = polygon_triangles(face, vertices, forbidden_edges=None)
            tris, newverts, num_refined = split_cap_chords_that_reuse_surface_edges(
                tris, all_verts, original_edges)
            assert len(newverts)>=len(all_verts)
            all_verts=newverts
            out.extend(tris)
            refinements+=num_refined
    new_v_count=len(all_verts)
    header2 = re.sub(rb'(?m)^element face \d+$',
                     b'element face ' + str(len(out)).encode(), header)
    header2 = re.sub(rb'(?m)^element vertex \d+$',
                     b'element vertex ' + str(new_v_count).encode(), header2)
    if header2 == header:
        raise ValueError('PLY header was not updated')
    blob = bytearray(header2)
    # Existing vertex XYZ bytes remain untouched; only extra cap vertices appended.
    blob.extend(vertex_bytes)
    for vert in all_verts[len(vertices):]:
        blob.extend(np.asarray(vert, dtype='<f4').tobytes())
    for f in out:
        blob.extend(struct.pack('<BIII', 3, *f))
    print(f'{"FIX" if apply else "DRY RUN"} {path}: '
          f'{len(faces)} source faces {dict(counts)} -> {len(out)} triangles; '
          f'added_verts={new_v_count-len(vertices)} conflicts_refined={refinements}',
          flush=True)
    # Validate geometry without writing anything for both dry-run and --apply.
    test_mesh = trimesh.Trimesh(vertices=np.asarray(all_verts),
                                faces=np.asarray(out, dtype=np.int64), process=False)
    if not (test_mesh.is_watertight and test_mesh.is_winding_consistent
            and test_mesh.is_volume):
        raise ValueError('Candidate triangulation is not a valid oriented volume: '
                         f'watertight={test_mesh.is_watertight} '
                         f'winding={test_mesh.is_winding_consistent} '
                         f'is_volume={test_mesh.is_volume}')
    if not apply:
        print('  VALID candidate: watertight=True winding=True is_volume=True',
              flush=True)
        return
    backup = path.with_suffix(path.suffix + '.pre_ngon_fix')
    if backup.exists():
        raise ValueError(f'Backup already exists: {backup}; refusing overwrite')
    fd, tempname = tempfile.mkstemp(prefix=path.stem+'-', suffix='.ply', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as file:
            file.write(blob)
        mesh = trimesh.load(tempname, process=False, force='mesh')
        if len(mesh.vertices) != new_v_count or len(mesh.faces) != len(out):
            raise ValueError('Trimesh parse returned wrong vertices or faces')
        if not (mesh.is_watertight and mesh.is_winding_consistent and mesh.is_volume):
            raise ValueError('Triangulated mesh fails volume check: '
                             f'watertight={mesh.is_watertight} '
                             f'winding={mesh.is_winding_consistent} '
                             f'is_volume={mesh.is_volume}')
        shutil.copy2(path, backup)
        os.replace(tempname, path)
        print(f'VALIDATED {path}: watertight=True winding=True is_volume=True; '
              f'backup={backup}',flush=True)
    finally:
        if os.path.exists(tempname):
            os.unlink(tempname)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('path', type=Path, help='Closed SDF folder or single PLY')
    ap.add_argument('--apply', action='store_true', help='write validated PLY and backup')
    args = ap.parse_args()
    if args.path.is_file():
        files = [args.path]
    else:
        files = sorted([*args.path.glob('parts_full/*_closed.ply'),
                        *args.path.glob('parts_base/*_closed.ply')])
    if not files:
        ap.error(f'No matching *_closed.ply files: {args.path}')
    failures = 0
    for path in files:
        try:
            convert(path, args.apply)
        except Exception as exc:
            failures+=1
            print(f'ERROR {path}: {type(exc).__name__}: {exc}',flush=True)
    if failures:
        raise SystemExit(f'{failures}/{len(files)} files failed; check errors before SDF preprocessing')
    print(f'Completed {len(files)} files (apply={args.apply}).',flush=True)

if __name__ == '__main__':
    main()
