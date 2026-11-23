#!/usr/bin/env python3
"""Rebuild Blender PLY n-gon caps from the actual triangle boundary loops.

For Oktopus CLOSED SDF part meshes only. This is *not* intended for original
open UDF meshes. Preserves all existing triangles and XYZ positions; removes
non-triangle polygon faces and reconstructs patches bounded by missing triangle
edges. When a boundary pinches at a vertex, separate vertex indices are used
for separate face fans (same XYZ), fixing non-manifold vertex topology.

Dry-run by default; --apply backs up each accepted input as
`<name>.ply.pre_boundary_cap_fix_v3` and atomically replaces it.

Requires numpy, trimesh, shapely, networkx.

V3 adds optional --fix-winding for a misoriented source triangle, and tries
additional planar projections if PCA yields a self-intersecting cap outline.
Topology/oriented-volume validation is NOT proof of no 3D self-intersection.
Inspect any repaired body/cap visually before SDF preprocessing.

Usage:
  python fix_closed_ply_boundary_caps.py Pack3MDataset/trex_9_v1/closed_sdf_meshes
  python fix_closed_ply_boundary_caps.py Pack3MDataset/trex_9_v1/closed_sdf_meshes --apply

Only meshes that pass edge manifold, vertex manifold, consistent winding,
watertightness, and positive-volume checks are written. Add
--drop-isolated-triangles only if the diagnostic identifies dangling triangles
as artifacts: it deletes just those triangles, and only if their combined area
is <=1% of the original triangle surface area. Does not certify the
absence of geometric self-intersection; inspect any complex reconstructed cap.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile

import networkx as nx
import numpy as np
import trimesh
from shapely.geometry import Polygon, LineString
from shapely.validation import explain_validity


def read_blender_ply(path):
    blob = path.read_bytes()
    marker = b'end_header\n'
    offset = blob.find(marker)
    if offset < 0:
        raise ValueError('Expected PLY end_header\\n')
    start = offset + len(marker)
    h = blob[:start].decode('ascii')
    if ('format binary_little_endian 1.0' not in h or
            not h.startswith('ply\n')):
        raise ValueError('Only binary little-endian Blender PLY supported')
    counts = []
    for label in ('vertex', 'face'):
        matches = re.findall(rf'(?m)^element {label} (\d+)$', h)
        if len(matches) != 1:
            raise ValueError(f'Expected one {label} element in header')
        counts.append(int(matches[0]))
    nv, nf = counts
    vp = h.split('element vertex ', 1)[1].split('element face ', 1)[0].splitlines()[1:]
    fp = h.split('element face ', 1)[1].split('end_header', 1)[0].splitlines()[1:]
    if vp != ['property float x', 'property float y', 'property float z']:
        raise ValueError('Only XYZ float32 vertices supported, got '+repr(vp))
    if fp != ['property list uchar uint vertex_indices']:
        raise ValueError('Only uint32 face indices supported, got '+repr(fp))
    vertex_end = start + nv * 12
    if vertex_end > len(blob):
        raise ValueError('Vertex data truncated')
    xyz = np.frombuffer(blob, dtype='<f4', count=nv*3, offset=start).reshape(-1,3).copy()
    p = vertex_end
    faces = []
    for i in range(nf):
        if p >= len(blob):
            raise ValueError(f'Face data truncated at face {i}')
        n = blob[p]
        if n < 3 or p+1+4*n > len(blob):
            raise ValueError(f'Face {i}: invalid/truncated index list length={n}')
        f = np.frombuffer(blob, dtype='<u4', count=n, offset=p+1).astype(np.int64)
        if np.any(f >= nv):
            raise ValueError(f'Face {i}: vertex index out of range')
        faces.append(f)
        p += 1+4*n
    if p != len(blob):
        raise ValueError(f'Unexpected trailing data: {len(blob)-p} bytes')
    return xyz, faces


def edge_occurrences(triangles):
    edges = defaultdict(list)
    for fi, f in enumerate(triangles):
        for a, b in zip(f, np.roll(f, -1)):
            a, b = int(a), int(b)
            edges[tuple(sorted((a,b)))].append((fi,a,b))
    return edges


def _project_candidate(p3, normal):
    """Deterministic orthonormal 2D coordinates for the selected viewing normal."""
    normal=np.asarray(normal,dtype=np.float64)
    normal=normal/np.linalg.norm(normal)
    axis=np.eye(3)[np.argmin(abs(normal))]
    u=np.cross(normal,axis)
    u=u/np.linalg.norm(u)
    v=np.cross(normal,u)
    return np.column_stack([p3@u,p3@v])


def _projection_candidates(points):
    """Try PCA first, then axes and deterministic oblique camera directions."""
    center=points.mean(axis=0)
    _,_,basis=np.linalg.svd(points-center, full_matrices=False)
    yield (points-center)@basis[:2].T, 'PCA'
    for dim in range(3):
        yield _project_candidate(points-center,np.eye(3)[dim]), f'axis-{dim}'
    # The loop may be curved so PCA can produce a crossing even when another
    # viewpoint gives a simple projected polygon. Deterministic Fibonacci sphere.
    for i in range(96):
        z=1-2*(i+0.5)/96
        theta=i*np.pi*(3-np.sqrt(5))
        radial=np.sqrt(max(0.,1-z*z))
        normal=np.array([radial*np.cos(theta),radial*np.sin(theta),z])
        yield _project_candidate(points-center,normal),f'view-{i}'


def cap_triangulation(indices, vertices):
    n=len(indices)
    if n==3:
        return [[int(v) for v in indices]]
    p3=np.asarray(vertices[np.asarray(indices)],dtype=np.float64)
    valid=[]
    invalid=[]
    for pts, label in _projection_candidates(p3):
        poly=Polygon(pts)
        if poly.is_valid and poly.area>1e-16:
            valid.append((pts,poly,label))
            # For speed and to preserve previous behavior, keep PCA if valid.
            if label=='PCA':break
        else:
            if len(invalid)<5:invalid.append(f'{label}: {explain_validity(poly)}')
    if not valid:
        raise ValueError(f'No simple planar projection of {n}-edge boundary. '
                         f'Tried PCA/axes/96 other directions. Examples: {invalid}. '
                         'Needs Blender boundary cleanup; refusing unsafe cap.')
    # Try triangulations for candidates. A valid polygon may not have a valid
    # triangulation if the polygon is pinched or has tiny edges.
    errors=[]
    for p2,polygon,label in valid:
        direction=1 if sum(p2[i,0]*p2[(i+1)%n,1]-p2[(i+1)%n,0]*p2[i,1] for i in range(n))>0 else -1
        scale=float(np.ptp(p2,axis=0).max())
        min_area=max(scale*scale*1e-13,1e-16)
        possible=[[False]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1,n):
                possible[i][j]= j==i+1 or (i==0 and j==n-1) or polygon.covers(LineString([p2[i],p2[j]]))
        dp=[[None]*n for _ in range(n)]
        for i in range(n-1):dp[i][i+1]=-1
        for size in range(2,n):
            for i in range(n-size):
                j=i+size
                if not possible[i][j]:continue
                for k in range(i+1,j):
                    if dp[i][k] is None or dp[k][j] is None:continue
                    a,b,c=p2[[i,k,j]]
                    signed_cross=((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))*direction
                    if signed_cross<=min_area:continue
                    if not polygon.covers(Polygon([a,b,c])):continue
                    dp[i][j]=k
                    break
        if dp[0][n-1] is None:
            errors.append(f'{label}: no acceptable triangulation')
            continue
        result=[]
        def descend(i,j):
            if j<=i+1:return
            k=dp[i][j]
            result.append([int(indices[i]),int(indices[k]),int(indices[j])])
            descend(i,k);descend(k,j)
        descend(0,n-1)
        if len(result)!=n-2:raise RuntimeError('Internal triangulation mismatch')
        if label!='PCA':print(f'  [alternate projection] used {label} for {n}-vertex cap',flush=True)
        return result
    raise ValueError(f'Tried valid projections but unable to triangulate {n}-vertex boundary: {errors[:6]}')


def orient_triangles_by_adjacency(triangles):
    """Orient a triangle mesh by parity constraints, without changing coordinates.

    Only flips existing face indices. Refuses nonorientable inconsistencies.
    Returns (oriented triangles, number flipped). Finds minimum flips within
    each connected component; volume validation later ensures a useful result.
    """
    triangles=[list(map(int,t)) for t in triangles]
    edges=edge_occurrences(triangles)
    adj=defaultdict(list)
    for edge,occ in edges.items():
        if len(occ)>2:
            raise ValueError(f'Edge {edge} has >2 incident original triangles')
        if len(occ)==2:
            (fi,a,b),(fj,c,d)=occ
            parity=int(a==c and b==d)
            adj[fi].append((fj,parity));adj[fj].append((fi,parity))
    flip={}
    for start in range(len(triangles)):
        if start in flip:continue
        flip[start]=0
        component=[]
        todo=[start]
        while todo:
            fi=todo.pop();component.append(fi)
            for fj,parity in adj[fi]:
                expected=flip[fi]^parity
                if fj in flip:
                    if flip[fj]!=expected:
                        raise ValueError('Inconsistent face orientation constraints; '
                                         'source mesh cannot be fixed by face flips alone')
                else:
                    flip[fj]=expected
                    todo.append(fj)
        if sum(flip[fi] for fi in component)>len(component)/2:
            for fi in component:flip[fi]^=1
    n_flipped=sum(flip.values())
    for fi,flag in flip.items():
        if flag:triangles[fi][1],triangles[fi][2]=triangles[fi][2],triangles[fi][1]
    return triangles,n_flipped


def avoid_surface_chord_conflicts(cap_faces, new_verts, source_edges):
    cap_faces=[list(f) for f in cap_faces]
    splits=0
    while True:
        cap_e=defaultdict(list)
        for fi,f in enumerate(cap_faces):
            for j in range(3):
                cap_e[tuple(sorted((int(f[j]),int(f[(j+1)%3]))))].append((fi,j))
        collision=next(((e,items) for e,items in cap_e.items()
                        if e in source_edges and len(items)==2),None)
        if collision is None: return cap_faces,new_verts,splits
        (a,b),items=collision
        newidx=len(new_verts)
        new_verts.append((new_verts[a]+new_verts[b])*np.float32(0.5))
        for fi,j in items:
            f=cap_faces[fi]
            aa,bb,cc=f[j],f[(j+1)%3],f[(j+2)%3]
            cap_faces[fi]=[aa,newidx,cc]
            cap_faces.append([newidx,bb,cc])
        splits+=1
        if splits>500: raise ValueError('Too many cap chord conflicts')


def split_pinch_vertices(triangles, vertices):
    """Separate disconnected edge-adjacent fans at a shared vertex."""
    farray=np.asarray(triangles,dtype=np.int64).copy()
    verts=[np.asarray(v,dtype=np.float32).copy() for v in vertices]
    incident=defaultdict(list)
    for fi,f in enumerate(farray):
        for v in f:
            incident[int(v)].append(fi)
    splits=[]
    for v, face_ids in list(incident.items()):
        if len(face_ids)<=1: continue
        neighbor_to_faces=defaultdict(list)
        for fi in face_ids:
            for u in farray[fi]:
                if u!=v: neighbor_to_faces[int(u)].append(fi)
        adjacency={fi:set() for fi in face_ids}
        for group in neighbor_to_faces.values():
            for fi in group:
                adjacency[fi].update(group)
        visited=set();comps=[]
        for fi in face_ids:
            if fi in visited:continue
            todo=[fi];component=[]
            while todo:
                x=todo.pop()
                if x in visited:continue
                visited.add(x);component.append(x)
                todo.extend(adjacency[x]-visited)
            comps.append(component)
        if len(comps)>1:
            for component in comps[1:]:
                fresh=len(verts)
                verts.append(verts[v].copy())
                for fi in component:
                    farray[fi,farray[fi]==v]=fresh
            splits.append((v,len(comps)))
    return farray, np.asarray(verts,dtype=np.float32),splits


def vertex_fan_count(triangles):
    incident=defaultdict(list)
    for fi, f in enumerate(triangles):
        for v in f:incident[int(v)].append(fi)
    bad=[]
    for v, face_ids in incident.items():
        neighbor_to_faces=defaultdict(list)
        for fi in face_ids:
            for u in triangles[fi]:
                if u!=v:neighbor_to_faces[int(u)].append(fi)
        adjacency={fi:set() for fi in face_ids}
        for fids in neighbor_to_faces.values():
            for fi in fids:adjacency[fi].update(fids)
        seen=set();components=0
        for fi in face_ids:
            if fi in seen:continue
            components+=1;todo=[fi]
            while todo:
                x=todo.pop()
                if x in seen:continue
                seen.add(x);todo.extend(adjacency[x]-seen)
        if components>1:bad.append((v,components))
    return bad


def repair(path, drop_isolated_triangles=False, fix_winding=False):
    original_vertices, all_faces=read_blender_ply(path)
    n_gon=Counter(map(len,all_faces))
    original_tris=[list(map(int,f)) for f in all_faces if len(f)==3]
    if not original_tris:
        raise ValueError('No original triangle faces; cannot infer original boundary')
    original_edges=edge_occurrences(original_tris)
    # Fully isolated surface triangles (three exposed edges) can appear when
    # Blender fills a figure-eight boundary with one inappropriate n-gon.
    # These are not valid parts of a closed surface; removal is opt-in.
    isolated=[]
    for fi,f in enumerate(original_tris):
        if all(len(original_edges[tuple(sorted((int(a),int(b))))]) == 1
               for a,b in zip(f, np.roll(f,-1))):
            isolated.append(fi)
    dropped=[]
    drop_area_fraction=0.0
    if isolated:
        old_vertices=original_vertices[np.asarray(original_tris,dtype=np.int64)]
        all_areas=0.5*np.linalg.norm(np.cross(old_vertices[:,1]-old_vertices[:,0],
                                                 old_vertices[:,2]-old_vertices[:,0]),axis=1)
        drop_area_fraction=float(all_areas[isolated].sum()/max(all_areas.sum(),1e-30))
        if not drop_isolated_triangles:
            raise ValueError(f'{len(isolated)} dangling triangles have all 3 edges exposed '
                             f'(face IDs within triangle-only list: {isolated[:20]}, '
                             f'area fraction={drop_area_fraction:.6%}); '
                             'rerun with --drop-isolated-triangles only if these are '
                             'artifacts of Blender capping')
        # Require removal be a tiny fraction of original geometry.
        if drop_area_fraction > 0.01:
            raise ValueError(f'Dangling triangle area is {drop_area_fraction:.3%} '
                             '(>1% of source); refusing automated deletion')
        dropped=isolated
        drop_set=set(isolated)
        original_tris=[f for fi,f in enumerate(original_tris) if fi not in drop_set]
        original_edges=edge_occurrences(original_tris)
    overused=[e for e,uses in original_edges.items() if len(uses)>2]
    if overused:
        raise ValueError(f'{len(overused)} original triangle edges have >2 incident faces; manual repair needed')
    winding_flips=0
    if fix_winding:
        original_tris,winding_flips=orient_triangles_by_adjacency(original_tris)
        original_edges=edge_occurrences(original_tris)
    # Reuse unchanged triangulated meshes if they are already truly valid.
    if len(all_faces)==len(original_tris):
        mesh=trimesh.Trimesh(vertices=original_vertices,faces=original_tris,process=False)
        if mesh.is_watertight and mesh.is_winding_consistent and mesh.is_volume and not vertex_fan_count(np.array(original_tris)):
            return None,{'status':'ALREADY VALID', 'faces':len(original_tris)}
    # Source faces need consistent winding for a reliable cap orientation.
    incorrectly_wound=[]
    for e,occ in original_edges.items():
        if len(occ)==2 and (occ[0][1],occ[0][2]) == (occ[1][1],occ[1][2]):
            incorrectly_wound.append(e)
    if incorrectly_wound:
        raise ValueError(f'{len(incorrectly_wound)} triangle edges have inconsistent winding; manual repair needed')
    boundary={e:occ[0] for e,occ in original_edges.items() if len(occ)==1}
    if not boundary:
        raise ValueError('No boundary edges to rebuild; invalid geometry needs manual inspection')
    graph=nx.Graph()
    graph.add_edges_from(boundary)
    degrees=Counter(d for _,d in graph.degree())
    if any(d%2 for _,d in graph.degree()):
        raise ValueError('Original triangle boundary has odd-degree vertices; cannot partition into closed cycles')
    cycles=nx.cycle_basis(graph)
    covered=Counter()
    for c in cycles:
        if len(c)<3:raise ValueError('Boundary cycle shorter than 3 vertices')
        for a,b in zip(c,c[1:]+c[:1]):
            covered[tuple(sorted((a,b)))] += 1
    if set(covered)!=set(boundary) or any(count!=1 for count in covered.values()):
        raise ValueError('Boundary graph has interleaved cycles; automated capping unsafe')
    generated=[]
    added_verts=[v.copy() for v in original_vertices]
    chord_splits=0
    for c in cycles:
        directions=[]
        for a,b in zip(c,c[1:]+c[:1]):
            _,ra,rb=boundary[tuple(sorted((a,b)))]
            directions.append(+1 if (a,b)==(ra,rb) else -1)
        if len(set(directions))!=1:
            raise ValueError('Boundary cycle has inconsistent edge orientations')
        # Generated cap must use opposite winding from triangles on the boundary.
        if directions[0]==1:
            c=list(reversed(c))
        newtris=cap_triangulation(c,original_vertices)
        newtris,added_verts,n=avoid_surface_chord_conflicts(newtris,added_verts,set(original_edges))
        chord_splits+=n
        generated.extend(newtris)
    combined=np.asarray(original_tris+generated,dtype=np.int64)
    combined,verts,pinch_splits=split_pinch_vertices(combined,added_verts)
    if len(set(map(tuple,np.sort(combined,axis=1))))!=len(combined):
        raise ValueError('Duplicate triangles after repair')
    ed=edge_occurrences(combined)
    incidence=Counter(map(len,ed.values()))
    badfans=vertex_fan_count(combined)
    mesh=trimesh.Trimesh(vertices=verts,faces=combined,process=False)
    metrics={'status':'REPAIRED CANDIDATE', 'input_faces':len(all_faces), 'winding_flips':winding_flips,
             'original_face_sizes':dict(n_gon),'dropped_dangling_triangles':len(dropped),
             'dropped_area_fraction':drop_area_fraction,'input_boundary_edges':len(boundary),
             'boundary_cycles':list(map(len,cycles)),
             'generated_cap_triangles':len(generated),'chord_splits':chord_splits,
             'pinch_vertex_splits':pinch_splits,
             'output_vertices':len(verts),'output_faces':len(combined),
             'edge_incidence':dict(incidence),
             'watertight':bool(mesh.is_watertight),
             'winding_consistent':bool(mesh.is_winding_consistent),
             'is_volume':bool(mesh.is_volume),
             'nonmanifold_vertex_fans':badfans}
    if (incidence!={2:len(ed)} or badfans or not mesh.is_watertight or
        not mesh.is_winding_consistent or not mesh.is_volume):
        raise ValueError(f'Candidate failed topology/volume validation: {metrics}')
    return (verts,combined),metrics


def write_binary_ply(dest, verts, faces):
    header=(f'ply\nformat binary_little_endian 1.0\n'
            f'comment Rebuilt artificial n-gon caps from source triangle boundaries\n'
            f'element vertex {len(verts)}\n'
            'property float x\nproperty float y\nproperty float z\n'
            f'element face {len(faces)}\n'
            'property list uchar uint vertex_indices\nend_header\n')
    with open(dest,'wb') as f:
        f.write(header.encode('ascii'))
        f.write(np.asarray(verts,dtype='<f4').tobytes())
        for tri in faces:
            f.write(struct.pack('<BIII',3,*[int(x) for x in tri]))


def process(path,apply,drop_isolated_triangles,fix_winding):
    candidate,metrics=repair(path,drop_isolated_triangles=drop_isolated_triangles,fix_winding=fix_winding)
    print(f'{path}: {metrics}',flush=True)
    if candidate is None or not apply: return
    backup=path.with_suffix(path.suffix+'.pre_boundary_cap_fix_v3')
    if backup.exists():
        raise ValueError(f'Backup already exists, refusing overwrite: {backup}')
    fd,temp=tempfile.mkstemp(prefix=path.stem+'-',suffix='.ply',dir=str(path.parent))
    try:
        os.close(fd)
        write_binary_ply(temp,*candidate)
        check=trimesh.load(temp,force='mesh',process=False)
        if (len(check.vertices)!=len(candidate[0]) or
            len(check.faces)!=len(candidate[1]) or
            not check.is_watertight or not check.is_winding_consistent or not check.is_volume):
            raise ValueError('Written PLY failed Trimesh reload/validation')
        shutil.copy2(path,backup)
        os.replace(temp,path)
        print(f'  REPLACED {path}; backup={backup}',flush=True)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('target',type=Path,help='One PLY or the closed_sdf_meshes folder')
    ap.add_argument('--apply',action='store_true',help='write validated repairs with backups')
    ap.add_argument('--drop-isolated-triangles',action='store_true',
                    help='explicitly remove dangling triangles with all 3 edges exposed (max 1%% of area)')
    ap.add_argument('--fix-winding',action='store_true', help='reorient existing triangle faces using adjacency constraints')
    a=ap.parse_args()
    if a.target.is_file():files=[a.target]
    else:files=sorted([*a.target.glob('parts_full/*_closed.ply'),
                       *a.target.glob('parts_base/*_closed.ply')])
    if not files:ap.error('No matching PLY files')
    errors=[]
    for p in files:
        try:process(p,a.apply,a.drop_isolated_triangles,a.fix_winding)
        except Exception as e:
            errors.append((str(p),str(e)))
            print(f'FAILED: {p}: {type(e).__name__}: {e}',flush=True)
    print(f'Completed {len(files)} files; failures={len(errors)}; apply={a.apply}',flush=True)
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
