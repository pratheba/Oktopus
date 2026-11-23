#!/usr/bin/env python3
"""Targeted *closed SDF* T-rex back-right-leg repair. Not for open UDF meshes.

Repairs a particular Blender PLY pathology: one face-orientation contradiction
between adjacent triangles, and a non-manifold vertex fan near the artificial
49-edge n-gon cap. For the validated case, remove only the smaller conflicting
triangle and reconstruct that triangle with correct orientation together with
all boundary caps after splitting disconnected vertex fans.

Uses triangulation/topology helpers from fix_closed_ply_boundary_caps_v3.py,
which must be alongside this file. Dry-run by default; --apply creates a
backup and atomically replaces ONLY files passing all checks.

Requirements: numpy, networkx, shapely, trimesh.

Usage (from Pack3MDataset, with both scripts in the current folder):
  python fix_trex_back_r_leg.py trex_9_v1/closed_sdf_meshes
  python fix_trex_back_r_leg.py trex_9_v1/closed_sdf_meshes --apply

This is a topological repair, not proof of absent 3D self-intersections.
Inspect repaired caps in Blender before costly SDF preprocessing.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from collections import Counter

import numpy as np
import networkx as nx
import trimesh

from fix_closed_ply_boundary_caps_v3 import (
    read_blender_ply, edge_occurrences, split_pinch_vertices,
    vertex_fan_count, cap_triangulation, avoid_surface_chord_conflicts,
    write_binary_ply,
)


def repair_closed_leg(path: Path):
    verts, all_faces = read_blender_ply(path)
    triangles = np.asarray([f for f in all_faces if len(f) == 3], dtype=np.int64)
    polygons = [f for f in all_faces if len(f) != 3]

    # If already fixed, report without further modification.
    if len(polygons) == 0:
        current = trimesh.Trimesh(vertices=verts, faces=triangles, process=False)
        if (current.is_watertight and current.is_winding_consistent
                and current.is_volume and not vertex_fan_count(triangles)):
            return None, {"status": "ALREADY VALID", "faces": len(triangles)}

    if len(polygons) != 1 or len(polygons[0]) != 49:
        raise ValueError(f"Expected one 49-sided artificial cap; found {[len(p) for p in polygons]}. Refusing automatic changes.")

    occurrences = edge_occurrences(triangles)
    overused = [e for e, occ in occurrences.items() if len(occ) > 2]
    if overused:
        raise ValueError(f"Found {len(overused)} edges with >2 triangle faces; refusing repair")

    conflicts = [(e, occ) for e, occ in occurrences.items()
                 if len(occ) == 2 and occ[0][1:] == occ[1][1:]]
    if len(conflicts) != 1:
        raise ValueError(f"Expected exactly one same-direction shared edge, found {len(conflicts)}. Refusing repair")

    edge, occ = conflicts[0]
    if edge != (15, 17):
        raise ValueError(f"This targeted repair is verified for conflict edge (15,17), found {edge}; refusing")

    fids = [occ[0][0], occ[1][0]]
    def face_area(fid):
        p = verts[triangles[fid]].astype(np.float64)
        return 0.5 * float(np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0])))
    loser = min(fids, key=face_area)
    # Avoid blindly deleting a real piece of geometry.
    total_area = float(trimesh.Trimesh(vertices=verts, faces=triangles, process=False).area)
    fraction = face_area(loser) / max(total_area, 1e-30)
    if fraction > 1e-3:
        raise ValueError(f"Conflicting triangle is {fraction:.3%} of surface (>0.1%); refusing")
    original_loser = triangles[loser].tolist()

    # Dropping the single sliver makes the original surface orientable.
    triangles = np.delete(triangles, loser, axis=0)
    oe = edge_occurrences(triangles)
    still_conflicting = [e for e, o in oe.items() if len(o) == 2 and o[0][1:] == o[1][1:]]
    if still_conflicting:
        raise ValueError(f"Still {len(still_conflicting)} inconsistent triangle edges after local surgery")

    # Separate disconnected fans *before* finding boundary loops. Otherwise
    # the 3 locally pinched vertices create ambiguous 4-valent boundary nodes.
    triangles, split_verts, pinches = split_pinch_vertices(triangles, verts)
    oe = edge_occurrences(triangles)
    boundary = {e: o[0] for e, o in oe.items() if len(o) == 1}
    graph = nx.Graph()
    graph.add_edges_from(boundary)
    if not boundary or any(d != 2 for _, d in graph.degree()):
        raise ValueError("Boundary is not a union of simple cycles after vertex-fan splitting")
    cycles = nx.cycle_basis(graph)
    if sum(len(c) for c in cycles) != len(boundary):
        raise ValueError("Some boundary edges were not covered by simple cycles")

    newverts = [p.copy() for p in split_verts]
    caps = []
    chord_splits = 0
    for cycle in cycles:
        directions = []
        for a, b in zip(cycle, cycle[1:] + cycle[:1]):
            _, ra, rb = boundary[tuple(sorted((a, b)))]
            directions.append(1 if (a, b) == (ra, rb) else -1)
        if len(set(directions)) != 1:
            raise ValueError("Boundary loop has inconsistent edge directions")
        if directions[0] == 1:
            cycle = list(reversed(cycle))
        candidate = cap_triangulation(cycle, split_verts)
        candidate, newverts, nsplits = avoid_surface_chord_conflicts(
            candidate, newverts, set(oe))
        chord_splits += nsplits
        caps.extend(candidate)

    joined = np.asarray([*triangles, *caps], dtype=np.int64)
    joined, newverts, post_split = split_pinch_vertices(joined, newverts)
    mesh = trimesh.Trimesh(vertices=newverts, faces=joined, process=False)
    incidences = Counter(map(len, edge_occurrences(joined).values()))
    bad_fans = vertex_fan_count(joined)
    components = len(mesh.split(only_watertight=False))
    metrics = {
        "status": "REPAIRED CANDIDATE",
        "conflicting_edge": edge,
        "conflicting_triangle_ids": fids,
        "replaced_triangle_id": loser,
        "replaced_triangle_vertices": original_loser,
        "replaced_triangle_area_fraction": fraction,
        "initial_vertices": len(verts),
        "initial_triangle_faces": len(triangles) + 1,
        "initial_ngon_edges": len(polygons[0]),
        "split_vertex_fans": pinches,
        "boundary_cycles_after_split": sorted(map(len, cycles)),
        "generated_cap_triangles": len(caps),
        "cap_chord_splits": chord_splits,
        "post_cap_pinch_splits": post_split,
        "output_vertices": len(newverts),
        "output_faces": len(joined),
        "edge_incidences": dict(incidences),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "is_volume": bool(mesh.is_volume),
        "components": components,
        "nonmanifold_vertex_fans": bad_fans,
        "volume": float(mesh.volume),
    }
    if (incidences != {2: len(edge_occurrences(joined))}
            or not mesh.is_watertight or not mesh.is_winding_consistent
            or not mesh.is_volume or bad_fans or components != 1):
        raise ValueError(f"Candidate failed manifold/orientation/volume checks: {metrics}")
    return (newverts, joined), metrics


def handle(path: Path, apply: bool):
    candidate, metrics = repair_closed_leg(path)
    print(f"{path}: {metrics}", flush=True)
    if candidate is None or not apply:
        return
    backup = path.with_name(path.name + '.pre_leg_topology_fix')
    if backup.exists():
        raise ValueError(f"Backup already exists: {backup}; refusing overwrite")
    fd, temp = tempfile.mkstemp(prefix=path.stem + '-', suffix='.ply', dir=path.parent)
    try:
        os.close(fd)
        write_binary_ply(temp, *candidate)
        check = trimesh.load(temp, force='mesh', process=False)
        if (len(check.vertices) != len(candidate[0])
                or len(check.faces) != len(candidate[1])
                or not check.is_watertight or not check.is_winding_consistent
                or not check.is_volume or vertex_fan_count(check.faces)):
            raise ValueError('Written PLY failed readback verification')
        shutil.copy2(path, backup)
        os.replace(temp, path)
        print(f'  REPLACED {path}; backup={backup}', flush=True)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('path', type=Path, help='closed_sdf_meshes directory or one CLOSED leg PLY')
    p.add_argument('--apply', action='store_true', help='write valid candidates, with .pre_leg_topology_fix backups')
    a = p.parse_args()
    if a.path.is_file():
        paths = [a.path]
    elif a.path.is_dir():
        paths = [a.path / sub / '003_back_r_leg_closed.ply'
                 for sub in ('parts_full', 'parts_base')]
    else:
        p.error('Path does not exist')
    failures = []
    for path in paths:
        try:
            handle(path, a.apply)
        except Exception as exc:
            print(f'FAILED: {path}: {type(exc).__name__}: {exc}', flush=True)
            failures.append(path)
    print(f'Checked {len(paths)} files; failures={len(failures)}; apply={a.apply}', flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
