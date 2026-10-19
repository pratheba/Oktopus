#!/usr/bin/env python3
"""0051: Offline single-curve pose sequence for Oktopus adapted meshes.

No network/SDF inference, no skinning engine, no scene rigging: deform the
rest-pose vertices using a smooth bend of ONE existing target skeleton curve.

Mesh vertices are bound once via nearest projection to the curve. A deformed
curve is constructed by integrating rotated rest-curve tangents; each vertex
keeps its rest offset from its binding point, rotated by the same local bend.
Face connectivity is unchanged. Manifoldness/collision-free motion is NOT
proved, especially at independent root/leaf attachments.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh


def load_curve_npz(path, curve_name):
    """Load curve keypoints directly from the standard Oktopus handle NPZ.

    Uses `segments` dictionaries as in ngc.handle_3dvec.Handle.load; unlike
    that loader, does not build huge cylinders or load torch checkpoints.
    """
    with np.load(path, allow_pickle=True) as data:
        if 'segments' not in data:
            raise ValueError('Expected std_handle.npz with `segments`')
        entries = data['segments']
        names = [str(e.get('name')) for e in entries]
        matches = [e for e in entries if str(e.get('name')) == curve_name]
        if len(matches) != 1:
            raise ValueError(f'Curve {curve_name!r} not uniquely found: available={names}')
        points = np.array(matches[0]['keypoints'], dtype=np.float64, copy=True)
        body = next((np.array(e['keypoints'], dtype=np.float64, copy=True)
                     for e in entries if str(e.get('name')) == 'body'), None)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError('Invalid curve keypoints')
    if not np.isfinite(points).all():
        raise ValueError('Curve keypoints are nonfinite')
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if lengths.sum() < 1e-8:
        raise ValueError('Degenerate target curve')
    if np.any(lengths < 1e-10):
        raise ValueError('Repeated keypoints in target curve; clean skeleton')
    return points, body


def orient_from_root(points, body_points, root_end):
    """Curve parameter goes from attachment (root) to free tip."""
    if root_end == 'auto':
        if body_points is None:
            raise ValueError('`--root-end auto` requires body curve in handle; set start/end explicitly')
        tree = cKDTree(body_points)
        distances = tree.query(points[[0, -1]])[0]
        if abs(distances[0] - distances[1]) < 1e-6:
            raise ValueError('Ambiguous root endpoint; set --root-end start/end')
        root_end = 'start' if distances[0] < distances[1] else 'end'
    if root_end not in ('start', 'end'):
        raise ValueError('root_end must be start, end, auto')
    return (points.copy() if root_end == 'start' else points[::-1].copy()), root_end


def sample_curve(keypoints, count=512):
    """Resample at normalized arc length, consistent with PWLA key_ts."""
    if count < 16:
        raise ValueError('At least 16 curve samples required')
    distances = np.linalg.norm(np.diff(keypoints, axis=0), axis=1)
    ts = np.r_[0., np.cumsum(distances)]
    ts /= ts[-1]
    u = np.linspace(0., 1., count)
    pts = np.stack([np.interp(u, ts, keypoints[:, j]) for j in range(3)], axis=-1)
    return u, pts


def project_mesh_vertices(vertices, curve_samples, u, batch=200000):
    """Bind vertices to rest curve using the nearest sampled centerline edge.

    Each vertex tests the two edges adjacent to the nearest curve sample. This
    is an efficient approximation to Oktopus's piecewise curve projection.
    """
    v = np.asarray(vertices, dtype=np.float64)
    tree = cKDTree(curve_samples)
    s_all = np.empty(len(v), dtype=np.float64)
    anchor = np.empty((len(v), 3), dtype=np.float64)
    n = len(curve_samples)
    for st in range(0, len(v), batch):
        chunk = v[st:st+batch]
        _, nearest = tree.query(chunk, k=1)
        candidates = np.stack([np.clip(nearest-1, 0, n-2),
                               np.clip(nearest, 0, n-2)], axis=1)
        a = curve_samples[candidates]
        d = curve_samples[candidates+1] - a
        t = np.clip(np.sum((chunk[:, None, :] - a)*d, axis=2) /
                    np.maximum(np.sum(d*d, axis=2), 1e-20), 0, 1)
        foot = a + t[..., None]*d
        err = np.sum((chunk[:, None, :] - foot)**2, axis=2)
        which = np.argmin(err, axis=1)
        inds = np.arange(len(chunk))
        k = candidates[inds, which]
        alpha = t[inds, which]
        anchor[st:st+len(chunk)] = foot[inds, which]
        s_all[st:st+len(chunk)] = u[k]*(1-alpha)+u[k+1]*alpha
    return s_all, v-anchor


def smooth_bend_profile(s, hold=0.12, full=0.85):
    if not (0.0 <= hold < full <= 1.0):
        raise ValueError('Require 0 <= hold < full <= 1')
    u = np.clip((np.asarray(s)-hold)/(full-hold), 0., 1.)
    return u*u*(3-2*u)


def bend_positions(rest_curve, s_vertex, offsets, *, angle_deg, bend_axis,
                   root_hold=0.12, full_bend=0.85):
    """Arc-length-preserving centerline integration and attached offset motion."""
    count = len(rest_curve)
    u = np.linspace(0, 1, count)
    axis = np.asarray(bend_axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    profile = smooth_bend_profile(u, root_hold, full_bend)
    mid_profile = smooth_bend_profile((u[:-1]+u[1:])/2, root_hold, full_bend)
    rot_mid = Rotation.from_rotvec(np.deg2rad(angle_deg)*mid_profile[:, None]*axis)
    increments = rot_mid.apply(np.diff(rest_curve, axis=0))
    posed = np.vstack([rest_curve[0], rest_curve[0]+np.cumsum(increments,axis=0)])
    center = np.stack([np.interp(s_vertex, u, posed[:, j]) for j in range(3)],axis=1)
    rotate_v = Rotation.from_rotvec(np.deg2rad(angle_deg)*
                                    smooth_bend_profile(s_vertex, root_hold, full_bend)[:, None]*axis)
    return center + rotate_v.apply(offsets), posed


def select_bend_axis(curve_samples, name):
    tan = curve_samples[min(4, len(curve_samples)-1)]-curve_samples[0]
    tan /= np.linalg.norm(tan)
    if name == 'x': ref = np.array([1., 0., 0.])
    elif name == 'y': ref = np.array([0., 1., 0.])
    elif name == 'z': ref = np.array([0., 0., 1.])
    else: raise ValueError('axis must be x, y, z')
    # Project chosen world-space direction normal to the root tangent: this
    # yields bending, rather than arbitrary roll about a tentacle tangent.
    axis = ref - tan*np.dot(ref,tan)
    norm = np.linalg.norm(axis)
    if norm < 1e-5:
        raise ValueError(f'Bend axis {name} parallel to root tangent; select a different axis')
    return axis/norm


def parse_mesh_args(items):
    parsed = {}
    for item in items:
        if '=' not in item:
            raise ValueError(f'Expected NAME=PATH, got {item!r}')
        name, value = item.split('=', 1)
        if not name or not value or name in parsed or '/' in name or '\\' in name:
            raise ValueError(f'Invalid/duplicate mesh name {name!r}')
        parsed[name] = Path(value)
    return parsed


def load_mesh(path):
    obj = trimesh.load(path, process=False)
    if isinstance(obj, trimesh.Scene):
        if len(obj.geometry) != 1:
            raise ValueError('Provide individual PLY meshes, not multi-part scenes')
        obj = obj.to_geometry()
    if not isinstance(obj, trimesh.Trimesh) or not len(obj.faces):
        raise ValueError(f'No triangle mesh in {path}')
    return obj


def mesh_quality(rest, pose):
    # Degenerate and severe edge stretch diagnostics; no collision certification.
    triangles = pose.vertices[pose.faces]
    area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                   triangles[:, 2]-triangles[:, 0]), axis=1)/2
    edges = np.asarray(pose.edges_unique, dtype=np.int64)
    a = np.linalg.norm(rest.vertices[edges[:, 0]]-rest.vertices[edges[:, 1]],axis=1)
    b = np.linalg.norm(pose.vertices[edges[:, 0]]-pose.vertices[edges[:, 1]],axis=1)
    valid = a > 1e-12
    stretch = (b[valid]/a[valid]) if valid.any() else np.array([1.0])
    return {'faces': int(len(pose.faces)), 'vertices': int(len(pose.vertices)),
            'tiny_triangle_count': int(np.count_nonzero(area < 1e-12)),
            'edge_stretch_p01': float(np.percentile(stretch, 1)),
            'edge_stretch_p99': float(np.percentile(stretch, 99)),
            'edge_stretch_max': float(np.max(stretch)),
            'input_watertight': bool(rest.is_watertight),
            'posed_watertight_topological': bool(pose.is_watertight)}


def export_sequence(args):
    curves, body = load_curve_npz(args.handle_npz, args.curve)
    curves, root = orient_from_root(curves, body, args.root_end)
    params, centers = sample_curve(curves, args.curve_samples)
    axis = select_bend_axis(centers, args.axis)
    moving_paths = parse_mesh_args(args.moving)
    static_paths = parse_mesh_args(args.static)
    if not moving_paths:
        raise ValueError('Pass at least one --moving NAME=PATH')
    if set(moving_paths) & set(static_paths):
        raise ValueError('Moving/static mesh names overlap')
    meshes = {name: load_mesh(p) for name,p in moving_paths.items()}
    static = {name: load_mesh(p) for name,p in static_paths.items()}
    bindings = {name: project_mesh_vertices(mesh.vertices, centers, params)
                for name,mesh in meshes.items()}
    out = Path(args.output);out.mkdir(parents=True, exist_ok=True)
    if args.frames < 3:
        raise ValueError('frames must be >= 3')
    report={'curve':args.curve,'handle_npz':str(args.handle_npz),
            'root_end':root,'axis':axis.tolist(),'angle_deg':args.angle_deg,
            'frames':args.frames,'moving':list(meshes),'static':list(static),
            'backend':'rest_vertex_curve_binding; no SDF re-evaluation',
            'connected_surface_validated':False,
            'collision_free_validated':False,'poses':[]}
    for i, phase in enumerate(np.linspace(0, np.pi, args.frames)):
        angle = float(args.angle_deg*np.sin(phase))
        scene=trimesh.Scene()
        for name,mesh in static.items():
            scene.add_geometry(mesh,geom_name=name,node_name=name)
        quality={}
        posed_skel = None
        for name, original in meshes.items():
            s, offsets = bindings[name]
            moved_points, posed_skel = bend_positions(centers, s, offsets,
                angle_deg=angle, bend_axis=axis, root_hold=args.root_hold,
                full_bend=args.full_bend)
            moved=original.copy()
            moved.vertices=moved_points
            # preserve connectivity; no process/weld/automatic fixes
            moved.export(out/f'{name}_{i:03d}.ply')
            scene.add_geometry(moved,geom_name=name,node_name=name)
            quality[name]=mesh_quality(original,moved)
        scene.export(out/f'pose_{i:03d}.glb')
        np.save(out/f'skeleton_{args.curve}_{i:03d}.npy',posed_skel)
        report['poses'].append({'frame':i,'angle_deg':angle,'quality':quality})
        print(f'[offline_pose] frame={i:03d} angle={angle:+.2f} deg saved', flush=True)
    (out/'pose_manifest.json').write_text(json.dumps(report,indent=2))
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--handle-npz',type=Path,required=True,
                   help='Avatar DATA_ROOT/oktopus_9_v1/handle/std_handle.npz')
    p.add_argument('--curve',default='t7',help='Select ONE target curve: t7 boots or t1 puffer')
    p.add_argument('--moving',action='append',default=[],metavar='NAME=PLY',
                   help='Mesh that follows selected curve; may be repeated')
    p.add_argument('--static',action='append',default=[],metavar='NAME=PLY',
                   help='Unchanged context mesh; may be repeated')
    p.add_argument('--output',type=Path,default=Path('inference/offline_motion_512'))
    p.add_argument('--frames',type=int,default=9)
    p.add_argument('--angle-deg',type=float,default=30)
    p.add_argument('--axis',choices=['x','y','z'],default='z')
    p.add_argument('--root-end',choices=['auto','start','end'],default='auto')
    p.add_argument('--root-hold',type=float,default=0.12)
    p.add_argument('--full-bend',type=float,default=0.85)
    p.add_argument('--curve-samples',type=int,default=512)
    args=p.parse_args(argv)
    export_sequence(args)


if __name__=='__main__':
    main()
