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



def _unit_rows(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(lengths < 1e-12):
        raise ValueError('Degenerate sampled curve tangent')
    return vectors / lengths


def _minimal_rotation_vectors(directions, target):
    """Shortest rotations that align each unit tangent with the target.

    Explicit 180-degree handling avoids NaNs when a curved centerline folds
    back against its root tangent. These are rotations, not skinning weights.
    """
    directions = _unit_rows(directions)
    target = _unit_rows(np.asarray(target, dtype=np.float64)[None, :])[0]
    dst = np.broadcast_to(target, directions.shape)
    cross = np.cross(directions, dst)
    sin_angle = np.linalg.norm(cross, axis=1)
    dot = np.clip(np.einsum('ij,ij->i', directions, dst), -1., 1.)
    angle = np.arctan2(sin_angle, dot)
    result = np.zeros_like(directions)
    regular = sin_angle > 1e-9
    result[regular] = (cross[regular] / sin_angle[regular, None]) * angle[regular, None]
    antiparallel = (~regular) & (dot < 0)
    if np.any(antiparallel):
        # Choose a deterministic perpendicular axis for the 180deg case.
        anti = directions[antiparallel]
        unit_axes = np.eye(3)[np.argmin(np.abs(anti), axis=1)]
        axes = _unit_rows(np.cross(anti, unit_axes))
        result[antiparallel] = np.pi * axes
    return result


def straighten_positions(rest_curve, s_vertex, offsets, *, progress,
                         root_hold=0.12, full_straight=0.85):
    """Straighten ONE centerline while keeping its root and arc length.

    After full_straight each tangent aligns with the tangent at the root hold
    point. For partial poses the rotation field is smoothly ramped in space
    and time. Vertices follow their bound centerline point and rotated rest
    offset. This is only a demo deformation, not a collision-free rig.
    """
    if not np.isfinite(progress) or not (0.0 <= progress <= 1.0):
        raise ValueError('straightening progress must be in [0, 1]')
    if not (0 <= root_hold < full_straight <= 1):
        raise ValueError('Require 0 <= root_hold < full_straight <= 1')
    rest_curve = np.asarray(rest_curve, dtype=np.float64)
    s_vertex = np.asarray(s_vertex, dtype=np.float64).reshape(-1)
    offsets = np.asarray(offsets, dtype=np.float64)
    if rest_curve.ndim != 2 or rest_curve.shape[1] != 3 or len(rest_curve) < 3:
        raise ValueError('Need an Nx3 centerline with at least three samples')
    if offsets.shape != (len(s_vertex), 3):
        raise ValueError('Offsets must be one 3D vector per vertex')
    if np.any((s_vertex < 0) | (s_vertex > 1)):
        raise ValueError('Curve vertex parameters must lie in [0,1]')
    u = np.linspace(0.0, 1.0, len(rest_curve))
    increments = np.diff(rest_curve, axis=0)
    tangent = _unit_rows(increments)
    mids = (u[:-1] + u[1:]) * 0.5
    # Use the tangent immediately before the held/attachment boundary so the
    # resulting straight portion continues in the root's outgoing direction.
    junction_edge = int(np.clip(np.searchsorted(mids, root_hold, side='right') - 1,
                                0, len(tangent) - 1))
    target_direction = tangent[junction_edge]
    full_rotations = _minimal_rotation_vectors(tangent, target_direction)
    segment_factor = progress * smooth_bend_profile(mids, root_hold, full_straight)
    rotations = Rotation.from_rotvec(full_rotations * segment_factor[:, None])
    posed_segments = rotations.apply(increments)
    posed_curve = np.vstack((rest_curve[0], rest_curve[0] +
                            np.cumsum(posed_segments, axis=0)))
    anchors = np.stack([np.interp(s_vertex, u, posed_curve[:, i])
                        for i in range(3)], axis=1)
    # Interpolate original *unit tangents* at each bound coordinate; construct
    # the offset rotation with exactly the same shortest-arc rule.
    vertex_tangent = np.stack([np.interp(s_vertex, mids, tangent[:, j])
                               for j in range(3)], axis=1)
    # At an exact U-turn, tangent interpolation could be zero; use nearest
    # segment tangent for these rare vertices instead of dividing by zero.
    bad = np.linalg.norm(vertex_tangent, axis=1) < 1e-9
    if np.any(bad):
        nearest = np.clip(np.searchsorted(mids, s_vertex[bad]), 0, len(tangent) - 1)
        vertex_tangent[bad] = tangent[nearest]
    vertex_rotations = _minimal_rotation_vectors(vertex_tangent, target_direction)
    vertex_factor = progress * smooth_bend_profile(s_vertex, root_hold, full_straight)
    offset_rot = Rotation.from_rotvec(vertex_rotations * vertex_factor[:, None])
    return anchors + offset_rot.apply(offsets), posed_curve


def load_other_curves(path, target, count=128):
    """Sample all other avatar curves to identify the target's own surface.

    Uses full original avatar geometry rather than cutting the tentacle from
    the body. The transition only drives target-owned vertices, preserving the
    original mesh topology and the body/tentacle attachment indices.
    """
    centers = []
    with np.load(path, allow_pickle=True) as data:
        if 'segments' not in data:
            raise ValueError('Expected std_handle.npz with segments')
        for segment in data['segments']:
            if str(segment.get('name')) == target:
                continue
            points = np.asarray(segment['keypoints'], dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
                continue
            if not np.isfinite(points).all():
                continue
            lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
            if np.any(lengths < 1e-10):
                continue
            _, sampled = sample_curve(points, count)
            centers.append(sampled)
    if not centers:
        raise ValueError('Need another skeleton curve (e.g. body) to protect the rest of the avatar')
    return np.concatenate(centers, axis=0)


def bind_connected_avatar(vertices, centers, params, other_centers, *,
                          distance_full=0.15, distance_zero=0.30,
                          owner_band=0.035, root_hold=0.12, root_blend_end=0.30,
                          batch=150000):
    """Automatically compute t7-only vertex influence, including root anchor.

    No disconnected mesh extraction and no skin-weight painting: we move
    vertices within the *original* avatar mesh and do not alter its faces.
    Blend influence is based on distances to target vs other skeleton curves.
    The first root_hold fraction stays exactly unchanged.
    """
    if not (0 < distance_full < distance_zero) or owner_band <= 0:
        raise ValueError('Require 0 < distance_full < distance_zero and owner_band > 0')
    if not (0 <= root_hold < root_blend_end <= 1):
        raise ValueError('Root hold/blend fractions must satisfy 0 <= hold < end <= 1')
    s, offsets = project_mesh_vertices(vertices, centers, params)
    selected_d = np.linalg.norm(offsets, axis=1)
    tree = cKDTree(other_centers)
    competing_d = np.empty(len(selected_d), dtype=np.float64)
    for start in range(0, len(selected_d), batch):
        stop = min(start + batch, len(selected_d))
        competing_d[start:stop] = tree.query(vertices[start:stop], k=1)[0]
    smooth = lambda t: t*t*(3-2*t)
    nearest = smooth(np.clip((competing_d - selected_d + owner_band)
                              / (2*owner_band), 0, 1))
    radius = smooth(np.clip((distance_zero-selected_d)
                             / (distance_zero-distance_full), 0, 1))
    root = smooth(np.clip((s-root_hold)/(root_blend_end-root_hold), 0, 1))
    weights = nearest * radius * root
    return s, offsets, weights


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
    # Demo diagnostics only; original avatar topology remains untouched.
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
    motion = getattr(args, 'motion', 'bend')
    if motion not in ('bend', 'straighten'):
        raise ValueError('motion must be bend or straighten')
    axis = select_bend_axis(centers, args.axis) if motion == 'bend' else None
    if motion == 'straighten' and not (0 <= args.root_hold < args.full_bend <= 1):
        raise ValueError('Straighten: require root-hold < full-bend <= 1')
    moving_paths = parse_mesh_args(args.moving)
    static_paths = parse_mesh_args(args.static)
    avatar_paths = parse_mesh_args(getattr(args, 'deform_avatar', []))
    if not moving_paths and not avatar_paths:
        raise ValueError('Pass at least one --moving or --deform-avatar NAME=PATH')
    if len(set(moving_paths) | set(static_paths) | set(avatar_paths)) != (
            len(moving_paths) + len(static_paths) + len(avatar_paths)):
        raise ValueError('Moving/avatar/static mesh names overlap')
    meshes = {name: load_mesh(p) for name,p in moving_paths.items()}
    avatar_meshes = {name: load_mesh(p) for name,p in avatar_paths.items()}
    meshes.update(avatar_meshes)
    static = {name: load_mesh(p) for name,p in static_paths.items()}
    bindings = {name: project_mesh_vertices(mesh.vertices, centers, params)
                for name,mesh in meshes.items() if name not in avatar_meshes}
    if avatar_meshes:
        others = load_other_curves(args.handle_npz, args.curve)
        for name, mesh in avatar_meshes.items():
            bindings[name] = bind_connected_avatar(
                mesh.vertices, centers, params, others,
                distance_full=getattr(args, 'avatar_distance_full', 0.15),
                distance_zero=getattr(args, 'avatar_distance_zero', 0.30),
                owner_band=getattr(args, 'avatar_owner_band', 0.035),
                root_hold=args.root_hold,
                root_blend_end=getattr(args, 'avatar_root_blend_end', 0.30))
            active = int(np.count_nonzero(bindings[name][2] > .01))
            print(f'[avatar_motion] {name}: {active}/{len(mesh.vertices)} vertices '
                  'follow selected curve; original mesh faces retained', flush=True)
            if active == 0:
                raise ValueError('No avatar vertices follow selected curve; check '
                                 'NPZ/mesh coordinate alignment and distance gates')
    out = Path(args.output);out.mkdir(parents=True, exist_ok=True)
    if args.frames < 3:
        raise ValueError('frames must be >= 3')
    report={'curve':args.curve,'handle_npz':str(args.handle_npz),
            'root_end':root,'motion':motion,
            'axis':(axis.tolist() if axis is not None else None),
            'angle_deg':(args.angle_deg if motion == 'bend' else None),
            'root_hold':args.root_hold,
            'full_bend_or_straight':args.full_bend,
            'frames':args.frames,'moving':list(meshes),'static':list(static),
            'deformed_full_avatar':list(avatar_meshes),
            'backend':'rest_vertex_curve_binding; no SDF re-evaluation',
            'connected_surface_validated':False,
            'collision_free_validated':False,'poses':[]}
    for i, phase in enumerate(np.linspace(0, np.pi, args.frames)):
        pose_amount = float(np.sin(phase))  # rest -> maximum -> rest
        angle = float(args.angle_deg*pose_amount) if motion == 'bend' else None
        def pose_vertices(s, offsets):
            if motion == 'straighten':
                return straighten_positions(centers, s, offsets,
                                            progress=pose_amount,
                                            root_hold=args.root_hold,
                                            full_straight=args.full_bend)
            return bend_positions(centers, s, offsets, angle_deg=angle,
                                  bend_axis=axis, root_hold=args.root_hold,
                                  full_bend=args.full_bend)
        scene=trimesh.Scene()
        for name,mesh in static.items():
            scene.add_geometry(mesh,geom_name=name,node_name=name)
        quality={}
        posed_skel = None
        for name, original in meshes.items():
            if name in avatar_meshes:
                s, offsets, weights = bindings[name]
                moved_points = np.array(original.vertices, dtype=np.float64, copy=True)
                active = weights > 1e-9
                if np.any(active):
                    raw, posed_skel = pose_vertices(s[active], offsets[active])
                    moved_points[active] += (raw-moved_points[active]) * weights[active,None]
            else:
                s, offsets = bindings[name]
                moved_points, posed_skel = pose_vertices(s, offsets)
            moved=original.copy()
            moved.vertices=moved_points
            # preserve connectivity; no process/weld/automatic fixes
            moved.export(out/f'{name}_{i:03d}.ply')
            scene.add_geometry(moved,geom_name=name,node_name=name)
            quality[name]=mesh_quality(original,moved)
        if posed_skel is None:
            _, posed_skel = pose_vertices(np.array([0.0]), np.zeros((1, 3)))
        scene.export(out/f'pose_{i:03d}.glb')
        np.save(out/f'skeleton_{args.curve}_{i:03d}.npy',posed_skel)
        report['poses'].append({'frame':i,'angle_deg':angle,
                                'straighten_progress':(pose_amount if motion == 'straighten' else None),
                                'quality':quality})
        description = (f'progress={pose_amount:.3f}' if motion == 'straighten'
                       else f'angle={angle:+.2f} deg')
        print(f'[offline_pose] frame={i:03d} motion={motion} {description} saved', flush=True)
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
    p.add_argument('--deform-avatar',action='append',default=[],metavar='NAME=PLY',
                   help='Whole connected octopus avatar mesh, selectively deform only t7 '
                        'vertices and leave the root and other tentacles anchored')
    p.add_argument('--avatar-distance-full',type=float,default=0.15,
                   help='All influence within this distance to selected curve')
    p.add_argument('--avatar-distance-zero',type=float,default=0.30,
                   help='No influence beyond this distance to selected curve')
    p.add_argument('--avatar-owner-band',type=float,default=0.035,
                   help='Smooth ownership transition between selected and other curves')
    p.add_argument('--avatar-root-blend-end',type=float,default=0.30,
                   help='First full root-to-tip influence point (root-hold remains anchored)')
    p.add_argument('--output',type=Path,default=Path('inference/offline_motion_512'))
    p.add_argument('--frames',type=int,default=9)
    p.add_argument('--motion', choices=['bend', 'straighten'], default='bend',
                   help='bend: original 0052 deformation; straighten: flatten the free tentacle '
                        'towards its root tangent (rest -> straight -> rest)')
    p.add_argument('--angle-deg',type=float,default=30,
                   help='Max bend angle (used only for --motion bend)')
    p.add_argument('--axis',choices=['x','y','z'],default='z')
    p.add_argument('--root-end',choices=['auto','start','end'],default='auto')
    p.add_argument('--root-hold',type=float,default=0.12)
    p.add_argument('--full-bend',type=float,default=0.85,
                   help='Curve fraction where full bending or straightening begins')
    p.add_argument('--curve-samples',type=int,default=512)
    args=p.parse_args(argv)
    export_sequence(args)


if __name__=='__main__':
    main()
