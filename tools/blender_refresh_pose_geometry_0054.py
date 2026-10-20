#!/usr/bin/env python3
"""0054: Refresh mesh/skeleton pose shape keys without losing Blender materials/camera.

This runs IN BLENDER on an existing 0051 animation .blend made from the same
rest mesh and same number of poses. It updates shape-key COORDINATES only;
no objects, materials, cameras, lights, or keyframe actions are replaced.

Example:
  blender -b /path/to/colored_camera_scene.blend \\
    --python tools/blender_refresh_pose_geometry_0054.py -- \\
    --pose-dir inference/demo_t7_0054 \\
    --output inference/demo_t7_0054/animation_preserve_camera_0054.blend
"""
import argparse
import json
from pathlib import Path
import sys

import bpy
import numpy as np


def options():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pose-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True,
                   help='Save as a NEW .blend to protect your original scene')
    return p.parse_args(argv)


def import_ply_temp(path):
    before = set(bpy.data.objects)
    if hasattr(bpy.ops.wm, 'ply_import'):
        bpy.ops.wm.ply_import(filepath=str(path))
    elif hasattr(bpy.ops.import_mesh, 'ply'):
        bpy.ops.import_mesh.ply(filepath=str(path))
    else:
        raise RuntimeError('Blender PLY importer not available')
    objs = [o for o in bpy.data.objects if o not in before and o.type == 'MESH']
    if len(objs) != 1:
        raise ValueError(f'Expected one mesh from {path}, got {len(objs)}')
    return objs[0]


def coords_and_faces(mesh):
    coords = np.empty(len(mesh.vertices)*3, dtype=np.float32)
    mesh.vertices.foreach_get('co', coords)
    loops = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get('vertex_index', loops)
    return coords, loops


def update_mesh(obj, name, folder, count):
    if obj is None or obj.type != 'MESH' or obj.data.shape_keys is None:
        raise ValueError(f'Existing animation object ANIM_{name} missing or has no shape keys')
    basis = obj.data.shape_keys.key_blocks.get('Basis')
    if basis is None:
        raise ValueError(f'ANIM_{name}: missing Basis shape key')
    existing_keys = [int(k.name[5:]) for k in obj.data.shape_keys.key_blocks
                     if k.name.startswith('pose_') and k.name[5:].isdigit()]
    # The sequence ends in a rest frame without an explicit last shape key.
    # This prevents silently mixing 9-pose and 15-pose animations.
    if not existing_keys or max(existing_keys) != count - 2:
        raise ValueError(f'ANIM_{name}: existing shape-key frame count differs '
                         'from the new pose folder; generate a fresh .blend')
    base = np.empty(len(basis.data)*3, dtype=np.float32)
    basis.data.foreach_get('co', base)
    original_faces = np.empty(len(obj.data.loops), dtype=np.int32)
    obj.data.loops.foreach_get('vertex_index', original_faces)
    for i in range(count):
        path = folder / f'{name}_{i:03d}.ply'
        if not path.is_file():
            raise FileNotFoundError(path)
        imported = import_ply_temp(path)
        data = imported.data
        try:
            coords, faces = coords_and_faces(data)
            if coords.shape != base.shape or not np.array_equal(faces, original_faces):
                raise ValueError(f'{name}, frame {i}: vertex/face ordering changed; '
                                 'rebuild the Blender scene instead')
            if not np.isfinite(coords).all():
                raise ValueError(f'{name}, frame {i}: nonfinite coordinates')
            if i == 0:
                if not np.allclose(coords, base, atol=2e-5, rtol=0):
                    raise ValueError(f'{name}: rest pose differs from existing .blend; '
                                     'cannot safely refresh old shape keys')
                continue
            block = obj.data.shape_keys.key_blocks.get(f'pose_{i:03d}')
            if block is None:
                if not np.allclose(coords, base, atol=2e-5, rtol=0):
                    raise ValueError(f'{name}: missing animated key pose_{i:03d}; '
                                     'use a newly generated Blender scene')
            else:
                block.data.foreach_set('co', coords)
        finally:
            bpy.data.objects.remove(imported, do_unlink=True)
            if data.users == 0:
                bpy.data.meshes.remove(data)
    print(f'[0054 refresh] mesh={name} poses={count}, materials retained', flush=True)


def update_skeleton(folder, curve, count):
    obj = bpy.data.objects.get(f'SKELETON_{curve}')
    if obj is None:
        print('[0054 refresh] no skeleton overlay in old scene; skip', flush=True)
        return
    if obj.data.shape_keys is None:
        raise ValueError('Existing skeleton overlay has no shape keys')
    keyblocks = obj.data.shape_keys.key_blocks
    rest = np.load(folder / f'skeleton_{curve}_000.npy', allow_pickle=False)
    basis = keyblocks.get('Basis')
    if basis is None or len(basis.data) != len(rest):
        raise ValueError('Existing skeleton geometry differs from pose source')
    for i in range(1,count):
        points = np.asarray(np.load(folder/f'skeleton_{curve}_{i:03d}.npy',
                                     allow_pickle=False), dtype=np.float32)
        if points.shape != rest.shape or not np.isfinite(points).all():
            raise ValueError(f'Invalid skeleton pose {i}')
        key = keyblocks.get(f'skeleton_pose_{i:03d}')
        if key is None:
            raise ValueError(f'Existing skeleton key missing at frame {i}')
        key.data.foreach_set('co', points.reshape(-1))
    print(f'[0054 refresh] skeleton={curve} poses={count}', flush=True)


def main():
    args = options()
    folder = args.pose_dir.expanduser().resolve()
    manifest = json.loads((folder/'pose_manifest.json').read_text())
    names = manifest['moving']
    count = int(manifest['frames'])
    curve = manifest['curve']
    if count < 3:
        raise ValueError('At least three pose frames are required')
    for name in names:
        update_mesh(bpy.data.objects.get(f'ANIM_{name}'), name, folder, count)
    update_skeleton(folder, curve, count)
    bpy.context.scene.frame_set(bpy.context.scene.frame_start)
    out = args.output.expanduser().resolve()
    if out.suffix.lower() != '.blend':
        raise ValueError('Output must be .blend')
    if out == Path(bpy.data.filepath).resolve():
        raise ValueError('Save to a NEW .blend path, not the input file')
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out))
    print(f'[0054 refresh] saved={out} with existing camera/materials/light', flush=True)


if __name__ == '__main__':
    main()
