"""Explicit ownership for root/leaf SDF and UDF processing.

This module intentionally does NOT claim that exporting an assembly makes it a
welded, manifold surface. Adjacent leaf surfaces are never Boolean-unioned.
A welded result requires an independently validated root/leaf attachment step.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import trimesh

_SAFE_PART = re.compile(r'^[A-Za-z][A-Za-z0-9_-]*$')


@dataclass(frozen=True)
class RootLeafPlan:
    item_to_part: dict
    group_to_part: dict
    part_names: tuple
    export_legacy_combined: bool
    adaptation_fingerprint: str = ''
    attachments: dict = None


def adaptation_fingerprint(adaptations):
    """Hash geometry-affecting adaptation configuration for UDF alignment.

    The entire adaptation list is used; if SDF and UDF differ even in optional
    correction fields, transfer must not silently assume corresponding meshes.
    """
    normalized = [{k: v for k, v in item.items() if not str(k).startswith('_')}
                  for item in adaptations]
    payload = json.dumps(normalized, sort_keys=True, separators=(',', ':'),
                         default=str).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _build_named_parts_plan(adaptations, spec):
    """YAML-owned part names and per-part SDF unions; no avatar-specific part names.

    A multi-adaptation part must use ONE shared blend_group in the original
    adaptation entries. That is the existing SDF group-union mechanism, not a
    new proximity-based mesh stitcher. Leaving other parts distinct is valid.
    """
    if any(k in spec for k in ('root', 'leaves')):
        raise ValueError('Use root_leaves.parts OR legacy root/leaves, not both')
    entries = spec['parts']
    if not isinstance(entries, dict) or not entries:
        raise ValueError('root_leaves.parts requires a nonempty part mapping')
    if spec.get('output_mode', 'separate_parts') != 'separate_parts':
        raise ValueError('root_leaves.parts supports only separate_parts outputs')
    item_to_part, group_to_part, attachments = {}, {}, {}
    for part, descriptor in entries.items():
        if not isinstance(part, str) or not _SAFE_PART.fullmatch(part):
            raise ValueError('root_leaves.parts part names must be safe identifiers')
        if isinstance(descriptor, str):
            descriptor = {'target_key': descriptor}
        if not isinstance(descriptor, dict):
            raise ValueError(f'Part {part!r} must have an adaptation selector')
        selectors = set(descriptor) & {'blend_group', 'target_key', 'adaptation_name', 'adaptation_indices'}
        unknown = set(descriptor) - selectors - {'attach_to'}
        if len(selectors) != 1 or unknown:
            raise ValueError(f'Part {part!r} requires exactly one selector '
                             '(blend_group, target_key, adaptation_name, adaptation_indices) '
                             f'and optional attach_to; unexpected keys={sorted(unknown)}')
        selector = next(iter(selectors))
        value = descriptor[selector]
        if selector == 'adaptation_indices':
            if (not isinstance(value, list) or not value or
                any(type(i) is not int or i < 0 or i >= len(adaptations) for i in value) or
                len(value) != len(set(value))):
                raise ValueError(f'Part {part!r} needs unique valid adaptation_indices')
            matches = list(value)
        else:
            if not isinstance(value, str) or not value:
                raise ValueError(f'Part {part!r} selector must be a nonempty string')
            key = 'name' if selector == 'adaptation_name' else selector
            matches = [i for i, a in enumerate(adaptations) if a.get(key) == value]
        if not matches:
            raise ValueError(f'Part {part!r} matches no adaptations')
        if any(i in item_to_part for i in matches):
            raise ValueError(f'Part {part!r} overlaps adaptations assigned to another part')
        if len(matches) > 1:
            groups = {adaptations[i].get('blend_group') for i in matches}
            if len(groups) != 1 or next(iter(groups)) in (None, '', False, 'none'):
                raise ValueError(f'Part {part!r} has multiple adaptations without a single shared blend_group; '
                                 'set blend_group explicitly on the intended pieces')
        for i in matches:
            item_to_part[i] = part
            group = adaptations[i].get('blend_group')
            if group not in (None, '', False, 'none'):
                prior = group_to_part.setdefault(str(group), part)
                if prior != part:
                    raise ValueError(f'blend_group {group!r} crosses parts {prior!r} and {part!r}')
        parent = descriptor.get('attach_to')
        if parent is not None:
            if not isinstance(parent, str) or parent == part:
                raise ValueError(f'Part {part!r}: attach_to must name a different part')
            attachments[part] = parent
    missing = sorted(set(range(len(adaptations))) - set(item_to_part))
    if missing:
        raise ValueError(f'root_leaves.parts leaves adaptations unassigned: {missing}')
    for part, parent in attachments.items():
        if parent not in entries:
            raise ValueError(f'Part {part!r}: attach_to={parent!r} does not exist')
        visited = {part}
        while parent in attachments:
            if parent in visited:
                raise ValueError(f'Cycle in root_leaves.parts attachments at {part!r}')
            visited.add(parent)
            parent = attachments[parent]
    return RootLeafPlan(
        item_to_part=item_to_part,
        group_to_part=group_to_part,
        part_names=tuple(entries),
        export_legacy_combined=bool(spec.get('export_legacy_combined', False)),
        adaptation_fingerprint=adaptation_fingerprint(adaptations),
        attachments=attachments,
    )


def build_root_leaf_plan(adaptations, spec):
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise TypeError('root_leaves must be a mapping')
    if 'parts' in spec:
        return _build_named_parts_plan(adaptations, spec)
    root_name = spec.get('root')
    leaves = spec.get('leaves')
    if not isinstance(root_name, str) or not root_name:
        raise ValueError('root_leaves.root must name exactly one adaptation')
    if not isinstance(leaves, dict) or not leaves:
        raise ValueError('root_leaves.leaves must map part names to selectors')
    if 'body' in leaves:
        raise ValueError('root_leaves: body is reserved for the root')
    if not all(isinstance(p, str) and _SAFE_PART.fullmatch(p) for p in leaves):
        raise ValueError('Every root_leaves part name must be a safe identifier')
    if spec.get('output_mode', 'separate_parts') != 'separate_parts':
        raise ValueError('root_leaves single connected output is not implemented: '
                         'trimming and manifold stitching must be validated first; '
                         'use output_mode: separate_parts for an unwelded assembly')

    root_matches = [i for i, item in enumerate(adaptations)
                    if item.get('name') == root_name]
    if len(root_matches) != 1:
        raise ValueError(f'root_leaves.root {root_name!r} matched '
                         f'{len(root_matches)} entries; expected one')
    root_i = root_matches[0]
    item_to_part = {root_i: 'body'}
    selector_groups = {}
    for part, descriptor in leaves.items():
        # Backward-compatible target_key strings, but new configurations
        # explicitly select intra-leaf blend groups.
        if isinstance(descriptor, str):
            descriptor = {'target_key': descriptor}
        if not isinstance(descriptor, dict):
            raise TypeError(f'root_leaves.leaves.{part} must be a selector')
        if descriptor.get('attach_to', 'root') not in ('root', 'body'):
            raise ValueError(f'Leaf {part!r}: only attach_to: root is supported')
        keys = set(descriptor) - {'attach_to'}
        if len(keys) != 1 or not keys <= {'blend_group', 'target_key', 'adaptation_name'}:
            raise ValueError(f'Leaf {part!r} needs exactly one blend_group, '
                             'target_key or adaptation_name selector')
        key = next(iter(keys))
        value = descriptor[key]
        if not isinstance(value, str) or not value:
            raise ValueError(f'Leaf {part!r} selector must be a nonempty string')
        if key == 'blend_group':
            matches = [i for i, item in enumerate(adaptations)
                       if item.get('blend_group') == value]
            # A root/leaf SDF part represents one target curve. Accidentally
            # reusing a blend group across different tentacles must fail closed.
            targets = {adaptations[i].get('target_key') for i in matches}
            if len(targets) > 1:
                raise ValueError(f'blend_group {value!r} spans multiple target curves: {targets}')
            selector_groups[value] = part
        elif key == 'target_key':
            matches = [i for i, item in enumerate(adaptations)
                       if item.get('target_key') == value]
        else:
            matches = [i for i, item in enumerate(adaptations)
                       if item.get('name') == value]
        if not matches:
            raise ValueError(f'Leaf {part!r} selector {key}={value!r} matched no adaptations')
        if root_i in matches:
            raise ValueError(f'Leaf {part!r} selector contains the root')
        for i in matches:
            if i in item_to_part:
                raise ValueError(f'Adaptation {i} assigned to multiple root_leaves parts')
            item_to_part[i] = part
    missing = set(range(len(adaptations))) - set(item_to_part)
    if missing:
        raise ValueError(f'Unassigned adaptation(s) {sorted(missing)}')

    group_to_part = {}
    part_members = {}
    for i, item in enumerate(adaptations):
        part = item_to_part[i]
        part_members.setdefault(part, []).append(i)
        group = item.get('blend_group')
        if group not in (None, '', False, 'none'):
            earlier = group_to_part.setdefault(str(group), part)
            if earlier != part:
                raise ValueError(f'blend_group {group!r} spans {earlier!r} and {part!r}')
    for part, members in part_members.items():
        if len(members) > 1:
            groups = {adaptations[i].get('blend_group') for i in members}
            if len(groups) != 1 or next(iter(groups)) in (None, '', False, 'none'):
                raise ValueError(f'Part {part!r} has multiple pieces without one shared blend_group')
    return RootLeafPlan(
        item_to_part=item_to_part,
        group_to_part=group_to_part,
        part_names=('body', *leaves.keys()),
        export_legacy_combined=bool(spec.get('export_legacy_combined', False)),
        adaptation_fingerprint=adaptation_fingerprint(adaptations),
    )


def _write_labeled_triangle_ply(path, vertices, faces, vertex_ids, face_ids):
    """Write portable PLY with explicit vertex AND face part_id properties.

    Trimesh's regular PLY exporter does not reliably retain arbitrary face
    attributes. We write the standard binary little-endian PLY fields directly.
    This records part ownership but does NOT weld disconnected surfaces.
    """
    path = Path(path)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    vertex_ids = np.asarray(vertex_ids, dtype=np.int32)
    face_ids = np.asarray(face_ids, dtype=np.int32)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or
            faces.ndim != 2 or faces.shape[1] != 3 or
            len(vertex_ids) != len(vertices) or len(face_ids) != len(faces)):
        raise ValueError('Part-labelled PLY requires Nx3 vertices/triangles and ownership arrays')
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError('PLY faces reference invalid vertices')
    if len(vertices) >= 2**31:
        raise ValueError('PLY int32 face indices cannot address this many vertices')
    header = (f'ply\nformat binary_little_endian 1.0\n'
              f'comment unstitched_root_leaf_assembly true\n'
              f'element vertex {len(vertices)}\n'
              'property double x\nproperty double y\nproperty double z\n'
              'property int part_id\n'
              f'element face {len(faces)}\n'
              'property list uchar int vertex_indices\n'
              'property int part_id\nend_header\n')
    v_dtype = np.dtype([('x','<f8'),('y','<f8'),('z','<f8'),('part_id','<i4')])
    f_dtype = np.dtype([('n','u1'),('a','<i4'),('b','<i4'),('c','<i4'),
                        ('part_id','<i4')])
    with path.open('wb') as stream:
        stream.write(header.encode('ascii'))
        for begin in range(0, len(vertices), 262144):
            end = min(begin+262144, len(vertices))
            out = np.empty(end-begin, dtype=v_dtype)
            out['x'], out['y'], out['z'] = vertices[begin:end].T
            out['part_id'] = vertex_ids[begin:end]
            stream.write(out.tobytes())
        for begin in range(0, len(faces), 262144):
            end = min(begin+262144, len(faces))
            out = np.empty(end-begin, dtype=f_dtype)
            out['n'] = 3
            out['a'], out['b'], out['c'] = faces[begin:end].T
            out['part_id'] = face_ids[begin:end]
            stream.write(out.tobytes())
    return path


def export_root_leaf_scene(meshes, output_folder, out_name, plan):
    """Export part meshes, a single-file UNSTITCHED PLY and an ownership manifest.

    The combined PLY deliberately keeps separate vertex/face sets; it is not a
    connected mesh. The face and vertex `part_id` fields let downstream rigging
    tools recover ownership even when reading the one-file representation.
    """
    missing = set(plan.part_names) - set(meshes)
    if missing:
        raise RuntimeError(f'Root-leaves output missing parts {sorted(missing)}')
    directory = Path(output_folder) / 'riggable_parts'
    directory.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()
    verts = []
    faces = []
    face_owner = []
    vertex_owner = []
    face_start = vertex_start = 0
    manifest = {
        'schema_version': 49,
        'connected_mesh': False,
        'stitched_root_leaf_interfaces': False,
        'leaf_leaf_boolean': False,
        'adaptation_fingerprint': plan.adaptation_fingerprint,
        'part_names': list(plan.part_names),
        'item_to_part': {str(k): v for k, v in sorted(plan.item_to_part.items())},
        'parts': {},
        'attachments': dict(plan.attachments or {}),
    }
    for part_id, part in enumerate(plan.part_names):
        mesh = meshes[part]
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            raise RuntimeError(f'Empty root-leaves part: {part}')
        mesh.export(directory / f'{part}.ply')
        scene.add_geometry(mesh, geom_name=part, node_name=part)
        verts.append(np.asarray(mesh.vertices, dtype=np.float64))
        faces.append(np.asarray(mesh.faces, dtype=np.int64) + vertex_start)
        face_owner.append(np.full(len(mesh.faces), part_id, dtype=np.int32))
        vertex_owner.append(np.full(len(mesh.vertices), part_id, dtype=np.int32))
        manifest['parts'][part] = {
            'part_id': part_id,
            'adaptation_indices': [i for i, p in sorted(plan.item_to_part.items()) if p == part],
            'vertex_range': [vertex_start, vertex_start + len(mesh.vertices)],
            'face_range': [face_start, face_start + len(mesh.faces)],
        }
        face_start += len(mesh.faces)
        vertex_start += len(mesh.vertices)
    one_file = Path(output_folder) / f'{out_name}_UNSTITCHED_parts.ply'
    _write_labeled_triangle_ply(one_file, np.concatenate(verts),
                               np.concatenate(faces), np.concatenate(vertex_owner),
                               np.concatenate(face_owner))
    manifest['unstitched_mesh'] = str(one_file)
    path = Path(output_folder) / f'{out_name}_riggable.glb'
    scene.export(path, file_type='glb')
    manifest['scene'] = str(path)
    manifest_path = Path(output_folder) / 'root_leaves_ownership.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf8')
    print('[root_leaves] saved', path, 'parts=', list(plan.part_names),
          'connected=False', 'leaf_leaf_boolean=False', flush=True)
    return path


def export_unstitched_parts_ply(meshes, output_path, plan):
    """One PLY asset with part_id on faces/vertices; does not weld shells."""
    vertices, faces, vertex_ids, face_ids = [], [], [], []
    offset = 0
    for part_id, part in enumerate(plan.part_names):
        mesh = meshes[part]
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            raise ValueError(f"Missing/empty mesh: {part}")
        vertices.append(np.asarray(mesh.vertices, dtype=np.float64))
        faces.append(np.asarray(mesh.faces, dtype=np.int64) + offset)
        vertex_ids.append(np.full(len(mesh.vertices), part_id, dtype=np.int32))
        face_ids.append(np.full(len(mesh.faces), part_id, dtype=np.int32))
        offset += len(mesh.vertices)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return _write_labeled_triangle_ply(target, np.concatenate(vertices),
                                      np.concatenate(faces), np.concatenate(vertex_ids),
                                      np.concatenate(face_ids))
