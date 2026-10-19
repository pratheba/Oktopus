"""Opt-in root/leaves rigging assembly; never performs leaf/leaf Boolean union.

The model exports one separate mesh per part. Existing intra-leaf blend_groups
are completed by the adaptation pipeline before calling the exporter.
"""
from dataclasses import dataclass
from pathlib import Path
import re
import trimesh

_SAFE_PART = re.compile(r'^[A-Za-z][A-Za-z0-9_-]*$')


@dataclass(frozen=True)
class RootLeafPlan:
    item_to_part: dict
    group_to_part: dict
    part_names: tuple
    export_legacy_combined: bool


def build_root_leaf_plan(adaptations, spec):
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise TypeError('root_leaves must be a mapping')
    root_name = spec.get('root')
    leaves = spec.get('leaves')
    if not isinstance(root_name, str) or not root_name:
        raise ValueError('root_leaves.root must name exactly one adaptation')
    if not isinstance(leaves, dict) or not leaves:
        raise ValueError('root_leaves.leaves must map part names to target_key strings')
    if 'body' in leaves:
        raise ValueError('root_leaves: body is reserved for the root')
    if not all(isinstance(p, str) and _SAFE_PART.fullmatch(p) for p in leaves):
        raise ValueError('Every root_leaves part name must be a safe identifier')
    targets = list(leaves.values())
    if not all(isinstance(t, str) and t for t in targets):
        raise ValueError('root_leaves.leaves values must be nonempty target_key strings')
    if len(set(targets)) != len(targets):
        raise ValueError('root_leaves target_keys must be unique')
    root_matches = [i for i, item in enumerate(adaptations) if item.get('name') == root_name]
    if len(root_matches) != 1:
        raise ValueError(f'root_leaves.root {root_name!r} matched {len(root_matches)} entries; expected one')
    root_i = root_matches[0]
    if adaptations[root_i]['target_key'] in targets:
        raise ValueError('root target_key cannot also be a leaf')
    target_parts = {target: part for part, target in leaves.items()}
    item_to_part = {root_i: 'body'}
    for i, item in enumerate(adaptations):
        if i == root_i:
            continue
        part = target_parts.get(item['target_key'])
        if part is None:
            raise ValueError(f'Unassigned adaptation {i}: {item.get("name", item["target_key"])}')
        item_to_part[i] = part
    for p in leaves:
        if p not in item_to_part.values():
            raise ValueError(f'Leaf {p!r} has no enabled adaptation')
    group_to_part = {}
    part_members = {}
    for i, item in enumerate(adaptations):
        part = item_to_part[i]
        part_members.setdefault(part, []).append(i)
        group = str(item.get('blend_group', 'none'))
        if group not in ('none', '', 'false'):
            earlier = group_to_part.setdefault(group, part)
            if earlier != part:
                raise ValueError(f'blend_group {group!r} spans {earlier!r} and {part!r}')
    for part, members in part_members.items():
        if len(members) > 1:
            groups = {str(adaptations[i].get('blend_group', 'none')) for i in members}
            if len(groups) != 1 or next(iter(groups)) in ('none', '', 'false'):
                raise ValueError(f'Part {part!r} has multiple pieces without one shared blend_group')
    return RootLeafPlan(
        item_to_part=item_to_part,
        group_to_part=group_to_part,
        part_names=('body', *leaves.keys()),
        export_legacy_combined=bool(spec.get('export_legacy_combined', False)),
    )


def export_root_leaf_scene(meshes, output_folder, out_name, plan):
    missing = set(plan.part_names) - set(meshes)
    if missing:
        raise RuntimeError(f'Root-leaves output missing parts {sorted(missing)}')
    directory = Path(output_folder) / 'riggable_parts'
    directory.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()
    for part in plan.part_names:
        mesh = meshes[part]
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            raise RuntimeError(f'Empty root-leaves part: {part}')
        mesh.export(directory / f'{part}.ply')
        scene.add_geometry(mesh, geom_name=part, node_name=part)
    path = Path(output_folder) / f'{out_name}_riggable.glb'
    scene.export(path, file_type='glb')
    print('[root_leaves] saved', path, 'parts=', list(plan.part_names), 'leaf_leaf_boolean=False')
    return path
