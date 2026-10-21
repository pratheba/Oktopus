"""Direct learned-UDF cutting of final, separate root/leaf SDF meshes.

Unlike per-item reference transfer, this path does not require intermediate
MC/DCSDD meshes.  Each final mesh is evaluated only by adaptations assigned
by the manifest-validated root/leaf plan; meshes are NEVER Boolean-unioned.
"""
from __future__ import annotations

import json
from pathlib import Path

import trimesh

from app.root_leaf_combination import export_unstitched_parts_ply


def load_final_root_leaf_meshes(*, plan, mc_parts_dir, root_path, require_manifest=True):
    directory = Path(mc_parts_dir).expanduser()
    if not directory.is_absolute():
        directory = Path(root_path) / directory
    directory = directory.resolve()
    missing = [part for part in plan.part_names if not (directory / f'{part}.ply').is_file()]
    if missing:
        raise FileNotFoundError('Missing final root/leaf mesh(s) ' + str(missing) + ' in ' + str(directory))

    manifest_path = directory.parent / 'root_leaves_ownership.json'
    if not manifest_path.is_file():
        if require_manifest:
            raise FileNotFoundError('Missing SDF ownership manifest: ' + str(manifest_path))
    else:
        stored = json.loads(manifest_path.read_text(encoding='utf-8'))
        expected = {
            'adaptation_fingerprint': plan.adaptation_fingerprint,
            'part_names': list(plan.part_names),
            'item_to_part': {str(k): v for k, v in sorted(plan.item_to_part.items())},
        }
        for key, value in expected.items():
            if stored.get(key) != value:
                raise ValueError(f'UDF/SDF final root/leaf {key} mismatch; use exactly the SDF adaptation list')

    result = {}
    for part in plan.part_names:
        mesh = trimesh.load(str(directory / f'{part}.ply'), process=False, force='mesh')
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            raise ValueError(f'Empty/non-triangle final root/leaf mesh: {directory / (part + ".ply")}')
        result[part] = mesh
    return directory, result


def export_direct_root_leaf_results(*, plan, source_dir, meshes, per_part_reports, output_folder):
    destination = Path(output_folder)
    out_dir = destination / 'riggable_parts_after_udf_cut'
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()
    report = {
        'mode': 'direct_udf_final_root_leaves',
        'input_dir': str(source_dir),
        'part_names': list(plan.part_names),
        'leaf_leaf_boolean': False,
        'connected_mesh': False,
        'attachment_stitching_done': False,
        'parts': {},
    }
    for part in plan.part_names:
        mesh = meshes[part]
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            raise RuntimeError(f'No faces remain in final UDF-cut part: {part}')
        part_path = out_dir / f'{part}.ply'
        mesh.export(part_path)
        scene.add_geometry(mesh, geom_name=part, node_name=part)
        report['parts'][part] = {
            **per_part_reports[part],
            'output': str(part_path),
        }
    one_file = export_unstitched_parts_ply(
        meshes, destination / 'riggable_after_udf_cut_UNSTITCHED_parts.ply', plan
    )
    report['single_file_unstitched'] = str(one_file)
    scene_path = destination / 'riggable_after_udf_cut.glb'
    scene.export(scene_path, file_type='glb')
    report['glb'] = str(scene_path)
    report_path = destination / 'riggable_after_udf_cut_report.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('[udf cut final root_leaves] saved', scene_path,
          'parts=', len(meshes), 'leaf_leaf_boolean=False', flush=True)
    return report
