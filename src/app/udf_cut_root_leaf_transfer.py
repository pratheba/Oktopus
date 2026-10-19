"""Transfer per-adaptation UDF-cut decisions to independent riggable MC parts.

No leaf-to-leaf operation is performed here. In particular this code never
concatenates part meshes and never computes a global Boolean/SDF union.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import trimesh

from udf_cut_combined_transfer import transfer_udf_cut_to_combined_mesh


def transfer_udf_cut_root_leaves(
    *,
    plan,
    mc_parts_dir: str,
    root_path: str,
    output_folder: str,
    removed_references: Sequence[trimesh.Trimesh],
    kept_references: Sequence[trimesh.Trimesh],
    config: Mapping,
    cleanup_fn=None,
    cleanup_kwargs=None,
):
    """Cut each named MC part using *only* its own adaptation's UDF evidence.

    An entirely uncut part is retained unchanged, rather than passing empty
    removed-reference sets to transfer_udf_cut_to_combined_mesh (which rejects
    empty references). This is common for intentionally open or clean parts.
    """
    if len(removed_references) != len(plan.item_to_part):
        raise ValueError('Missing per-adaptation removed references for root-leaves UDF transfer')
    if len(kept_references) != len(plan.item_to_part):
        raise ValueError('Missing per-adaptation kept references for root-leaves UDF transfer')

    src_dir = Path(mc_parts_dir).expanduser()
    if not src_dir.is_absolute():
        src_dir = Path(root_path) / src_dir
    src_dir = src_dir.resolve()
    missing = [part for part in plan.part_names if not (src_dir / (part + '.ply')).is_file()]
    if missing:
        raise FileNotFoundError(
            'Missing root-leaves MC part(s) ' + str(missing)
            + ' in ' + str(src_dir)
            + '; run the matching 0043 SDF root_leaves YAML first'
        )

    out_dir = Path(output_folder) / 'riggable_parts_after_udf_cut'
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()
    report = {'input_dir': str(src_dir), 'parts': {}, 'leaf_leaf_boolean': False}
    result = {}
    for part in plan.part_names:
        part_path = src_dir / (part + '.ply')
        mesh = trimesh.load(str(part_path), process=False, force='mesh')
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            raise ValueError('Empty/non-triangle riggable MC part: ' + str(part_path))
        indices = [i for i, owner in sorted(plan.item_to_part.items()) if owner == part]
        removed = [removed_references[i] for i in indices]
        kept = [kept_references[i] for i in indices]
        removed_faces = sum(len(r.faces) for r in removed)
        kept_faces = sum(len(r.faces) for r in kept)
        method = 'transfer'
        transfer_report = None
        cleanup_report = None
        if removed_faces and kept_faces:
            mesh_cut, _removed_mesh, transfer_report = transfer_udf_cut_to_combined_mesh(
                mesh,
                removed_reference_meshes=removed,
                kept_reference_meshes=kept,
                radius_edge_fraction=float(config.get('udf_cut_combined_transfer_radius_edges', 2.0)),
                seed_advantage_edge_fraction=float(config.get('udf_cut_combined_transfer_seed_advantage_edges', .30)),
                grow_advantage_edge_fraction=float(config.get('udf_cut_combined_transfer_grow_advantage_edges', 0.0)),
                min_faces=int(config.get('udf_cut_combined_transfer_min_faces', 8)),
                min_seed_faces=int(config.get('udf_cut_combined_transfer_min_seed_faces', 2)),
                min_seed_fraction=float(config.get('udf_cut_combined_transfer_min_seed_fraction', .05)),
                min_valid_samples=int(config.get('udf_cut_combined_transfer_min_valid_samples', 2)),
            )
            if len(mesh_cut.faces) == 0:
                raise RuntimeError('UDF transfer removed all faces from part ' + part)
            if config.get('udf_cut_cleanup_boundary', False):
                if cleanup_fn is None:
                    raise ValueError('UDF boundary cleanup requested but no cleanup function supplied')
                mesh_cut, cleanup_report = cleanup_fn(mesh_cut, **(cleanup_kwargs or {}))
            result_mesh = mesh_cut
        else:
            method = 'no_cut_no_removed_evidence' if not removed_faces else 'no_cut_no_kept_evidence'
            result_mesh = mesh
        if not len(result_mesh.faces):
            raise RuntimeError('Empty riggable part after UDF: ' + part)
        destination = out_dir / (part + '.ply')
        result_mesh.export(destination)
        scene.add_geometry(result_mesh, geom_name=part, node_name=part)
        result[part] = result_mesh
        report['parts'][part] = {
            'input': str(part_path),
            'output': str(destination),
            'adaptation_indices': indices,
            'reference_removed_faces': int(removed_faces),
            'reference_kept_faces': int(kept_faces),
            'faces_before': int(len(mesh.faces)),
            'faces_after': int(len(result_mesh.faces)),
            'method': method,
            'transfer': transfer_report,
            'cleanup': cleanup_report,
        }
        print('[udf cut root_leaves]', 'part=', part, 'indices=', indices,
              'faces=', len(mesh.faces), '->', len(result_mesh.faces),
              'mode=', method, flush=True)

    glb_path = Path(output_folder) / 'riggable_after_udf_cut.glb'
    scene.export(glb_path, file_type='glb')
    report['glb'] = str(glb_path)
    report_path = Path(output_folder) / 'riggable_after_udf_cut_report.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('[udf cut root_leaves] saved', str(glb_path),
          'parts=', len(result), 'leaf_leaf_boolean=False', flush=True)
    return result, report
