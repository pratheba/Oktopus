#!/usr/bin/env python3
"""Promote inspected 0081 UDF *preview* meshes to an 8-part riggable assembly.

No inference, thresholding, face removal, blending, Boolean union, welding or
boundary smoothing is performed. This DOES NOT stitch root/leaf interfaces.

Use ONLY after visually inspecting all eight 0081 proposed_kept_cleaned.ply
files. The original SDF meshes, the 0081 preview files, and any existing
production assembly are never overwritten.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from app.root_leaf_combination import RootLeafPlan  # noqa: E402
from app.udf_cut_final_root_leaves import export_direct_root_leaf_results  # noqa: E402


def _mesh(path: Path) -> trimesh.Trimesh:
    if not path.is_file():
        raise FileNotFoundError(f'Required mesh missing: {path}')
    mesh = trimesh.load(path, process=False, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f'Empty/non-triangle mesh: {path}')
    vertices = np.asarray(mesh.vertices)
    if not np.all(np.isfinite(vertices)):
        raise ValueError(f'Non-finite vertices in {path}')
    return mesh


def _validated_preview(sdf_mesh: trimesh.Trimesh, preview: trimesh.Trimesh,
                       *, tolerance: float, part: str, summary: dict, locality_path: Path) -> dict:
    """Fail closed on stale/foreign 0081 previews, no hidden geometry changes."""
    n0, n1 = len(sdf_mesh.faces), len(preview.faces)
    if n1 > n0:
        raise ValueError(f'{part}: preview has more triangles than original')
    if int(summary.get('faces_after_preview', -1)) != n1:
        raise ValueError(f'{part}: summary face count disagrees with preview')
    if summary.get('mode') != 'surface_and_arclength_local_0081':
        raise ValueError(f'{part}: not an 0081 curve-local summary')
    if not locality_path.is_file():
        raise FileNotFoundError(f'Missing 0081 face evidence: {locality_path}')
    with np.load(locality_path, allow_pickle=False) as cache:
        required = {'selected_faces', 'candidate_faces', 'seed_faces',
                    'face_centroids_xyz'}
        if not required.issubset(cache.files):
            raise ValueError(f'{part}: 0081 locality cache incomplete')
        selected = np.asarray(cache['selected_faces'], dtype=bool)
        candidate = np.asarray(cache['candidate_faces'], dtype=bool)
        seeds = np.asarray(cache['seed_faces'], dtype=bool)
        centers = np.asarray(cache['face_centroids_xyz'], dtype=np.float64)
        if (selected.shape != (n0,) or candidate.shape != (n0,)
                or seeds.shape != (n0,) or centers.shape != (n0, 3)):
            raise ValueError(f'{part}: 0081 locality dimensions do not match SDF mesh')
        if not np.all(seeds <= selected) or not np.all(selected <= candidate):
            raise ValueError(f'{part}: invalid 0081 selected/seed masks')
        center_error = np.max(np.linalg.norm(
            centers - np.asarray(sdf_mesh.triangles_center), axis=1))
        if center_error > tolerance:
            raise ValueError(f'{part}: locality evidence comes from different SDF geometry '
                             f'(centroid error={center_error:g})')
        selected_count = int(np.count_nonzero(selected))
        if selected_count != int(summary.get('selected_faces', -1)):
            raise ValueError(f'{part}: selected count mismatch')
    # This step must assemble exactly the 0081 preview geometry, not perform
    # an unrecorded smoothing/reconstruction. All vertices must be present in
    # the original SDF part within only float32 PLY serialization tolerance.
    if tolerance <= 0:
        raise ValueError('tolerance must be positive')
    distance, _ = cKDTree(np.asarray(sdf_mesh.vertices)).query(
        np.asarray(preview.vertices), k=1, workers=1)
    max_error = float(np.max(distance, initial=0.0))
    if max_error > tolerance:
        raise ValueError(f'{part}: preview vertices differ from SDF input '
                         f'(max nearest-vertex delta={max_error:g})')
    # The preview can delete detached fragments in addition to 0081-selected
    # faces, but must never retain a face count larger than that selection allows.
    if n1 > n0 - selected_count:
        raise ValueError(f'{part}: preview still contains faces selected for removal')
    return {
        'original_faces': n0,
        'selected_faces_0081': selected_count,
        'cleanup_removed_faces': n0 - selected_count - n1,
        'faces_after_preview': n1,
        'max_original_vertex_delta': max_error,
        'max_original_centroid_delta': float(center_error),
    }


def _write_ownership_manifest(*, original: dict, plan: RootLeafPlan,
                              meshes: dict, destination: Path, final_dir: Path,
                              sdf_run: Path, udf_run: Path) -> None:
    manifest = {
        'schema_version': original.get('schema_version', 49),
        'adaptation_fingerprint': plan.adaptation_fingerprint,
        'part_names': list(plan.part_names),
        'item_to_part': {str(k): v for k, v in sorted(plan.item_to_part.items())},
        'connected_mesh': False,
        'stitched_root_leaf_interfaces': False,
        'leaf_leaf_boolean': False,
        'source_sdf_run': str(sdf_run),
        'source_udf_preview_run': str(udf_run),
        'provenance': '0081_proposed_kept_cleaned_no_geometry_modification',
        'parts': {},
        'unstitched_mesh': str(final_dir / 'riggable_after_udf_cut_UNSTITCHED_parts.ply'),
        'scene': str(final_dir / 'riggable_after_udf_cut.glb'),
    }
    vstart = fstart = 0
    for part_id, name in enumerate(plan.part_names):
        mesh = meshes[name]
        indices = [i for i, p in sorted(plan.item_to_part.items()) if p == name]
        manifest['parts'][name] = {
            'part_id': part_id,
            'adaptation_indices': indices,
            'vertex_range': [vstart, vstart + len(mesh.vertices)],
            'face_range': [fstart, fstart + len(mesh.faces)],
        }
        vstart += len(mesh.vertices)
        fstart += len(mesh.faces)
    (destination / 'root_leaves_ownership.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8')


def promote(*, sdf_run: Path, udf_run: Path, out: Path,
            tolerance: float = 2e-5) -> dict:
    sdf_run = Path(sdf_run).resolve()
    udf_run = Path(udf_run).resolve()
    out = Path(out).resolve()
    if not sdf_run.is_dir() or not udf_run.is_dir():
        raise FileNotFoundError('SDF and UDF run directories must both exist')
    if out == sdf_run or out == udf_run or out in sdf_run.parents or out in udf_run.parents:
        raise ValueError('Output must be separate from SDF and UDF input directories')
    if sdf_run in out.parents or udf_run in out.parents:
        raise ValueError('Output cannot be nested inside either input directory')
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {out}')
    manifest_path = sdf_run / 'root_leaves_ownership.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Missing SDF part ownership manifest: {manifest_path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    names = manifest.get('part_names')
    assignment = manifest.get('item_to_part')
    if not isinstance(names, list) or not names or len(set(names)) != len(names):
        raise ValueError('Invalid part_names in SDF ownership manifest')
    if not isinstance(assignment, dict) or any(v not in names for v in assignment.values()):
        raise ValueError('Invalid item_to_part in SDF ownership manifest')
    if len(assignment) != 15 or len(names) != 8:
        raise ValueError(f'Expected 15 adaptations and 8 final parts, got '
                         f'{len(assignment)} adaptations and {len(names)} parts')
    if not isinstance(manifest.get('adaptation_fingerprint'), str) or not manifest['adaptation_fingerprint']:
        raise ValueError('SDF ownership manifest is missing adaptation fingerprint')
    item_to_part = {int(k): v for k, v in assignment.items()}
    if sorted(item_to_part) != list(range(15)):
        raise ValueError('SDF adaptation indices must be 0..14')
    plan = RootLeafPlan(item_to_part=item_to_part, group_to_part={},
                        part_names=tuple(names), export_legacy_combined=False,
                        adaptation_fingerprint=manifest['adaptation_fingerprint'])
    meshes, reports = {}, {}
    for part in names:
        original = _mesh(sdf_run / 'riggable_parts' / (part + '.ply'))
        folder = udf_run / 'udf_cut_curve_local_0081' / part
        preview = _mesh(folder / 'proposed_kept_cleaned.ply')
        summary_path = folder / 'summary.json'
        if not summary_path.is_file():
            raise FileNotFoundError(f'Missing 0081 summary: {summary_path}')
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        review = _validated_preview(original, preview, tolerance=tolerance,
                                    part=part, summary=summary,
                                    locality_path=folder / 'locality.npz')
        meshes[part] = preview
        reports[part] = {
            'input': str(sdf_run / 'riggable_parts' / (part + '.ply')),
            'preview': str(folder / 'proposed_kept_cleaned.ply'),
            'adaptation_indices': [k for k, v in sorted(item_to_part.items()) if v == part],
            'faces_before': len(original.faces),
            'faces_removed': len(original.faces) - len(preview.faces),
            'faces_after': len(preview.faces),
            'promotion_0082': review,
        }
        print(f'[udf assembly 0082] validated part={part} '
              f'faces={len(original.faces)}->{len(preview.faces)} '
              f'udf_selected={review["selected_faces_0081"]} '
              f'detached={review["cleanup_removed_faces"]}', flush=True)

    # Export to a fresh sibling directory and only publish after all exports
    # succeed. The final report paths are rewritten after the atomic move.
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.udf0082-', dir=out.parent))
    try:
        result = export_direct_root_leaf_results(
            plan=plan, source_dir=sdf_run / 'riggable_parts', meshes=meshes,
            per_part_reports=reports, output_folder=stage,
        )
        _write_ownership_manifest(original=manifest, plan=plan, meshes=meshes,
                                  destination=stage, final_dir=out,
                                  sdf_run=sdf_run, udf_run=udf_run)
        # export_direct_root_leaf_results writes stage paths. Rewrite only
        # fields pointing to our own stage, never external inputs.
        report_path = stage / 'riggable_after_udf_cut_report.json'
        report = json.loads(report_path.read_text(encoding='utf-8'))
        def rewrite(value):
            if isinstance(value, str) and value.startswith(str(stage) + os.sep):
                return str(out) + value[len(str(stage)):]
            if isinstance(value, dict):
                return {k: rewrite(v) for k, v in value.items()}
            if isinstance(value, list):
                return [rewrite(v) for v in value]
            return value
        report = rewrite(report)
        report['promotion_0082'] = {
            'source_sdf_run': str(sdf_run),
            'source_0081_run': str(udf_run),
            'originals_untouched': True,
            'mesh_welding_performed': False,
            'boundary_smoothing_performed': False,
            'parts_validated': len(meshes),
            'vertex_match_tolerance_world': tolerance,
            'promoted_preview_files': {name: reports[name]['preview'] for name in names},
        }
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        # Verify GLB contains one node per named part and exact triangle counts.
        scene = trimesh.load(stage / 'riggable_after_udf_cut.glb', force='scene')
        if not isinstance(scene, trimesh.Scene) or len(scene.geometry) != len(names):
            raise RuntimeError('Exported GLB lost one or more separately named parts')
        for part in names:
            if part not in scene.geometry or len(scene.geometry[part].faces) != len(meshes[part].faces):
                raise RuntimeError(f'Exported GLB identity/triangle mismatch: {part}')
        os.replace(stage, out)
        print('[udf assembly 0082] saved', out / 'riggable_after_udf_cut.glb',
              'parts=', len(names), 'stitched=False', flush=True)
        return report
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdf-run-dir', type=Path, required=True,
                        help='0067 SDF shape folder containing riggable_parts and root_leaves_ownership.json')
    parser.add_argument('--udf-run-dir', type=Path, required=True,
                        help='0081 UDF shape folder containing udf_cut_curve_local_0081')
    parser.add_argument('--out', type=Path, required=True,
                        help='NEW output folder for cut-part GLB assembly (refuses overwrite)')
    parser.add_argument('--tolerance', type=float, default=2e-5,
                        help='maximum vertex/face-center matching error, world units (default 2e-5)')
    args = parser.parse_args()
    promote(sdf_run=args.sdf_run_dir, udf_run=args.udf_run_dir,
            out=args.out, tolerance=args.tolerance)


if __name__ == '__main__':
    main()
