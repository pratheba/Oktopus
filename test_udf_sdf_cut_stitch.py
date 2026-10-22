#!/usr/bin/env python3
"""One command: learned-UDF inference -> 0081 cut evidence -> riggable assembly -> stitch.

This reads the EXISTING 0067 SDF meshes named by the chosen UDF YAML; it does
not regenerate SDF or require --mc-mesh. UDF model inference is run once using
Oktopus's original test_sdf_udf_cut.py. The 0081 curve-local proposed meshes
are independently validated and exported by the already-installed 0082
assembly exporter. Finally a strict 7-leaf boundary stitch is attempted.

If sewing is geometrically impossible the command EXITS NONZERO and saves a
failure report. The eight correct cut PLYs and GLB remain available; it never
mislabels mere concatenation as STITCHED.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import yaml
import trimesh

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.promote_udf_0081_assembly import promote
from tools.udf_stitch_0084 import PARTS, _load_part, stitch_parts, preflight
sys.path.insert(0, str(ROOT / 'src' / 'app'))
from udf_cut_fused_exterior_0092 import generate_fused_cut


def _read_yaml(path: str) -> tuple[Path, dict]:
    source = Path(path)
    if not source.is_absolute():
        source = ROOT / 'exp' / 'train' / 'manipulation' / source
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f'Missing UDF YAML: {source}')
    data = yaml.safe_load(source.read_text(encoding='utf8'))
    if not isinstance(data, dict):
        raise ValueError('UDF configuration must be a mapping')
    return source, data


def _locate_sdf_and_output(config: dict, out_name: str, shape: str) -> tuple[Path, Path, Path]:
    opts = config.get('udf_full_pipeline_0084') or {}
    if not isinstance(opts, dict) or opts.get('enabled') is not True:
        raise ValueError('Set udf_full_pipeline_0084.enabled: true in the YAML')
    if (config.get('udf_cut_direct_root_leaves') is not True or
            config.get('udf_cut_curve_local_preview_0081') is not True):
        raise ValueError('0084 requires 0081 direct-final curve-local UDF cutting')
    if config.get('udf_cut_direct_dry_run_parts_0073') is None:
        raise ValueError('0084 requires dry-run 0081 previews to preserve input until promotion')
    if not isinstance(out_name, str) or not out_name or Path(out_name).is_absolute() or '..' in Path(out_name).parts:
        raise ValueError('-o must be a relative, new inference run name')
    if not shape or Path(shape).name != shape or shape in {'.', '..'}:
        raise ValueError('-s must be a single shape name')
    rel = config.get('udf_cut_root_leaves_mesh_dir')
    if not isinstance(rel, str):
        raise ValueError('YAML must specify udf_cut_root_leaves_mesh_dir')
    sdf_part_dir = (ROOT / rel).resolve() if not Path(rel).is_absolute() else Path(rel).resolve()
    if sdf_part_dir.name != 'riggable_parts':
        raise ValueError('udf_cut_root_leaves_mesh_dir must point to riggable_parts')
    sdf_shape = sdf_part_dir.parent
    if sdf_shape.name != shape:
        raise ValueError('SDF input shape and -s must match')
    try:
        sdf_rel = sdf_shape.relative_to(ROOT / 'inference')
    except ValueError as exc:
        raise ValueError('0084 expects SDF meshes under <repo>/inference/<model-size>/...') from exc
    if len(sdf_rel.parts) < 3:
        raise ValueError('Expected inference/<num_samples>/<sdf_run>/<shape>/riggable_parts')
    sample_dir = ROOT / 'inference' / sdf_rel.parts[0]
    run_base = (sample_dir / out_name).resolve()
    if not run_base.is_relative_to(sample_dir.resolve()) or run_base == sdf_shape or sdf_shape.is_relative_to(run_base):
        raise ValueError('Output must not modify existing SDF run')
    final_shape = run_base / shape
    udf_stage = run_base / '_udf_inference_0084' / shape
    if final_shape.exists() or udf_stage.exists():
        raise FileExistsError(f'Output already exists: {final_shape if final_shape.exists() else udf_stage}')
    if not sdf_part_dir.is_dir():
        raise FileNotFoundError(f'SDF parts directory does not exist: {sdf_part_dir}')
    if not (sdf_shape / 'root_leaves_ownership.json').is_file():
        raise FileNotFoundError('Missing original SDF root_leaves_ownership.json')
    return sdf_shape, udf_stage, final_shape


def _run_inference(args, passthrough: list[str], yaml_path: Path, stage: Path) -> None:
    # Relative -o must resolve to inference/<samples>/<output-run>/_udf_inference_0084.
    samples = stage.parents[2].name
    out_base = stage.parent.relative_to(ROOT / 'inference' / samples)
    if '--mc-mesh' in passthrough or any(x.startswith('--mc-mesh=') for x in passthrough):
        raise ValueError('--mc-mesh is not used in the direct-final UDF pipeline')
    cmd = [sys.executable, str(ROOT / 'test_sdf_udf_cut.py'),
           '-c', args.config, '-o', str(out_base), '-s', args.shape,
           '-y', str(yaml_path), '-r', str(args.resolution),
           '-ckpt', args.checkpoint,
           '--udf-far-value', str(args.udf_far_value),
           '--udf-model-batch-size', str(args.udf_model_batch_size),
           '--udf-cut-query-batch-size', str(args.udf_cut_query_batch_size)]
    cmd += passthrough
    print('[udf full 0084] infer+cut:', ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not stage.is_dir():
        raise RuntimeError(f'UDF runner did not create expected stage: {stage}')
    for name in PARTS:
        expected = stage / 'udf_cut_curve_local_0081' / name / 'proposed_kept_cleaned.ply'
        if not expected.is_file():
            raise RuntimeError(f'0081 did not export the preview for {name}: {expected}')


def _json_atomic(path: Path, contents: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.tmp')
    try:
        temporary.write_text(json.dumps(contents, indent=2), encoding='utf8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _export_stitch(final: Path, settings: dict) -> dict:
    part_dir = final / 'riggable_parts_after_udf_cut'
    meshes = {name: _load_part(part_dir / (name + '.ply')) for name in PARTS}
    boundary = preflight(meshes)
    _json_atomic(final / 'riggable_after_udf_cut_STITCH_preflight.json', boundary)
    params = dict(
        max_seam_gap=float(settings.get('max_seam_gap', .025)),
        max_bridge_edge=float(settings.get('max_bridge_edge', .06)),
        max_perimeter_ratio=float(settings.get('max_perimeter_ratio', 1.65)),
    )
    target = final / 'riggable_after_udf_cut_STITCHED.ply'
    status_path = final / 'riggable_after_udf_cut_STITCHED_report.json'
    try:
        if not settings.get('enabled', True):
            raise RuntimeError('Stitching disabled by YAML')
        result, report = stitch_parts(meshes, **params)
        # No STITCHED.ply until geometry checks pass.
        with tempfile.NamedTemporaryFile(prefix='.stitch-', suffix='.ply', dir=final, delete=False) as f:
            scratch = Path(f.name)
        try:
            result.export(scratch)
            verify = trimesh.load(scratch, process=False, force='mesh')
            if len(verify.faces) != int(report['faces_after']):
                raise RuntimeError('STITCHED PLY export lost faces')
            os.replace(scratch, target)
        finally:
            scratch.unlink(missing_ok=True)
        report['status'] = 'STITCHED'
        report['source_cut_parts_dir'] = str(part_dir)
        report['stitched_ply'] = str(target)
        _json_atomic(status_path, report)
        print(f'[udf full 0084] STITCHED file={target} faces={len(result.faces)}', flush=True)
        return report
    except Exception as exc:
        report = {
            'status': 'STITCH_FAILED', 'reason': str(exc), 'params': params,
            'preflight': boundary, 'source_cut_parts_dir': str(part_dir),
            'stitched_ply': None,
            'note': 'Eight cut PLY parts and GLB were exported successfully. '
                    'No stitched file was written: attachment boundary repair is needed.'
        }
        _json_atomic(status_path, report)
        print(f'[udf full 0084] STITCH_FAILED: {exc}', file=sys.stderr, flush=True)
        return report


def execute(args, passthrough: list[str] | None = None) -> dict:
    yaml_path, config = _read_yaml(args.yaml)
    sdf, udf_stage, final = _locate_sdf_and_output(config, args.out, args.shape)
    stitch_cfg = config['udf_full_pipeline_0084'].get('stitch') or {}
    if not isinstance(stitch_cfg, dict):
        raise ValueError('udf_full_pipeline_0084.stitch must be a mapping')
    print('[udf full 0084] 1/3 inference and 0081 curve-local cut candidates', flush=True)
    start = time.monotonic()
    _run_inference(args, passthrough or [], yaml_path, udf_stage)
    print('[udf full 0084] 2/3 validate and assemble 8 riggable cut meshes', flush=True)
    promote(sdf_run=sdf, udf_run=udf_stage, out=final,
            tolerance=float(config['udf_full_pipeline_0084'].get('preview_vertex_tolerance', 2e-5)))
    fused_cfg = config.get('udf_fused_exterior_0092') or {}
    if not isinstance(fused_cfg, dict):
        raise ValueError('udf_fused_exterior_0092 must be a mapping')
    if fused_cfg.get('enabled', False):
        print('[udf fused 0092] 3/3 cut true SDF Boolean exterior using saved learned-UDF evidence; NO stitching', flush=True)
        report = generate_fused_cut(
            sdf_shape=sdf, udf_shape=udf_stage, out=final,
            distance_edge_factor=float(fused_cfg.get('distance_edge_factor', 1.75)),
            normal_dot=float(fused_cfg.get('normal_dot', .55)),
            allow_zero_seed_caps=bool(fused_cfg.get('zero_seed_cap_fallback', True)),
        )
    else:
        print('[udf full 0084] 3/3 stitch 7 root/leaf attachments', flush=True)
        report = _export_stitch(final, stitch_cfg)
    is_fused = bool(fused_cfg.get('enabled', False))
    summary = {
        'status': report['status'] if is_fused else ('COMPLETE' if report['status'] == 'STITCHED' else 'STITCH_FAILED'),
        'sdf_input': str(sdf), 'udf_inference_stage': str(udf_stage),
        'output': str(final), 'elapsed_s': round(time.monotonic() - start, 3),
        'glb': str(final / 'riggable_after_udf_cut.glb'),
        'riggable_parts': str(final / 'riggable_parts_after_udf_cut'),
        'unstitched_ply': str(final / 'riggable_after_udf_cut_UNSTITCHED_parts.ply'),
        'stitched_ply': None if is_fused else report.get('stitched_ply'),
        'stitch_report': None if is_fused else str(final / 'riggable_after_udf_cut_STITCHED_report.json'),
        'fused_sdf_cut_ply': report.get('output_ply') if is_fused else None,
        'fused_sdf_cut_report': str(final / 'riggable_fused_sdf_exterior_after_udf_cut_report.json') if is_fused else None,
    }
    _json_atomic(final / 'udf_sdf_cut_stitch_0084_report.json', summary)
    print('[udf full 0084] summary:', json.dumps(summary, indent=2), flush=True)
    # Keep the eight-part GLB manifest semantically separate from the stitched PLY.
    # The former contains independent riggable nodes even after a combined
    # stitched mesh is created alongside it.
    assembly_report_path = final / 'riggable_after_udf_cut_report.json'
    assembly_report = json.loads(assembly_report_path.read_text(encoding='utf8'))
    if is_fused:
        assembly_report['fused_sdf_cut_0092'] = {
            'status': report['status'], 'path': report['output_ply'],
            'method': report['method'], 'direct_fused_model_inference': False,
        }
    else:
        assembly_report['stitched_combined_ply'] = {
            'status': report['status'], 'path': report.get('stitched_ply'),
            'connected': report['status'] == 'STITCHED',
            'report': str(final / 'riggable_after_udf_cut_STITCHED_report.json'),
        }
    _json_atomic(assembly_report_path, assembly_report)
    if not is_fused and report['status'] != 'STITCHED' and stitch_cfg.get('require_success', True):
        raise RuntimeError('Real root/leaf stitch failed; see '
                           + str(final / 'riggable_after_udf_cut_STITCHED_report.json'))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', required=True, help='UDF model config')
    parser.add_argument('-o', '--out', required=True, help='NEW run name under inference/<model-size>')
    parser.add_argument('-s', '--shape', required=True, help='shape, e.g. oktopus_9_v1')
    parser.add_argument('-y', '--yaml', required=True, help='0084 UDF YAML under exp/train/manipulation')
    parser.add_argument('-r', '--resolution', type=int, default=512)
    parser.add_argument('-ckpt', '--checkpoint', default='final')
    parser.add_argument('--udf-far-value', type=float, default=.1)
    parser.add_argument('--udf-model-batch-size', type=int, default=32768)
    parser.add_argument('--udf-cut-query-batch-size', type=int, default=32768)
    return parser


def main():
    args, extra = build_parser().parse_known_args()
    execute(args, extra)


if __name__ == '__main__':
    main()
