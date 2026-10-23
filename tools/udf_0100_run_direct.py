#!/usr/bin/env python3
"""Experimental learned-UDF queries ON the existing fused SDF mesh.

Routes fused faces using the SDF's own riggable_parts + ownership manifest.
No previous UDF/0092 run is needed. Queries UDF on fused SDF geometry.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import uuid
import yaml

ROOT = Path(__file__).resolve().parents[1]
YAMLS = ROOT / 'exp/train/manipulation'


def parse_parts(value):
    """Preserve the runtime's 'all' sentinel; otherwise return part names."""
    raw = value.strip()
    if raw.lower() == 'all':
        return 'all'
    parts = [part.strip() for part in raw.split(',')]
    if not parts or any(not part or part.lower() == 'all' for part in parts):
        raise ValueError('Use --parts all or comma-separated part names (no empty entries)')
    if len(parts) != len(set(parts)):
        raise ValueError('Duplicate part name in --parts')
    return parts


def needs_nonbody_opt_in(parts):
    """Tentacle thresholds are not calibrated by the body-only 0100 trial."""
    return parts == 'all' or any(part != 'body' for part in parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sdf-run', required=True)
    ap.add_argument('--udf-run', default=None,
                    help='DEPRECATED and ignored: this tool no longer needs a previous UDF run')
    ap.add_argument('--out', required=True, help='NEW output run name')
    ap.add_argument('--shape', default='oktopus_9_v1')
    ap.add_argument('--model-size', default='4M')
    ap.add_argument('--resolution', type=int, default=1024)
    ap.add_argument('--yaml', default='oct22_adapt_oktopus_puffer_full_udf_fused_0092.yaml')
    ap.add_argument('--config', default='config/config_udf_grid_b13d3_oktopus_puffer.yaml')
    ap.add_argument('--parts', default='all',
                    help='Part names separated by commas, or all (default: all)')
    ap.add_argument('--ckpt', default='final')
    ap.add_argument('--face-batch-size', type=int, default=16384)
    ap.add_argument('--allow-uncalibrated-nonbody', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--skip-connected-cleanup', action='store_true',
                    help='Save direct classifications only; do not run connected unknown cleanup')
    args = ap.parse_args()
    try:
        selected_parts = parse_parts(args.parts)
    except ValueError as exc:
        ap.error(str(exc))
    if needs_nonbody_opt_in(selected_parts) and not args.allow_uncalibrated_nonbody:
        ap.error('Non-body 0100 UDF distance thresholds are not calibrated. '
                 'For an experimental all-parts preview, pass '
                 '--allow-uncalibrated-nonbody; or use --parts body.')
    for item in (args.sdf_run, args.out, args.shape):
        if not item or Path(item).name != item or item in ('.', '..'):
            ap.error('run names and shape must be single directory names')
    config = YAMLS / args.yaml
    if Path(args.yaml).name != args.yaml or not config.is_file():
        ap.error('YAML must be an existing filename directly in exp/train/manipulation')
    udf_cfg = yaml.safe_load(config.read_text(encoding='utf-8'))
    sdf = ROOT / 'inference' / args.model_size / args.sdf_run / args.shape
    dest = ROOT / 'inference' / args.model_size / args.out / args.shape
    if dest.exists() and not args.dry_run:
        ap.error('Output already exists; use a new --out')
    if not args.dry_run:
        for path in (sdf / 'riggable_fused_sdf_exterior.ply',
                     sdf / 'root_leaves_ownership.json', ROOT / args.config):
            if not path.is_file():
                ap.error(f'Required SDF/model input missing: {path}')
        if not (sdf / 'riggable_parts').is_dir():
            ap.error(f'Missing SDF riggable_parts: {sdf / "riggable_parts"}')
    if args.udf_run:
        print('[udf direct fused 0107] --udf-run is deprecated and ignored; '
              'no saved 0092 evidence is read')
    udf_cfg['udf_direct_fused_0100'] = {
        'enabled': True,
        'sdf_shape': str(sdf),
        'parts': selected_parts,
        'face_batch_size': args.face_batch_size,
        'allow_uncalibrated_nonbody': args.allow_uncalibrated_nonbody,
    }
    # Preserve original 15 adaptation entries and fingerprint. The runtime
    # flag has a separate classifier and never invokes the old 0081 pipeline.
    runtime = YAMLS / ('.udf_0100_' + uuid.uuid4().hex + '.yaml')
    cmd = [sys.executable, str(ROOT/'test_sdf_udf_cut.py'),
           '-c', args.config, '-o', args.out, '-s', args.shape,
           '-y', str(runtime), '-r', str(args.resolution),
           '-ckpt', args.ckpt, '--udf-far-value', '0.1',
           '--udf-model-batch-size', '32768', '--udf-cut-query-batch-size', '32768']
    print('[udf direct fused 0100] source:', sdf)
    print('[udf direct fused 0107] routing from SDF riggable_parts + root_leaves manifest')
    print('[udf direct fused 0100] query parts:', selected_parts)
    print('[udf direct fused 0100] new output:', dest / 'udf_direct_fused_0100')
    if args.dry_run:
        print('Command:', ' '.join(cmd))
        return
    try:
        runtime.write_text(yaml.safe_dump(udf_cfg, sort_keys=False), encoding='utf-8')
        subprocess.run(cmd, cwd=ROOT, check=True)
    finally:
        runtime.unlink(missing_ok=True)
    if not args.skip_connected_cleanup:
        # Direct UDF and connected-region cleanup are a single user-facing run.
        # Import the tested 0103 implementation: no second shell invocation.
        from udf_0103_connected_unknown_preview import run as connected_cleanup
        cleanup_dest = dest / 'udf_connected_final_0106'
        cleanup_args = argparse.Namespace(
            direct=str(dest / 'udf_direct_fused_0100'),
            out=str(cleanup_dest),
            parts=args.parts,
            allow_uncalibrated_nonbody=args.allow_uncalibrated_nonbody,
            min_area_fraction=0.005,
            min_invalid_fraction=0.60,
            min_removed_contact_fraction=0.80,
            min_removed_contact_edges=4,
            min_main_fraction=0.85,
            save_precleanup_preview=False,
        )
        cleanup_report = connected_cleanup(cleanup_args)
        print('[udf 0106] cleaned fused mesh:',
              cleanup_dest / 'fused_kept_connected_unknown_preview.ply')
        print('[udf 0106] cleanup:', cleanup_report['cleanup_largest_component'])


if __name__ == '__main__':
    main()
