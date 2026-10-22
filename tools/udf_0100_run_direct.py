#!/usr/bin/env python3
"""Experimental learned-UDF queries ON the existing fused SDF mesh.

Uses saved 0092 nearest-source-PART routing (never its cutting decisions).
Runs UDF model inference again on the fused mesh, not SDF inference.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import uuid
import yaml

ROOT = Path(__file__).resolve().parents[1]
YAMLS = ROOT / 'exp/train/manipulation'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sdf-run', required=True)
    ap.add_argument('--udf-run', required=True, help='Completed 0092 run: correspondence only')
    ap.add_argument('--out', required=True, help='NEW output run name')
    ap.add_argument('--shape', default='oktopus_9_v1')
    ap.add_argument('--model-size', default='4M')
    ap.add_argument('--resolution', type=int, default=1024)
    ap.add_argument('--yaml', default='oct22_adapt_oktopus_puffer_full_udf_fused_0092.yaml')
    ap.add_argument('--config', default='config/config_udf_grid_b13d3_oktopus_puffer.yaml')
    ap.add_argument('--parts', default='body', help='Comma-separated parts; start with body')
    ap.add_argument('--ckpt', default='final')
    ap.add_argument('--face-batch-size', type=int, default=16384)
    ap.add_argument('--allow-uncalibrated-nonbody', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    for item in (args.sdf_run, args.udf_run, args.out, args.shape):
        if not item or Path(item).name != item or item in ('.', '..'):
            ap.error('run names and shape must be single directory names')
    config = YAMLS / args.yaml
    if Path(args.yaml).name != args.yaml or not config.is_file():
        ap.error('YAML must be an existing filename directly in exp/train/manipulation')
    udf_cfg = yaml.safe_load(config.read_text(encoding='utf-8'))
    sdf = ROOT / 'inference' / args.model_size / args.sdf_run / args.shape
    prior = ROOT / 'inference' / args.model_size / args.udf_run / args.shape
    transfer = prior / 'fused_udf_transfer_evidence_0092.npz'
    dest = ROOT / 'inference' / args.model_size / args.out / args.shape
    if dest.exists() and not args.dry_run:
        ap.error('Output already exists; use a new --out')
    if not args.dry_run:
        # Reject stale cross-run correspondence. This metadata comes from
        # 0092 and identifies the actual fused SDF it was computed against.
        prior_report = prior / 'riggable_fused_sdf_exterior_after_udf_cut_report.json'
        if not prior_report.is_file():
            ap.error(f'Cannot verify correspondence without 0092 report: {prior_report}')
        import json
        actual = json.loads(prior_report.read_text(encoding='utf-8')).get('sdf_fused_input')
        if not actual or Path(actual).resolve() != (sdf / 'riggable_fused_sdf_exterior.ply').resolve():
            ap.error('0092 correspondence was generated for another SDF run; refusing')
        for path in (sdf / 'riggable_fused_sdf_exterior.ply',
                     sdf / 'root_leaves_ownership.json', transfer,
                     ROOT / args.config):
            if not path.is_file():
                ap.error(f'Required input missing: {path}')
    udf_cfg['udf_direct_fused_0100'] = {
        'enabled': True,
        'sdf_shape': str(sdf),
        'correspondence_0092': str(transfer),
        'correspondence_confirmed_same_sdf': True,
        'parts': args.parts.split(','),
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
    print('[udf direct fused 0100] routing only:', transfer)
    print('[udf direct fused 0100] query parts:', args.parts)
    print('[udf direct fused 0100] new output:', dest / 'udf_direct_fused_0100')
    if args.dry_run:
        print('Command:', ' '.join(cmd))
        return
    try:
        runtime.write_text(yaml.safe_dump(udf_cfg, sort_keys=False), encoding='utf-8')
        subprocess.run(cmd, cwd=ROOT, check=True)
    finally:
        runtime.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
