#!/usr/bin/env python3
"""Oct-22 SDF runner + separate existing puffer 0084 UDF-parts runner.

SDF: generic rigid, wrap, mixed. Part ownership and optional fused exterior
are defined by YAML root_leaves.parts and sdf_fused_exterior_0087.
UDF: only the existing puffer eight-parts 0084 implementation; not fused UDF.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

import yaml

ROOT = Path(__file__).resolve().parents[1]
YAML_DIR = ROOT / 'exp/train/manipulation'
PUFFER_UDF_PARTS = ('tentacle_t1', 'tentacle_t2', 'tentacle_t3', 'tentacle_t4',
         'tentacle_t5', 'tentacle_t6', 'tentacle_t8')


def full_path(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def infer_case_yaml(case, suffix):
    return f'oct22_adapt_{case}_{suffix}_optimization.yaml'


def config_for_case(case):
    if case == 'oktopus_puffer_full':
        return 'config/config_grid_b13d3_oktopus_puffer.yaml'
    if case.startswith('oktopus_boots_'):
        return 'config/config_grid_b13d3_oktopus_boots1.yaml'
    if case.startswith('oktopus_cone_'):
        return 'config/config_grid_b13d3_oktopus_cone.yaml'
    return None


def load_yaml(path):
    if path is None:
        raise ValueError('A YAML file is required')
    p = Path(path)
    if not p.is_absolute():
        p = YAML_DIR / p.name if p.parent == Path('.') else full_path(p)
    p = p.resolve()
    if not p.is_file():
        raise ValueError(f'YAML not found: {p}')
    if p.parent != YAML_DIR.resolve():
        raise ValueError(f'YAML must live directly in {YAML_DIR} (test_dc.py uses this location): {p}')
    d = yaml.safe_load(p.read_text(encoding='utf8'))
    if not isinstance(d, dict) or not isinstance(d.get('adaptations'), list) or not d['adaptations']:
        raise ValueError(f'Expected YAML mapping with nonempty adaptations: {p}')
    return p, d


def adaptation_modes(data):
    counts = {'rigid': 0, 'wrap': 0}
    for idx, a in enumerate(data['adaptations']):
        if not isinstance(a, dict):
            raise ValueError(f'adaptations[{idx}] must be a mapping')
        rigid, wrap = a.get('rigid_radius') is True, a.get('wrap_radius') is True
        if rigid == wrap:
            raise ValueError(f'adaptations[{idx}] target={a.get("target_key")} has '
                             'ambiguous radius mode. Set exactly one of '
                             'rigid_radius and wrap_radius true in its source configuration.')
        counts['rigid' if rigid else 'wrap'] += 1
    kind = 'mixed' if min(counts.values()) else 'rigid' if counts['rigid'] else 'wrap'
    return kind, counts


def root_leaf_is_full_puffer(data):
    plan = data.get('root_leaves') or {}
    return bool(data.get('combine_adaptations') is True and
                isinstance(plan, dict) and
                set((plan.get('leaves') or {}).keys()) == set(PUFFER_UDF_PARTS) and
                len(data['adaptations']) == 15 and
                data['adaptations'][0].get('target_key') == 'oktopus_9_v1|body')


def check_basics(args):
    if not args.out or Path(args.out).name != args.out or args.out in {'.', '..'}:
        raise ValueError('--out must be a simple new run name (no path separators)')
    if Path(args.shape).name != args.shape:
        raise ValueError('--shape must be a simple name')
    if args.resolution < 16:
        raise ValueError('resolution must be >= 16')



def verify_fused_yaml_forwarding(source_path):
    """Fail early if action_part_adapt would silently drop the 0087 YAML flag."""
    source_path = Path(source_path)
    if not source_path.is_file():
        raise ValueError(f'SDF implementation missing: {source_path}')
    source = source_path.read_text(encoding='utf8')
    start = 'for config_key, config_value in raw_config.items():'
    stop = 'adaptation_items = raw_config.get('
    if start not in source or stop not in source.split(start, 1)[1]:
        raise ValueError('Cannot verify 0087 SDF YAML forwarding; inspect agent_3dvec_sdf.py')
    forwarding = source.split(start, 1)[1].split(stop, 1)[0]
    if "'sdf_fused_exterior_0087'" not in forwarding and '"sdf_fused_exterior_0087"' not in forwarding:
        raise ValueError('0087 fused YAML flag is NOT forwarded by agent_3dvec_sdf.py; '
                         'apply the 0089 fusion-activation fix before inference')


def sdf(args):
    check_basics(args)
    filename = args.yaml or infer_case_yaml(args.case, 'sdf')
    path, data = load_yaml(filename)
    kind, counts = adaptation_modes(data)
    config = args.config or config_for_case(args.case)
    if not config:
        raise ValueError('Choose --config config/<matching-model>.yaml for this experiment')
    if not args.dry_run and not full_path(config).is_file():
        raise ValueError(f'Model config not found: {config}')
    fused = data.get('sdf_fused_exterior_0087') is True
    targets = {str(a.get('target_key', '')).split('|')[0] for a in data['adaptations']}
    if targets != {args.shape}:
        raise ValueError(f'YAML target avatar(s) {sorted(targets)} differ from --shape {args.shape!r}; '
                         f'use --shape {next(iter(targets))!r} with a matching model --config')
    combine = data.get('combine_adaptations') is True
    has_root = bool(data.get('root_leaves'))
    if has_root and not combine:
        raise ValueError('root_leaves requires combine_adaptations: true')
    if has_root:
        # Validate part assignment and blend-group ownership before GPU inference.
        app = str(ROOT / 'src' / 'app')
        if app not in sys.path:
            sys.path.insert(0, app)
        from root_leaf_combination import build_root_leaf_plan
        plan = build_root_leaf_plan(data['adaptations'], data['root_leaves'])
        print(f'[oct22 SDF] YAML-owned parts={list(plan.part_names)} '
              f'attachments={plan.attachments or {}}', flush=True)
    if fused:
        # 0089: preflight the actual action's YAML allowlist, not just YAML.
        verify_fused_yaml_forwarding(ROOT / 'src/app/agent_3dvec_sdf.py')
    if fused and not args.dry_run and not (ROOT / 'src/app/sdf_fused_report_0087.py').is_file():
        raise ValueError('0087 SDF code not installed: apply the 0087 patch first')
    if args.require_fused and not fused:
        raise ValueError('Requested fused exterior, but YAML has no sdf_fused_exterior_0087: true')
    if args.stream and (fused or combine or has_root or kind == 'mixed'):
        raise ValueError('--stream only offered for independent rigid pieces; not for fused/combined SDF')
    if args.stream and kind != 'rigid':
        raise ValueError('--stream currently restricted to independent rigid pieces')
    output = ROOT / 'inference' / args.model_size / args.out / args.shape
    if output.exists() and not args.dry_run:
        raise ValueError(f'Output already exists: {output}. Use a fresh --out name.')
    repo = full_path(args.dcsdd_repo).resolve()
    if not args.dry_run and not repo.is_dir():
        raise ValueError(f'DCSDD repository is not available: {repo}; provide --dcsdd-repo /path/to/dcsdd')
    scratch = full_path(args.memmap_dir).resolve()
    if not args.dry_run:
        scratch.mkdir(parents=True, exist_ok=True)
    print(f'[oct22 SDF] mode={kind} counts={counts} combined={combine} root_leaves={has_root} '
          f'true_fused_exterior={fused}', flush=True)
    if kind == 'mixed' and not combine:
        print('[oct22 SDF] NOTE: mixed rigid/wrap does NOT mean fused/merged. '
              'Your YAML has combine_adaptations unset/false.', flush=True)
    if fused:
        print('[oct22 SDF] 0087 scalar-field Boolean union enabled; '
              'YAML-defined separate parts remain when root_leaves is present.', flush=True)
    cmd = [sys.executable, str(ROOT / 'test_dc.py'), '-c', config, '-o', args.out,
           '-s', args.shape, '-y', path.name, '-r', str(args.resolution),
           '-ckpt', args.checkpoint, '-a', 'sdf_dc',
           '--dcsdd-repo', str(repo), '--dcsdd-methods', 'ours',
           '--dcsdd-outer-iters', str(args.outer_iters), '--dcsdd-inner-iters', str(args.inner_iters),
           '--dcsdd-mu', '0.5', '--dcsdd-dc-weight', '0.4', '--dcsdd-verbose',
           '--sdf-grid-memmap-dir', str(scratch), '--dcsdd-crop-grid',
           '--dcsdd-fast-bbox-marking', '--dcsdd-bounded-filtering',
           '--dcsdd-filter-batch-size', '262144', '--sdf-localize-after-gate',
           '--surface-extraction', 'ours']
    if args.stream:
        cmd += ['--dcsdd-compact-staging', '--dcsdd-stream-inference',
                '--dcsdd-stream-crop-first', '--dcsdd-stream-replay-metadata',
                '--dcsdd-stream-one-pass', '--dcsdd-stream-replay-dir', str(scratch)]
    print('$ ' + shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)
    if fused:
        mesh = output / 'riggable_fused_sdf_exterior.ply'
        report = output / 'riggable_fused_sdf_exterior_report.json'
        if not mesh.is_file() or not report.is_file():
            raise RuntimeError(f'0087 fused output missing. Expected {mesh} and {report}')
        print(f'[oct22 SDF] FUSED EXTERIOR: {mesh}', flush=True)
        print(f'[oct22 SDF] FUSION REPORT: {report}', flush=True)
        if has_root:
            print(f'[oct22 SDF] RIGGABLE PARTS: {output / "riggable_parts"}', flush=True)
    else:
        print(f'[oct22 SDF] output: {output}', flush=True)


def udf_parts(args):
    check_basics(args)
    if args.case != 'oktopus_puffer_full':
        raise ValueError('0084 UDF parts+stitch currently supports only the full puffer, '
                         'not arbitrary rigid/wrap/other mixed experiments')
    filename = args.yaml or infer_case_yaml(args.case, 'udf')
    path, data = load_yaml(filename)
    if not root_leaf_is_full_puffer(data) or not data.get('udf_full_pipeline_0084', {}).get('enabled'):
        raise ValueError('UDF YAML is not the full-puffer 0084 configuration')
    if not data.get('udf_cut_curve_local_preview_0081'):
        raise ValueError('UDF YAML requires curve-local 0081 enabled')
    config = args.config or 'config/config_udf_grid_b13d3_oktopus_puffer.yaml'
    run = args.sdf_run or f'puffer_oct22_fused_sdf_r{args.resolution}'
    if Path(run).name != run or run in ('.', '..'):
        raise ValueError('--sdf-run must be a single SDF run directory name')
    shape_base = ROOT / 'inference' / args.model_size / run / args.shape
    parts = shape_base / 'riggable_parts'
    if not args.dry_run and not parts.is_dir():
        raise ValueError(f'No SDF riggable parts in {parts}')
    if not args.dry_run and not (shape_base / 'root_leaves_ownership.json').is_file():
        raise ValueError(f'Missing ownership manifest in {shape_base}')
    if not args.dry_run and not full_path(config).is_file():
        raise ValueError(f'Missing UDF model config: {config}')
    if not args.dry_run:
        # Verify both imports now rather than wasting a 70+ second UDF pass.
        cmd = [sys.executable, '-c',
               'import tools; print("tools package", tools.__file__); '
               'from tools.promote_udf_0081_assembly import promote; '
               'from tools.udf_stitch_0084 import stitch_parts; '
               'print("0084 imports OK")']
        subprocess.run(cmd, cwd=ROOT, check=True)
    if data.get('udf_cut_root_leaves_mesh_dir') == str(parts.relative_to(ROOT)):
        runtime = path
        cleanup = False
    else:
        runtime = YAML_DIR / f'.oct22_udf_runtime_{uuid.uuid4().hex}.yaml'
        cleanup = True
        if not args.dry_run:
            edited = copy.deepcopy(data)
            edited['udf_cut_root_leaves_mesh_dir'] = str(parts.relative_to(ROOT))
            runtime.write_text(yaml.safe_dump(edited, sort_keys=False), encoding='utf8')
    print('[oct22 UDF] Input:', parts, flush=True)
    if (data.get('udf_fused_exterior_0092') or {}).get('enabled', False):
        if not args.dry_run and not (shape_base / 'riggable_fused_sdf_exterior.ply').is_file():
            raise ValueError('0092 requires the true 0087 fused SDF exterior PLY')
        print('[oct22 UDF] 0092 enabled: eight-part learned evidence -> cut TRUE fused SDF; no loop stitching.', flush=True)
    else:
        print('[oct22 UDF] 0084 legacy: cuts eight parts and attempts boundary stitching; not SDF Boolean.', flush=True)
    cmd = [sys.executable, str(ROOT / 'test_udf_sdf_cut_stitch.py'), '-c', config,
           '-o', args.out, '-s', args.shape, '-y', str(runtime if not args.dry_run else path),
           '-r', str(args.resolution), '-ckpt', args.checkpoint,
           '--udf-far-value', '0.1', '--udf-model-batch-size', '32768',
           '--udf-cut-query-batch-size', '32768']
    print('$ ' + shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    try:
        subprocess.run(cmd, cwd=ROOT, check=True)
    finally:
        if cleanup:
            runtime.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('stage', choices=('sdf', 'udf-parts'))
    p.add_argument('--case', default='oktopus_puffer_full', help='e.g. oktopus_puffer_full, oktopus_boots_t7_boot1_only, oktopus_cone_post_sdf, trex2_puffer')
    p.add_argument('--yaml', help='YAML filename in exp/train/manipulation (stage-specific)')
    p.add_argument('--config', help='model config YAML; defaults provided for puffer/boots/cone')
    p.add_argument('--shape', default='oktopus_9_v1')
    p.add_argument('--resolution', type=int, default=512)
    p.add_argument('--out', help='new output run directory name; auto-named when omitted')
    p.add_argument('--sdf-run', help='existing SDF run for udf-parts stage')
    p.add_argument('--model-size', default='4M')
    p.add_argument('--checkpoint', default='final')
    p.add_argument('--dcsdd-repo', default='third_party/dcsdd')
    p.add_argument('--memmap-dir', default='logs')
    p.add_argument('--outer-iters', type=int, default=5)
    p.add_argument('--inner-iters', type=int, default=3)
    p.add_argument('--require-fused', action='store_true', help='fail SDF run if YAML does not enable 0087 true fusion')
    p.add_argument('--stream', action='store_true', help='rigid independent DCSDD streaming; NOT mixed or fused')
    p.add_argument('--dry-run', action='store_true', help='validate YAML and print command without inference')
    args = p.parse_args()
    if not args.out:
        prefix = 'puffer_oct22_fused_sdf' if args.stage == 'sdf' and args.case == 'oktopus_puffer_full' else (args.case + ('_sdf' if args.stage == 'sdf' else '_udf_parts'))
        args.out = f'{prefix}_r{args.resolution}'
    try:
        if args.stage == 'sdf':
            sdf(args)
        else:
            udf_parts(args)
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as e:
        print(f'[oct22 ERROR] {e}', file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == '__main__':
    main()
