#!/usr/bin/env python3
"""Read-only side-by-side diagnosis of two saved 0081 UDF selections.

Compares source-face geometry, curve arclength, trusted seeds, candidate faces,
selected faces, and blocked candidates. No UDF/SDF calls and no mesh edits.

Examples:
 python tools/udf_0081_compare_runs.py --discover inference/4M
 python tools/udf_0081_compare_runs.py \
   --old inference/4M/puffer_0081_curve_local_preview_r512 \
   --new inference/4M/adapt_oktopus_puffer_udf_r1024 \
   --out /tmp/puffer_0081_compare.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PARTS = ('body', 'tentacle_t1', 'tentacle_t2', 'tentacle_t3',
         'tentacle_t4', 'tentacle_t5', 'tentacle_t6', 'tentacle_t8')
BINS = np.linspace(0.0, 1.0, 21)
MASKS = ('seed', 'candidate', 'selected', 'blocked')


def find_locality(path: Path, shape: str) -> Path:
    """Allow a run, shape, 0084 wrapper, or 0081 diagnostics directory."""
    p = Path(path).expanduser()
    candidates = (
        p,
        p / 'udf_cut_curve_local_0081',
        p / shape / 'udf_cut_curve_local_0081',
        p / '_udf_inference_0084' / shape / 'udf_cut_curve_local_0081',
    )
    for c in candidates:
        if (c / 'body' / 'locality.npz').is_file():
            return c.resolve()
    raise FileNotFoundError(
        f'0081 evidence not found for {p}. Expected body/locality.npz '
        f'in a shape folder or _udf_inference_0084/{shape}/udf_cut_curve_local_0081. '
        'Run --discover inference/4M to find existing evidence.')


def discover_runs(root: Path, shape: str) -> list[str]:
    """List only named runs with 0081 evidence; no modification."""
    root = Path(root)
    found = []
    if not root.is_dir():
        raise FileNotFoundError(root)
    for run in sorted(root.iterdir()):
        if not run.is_dir():
            continue
        try:
            loc = find_locality(run, shape)
        except FileNotFoundError:
            continue
        found.append(f'{run}  ->  {loc}')
    return found


def read_part(locality: Path, part: str) -> dict | None:
    p = locality / part / 'locality.npz'
    if not p.is_file():
        return None
    with np.load(p, allow_pickle=False) as z:
        required = ('face_curve_arclength', 'curve_length_world', 'seed_faces',
                    'candidate_faces', 'selected_faces', 'blocked_faces',
                    'face_centroids_xyz')
        absent = [x for x in required if x not in z]
        if absent:
            raise ValueError(f'{p}: missing {absent}')
        result = {
            'arc': np.asarray(z['face_curve_arclength'], dtype=np.float64),
            'curve_length_world': float(z['curve_length_world']),
            'seed': np.asarray(z['seed_faces'], dtype=bool),
            'candidate': np.asarray(z['candidate_faces'], dtype=bool),
            'selected': np.asarray(z['selected_faces'], dtype=bool),
            'blocked': np.asarray(z['blocked_faces'], dtype=bool),
            'centroids': np.asarray(z['face_centroids_xyz'], dtype=np.float64),
        }
    n = len(result['arc'])
    if (n == 0 or result['centroids'].shape != (n, 3) or
        any(len(result[k]) != n for k in MASKS) or
        not np.isfinite(result['arc']).all() or
        not np.isfinite(result['centroids']).all() or
        result['curve_length_world'] <= 0):
        raise ValueError(f'{p}: invalid lengths, projection, or positions')
    if (np.any(result['seed'] & ~result['candidate']) or
        np.any(result['selected'] & ~result['candidate']) or
        np.any(result['seed'] & ~result['selected']) or
        np.any(result['blocked'] != (result['candidate'] & ~result['selected']))):
        raise ValueError(f'{p}: inconsistent 0081 masks (selection may have changed)')
    return result


def _stats_for_interval(d: dict, mask: np.ndarray) -> dict:
    n = int(mask.sum())
    row = {'faces': n}
    for name in MASKS:
        row[name] = int(np.count_nonzero(d[name] & mask))
    row['blocked_per_candidate_pct'] = round(100 * row['blocked'] / row['candidate'], 3) if row['candidate'] else None
    return row


def summary(d: dict) -> dict:
    s = d['arc']
    all_faces = np.ones(len(s), bool)
    selection = d['selected']
    seed_s = s[d['seed']]
    centers = d['centroids']
    bounds_low = np.min(centers, axis=0)
    bounds_high = np.max(centers, axis=0)
    bins = []
    for i in range(20):
        in_bin = (s >= BINS[i]) & ((s < BINS[i + 1]) if i != 19 else (s <= BINS[i + 1]))
        bins.append({'arc_interval': [round(float(BINS[i]), 3), round(float(BINS[i + 1]), 3)],
                     **_stats_for_interval(d, in_bin)})
    return {
        'n_faces': len(s),
        'curve_length_world': round(d['curve_length_world'], 9),
        'centroid_bbox_min': bounds_low.round(7).tolist(),
        'centroid_bbox_max': bounds_high.round(7).tolist(),
        'centroid_bbox_diagonal': round(float(np.linalg.norm(bounds_high - bounds_low)), 7),
        'seed_arc_min_max': [round(float(np.min(seed_s)), 6), round(float(np.max(seed_s)), 6)] if len(seed_s) else None,
        'seed_arc_quantiles_02_50_98': np.quantile(seed_s, [.02, .5, .98]).round(6).tolist() if len(seed_s) else None,
        'total': _stats_for_interval(d, all_faces),
        'terminal_low_s_le_012': _stats_for_interval(d, s <= .12),
        'terminal_high_s_ge_088': _stats_for_interval(d, s >= .88),
        'cap_high_s_ge_089': _stats_for_interval(d, s >= .89),
        'cap_high_s_ge_090': _stats_for_interval(d, s >= .90),
        'bins': bins,
    }


def spatial_difference(a: dict, b: dict, max_centroids: int = 5000) -> dict:
    """Approximate surface disparity, resolution independent (not face identity)."""
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return {'status': 'skipped_scipy_not_available'}
    def sample(c):
        ids = np.linspace(0, len(c) - 1, min(max_centroids, len(c)), dtype=np.int64)
        return c[ids]
    pa, pb = sample(a['centroids']), sample(b['centroids'])
    ab = cKDTree(pb).query(pa, k=1)[0]
    ba = cKDTree(pa).query(pb, k=1)[0]
    a_box = np.ptp(a['centroids'], axis=0)
    b_box = np.ptp(b['centroids'], axis=0)
    extent = max(float(np.linalg.norm(a_box)), float(np.linalg.norm(b_box)), 1e-12)
    return {
        'method': 'sampled_face_centroids_nearest_neighbor_nonregistration',
        'sampled_old': len(pa), 'sampled_new': len(pb),
        'old_to_new_median_p95_world': np.quantile(ab, [.5, .95]).round(7).tolist(),
        'new_to_old_median_p95_world': np.quantile(ba, [.5, .95]).round(7).tolist(),
        'max_p95_relative_to_bbox_pct': round(100 * max(float(np.quantile(ab, .95)), float(np.quantile(ba, .95))) / extent, 4),
        'warning': 'Samples of face centroids are approximate, not a registration or exact surface-distance test.',
    }


def compare(old: Path, new: Path, *, parts=PARTS, max_centroids=5000) -> dict:
    result = {'old_evidence': str(old), 'new_evidence': str(new),
              'method': 'saved 0081 masks; no inference or cutting', 'parts': {}}
    for part in parts:
        a = read_part(old, part)
        b = read_part(new, part)
        if a is None or b is None:
            result['parts'][part] = {'status': 'missing_locality',
                                     'old_exists': a is not None, 'new_exists': b is not None}
            continue
        aa, bb = summary(a), summary(b)
        result['parts'][part] = {
            'old': aa, 'new': bb,
            'relative': {
                'face_count_ratio_new_old': round(bb['n_faces'] / aa['n_faces'], 4),
                'curve_length_ratio_new_old': round(bb['curve_length_world'] / aa['curve_length_world'], 5),
                'old_selected_per_candidate_pct': round(100 * aa['total']['selected'] / max(aa['total']['candidate'], 1), 3),
                'new_selected_per_candidate_pct': round(100 * bb['total']['selected'] / max(bb['total']['candidate'], 1), 3),
                'old_high_cap_blocked_per_candidate_pct': aa['cap_high_s_ge_089']['blocked_per_candidate_pct'],
                'new_high_cap_blocked_per_candidate_pct': bb['cap_high_s_ge_089']['blocked_per_candidate_pct'],
            },
            'spatial_difference': spatial_difference(a, b, max_centroids=max_centroids),
        }
    return result


def print_comparison(report: dict) -> None:
    print('0081 OFFLINE RUN COMPARISON (saved evidence only)')
    print('old:', report['old_evidence'])
    print('new:', report['new_evidence'])
    print('part               old faces  new faces    old seeds  new seeds  old sel/cand  new sel/cand  old high blocked  new high blocked')
    for part, info in report['parts'].items():
        if info.get('status') == 'missing_locality':
            print(f'{part:<18} MISSING OLD={info["old_exists"]} NEW={info["new_exists"]}')
            continue
        a, b, r = info['old'], info['new'], info['relative']
        pct = lambda x: 'n/a' if x is None else f'{x:.1f}%'
        print(f'{part:<18} {a["n_faces"]:>9,} {b["n_faces"]:>9,} {a["total"]["seed"]:>11,} '
              f'{b["total"]["seed"]:>10,} {r["old_selected_per_candidate_pct"]:>10.1f}% '
              f'{r["new_selected_per_candidate_pct"]:>10.1f}% '
              f'{pct(r["old_high_cap_blocked_per_candidate_pct"]):>17} '
              f'{pct(r["new_high_cap_blocked_per_candidate_pct"]):>17}')
    if 'body' not in report['parts'] or 'old' not in report['parts']['body']:
        return
    d = report['parts']['body']
    print('\nBODY: high-arc end (s >= 0.89)')
    for key in ('old', 'new'):
        s = d[key]
        print(f'{key}: curve_length={s["curve_length_world"]:.6f} '
              f'seed_s_range={s["seed_arc_min_max"]} high_arc={s["cap_high_s_ge_089"]}')
    print('\nBODY: curve arclength bins (counts are resolution-dependent; compare ratios)')
    print('range       old candidate/seed/selected/blocked       new candidate/seed/selected/blocked')
    for a, b in zip(d['old']['bins'], d['new']['bins']):
        n = lambda r: f'{r["candidate"]:>7}/{r["seed"]:>6}/{r["selected"]:>7}/{r["blocked"]:>7}'
        print(f'{a["arc_interval"][0]:.2f}-{a["arc_interval"][1]:.2f}     {n(a)}      {n(b)}')
    print('\nSpatial comparison:', d['spatial_difference'])
    print('\nInterpretation rule: if old high-cap seeds were present but new seeds are absent, '
          'investigate evidence/projection and source geometry first. If both have seeds but '
          'selection differs, investigate locality. Compare body source meshes before modifying 0081.')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--old', type=Path, help='Working 0081 run directory (512 or otherwise)')
    ap.add_argument('--new', type=Path, help='Failing 0081/0084 run directory')
    ap.add_argument('--discover', type=Path, help='Find existing runs containing saved 0081 localities')
    ap.add_argument('--shape', default='oktopus_9_v1')
    ap.add_argument('--out', type=Path, help='Optional JSON report path (only file written)')
    ap.add_argument('--max-centroids', type=int, default=5000)
    args = ap.parse_args(argv)
    if args.discover:
        choices = discover_runs(args.discover, args.shape)
        print('\n'.join(choices) if choices else 'No saved 0081 locality.npz found under named runs.')
        return 0
    if not args.old or not args.new:
        ap.error('--old and --new are required unless --discover is supplied')
    if args.max_centroids < 100:
        ap.error('--max-centroids must be >= 100')
    report = compare(find_locality(args.old, args.shape),
                     find_locality(args.new, args.shape), max_centroids=args.max_centroids)
    print_comparison(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding='utf8')
        print('Saved JSON:', args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
