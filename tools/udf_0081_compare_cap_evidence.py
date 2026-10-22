#!/usr/bin/env python3
"""Read-only comparison of 0074/0079 evidence behind 0081 body cap seeds.

Requires saved UDF diagnostics only, never reloads models or changes geometry.
Example:
    python tools/udf_0081_compare_cap_evidence.py \
      --old inference/4M/puffer_0081_curve_local_preview_r512 \
      --new inference/4M/adapt_oktopus_puffer_udf_r1024
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from udf_0081_compare_runs import find_locality

CATEGORY_NAMES = (
    'not_assigned', 'assigned_no_raw_udf', 'raw_valid_abstained',
    'effective_no_fused', 'below_grow', 'grow_only',
    'seed_not_removed', 'removed',
)
WINDOWS = ((.85, .89), (.89, .90), (.90, .95), (.89, .95), (.40, .75))


def load_evidence(run: Path, shape: str, part: str = 'body'):
    local_dir = find_locality(run, shape)
    run_base = local_dir.parent
    evidence_path = run_base / 'udf_cut_evidence_0074' / part / 'face_evidence.npz'
    votes_path = run_base / 'udf_cut_reconcile_0079' / part / 'votes.npz'
    if not evidence_path.is_file() or not votes_path.is_file():
        raise FileNotFoundError(
            f'Missing saved 0074 or 0079 evidence in {run_base};\n'
            f'  expected {evidence_path}\n  and {votes_path}\n'
            'These files are created with udf_cut_evidence_0074=true and '
            'udf_cut_reconcile_0079=true.')
    with np.load(local_dir / part / 'locality.npz', allow_pickle=False) as z:
        s = np.asarray(z['face_curve_arclength'], dtype=float)
        seeds = np.asarray(z['seed_faces'], dtype=bool)
        candidate = np.asarray(z['candidate_faces'], dtype=bool)
        selected = np.asarray(z['selected_faces'], dtype=bool)
    with np.load(evidence_path, allow_pickle=False) as z:
        category = np.asarray(z['face_category'], dtype=np.uint8)
        assignment_count = np.asarray(z['assigned_owners_count'])
        raw_count = np.asarray(z['raw_valid_owners_count'])
        effective_count = np.asarray(z['effective_valid_owners_count'])
        owners = {}
        for k in z.files:
            if k.startswith('item_') and k.endswith('_faces'):
                prefix = k[:-len('_faces')]
                for suffix in ('_raw_scores', '_raw_valid', '_effective_valid'):
                    if prefix + suffix not in z:
                        raise ValueError(f'Missing {prefix+suffix} in {evidence_path}')
                owners[prefix] = (
                    np.asarray(z[k], dtype=np.int64),
                    np.asarray(z[prefix + '_raw_scores'], dtype=np.float64),
                    np.asarray(z[prefix + '_raw_valid'], dtype=bool),
                    np.asarray(z[prefix + '_effective_valid'], dtype=bool),
                )
    with np.load(votes_path, allow_pickle=False) as z:
        vote_names = ('assigned', 'valid', 'remove_votes', 'keep_votes',
                      'class_agreed_remove', 'class_missing_evidence',
                      'class_kept_by_owner', 'class_keep_conflict')
        votes = {k: np.asarray(z[k]) for k in vote_names if k in z.files}
    n = len(s)
    for name, value in [('seed', seeds), ('candidate', candidate), ('selected', selected),
                        ('category', category), ('assignments', assignment_count),
                        ('raw_count', raw_count), ('effective_count', effective_count),
                        *votes.items()]:
        if len(value) != n:
            raise ValueError(f'{name} length mismatch for {run}')
    for name, (ids, scores, raw_valid, effective) in owners.items():
        if (np.any(ids >= n) or np.any(ids < 0) or
                not (len(ids) == len(scores) == len(raw_valid) == len(effective))):
            raise ValueError(f'{name} has invalid ownership arrays for {run}')
    if np.any(seeds & ~candidate) or np.any(seeds & ~selected):
        raise ValueError(f'Inconsistent 0081 masks in {run}')
    return dict(s=s, seeds=seeds, candidates=candidate, selected=selected,
                category=category, assigned=assignment_count, raw=raw_count,
                effective=effective_count, votes=votes, owners=owners)


def _count(mask):
    return int(np.count_nonzero(mask))


def summarize_region(d, lo: float, hi: float):
    mask = (d['s'] >= lo) & (d['s'] < hi)
    row = {'faces': _count(mask), 'seed': _count(mask & d['seeds']),
           'candidate': _count(mask & d['candidates']),
           'selected': _count(mask & d['selected']),
           'assigned': _count(mask & (d['assigned'] > 0)),
           'raw_valid_face': _count(mask & (d['raw'] > 0)),
           'effective_face': _count(mask & (d['effective'] > 0)),
           'categories': {name: _count(mask & (d['category'] == i))
                          for i, name in enumerate(CATEGORY_NAMES)}}
    row['votes'] = {key: _count(mask & (value > 0))
                    for key, value in d['votes'].items()}
    row['owners'] = {}
    for name, (ids, scores, raw_valid, effective) in d['owners'].items():
        sel = mask[ids]
        effective_sel = sel & effective & np.isfinite(scores)
        vals = scores[effective_sel]
        row['owners'][name] = {
            'assigned': _count(sel), 'raw_valid': _count(sel & raw_valid),
            'effective_valid': _count(effective_sel),
            'score_quantiles_p50_p90_p99': [float(x) for x in np.quantile(vals, [.5, .9, .99])] if len(vals) else None,
            'scores_ge_0p02': _count(effective_sel & (scores >= .02)),
            'scores_ge_0p01': _count(effective_sel & (scores >= .01)),
        }
    return row


def _format_counts(row):
    return (f"faces={row['faces']:,} candidate={row['candidate']:,} "
            f"seed={row['seed']:,} selected={row['selected']:,} "
            f"assigned={row['assigned']:,} raw_valid={row['raw_valid_face']:,} "
            f"effective={row['effective_face']:,}")


def print_comparison(old, new):
    print('0074/0079 EVIDENCE → 0081 CAP-SEED REGRESSION (saved data only)')
    print('The score thresholds shown below (0.02/0.01) are the default body YAML values;')
    print('verify the actual body thresholds in each UDF run log/config if overridden.')
    for lo, hi in WINDOWS:
        print(f'\nARC s=[{lo:.2f},{hi:.2f})')
        for label, d in [('old', old), ('new', new)]:
            row = summarize_region(d, lo, hi)
            print(f' {label} {_format_counts(row)}')
            print('   category:', ', '.join(f'{k}={v:,}' for k, v in row['categories'].items() if v))
            print('   votes:', ', '.join(f'{k}={v:,}' for k, v in row['votes'].items() if v))
            for owner, stats in row['owners'].items():
                if stats['assigned']:
                    print(f'   {owner}: assigned={stats["assigned"]:,} '
                          f'raw={stats["raw_valid"]:,} effective={stats["effective_valid"]:,} '
                          f'>=.02={stats["scores_ge_0p02"]:,} '
                          f'>=.01={stats["scores_ge_0p01"]:,} '
                          f'p50/p90/p99={stats["score_quantiles_p50_p90_p99"]}')
    print('\nDIAGNOSIS GUIDE for s=[.85,.90):')
    print('  Old agreed-remove seeds but new raw_valid=0: localization/UDF query validity changed.')
    print('  New valid scores but no scores>=seed: evidence/model/threshold or surface position changed.')
    print('  New scores>=seed but no agreed_remove: inspect 0079 connected-component rejection/votes.')
    print('  New agreed_remove present but no 0081 seeds: face alignment or saved-mask inconsistency.')
    print('  This alone does not prove why the learned UDF values differ: compare the SDF part surfaces if needed.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', required=True, type=Path)
    parser.add_argument('--new', required=True, type=Path)
    parser.add_argument('--shape', default='oktopus_9_v1')
    args = parser.parse_args(argv)
    print_comparison(load_evidence(args.old, args.shape), load_evidence(args.new, args.shape))


if __name__ == '__main__':
    main()
