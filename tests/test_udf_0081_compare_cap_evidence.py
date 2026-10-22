"""Regression tests for read-only saved UDF body cap evidence comparison."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from udf_0081_compare_cap_evidence import load_evidence, summarize_region


def _fixture(root: Path, *, raw_valid: bool, seed: bool):
    base = root / '_udf_inference_0084' / 'oktopus_9_v1'
    locality = base / 'udf_cut_curve_local_0081' / 'body'
    evidence = base / 'udf_cut_evidence_0074' / 'body'
    votes = base / 'udf_cut_reconcile_0079' / 'body'
    for p in [locality, evidence, votes]:
        p.mkdir(parents=True)
    s = np.asarray([.4, .87, .895, .901], dtype='float32')
    seeds = np.asarray([True, False, seed, False])
    candidate = np.asarray([True, True, True, True])
    np.savez_compressed(locality / 'locality.npz', face_curve_arclength=s, seed_faces=seeds,
                        candidate_faces=candidate, selected_faces=seeds)
    cat = np.asarray([7, 1, 7 if seed else (6 if raw_valid else 1), 1], dtype='uint8')
    np.savez_compressed(evidence / 'face_evidence.npz', face_category=cat,
                        assigned_owners_count=np.ones(4), raw_valid_owners_count=np.asarray([1, 0, int(raw_valid), 0]),
                        effective_valid_owners_count=np.asarray([1, 0, int(raw_valid), 0]),
                        item_00_faces=np.arange(4),
                        item_00_raw_scores=np.asarray([.03, .1, .03, .1]),
                        item_00_raw_valid=np.asarray([True, False, raw_valid, False]),
                        item_00_effective_valid=np.asarray([True, False, raw_valid, False]))
    np.savez_compressed(votes / 'votes.npz', class_agreed_remove=seeds,
                        class_missing_evidence=np.asarray([False, True, not raw_valid, True]),
                        assigned=np.ones(4), valid=np.asarray([1, 0, int(raw_valid), 0]))
    return root


def test_detects_seed_loss_because_raw_invalid(tmp_path):
    old = load_evidence(_fixture(tmp_path / 'old', raw_valid=True, seed=True), 'oktopus_9_v1')
    new = load_evidence(_fixture(tmp_path / 'new', raw_valid=False, seed=False), 'oktopus_9_v1')
    a, b = summarize_region(old, .89, .9), summarize_region(new, .89, .9)
    assert a['seed'] == 1 and b['seed'] == 0
    assert a['owners']['item_00']['raw_valid'] == 1
    assert b['owners']['item_00']['raw_valid'] == 0
    assert a['votes']['class_agreed_remove'] == 1


def test_detects_seed_loss_despite_valid_score(tmp_path):
    new = load_evidence(_fixture(tmp_path / 'new', raw_valid=True, seed=False), 'oktopus_9_v1')
    b = summarize_region(new, .89, .9)
    assert b['owners']['item_00']['scores_ge_0p02'] == 1
    assert b['seed'] == 0
    assert b['votes'].get('class_agreed_remove', 0) == 0


def test_fails_for_missing_evidence(tmp_path):
    root = _fixture(tmp_path / 'no_data', raw_valid=False, seed=False)
    p = root / '_udf_inference_0084' / 'oktopus_9_v1' / 'udf_cut_evidence_0074' / 'body' / 'face_evidence.npz'
    p.unlink()
    with pytest.raises(FileNotFoundError, match='Missing saved 0074 or 0079'):
        load_evidence(root, 'oktopus_9_v1')
