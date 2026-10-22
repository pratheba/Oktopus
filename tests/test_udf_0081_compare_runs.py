"""Read-only saved-0081 cross-run diagnostic regression tests."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from udf_0081_compare_runs import find_locality, compare, read_part, discover_runs


def _save(root, *, cap_seeds):
    loc = root / '_udf_inference_0084' / 'oktopus_9_v1' / 'udf_cut_curve_local_0081'
    p = loc / 'body'
    p.mkdir(parents=True)
    s = np.array([.05, .25, .40, .50, .75, .91, .91, .92], dtype='float32')
    seed = np.array([0, 0, 1, 1, 0, cap_seeds, cap_seeds, cap_seeds], bool)
    cand = np.array([0, 0, 1, 1, 0, 1, 1, 1], bool)
    selected = seed.copy()
    blocked = cand & ~selected
    np.savez_compressed(p / 'locality.npz', face_curve_arclength=s,
                        curve_length_world=1.0, seed_faces=seed, candidate_faces=cand,
                        selected_faces=selected, blocked_faces=blocked,
                        face_centroids_xyz=np.arange(24).reshape(8, 3).astype('float32') / 100)
    return loc


def test_compare_detects_missing_high_arc_seeds(tmp_path):
    _save(tmp_path / 'working', cap_seeds=True)
    _save(tmp_path / 'failing', cap_seeds=False)
    old = find_locality(tmp_path / 'working', 'oktopus_9_v1')
    new = find_locality(tmp_path / 'failing', 'oktopus_9_v1')
    report = compare(old, new, parts=('body',))['parts']['body']
    assert report['old']['cap_high_s_ge_089']['seed'] == 3
    assert report['new']['cap_high_s_ge_089']['seed'] == 0
    assert report['new']['cap_high_s_ge_089']['blocked'] == 3
    assert len(discover_runs(tmp_path, 'oktopus_9_v1')) == 2


def test_bad_mask_detected(tmp_path):
    loc = _save(tmp_path / 'run', cap_seeds=False)
    p = loc / 'body' / 'locality.npz'
    with np.load(p) as z:
        data = {k: z[k] for k in z.files}
    data['blocked_faces'][5] = False
    np.savez_compressed(p, **data)
    import pytest
    with pytest.raises(ValueError, match='inconsistent 0081 masks'):
        read_part(loc, 'body')
