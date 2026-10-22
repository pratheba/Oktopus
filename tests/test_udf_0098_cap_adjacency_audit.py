"""Read-only 0098 adjacency audit checks."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from udf_0097_physical_support_probe import boundary_face_mask
from udf_0098_cap_adjacency_audit import audit_window
from test_udf_0097_physical_support_probe import plane_grid


def case_for_grid():
    mesh = plane_grid(3, 3)
    n = len(mesh.faces)
    scores = np.zeros(n)
    scores[8] = 0.1
    return dict(mesh=mesh, arc=np.full(n, .895), scores=scores,
                valid=np.ones(n, dtype=bool), seed=.02, grow=.01)


def test_edges_to_weak_are_counted():
    case = case_for_grid()
    report = audit_window(case, .89, .90, radius=2.)
    assert report['anchor_strong_faces'] == 1
    assert report['strong_faces_with_weak_valid_neighbors_within_radius'] == 1
    assert report['below_grow_edges_within_radius'] > 0
    assert report['edges_to_invalid_or_missing_raw_evidence'] == 0


def test_invalid_neighbors_do_not_count_as_support():
    case = case_for_grid()
    case['valid'][:] = False
    case['valid'][8] = True
    report = audit_window(case, .89, .90, radius=2.)
    assert report['anchor_strong_faces'] == 1
    assert report['strong_faces_with_weak_valid_neighbors_within_radius'] == 0
    assert report['edges_to_invalid_or_missing_raw_evidence'] > 0


def test_window_cropping_visible_in_report():
    case = case_for_grid()
    # Set all other faces just outside the crop; the graph retains their edges.
    case['arc'][:] = .90001
    case['arc'][8] = .895
    report = audit_window(case, .89, .90, radius=2.)
    assert report['anchor_strong_faces'] == 1
    assert report['edges_to_valid_faces_outside_window'] > 0
    assert report['below_grow_edges_within_window'] == 0
