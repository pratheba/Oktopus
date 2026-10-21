"""Exact legacy 2026-10-18 absolute-hysteresis/metric-calibration regression tests."""
import sys
import types
from pathlib import Path
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'app'))
from agent_3dvec_udf_cut import AgentUDFCut


def test_absolute_hysteresis_keeps_raw_thresholds_with_supported_samples():
    scores = np.r_[np.full(400, .006), np.full(400, .04), np.full(200, .1)]
    seed, grow, report = AgentUDFCut._absolute_hysteresis_with_saturation_guard(
        scores, seed_world=.02, grow_world=.01, far_world=.1,
        far_band_world=.002, min_informative_fraction=.05,
        min_informative_faces=128,
    )
    assert seed == .02 and grow == .01
    assert report['informative_faces'] == 800
    assert not report['saturated']


def test_absolute_hysteresis_saturation_does_not_remove_far_plateau():
    scores = np.full(1000, .1001)
    seed, grow, report = AgentUDFCut._absolute_hysteresis_with_saturation_guard(
        scores, seed_world=.02, grow_world=.01, far_world=.1,
        far_band_world=.002, min_informative_fraction=.05,
        min_informative_faces=128,
    )
    assert report['saturated']
    assert seed == grow and seed > np.max(scores)


def test_original_metric_calibration_reconstructs_world_gradient_scale():
    agent = object.__new__(AgentUDFCut)
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=.025)
    # g(x) = 0.02 + 2.5*x, so ||grad_world(g)|| = 2.5.
    def fake_raw(self, points, **kwargs):
        points = np.asarray(points)
        return .02 + 2.5 * points[:, 0], np.ones(len(points), dtype=bool)
    agent._query_adapted_raw_udf = types.MethodType(fake_raw, agent)
    result = agent._estimate_udf_metric_calibration(
        mesh,
        face_scores=np.full(len(mesh.faces), .01),
        usable_faces=np.ones(len(mesh.faces), dtype=bool),
        avatar_curve=None, adapt_arg={}, accessory_key='test',
        model_batch_size=4096, far_world=.1,
        seed_world=.02, grow_world=.01,
        max_samples=256, min_samples=32,
    )
    assert result['status'] == 'ok'
    np.testing.assert_allclose(result['metric_scale'], 2.5, rtol=1e-10)
    np.testing.assert_allclose(result['proposed_seed_raw'], .05, rtol=1e-10)
    np.testing.assert_allclose(result['proposed_grow_raw'], .025, rtol=1e-10)


def test_fragment_cleanup_restored_from_legacy_cut():
    big = trimesh.creation.icosphere(subdivisions=1, radius=.1)
    tiny = trimesh.Trimesh(vertices=np.array([[2.,0,0],[2.,1,0],[2.,0,1]]),
                           faces=np.array([[0,1,2]]), process=False)
    mesh = trimesh.util.concatenate([big, tiny])
    kept, removed, report = AgentUDFCut._cleanup_tiny_vertex_components(
        mesh, min_faces=10, min_fraction_of_largest=0.0,
    )
    assert len(kept.faces) == len(big.faces)
    assert len(removed.faces) == 1
    assert report['removed_faces'] == 1
