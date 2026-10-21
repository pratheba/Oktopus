"""0073: restored absolute-hysteresis gradient calibration on final owned faces."""
import sys
import types
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'app'))
from agent_3dvec_udf_cut import AgentUDFCut


def _agent(gradient=3.0):
    agent = object.__new__(AgentUDFCut)
    def udf(self, points, **kwargs):
        points = np.asarray(points)
        return .02 + gradient * points[:, 0], np.ones(len(points), dtype=bool)
    agent._query_adapted_raw_udf = types.MethodType(udf, agent)
    return agent


def _owner():
    return {'item_index': 3, 'seed_world': .02, 'grow_world': .01,
            'avatar_curve': None, 'adapt_arg': {}, 'accessory_key': 'test'}


def _input():
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=.025)
    scores = np.full(len(mesh.faces), .010, dtype=float)
    usable = np.ones(len(mesh.faces), dtype=bool)
    return mesh, scores, usable


def test_restores_legacy_gradient_metric_on_final_mesh():
    agent = _agent(3.0)
    mesh, scores, usable = _input()
    seed, grow, report = agent._direct_metric_calibrate_owner_0073(
        mesh, face_scores=scores, usable_faces=usable, owner=_owner(),
        model_batch_size=32768, far_world=.1,
        config={'udf_cut_direct_min_informative_faces': 32})
    assert report['status'] == 'ok'
    np.testing.assert_allclose([seed, grow], [.06, .03], atol=1e-10)
    assert report['saturation']['mode'] == 'absolute_hysteresis'


def test_far_plateau_cannot_propose_cut():
    agent = _agent()
    mesh, scores, usable = _input()
    scores[:] = .1
    seed, grow, report = agent._direct_metric_calibrate_owner_0073(
        mesh, face_scores=scores, usable_faces=usable, owner=_owner(),
        model_batch_size=32768, far_world=.1, config={})
    assert report['status'] == 'saturated'
    assert seed >= .1 and seed == grow


def test_missing_support_does_not_calibrate():
    agent = _agent()
    mesh, scores, usable = _input()
    usable[:] = False
    seed, grow, report = agent._direct_metric_calibrate_owner_0073(
        mesh, face_scores=scores, usable_faces=usable, owner=_owner(),
        model_batch_size=32768, far_world=.1, config={})
    assert report['status'] == 'no_usable_faces'
    assert seed == .02 and grow == .01


def test_metric_mask_does_not_use_unowned_faces():
    agent = _agent(2.0)
    mesh, scores, usable = _input()
    usable[len(usable)//2:] = False
    seed, grow, report = agent._direct_metric_calibrate_owner_0073(
        mesh, face_scores=scores, usable_faces=usable, owner=_owner(),
        model_batch_size=32768, far_world=.1,
        config={'udf_cut_metric_calibration_min_samples': 16,
                'udf_cut_direct_min_informative_faces': 16})
    assert report['sample_count'] == np.sum(usable)
    np.testing.assert_allclose([seed, grow], [.04, .02], atol=1e-10)


def test_pair_yaml_has_same_adaptation_fingerprint_and_no_mc_input():
    import yaml
    from root_leaf_combination import build_root_leaf_plan
    root = Path(__file__).resolve().parents[1]
    base = yaml.safe_load((root/'exp/train/manipulation/udf_adapt_oktopus_puffer_0067_final.yaml').read_text())
    diag = yaml.safe_load((root/'exp/train/manipulation/udf_adapt_oktopus_puffer_0073_metric_diagnostic.yaml').read_text())
    apply = yaml.safe_load((root/'exp/train/manipulation/udf_adapt_oktopus_puffer_0073_metric_apply.yaml').read_text())
    assert diag['adaptations'] == apply['adaptations'] == base['adaptations']
    for cfg in [diag, apply]:
        assert cfg['udf_cut_direct_root_leaves'] is True
        assert cfg['udf_cut_root_leaves_mesh_dir'] == base['udf_cut_root_leaves_mesh_dir']
        assert cfg['udf_cut_auto_thresholds']['enabled'] is False
        assert cfg['udf_cut_direct_metric_parts_0073']
        assert not cfg.get('udf_cut_direct_cap_guard_parts_0071')
    assert 'udf_cut_direct_dry_run_parts_0073' in diag
    assert 'udf_cut_direct_dry_run_parts_0073' not in apply


def test_metric_dry_run_preserves_final_part_but_reports_candidate(tmp_path):
    import json
    from root_leaf_combination import build_root_leaf_plan
    from udf_cut_final_root_leaves import export_direct_root_leaf_results

    items = [{'name':'body', 'target_key':'o|body', 'accessory_key':'a|body'},
             {'name':'arm', 'target_key':'o|t1', 'accessory_key':'a|arm'}]
    mapping = {'root':'body', 'leaves':{'tentacle_t1':{'target_key':'o|t1'}}}
    plan = build_root_leaf_plan(items, mapping)
    source = tmp_path / 'riggable_parts'
    source.mkdir()
    meshes = {part: trimesh.creation.icosphere(subdivisions=1) for part in plan.part_names}
    for part, mesh in meshes.items():
        mesh.export(source / (part+'.ply'))
    (source.parent/'root_leaves_ownership.json').write_text(json.dumps({
        'adaptation_fingerprint': plan.adaptation_fingerprint,
        'part_names': list(plan.part_names),
        'item_to_part': {str(i):p for i,p in plan.item_to_part.items()},
    }))
    agent = object.__new__(AgentUDFCut)
    agent.device = 'cpu'
    agent.feat_dict = {x:i for i,x in enumerate(['o|body','o|t1','a|body','a|arm'])}
    agent.curve_from_key = lambda key: object()
    cut_calls = []
    def cut(self, mesh, **kwargs):
        cut_calls.append((kwargs['owners'][0]['item_index'],kwargs['owners'][0].get('metric_0073')))
        removed = AgentUDFCut._mesh_from_face_mask(mesh, np.arange(len(mesh.faces)) < 8)
        kept = AgentUDFCut._mesh_from_face_mask(mesh, np.arange(len(mesh.faces)) >= 8)
        return kept, removed, {'proposed_removed':8}, np.zeros(len(mesh.faces))
    agent._direct_udf_cut_combined_mesh = types.MethodType(cut, agent)
    output = tmp_path/'out'
    report = agent._cut_final_root_leaf_parts(
        adaptation_items=items, extraction_config={
            'udf_cut_direct_metric_parts_0073':['tentacle_t1'],
            'udf_cut_direct_dry_run_parts_0073':['tentacle_t1'],
            'udf_cut_direct_export_removed_0070':True,
            'udf_cut_auto_thresholds':{'enabled':False},
        }, root_path=str(tmp_path), output_folder=str(output),
        target_curves={i['target_key']:object() for i in items},
        seam_plan={}, apply_seam_warp=False,
        rig_parts_plan=plan, rig_parts_spec=str(source),
        model_batch_size=32, query_batch_size=32, far_world=.1,
    )
    assert cut_calls == [(0,False),(1,True)]
    assert report['parts']['tentacle_t1']['status'] == 'dry_run_preserved_0073'
    assert report['parts']['tentacle_t1']['candidate_removed_0073'] == 8
    assert report['parts']['tentacle_t1']['faces_before'] == report['parts']['tentacle_t1']['faces_after']
    assert report['parts']['body']['faces_after'] == len(meshes['body'].faces)-8
    assert (output/'udf_cut_removed_diagnostics_0070'/'tentacle_t1_removed.ply').exists()
