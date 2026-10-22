"""0081: folded-curve shortcut regression and shared-edge locality."""
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'app'))
from udf_cut_curve_local_0081 import (LocalExpansionSettings, expand_face_mask,
                                      project_faces_to_curve_arclength)


class HairpinCurve:
    class Core:
        key_ts = np.array([0., 1.])

        def interpolate(self, ts):
            ts = np.asarray(ts)
            # Left-to-right, short U-turn, then right-to-left only 0.06 away.
            xyz = np.zeros((len(ts), 3))
            first = ts <= 0.45
            turn = (ts > 0.45) & (ts <= 0.55)
            last = ts > 0.55
            xyz[first, 0] = ts[first] / 0.45
            xyz[turn, 0] = 1.
            xyz[turn, 1] = (ts[turn] - 0.45) / 0.10 * 0.06
            xyz[last, 0] = 1. - (ts[last] - 0.55) / 0.45
            xyz[last, 1] = 0.06
            return {'points': xyz}

    core = Core()


def _ribbon():
    ts = np.linspace(0, 1, 101)
    centers = HairpinCurve().core.interpolate(ts)['points']
    vertices = []
    for center in centers:
        vertices.append(center + [0, 0, -0.006])
        vertices.append(center + [0, 0, 0.006])
    faces = []
    for i in range(len(centers)-1):
        a = 2*i
        faces.extend([[a, a+1, a+2], [a+1, a+3, a+2]])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def test_folded_tentacle_far_along_curve_not_selected_despite_xyz_proximity():
    mesh = _ribbon()
    arc, length = project_faces_to_curve_arclength(mesh, HairpinCurve(), samples=1025)
    # Seed is on first arm; end of second arm is near in XYZ but remote along arc.
    seeds = np.zeros(len(mesh.faces), dtype=bool)
    seeds[24:30] = True
    candidates = np.ones(len(mesh.faces), dtype=bool)
    selected, stats = expand_face_mask(mesh, seed_mask=seeds, candidate_mask=candidates,
             curve_s=arc, curve_length_world=length,
             settings=LocalExpansionSettings(max_curve_fraction=.10,
                max_neighbor_curve_jump=.07, max_surface_walk_fraction=.40,
                min_seed_component_faces=2))
    assert selected[24:30].all()
    assert selected[30:40].any()       # contiguous early-arm expansion works
    assert not selected[170:190].any() # folded-back arm must remain
    assert stats['blocked_candidates'] > 0


def test_candidate_disconnected_even_when_xyz_near():
    a = trimesh.Trimesh(vertices=[[0,0,0],[.01,0,0],[0,.01,0]], faces=[[0,1,2]], process=False)
    b = trimesh.Trimesh(vertices=[[.001,0,.001],[.011,0,.001],[.001,.01,.001]], faces=[[0,1,2]], process=False)
    merged = trimesh.util.concatenate([a,b])
    selected, stats = expand_face_mask(merged, seed_mask=[1,0], candidate_mask=[1,1],
                       curve_s=[.20,.20], curve_length_world=1,
                       settings=LocalExpansionSettings(min_seed_component_faces=1))
    assert selected.tolist() == [True,False]
    assert stats['blocked_candidates'] == 1


def test_nearby_connected_cap_faces_can_be_expanded():
    verts=[[0,0,0],[.1,0,0],[.0,.1,0],[.1,.1,0], [.2,0,0],[.2,.1,0]]
    faces=[[0,1,2],[1,3,2],[1,4,3],[4,5,3]]
    mesh=trimesh.Trimesh(vertices=verts,faces=faces,process=False)
    selected, stats=expand_face_mask(mesh, seed_mask=[1,1,0,0], candidate_mask=[1,1,1,1],
                      curve_s=[.45,.45,.45,.45], curve_length_world=1,
                      settings=LocalExpansionSettings(min_seed_component_faces=2))
    assert selected.tolist()==[True,True,True,True]
    assert stats['added_faces']==2


def test_ambiguous_curve_projection_jump_blocks_surface_shortcut():
    verts=[[0,0,0],[.1,0,0],[.1,.1,0],[.2,.1,0]]
    mesh=trimesh.Trimesh(vertices=verts,faces=[[0,1,2],[1,3,2]],process=False)
    selection, stat=expand_face_mask(mesh,seed_mask=[1,0],candidate_mask=[1,1],
                  curve_s=[.1,.9], curve_length_world=1,
                  settings=LocalExpansionSettings(min_seed_component_faces=1))
    assert selection.tolist()==[True,False]
    assert stat['blocked_mesh_links_curve_jump']==1


def test_seed_component_too_small_to_expand_but_seed_kept():
    mesh=trimesh.Trimesh(vertices=[[0,0,0],[.1,0,0],[.1,.1,0],[.2,.1,0]],
                          faces=[[0,1,2],[1,3,2]],process=False)
    chosen, stats=expand_face_mask(mesh,seed_mask=[1,0],candidate_mask=[1,1],
                                 curve_s=[.3,.3],curve_length_world=1)
    assert chosen.tolist()==[True,False]
    assert stats['small_seed_components_not_expanded']==1


def test_missing_seed_or_invalid_configuration():
    mesh=trimesh.creation.icosphere(subdivisions=1)
    n=len(mesh.faces)
    selected, info=expand_face_mask(mesh, seed_mask=np.zeros(n, bool),
        candidate_mask=np.ones(n, bool), curve_s=np.full(n,.5),curve_length_world=1)
    assert not selected.any() and info['selected_faces']==0
    with pytest.raises(ValueError):
        LocalExpansionSettings(max_curve_fraction=0)
    with pytest.raises(ValueError):
        expand_face_mask(mesh,seed_mask=np.ones(n,bool),
            candidate_mask=np.zeros(n,bool),curve_s=np.full(n,.5),curve_length_world=1)


def test_replay_refuses_wrong_original_mesh(tmp_path):
    from importlib.util import spec_from_file_location, module_from_spec
    script = Path(__file__).resolve().parents[1] / 'tools' / 'udf_0081_replay_locality.py'
    spec = spec_from_file_location('replay_0081', script)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    mesh = trimesh.creation.box()
    src = tmp_path/'mesh.ply'
    mesh.export(src)
    evidence = tmp_path/'locality.npz'
    n=len(mesh.faces)
    np.savez(evidence, face_curve_arclength=np.full(n,.5),curve_length_world=np.array(1.),
             seed_faces=np.zeros(n,bool), candidate_faces=np.ones(n,bool),
             face_centroids_xyz=np.asarray(mesh.triangles_center)+.25)
    with pytest.raises(ValueError,match='not aligned'):
        module.preview_one(src, evidence, tmp_path/'out',LocalExpansionSettings())


def test_replay_outputs_preview_without_mutating_input(tmp_path):
    from importlib.util import spec_from_file_location, module_from_spec
    script = Path(__file__).resolve().parents[1] / 'tools' / 'udf_0081_replay_locality.py'
    spec=spec_from_file_location('replay_0081_good',script)
    mod=module_from_spec(spec);spec.loader.exec_module(mod)
    mesh=trimesh.creation.box()
    original=tmp_path/'body.ply';mesh.export(original)
    imported=trimesh.load(original, force='mesh', process=False)
    n=len(imported.faces)
    seed=np.zeros(n,dtype=bool);seed[:2]=True
    # No candidate outside seed in this test, so output is predictable.
    data=tmp_path/'locality.npz'
    np.savez_compressed(data,face_curve_arclength=np.full(n,.5),curve_length_world=np.array(1.),
             seed_faces=seed,candidate_faces=seed,
             face_centroids_xyz=np.asarray(imported.triangles_center,dtype=np.float32))
    report=mod.preview_one(original,data,tmp_path/'out',LocalExpansionSettings(min_seed_component_faces=1))
    assert report['selected_faces']==2
    assert (tmp_path/'out'/'proposed_kept_cleaned.ply').is_file()
    assert len(trimesh.load(original,force='mesh').faces)==n
