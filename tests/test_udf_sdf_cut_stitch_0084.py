"""Single-entrypoint UDF + preview promotion + real stitching tests."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
import yaml

import test_udf_sdf_cut_stitch as flow
from tools.udf_stitch_0084 import PARTS

ASSIGNMENT = ['body','tentacle_t1','tentacle_t1','tentacle_t1',
              'tentacle_t2','tentacle_t2','tentacle_t3',
              'tentacle_t4','tentacle_t4','tentacle_t4',
              'tentacle_t5','tentacle_t5','tentacle_t5',
              'tentacle_t6','tentacle_t8']


def _root_with_seven_holes():
    nx, ny = 16, 4
    verts = [[float(x), float(y), 0.] for y in range(ny+1) for x in range(nx+1)]
    def idx(x, y): return y*(nx+1)+x
    missing = {(x, 1) for x in range(1, 14, 2)}
    faces = []
    for y in range(ny):
        for x in range(nx):
            if (x, y) in missing:
                continue
            a,b,c,d = idx(x,y),idx(x+1,y),idx(x+1,y+1),idx(x,y+1)
            faces += [(a,b,c),(a,c,d)]
    return trimesh.Trimesh(vertices=verts,faces=faces,process=False)


def _leaf(x):
    base = np.array([[x,1,.004],[x+1,1,.004],[x+1,2,.004],[x,2,.004]],float)
    verts = np.vstack([base,base + [0,0,.2]])
    faces=[]
    for i in range(4):
        j=(i+1)%4
        faces += [(i,j,i+4),(j,j+4,i+4)]
    faces += [(4,5,6),(4,6,7)]
    return trimesh.Trimesh(vertices=verts,faces=faces,process=False)


def make_fixture(monkeypatch, tmp_path, *, stitchable=True):
    monkeypatch.setattr(flow, 'ROOT', tmp_path)
    sdf = tmp_path/'inference'/'4M'/'puffer_0067_final_parts_r512'/'oktopus_9_v1'
    parts = sdf/'riggable_parts'
    parts.mkdir(parents=True)
    meshes = {'body': _root_with_seven_holes() if stitchable else trimesh.creation.icosphere(subdivisions=1)}
    for j, part in enumerate(PARTS[1:]):
        meshes[part] = _leaf(1+2*j)
    for part, mesh in meshes.items():
        mesh.export(parts/(part+'.ply'))
    (sdf/'root_leaves_ownership.json').write_text(json.dumps({
        'schema_version':49,'part_names':list(PARTS),
        'item_to_part':{str(i):name for i,name in enumerate(ASSIGNMENT)},
        'adaptation_fingerprint':'stable_test_fingerprint'}))
    yaml_path = tmp_path/'exp'/'train'/'manipulation'/'full.yaml'
    yaml_path.parent.mkdir(parents=True)
    data = {'udf_full_pipeline_0084': {'enabled':True,
            'stitch': {'enabled':True,'require_success':True,
                       'max_seam_gap':.02,'max_bridge_edge':1.5,
                       'max_perimeter_ratio':1.65}},
            'udf_cut_direct_root_leaves': True,
            'udf_cut_curve_local_preview_0081': True,
            'udf_cut_direct_dry_run_parts_0073': list(PARTS),
            'udf_cut_root_leaves_mesh_dir': str(parts.relative_to(tmp_path))}
    yaml_path.write_text(yaml.safe_dump(data))
    return sdf, yaml_path


def fake_inference(args, rest, yaml_path, stage):
    config = yaml.safe_load(yaml_path.read_text())
    parts = flow.ROOT / config['udf_cut_root_leaves_mesh_dir']
    for name in PARTS:
        mesh = trimesh.load(parts/(name+'.ply'),process=False)
        folder = stage/'udf_cut_curve_local_0081'/name
        folder.mkdir(parents=True)
        mesh.export(folder/'proposed_kept_cleaned.ply')
        n=len(mesh.faces)
        (folder/'summary.json').write_text(json.dumps({
            'mode':'surface_and_arclength_local_0081',
            'selected_faces':0,'faces_after_preview':n}))
        np.savez_compressed(folder/'locality.npz',
            selected_faces=np.zeros(n,bool),candidate_faces=np.zeros(n,bool),
            seed_faces=np.zeros(n,bool),face_centroids_xyz=np.asarray(mesh.triangles_center,dtype=np.float32))


def user_args():
    return SimpleNamespace(config='config/my_udf.yaml',out='one_step',shape='oktopus_9_v1',
        yaml='full.yaml',resolution=512,checkpoint='final',udf_far_value=.1,
        udf_model_batch_size=32768,udf_cut_query_batch_size=32768)


def test_complete_one_command_produces_cut_parts_glb_and_stitched_ply(monkeypatch,tmp_path):
    make_fixture(monkeypatch,tmp_path)
    monkeypatch.setattr(flow,'_run_inference',fake_inference)
    output=flow.execute(user_args())
    dest=tmp_path/'inference'/'4M'/'one_step'/'oktopus_9_v1'
    assert output['status']=='COMPLETE'
    assert (dest/'riggable_after_udf_cut_STITCHED.ply').exists()
    assert (dest/'riggable_after_udf_cut.glb').exists()
    assert (dest/'riggable_after_udf_cut_UNSTITCHED_parts.ply').exists()
    assert all((dest/'riggable_parts_after_udf_cut'/(part+'.ply')).exists() for part in PARTS)
    assert json.loads((dest/'riggable_after_udf_cut_STITCHED_report.json').read_text())['edge_connected_components_after']==1
    assert len(trimesh.load(dest/'riggable_after_udf_cut.glb',force='scene').geometry)==8
    assert json.loads((dest/'udf_sdf_cut_stitch_0084_report.json').read_text())['status']=='COMPLETE'


def test_failed_real_stitch_keeps_parts_but_never_writes_fake_stitched_ply(monkeypatch,tmp_path):
    make_fixture(monkeypatch,tmp_path,stitchable=False)
    monkeypatch.setattr(flow,'_run_inference',fake_inference)
    with pytest.raises(RuntimeError, match='Real root/leaf stitch failed'):
        flow.execute(user_args())
    dest=tmp_path/'inference'/'4M'/'one_step'/'oktopus_9_v1'
    assert (dest/'riggable_after_udf_cut.glb').exists()
    assert not (dest/'riggable_after_udf_cut_STITCHED.ply').exists()
    assert json.loads((dest/'riggable_after_udf_cut_STITCHED_report.json').read_text())['status']=='STITCH_FAILED'


def test_no_mc_mesh_and_stage_resolves_to_sdf_model_scale(monkeypatch,tmp_path):
    _, yaml_path = make_fixture(monkeypatch,tmp_path)
    config=yaml.safe_load(yaml_path.read_text())
    _, stage, out=flow._locate_sdf_and_output(config,'my_output','oktopus_9_v1')
    assert '/inference/4M/my_output/_udf_inference_0084/oktopus_9_v1' in str(stage)
    assert out.parent.name=='my_output'
    with pytest.raises(ValueError,match='--mc-mesh'):
        flow._run_inference(user_args(),['--mc-mesh','invalid.ply'],yaml_path,stage)
    config['udf_cut_curve_local_preview_0081']=False
    with pytest.raises(ValueError,match='0081 direct-final'):
        flow._locate_sdf_and_output(config,'another','oktopus_9_v1')


def test_refuses_overwrite(monkeypatch,tmp_path):
    make_fixture(monkeypatch,tmp_path)
    monkeypatch.setattr(flow,'_run_inference',fake_inference)
    flow.execute(user_args())
    with pytest.raises(FileExistsError):
        flow.execute(user_args())
