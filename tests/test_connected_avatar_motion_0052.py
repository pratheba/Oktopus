"""Demo-only connected avatar t7 bend with static attachment and moving boot."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from offline_tentacle_motion_0051 import (
    bind_connected_avatar, export_sequence, sample_curve, bend_positions,
)


class ConnectedAvatar0052Tests(unittest.TestCase):
    def test_vertex_weighting_freezes_attachment_and_other_limbs(self):
        pts = np.column_stack([np.linspace(0., 1., 25), np.zeros(25), np.zeros(25)])
        u, curve = sample_curve(pts, 128)
        body = np.column_stack([np.linspace(-.6, 0., 25), np.zeros(25), np.zeros(25)])
        other = np.column_stack([np.linspace(0., 1., 25), np.ones(25), np.zeros(25)])
        verts = np.asarray([
            [0.015,.04,0],  # root anchored
            [0.1,.06,0],    # root anchored
            [.8,.06,0],    # moving tentacle
            [.8,.15,0],    # moving tentacle
            [.8,1.02,0],   # competing limb, stays fixed
            [-.4,.05,0],   # body stays fixed
        ],dtype=np.float64)
        s, offsets, weights=bind_connected_avatar(
            verts,curve,u,np.vstack([body,other]),
            distance_full=.16,distance_zero=.3,owner_band=.03,
            root_hold=.12,root_blend_end=.3)
        np.testing.assert_array_equal(weights[[0,1,4,5]], np.zeros(4))
        self.assertGreater(weights[2], .9)
        raw,_=bend_positions(curve,s,offsets,angle_deg=25,bend_axis=[0,0,1])
        posed=verts+(raw-verts)*weights[:,None]
        np.testing.assert_array_equal(posed[[0,1,4,5]],verts[[0,1,4,5]])
        self.assertGreater(np.linalg.norm(posed[2]-verts[2]),.07)

    def test_end_to_end_connected_avatar_and_boot_export(self):
        with tempfile.TemporaryDirectory() as temp:
            home=Path(temp)
            target=np.column_stack([np.linspace(0.,1.,24),np.zeros(24),np.zeros(24)])
            body=np.column_stack([np.linspace(-.7,0.,20),np.zeros(20),np.zeros(20)])
            other=np.column_stack([np.linspace(0.,1.,20),np.ones(20),np.zeros(20)])
            npz=home/'std_handle.npz'
            np.savez(npz,segments=np.array([
                {'name':'body','keypoints':body},
                {'name':'t7','keypoints':target},
                {'name':'t8','keypoints':other},
            ],dtype=object))
            # A single continuous mesh with triangles from the body through t7.
            xs=np.linspace(-.6,1.,49)
            verts=np.asarray([[x,y,0.0] for x in xs for y in (-.055,.055)])
            faces=[]
            for i in range(len(xs)-1):
                k=2*i
                faces += [[k,k+1,k+2],[k+1,k+3,k+2]]
            avatar=trimesh.Trimesh(vertices=verts,faces=faces,process=False)
            avatar_file=home/'avatar.ply';avatar.export(avatar_file)
            boot=trimesh.creation.icosphere(subdivisions=1,radius=.07)
            boot.apply_translation([.8,0,0])
            boot_file=home/'boot.ply';boot.export(boot_file)
            out=home/'poses'
            args=SimpleNamespace(handle_npz=npz,curve='t7',moving=[f'boot={boot_file}'],
                 deform_avatar=[f'avatar={avatar_file}'],static=[],output=out,
                 frames=3,angle_deg=25,axis='z',root_end='auto',
                 root_hold=.12,full_bend=.85,curve_samples=128,
                 avatar_distance_full=.16,avatar_distance_zero=.28,
                 avatar_owner_band=.035,avatar_root_blend_end=.3)
            report=export_sequence(args)
            self.assertEqual(report['deformed_full_avatar'],['avatar'])
            posed=trimesh.load(out/'avatar_001.ply',process=False)
            rest=trimesh.load(avatar_file,process=False)
            np.testing.assert_array_equal(posed.faces,rest.faces)
            self.assertEqual(len(posed.vertices),len(rest.vertices))
            np.testing.assert_allclose(posed.vertices[:8],rest.vertices[:8],atol=1e-8)
            self.assertGreater(np.linalg.norm(posed.vertices[-1]-rest.vertices[-1]),.07)
            self.assertTrue((out/'boot_001.ply').is_file())
            self.assertTrue((out/'pose_001.glb').is_file())
            self.assertTrue((out/'skeleton_t7_001.npy').is_file())
            self.assertEqual(json.loads((out/'pose_manifest.json').read_text())['moving'], ['boot','avatar'])


if __name__=='__main__':
    unittest.main()
