"""0053: straightening a curved skeletal tentacle with attached boot and fixed root."""
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
    sample_curve, straighten_positions, project_mesh_vertices, export_sequence,
    smooth_bend_profile,
)


class Straighten0053Tests(unittest.TestCase):
    def setUp(self):
        t = np.linspace(0, 1, 41)
        # Curved centerline with an almost-horizontal root and a bent tip.
        self.keypoints = np.column_stack((t, .25*t*t, .09*np.sin(np.pi*t/2)**2))
        self.u, self.curve = sample_curve(self.keypoints, 256)

    def test_centerline_straightens_after_transition_preserving_arc_length(self):
        moved, posed = straighten_positions(self.curve, np.array([.5]),
                np.zeros((1,3)), progress=1, root_hold=.12, full_straight=.85)
        np.testing.assert_allclose(posed[self.u <= .12],
                                   self.curve[self.u <= .12], atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            np.linalg.norm(np.diff(posed,axis=0), axis=1),
            np.linalg.norm(np.diff(self.curve,axis=0), axis=1), atol=3e-15, rtol=0)
        tangent = np.diff(posed[self.u >= .85],axis=0)
        tangent /= np.linalg.norm(tangent,axis=1,keepdims=True)
        np.testing.assert_allclose(tangent, np.broadcast_to(tangent[0],tangent.shape),
                                   atol=2e-7, rtol=0)
        self.assertGreater(np.linalg.norm(posed[-1]-self.curve[-1]), .1)

    def test_rest_identity_and_halfway_pose(self):
        boot = trimesh.creation.icosphere(subdivisions=1, radius=.055)
        boot.apply_translation([.9,.22,.07])
        s,offsets = project_mesh_vertices(boot.vertices,self.curve,self.u)
        at_rest,rest_skel=straighten_positions(self.curve,s,offsets,progress=0)
        np.testing.assert_allclose(at_rest,boot.vertices,atol=3e-12,rtol=0)
        np.testing.assert_allclose(rest_skel,self.curve,atol=3e-12,rtol=0)
        half,half_skel=straighten_positions(self.curve,s,offsets,progress=.5)
        full,full_skel=straighten_positions(self.curve,s,offsets,progress=1)
        self.assertTrue(np.isfinite(full).all())
        self.assertTrue(np.isfinite(half).all())
        self.assertGreater(np.linalg.norm(full.mean(axis=0)-boot.vertices.mean(axis=0)),.06)
        self.assertLess(np.linalg.norm(half_skel[-1]-rest_skel[-1]),
                        np.linalg.norm(full_skel[-1]-rest_skel[-1]))
        with self.assertRaisesRegex(ValueError, 'progress'):
            straighten_positions(self.curve,s,offsets,progress=1.5)

    def test_end_to_end_connected_avatar_and_boot_frame_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            handle=folder/'std_handle.npz'
            body=np.column_stack((np.linspace(-.3,0,12),np.zeros(12),np.zeros(12)))
            other=np.column_stack((np.linspace(0,.9,24),np.ones(24),np.zeros(24)))
            np.savez(handle,segments=np.array([
                {'name':'body','keypoints':body},
                {'name':'t7','keypoints':self.keypoints},
                {'name':'t8','keypoints':other}], dtype=object))
            # A connected strip across the attachment and along the tentacle.
            center_x=np.linspace(-.3,1,65)
            center_y=.25*np.clip(center_x,0,None)**2
            vertex=np.array([[x,y+delta,.09*np.sin(np.pi*np.clip(x,0,1)/2)**2]
                             for x,y in zip(center_x,center_y) for delta in (-.045,.045)])
            faces=[]
            for k in range(len(center_x)-1):
                j=k*2
                faces.extend([[j,j+1,j+2],[j+1,j+3,j+2]])
            avatar=trimesh.Trimesh(vertices=vertex,faces=faces,process=False)
            avatar_file=folder/'octopus.ply'
            avatar.export(avatar_file)
            boot=trimesh.creation.icosphere(subdivisions=1,radius=.055)
            boot.apply_translation([.93,.22,.09]);boot_file=folder/'boot.ply'
            boot.export(boot_file)
            args=SimpleNamespace(handle_npz=handle,curve='t7',motion='straighten',
                 moving=[f'boot={boot_file}'],deform_avatar=[f'octopus={avatar_file}'],
                 static=[],output=folder/'poses',frames=5,angle_deg=25,axis='z',
                 root_end='start',root_hold=.12,full_bend=.85,curve_samples=256,
                 avatar_distance_full=.15,avatar_distance_zero=.3,
                 avatar_owner_band=.035,avatar_root_blend_end=.3)
            report=export_sequence(args)
            self.assertEqual(report['motion'],'straighten')
            self.assertEqual(report['frames'],5)
            self.assertIsNone(report['axis'])
            self.assertAlmostEqual(report['poses'][2]['straighten_progress'],1.)
            self.assertAlmostEqual(report['poses'][0]['straighten_progress'],0.)
            self.assertIsNone(report['poses'][2]['angle_deg'])
            before=trimesh.load(avatar_file,process=False)
            peak=trimesh.load(args.output/'octopus_002.ply',process=False)
            last=trimesh.load(args.output/'octopus_004.ply',process=False)
            np.testing.assert_array_equal(peak.faces,before.faces)
            np.testing.assert_allclose(peak.vertices[:12],before.vertices[:12],atol=2e-7)
            np.testing.assert_allclose(last.vertices,before.vertices,atol=2e-7)
            self.assertGreater(np.linalg.norm(peak.vertices[-1]-before.vertices[-1]),.05)
            boot_start=trimesh.load(args.output/'boot_000.ply',process=False)
            boot_peak=trimesh.load(args.output/'boot_002.ply',process=False)
            self.assertGreater(np.linalg.norm(boot_peak.vertices.mean(axis=0)-
                                             boot_start.vertices.mean(axis=0)),.05)
            npy_start=np.load(args.output/'skeleton_t7_000.npy')
            npy_peak=np.load(args.output/'skeleton_t7_002.npy')
            np.testing.assert_allclose(npy_start[:25],npy_peak[:25],atol=1e-10)
            self.assertGreater(np.linalg.norm(npy_peak[-1]-npy_start[-1]),.1)
            self.assertTrue((args.output/'pose_004.glb').exists())
            self.assertEqual(json.loads((args.output/'pose_manifest.json').read_text())['motion'],'straighten')


if __name__=='__main__': unittest.main()
