import json
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import trimesh

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'tools'))
from offline_tentacle_motion_0051 import (
    bend_positions, load_curve_npz, orient_from_root, sample_curve,
    project_mesh_vertices, select_bend_axis, export_sequence,
)


class Motion0051Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.points = np.stack((np.linspace(0., 1., 21),
                               0.03*np.sin(np.linspace(0., 2*np.pi, 21)),
                               np.zeros(21)),axis=1)

    def test_zero_angle_is_exact_vertex_identity(self):
        u, points = sample_curve(self.points, 256)
        mesh = trimesh.creation.icosphere(subdivisions=2,radius=.09)
        mesh.apply_translation([.7, .07, .01])
        s,offset = project_mesh_vertices(mesh.vertices,points,u)
        posed,skel=bend_positions(points,s,offset,angle_deg=0,
                                 bend_axis=[0,0,1])
        np.testing.assert_allclose(posed, mesh.vertices, atol=5e-13)
        np.testing.assert_allclose(skel,points,atol=5e-13)

    def test_root_anchor_fixed_and_tip_moves_and_connectivity_unchanged(self):
        u,points = sample_curve(self.points, 256)
        mesh=trimesh.creation.cylinder(radius=.06,height=.45,sections=20)
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[0,1,0]))
        mesh.apply_translation([.6,0,0])
        s,offset=project_mesh_vertices(mesh.vertices,points,u)
        posed,skel=bend_positions(points,s,offset,angle_deg=30,bend_axis=[0,0,1])
        self.assertGreater(np.linalg.norm(skel[-1]-points[-1]),0.1)
        np.testing.assert_allclose(skel[0],points[0],atol=1e-14)
        root_s=np.array([0.01,0.06,0.1]); root_off=np.zeros((3,3))
        a,b=bend_positions(points,root_s,root_off,angle_deg=30,bend_axis=[0,0,1])
        np.testing.assert_allclose(a, np.stack([np.interp(root_s,u,points[:,i])
                                                   for i in range(3)],axis=-1),atol=1e-4)
        self.assertEqual(len(posed),len(mesh.vertices))

    def test_load_selects_name_and_root_by_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=pathlib.Path(tmp)/'handle.npz'
            body=np.stack([np.linspace(-.2,0,5),np.zeros(5),np.zeros(5)],axis=-1)
            np.savez(path,segments=np.array([{'name':'body','keypoints':body},
                                             {'name':'t7','keypoints':self.points}],dtype=object))
            pts,b=load_curve_npz(path,'t7')
            _,root=orient_from_root(pts,b,'auto')
            self.assertEqual(root,'start')
            with self.assertRaisesRegex(ValueError,'not uniquely found'):
                load_curve_npz(path,'t9')

    def test_cli_generates_frame_mesh_scenes_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp)
            handle=root/'std_handle.npz'
            body=np.stack([np.linspace(-.2,0,5),np.zeros(5),np.zeros(5)],axis=-1)
            np.savez(handle,segments=np.array([{'name':'body','keypoints':body},
                    {'name':'t7','keypoints':self.points}],dtype=object))
            mesh=trimesh.creation.icosphere(subdivisions=1,radius=.15)
            mesh.apply_translation([.6,.02,0.])
            input_mesh=root/'boot.ply';mesh.export(input_mesh)
            out=root/'poses'
            args=SimpleNamespace(handle_npz=handle,curve='t7',moving=['boot='+str(input_mesh)],
                      static=[],output=out,frames=3,angle_deg=20,
                      axis='z',root_end='auto',root_hold=.12,full_bend=.85,
                      curve_samples=128)
            result=export_sequence(args)
            self.assertEqual(result['frames'],3)
            self.assertEqual(len(result['poses']),3)
            self.assertEqual(result['poses'][0]['quality']['boot']['faces'],len(mesh.faces))
            for i in range(3):
                self.assertTrue((out/f'boot_{i:03d}.ply').exists())
                self.assertTrue((out/f'pose_{i:03d}.glb').exists())
                self.assertTrue((out/f'skeleton_t7_{i:03d}.npy').exists())
            original=trimesh.load(input_mesh,process=False)
            frame0=trimesh.load(out/'boot_000.ply',process=False)
            np.testing.assert_allclose(original.vertices,frame0.vertices,atol=1e-7)
            manifest=json.loads((out/'pose_manifest.json').read_text())
            self.assertFalse(manifest['connected_surface_validated'])


if __name__=='__main__': unittest.main()
