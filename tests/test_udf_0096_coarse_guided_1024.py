import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import trimesh

SRC = Path(__file__).resolve().parents[1] / 'tools' / 'udf_0096_coarse_guided_1024.py'
spec = importlib.util.spec_from_file_location('guidance_0096', SRC)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def square(z=0, shift=0):
    vertices = np.asarray([[shift,0,z],[shift+1,0,z],[shift+1,1,z],[shift,1,z]],float)
    faces=np.asarray([[0,1,2],[0,2,3]])
    return trimesh.Trimesh(vertices=vertices,faces=faces,process=False)


def masks(mesh, *, selected, candidate, s=.9):
    return dict(seed_faces=np.array(selected,dtype=bool),
                selected_faces=np.array(selected,dtype=bool),
                candidate_faces=np.array(candidate,dtype=bool),
                blocked_faces=np.array(candidate,dtype=bool)&~np.array(selected,dtype=bool),
                face_curve_arclength=np.full(len(mesh.faces),s,dtype=np.float32),
                curve_length_world=np.asarray(1.,dtype=float),
                face_centroids_xyz=np.asarray(mesh.triangles_center,dtype=np.float32))

class GuidedTest(unittest.TestCase):
    def test_aligned_transfer(self):
        a=square(); b=square()
        old=masks(a,selected=[True,False],candidate=[True,True]); new=masks(b,selected=[False,False],candidate=[True,True])
        selected,matched,_,_,stats=m.transfer_guidance(a,b,old,new,arc_min=.85,arc_max=.95)
        self.assertListEqual(selected.tolist(),[True,False]); self.assertEqual(stats['matched_pct'],100)

    def test_unselected_and_out_of_roi_cannot_transfer(self):
        a=square(); b=square()
        old=masks(a,selected=[True,True],candidate=[True,True]); new=masks(b,selected=[False,False],candidate=[True,False],s=.5)
        sel,*_=m.transfer_guidance(a,b,old,new,arc_min=.85,arc_max=.95)
        self.assertEqual(int(sel.sum()),0)

    def test_different_curve_arclength_is_rejected(self):
        a=square(); b=square()
        old=masks(a,selected=[True,True],candidate=[True,True],s=.70); new=masks(b,selected=[False,False],candidate=[True,True],s=.9)
        sel,matched,*_=m.transfer_guidance(a,b,old,new,arc_min=.85,arc_max=.95)
        self.assertFalse(matched.any()); self.assertFalse(sel.any())

    def test_unaligned_geometry_rejected(self):
        a=square(); b=square(z=2.)
        old=masks(a,selected=[True,True],candidate=[True,True]); new=masks(b,selected=[False,False],candidate=[True,True])
        sel,matched,*_=m.transfer_guidance(a,b,old,new,arc_min=.85,arc_max=.95)
        self.assertFalse(matched.any()); self.assertFalse(sel.any())

    def test_reversed_normal_rejected(self):
        a=square(); b=square()
        b.faces=np.asarray(b.faces)[:,::-1]
        old=masks(a,selected=[True,True],candidate=[True,True]); new=masks(b,selected=[False,False],candidate=[True,True])
        sel,matched,*_=m.transfer_guidance(a,b,old,new,arc_min=.85,arc_max=.95)
        self.assertFalse(matched.any()); self.assertFalse(sel.any())

    def test_refuse_face_mismatch(self):
        a=square(); old=masks(a,selected=[True,True],candidate=[True,True]); old['face_centroids_xyz'][0,0]+=0.2
        with self.assertRaisesRegex(ValueError,'does not match'):
            m.require_aligned(a,old,'coarse')

    def test_staged_evidence_preserves_old_seed_and_other_parts(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td)
            co=square(); hi=square()
            oldroot=td/'old'; newroot=td/'new'; sdf=td/'fine_sdf'; oldmesh=td/'oldmesh.ply'
            co.export(oldmesh)
            (sdf/'riggable_parts').mkdir(parents=True); hi.export(sdf/'riggable_parts'/'body.ply')
            old=masks(co,selected=[True,False],candidate=[True,True]); new=masks(hi,selected=[False,False],candidate=[True,True])
            for root,parts in ((oldroot,{'body':old}),(newroot,{'body':new,'tentacle_t1':new})):
                for part,arrays in parts.items():
                    dir=root/'udf_cut_curve_local_0081'/part
                    dir.mkdir(parents=True)
                    np.savez_compressed(dir/'locality.npz',**arrays)
            args=SimpleNamespace(old_udf_shape=str(oldroot),new_udf_shape=str(newroot),old_mesh=str(oldmesh),
                new_sdf_shape=str(sdf),out=str(td/'guided'),part='body',arc_min=.85,arc_max=.95,
                normal_dot=.75,max_arc_delta=.015,distance_edge_factor=1.75,min_coverage_pct=75.,neighbors=12,batch_size=2)
            self.assertEqual(m.run(args),0)
            staged=td/'guided'/'staged_udf_shape'/'udf_cut_curve_local_0081'
            out=m.load_evidence(staged/'body'/'locality.npz')
            self.assertTrue(out['selected_faces'][0]); self.assertFalse(out['seed_faces'].any())
            self.assertTrue((staged/'tentacle_t1'/'locality.npz').exists())
            with self.assertRaises(FileExistsError): m.run(args)

if __name__ == '__main__': unittest.main()
