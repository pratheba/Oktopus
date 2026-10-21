"""Regression for 0054 choosing a tiny tip fragment instead of the real limb."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from offline_tentacle_motion_0051 import (sample_curve, bind_connected_avatar,
                                         isolate_target_surface_by_topology,
                                         bend_positions)


def ribbon(verts, faces, x0, x1, n, y, z=0):
    ids = []
    for x in np.linspace(x0, x1, n):
        j = len(verts)
        verts.extend(([x, y-.025, z], [x, y+.025, z]))
        ids.extend([j, j+1])
    for j in range(n-1):
        a,b,c,d = ids[2*j:2*j+2] + ids[2*j+2:2*j+4]
        faces.append([a,b,c]); faces.append([b,d,c])
    return np.asarray(ids,dtype=int)


class TopologyRecovery0055(unittest.TestCase):
    def test_tiny_close_to_tip_fragment_does_not_freeze_main_tentacle(self):
        verts, faces = [], []
        target = ribbon(verts, faces, .10,.98, 41, y=.07)
        # This decoy is closer to the skeleton TIP than the actual surface.
        tiny = ribbon(verts, faces, .89,1.,5, y=.005)
        verts=np.asarray(verts,float); faces=np.asarray(faces,int)
        u, c = sample_curve(np.array([[0.,0.,0.],[1.,0.,0.]]), 128)
        competitor = np.array([[0.,1.,0.],[1.,1.,0.]])
        s, offsets, selected = bind_connected_avatar(
            verts, c, u, competitor, faces=faces,
            distance_full=.15, distance_zero=.30,
            owner_band=.035,root_hold=.12,root_blend_end=.30)
        self.assertGreater(np.mean(selected[target]>.1), .60)
        self.assertTrue(np.all(selected[tiny] == 0))
        posed, _ = bend_positions(c,s,offsets,angle_deg=30,bend_axis=[0,0,1])
        result=verts+(posed-verts)*selected[:,None]
        self.assertGreater(np.max(np.linalg.norm(result[target]-verts[target],axis=1)),.1)
        np.testing.assert_array_equal(result[tiny], verts[tiny])

    def test_zero_weight_seam_does_not_split_real_surface(self):
        v,f=[],[]
        target=ribbon(v,f,.1,.96,45,y=.07)
        neighbor=ribbon(v,f,.1,.96,45,y=.22)
        v=np.asarray(v,float); f=np.asarray(f,int)
        u,c=sample_curve(np.array([[0.,0.,0.],[1.,0.,0.]]), 128)
        s=np.clip(v[:,0],0,1)
        weights=np.ones(len(v))
        weights[neighbor]=.1  # Simulates original leakage onto nearby surface
        weights[(s>.48)&(s<.55)]=0.0  # A seam splits the 0054 candidate graph
        weights[s<=.12]=0.0
        restricted=isolate_target_surface_by_topology(v,f,c,s,weights,
                         root_hold=.12,distance_zero=.30)
        self.assertTrue(np.all(restricted[neighbor]==0))
        good=target[(s[target]>.60)]
        self.assertTrue(np.all(restricted[good]>.1))
        left=target[(s[target]>.30)&(s[target]<.45)]
        self.assertTrue(np.all(restricted[left]>.1))
        self.assertEqual(len(restricted),len(v))

    def test_fails_loudly_if_no_movable_surface(self):
        v,f=[],[]
        ribbon(v,f,.1,.98,41,y=.07)
        v=np.asarray(v,float);f=np.asarray(f,int)
        u,c=sample_curve(np.array([[0.,0.,0.],[1.,0.,0.]]),128)
        with self.assertRaisesRegex(ValueError,'no nonzero'):
            isolate_target_surface_by_topology(v,f,c,v[:,0],np.zeros(len(v)),root_hold=.12)


if __name__ == '__main__':unittest.main()
