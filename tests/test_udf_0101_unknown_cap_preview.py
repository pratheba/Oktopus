import sys
import unittest
from pathlib import Path
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from udf_0101_unknown_cap_preview import cap_candidates


def grid(z):
    vertices = np.array([(x, y, z(x, y)) for y in range(8) for x in range(8)], float)
    faces=[]
    for j in range(7):
        for i in range(7):
            v=j*8+i
            faces.extend([(v,v+1,v+8),(v+1,v+9,v+8)])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


class TestUnknownCapPreview(unittest.TestCase):
    def setUp(self):
        self.m = grid(lambda x,y: 0)
        ctr = self.m.triangles_center
        self.unknown = ctr[:,0] <= 5
        self.remove = ctr[:,0] > 5
        self.part = np.ones(len(self.m.faces),bool)
        self.invalid = self.unknown.copy()

    def classify(self, mesh=None, **masks):
        m=mesh or self.m
        data=dict(part=self.part,unknown=self.unknown,
                  removed=self.remove, invalid=self.invalid)
        data.update(masks)
        return cap_candidates(face_adjacency=m.face_adjacency,
                              centers=m.triangles_center, min_faces=4,
                              min_fraction=.3, min_invalid_fraction=.6,
                              max_planarity=.075, **data)

    def test_flat_large_unknown_adjacent_to_removed_is_candidate(self):
        mask, report=self.classify()
        self.assertEqual(mask.sum(),self.unknown.sum())
        self.assertTrue(report[0]['preview_eligible'])

    def test_no_absent_neighbor_does_not_imply_removal(self):
        mask, report=self.classify(removed=np.zeros_like(self.remove))
        self.assertFalse(mask.any())
        self.assertIn('no_confirmed_removed_neighbor',report[0]['reasons'])

    def test_valid_ambiguous_is_not_evidence_of_missing_surface(self):
        mask, report=self.classify(invalid=np.zeros_like(self.invalid))
        self.assertFalse(mask.any())

    def test_curved_unknown_not_accepted(self):
        m=grid(lambda x,y: 2*np.sin(.65*x))
        mask, report=self.classify(mesh=m)
        self.assertFalse(mask.any())
        self.assertIn('not_planar',report[0]['reasons'])

    def test_does_not_promote_other_parts(self):
        part=self.part.copy()
        part[:]=False
        mask, report=self.classify(part=part)
        self.assertFalse(mask.any())
        self.assertEqual(report,[])

    def test_unknown_removed_overlap_rejected(self):
        with self.assertRaises(ValueError):
            self.classify(removed=self.unknown.copy())


if __name__=='__main__':
    unittest.main()
