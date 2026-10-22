"""0103: shared-edge selection must not jump between close folded surfaces."""
import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from udf_0103_connected_unknown_preview import choose_connected_unknown


def plane(z=0., nx=8, ny=8):
    v = np.array([(float(x), float(y), z) for y in range(ny) for x in range(nx)])
    f = []
    for y in range(ny - 1):
        for x in range(nx - 1):
            i = y * nx + x
            f += [(i, i + 1, i + nx), (i + 1, i + nx + 1, i + nx)]
    return trimesh.Trimesh(vertices=v, faces=f, process=False)


def classify(mesh, *, unknown, removed, supported, invalid, part=None, **kwargs):
    n = len(mesh.faces)
    if part is None:
        part = np.ones(n, bool)
    return choose_connected_unknown(
        face_adjacency=mesh.face_adjacency,
        face_area=mesh.area_faces,
        part=part, unknown=unknown, removed=removed,
        supported=supported, invalid=invalid,
        min_area_fraction=0.01, min_invalid_fraction=.60,
        min_removed_contact_fraction=.80,
        min_removed_contact_edges=2, **kwargs)


class Test0103ConnectedUnknown(unittest.TestCase):
    def setUp(self):
        self.m = plane()
        x = self.m.triangles_center[:, 0]
        self.unknown = x <= 5
        self.removed = ~self.unknown
        self.supported = np.zeros(len(x), bool)
        self.invalid = self.unknown.copy()

    def go(self, mesh=None, **change):
        base = dict(unknown=self.unknown, removed=self.removed,
                    supported=self.supported, invalid=self.invalid)
        base.update(change)
        return classify(mesh if mesh is not None else self.m, **base)

    def test_selects_nonplanar_connected_unknown(self):
        # Geometry is curved; planarity is intentionally not a gate.
        m = self.m.copy()
        m.vertices[:, 2] = 2 * np.sin(0.7 * m.vertices[:, 0])
        mask, reports = self.go(mesh=m)
        self.assertEqual(int(mask.sum()), int(self.unknown.sum()))
        self.assertTrue(reports[0]['preview_eligible'])

    def test_strong_supported_boundary_blocks_preview(self):
        supported = self.removed.copy()
        removed = np.zeros_like(self.removed)
        mask, reports = self.go(supported=supported, removed=removed)
        self.assertFalse(mask.any())
        self.assertIn('boundary_touches_supported_surface', reports[0]['reasons'])

    def test_valid_unknown_not_promoted(self):
        mask, reports = self.go(invalid=np.zeros_like(self.invalid))
        self.assertFalse(mask.any())
        self.assertIn('insufficient_invalid_udf_evidence', reports[0]['reasons'])

    def test_does_not_cross_close_disconnected_fold(self):
        # The second sheet is only 1e-6 units away but has no shared vertices.
        a, b = self.m, plane(z=1e-6)
        combined = trimesh.util.concatenate((a, b))
        n = len(a.faces)
        unknown = np.r_[self.unknown, np.ones(n, bool)]
        removed = np.r_[self.removed, np.zeros(n, bool)]
        invalid = unknown.copy()
        supported = np.zeros(len(unknown), bool)
        mask, _ = classify(combined, unknown=unknown, removed=removed,
                           supported=supported, invalid=invalid)
        self.assertEqual(int(mask[:n].sum()), int(self.unknown.sum()))
        self.assertFalse(mask[n:].any())

    def test_does_not_cross_part_boundary(self):
        m = self.m
        # A shared edge exists but neighbors are routed to a different part.
        part = self.unknown.copy()
        mask, _ = self.go(part=part)
        self.assertFalse(mask.any())

    def test_overlapping_categories_rejected(self):
        with self.assertRaises(ValueError):
            self.go(supported=self.unknown)

    def test_no_removed_neighbors_blocks(self):
        mask, rep = self.go(removed=np.zeros_like(self.removed))
        self.assertFalse(mask.any())
        self.assertIn('few_confirmed_removed_contacts', rep[0]['reasons'])

    def test_physical_area_is_resolution_independent(self):
        coarse = self.m
        fine = plane(nx=16, ny=16)
        # Threshold high enough that a tiny *area* region fails both,
        # independent of face count.
        for m in (coarse, fine):
            n = len(m.faces)
            small = np.zeros(n, bool)
            small[0] = True
            ids = m.face_adjacency[np.any(m.face_adjacency == 0, axis=1)]
            rm = np.zeros(n, bool)
            if len(ids): rm[ids[0, 1] if ids[0, 0] == 0 else ids[0, 0]] = True
            # Check classification cannot select unconfirmed, tiny patch.
            keep, _ = classify(m, unknown=small, removed=rm,
                               supported=np.zeros(n, bool), invalid=small)
            self.assertFalse(keep.any())


if __name__ == '__main__':
    unittest.main()
