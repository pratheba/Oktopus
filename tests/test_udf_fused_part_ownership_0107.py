"""0107: SDF-only fused-part ownership, no saved UDF evidence or model/GPU."""
import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/app'))
from udf_fused_part_ownership_0107 import route_fused_faces_from_sdf_parts, mesh_from_mask


class TestSdfOnlyFusedOwnership0107(unittest.TestCase):
    def test_exact_part_identity(self):
        box = trimesh.creation.box()
        ids, report = route_fused_faces_from_sdf_parts(box, {'body': box})
        np.testing.assert_array_equal(ids, np.zeros(len(box.faces), np.int32))
        self.assertEqual(report['unmatched_faces'], 0)
        self.assertFalse(report['uses_0092_evidence'])
        self.assertEqual(report['part_face_counts']['body'], len(box.faces))

    def test_two_spatially_separate_parts(self):
        body = trimesh.creation.box(extents=[1, 1, 1])
        tentacle = body.copy()
        tentacle.apply_translation([4, 0, 0])
        fused = trimesh.util.concatenate([body, tentacle])
        ids, report = route_fused_faces_from_sdf_parts(
            fused, {'body': body, 'tentacle': tentacle})
        np.testing.assert_array_equal(ids[:len(body.faces)], 0)
        np.testing.assert_array_equal(ids[len(body.faces):], 1)
        self.assertEqual(report['unmatched_faces'], 0)
        self.assertEqual(report['part_face_counts']['tentacle'], len(tentacle.faces))

    def test_backside_nearby_never_claims_owner(self):
        body = trimesh.creation.box()
        inverted = body.copy()
        inverted.invert()
        with self.assertRaisesRegex(ValueError, 'unmatched'):
            route_fused_faces_from_sdf_parts(
                inverted, {'body': body}, distance_edge_factor=.1,
                max_unmatched_fraction=.05)

    def test_distant_geometry_fails_closed(self):
        body = trimesh.creation.box()
        other = body.copy()
        other.apply_translation([10, 0, 0])
        with self.assertRaisesRegex(ValueError, 'unmatched'):
            route_fused_faces_from_sdf_parts(other, {'body': body})

    def test_optional_unmatched_preserved_as_minus_one(self):
        body = trimesh.creation.box()
        other = body.copy()
        other.apply_translation([20, 0, 0])
        joined = trimesh.util.concatenate([body, other])
        ids, report = route_fused_faces_from_sdf_parts(
            joined, {'body': body}, max_unmatched_fraction=.6)
        self.assertTrue(np.all(ids[:len(body.faces)] == 0))
        self.assertTrue(np.all(ids[len(body.faces):] == -1))
        self.assertEqual(report['unmatched_faces'], len(other.faces))

    def test_invalid_mesh_rejected(self):
        body = trimesh.creation.box()
        with self.assertRaisesRegex(ValueError, 'at least one'):
            route_fused_faces_from_sdf_parts(body, {})
        with self.assertRaisesRegex(ValueError, 'distance_edge_factor'):
            route_fused_faces_from_sdf_parts(body, {'body': body}, distance_edge_factor=-1)

    def test_masked_mesh(self):
        box = trimesh.creation.box()
        mask = np.arange(len(box.faces)) < 3
        m = mesh_from_mask(box, mask)
        self.assertEqual(len(m.faces), 3)
        self.assertEqual(len(box.faces), 12)

    def test_parity_with_0092_geometry_router_for_identity(self):
        # Compare with the historical correspondence algorithm on a synthetic
        # case, without reading or needing any 0092 output evidence.
        from udf_cut_fused_exterior_0092 import transfer_cut_to_fused
        box = trimesh.creation.box()
        tube = box.copy()
        tube.apply_translation([3, 0, 0])
        fused = trimesh.util.concatenate([box, tube])
        names = {'body': box, 'tentacle': tube}
        new, _ = route_fused_faces_from_sdf_parts(fused, names)
        _, old, _, _ = transfer_cut_to_fused(
            fused, {n: (m, np.zeros(len(m.faces), bool)) for n, m in names.items()})
        np.testing.assert_array_equal(new, old)


if __name__ == '__main__':
    unittest.main()
