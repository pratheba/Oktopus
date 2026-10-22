import unittest
import numpy as np
import trimesh

from app.sdf_puffer_sparse_groups import merge_sparse_field_min
from app.sdf_fused_report_0087 import fused_surface_report


class FakeGrid:
    def __init__(self, n=10):
        self.val_grid = np.full(n, 10., dtype=np.float32)
        self.empty_marks = np.ones(n, dtype=bool)


class TestSparseMinUnion(unittest.TestCase):
    def test_min_and_unknown(self):
        g = FakeGrid()
        self.assertEqual(merge_sparse_field_min(g, [1, 2, 4], [.5, -.3, .7], batch_size=2), 3)
        merge_sparse_field_min(g, [2, 3, 4], [.2, -.8, -.1], batch_size=1)
        np.testing.assert_allclose(g.val_grid[[1, 2, 3, 4]], [.5, -.3, -.8, -.1], atol=1e-6)
        self.assertTrue(np.all(g.empty_marks[[0, 5, 6, 7, 8, 9]]))
        self.assertTrue(np.all(~g.empty_marks[[1, 2, 3, 4]]))

    def test_preserves_negative_voxels_on_second_positive_part(self):
        g = FakeGrid()
        merge_sparse_field_min(g, [2], [-.5])
        merge_sparse_field_min(g, [2], [.5])
        self.assertEqual(float(g.val_grid[2]), -.5)

    def test_invalid_sparse_inputs_fail(self):
        g = FakeGrid()
        for idx, vals in [([2, 2], [0, 1]), ([3, 1], [0, 1]), ([11], [0]), ([1], [np.nan])]:
            with self.assertRaises((ValueError, IndexError)):
                merge_sparse_field_min(g, idx, vals)
        with self.assertRaises(ValueError):
            merge_sparse_field_min(g, [0], [0], batch_size=0)

    def test_independent_meshes_no_longer_just_concatenated(self):
        # Fused SDF zero level set should have one boundary for overlapping
        # solids. Here verify the diagnostic counts on a *single* output mesh.
        box = trimesh.creation.box()
        report = fused_surface_report(box)
        self.assertEqual(report['face_components'], 1)
        self.assertEqual(report['boundary_edges'], 0)
        self.assertEqual(report['nonmanifold_edges'], 0)
        self.assertTrue(report['watertight'])

    def test_min_union_excludes_internal_sphere_surfaces(self):
        # Extract from a genuine scalar field union, not triangle concatenation.
        from skimage.measure import marching_cubes
        a = np.linspace(-1, 1, 44, dtype=np.float32)
        x, y, z = np.meshgrid(a, a, a, indexing='ij')
        body = np.sqrt((x + .22)**2 + y**2 + z**2) - .6
        leaf = np.sqrt((x - .22)**2 + y**2 + z**2) - .6
        val = np.minimum(body, leaf)
        verts, faces, _normal, _values = marching_cubes(
            val, level=0., spacing=(float(a[1]-a[0]),) * 3)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        diagnostics = fused_surface_report(mesh)
        self.assertEqual(diagnostics['face_components'], 1)
        self.assertEqual(diagnostics['boundary_edges'], 0)
        self.assertEqual(diagnostics['nonmanifold_edges'], 0)
        # Actual fused sphere exterior has fewer faces than two separately
        # extracted complete sphere meshes; their interior caps are absent.
        _v, f_body, _, _ = marching_cubes(body, level=0.)
        _v, f_leaf, _, _ = marching_cubes(leaf, level=0.)
        self.assertLess(len(mesh.faces), len(f_body) + len(f_leaf))

    def test_open_mesh_reports_boundary(self):
        box = trimesh.creation.box()
        mesh = trimesh.Trimesh(vertices=box.vertices, faces=box.faces[:-1], process=False)
        self.assertGreater(fused_surface_report(mesh)['boundary_edges'], 0)


if __name__ == '__main__':
    unittest.main()
