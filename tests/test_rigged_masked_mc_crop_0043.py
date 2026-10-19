import unittest
import numpy as np
from ngc.handle_utils.mc_utils import MCGrid


class MaskedCropTest(unittest.TestCase):
    def test_crop_preserves_full_masked_mc_triangles(self):
        grid = MCGrid({'reso': 38, 'size': 1.0, 'level': 0.0})
        n = grid.reso + 1
        x = np.linspace(-1, 1, n)
        a, b, c = np.meshgrid(x, x, x, indexing='ij')
        sdf = np.sqrt((a-.25)**2 + (b+.20)**2 + (c-.1)**2) - .43
        valid = (a > -.05) & (a < .8) & (b > -.6) & (b < .5)
        idx = np.flatnonzero(valid)
        grid.update_grid(sdf.ravel()[idx], idx, mark=True)
        old = grid.extract_mesh_masked()
        new = grid.extract_mesh_masked_cropped()
        self.assertGreater(len(old.faces), 0)
        self.assertEqual(len(old.faces), len(new.faces))
        self.assertEqual(len(old.vertices), len(new.vertices))
        # MC may enumerate triangles in different order after translating the
        # crop. Sorting vertices compares numerical geometry, not face order.
        old_v = np.array(sorted(map(tuple, np.round(old.vertices, 12))))
        new_v = np.array(sorted(map(tuple, np.round(new.vertices, 12))))
        np.testing.assert_allclose(old_v, new_v, atol=1e-7, rtol=0)


if __name__ == '__main__':
    unittest.main()
