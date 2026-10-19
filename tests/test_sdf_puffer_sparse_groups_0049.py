"""0049: dense-versus-sparse composition and DCSDD crop parity."""
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from app.sdf_puffer_sparse_groups import SparseSDFGroup, crop_final_piece
from app.agent_3dvec_sdf import AgentSDF
from app.agent_3dvec_sdf_dc import AgentSDFDC
from app.post_sdf_seam_blend import overlap_crossfade_sdf
from utils.mc_utils import MCGrid


class SparsePuffer0049(unittest.TestCase):
    def test_three_section_smooth_union_matches_legacy_grid(self):
        rng = np.random.default_rng(491)
        group = SparseSDFGroup(blend_delta=.005)
        dense = np.full(1000, 10.0)
        seen = np.zeros(1000, dtype=bool)
        for _ in range(3):
            idx = np.sort(rng.choice(1000, size=300, replace=False))
            vals = rng.normal(0, .2, size=300)
            # Exactly the pre-0049 group merge logic.
            was = seen[idx]
            if np.any(was):
                dense[idx[was]] = AgentSDF.smooth_union_sdf(
                    dense[idx[was]], vals[was], .005)
            dense[idx[~was]] = vals[~was]
            seen[idx] = True
            group.add(idx, vals, smooth_union=AgentSDF.smooth_union_sdf)
        np.testing.assert_array_equal(group.indices, np.flatnonzero(seen))
        np.testing.assert_allclose(group.values, dense[seen], atol=0, rtol=0)

    def test_post_sdf_seam_and_stable_legacy_first_wins(self):
        rng = np.random.default_rng(492)
        pair = SimpleNamespace(source_lo=.2, source_hi=.5, edge_hold_fraction=.1)
        a = np.arange(0, 50, 2, dtype=np.int64)
        b = np.arange(15, 55, dtype=np.int64)
        av, bv = rng.normal(size=len(a)), rng.normal(size=len(b))
        coords = rng.uniform(0, 1, size=len(b))
        dense = np.full(60, 10.0)
        present = np.zeros(60, dtype=bool)
        dense[a] = av; present[a] = True
        overlap = present[b]
        dense[b[overlap]] = overlap_crossfade_sdf(
            dense[b[overlap]], bv[overlap], coords[overlap], pair)
        dense[b[~overlap]] = bv[~overlap]
        present[b] = True
        group = SparseSDFGroup()
        group.add(a, av)
        group.add(b, bv, post_pair=pair, source_s=coords,
                  overlap_crossfade=overlap_crossfade_sdf)
        np.testing.assert_array_equal(group.indices, np.flatnonzero(present))
        np.testing.assert_array_equal(group.values, dense[present])
        fallback = SparseSDFGroup()
        fallback.add(a, av)
        fallback.add(b, bv)
        np.testing.assert_array_equal(fallback.values[np.searchsorted(fallback.indices, a)], av)

    def test_cropped_individual_identical_to_fullgrid_dc_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            full = MCGrid({'reso': 40, 'size': .6, 'level': 0.0, 'memmap_dir':tmp})
            full.clear_grid()
            rng = np.random.default_rng(493)
            ids = np.sort(rng.choice(41**3, size=16000, replace=False))
            pts = full.idx2pts(ids)
            vals = np.sqrt(np.sum((pts - [.03, -.02, .01])**2, axis=1)) - .18
            full.update_grid(vals, ids, mark=True, mode='overwrite')
            crop, start, side = crop_final_piece(
                full, ids, vals, band=AgentSDFDC._dcsdd_band(
                    full.grid_config['step'], {}),
                bounds_fn=AgentSDFDC._crop_cube_bounds)
            probe = AgentSDFDC()
            baseline_s, baseline_u, baseline_empty, n = probe._grid_to_S_U(
                full, {'dcsdd_crop_grid': True})
            cropped_s, cropped_u, cropped_empty, n2 = probe._grid_to_S_U(
                crop, {'dcsdd_crop_grid': False})
            self.assertEqual(side, n)
            self.assertEqual(n, n2)
            np.testing.assert_array_equal(cropped_s, baseline_s)
            np.testing.assert_array_equal(cropped_empty, baseline_empty)
            np.testing.assert_allclose(cropped_u, baseline_u, atol=2e-15)

    def test_duplicate_voxels_rejected(self):
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            SparseSDFGroup().add([1, 1], [-.5, .5])


if __name__ == '__main__':
    unittest.main()
