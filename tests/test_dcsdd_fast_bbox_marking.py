"""DCSDD opt-in fast broad marking must be identical to legacy voxel marking."""

import numpy as np
import pytest

from ngc.handle_utils.mesh_utils import CylindersMesh
from utils.mc_utils import MCGrid as AppMCGrid
from ngc.handle_utils.mc_utils import MCGrid as NgcMCGrid


@pytest.mark.parametrize("grid_cls", [AppMCGrid, NgcMCGrid])
@pytest.mark.parametrize("reso", [12, 32, 64])
@pytest.mark.parametrize("extend_bbox", [False, True])
@pytest.mark.parametrize("pattern", ["inside", "clipped", "outside", "narrow"])
def test_fast_bbox_same_candidates_as_meshgrid(grid_cls, reso, extend_bbox, pattern):
    grid = grid_cls({"reso": reso, "size": 1.2, "level": 0.0})
    rng = np.random.default_rng(130 + reso)
    cylinder = CylindersMesh()
    points = rng.uniform(-0.5, 0.5, size=(18, 20, 3))
    if pattern == "clipped":
        points[:, :, 0] += 1.25
    elif pattern == "outside":
        points[:, :, 0] += 4.0
    elif pattern == "narrow":
        points = points * 0.02
    cylinder.cylinder_samples = points

    legacy_xyz, legacy_kidx = cylinder.filter_grid(
        grid, extend_bbox=extend_bbox
    )
    fast_xyz, fast_kidx = cylinder.filter_grid(
        grid, extend_bbox=extend_bbox, fast_bbox=True
    )
    np.testing.assert_array_equal(fast_kidx, legacy_kidx)
    np.testing.assert_array_equal(fast_xyz, legacy_xyz)


def test_fast_bbox_same_for_overlapping_large_boxes():
    grid = AppMCGrid({"reso": 96, "size": 1.2, "level": 0.0})
    cylinder = CylindersMesh()
    rng = np.random.default_rng(4)
    # Reproduces the high-overlap broad segments that made the DCSDD run slow.
    cylinder.cylinder_samples = rng.uniform(-1.1, 1.1, (21, 24, 3))
    old_pts, old_idx = cylinder.filter_grid(grid)
    new_pts, new_idx = cylinder.filter_grid(grid, fast_bbox=True)
    np.testing.assert_array_equal(old_idx, new_idx)
    np.testing.assert_array_equal(old_pts, new_pts)
