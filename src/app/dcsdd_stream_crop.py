"""Crop-first grid composition for opt-in independent DCSDD streaming.

All indices here are *global* MCGrid x-fast indices. The negative SDF bbox is
computed from final, globally calibrated SDF values, then only its padded
cube is allocated. Values not written inside that cube retain MCGrid's +10
background and empty=True semantics. This module has no effect on the
non-streaming and non-cropped pipelines.
"""
import numpy as np


class NegativeVoxelBounds:
    """Track exact global grid bbox of finite, evaluated negative predictions."""
    def __init__(self, n1, level=0.0):
        self.n1 = int(n1)
        self.level = float(level)
        self.lo = np.array([self.n1] * 3, dtype=np.int64)
        self.hi = np.array([-1] * 3, dtype=np.int64)
        self.count = 0

    def add(self, indices, values):
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        values = np.asarray(values).reshape(-1)
        if len(indices) != len(values):
            raise ValueError('indices and values have different lengths')
        if indices.size and (indices.min() < 0 or indices.max() >= self.n1 ** 3):
            raise ValueError('Global voxel index outside grid')
        neg = indices[np.isfinite(values) & (values < self.level)]
        if neg.size == 0:
            return
        n = self.n1
        x = neg % n
        y = (neg // n) % n
        z = neg // (n * n)
        self.lo = np.minimum(self.lo, [x.min(), y.min(), z.min()])
        self.hi = np.maximum(self.hi, [x.max(), y.max(), z.max()])
        self.count += int(len(neg))

    def bounds(self):
        if not self.count:
            raise RuntimeError('No evaluated negative SDF samples')
        return self.lo.copy(), self.hi.copy()


def make_crop_grid(global_grid, starts, side, *, max_ram_samples=100_000_000):
    """Return a real MCGrid with identical world-space voxel spacing.

    Use ordinary RAM for a modest crop, avoiding slow random Lustre memmap
    writes. Larger crops retain the disk-backed option. Never initialize the
    full global grid. MCGrid's x-fast ordering is unchanged.
    """
    from utils.mc_utils import MCGrid

    starts = np.asarray(starts, dtype=np.int64)
    side = int(side)
    if side < 2 or starts.shape != (3,) or np.any(starts < 0):
        raise ValueError('Invalid crop cube')
    full = int(global_grid.reso) + 1
    if np.any(starts + side > full):
        raise ValueError('Crop cube lies outside the global grid')
    step = float(global_grid.grid_config['step'])
    origin = np.asarray(global_grid.grid_config['origin'], dtype=np.float64)
    # MCGrid derives step from 2*size/reso. Algebraically identical step;
    # explicitly restore the original scalar to avoid roundoff drift.
    size = step * (side - 1) / 2.0
    memmap_dir = getattr(global_grid, 'memmap_dir', None) if side ** 3 > max_ram_samples else None
    grid = MCGrid({'reso': side - 1, 'size': size,
                   'level': global_grid.grid_config['level'],
                   'memmap_dir': memmap_dir})
    grid.grid_config['origin'] = origin + starts * step
    grid.grid_config['step'] = step
    grid.crop_start_global = starts.copy()
    grid.global_resolution = global_grid.reso
    return grid


def write_global_chunk_to_crop(crop_grid, global_indices, values, *, n1, starts):
    """Write samples within the crop, preserving SDF, validity and index order."""
    idx = np.asarray(global_indices, dtype=np.int64).reshape(-1)
    values = np.asarray(values).reshape(-1)
    if len(idx) != len(values):
        raise ValueError('indices and values have different lengths')
    if len(idx) == 0:
        return 0
    n1 = int(n1)
    starts = np.asarray(starts, dtype=np.int64)
    side = int(crop_grid.reso) + 1
    x = idx % n1 - starts[0]
    y = (idx // n1) % n1 - starts[1]
    z = idx // (n1 * n1) - starts[2]
    kept = ((x >= 0) & (x < side) &
            (y >= 0) & (y < side) &
            (z >= 0) & (z < side))
    if not np.any(kept):
        return 0
    local = x[kept] + side * (y[kept] + side * z[kept])
    crop_grid.update_grid(values[kept], local, mode='overwrite', mark=True)
    return int(kept.sum())
