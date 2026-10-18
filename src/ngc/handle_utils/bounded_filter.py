"""Memory-bounded, exact voxel-box union for an MCGrid with x-fast indices.

Only used by the opt-in DCSDD adaptation path.  Never creates a full
(reso+1)**3 Boolean mask, and never changes the broad-phase box extents.
"""
import numpy as np


def _box_ranges(cylinder_samples, mc_grid, extend_bbox=False):
    n = int(mc_grid.reso)
    origin = np.asarray(mc_grid.grid_config["origin"])
    step = float(mc_grid.grid_config["step"])
    boxes = []
    for i in range(len(cylinder_samples) - 1):
        part = cylinder_samples[i:i + 2].reshape(-1, 3)
        hi = np.floor((part.max(axis=0) - origin) / step) + 1
        lo = np.floor((part.min(axis=0) - origin) / step)
        if extend_bbox:
            lo -= 1
            hi += 1
        # Exactly MCGrid.generate_samples's asymmetric clipping.
        lo = np.maximum(lo, 0).astype(np.int64)
        hi = np.minimum(hi, n).astype(np.int64)
        if np.all(lo <= hi):
            boxes.append((lo, hi))
    return boxes


def iter_bbox_kidx(cylinder_samples, mc_grid, *, extend_bbox=False,
                   batch_size=262144, max_slab_voxels=4_194_304):
    """Yield globally ascending flat indices of the union of cylinder AABBs.

    Bounded by max_slab_voxels for the mark array, plus batch_size for output.
    No coordinate samples or full-grid masks are retained between slabs.
    """
    if batch_size <= 0 or max_slab_voxels <= 0:
        raise ValueError("batch_size and max_slab_voxels must be positive")
    boxes = _box_ranges(cylinder_samples, mc_grid, extend_bbox)
    if not boxes:
        return
    los = np.stack([b[0] for b in boxes])
    his = np.stack([b[1] for b in boxes])
    lo = los.min(axis=0)
    hi = his.max(axis=0)
    nx, ny = (hi[:2] - lo[:2] + 1).tolist()
    nz = int(hi[2] - lo[2] + 1)
    # A single XY plane can exceed max_slab_voxels; in that case we still
    # need one XY plane (worst ~4 MiB at 2048, not 8.6 GiB).
    dz = max(1, int(max_slab_voxels // (nx * ny)))
    n1 = int(mc_grid.reso) + 1
    for slab_start in range(0, nz, dz):
        z0 = int(lo[2] + slab_start)
        z1 = min(int(hi[2]), z0 + dz - 1)
        marks = np.zeros((nx, ny, z1 - z0 + 1), dtype=np.bool_, order='F')
        for b_lo, b_hi in boxes:
            if b_hi[2] < z0 or b_lo[2] > z1:
                continue
            marks[
                b_lo[0]-lo[0]:b_hi[0]-lo[0]+1,
                b_lo[1]-lo[1]:b_hi[1]-lo[1]+1,
                max(int(b_lo[2]), z0)-z0:min(int(b_hi[2]), z1)-z0+1
            ] = True
        local = np.flatnonzero(marks.ravel(order='F'))
        del marks
        for start in range(0, len(local), batch_size):
            flat = local[start:start+batch_size]
            x = flat % nx
            yz = flat // nx
            y = yz % ny
            z = yz // ny
            yield ((x + lo[0]) + n1 * ((y + lo[1]) + n1 * (z + z0))).astype(np.int64)
