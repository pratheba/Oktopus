"""0049: optional sparse composition for a completed puffer blend group.

The incoming arrays are the FINAL corrected SDF values from the existing
per-adaptation pipeline (including avatar carving/snug). Only dense SDF grid
allocation/extraction is replaced, not curve mapping or correction semantics.

This is an experimental opt-in: groups must be processed in adaptation order.
It avoids a full (reso+1)^3 grid per group and per individual piece.
"""
import numpy as np
from app.dcsdd_stream_crop import NegativeVoxelBounds, make_crop_grid, write_global_chunk_to_crop


def crop_final_piece(global_grid, indices, sdf, *, band, bounds_fn, batch_size=262144):
    """Replicate DCSDD's exact negative-bbox crop directly from sparse samples."""
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    sdf = np.asarray(sdf, dtype=np.float64).reshape(-1)
    if indices.size != sdf.size:
        raise ValueError('Indices/SDF length mismatch')
    n1 = int(global_grid.reso) + 1
    bounds = NegativeVoxelBounds(n1, level=0.0)
    for i in range(0, len(indices), batch_size):
        bounds.add(indices[i:i+batch_size], sdf[i:i+batch_size])
    lo, hi = bounds.bounds()
    step = float(global_grid.grid_config['step'])
    pad = int(np.ceil(float(band) / step)) + 2
    starts, side = bounds_fn(lo, hi, n1, pad)
    crop = make_crop_grid(global_grid, starts, side)
    for i in range(0, len(indices), batch_size):
        write_global_chunk_to_crop(crop, indices[i:i+batch_size], sdf[i:i+batch_size],
                                   n1=n1, starts=starts)
    return crop, starts, side


class SparseSDFGroup:
    """Accumulate a group by sorted global indices, preserving overlap rules.

    No implicit Boolean between distinct groups. This keeps all evaluated
    samples (including positive support-clamped ones), not just near-zero SDF.
    """
    def __init__(self, blend_delta=0.0):
        self.indices = np.empty(0, dtype=np.int64)
        self.values = np.empty(0, dtype=np.float64)
        self.blend_delta = float(blend_delta)
        self.pieces = 0

    def add(self, indices, values, *, post_pair=None, source_s=None,
            overlap_crossfade=None, smooth_union=None):
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        val = np.asarray(values, dtype=np.float64).reshape(-1)
        if len(idx) != len(val):
            raise ValueError('SDF group index/value length mismatch')
        if len(idx) and (idx[0] < 0 or np.any(idx[1:] <= idx[:-1])):
            raise ValueError('Sparse SDF requires strictly increasing, unique sample indices')
        if not self.pieces:
            self.indices = idx.copy()
            self.values = val.copy()
            self.pieces = 1
            return {'overlap': 0, 'total': len(idx)}
        old = self.indices
        old_val = self.values
        # searchsorted also permits a 0-length incoming part.
        pos = np.searchsorted(old, idx)
        overlap = np.zeros(len(idx), dtype=bool)
        valid = pos < len(old)
        overlap[valid] = old[pos[valid]] == idx[valid]
        n_overlap = int(np.count_nonzero(overlap))
        if n_overlap:
            source = None
            if post_pair is not None:
                if overlap_crossfade is None or source_s is None:
                    raise ValueError('Post-SDF seam requires source_s and crossfade function')
                source = np.asarray(source_s, dtype=np.float64).reshape(-1)
                if len(source) != len(idx):
                    raise ValueError('Post-SDF seam source length mismatch')
                old_val[pos[overlap]] = overlap_crossfade(
                    old_val[pos[overlap]], val[overlap], source[overlap], post_pair)
            elif self.blend_delta > 0.0:
                if smooth_union is None:
                    raise ValueError('Smooth union callback missing')
                old_val[pos[overlap]] = smooth_union(
                    old_val[pos[overlap]], val[overlap], self.blend_delta)
            # Otherwise existing grid values win, as in legacy accumulation.
        if np.any(~overlap):
            novel = idx[~overlap]
            novel_val = val[~overlap]
            # All novel voxel indices are disjoint; merge deterministically.
            merged = np.concatenate((old, novel))
            order = np.argsort(merged, kind='stable')
            self.indices = merged[order]
            self.values = np.concatenate((old_val, novel_val))[order]
        self.pieces += 1
        return {'overlap': n_overlap, 'total': len(self.indices)}
