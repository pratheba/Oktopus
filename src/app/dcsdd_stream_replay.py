"""Opt-in 0038 Boot 1 replay without repeated curve localization.

The prediction pass stores exactly the accepted world-grid index, accessory
curve coordinate and (once-computed) radial norm for each accepted sample.
These are precisely the inputs required for *post*-calibration composition;
we never change inference ordering, support calibration, or voxel indexing.
"""
from pathlib import Path
from time import perf_counter

import numpy as np


REPLAY_DTYPE = np.dtype([
    ('index', np.int64),
    ('coord', np.float64),
    ('rho', np.float64),
])


class BootReplayMetadata:
    def __init__(self, *, n, outdir):
        self.total = int(n)
        if self.total <= 0:
            raise ValueError('Replay requires a nonempty accepted sample set')
        self.path = Path(outdir) / 'replay_metadata.bin'
        self.records = np.memmap(str(self.path), mode='w+',
                                 dtype=REPLAY_DTYPE, shape=(self.total,))
        self.written = 0
        self.write_s = 0.0

    @property
    def bytes(self):
        return self.total * REPLAY_DTYPE.itemsize

    def push(self, acc, indices):
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        n = len(indices)
        if self.written + n > self.total:
            raise ValueError('More accepted samples than counted')
        coords = np.asarray(acc['coords']).reshape(-1)
        local = np.asarray(acc['samples_local'])
        if len(coords) != n or local.shape != (n, 3):
            raise ValueError('Invalid replay localization batch')
        t0 = perf_counter()
        # Compute rho with the exact arithmetic order in clamp_with_profile.
        radial = np.sqrt(local[:, 1] * local[:, 1] + local[:, 2] * local[:, 2])
        view = self.records[self.written:self.written + n]
        view['index'] = indices
        view['coord'] = coords
        view['rho'] = radial
        self.written += n
        self.write_s += perf_counter() - t0

    def finish(self):
        if self.written != self.total:
            raise ValueError(f'Metadata incomplete: {self.written}/{self.total}')
        self.records.flush()

    def iter_chunks(self, batch_size):
        if self.written != self.total:
            raise ValueError('Call finish before replay')
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        for start in range(0, self.total, batch_size):
            end = min(self.total, start + batch_size)
            yield start, end, self.records[start:end]


def compose_replay_chunk(raw, coords, rho, profiles, *, rho_limit=1.,
                         positive_value=1., sigma=.01):
    """Same independent rigid-boot support clamp and base/detail composition.

    `rho` is precomputed from the original samples_local with exactly the
    multiplication/addition/sqrt order used by clamp_with_profile. Every
    remaining scalar operation is the existing formula, including the gate.
    """
    coords = np.asarray(coords).reshape(-1)
    rho = np.asarray(rho).reshape(-1)
    dist = np.asarray(raw['dist']).reshape(-1)
    base = np.asarray(raw['dist_base']).reshape(-1)
    detail = np.asarray(raw['dist_detail']).reshape(-1)
    if not (len(coords) == len(rho) == len(dist) == len(base) == len(detail)):
        raise ValueError('Metadata and predictions are misaligned')
    support_dist = np.interp(coords, profiles['dist'][0], profiles['dist'][1])
    support_base = np.interp(coords, profiles['dist_base'][0], profiles['dist_base'][1])
    s_dist = np.clip((rho - support_dist) / (0.25 + 1e-12), -1., 1.) * float(positive_value)
    s_base = np.clip((rho - support_base) / (0.25 + 1e-12), -1., 1.) * float(positive_value)
    # Same clamp and bad-mask operations as clamp_with_profile.
    clamped_dist = np.maximum(dist, s_dist)
    bad = (rho > float(rho_limit)) & (clamped_dist < 0.)
    valid = ~bad
    base = np.maximum(base, s_base)
    base_bad = (rho > float(rho_limit)) & (base < 0.)
    base = np.asarray(base).copy()
    base[base_bad] = float(positive_value)
    detail = np.where(valid, detail, 0.)
    gate = np.exp(-(base * base) / (2. * float(sigma) ** 2 + 1e-12))
    return base + gate * detail, int(np.count_nonzero(bad)), (float(rho.min()),float(rho.max()))
