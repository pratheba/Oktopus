"""Shared 0047 *per-adaptation* SDF evaluator for rigid and wrap mappings.

Curve.filter_grid_adapt remains the coordinate-mapping authority.  This class
handles only the common stream protocol: increasing global voxel indices,
pointwise prediction partitions, global support calibration, compact replay,
and final-value composition.  It does NOT choose a mapping, seam rule, or
attachment topology, and deliberately does not approximate snug fields.
"""
from pathlib import Path
from time import perf_counter

import numpy as np

from app.dcsdd_stream_one_pass import OnePassBootStreamWriter
from app.dcsdd_stream_replay import compose_replay_chunk


class SDFStreamPiece:
    """A single adaptation's count-free SDF prediction and replay stream.

    `push` is a filter_grid_adapt callback for rigid and wrap mappings; when
    `record_avatar_source` is true it accepts the optional third avatar-data
    argument and records its longitudinal seam coordinate in voxel order.
    `finish` validates the accepted count and completes global calibration.
    """

    def __init__(self, *, batch_size, outdir, inference,
                 record_avatar_source=False):
        self.outdir = Path(outdir)
        self.writer = OnePassBootStreamWriter(
            batch_size=batch_size, outdir=outdir, inference=inference)
        self.replay = self.writer.replay
        self.record_avatar_source = bool(record_avatar_source)
        self._source_path = self.outdir / 'avatar_source_s.bin'
        self._source_file = (self._source_path.open('wb', buffering=1024 * 1024)
                             if self.record_avatar_source else None)
        self._source_values = None
        self._last_index = -1
        self.profiles = None
        self.source_write_s = 0.0
        self.prediction_finalize_s = 0.0
        self.replay_finalize_s = 0.0

    @property
    def accepted(self):
        return self.writer.accepted

    @property
    def source_s(self):
        if not self.record_avatar_source:
            return None
        if self.profiles is None:
            raise ValueError('Finish the SDF stream before reading source coordinates')
        return self._source_values

    def push(self, accessory_data, indices, avatar_data=None):
        """Append one accepted batch; preserve original global voxel order."""
        if self.profiles is not None:
            raise ValueError('Cannot append to a finalized SDF stream')
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if len(indices):
            if indices[0] <= self._last_index or np.any(indices[1:] <= indices[:-1]):
                raise ValueError('SDF stream indices must be unique and increasing')
        if self.record_avatar_source:
            if avatar_data is None or 'coords' not in avatar_data:
                raise ValueError('Wrap seam streaming requires avatar source coords')
            source_s = np.asarray(avatar_data['coords'], dtype=np.float64).reshape(-1)
            if len(source_s) != len(indices):
                raise ValueError('Avatar seam coordinate/replay index mismatch')
        else:
            source_s = None
        # Writer validates inference columns and append order before accepting.
        self.writer.push(accessory_data, indices)
        if len(indices):
            self._last_index = int(indices[-1])
        if source_s is not None:
            t0 = perf_counter()
            self._source_file.write(np.ascontiguousarray(source_s).tobytes())
            self.source_write_s += perf_counter() - t0

    def finish(self, *, expected_accepted, fallback):
        """Finalize predictions, global support profiles, and replay metadata."""
        if self.profiles is not None:
            raise ValueError('SDF stream was already finalized')
        if self.accepted <= 0 or self.accepted != int(expected_accepted):
            raise RuntimeError('SDF stream accepted-count mismatch or empty stream')
        if self._source_file is not None:
            self._source_file.close()
            self._source_file = None
            if self._source_path.stat().st_size != self.accepted * np.dtype('float64').itemsize:
                raise RuntimeError('SDF stream avatar seam records are incomplete')
        t0 = perf_counter()
        profiles = self.writer.finish(fallback=fallback)
        self.prediction_finalize_s = perf_counter() - t0
        t0 = perf_counter()
        self.replay.finish()
        self.replay_finalize_s = perf_counter() - t0
        if self.record_avatar_source:
            self._source_values = np.memmap(
                self._source_path, dtype=np.float64, mode='r', shape=(self.accepted,))
        self.profiles = profiles
        return profiles

    def compose_chunk(self, start, end, records, *, rho_limit,
                      positive_value, sigma):
        """Identical globally calibrated composition for rigid or wrap."""
        if self.profiles is None:
            raise ValueError('Finalize SDF stream before composing')
        raw = {key: val[start:end] for key, val in self.writer.raw.items()}
        return compose_replay_chunk(
            raw, records['coord'], records['rho'], self.profiles,
            rho_limit=rho_limit, positive_value=positive_value, sigma=sigma)

    def compose_values(self, *, sigma, rho_limit, positive_value,
                       chunk_size, label=''):
        """Make compact per-accepted-sample SDF values, not a dense 3D grid."""
        path = self.outdir / 'final_sdf.bin'
        result = np.memmap(path, dtype=np.float64, mode='w+',
                           shape=(self.accepted,))
        bad_sum = 0
        min_rho, max_rho = np.inf, -np.inf
        for start, end, records in self.replay.iter_chunks(chunk_size):
            values, bad, (rmin, rmax) = self.compose_chunk(
                start, end, records, sigma=sigma, rho_limit=rho_limit,
                positive_value=positive_value)
            result[start:end] = values
            bad_sum += bad
            min_rho = min(min_rho, rmin)
            max_rho = max(max_rho, rmax)
        result.flush()
        print(f'[sdf_stream_piece] {label} clamped_bad={bad_sum}/{self.accepted} '
              f'rho=[{min_rho:.3f},{max_rho:.3f}]', flush=True)
        return result
