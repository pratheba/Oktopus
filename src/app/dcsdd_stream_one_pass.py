"""0040: automatic, count-free streaming for independent rigid SDF boots.

Localization is traversed once. Accepted samples are inferred in fixed-size
batches and binary prediction/replay records are appended sequentially.

Unlike the legacy `np.array_split` inference partitions, these batches do not
require knowing the final accepted count. For pointwise models the mathematical
queries are unchanged, but GPU execution can have minor batch-shape numerical
differences: use the opt-in flag and compare SDF/mesh output before production.
"""
from collections import deque
from pathlib import Path
from time import perf_counter
import numpy as np

from app.dcsdd_stream_boot import (
    INFERENCE_KEYS, PREDICTION_KEYS, GlobalRhoCalibration,
)
from app.dcsdd_stream_replay import REPLAY_DTYPE


class AppendReplayMetadata:
    """Sequentially append only voxel index, curve coordinate and radial norm."""

    def __init__(self, outdir):
        self.path = Path(outdir) / 'replay_metadata.bin'
        self.file = self.path.open('wb', buffering=1024 * 1024)
        self.records = None
        self.written = 0
        self.write_s = 0.0

    @property
    def bytes(self):
        return self.written * REPLAY_DTYPE.itemsize

    def push(self, acc, indices):
        if self.file is None:
            raise ValueError('Replay is already finalized')
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        coords = np.asarray(acc['coords']).reshape(-1)
        local = np.asarray(acc['samples_local'])
        n = len(idx)
        if len(coords) != n or local.shape != (n, 3):
            raise ValueError('Invalid accepted localization data')
        t0 = perf_counter()
        records = np.empty(n, dtype=REPLAY_DTYPE)
        records['index'] = idx
        records['coord'] = coords
        records['rho'] = np.sqrt(local[:, 1] * local[:, 1] +
                                 local[:, 2] * local[:, 2])
        self.file.write(records.tobytes())
        self.written += n
        self.write_s += perf_counter() - t0

    def finish(self):
        if self.file is None:
            raise ValueError('Replay already finalized')
        self.file.close()
        self.file = None
        if self.written <= 0:
            raise ValueError('Cannot finalize empty replay')
        self.records = np.memmap(self.path, dtype=REPLAY_DTYPE,
                                 mode='r', shape=(self.written,))

    def iter_chunks(self, batch_size):
        if self.records is None:
            raise ValueError('Replay must be finalized before reading')
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        for start in range(0, self.written, batch_size):
            end = min(start + batch_size, self.written)
            yield start, end, self.records[start:end]


class OnePassBootStreamWriter:
    """One localization pass; infer in fixed accepted-sample batches.

    `push` accepts localization chunks with arbitrary sizes. It accumulates up
    to `batch_size` accepted rows and invokes inference immediately. Exactly
    three scalar prediction columns and compact replay records are appended
    to ordinary buffered files. At `finish`, files are read via memmap.
    """

    def __init__(self, *, batch_size, outdir, inference):
        if int(batch_size) <= 0:
            raise ValueError('batch_size must be positive')
        self.batch_size = int(batch_size)
        self.outdir = Path(outdir)
        self.inference = inference
        self.replay = AppendReplayMetadata(self.outdir)
        self.buffers = {key: deque() for key in INFERENCE_KEYS}
        self.pending = 0
        self.written = 0
        self.accepted = 0
        self.raw = {}
        self.files = {}
        self.dtypes = {}
        self.cal = {k: GlobalRhoCalibration() for k in ('dist', 'dist_base')}
        self.prediction_bytes = 0
        self.n_inference_calls = 0
        self.inference_compute_s = 0.0
        self.prediction_write_s = 0.0
        self.batch_sizes = []
        self.finished = False

    def push(self, acc, indices):
        if self.finished:
            raise ValueError('Cannot append after finalizing one-pass writer')
        n = len(indices)
        if len(np.asarray(acc['coords'])) != n:
            raise ValueError('Prediction data and indices differ in length')
        self.replay.push(acc, indices)
        self.accepted += n
        for key in INFERENCE_KEYS:
            part = np.asarray(acc[key])
            if len(part) != n:
                raise ValueError('Invalid localization length for ' + key)
            if n:
                self.buffers[key].append(part)
        self.pending += n
        while self.pending >= self.batch_size:
            self._infer(self.batch_size)

    def _take(self, size):
        batch = {}
        for key in INFERENCE_KEYS:
            pieces = []
            left = size
            fifo = self.buffers[key]
            while left:
                item = fifo.popleft()
                take = min(len(item), left)
                pieces.append(item[:take])
                if take < len(item):
                    fifo.appendleft(item[take:])
                left -= take
            batch[key] = (pieces[0] if len(pieces) == 1 else
                          np.concatenate(pieces, axis=0))
        return batch

    def _infer(self, size):
        if size <= 0 or size > self.pending:
            raise ValueError('Invalid inference partition')
        batch = self._take(size)
        t0 = perf_counter()
        result = self.inference(batch)
        self.inference_compute_s += perf_counter() - t0
        self.n_inference_calls += 1
        self.batch_sizes.append(size)
        t0 = perf_counter()
        for key in PREDICTION_KEYS:
            if result.get(key) is None:
                raise ValueError('Missing model output ' + key)
            arr = np.asarray(result[key]).reshape(-1)
            if len(arr) != size:
                raise ValueError(f'Prediction {key} length mismatch')
            if key not in self.files:
                self.dtypes[key] = arr.dtype
                self.files[key] = (self.outdir / (key + '.bin')).open(
                    'wb', buffering=1024 * 1024)
            if arr.dtype != self.dtypes[key]:
                raise ValueError('Prediction dtype changed for ' + key)
            self.files[key].write(np.ascontiguousarray(arr).tobytes())
            self.prediction_bytes += arr.nbytes
        for key in self.cal:
            self.cal[key].add(result[key], batch['samples_local'], batch['coords'])
        self.pending -= size
        self.written += size
        self.prediction_write_s += perf_counter() - t0

    def finish(self, fallback=1.2):
        if self.finished:
            raise ValueError('One-pass writer already finalized')
        if self.pending:
            self._infer(self.pending)
        if self.written != self.accepted or self.written <= 0:
            raise ValueError('Accepted prediction count mismatch or empty stream')
        for handle in self.files.values():
            handle.close()
        self.files.clear()
        self.raw = {
            key: np.memmap(self.outdir / (key + '.bin'), mode='r',
                           dtype=self.dtypes[key], shape=(self.written,))
            for key in PREDICTION_KEYS
        }
        self.finished = True
        return {key: cal.finish(fallback=fallback) for key, cal in self.cal.items()}
