"""0053: Direct bounded model prediction with whole-section support calibration.

Each accepted localization batch is immediately inferred, and only predicted
scalar fields, accepted voxel indices and support-calibration features are
appended to disk. The ORIGINAL whole-section support clamp runs after finish().
No per-chunk rho quantiles are used. A one-pass localization/model stage; a
second, cheap correction/composition stage replays the predictions.
"""
import os
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np


MODEL_INPUT_KEYS = ('samples_local', 'coords', 'angles', 'radius', 'rho', 'rho_n')


def needs_avatar_inference(arg):
    return (bool(arg.get('cut_avatar', False))
            or bool(arg.get('auto_avatar_snug_field', False))
            or bool(arg.get('detail_avatar_gate', False))
            or bool(arg.get('final_carve', False))
            or str(arg.get('snug_mode', 'multiplicative')).lower() == 'additive'
            or str(arg.get('accessory_offset_mode', 'none')).lower() in ('local', 'both'))


def validate_prediction_stream(arg):
    """Fail before localization for branches requiring nonlocal query corrections."""
    blockers = [flag for flag in ('auto_avatar_snug_field', 'use_tiled_detail')
                if arg.get(flag, False)]
    # A spatial curvature proxy requires the full avatar grid, not predictions
    # stored in independent voxel batches.
    if arg.get('final_carve', False) and float(arg.get('carve_k_bulge', 0.0)) > 0:
        blockers.append('spatial curvature final_carve')
    if arg.get('use_accessory_support_clamp', False) is False:
        # This is a policy guard for the initial, comparison-only puffer launch.
        blockers.append('support clamp required for 0053 calibration parity')
    if blockers:
        raise ValueError('0053 direct prediction streaming unsupported options: '
                         + ', '.join(blockers))


class PredictionStream0053:
    """Append-only disk-backed predictions in global accepted-index order.

    Shapes/dtypes are recorded on first write. No n_candidate or n_accepted
    allocation is required. Raw binary files can be opened as memmaps without
    copying the full dataset at the end of localization.
    """
    def __init__(self, agent, accessory_key, avatar_key, arg,
                 seam_corrections, scratch_dir, batch_size):
        validate_prediction_stream(arg)
        self.agent = agent
        self.accessory_key = accessory_key
        self.avatar_key = avatar_key
        self.arg = arg
        self.seam_corrections = seam_corrections
        self.batch_size = int(batch_size)
        self.avatar_needed = needs_avatar_inference(arg)
        if not scratch_dir or not os.path.isdir(scratch_dir):
            raise ValueError('0053 requires existing writable --sdf-grid-memmap-dir')
        self.temp = tempfile.TemporaryDirectory(prefix='oktopus-0053-', dir=scratch_dir)
        self.rows = 0
        self.files = {}
        self.schemas = {}
        self.bytes_written = 0
        self.model_s = 0.0
        self.write_s = 0.0
        self._finished = False
        # Accumulate accepted (not candidate) rows to avoid tiny GPU calls.
        # Only one batch of model inputs stays in RAM at a time.
        self.pending = []
        self.pending_rows = 0

    def _write(self, field, array):
        a = np.asarray(array)
        if a.ndim == 0:
            a = a.reshape(1)
        if not a.flags.c_contiguous:
            a = np.ascontiguousarray(a)
        shape_tail = a.shape[1:]
        schema = (a.dtype, shape_tail)
        if field not in self.schemas:
            path = Path(self.temp.name) / (field + '.bin')
            self.files[field] = open(path, 'wb')
            self.schemas[field] = schema
        elif self.schemas[field] != schema:
            raise ValueError(f'0053 streamed field {field} changed dtype or shape')
        self.files[field].write(a.tobytes(order='C'))
        self.bytes_written += a.nbytes

    def push(self, acc, indices, av):
        if self._finished:
            raise RuntimeError('0053 prediction stream already finished')
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        n = len(indices)
        if not n:
            return
        if set(MODEL_INPUT_KEYS) - set(acc):
            raise ValueError('0053 missing accessory model features')
        if 'coords' not in av or 'angles' not in av:
            raise ValueError('0053 requires avatar coords and angles for seam ownership')
        if self.avatar_needed and (set(MODEL_INPUT_KEYS) - set(av)):
            raise ValueError('0053 missing avatar model features')
        av_keys = MODEL_INPUT_KEYS if self.avatar_needed else ('coords', 'angles')
        offset = 0
        while offset < n:
            take = min(n - offset, self.batch_size - self.pending_rows)
            chunk = slice(offset, offset + take)
            self.pending.append((indices[chunk].copy(),
                                 {k: np.asarray(acc[k])[chunk].copy()
                                  for k in MODEL_INPUT_KEYS},
                                 {k: np.asarray(av[k])[chunk].copy()
                                  for k in av_keys}))
            self.pending_rows += take
            offset += take
            if self.pending_rows == self.batch_size:
                self._flush()

    def _flush(self):
        if not self.pending_rows:
            return
        n = self.pending_rows
        if len(self.pending) == 1:
            indices, acc, av = self.pending[0]
        else:
            indices = np.concatenate([p[0] for p in self.pending])
            acc = {k: np.concatenate([p[1][k] for p in self.pending])
                   for k in MODEL_INPUT_KEYS}
            av = {k: np.concatenate([p[2][k] for p in self.pending])
                  for k in self.pending[0][2]}
        self.pending.clear()
        self.pending_rows = 0
        # Exactly the same query-space seam warp as in staged inference.
        if self.seam_corrections:
            from app.seam_optimization import warp_accessory_queries
            acc = warp_accessory_queries(acc, av, self.seam_corrections, self.arg)
        t0 = perf_counter()
        acc_pred = self.agent._inference_full_vals(
            acc, self.accessory_key, batch_size=self.batch_size)
        av_pred = (self.agent._inference_full_vals(
            av, self.avatar_key, batch_size=self.batch_size)
                   if self.avatar_needed else None)
        self.model_s += perf_counter() - t0
        if any(k not in acc_pred or acc_pred[k] is None
               for k in ('dist', 'dist_base', 'dist_detail')):
            raise ValueError('0053 inference_full requires dist, dist_base, dist_detail')
        t0 = perf_counter()
        self._write('index', indices)
        self._write('acc_coords', np.asarray(acc['coords']).reshape(n))
        self._write('acc_local', np.asarray(acc['samples_local']).reshape(n, 3))
        self._write('av_coords', np.asarray(av['coords']).reshape(n))
        self._write('av_angles', np.asarray(av['angles']).reshape(n))
        for name in ('dist', 'dist_base', 'dist_detail'):
            pred = np.asarray(acc_pred[name]).reshape(-1)
            if pred.size != n:
                raise ValueError(f'0053 inference {name} count mismatch')
            self._write('acc_' + name, pred)
        if av_pred is not None:
            pred = np.asarray(av_pred['dist']).reshape(-1)
            if pred.size != n:
                raise ValueError('0053 avatar inference count mismatch')
            self._write('av_dist', pred)
        self.rows += n
        self.write_s += perf_counter() - t0

    def finish(self):
        if self._finished:
            raise RuntimeError('0053 finish called twice')
        self._finished = True
        self._flush()
        for f in self.files.values():
            f.close()
        self.files.clear()
        if not self.rows:
            raise ValueError('0053 no accepted voxels')
        result = {}
        for name, (dtype, tail) in self.schemas.items():
            shape = (self.rows,) + tail
            result[name] = np.memmap(Path(self.temp.name) / (name + '.bin'),
                                     mode='r+', dtype=dtype, shape=shape)
        print('[sdf_puffer_0053] predicted_rows=', self.rows,
              'predictions_and_calibration_GB=', f'{self.bytes_written/1e9:.3f}',
              'model_s=', f'{self.model_s:.3f}',
              'prediction_write_s=', f'{self.write_s:.3f}',
              'global_support_clamp=deferred_exact', flush=True)
        return result

    def cleanup(self):
        self.pending.clear()
        for f in self.files.values():
            f.close()
        self.files.clear()
        self.temp.cleanup()
