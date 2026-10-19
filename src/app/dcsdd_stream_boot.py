"""Opt-in streaming SDF preparation for independent rigid boots.

The final support clamp uses GLOBAL, per-curve-bin quantiles of predicted SDF.
Consequently a one-pass per-chunk clamp is NOT equivalent to the existing path.
This module stages only three *predicted scalar fields*, not full localization
records, and re-localizes once to apply the exact global calibration. The
network is evaluated only once, with the original array_split batch partition.
"""
from collections import deque
from pathlib import Path
from time import perf_counter
import numpy as np


INFERENCE_KEYS = ('samples_local', 'coords', 'angles', 'radius', 'rho', 'rho_n')
PREDICTION_KEYS = ('dist', 'dist_base', 'dist_detail')


def inference_partition_sizes(n, batch_size):
    """Exactly _inference_full_vals: np.array_split(arange(n), n//size + 1)."""
    if n < 0 or batch_size <= 0:
        raise ValueError('Invalid sample count or batch size')
    if n <= batch_size:
        return (n,) if n else ()
    parts = n // batch_size + 1
    q, r = divmod(n, parts)
    return (q + 1,) * r + (q,) * (parts - r)


class GlobalRhoCalibration:
    """Exact profile equivalent to AgentBase.estimate_rho_limit_from_pred().

    Retain *only* the near-zero rho samples required for np.quantile, never
    the world coordinates or the complete (N,6+) localized result.
    """
    def __init__(self, n_bins=48, surface_band=0.03):
        self.n_bins = int(n_bins)
        self.surface_band = float(surface_band)
        self.edges = np.linspace(0., 1., self.n_bins + 1)
        self.centers = (self.edges[:-1] + self.edges[1:]) * .5
        self.values = [[] for _ in range(self.n_bins)]
        self.count = 0

    def add(self, pred, samples_local, coords):
        pred = np.asarray(pred).reshape(-1)
        sl = np.asarray(samples_local)
        coord = np.asarray(coords).reshape(-1)
        near = np.abs(pred) < self.surface_band
        bins = np.searchsorted(self.edges, coord[near], side='right') - 1
        rho = np.sqrt(sl[near, 1] * sl[near, 1] + sl[near, 2] * sl[near, 2])
        for i in np.unique(bins):
            if 0 <= i < self.n_bins:
                part = rho[bins == i]
                self.values[int(i)].append(part)
                self.count += len(part)

    def finish(self, q=0.98, margin=0.08, smooth_s=2.0, fallback=1.2):
        from scipy.ndimage import gaussian_filter1d
        limit = np.full(self.n_bins, np.nan, dtype=np.float64)
        for i, parts in enumerate(self.values):
            n = sum(map(len, parts))
            if n >= 10:
                # Same quantile and bin edges as AgentBase. No approximation.
                limit[i] = np.quantile(np.concatenate(parts), float(q)) + float(margin)
        good = np.isfinite(limit)
        if np.any(good):
            limit[~good] = np.interp(self.centers[~good], self.centers[good], limit[good])
        else:
            limit[:] = float(fallback)
        if smooth_s > 0:
            limit = gaussian_filter1d(limit, sigma=float(smooth_s), mode='nearest')
        return self.centers, limit


def clamp_with_profile(pred, data, profile, *, rho_limit=1., positive_value=1.):
    """AgentBase.clamp_pred_sdf_by_support with precomputed GLOBAL profile.

    Must use samples_local for rho_n (not data['rho_n']), as in the original.
    """
    sl = np.asarray(data['samples_local'])
    coords = np.asarray(data['coords']).reshape(-1)
    u, v = sl[:, 1], sl[:, 2]
    rho_n = np.sqrt(u * u + v * v)
    bins, limits = profile
    rho_limit_i = np.interp(coords, bins, limits)
    support_sdf = np.clip((rho_n - rho_limit_i) / (0.25 + 1e-12), -1., 1.) * float(positive_value)
    result = np.maximum(pred, support_sdf)
    result = np.asarray(result).reshape(-1).copy()
    bad = (rho_n > float(rho_limit)) & (result < 0.)
    result[bad] = float(positive_value)
    return result, ~bad, rho_n, int(np.count_nonzero(bad))


class PartitionedPredictionWriter:
    """Accept irregular localization batches; infer on original global partitions.

    The predictions are stored in three contiguous scalar files. Does not store
    indices, frames, avatar data, coordinates or other localization fields.
    """
    def __init__(self, *, n, batch_size, outdir, inference):
        self.total = int(n)
        self.partition_sizes = deque(inference_partition_sizes(n, batch_size))
        self.outdir = Path(outdir)
        self.inference = inference
        self.buffers = {key: deque() for key in INFERENCE_KEYS}
        self.pending = 0
        self.written = 0
        self.raw = {}
        self.cal = {k: GlobalRhoCalibration() for k in ('dist', 'dist_base')}
        self.prediction_bytes = 0
        self.n_inference_calls = 0
        self.inference_compute_s = 0.0
        self.prediction_write_s = 0.0

    def push(self, acc):
        if not self.partition_sizes:
            if len(np.asarray(acc['coords'])):
                raise AssertionError('Received more accepted samples than counted')
            return
        n = len(acc['coords'])
        for key in INFERENCE_KEYS:
            self.buffers[key].append(np.asarray(acc[key]))
        self.pending += n
        while self.partition_sizes and self.pending >= self.partition_sizes[0]:
            size = self.partition_sizes.popleft()
            model_data = {}
            for key in INFERENCE_KEYS:
                slices = []
                take = size
                buff = self.buffers[key]
                while take:
                    src = buff.popleft()
                    m = min(len(src), take)
                    slices.append(src[:m])
                    if m < len(src):
                        buff.appendleft(src[m:])
                    take -= m
                model_data[key] = np.concatenate(slices, axis=0) if len(slices) > 1 else slices[0]
            inference_t0 = perf_counter()
            out = self.inference(model_data)
            self.inference_compute_s += perf_counter() - inference_t0
            self.n_inference_calls += 1
            write_t0 = perf_counter()
            for key in PREDICTION_KEYS:
                if key not in out or out[key] is None:
                    raise ValueError('model.inference_full must return ' + key)
                v = np.asarray(out[key]).reshape(-1)
                if len(v) != size:
                    raise ValueError(f'Inference {key} returned {len(v)}, expected {size}')
                if key not in self.raw:
                    self.raw[key] = np.memmap(
                        str(self.outdir / (key + '.bin')),
                        mode='w+', dtype=v.dtype, shape=(max(1,self.total),))
                self.raw[key][self.written:self.written+size] = v
                self.prediction_bytes += v.nbytes
            for key in self.cal:
                self.cal[key].add(out[key], model_data['samples_local'], model_data['coords'])
            self.written += size
            self.pending -= size
            self.prediction_write_s += perf_counter() - write_t0

    def finish(self, fallback=1.2):
        if self.partition_sizes or self.pending or self.written != self.total:
            raise AssertionError(f'Not all samples inferred: {self.written}/{self.total}')
        for arr in self.raw.values():
            arr.flush()
        return {key: cal.finish(fallback=fallback) for key, cal in self.cal.items()}


def compose_boot_chunk(raw, data, profiles, *, rho_limit=1., positive_value=1., sigma=0.01, support=True):
    """Boot1's existing base/detail recomposition with full-run support profiles."""
    dist = np.asarray(raw['dist'])
    base = np.asarray(raw['dist_base'])
    detail = np.asarray(raw['dist_detail'])
    if support:
        _, valid, rho, bad = clamp_with_profile(
            dist, data, profiles['dist'], rho_limit=rho_limit,
            positive_value=positive_value)
        base, _, _, _ = clamp_with_profile(
            base, data, profiles['dist_base'], rho_limit=rho_limit,
            positive_value=positive_value)
        detail = np.where(valid, detail, 0.)
    else:
        rho = np.sqrt(np.sum(np.square(data['samples_local'][:, 1:3]), axis=1))
        bad = 0
    gate = np.exp(-(base * base) / (2. * float(sigma) ** 2 + 1e-12))
    return base + gate * detail, bad, (float(rho.min()), float(rho.max()))
