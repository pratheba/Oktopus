import os, pickle
import numpy as np
import os.path as op
import trimesh
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation, Slerp
from handle_utils import CylindersMesh
from scipy.ndimage import gaussian_filter1d
from curve_utils import *


def interpolate(self, non_uniform_linear_skeletal_points, points=True, radius=True, frame=True):
    res = {}
    # key_points: (N, 3); key_radius: (N, 2)
    ts = non_uniform_linear_skeletal_points
    if points:
        pts_ts = np.stack([
            np.interp(ts, self.key_ts, self.key_points[:, 0]),
            np.interp(ts, self.key_ts, self.key_points[:, 1]),
            np.interp(ts, self.key_ts, self.key_points[:, 2])
        ]).T
        res['points'] = pts_ts
    
    if radius:
        rs_ts = np.stack([
            np.interp(ts, self.key_ts, self.key_radius[:, 0]),
            np.interp(ts, self.key_ts, self.key_radius[:, 1])
        ]).T
        res['radius'] = rs_ts

    if frame:
        # requirement of Slerp from Scipy
        ts_rot = np.clip(ts, a_min=1e-10, a_max=(1 - 1e-10))
        idx = np.searchsorted(self.key_ts, ts_rot)
        idx = np.clip(idx, 0, len(self.key_ts)-1)
        frame_ts= self.key_frame[idx]
        #frame_ts = self.rot_slerp(ts_rot).as_matrix()
        res['frame'] = frame_ts

    return res

def interpolate_mix(self, ts, mix_arg):
    new_curve = mix_arg['curve_handle']
    func1 = mix_arg['mix_func1']
    func2 = mix_arg['mix_func2']
    ts1, weights1 = func1(ts)
    ts2, weights2 = func2(ts)
    #print("weights 1 ", weights1)
    #print("weights 2 ", weights2)
    #print("ts 2 ", ts2)
    #print("ts 1", ts1)
    #exit()

    rs1 = np.stack([
        np.interp(ts1, self.key_ts, self.key_radius[:, 0]),
        np.interp(ts1, self.key_ts, self.key_radius[:, 1])
    ]).T
    rs1_mean = rs1.mean(axis=1)
    scales1 = rs1 / rs1_mean[:, None]
    #print("scaled1 = ", scales1, flush=True)
    #print("rs1 = ", rs1_mean, flush=True)

    intpl2 = new_curve.core.interpolate(ts2, points=False, frame=False)
    rs2 = intpl2['radius']
    rs2_mean = rs2.mean(axis=1)
    scales2 = rs2 / rs2_mean[:, None]
    #print("scaled2 = ", scales2, flush=True)
    #print("rs2 = ", rs2_mean, flush=True)

    scales = scales1*weights1[:,None] + scales2*weights2[:,None]
    radius = rs1_mean[:,None]*scales

    intpl = self.interpolate(ts, radius=False)
    intpl['radius'] = radius
    #exit()
    return intpl


def interpolate_adapt(self, ts, adapt_arg):
    avatar_arclen_coords = self.localize_adapt(ts, adapt_arg)
    avatar_curve_handle = adapt_arg['avatar_curve_handle']

    # source yz radius at coords_src
    accessory_intpl = self.interpolate(ts)    # uses self.key_ts
    avatar_intpl = avatar_curve_handle.core.interpolate(avatar_arclen_coords)

    return accessory_intpl, avatar_intpl, avatar_arclen_coords

def interpolate_wrap_radius(self, ts, theta):
    if not hasattr(self, "wrap_radius"):
        return None

    theta = (theta + np.pi) % (2*np.pi) - np.pi

    out = np.empty_like(ts, dtype=np.float64)

    for n in range(len(ts)):
        # interpolate in s first
        r_theta_grid = np.empty(len(self.wrap_theta_bins), dtype=np.float64)
        for k in range(len(self.wrap_theta_bins)):
            r_theta_grid[k] = np.interp(ts[n], self.wrap_s_bins, self.wrap_radius[:, k])

        # periodic interpolation in theta
        theta_grid = self.wrap_theta_bins
        theta_ext = np.concatenate([theta_grid - 2*np.pi, theta_grid, theta_grid + 2*np.pi])
        r_ext = np.concatenate([r_theta_grid, r_theta_grid, r_theta_grid])

        out[n] = np.interp(theta[n], theta_ext, r_ext)

    return out

def interpolate_wrap_radius_test1(self, ts, theta, wrap_theta_bins, wrap_s_bins, wrap_radius):
    """
    Vectorized interpolation of directional wrap radius.

    Args:
        ts:    (N,) curve coordinates in [0,1]
        theta: (N,) angles in radians

    Returns:
        r:     (N,) interpolated directional radius
    """

    ts = np.asarray(ts, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)

    s_bins = wrap_s_bins              # (Ns,)
    theta_bins = wrap_theta_bins      # (Nt,)
    wrap = wrap_radius                # (Ns, Nt)

    Ns = len(s_bins)
    Nt = len(theta_bins)
    N = len(ts)

    # -----------------------------
    # 1) linear interpolation in s
    # -----------------------------
    s_idx1 = np.searchsorted(s_bins, ts, side='right')
    s_idx1 = np.clip(s_idx1, 1, Ns - 1)
    s_idx0 = s_idx1 - 1

    s0 = s_bins[s_idx0]
    s1 = s_bins[s_idx1]
    ws = (ts - s0) / (s1 - s0 + 1e-12)   # (N,)

    # gather wrap values at the two neighboring s bins
    r0 = wrap[s_idx0, :]   # (N, Nt)
    r1 = wrap[s_idx1, :]   # (N, Nt)

    r_s = (1.0 - ws[:, None]) * r0 + ws[:, None] * r1   # (N, Nt)

    # --------------------------------
    # 2) periodic linear interpolation in theta
    # --------------------------------
    period = 2.0 * np.pi
    theta0 = theta_bins[0]
    dtheta = theta_bins[1] - theta_bins[0]

    # map theta into same periodic interval as theta_bins
    theta_wrap = ((theta - theta0) % period) + theta0

    # fractional theta-bin coordinate
    t = (theta_wrap - theta0) / dtheta
    th_idx0 = np.floor(t).astype(np.int64) % Nt
    th_idx1 = (th_idx0 + 1) % Nt
    wt = t - np.floor(t)   # (N,)

    rows = np.arange(N)
    rv0 = r_s[rows, th_idx0]
    rv1 = r_s[rows, th_idx1]

    out = (1.0 - wt) * rv0 + wt * rv1
    return out

def interpolate_wrap_radius_test(self, ts, theta, wrap_theta_bins, wrap_s_bins, wrap_radius):

    theta = (theta + np.pi) % (2*np.pi) - np.pi

    out = np.empty_like(ts, dtype=np.float64)


    print(len(ts))
    print(len(wrap_theta_bins))
    exit()

    for n in range(len(ts)):
        # interpolate in s first
        print(n, flush=True)
        r_theta_grid = np.empty(len(wrap_theta_bins), dtype=np.float64)
        for k in range(len(wrap_theta_bins)):
            r_theta_grid[k] = np.interp(ts[n], wrap_s_bins, wrap_radius[:, k])

        # periodic interpolation in theta
        theta_grid = wrap_theta_bins
        theta_ext = np.concatenate([theta_grid - 2*np.pi, theta_grid, theta_grid + 2*np.pi])
        r_ext = np.concatenate([r_theta_grid, r_theta_grid, r_theta_grid])

        out[n] = np.interp(theta[n], theta_ext, r_ext)

    return out

def interpolate_stretch(self, ts, stretch_arg):
    #func = stretch_arg['mix_func']
    #ts_new = func(ts)
    radius = np.stack([
        np.interp(ts, self.key_ts, self.key_radius[:, 0]),
        np.interp(ts, self.key_ts, self.key_radius[:, 1])
    ]).T
    intpl = self.interpolate(ts, radius=False)

    intpl['radius'] = radius 
    return intpl, ts

def interpolate_theta(self, theta, wrap_radius, theta_bins):
    period = 2.0 * np.pi
    theta0 = theta_bins[0]
    theta = ((theta - theta0) % period) + theta0 # np.mod(theta, 2.0 * np.pi)

    num_theta_bins = theta_bins.shape[0]
    dtheta_bins = theta_bins[1] - theta_bins[0]

    delta_theta = (theta - theta0) / dtheta_bins
    t0 = np.floor(delta_theta).astype(np.int64) % num_theta_bins
    t1 = (t0 + 1) % num_theta_bins
    theta_period = delta_theta - np.floor(delta_theta)

    wrap_theta = (1.0 - theta_period[:, None]) * wrap_radius[:, t0].T  + theta_period[:, None]*wrap_radius[:, t1].T
    return wrap_theta


def _interpolate_wrap_radius_vectorized_0061(ts, theta, wrap, theta_bins, s_bins):
    """Bilinear wrap lookup in O(N) memory, with no per-point Python loop.

    Use the ORIGINAL angular wrapping/index arithmetic (interpolate_theta),
    then interpolate just the two adjacent s rows. Unlike the existing
    interpolate_theta -> (N,Ns) -> np.interp per row path, this constructs
    only O(N) temporaries; it supports nonuniform, strictly increasing s_bins.
    The caller handles unsupported/degenerate longitudinal bins by falling
    back to the original implementation.
    """
    n_s, n_theta = wrap.shape
    theta0 = theta_bins[0]
    dtheta = theta_bins[1] - theta_bins[0]
    # Keep the same periodic wrap and index/weight arithmetic as
    # interpolate_theta. In particular, do not approximate theta by its
    # nearest bin or assume theta is pre-normalized.
    theta_mapped = ((theta - theta0) % (2.0 * np.pi)) + theta0
    delta_theta = (theta_mapped - theta0) / dtheta
    theta_floor = np.floor(delta_theta)
    idx0 = theta_floor.astype(np.int64) % n_theta
    idx1 = (idx0 + 1) % n_theta
    wt = delta_theta - theta_floor

    # np.interp clamps to the ends, and uses the right interval when the
    # query lands exactly on a bin. Keep both semantics.
    s_clip = np.clip(ts, s_bins[0], s_bins[-1])
    upper = np.searchsorted(s_bins, s_clip, side='right')
    upper = np.clip(upper, 1, n_s - 1)
    lower = upper - 1
    ws = (s_clip - s_bins[lower]) / (s_bins[upper] - s_bins[lower])

    # The old path interpolates ANGLE first and then LONGITUDINAL POSITION.
    # Preserve that operation order to minimize floating-point differences.
    r0 = (1.0 - wt) * wrap[lower, idx0] + wt * wrap[lower, idx1]
    r1 = (1.0 - wt) * wrap[upper, idx0] + wt * wrap[upper, idx1]
    return r0 + ws * (r1 - r0)


def interpolate_wrap_radius1(self, ts, theta, wrap, theta_bins, s_bins,
                             *, vectorized_0061=False, audit_0061=False):
    ts = np.asarray(ts, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    wrap = np.asarray(wrap, dtype=np.float64)
    theta_bins = np.asarray(theta_bins, dtype=np.float64)
    s_bins = np.asarray(s_bins, np.float64)

    assert wrap.ndim == 2, f"Expected wrap to be 2D, got {wrap.shape}"

    # This is the proven legacy reference used by the 0061 audit. Do not
    # change its numerical behavior, including endpoint clamping.
    def reference():
        wrap_theta = interpolate_theta(self, theta, wrap, theta_bins)
        out = np.empty(ts.shape[0], dtype=np.float64)
        ts_clip = np.clip(ts, s_bins[0], s_bins[-1])
        for i in range(ts.shape[0]):
            out[i] = np.interp(ts_clip[i], s_bins, wrap_theta[i])
        return out

    if not vectorized_0061:
        return reference()

    # np.interp has no meaningful interpolation contract for non-monotone
    # xp; for single/duplicate s positions, fall back rather than divide by
    # zero or silently change historical behavior. Also guard short angular
    # tables, which cannot be interpolated by the original fast theta path.
    supported = (s_bins.ndim == 1 and theta_bins.ndim == 1
                 and wrap.shape == (len(s_bins), len(theta_bins))
                 and len(s_bins) >= 2 and len(theta_bins) >= 2
                 and np.all(np.isfinite(s_bins))
                 and np.all(np.diff(s_bins) > 0.0))
    if supported:
        result = _interpolate_wrap_radius_vectorized_0061(
            ts, theta, wrap, theta_bins, s_bins)
    else:
        result = reference()

    stats = getattr(self, '_wrap_vector_stats_0061', None)
    if stats is None:
        stats = {'calls': 0, 'points': 0, 'fallback_calls': 0,
                 'audited_points': 0, 'max_abs_error': 0.0}
        self._wrap_vector_stats_0061 = stats
    stats['calls'] += 1
    stats['points'] += ts.size
    stats['fallback_calls'] += not supported

    if audit_0061:
        legacy = reference() if supported else result
        # Vectorized reassociation changes last-bit rounding. Preserve a tight
        # absolute+relative numerical tolerance; do not claim bitwise equality.
        np.testing.assert_allclose(result, legacy, rtol=1e-12, atol=1e-12,
                                   equal_nan=True, err_msg='0061 wrap parity')
        finite = np.isfinite(result) & np.isfinite(legacy)
        if np.any(finite):
            stats['max_abs_error'] = max(
                stats['max_abs_error'],
                float(np.max(np.abs(result[finite] - legacy[finite]))))
        stats['audited_points'] += ts.size
    return result

def interpolate_wrap_radius2(self, ts, theta, wrap, theta_bins, s_bins):
        ts = np.asarray(ts, dtype=np.float64)
        theta = np.asarray(theta, dtype=np.float64)


        Ns = len(s_bins)
        Nt = len(theta_bins)
        N = len(ts)

        # -----------------------------
        # 1) linear interpolation in s
        # -----------------------------
        s_idx1 = np.searchsorted(s_bins, ts, side='right')
        s_idx1 = np.clip(s_idx1, 1, Ns - 1)
        s_idx0 = s_idx1 - 1

        s0 = s_bins[s_idx0]
        s1 = s_bins[s_idx1]
        ws = (ts - s0) / (s1 - s0 + 1e-12)   # (N,)

        # gather wrap values at the two neighboring s bins
        #print(wrap)
        r0 = wrap[s_idx0, :]   # (N, Nt)
        r1 = wrap[s_idx1, :]   # (N, Nt)

        r_s = (1.0 - ws[:, None]) * r0 + ws[:, None] * r1   # (N, Nt)

        # --------------------------------
        # 2) periodic linear interpolation in theta
        # --------------------------------
        period = 2.0 * np.pi
        theta0 = theta_bins[0]
        dtheta = theta_bins[1] - theta_bins[0]

        # map theta into same periodic interval as theta_bins
        theta_wrap = ((theta - theta0) % period) + theta0

        # fractional theta-bin coordinate
        t = (theta_wrap - theta0) / dtheta
        th_idx0 = np.floor(t).astype(np.int64) % Nt
        th_idx1 = (th_idx0 + 1) % Nt
        wt = t - np.floor(t)   # (N,)

        rows = np.arange(N)
        rv0 = r_s[rows, th_idx0]
        rv1 = r_s[rows, th_idx1]

        out = (1.0 - wt) * rv0 + wt * rv1
        return out

def interpolate_occ_profile3(self, ts, occ_profile, s_bins):
    ts = np.asarray(ts, dtype=np.float64)

    if isinstance(occ_profile, np.ndarray) and occ_profile.dtype == object:
        if occ_profile.ndim == 0:
            occ_profile = occ_profile.item()
        elif occ_profile.ndim == 1 and occ_profile.shape[0] == 1:
            occ_profile = occ_profile[0]

    if isinstance(s_bins, np.ndarray) and s_bins.dtype == object:
        if s_bins.ndim == 0:
            s_bins = s_bins.item()
        elif s_bins.ndim == 1 and s_bins.shape[0] == 1:
            s_bins = s_bins[0]

    occ_profile = np.asarray(occ_profile, dtype=np.float64).reshape(-1)
    s_bins = np.asarray(s_bins, dtype=np.float64).reshape(-1)

    ts = np.clip(ts, s_bins[0], s_bins[-1])
    return np.interp(ts, s_bins, occ_profile)



def interpolate_occ_profile2(self, ts, occ_profile, s_bins):
    ts = np.asarray(ts, dtype=np.float64)
    occ_profile = np.asarray(occ_profile, dtype=np.float64)
    s_bins = np.asarray(s_bins, dtype=np.float64)

    ts = np.clip(ts, s_bins[0], s_bins[-1])
    return np.interp(ts, s_bins, occ_profile)

def interpolate_occ_profile1(self, ts, occ_profile, s_bins):
    ts = np.asarray(ts, dtype=np.float64)
    occ_profile = np.asarray(occ_profile, dtype=np.float64)
    s_bins = np.asarray(s_bins, dtype=np.float64)

    ts = np.clip(ts, s_bins[0], s_bins[-1])
    s1 = np.searchsorted(s_bins, ts, side='right')
    s1 = np.clip(s1, 1, len(s_bins) - 1)
    s0 = s1 - 1

    s0v = s_bins[s0]
    s1v = s_bins[s1]
    a = (ts - s0v) / np.maximum(s1v - s0v, 1e-12)

    return (1.0 - a) * occ_profile[s0] + a * occ_profile[s1]
