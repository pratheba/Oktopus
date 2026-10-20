# Auto-split from PWLA_curve_handle.py -- _GeometryMixin
# Curve projection + analytic cylinder distance (used by BOTH sdf and udf).
import os as _pwla_os, sys as _pwla_sys
_pwla_sys.path.append(_pwla_os.path.abspath(_pwla_os.path.join(_pwla_os.path.dirname(__file__), '..')))

import os, pickle
import numpy as np
import os.path as op
import trimesh
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import PchipInterpolator
from handle_utils import CylindersMesh
from scipy.ndimage import gaussian_filter1d
from curve_utils.visualize_util import *
from curve_utils.curve_utils import *
from curve_functions._interpolate import interpolate_occ_profile1, interpolate_wrap_radius1
from curve_functions._frame import *
from curve_functions._update import update_wrap_profile_from_coords, update_wrap_occupancy_from_coords
#from curve_functions._localize import localize_samples


#n_sample_curve = 200
#n_sample_circle = 120

n_sample_curve = 200
n_sample_circle = 120
#n_sample_points = 12



class _GeometryMixin:

    def is_points_in_edge(self, points, vt0, vt1):
        # project points on line segment(edge), return if inside the edge 
        v0, t0 = vt0
        v1, t1 = vt1

        length = np.linalg.norm(v1 - v0)
        length_safe = length + 1e-12   # guard degenerate (zero-length) edges
        vec = (v1 - v0) / length_safe
        proj_len = (points - v0) @ vec
        inside_flag = np.logical_and(proj_len >= 0., proj_len <= length)

        # calculate natural coords for projected points
        ts = t0 + ((t1 - t0)/length_safe)* proj_len
        return inside_flag, ts

    def _projection_skeleton(self, N_discrete):
        """Build the (non-uniform) polyline skeleton + KDTree used for projection."""
        uniform_linear_points = np.linspace(0., 1., N_discrete, endpoint=False)
        ii = np.searchsorted(uniform_linear_points, self.key_ts)
        non_uniform_linear_points = np.insert(uniform_linear_points, ii, self.key_ts)
        non_uniform_linear_points = np.unique(non_uniform_linear_points)
        skeletal_verts = self.interpolate(non_uniform_linear_points, radius=False, frame=False)['points']
        tree = KDTree(skeletal_verts)
        return non_uniform_linear_points, skeletal_verts, tree

    def _assign_ts(self, samples, vidx, non_uniform_linear_points, skeletal_verts, outside):
        """Refine nearest-skeleton assignments onto adjacent curve edges in O(N).

        The reference implementation scanned the full ``vidx`` array once for
        every skeleton vertex. With ~200 curve samples and millions of grid
        candidates that creates a large O(N*V) memory-scan cost. Each query
        only depends on its own nearest vertex and at most two adjacent edges,
        so the same endpoint/XOR rules can be evaluated point-wise.
        """
        samples = np.asarray(samples, dtype=np.float64)
        vidx = np.asarray(vidx, dtype=np.int64).reshape(-1)
        params = np.asarray(non_uniform_linear_points, dtype=np.float64).reshape(-1)
        verts = np.asarray(skeletal_verts, dtype=np.float64)

        if samples.shape[0] != vidx.shape[0]:
            raise ValueError(
                "Projection sample/index length mismatch: "
                f"samples={samples.shape[0]} vidx={vidx.shape[0]}"
            )

        n = samples.shape[0]
        num_vert = verts.shape[0]
        result = np.full(n, -1.0, dtype=np.float64)
        if n == 0 or num_vert == 0:
            return result
        if num_vert == 1:
            result[:] = params[0]
            return result
        if np.any((vidx < 0) | (vidx >= num_vert)):
            raise ValueError("Projection vertex index out of range")

        middle = (vidx > 0) & (vidx < num_vert - 1)
        first = vidx == 0
        last = vidx == num_vert - 1

        # Interior points default to the nearest skeleton vertex parameter.
        result[middle] = params[vidx[middle]]
        if outside:
            result[first] = params[0]
            result[last] = params[-1]

        def project_edges(mask, i0, i1):
            ids = np.flatnonzero(mask)
            if ids.size == 0:
                return ids, np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float64)

            a = verts[i0[ids]]
            b = verts[i1[ids]]
            t0 = params[i0[ids]]
            t1 = params[i1[ids]]
            edge = b - a
            length = np.linalg.norm(edge, axis=1)
            length_safe = length + 1e-12
            direction = edge / length_safe[:, None]
            proj_len = np.einsum("ij,ij->i", samples[ids] - a, direction)
            inside = (proj_len >= 0.0) & (proj_len <= length)
            ts = t0 + ((t1 - t0) / length_safe) * proj_len
            return ids, inside, ts

        right_ids, right_inside, right_ts = project_edges(
            vidx < num_vert - 1,
            vidx,
            np.minimum(vidx + 1, num_vert - 1),
        )
        left_ids, left_inside, left_ts = project_edges(
            vidx > 0,
            np.maximum(vidx - 1, 0),
            vidx,
        )

        # Endpoint semantics are identical to the reference implementation.
        if right_ids.size:
            take = (vidx[right_ids] == 0) & right_inside
            result[right_ids[take]] = right_ts[take]
        if left_ids.size:
            take = (vidx[left_ids] == num_vert - 1) & left_inside
            result[left_ids[take]] = left_ts[take]

        if np.any(middle):
            # Interior vertices use XOR: exactly one adjacent segment must
            # accept the projection, otherwise retain the nearest vertex t.
            right_ok = np.zeros(n, dtype=bool)
            left_ok = np.zeros(n, dtype=bool)
            right_value = np.zeros(n, dtype=np.float64)
            left_value = np.zeros(n, dtype=np.float64)
            right_ok[right_ids] = right_inside
            left_ok[left_ids] = left_inside
            right_value[right_ids] = right_ts
            left_value[left_ids] = left_ts

            mid_ids = np.flatnonzero(middle)
            use_edge = right_ok[mid_ids] ^ left_ok[mid_ids]
            chosen = mid_ids[use_edge]
            choose_right = right_ok[chosen]
            result[chosen[choose_right]] = right_value[chosen[choose_right]]
            result[chosen[~choose_right]] = left_value[chosen[~choose_right]]

        return result

    def _projection_edges_0055(self, verts, params):
        """Reuse adjacent-segment geometry for exact interval refinement.

        The original _assign_ts rebuilds 2 segment vectors, lengths and
        directions for every grid batch. An interval tree is fixed during a
        given adaptation; cache only its tiny (roughly 200-edge) geometry.
        The strong reference to ``verts`` prevents id-reuse cache collisions.
        """
        entry = getattr(self, '_projection_edges_cache_0055', None)
        if entry is not None and entry[0] is verts and entry[1] is params:
            return entry[2]
        edge = np.asarray(verts[1:] - verts[:-1], dtype=np.float64)
        length = np.linalg.norm(edge, axis=1)
        length_safe = length + 1e-12
        direction = edge / length_safe[:, None]
        dt_over_len = (params[1:] - params[:-1]) / length_safe
        data = (length, direction, dt_over_len)
        self._projection_edges_cache_0055 = (verts, params, data)
        return data

    def _assign_ts_fast_0055(self, samples, vidx, params, verts, outside=False):
        """Same nearest-vertex + adjacent-edge XOR rules as _assign_ts.

        Gather per-sample precomputed segments rather than rebuilding lengths,
        directions, per-edge index lists, and full-size boolean scatter arrays.
        Endpoint behavior, including the ``-1`` sentinel, is unchanged.
        """
        samples = np.asarray(samples, dtype=np.float64)
        vidx = np.asarray(vidx, dtype=np.int64).reshape(-1)
        params = np.asarray(params, dtype=np.float64).reshape(-1)
        verts = np.asarray(verts, dtype=np.float64)
        if samples.shape[0] != vidx.shape[0]:
            raise ValueError('Projection sample/index length mismatch: '
                             f'samples={samples.shape[0]} vidx={vidx.shape[0]}')
        n = samples.shape[0]
        nv = verts.shape[0]
        result = np.full(n, -1.0, dtype=np.float64)
        if n == 0 or nv == 0:
            return result
        if nv == 1:
            result[:] = params[0]
            return result
        if np.any((vidx < 0) | (vidx >= nv)):
            raise ValueError('Projection vertex index out of range')
        length, direction, dt_over_len = self._projection_edges_0055(verts, params)
        last = nv - 1
        interior = (vidx > 0) & (vidx < last)
        first = vidx == 0
        end = vidx == last
        result[interior] = params[vidx[interior]]
        if outside:
            result[first] = params[0]
            result[end] = params[-1]

        # The two adjacent edges are gathered in vertex-index order. At the
        # endpoints one of these is a placeholder, masked out below.
        ri = np.minimum(vidx, last - 1)
        li = np.maximum(vidx - 1, 0)
        rproj = np.einsum('ij,ij->i', samples - verts[ri], direction[ri])
        lproj = np.einsum('ij,ij->i', samples - verts[li], direction[li])
        rok = (~end) & (rproj >= 0.0) & (rproj <= length[ri])
        lok = (~first) & (lproj >= 0.0) & (lproj <= length[li])

        take_r = (first & rok) | (interior & rok & ~lok)
        take_l = (end & lok) | (interior & lok & ~rok)
        if np.any(take_r):
            indices = ri[take_r]
            result[take_r] = params[indices] + dt_over_len[indices] * rproj[take_r]
        if np.any(take_l):
            indices = li[take_l]
            result[take_l] = params[indices] + dt_over_len[indices] * lproj[take_l]
        return result

    def curve_projection(self, samples, N_discrete=n_sample_curve, outside=False):
        non_uniform_linear_points, skeletal_verts, tree = self._projection_skeleton(N_discrete)
        # not accurate for radius-varying skeleton
        _, vidx = tree.query(samples)
        # basically project samples onto the piecewise linear curve
        return self._assign_ts(samples, vidx, non_uniform_linear_points, skeletal_verts, outside)

    def curve_projection_candidates(self, samples, K, N_discrete=n_sample_curve, outside=False):
        """Return (N, K) candidate arc-length params: the refined projection onto
        each of the K nearest skeleton vertices.  For a looping / self-close curve
        these candidates land on DIFFERENT arms, letting the caller pick the arm a
        point actually belongs to (see localize_samples' k_project path).
        K == 1 reproduces curve_projection exactly.
        """
        non_uniform_linear_points, skeletal_verts, tree = self._projection_skeleton(N_discrete)
        K = int(max(1, min(K, skeletal_verts.shape[0])))
        _, vidx_k = tree.query(samples, k=K)
        if vidx_k.ndim == 1:
            vidx_k = vidx_k[:, None]
        cols = [
            self._assign_ts(samples, vidx_k[:, j], non_uniform_linear_points, skeletal_verts, outside)
            for j in range(vidx_k.shape[1])
        ]
        return np.stack(cols, axis=1)

    def _projection_skeleton_interval(self, s0, s1, N_discrete,
                                      cache_0054=False, leafsize_0056=10):
        """Build the identical interval projection tree, optionally reusing it.

        Cache is keyed by the exact source curve geometry and interval. A changed
        key-point position or parameter invalidates it automatically. The cache
        has only one entry, so it cannot accumulate one tree per tentacle.
        """
        s_min = max(0.0, min(float(s0), float(s1)))
        s_max = min(1.0, max(float(s0), float(s1)))
        leafsize_0056 = int(leafsize_0056)
        if not 1 <= leafsize_0056 <= 128:
            raise ValueError('0056 KD-tree leaf size must be in [1, 128]')
        if abs(s_max - s_min) < 1e-12:
            return None, None, None, s_min, s_max
        cache_key = None
        if cache_0054:
            # interpolate(..., radius=False, frame=False) depends only on
            # key_ts/key_points. Their bytes are tiny compared with a voxel
            # batch and detect *in-place* edits as well as array replacement.
            ts = np.asarray(self.key_ts)
            pos = np.asarray(self.key_points)
            cache_key = (s_min, s_max, int(N_discrete), leafsize_0056,
                         ts.shape, ts.dtype.str, ts.tobytes(),
                         pos.shape, pos.dtype.str, pos.tobytes())
            entry = getattr(self, '_projection_interval_cache_0054', None)
            if entry is not None and entry[0] == cache_key:
                return entry[1]
        uniform_linear_points = np.linspace(s_min, s_max, int(N_discrete), endpoint=True)
        key_inside = self.key_ts[(self.key_ts >= s_min) & (self.key_ts <= s_max)]
        non_uniform_linear_points = np.unique(np.concatenate([
            np.array([s_min, s_max], dtype=np.float64),
            uniform_linear_points,
            key_inside,
        ]))
        skeletal_verts = self.interpolate(
            non_uniform_linear_points, radius=False, frame=False
        )["points"]
        tree = KDTree(skeletal_verts, leafsize=leafsize_0056)
        result = (non_uniform_linear_points, skeletal_verts, tree, s_min, s_max)
        if cache_key is not None:
            self._projection_interval_cache_0054 = (cache_key, result)
        return result

    def curve_projection_interval(self, samples, s0, s1, N_discrete=n_sample_curve, outside=False,
                                  projection_workers_0054=1,
                                  projection_cache_0054=False,
                                  projection_refine_fast_0055=False,
                                  projection_profile_split_0055=False,
                                  projection_leafsize_0056=10):
        """
        Project samples only onto a source interval [s0, s1].

        This avoids adapt samples snapping to a nearby but wrong branch of the
        full avatar curve before the src_0/src_1 crop.
        """
        samples = np.asarray(samples, dtype=np.float64)
        pts, verts, tree, s_min, s_max = self._projection_skeleton_interval(
            s0, s1, N_discrete, cache_0054=projection_cache_0054,
            leafsize_0056=projection_leafsize_0056)
        if tree is None:
            return np.full(samples.shape[0], s_min, dtype=np.float64)
        workers = int(projection_workers_0054)
        if not 1 <= workers <= 16:
            raise ValueError('0054 projection workers must be between 1 and 16')
        # Preserve the historical SciPy call when single-threaded.
        from time import perf_counter
        start_knn = perf_counter() if projection_profile_split_0055 else None
        _, vidx = (tree.query(samples) if workers == 1
                   else tree.query(samples, workers=workers))
        start_refine = perf_counter() if projection_profile_split_0055 else None
        ts = (self._assign_ts_fast_0055(samples, vidx, pts, verts, outside)
              if projection_refine_fast_0055
              else self._assign_ts(samples, vidx, pts, verts, outside))
        if start_knn is not None:
            self._projection_knn_seconds_0055 = (
                getattr(self, '_projection_knn_seconds_0055', 0.0)
                + start_refine - start_knn)
            self._projection_refine_seconds_0055 = (
                getattr(self, '_projection_refine_seconds_0055', 0.0)
                + perf_counter() - start_refine)
        good = ts >= 0.0
        ts[good] = np.clip(ts[good], s_min, s_max)
        return ts

    def curve_projection_interval_candidates(self, samples, s0, s1, K, N_discrete=n_sample_curve, outside=False,
                                             projection_workers_0054=1,
                                             projection_cache_0054=False,
                                             projection_refine_fast_0055=False,
                                             projection_profile_split_0055=False,
                                             projection_leafsize_0056=10):
        """(N, K) candidate arc-length params restricted to [s0, s1] -- the
        loop-aware companion to curve_projection_interval. For a curve that loops
        WITHIN the interval, the K nearest skeleton candidates land on different
        arms, letting localize_samples' k_project path pick the correct one.
        K == 1 reproduces curve_projection_interval exactly."""
        samples = np.asarray(samples, dtype=np.float64)
        pts, verts, tree, s_min, s_max = self._projection_skeleton_interval(
            s0, s1, N_discrete, cache_0054=projection_cache_0054,
            leafsize_0056=projection_leafsize_0056)
        if tree is None:
            return np.full((samples.shape[0], 1), s_min, dtype=np.float64)
        K = int(max(1, min(K, verts.shape[0])))
        workers = int(projection_workers_0054)
        if not 1 <= workers <= 16:
            raise ValueError('0054 projection workers must be between 1 and 16')
        from time import perf_counter
        start_knn = perf_counter() if projection_profile_split_0055 else None
        _, vidx_k = (tree.query(samples, k=K) if workers == 1
                     else tree.query(samples, k=K, workers=workers))
        start_refine = perf_counter() if projection_profile_split_0055 else None
        if vidx_k.ndim == 1:
            vidx_k = vidx_k[:, None]
        cols = []
        for j in range(vidx_k.shape[1]):
            ts = (self._assign_ts_fast_0055(samples, vidx_k[:, j], pts, verts, outside)
                  if projection_refine_fast_0055
                  else self._assign_ts(samples, vidx_k[:, j], pts, verts, outside))
            good = ts >= 0.0
            ts[good] = np.clip(ts[good], s_min, s_max)
            cols.append(ts)
        if start_knn is not None:
            self._projection_knn_seconds_0055 = (
                getattr(self, '_projection_knn_seconds_0055', 0.0)
                + start_refine - start_knn)
            self._projection_refine_seconds_0055 = (
                getattr(self, '_projection_refine_seconds_0055', 0.0)
                + perf_counter() - start_refine)
        return np.stack(cols, axis=1)

    def calc_cylinder_SDF(self, vs):
        ts = self.curve_projection(vs, outside=True)

        intpl = self.interpolate(ts, radius_type='cylinder')
        proj_vs = intpl['points']
        yz_rs = intpl['radius']
        frame_mat = intpl['frame']

        x_rs = self.calc_x_radius(ts)
        radius = np.concatenate([x_rs[:,None], yz_rs], axis=1)

        # frame: (N, 3,3), vs (N, 3)
        samples_local = np.einsum('nij,nj->ni', frame_mat, (vs - proj_vs))
        samples_local /= radius
        norms_cyl = np.linalg.norm(samples_local, axis=1)
        
        vx = 2*ts - 1
        samples_local[:, 0] += vx
        xpos = vx >= 0.
        xneg = np.logical_not(xpos)

        norms_max = np.linalg.norm(samples_local, axis=1, ord=np.inf)
        #if self.start_ball_x is None:
        norms_cyl[xneg] = np.maximum(norms_cyl[xneg], norms_max[xneg])

        #if self.end_ball_x is None:
        norms_cyl[xpos] = np.maximum(norms_cyl[xpos], norms_max[xpos])
            
        return norms_cyl - 1.
    
    def calc_std_cylinder_SDF(self, vs):
        # assume this is a standard cylinder
        # vs all inside the cylinder
        ts = self.curve_projection(vs)
        intpl = self.interpolate(ts)
        proj_vs = intpl['points']
        yz_rs = intpl['radius']
        
        dist_cyl = np.linalg.norm(vs - proj_vs, axis=1)
        dist_cyl = yz_rs[:,0] - dist_cyl

        tsd = np.minimum(ts, 1-ts)
        curve_length, _ = self.calc_curve_length()
        dist_side = tsd*curve_length

        dist = np.minimum(dist_cyl, dist_side)
        return -dist
    
    def calc_global_implicit(self, vs, return_coords=False):
        ts = self.curve_projection(vs, outside=True)

        intpl = self.interpolate(ts)
        proj_vs = intpl['points']
        yz_rs = intpl['radius']
        frame_mat = intpl['frame']

        x_rs = self.calc_x_radius(ts)
        radius = np.concatenate([x_rs[:,None], yz_rs], axis=1)

        # frame: (N, 3,3), vs (N, 3)
        samples_local = np.einsum('nij,nj->ni', frame_mat, (vs - proj_vs))
        samples_local /= radius
        norms_cyl = np.linalg.norm(samples_local, axis=1)
        # return norms_cyl - 1.
        signs = np.sign(norms_cyl - 1.)
        
        nearest_local = samples_local / norms_cyl[:, None]
        nearest_global = nearest_local* radius
        # NOTE: here it should be the inverse of frames
        # so it is the transpose(since they are unitary mats)
        # and we simply modify it in einsum: nij,nj->ni => nji,nj->ni
        nearest_global = np.einsum('nji,nj->ni', frame_mat, nearest_global)
        nearest_global += proj_vs
        ds = np.linalg.norm(nearest_global - vs, axis=1)
        if return_coords:
            vx = 2*ts - 1
            samples_local[:, 0] += vx
            return ds*signs, samples_local, ts
        else:
            return ds*signs
