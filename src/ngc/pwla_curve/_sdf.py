# Auto-split from PWLA_curve_handle.py -- _SDFMixin
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



class _SDFMixin:

    def is_points_in_edge(self, points, vt0, vt1):
        # project points on line segment(edge), return if inside the edge 
        v0, t0 = vt0
        v1, t1 = vt1

        length = np.linalg.norm(v1 - v0)
        vec = (v1 - v0) / length
        proj_len = (points - v0) @ vec
        inside_flag = np.logical_and(proj_len >= 0., proj_len <= length)

        # calculate natural coords for projected points
        ts = t0 + ((t1 - t0)/length)* proj_len
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
        """Refine one arc-length param per sample given an assigned skeleton vertex id.

        This is exactly the per-vertex edge-projection logic that curve_projection
        used inline; factored out so it can be reused per candidate column.
        """
        samples3D_to_skeleton = -1 * np.ones(samples.shape[0])
        num_vert = skeletal_verts.shape[0]
        for vid in range(num_vert):
            sample_index = np.argwhere(vidx == vid).flatten()
            if len(sample_index) == 0:
                continue

            samples_v = samples[sample_index]

            if 0 < vid < num_vert - 1:
                # middle part
                samples3D_to_skeleton[sample_index] = non_uniform_linear_points[vid]

                in1, px1 = self.is_points_in_edge(
                    samples_v,
                    (skeletal_verts[vid], non_uniform_linear_points[vid]),
                    (skeletal_verts[vid+1], non_uniform_linear_points[vid+1])
                )
                in2, px2 = self.is_points_in_edge(
                    samples_v,
                    (skeletal_verts[vid-1], non_uniform_linear_points[vid-1]),
                    (skeletal_verts[vid], non_uniform_linear_points[vid])
                )
                in_p = np.logical_xor(in1, in2)
                px = (in1*px1 + in2*px2)[in_p]
                samples3D_to_skeleton[sample_index[in_p]] = px

            elif vid == 0:
                if num_vert == 1:
                    samples3D_to_skeleton[sample_index] = non_uniform_linear_points[vid]
                    continue
                in1, px1 = self.is_points_in_edge(
                    samples_v,
                    (skeletal_verts[vid], non_uniform_linear_points[vid]),
                    (skeletal_verts[vid+1], non_uniform_linear_points[vid+1])
                )
                if outside:
                    samples3D_to_skeleton[sample_index] = non_uniform_linear_points[vid]
                samples3D_to_skeleton[sample_index[in1]] = px1[in1]

            else:
                in2, px2 = self.is_points_in_edge(
                    samples_v,
                    (skeletal_verts[vid-1], non_uniform_linear_points[vid-1]),
                    (skeletal_verts[vid], non_uniform_linear_points[vid]),
                )
                if outside:
                    samples3D_to_skeleton[sample_index] = non_uniform_linear_points[vid]
                samples3D_to_skeleton[sample_index[in2]] = px2[in2]

        return samples3D_to_skeleton

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

    def _projection_skeleton_interval(self, s0, s1, N_discrete):
        """Build the polyline skeleton + KDTree restricted to [s0, s1]."""
        s_min = max(0.0, min(float(s0), float(s1)))
        s_max = min(1.0, max(float(s0), float(s1)))
        if abs(s_max - s_min) < 1e-12:
            return None, None, None, s_min, s_max
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
        tree = KDTree(skeletal_verts)
        return non_uniform_linear_points, skeletal_verts, tree, s_min, s_max

    def curve_projection_interval(self, samples, s0, s1, N_discrete=n_sample_curve, outside=False):
        """
        Project samples only onto a source interval [s0, s1].

        This avoids adapt samples snapping to a nearby but wrong branch of the
        full avatar curve before the src_0/src_1 crop.
        """
        samples = np.asarray(samples, dtype=np.float64)
        pts, verts, tree, s_min, s_max = self._projection_skeleton_interval(s0, s1, N_discrete)
        if tree is None:
            return np.full(samples.shape[0], s_min, dtype=np.float64)
        _, vidx = tree.query(samples)
        ts = self._assign_ts(samples, vidx, pts, verts, outside)
        good = ts >= 0.0
        ts[good] = np.clip(ts[good], s_min, s_max)
        return ts

    def curve_projection_interval_candidates(self, samples, s0, s1, K, N_discrete=n_sample_curve, outside=False):
        """(N, K) candidate arc-length params restricted to [s0, s1] -- the
        loop-aware companion to curve_projection_interval. For a curve that loops
        WITHIN the interval, the K nearest skeleton candidates land on different
        arms, letting localize_samples' k_project path pick the correct one.
        K == 1 reproduces curve_projection_interval exactly."""
        samples = np.asarray(samples, dtype=np.float64)
        pts, verts, tree, s_min, s_max = self._projection_skeleton_interval(s0, s1, N_discrete)
        if tree is None:
            return np.full((samples.shape[0], 1), s_min, dtype=np.float64)
        K = int(max(1, min(K, verts.shape[0])))
        _, vidx_k = tree.query(samples, k=K)
        if vidx_k.ndim == 1:
            vidx_k = vidx_k[:, None]
        cols = []
        for j in range(vidx_k.shape[1]):
            ts = self._assign_ts(samples, vidx_k[:, j], pts, verts, outside)
            good = ts >= 0.0
            ts[good] = np.clip(ts[good], s_min, s_max)
            cols.append(ts)
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
