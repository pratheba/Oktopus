# Auto-split from PWLA_curve_handle.py -- _InterpolateMixin
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



class _InterpolateMixin:

    def inverse_transform(self, samples_local, coords):
        """
        samples_local: (N,3) where x is packed: x = w/xr + (2*coords-1)
        coords: (N,) in [0,1]
        Returns: points_world (N,3)
        """
        coords = np.asarray(coords)
        sl = np.asarray(samples_local)

        # interpolate curve quantities at coords
        intpl = self.interpolate(coords)      # must return points, radius(yz), frame
        proj_vs = intpl["points"]             # (N,3)
        yz_radius = intpl["radius"]           # (N,2)
        frame_mat = intpl["frame"]            # (N,3,3) world->local (as used in localize_samples)

        # build full radius [x, y, z]
        x_radius = self.calc_x_radius(coords)             # (N,)
        radius = np.concatenate([x_radius[:, None], yz_radius], axis=1)  # (N,3)

        # unpack x (remove vx)
        vx = 2.0 * coords - 1.0
        local_n = sl.copy()
        local_n[:, 0] -= vx

        # unnormalize
        local0 = local_n * radius   # (N,3) now equals [w,u,v] in local frame

        # back to world: since localize used frame_mat @ (p - proj),
        # inverse uses (p - proj) = frame_mat^T @ local0
        world_offset = np.einsum("nij,nj->ni", np.transpose(frame_mat, (0,2,1)), local0)
        points_world = proj_vs + world_offset
        return points_world

    def interpolate(self, non_uniform_linear_skeletal_points, points=True, radius=True, frame=True, radius_type='train'):
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
            if radius_type == 'train':
                key_radius = self.key_train_radius
            elif radius_type == 'cylinder':
                key_radius = self.key_cylinder_radius
            else:
                raise ValueError(f'Unknown radius type: {radius_type}')
            #print(self.key_ts)
            #print(key_radius)
            #print("********")
            rs_ts = np.stack([
                np.interp(ts, self.key_ts, key_radius[:, 0]),
                np.interp(ts, self.key_ts, key_radius[:, 1])
            ]).T
            res['radius'] = rs_ts

        if frame:
            ts_rot = np.clip(ts, a_min=0.0, a_max=1.0)
            res["frame"] = self._interp_frames(self.key_ts, self.key_frame, ts_rot)
            # requirement of Slerp from Scipy
            ############ UNCOMMENT IF NOT WORKING ############################
            #ts_rot = np.clip(ts, a_min=1e-10, a_max=(1 - 1e-10))
            #idx = np.searchsorted(self.key_ts, ts_rot)
            #idx = np.clip(idx, 0, len(self.key_ts)-1)
            #frame_ts= self.key_frame[idx]
            #frame_ts = self.rot_slerp(ts_rot).as_matrix()
            #res['frame'] = frame_ts

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

    def map_coords_by_arclen(self, coords_src):
        # coords_src are in [0,1]
        a = np.interp(coords_src, ts_src, A_src)     # src ts -> src arc fraction
        coords_tgt = np.interp(a, A_tgt, ts_tgt)     # target arc fraction -> target ts
        return coords_tgt


    def interpolate_snug_scale_field(self, scale_field, s, theta):
        """
        Interpolate a local avatar snug correction field.

        scale_field:
            {
                "scale":      (Ns, Nt)
                "s_bins":     (Ns,)
                "theta_bins": (Nt,)
            }

        Returns:
            scale(s, theta), shape (N,)

        Usage:
            r_src_corrected = r_src * scale(s, theta)

        scale < 1.0  => tighter accessory
        scale > 1.0  => looser / more clearance
        """
        if scale_field is None:
            return np.ones_like(np.asarray(s, dtype=np.float64))

        scale = np.asarray(scale_field["scale"], dtype=np.float64)
        s_bins = np.asarray(scale_field["s_bins"], dtype=np.float64)
        theta_bins = np.asarray(scale_field["theta_bins"], dtype=np.float64)

        s = np.asarray(s, dtype=np.float64).reshape(-1)
        theta = np.asarray(theta, dtype=np.float64).reshape(-1)

        Ns, Nt = scale.shape

        if len(s_bins) != Ns:
            print(
                "[snug_scale] rebuilding s_bins because length mismatch:",
                "scale=", scale.shape,
                "s_bins=", len(s_bins),
            )
            s_bins = np.linspace(0.0, 1.0, Ns)

        if len(theta_bins) != Nt:
            print(
                "[snug_scale] rebuilding theta_bins because length mismatch:",
                "scale=", scale.shape,
                "theta_bins=", len(theta_bins),
            )
            theta_bins = np.linspace(-np.pi, np.pi, Nt, endpoint=False)

        # ---- interpolate along s ----
        s_clip = np.clip(s, s_bins[0], s_bins[-1])
        i1 = np.searchsorted(s_bins, s_clip, side="right")
        i1 = np.clip(i1, 1, Ns - 1)
        i0 = i1 - 1

        s0 = s_bins[i0]
        s1 = s_bins[i1]
        ws = (s_clip - s0) / (s1 - s0 + 1e-12)

        scale_s = (1.0 - ws[:, None]) * scale[i0, :] + ws[:, None] * scale[i1, :]

        # ---- periodic theta interpolation ----
        period = 2.0 * np.pi
        theta0 = theta_bins[0]
        dtheta = period / float(Nt)

        theta_wrap = ((theta - theta0) % period) + theta0
        t = (theta_wrap - theta0) / dtheta

        j0 = np.floor(t).astype(np.int64) % Nt
        j1 = (j0 + 1) % Nt
        wt = t - np.floor(t)

        rows = np.arange(len(s))
        out = (1.0 - wt) * scale_s[rows, j0] + wt * scale_s[rows, j1]
        return out

    def interpolate_snug_delta_field(self, snug_field, s, theta):
        """
        Interpolate the additive (signed SDF offset) snug field at samples
        (s, theta).

        snug_field is the same dict produced by build_avatar_snug_scale_field;
        it must contain the "delta" key alongside "scale".

        Returns an array of shape (N,) of additive SDF offsets in world units.
        Convention:
            delta > 0  -> avatar is closer than target (penetration / tight)
                          -> caller should subtract this from vals_final to
                             push the iso-surface OUTWARD.
            delta < 0  -> avatar is farther than target (loose)
                          -> subtracting makes vals_final larger, pulling the
                             iso-surface INWARD by |delta|.
        """
        if snug_field is None or "delta" not in snug_field:
            return np.zeros_like(np.asarray(s, dtype=np.float64))

        delta = np.asarray(snug_field["delta"], dtype=np.float64)
        s_bins = np.asarray(snug_field["s_bins"], dtype=np.float64)
        theta_bins = np.asarray(snug_field["theta_bins"], dtype=np.float64)

        s = np.asarray(s, dtype=np.float64).reshape(-1)
        theta = np.asarray(theta, dtype=np.float64).reshape(-1)

        Ns, Nt = delta.shape

        if len(s_bins) != Ns:
            s_bins = np.linspace(0.0, 1.0, Ns)
        if len(theta_bins) != Nt:
            theta_bins = np.linspace(-np.pi, np.pi, Nt, endpoint=False)

        # interpolate along s
        s_clip = np.clip(s, s_bins[0], s_bins[-1])
        i1 = np.searchsorted(s_bins, s_clip, side="right")
        i1 = np.clip(i1, 1, Ns - 1)
        i0 = i1 - 1
        s0 = s_bins[i0]
        s1 = s_bins[i1]
        ws = (s_clip - s0) / (s1 - s0 + 1e-12)
        delta_s = (1.0 - ws[:, None]) * delta[i0, :] + ws[:, None] * delta[i1, :]

        # periodic theta interpolation
        period = 2.0 * np.pi
        theta0 = theta_bins[0]
        dtheta = period / float(Nt)
        theta_wrap = ((theta - theta0) % period) + theta0
        t = (theta_wrap - theta0) / dtheta
        j0 = np.floor(t).astype(np.int64) % Nt
        j1 = (j0 + 1) % Nt
        wt = t - np.floor(t)

        rows = np.arange(len(s))
        out = (1.0 - wt) * delta_s[rows, j0] + wt * delta_s[rows, j1]
        return out

    def interpolate_adapt(self, ts, adapt_arg):
        avatar_arclen_coords = self.localize_adapt(ts, adapt_arg)
        avatar_curve_handle = adapt_arg['avatar_curve_handle']

        # source yz radius at coords_src
        accessory_intpl = self.interpolate(ts)    # uses self.key_ts
        avatar_intpl = avatar_curve_handle.core.interpolate(avatar_arclen_coords)

        return accessory_intpl, avatar_intpl, avatar_arclen_coords

    def interpolate_wrap_radius(self, ts, theta):
        """
        Vectorized interpolation of directional wrap radius.

        Args:
            ts:    (N,) curve coordinates in [0,1]
            theta: (N,) angles in radians

        Returns:
            r:     (N,) interpolated directional radius
        """
        if self.key_wrap_radius is None or self.wrap_s_bins is None or self.wrap_theta_bins is None:
            # fallback to ellipse-equivalent radial support
            intpl = self.interpolate(ts, points=False, frame=False)
            ry = intpl['radius'][:, 0]
            rz = intpl['radius'][:, 1]
            ct = np.cos(theta)
            st = np.sin(theta)
            denom = np.sqrt((ct * ct) / (ry * ry + 1e-12) + (st * st) / (rz * rz + 1e-12))
            return 1.0 / (denom + 1e-12)

        ts = np.asarray(ts, dtype=np.float64)
        theta = np.asarray(theta, dtype=np.float64)

        s_bins = self.wrap_s_bins              # (Ns,)
        theta_bins = self.wrap_theta_bins      # (Nt,)
        wrap = self.key_wrap_radius                # (Ns, Nt)

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

#    def interpolate_wrap_radius1(self, ts, theta, wrap, theta_bins, s_bins):
#        ts = np.asarray(ts, dtype=np.float64)
#        theta = np.asarray(theta, dtype=np.float64)
#
#        Ns = len(s_bins)
#        Nt = len(theta_bins)
#        N = len(ts)
#
#        # -----------------------------
#        # 1) linear interpolation in s
#        # -----------------------------
#        s_idx1 = np.searchsorted(s_bins, ts, side='right')
#        s_idx1 = np.clip(s_idx1, 1, Ns - 1)
#        s_idx0 = s_idx1 - 1
#
#        s0 = s_bins[s_idx0]
#        s1 = s_bins[s_idx1]
#        ws = (ts - s0) / (s1 - s0 + 1e-12)   # (N,)
#
#        # gather wrap values at the two neighboring s bins
#        #print(wrap)
#        r0 = wrap[s_idx0, :]   # (N, Nt)
#        r1 = wrap[s_idx1, :]   # (N, Nt)
#
#        r_s = (1.0 - ws[:, None]) * r0 + ws[:, None] * r1   # (N, Nt)
#
#        # --------------------------------
#        # 2) periodic linear interpolation in theta
#        # --------------------------------
#        period = 2.0 * np.pi
#        theta0 = theta_bins[0]
#        dtheta = theta_bins[1] - theta_bins[0]
#
#        # map theta into same periodic interval as theta_bins
#        theta_wrap = ((theta - theta0) % period) + theta0
#
#        # fractional theta-bin coordinate
#        t = (theta_wrap - theta0) / dtheta
#        th_idx0 = np.floor(t).astype(np.int64) % Nt
#        th_idx1 = (th_idx0 + 1) % Nt
#        wt = t - np.floor(t)   # (N,)
#
#        rows = np.arange(N)
#        rv0 = r_s[rows, th_idx0]
#        rv1 = r_s[rows, th_idx1]
#
#        out = (1.0 - wt) * rv0 + wt * rv1
#        return out

    def periodic_interpolate(self, ts_detail, ts, radius):
        x = np.mod(ts_detail, 1.0)

        ts_periodic = np.concatenate([ts, ts[1:]])
        radius_periodic = np.concatenate([radius, radius[1:]])

        x_copy = x.copy()
        x_copy[x_copy < ts_periodic[0]] += 1.0

        return np.interp(x_copy, ts_periodic, radius_periodic)

    def interpolate_radius_field(self, ts, radius_field):
        ts = np.asarray(ts, dtype=np.float64)
        radius_field = np.asarray(radius_field, dtype=np.float64)

        return np.stack([
            np.interp(ts, self.key_ts, radius_field[:, 0]),
            np.interp(ts, self.key_ts, radius_field[:, 1]),
        ], axis=1)


    def interpolate_stretch1(self, ts, stretch_arg):
        t0 = float(stretch_arg['t0'])
        t1 = float(stretch_arg['t1'])
        stretch_scale = float(stretch_arg.get('stretch_scale', stretch_arg.get('length', 1.0)))
        direction = stretch_arg.get('direction', 'forward')
        eps = 1e-12

        old_len = t1 - t0
        new_len = old_len * stretch_scale
        delta = new_len - old_len

        ts_new = ts.copy()

        mid = (ts >= t0) & (ts <= t1)

        if direction == 'forward':
            post = ts > t1
            ts_new[mid] = t0 + ((ts[mid] - t0) / (old_len + eps)) * new_len
            ts_new[post] = ts[post] + delta

        elif direction == 'backward':
            pre = ts < t0
            ts_new[mid] = t1 - ((t1 - ts[mid]) / (old_len + eps)) * new_len
            ts_new[pre] = ts[pre] - delta

        ts_new = np.clip(ts_new, 0.0, 1.0)

        radius = np.stack([
            np.interp(ts_new, self.key_ts, self.key_radius[:, 0]),
            np.interp(ts_new, self.key_ts, self.key_radius[:, 1])
        ]).T

        intpl = self.interpolate(ts_new, radius=False)
        intpl['radius'] = radius
        return intpl, ts_new



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


    def inverse_transform(self, samples_local, ts):
        x_proj = 2*ts - 1
        res = self.interpolate(ts)
        verts, yz_rs, frame = res['points'], res['radius'], res['frame']

        x_rs = self.calc_x_radius(ts)
        radius = np.concatenate([x_rs[:,None], yz_rs], axis=1)
        # F(v - Pv) = (1,ry,rz)*(v_ - (v_x,0,0))
        # i.e. samples_local[:,0] - samples_x
        samples_global = samples_local.copy()
        samples_global[:, 0] -= x_proj
        samples_global *= radius
        # NOTE: here it should be the inverse of frames
        # so it is the transpose(since they are unitary mats)
        # and we simply modify it in einsum: nij,nj->ni => nji,nj->ni
        samples_global = np.einsum('nji,nj->ni', frame, samples_global)
        samples_global += verts
        return samples_global

    def generate_samples(self, num_samples):
        # mapping: cylinder <-> y^2+z^2 = 1, -1 <= x <= 1
        samples = np.random.uniform(-1., 1., size=(num_samples, 3))
        yz_norms = np.linalg.norm(samples[:,1:], axis=1)
        inside = yz_norms <= 1
        samples = samples[inside]

        # coords: [-1,1] to [0,1]
        samples_x = samples[:, 0]
        ts = (samples_x + 1) / 2
        res = self.interpolate(ts, radius_type='cylinder')
        verts, yz_rs, frame = res['points'], res['radius'], res['frame']

        # F(v - Pv) = (1,ry,rz)*(v_ - (v_x,0,0))
        # i.e. samples[:,0] - samples_x
        samples_global = np.zeros((samples.shape[0], 3))
        samples_global[:, 1:] = samples[:, 1:]*yz_rs
        # NOTE: here it should be the inverse of frames
        # so it is the transpose(since they are unitary mats)
        # and we simply modify it in einsum: nij,nj->ni => nji,nj->ni
        samples_global = np.einsum('nji,nj->ni', frame, samples_global)
        samples_global += verts

        # sdf_scales = np.sqrt(np.product(yz_rs, axis=1))
        return {
            'samples': samples_global,
            'samples_local':samples,
            'radius': yz_rs,
            'coords': ts,
        }


    def export_data(self):
        return {
            'key_points': self.key_points,
            'key_radius': self.key_radius,
            'z_axis': self.z_axis,
            'ball': {
                'start_x': self.start_ball_x,
                'end_x': self.end_ball_x,
            }
        }
    
    def export_vis(self):
        vidx = np.arange(self.key_points.shape[0])
        return {
            'vertices': self.key_points,
            'edges': np.asarray([vidx[:-1], vidx[1:]]).T,
        }
