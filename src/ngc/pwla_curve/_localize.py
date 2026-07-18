# Auto-split from PWLA_curve_handle.py -- _LocalizeMixin
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



class _LocalizeMixin:


    def localize_samples_test(self, idx, pointcloudsamples, norm=1.0):
        sample_keypoint_map = self.curve_projection(pointcloudsamples)
        sample_keypoint_map_range = np.logical_and(sample_keypoint_map >= 0., sample_keypoint_map <= 1.)
        sample_index = np.arange(pointcloudsamples.shape[0])

        ### Keep only the points that fall within the rane of 0 and 1
        sample_keypoint_map = sample_keypoint_map[sample_keypoint_map_range]
        pointcloudsamples = pointcloudsamples[sample_keypoint_map_range]
        sample_index = sample_index[sample_keypoint_map_range]

        # interpolate with the new additional non linear skeletal keypoints
        intpl = self.interpolate(sample_keypoint_map)

        ## The new keypoiints in 3D world coord system based on the curve projection from the surface/space samples
        proj_vs = intpl['points']
        yz_radius = intpl['radius']
        frame_mat = intpl['frame']

        # If the end ball is None then x_radius  = 1.0 
        x_radius = self.calc_x_radius(sample_keypoint_map)
        radius = np.concatenate([x_radius[:,None], yz_radius], axis=1)

        # frame: (N, 3,3), vs (N, 3)
        # The vector of the keypoint to the skeletam point is rotated using the rotation from
        samples_local0 = np.einsum('nij,nj->ni', frame_mat, (pointcloudsamples - proj_vs))
        # And all are bounding to radius
        samples_local = samples_local0.copy()
        #stats = compute_local_centering_stats(samples_local, sample_keypoint_map, n_bins=24, min_count=150)
        #C_old, C_new, C_key_new = compute_centered_curve_world(self, stats)
        #C_old, C_new = compute_centered_curve_world(self, stats)
        #C_new_smooth = gaussian_filter1d(C_new, sigma=2, axis=0)
        #export_curve_points_as_ply(C_old, C_new_smooth, str(idx)+"_curve_compare_points.ply")
        #export_shape_and_curves_as_ply(
        #    points=pointcloudsamples,
        #    C_old=C_old,
        #    C_new=C_new_smooth,
        #    out_path=str(idx)+"_shape_and_curves.ply"
        #)
        trimesh.Trimesh(vertices = np.array(samples_local), process=False).export(str(idx)+"_localsample.ply")

        #plot_centroid_offsets_from_origin(stats)
        #plot_centered_curve_local_projections(stats)
        #plot_centroid_path_with_origin(stats)
        #plot_local_centering_stats(stats)
        #plot_local_bins(stats, bins=[10, 25, 40, 60, 80])   
        #plot_local_bins_with_drift_clean(stats, bins=[10, 25, 40, 60, 80])


    def _select_projection_candidate(self, samples, ts_cand, radius_type,
                                     runtime_cylinder_radius_scale=1.0,
                                     runtime_cylinder_radius_add=0.0):
        """Given (N,K) candidate arc-length params, pick per sample the candidate
        with the smallest cylinder-normalized radial distance -- i.e. the curve arm
        the point actually belongs to.  This mirrors the inside_cyl gate exactly, so
        a surface point on a looped arm stops snapping to the opposite arm."""
        samples = np.asarray(samples, dtype=np.float64)
        N = samples.shape[0]
        K = ts_cand.shape[1]
        best_norm = np.full(N, np.inf)
        best_t = ts_cand[:, 0].astype(np.float64).copy()
        for j in range(K):
            t = ts_cand[:, j].astype(np.float64)
            valid = (t >= 0.0) & (t <= 1.0)
            if not np.any(valid):
                continue
            tv = np.clip(t, 0.0, 1.0)
            intpl = self.interpolate(tv, radius_type="cylinder", radius=True, frame=True)
            proj = np.asarray(intpl['points'], dtype=np.float64)
            frame = np.asarray(intpl['frame'], dtype=np.float64)
            r_cyl = np.asarray(intpl['radius'], dtype=np.float64)
            r_cyl = r_cyl * float(runtime_cylinder_radius_scale) + float(runtime_cylinder_radius_add)
            x_radius = self.calc_x_radius(tv)
            radius_cyl = np.concatenate([x_radius[:, None], r_cyl], axis=1)
            local = np.einsum('nij,nj->ni', frame, (samples - proj))
            local_cyl = local / (radius_cyl + 1e-12)
            norms = np.linalg.norm(local_cyl, axis=1)
            norms[~valid] = np.inf
            take = norms < best_norm
            best_norm[take] = norms[take]
            best_t[take] = t[take]
        return best_t

    def localize_samples(self, pointcloudsamples, return_sdf=False, norm=1.0, update_curve=False, update_radius=False, outside=False, name='', radius_type='cylinder', runtime_cylinder_radius_scale=1.0, runtime_cylinder_radius_add=0.0, projection_s0=None, projection_s1=None, k_project=1):
        # Owned-volume gate: drop samples that fall outside the dilated voxel
        # mask of surface_points_owned BEFORE running the cylinder projection.
        # The returned `inside` indices still index into the ORIGINAL input.
        # No-op when the mask isn't built (owned points missing or gate disabled).

        if projection_s0 is not None and projection_s1 is not None:
            if k_project is not None and int(k_project) > 1:
                # Loop-aware interval projection: pick, per sample, the in-interval
                # arm with the smallest cylinder-normalized radius.
                ts_cand = self.curve_projection_interval_candidates(
                    pointcloudsamples, projection_s0, projection_s1,
                    int(k_project), outside=outside,
                )
                sample_keypoint_map = self._select_projection_candidate(
                    pointcloudsamples, ts_cand, radius_type,
                    runtime_cylinder_radius_scale, runtime_cylinder_radius_add,
                )
            else:
                sample_keypoint_map = self.curve_projection_interval(
                    pointcloudsamples,
                    projection_s0,
                    projection_s1,
                    outside=outside,
                )
        elif k_project is not None and int(k_project) > 1:
            # Loop-aware projection: evaluate the K nearest skeleton candidates and
            # keep, per sample, the arm with the smallest cylinder-normalized radius.
            ts_cand = self.curve_projection_candidates(
                pointcloudsamples, int(k_project), outside=outside
            )
            sample_keypoint_map = self._select_projection_candidate(
                pointcloudsamples, ts_cand, radius_type,
                runtime_cylinder_radius_scale, runtime_cylinder_radius_add,
            )
        else:
            sample_keypoint_map = self.curve_projection(pointcloudsamples, outside=outside)
        sample_keypoint_map_range = np.logical_and(sample_keypoint_map >= 0., sample_keypoint_map <= 1.)
        sample_index = np.arange(pointcloudsamples.shape[0])

        ### Keep only the points that fall within the rane of 0 and 1
        if update_curve:
            pointcloudsamples0 = pointcloudsamples.copy()
        sample_keypoint_map = sample_keypoint_map[sample_keypoint_map_range]
        pointcloudsamples = pointcloudsamples[sample_keypoint_map_range]
        sample_index = sample_index[sample_keypoint_map_range]

        # interpolate with the new additional non linear skeletal keypoints
        intpl = self.interpolate(sample_keypoint_map, radius_type=radius_type, radius=False, frame=True)

        ## The new keypoiints in 3D world coord system based on the curve projection from the surface/space samples
        proj_vs = intpl['points']
        #yz_radius = intpl['radius']
        frame_mat = intpl['frame']

        # frame: (N, 3,3), vs (N, 3)
        # The vector of the keypoint to the skeletam point is rotated using the rotation from
        #print("frame_mat nan:", np.isnan(frame_mat).any())
        #print("proj_vs nan:", np.isnan(proj_vs).any())
        #print("yz_radius nan:", np.isnan(yz_radius).any())
        samples_local0 = np.einsum('nij,nj->ni', frame_mat, (pointcloudsamples - proj_vs))
        #print(samples_local0)
        # And all are bounding to radius
        w, u, v = samples_local0[:,0], samples_local0[:, 1], samples_local0[:, 2]
        # train radius for model coords
        r_train = self.interpolate(sample_keypoint_map, points=False, frame=False, radius=True, radius_type="train")["radius"]
        # cylinder radius for inside filtering
        r_cyl = self.interpolate(sample_keypoint_map, points=False, frame=False, radius=True, radius_type="cylinder")["radius"]
        r_cyl = (
            r_cyl * float(runtime_cylinder_radius_scale)
            + float(runtime_cylinder_radius_add)
        )
        x_radius = self.calc_x_radius(sample_keypoint_map)

        radius_train_temp = np.concatenate([x_radius[:, None], r_train], axis=1)
        radius_cyl_temp = np.concatenate([x_radius[:, None], r_cyl], axis=1)

        samples_local_train = samples_local0 / (radius_train_temp + 1e-12)
        samples_local_cyl = samples_local0 / (radius_cyl_temp + 1e-12)

        norms_cyl = np.linalg.norm(samples_local_cyl, axis=1)
        inside_cyl = norms_cyl <= norm

        rho = np.sqrt(u**2 + v**2)

        u_n = samples_local_train[:, 1]
        v_n = samples_local_train[:, 2]

        angle = np.arctan2(v_n, u_n)
        rho_n = np.sqrt(u_n**2 + v_n**2)

        if return_sdf:
            return norms_cyl - 1, sample_index

        inside = sample_index[inside_cyl]

        
        # in std cylinder
        #norms = np.linalg.norm(samples_local, axis=1)
        if return_sdf:
            return norms_cyl - 1, sidx
        
        inside = sample_index[inside_cyl]

        # NOTE: vs -> (vx, *, *). (vert -> (vx, 0, 0))
        # [0,1] -> [-1,1]
        vx = 2*sample_keypoint_map - 1.0
        samples_local_train[:, 0] += vx
        
        return {
            'samples': pointcloudsamples[inside_cyl],
            'samples_local': samples_local_train[inside_cyl],
            'coords': sample_keypoint_map[inside_cyl],
            'rho': rho[inside_cyl],
            'rho_n': rho_n[inside_cyl],
            'angles': angle[inside_cyl],
            'radius': r_train[inside_cyl],
            'radius_cylinder': r_cyl[inside_cyl],
            'frame_mat': frame_mat[inside_cyl]
        }, inside
    
    def localize_samples_global(self, vs):
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
        vx = 2*ts - 1
        samples_local[:, 0] += vx
        return samples_local, ts


    def localize_samples_mix(self, vs, mix_arg):
        ts = self.curve_projection(vs)
        ts_range = np.logical_and(ts >= 0., ts <= 1.)
        sidx = np.arange(vs.shape[0])
        ts = ts[ts_range]
        vs = vs[ts_range]
        sidx = sidx[ts_range]
        
        intpl = self.interpolate_mix(ts, mix_arg)
        proj_vs = intpl['points']
        yz_rs = intpl['radius']
        frame_mat = intpl['frame']

        x_rs = self.calc_x_radius(ts)
        radius = np.concatenate([x_rs[:,None], yz_rs], axis=1)

        # frame: (N, 3,3), vs (N, 3)
        samples_local0 = np.einsum('nij,nj->ni', frame_mat, (vs - proj_vs))
        w, u, v = samples_local0[:,0], samples_local0[:, 1], samples_local0[:, 2]
        samples_local = samples_local0.copy()
        samples_local /= (radius + 1e-12)
        rho = np.sqrt(v**2 + u**2)
        #angle = np.arctan2(v, u)
        u_n = samples_local[:,1] #u / (radius[:,1] + 1e-12)
        v_n = samples_local[:,2] #v / (radius[:,2] + 1e-12)
        angle = np.arctan2(v_n, u_n)
        rho_n = np.sqrt(v_n**2 + u_n**2)

        # in std cylinder
        norms = np.linalg.norm(samples_local, axis=1)
        inside_cyl = norms <= 1
        inside = sidx[inside_cyl]

        # NOTE: vs -> (vx, *, *). (vert -> (vx, 0, 0))
        # [0,1] -> [-1,1]
        vx = 2*ts - 1
        samples_local[:, 0] += vx
        return {
            'samples': vs[inside_cyl],
            'samples_local': samples_local[inside_cyl],
            'coords': ts[inside_cyl],
            'rho': rho[inside_cyl],
            'rho_n': rho_n[inside_cyl],
            'angles': angle[inside_cyl],
            'radius': yz_rs[inside_cyl],
        }, inside


    def localize_adapt(self, ts, adapt_arg):
        curve_length, cum_length = self.calc_curve_length()
        accessory_arclen = cum_length / (curve_length  + 1e-12)

        avatar_curve_handle = adapt_arg['avatar_curve_handle']
        avatar_curve_length, cum_length = avatar_curve_handle.core.calc_curve_length()
        avatar_arclen = cum_length / (avatar_curve_length + 1e-12)

        acc_arclen_coords = np.interp(ts, self.key_ts, accessory_arclen)
        avatar_arclen_coords = np.interp(acc_arclen_coords, avatar_arclen, avatar_curve_handle.core.key_ts)
        return avatar_arclen_coords

    def localize_stretch(self, stretch_arg):
        self.key_ts0 = self.key_ts.copy()
        self.key_radius0 = self.key_radius.copy()
        self.key_points0 = self.key_points.copy()
        self.key_frame0 = self.key_frame.copy()
        #self.rotation0 = self.rotation.copy()
        #self.rot_slerp0 = self.rot_slerp.copy()

        #points = self.stretch_uniform(stretch_arg)
        #points = self.stretch_from_end_smooth_nonuniform(stretch_arg)
        #points = self.stretch_from_end_smooth_nonuniform
        mode = stretch_arg.get("mode", None)
        if mode == "end_extension":
            points = self.stretch_end_extension(stretch_arg)
        elif mode == "start_extension":
            points = self.stretch_start_extension(stretch_arg)
        elif ('t0' in stretch_arg) and ('t1' in stretch_arg):
            points = self.stretch_interval_smooth_nonuniform(stretch_arg)
        else:
            points = self.stretch_from_end_smooth_nonuniform(stretch_arg)
        self.key_points = points
        self.update_coords()
        self.update_frame()
        #self.update_radius()

    def restore_stretch(self):
        self.key_ts = self.key_ts0.copy()
        self.key_radius = self.key_radius0.copy()
        self.key_points = self.key_points0.copy()
        self.key_frame = self.key_frame0.copy()
        #self.rotation = self.rotation0.copy()
        #self.rot_slerp = self.rot_slerp0.copy()
    def localize_samples_stretch(self, vs, stretch_arg, return_sdf=False):
        # for stretch or offset 
        # Project samples from surface onto the curve to get ts - they key points
        #return self.localize_samples(vs)
        ts = self.curve_projection(vs)
        valid_range = np.logical_and(ts >= 0., ts <= 1.)

        sidx = np.arange(vs.shape[0])
        ts = ts[valid_range]
        vs = vs[valid_range]
        sidx = sidx[valid_range]
        

        # Interpolate the keypoints (points on curve), radius and frame
        intpl, ts_new = self.interpolate_stretch(ts, stretch_arg)
        #intpl, ts_new = self.interpolate(ts) #, stretch_arg)
        #intpl = self.interpolate(ts) #, stretch_arg)
        #ts_new = ts.copy()
        proj_vs = intpl['points']
        frame_mat = intpl['frame']
        yz_radius = intpl['radius']


        # Build local coordinate system
        x_rs = self.calc_x_radius(ts_new)
        radius = np.concatenate([x_rs[:,None], yz_radius], axis=1)
        # frame: (N, 3,3), vs (N, 3)
        samples_local0 = np.einsum('nij,nj->ni', frame_mat, (vs - proj_vs))
        w, u, v = samples_local0[:,0], samples_local0[:, 1], samples_local0[:, 2]
        #samples_local = np.einsum('nij,nj->ni', frame_mat, (vs - proj_vs))
        #w, u, v = samples_local[:,0], samples_local[:, 1], samples_local[:, 2]
        rho = np.sqrt(v**2 + u**2)
        theta = np.arctan2(v, u)


        samples_local = samples_local0.copy()
        samples_local /= (radius + 1e-12)
        #samples_local /= radius
        norms = np.linalg.norm(samples_local, axis=1)
        inside_cyl = norms <= 1.0
        inside = sidx[inside_cyl]
        u_n = samples_local[:,1] # u / (radius[:,1] + 1e-12)
        v_n = samples_local[:,2] # v / (radius[:,2] + 1e-12)
        #u_n = u / (radius[:,1]**2 + 1e-12)
        #v_n = v / (radius[:,2]**2 + 1e-12)
        angle = np.arctan2(v_n, u_n)
        rho_n = np.sqrt(v_n**2 + u_n**2)


       # geometry coords mapped to [-1, 1]
        vx_base = 2.0*ts_new - 1.0
        #vx_base = 2.0*ts - 1.0
        samples_local[:, 0] += vx_base

        stretch_scale = float(stretch_arg.get('stretch_scale', stretch_arg.get('length', 1.0)))
        detail_tiles_cfg = stretch_arg.get('detail_tiles', 1.0)
        mode = stretch_arg.get("mode", None)

        if detail_tiles_cfg == 'auto':
            detail_tiles = stretch_scale
        else:
            detail_tiles = float(detail_tiles_cfg)

        eps_region = stretch_arg.get('eps_region', 0.03)
        eps_seam = stretch_arg.get('eps_seam', 0.05)

        # Default: no detail remap
        ts_used = ts_new.copy()
        w_seam = np.ones_like(ts_new, dtype=np.float64)

        # Only do interval-style wrapping if both t0 and t1 are present
        # and the mode is an interval mode.
        interval_modes = {None, "interval_forward", "interval_backward", "interval"}

        if ('t0' in stretch_arg) and ('t1' in stretch_arg) and (mode in interval_modes):
            t0 = float(stretch_arg['t0'])
            t1 = float(stretch_arg['t1'])

            w_region = make_detail_mask(ts_new, t0, t1, eps_region)
            eps = 1e-12
            tau = np.clip((ts_new - t0) / ((t1 - t0) + eps), 0.0, 1.0)
            ts_tile_phase = np.mod((detail_tiles * tau), 1.0)

            w_seam = seam_fade(ts_tile_phase, eps_seam)

            ts_wrapped = t0 + (t1 - t0) * ts_tile_phase
            use_wrap = w_region > 0.5
            ts_used[use_wrap] = ts_wrapped[use_wrap]

        elif ('t0' in stretch_arg) and (mode == "end_extension"):
            t0 = float(stretch_arg['t0'])

            w_region = (ts_new >= t0).astype(np.float64)
            eps = 1e-12
            tau = np.clip((ts_new - t0) / ((1.0 - t0) + eps), 0.0, 1.0)
            ts_tile_phase = np.mod((detail_tiles * tau), 1.0)

            w_seam = seam_fade(ts_tile_phase, eps_seam)

            ts_wrapped = t0 + (1.0 - t0) * ts_tile_phase
            use_wrap = w_region > 0.5
            ts_used[use_wrap] = ts_wrapped[use_wrap]

        elif ('t1' in stretch_arg) and (mode == "start_extension"):
            t1 = float(stretch_arg['t1'])

            w_region = (ts_new <= t1).astype(np.float64)
            eps = 1e-12
            tau = np.clip(ts_new / (t1 + eps), 0.0, 1.0)
            ts_tile_phase = np.mod((detail_tiles * tau), 1.0)

            w_seam = seam_fade(ts_tile_phase, eps_seam)

            ts_wrapped = t1 * ts_tile_phase
            use_wrap = w_region > 0.5
            ts_used[use_wrap] = ts_wrapped[use_wrap]

        vx_used = 2.0 * ts_used - 1.0




        #ts_detail = np.mod(stretch_arg['length'] * ts_new, 1.0)
        # NOTE: vs -> (vx, *, *). (vert -> (vx, 0, 0))
        # [0,1] -> [-1,1]
        samples_detail  = samples_local0.copy()
        samples_detail /= (radius + 1e-12)
        #x_radius_detail = self.calc_x_radius(ts_detail)
        #r_detail = np.concatenate([x_rs[:,None], intpl['radius_detail']], axis=1)
        #samples_detail /= r_detail #intpl['radius_detail']
        samples_detail[:,0] += vx_used

        return {
            #'samples': vs[inside_cyl],
            'samples_local': samples_local[inside_cyl],
            'samples_detail': samples_detail[inside_cyl],
            'coords': ts_new[inside_cyl],
            #'coords': ts[inside_cyl],
            'coords_detail': ts_used[inside_cyl], 
            'w_seam': w_seam[inside_cyl],
            'rho': rho[inside_cyl],
            'rho_n': rho_n[inside_cyl],
            'angles': angle[inside_cyl],
            'radius': yz_radius[inside_cyl],
        }, inside

    def normalized_arclen_keypoints(self):
        """
        Returns A[k] in [0,1] at each keypoint, monotonic.
        """
        curve_length, cum_length = self.calc_curve_length()  # your version returns (L, cumulative)
        return cum_length / (curve_length + 1e-12)

    def _compute_anchor_from_support(self, support_data, at="end", coord=None):
        coords = np.asarray(support_data["coords"], dtype=np.float64)

        if at == "start":
            idx = 0
        elif at == "end":
            idx = -1
        elif at == "coord":
            if coord is None:
                raise ValueError("coord must be provided when at='coord'")
            idx = int(np.argmin(np.abs(coords - float(coord))))
        else:
            raise ValueError(f"Unknown anchor mode: {at}")

        out = {
            "point": support_data["points"][idx].copy(),
            "frame": support_data["frame"][idx].copy(),
            "radius": support_data["radius"][idx].copy(),
            "coord": float(support_data["coords"][idx]),
        }

        if "x_radius" in support_data:
            out["x_radius"] = float(support_data["x_radius"][idx])
        else:
            out["x_radius"] = 1.0
        return out

    def _compute_anchor_from_support_old(self, support_data, attach="end", coord=None):
        coords = np.array(support_data["coords"], dtype=np.float64)
        if attach == "start":
            idx = 0
        elif attach == "end":
            idx = -1
        elif attach == "coord":
            if coord is None:
                raise ValueError("coord must be provided when at='coord'")
            idx = int(np.argmin(np.abs(coords - float(coord))))
        else:
            raise ValueError(f"Unknown anchor mode: {attach}")
        out = {
            "point": support_data["points"][idx].copy(),
            "frame": support_data["frame"][idx].copy(),
            "radius": support_data["radius"][idx].copy(),
            "coord": float(support_data["coords"][idx]),
        }

        if "x_radius" in support_data:
            out["x_radius"] = float(support_data["x_radius"][idx])
        else:
            out["x_radius"] = 1.0
        return out


    def _build_dependent_support_from_anchor(
        self,
        dep_template,
        parent_anchor,
        scale_w=1.0,
        scale_y=1.0,
        scale_z=1.0,
        radius_scale_y=1.0,
        radius_scale_z=1.0,
    ):
        """
        dep_template:
            local_points : (K,3) in parent-anchor local [w,u,v]
            local_frames : (K,3,3) relative to parent-anchor frame
            radius       : (K,2)
            coords       : (K,)
            x_radius     : (K,) optional
        """
        Fp = parent_anchor["frame"]   # rows [T,N,B]
        p0 = parent_anchor["point"]

        local_points = dep_template["local_points"].copy()
        local_points[:, 0] *= scale_w
        local_points[:, 1] *= scale_y
        local_points[:, 2] *= scale_z

        points = p0[None, :] + local_points @ Fp
        frames = np.einsum("kij,jm->kim", dep_template["local_frames"], Fp)

        radius = dep_template["radius"].copy()
        radius[:, 0] *= radius_scale_y
        radius[:, 1] *= radius_scale_z

        out = {
            "points": points,
            "frame": frames,
            "radius": radius,
            "coords": dep_template["coords"].copy(),
        }

        if "x_radius" in dep_template:
            out["x_radius"] = dep_template["x_radius"].copy() * scale_w

        return out


    def _interpolate_dependent_support(self, support, query_coords):
        s = np.asarray(support["coords"], dtype=np.float64)
        q = np.clip(np.asarray(query_coords, dtype=np.float64), s[0], s[-1])

        points = np.zeros((len(q), 3), dtype=np.float64)
        for j in range(3):
            points[:, j] = np.interp(q, s, support["points"][:, j])

        frames = self._interp_frames(s, support["frame"], q)

        radius = np.zeros((len(q), 2), dtype=np.float64)
        radius[:, 0] = np.interp(q, s, support["radius"][:, 0])
        radius[:, 1] = np.interp(q, s, support["radius"][:, 1])

        out = {
            "points": points,
            "frame": frames,
            "radius": radius,
            "coords": q,
        }

        if "x_radius" in support:
            out["x_radius"] = np.interp(q, s, support["x_radius"])

        return out
    
    def map_coords_to_by_arclen(
        self,
        coords_src,
        target_core,
        src_0: float = 0.0,
        src_1: float = 1.0,
        tgt_0: float = 0.0,
        tgt_1: float = 1.0):
        eps = 1e-12
        coords_src = np.asarray(coords_src, dtype=np.float64)

        arclen_src = np.asarray(self.normalized_arclen_keypoints(), dtype=np.float64)
        arclen_tgt = np.asarray(target_core.normalized_arclen_keypoints(), dtype=np.float64)

        ts_src = np.asarray(self.key_ts, dtype=np.float64)
        ts_tgt = np.asarray(target_core.key_ts, dtype=np.float64)

        coords_src_clip = np.clip(coords_src, min(src_0, src_1), max(src_0, src_1))

        arc_src = np.interp(coords_src_clip, ts_src, arclen_src)
        arc_src_0 = np.interp(src_0, ts_src, arclen_src)
        arc_src_1 = np.interp(src_1, ts_src, arclen_src)

        denom = max(abs(arc_src_1 - arc_src_0), eps)
        u = (arc_src - arc_src_0) / denom
        if arc_src_1 < arc_src_0:
            u = -u
        u = np.clip(u, 0.0, 1.0)
        u = u * u * (3.0 - 2.0 * u)

        arc_tgt_0 = np.interp(tgt_0, ts_tgt, arclen_tgt)
        arc_tgt_1 = np.interp(tgt_1, ts_tgt, arclen_tgt)
        arc_tgt = arc_tgt_0 + u * (arc_tgt_1 - arc_tgt_0)

        keep = np.r_[True, np.diff(arclen_tgt) > eps]
        arclen_tgt_mono = arclen_tgt[keep]
        ts_tgt_mono = ts_tgt[keep]

        if arclen_tgt_mono.shape[0] < 2:
            return np.full_like(coords_src, fill_value=tgt_0, dtype=np.float64)

        coords_tgt = np.interp(arc_tgt, arclen_tgt_mono, ts_tgt_mono)
        return coords_tgt

        return accessory_data, avatar_data, inside_final

    def localize_samples_split_dependent(self, vs, dep_arg):
        """
        Same as dependent localization, but dep_template is built from a suffix
        of the SAME original accessory curve.
        """
        return self.localize_samples_dependent(vs, dep_arg)

    def localize_samples_dependent(self, vs, dep_arg):
        """
        Dependent support localization:
        - localize samples on avatar as source
        - crop source interval [src_0, src_1]
        - map to dependent coords [tgt_0, tgt_1]
        - build dependent support from parent anchor
        - apply shaft-like scaling for local quantities
        """
        avatar_data, inside = self.localize_samples(vs, norm=10.0)

        src_0 = dep_arg["src_0"]
        src_1 = dep_arg["src_1"]
        tgt_0 = dep_arg["tgt_0"]
        tgt_1 = dep_arg["tgt_1"]

        avatar_coords = avatar_data["coords"]
        valid_map = (avatar_coords >= min(src_0, src_1)) & (avatar_coords <= max(src_0, src_1))

        for k, v in avatar_data.items():
            if isinstance(v, np.ndarray) and v.shape[0] == valid_map.shape[0]:
                avatar_data[k] = v[valid_map]

        inside_final = inside.copy()
        inside_final = inside_final[valid_map]

        avatar_coords = avatar_data["coords"]
        avatar_samples_local = avatar_data["samples_local"].copy()

        vx_avatar = 2.0 * avatar_coords - 1.0
        w_n_avatar = avatar_samples_local[:, 0] - vx_avatar
        u_n_avatar = avatar_samples_local[:, 1]
        v_n_avatar = avatar_samples_local[:, 2]

        avatar_radius_y = avatar_data["radius"][:, 0]
        avatar_radius_z = avatar_data["radius"][:, 1]
        tangent_avatar = self.calc_x_radius(avatar_coords)

        w_avatar = w_n_avatar * (tangent_avatar + 1e-12)
        u_avatar = u_n_avatar * (avatar_radius_y + 1e-12)
        v_avatar = v_n_avatar * (avatar_radius_z + 1e-12)

        rho_avatar = np.sqrt(u_avatar**2 + v_avatar**2)
        theta_avatar = np.arctan2(v_avatar, u_avatar)

        dep_coords = tgt_0 + ((avatar_coords - src_0) / (src_1 - src_0 + 1e-12)) * (tgt_1 - tgt_0)

        parent_support_data = dep_arg["parent_support_data"]
        parent_anchor = self._compute_anchor_from_support(
            parent_support_data,
            at=dep_arg.get("parent_anchor_at", "end"),
            coord=dep_arg.get("parent_anchor_coord", None)
        )

        dep_template = dep_arg["dep_template"]
        parent_anchor_meta = dep_template["parent_anchor_meta"]

        global_scale = float(dep_arg.get("scale", 1.0))
        use_parent_aniso = bool(dep_arg.get("use_parent_anisotropic_scale", True))

        if use_parent_aniso:
            scale_w = global_scale * (
                parent_anchor["x_radius"] / (parent_anchor_meta["x_radius"] + 1e-12)
            )
            scale_y = global_scale * (
                parent_anchor["radius"][0] / (parent_anchor_meta["radius"][0] + 1e-12)
            )
            scale_z = global_scale * (
                parent_anchor["radius"][1] / (parent_anchor_meta["radius"][1] + 1e-12)
            )
        else:
            scale_w = global_scale
            scale_y = global_scale
            scale_z = global_scale

        dep_support = self._build_dependent_support_from_anchor(
            dep_template,
            parent_anchor,
            scale_w=scale_w,
            scale_y=scale_y,
            scale_z=scale_z,
            radius_scale_y=scale_y,
            radius_scale_z=scale_z,
        )

        dep_intpl = self._interpolate_dependent_support(dep_support, dep_coords)

        tangent_dep = dep_intpl.get("x_radius", np.ones_like(dep_coords))
        dep_radius_y = dep_intpl["radius"][:, 0]
        dep_radius_z = dep_intpl["radius"][:, 1]

        delta_theta = np.deg2rad(float(dep_arg.get("rot_deg", 0.0)))
        #translate_local = adapt_arg.get("translate_local", None)
        theta_tgt = theta_avatar + delta_theta

        scale_w_sample = tangent_dep / (tangent_avatar + 1e-12)
        scale_y_sample = dep_radius_y / (avatar_radius_y + 1e-12)
        scale_z_sample = dep_radius_z / (avatar_radius_z + 1e-12)

        if dep_arg.get("wrap_radius", False):
            source_npz = np.load(dep_arg["wrap_npz_src"], allow_pickle=True)["arr_0"].item()[dep_arg["wrap_src_key"]]
            target_npz = np.load(dep_arg["wrap_npz_tgt"], allow_pickle=True)["arr_0"].item()[dep_arg["wrap_tgt_key"]]

            r_src = interpolate_wrap_radius1(
                self, avatar_coords, theta_avatar,
                source_npz["key_wrap_radius"],
                source_npz["wrap_theta_bins"],
                source_npz["wrap_s_bins"]
            )
            r_tgt = interpolate_wrap_radius1(
                dep_arg["target_core_for_wrap"], dep_coords, theta_tgt,
                target_npz["key_wrap_radius"],
                target_npz["wrap_theta_bins"],
                target_npz["wrap_s_bins"]
            )

            scale_rho = (global_scale * r_tgt) / (r_src + 1e-12)
        else:
            scale_rho = global_scale * 0.5 * (scale_y_sample + scale_z_sample)

        rho_dep = rho_avatar * scale_rho
        u_dep = rho_dep * np.cos(theta_tgt)
        v_dep = rho_dep * np.sin(theta_tgt)
        w_dep = w_avatar * scale_w_sample

        w_n_dep = w_dep / (tangent_dep + 1e-12)
        u_n_dep = u_dep / (dep_radius_y + 1e-12)
        v_n_dep = v_dep / (dep_radius_z + 1e-12)

        vx_dep = 2.0 * dep_coords - 1.0
        samples_local_dep = np.stack([w_n_dep + vx_dep, u_n_dep, v_n_dep], axis=1)

        rho_n_dep = np.sqrt(u_n_dep**2 + v_n_dep**2)
        angles_dep = np.arctan2(v_n_dep, u_n_dep)

        dependent_data = dict(avatar_data)
        dependent_data["coords"] = dep_coords
        dependent_data["samples_local"] = samples_local_dep
        dependent_data["angles"] = angles_dep
        dependent_data["rho_n"] = rho_n_dep
        dependent_data["rho"] = rho_dep
        dependent_data["radius"] = dep_intpl["radius"]
        dependent_data["frame"] = dep_intpl["frame"]
        dependent_data["points"] = dep_intpl["points"]
        dependent_data["x_radius"] = tangent_dep

        return dependent_data, avatar_data, inside_final

    def build_adapt_control_field(
        self,
        coords_query,
        n_control=12,
        smooth_points_sigma=1.0,
        smooth_radius_sigma=1.0,
        rebuild_frames=True,
        preserve_endpoints=True,
        radius_type="train",
    ):
        """
        Build the actual control field used by adapt inference.

        This is different from wrap_adapt_n_keypoints:
          - wrap_adapt_n_keypoints only smooths/downsamples key_wrap_radius.
          - this function smooths/rebuilds points, frames and radii used to
            recompute samples_local. Therefore it directly affects inference.

        The control coordinates include first and last query coordinate, with
        n_control-2 equally spaced controls in between.
        """
        coords_query = np.asarray(coords_query, dtype=np.float64).reshape(-1)
        if coords_query.shape[0] == 0:
            raise ValueError("build_adapt_control_field got empty coords_query")

        c0 = float(np.min(coords_query))
        c1 = float(np.max(coords_query))
        if abs(c1 - c0) < 1e-12:
            c1 = min(1.0, c0 + 1e-6)
            c0 = max(0.0, c0 - 1e-6)

        n_control = int(max(2, n_control))
        ctrl_s = np.linspace(c0, c1, n_control)

        ctrl = self.interpolate(ctrl_s, points=True, radius=True, frame=True, radius_type=radius_type)
        ctrl_points = np.asarray(ctrl["points"], dtype=np.float64).copy()
        ctrl_radius = np.asarray(ctrl["radius"], dtype=np.float64).copy()
        ctrl_frames = np.asarray(ctrl["frame"], dtype=np.float64).copy()
        ctrl_x_radius = self.calc_x_radius(ctrl_s).astype(np.float64).copy()

        if smooth_points_sigma and smooth_points_sigma > 0:
            pts0 = ctrl_points.copy()
            ctrl_points = gaussian_filter1d(
                ctrl_points,
                sigma=float(smooth_points_sigma),
                axis=0,
                mode="nearest",
            )
            if preserve_endpoints:
                ctrl_points[0] = pts0[0]
                ctrl_points[-1] = pts0[-1]

        if smooth_radius_sigma and smooth_radius_sigma > 0:
            ctrl_radius = gaussian_filter1d(
                ctrl_radius,
                sigma=float(smooth_radius_sigma),
                axis=0,
                mode="nearest",
            )
            ctrl_x_radius = gaussian_filter1d(
                ctrl_x_radius,
                sigma=float(smooth_radius_sigma),
                axis=0,
                mode="nearest",
            )

        if rebuild_frames:
            old_z = self.z_axis
            try:
                # Preserve the existing first-frame orientation but rebuild
                # low-twist frames from the smoothed control curve.
                self.z_axis = ctrl_frames[:, 2, :].copy()
                ctrl_frames = self.get_new_frame(ctrl_points)
            finally:
                self.z_axis = old_z

        points_q = np.stack([
            np.interp(coords_query, ctrl_s, ctrl_points[:, 0]),
            np.interp(coords_query, ctrl_s, ctrl_points[:, 1]),
            np.interp(coords_query, ctrl_s, ctrl_points[:, 2]),
        ], axis=1)

        radius_q = np.stack([
            np.interp(coords_query, ctrl_s, ctrl_radius[:, 0]),
            np.interp(coords_query, ctrl_s, ctrl_radius[:, 1]),
        ], axis=1)

        x_radius_q = np.interp(coords_query, ctrl_s, ctrl_x_radius)
        frames_q = self._interp_frames(ctrl_s, ctrl_frames, coords_query)

        return {
            "coords": coords_query,
            "points": points_q,
            "frame": frames_q,
            "radius": radius_q,
            "x_radius": x_radius_q,
            "control_coords": ctrl_s,
            "control_points": ctrl_points,
            "control_frame": ctrl_frames,
            "control_radius": ctrl_radius,
        }

    def localize_samples_adapt(self, vs, adapt_arg):
        """
        Direct accessory adaptation.

        Important distinction:
          - wrap_adapt_n_keypoints affects only the wrap radius table
            r_src/r_tgt used by wrap_radius.
          - use_adapt_control_field/adapt_control_n_keypoints affects the
            actual points, frames and radii used to compute samples_local.

        Clean inference order:
          1. localize query samples on avatar curve and crop to src interval
          2. optionally rebuild avatar from equally spaced control keypoints
          3. recompute avatar w/u/v/rho/theta from rebuilt avatar frames
          4. map avatar coords to accessory coords
          5. optionally rebuild accessory from equally spaced control keypoints
          6. apply wrap_radius / smooth_rigid_radius / rigid_radius / default
          7. pack accessory samples_local for model inference
        """

        accessory_curve_handle = adapt_arg["accessory_curve_handle"]
        accessory_curve_handle.core.update_coords()
        accessory_curve_handle.core.update_frame()

        src_0 = float(adapt_arg["src_0"])
        src_1 = float(adapt_arg["src_1"])
        tgt_0 = float(adapt_arg["tgt_0"])
        tgt_1 = float(adapt_arg["tgt_1"])
        delta_theta = np.deg2rad(float(adapt_arg.get("rot_deg", 0.0)))

        use_interval_projection = bool(adapt_arg.get("use_interval_projection", True))

        # ------------------------------------------------------------
        # 1) Localize grid/world samples on avatar/source curve.
        #    In adapt mode, restrict projection to src_0/src_1 by default.
        #    This prevents samples near a crossing/bend from snapping to a
        #    nearby wrong branch of the full avatar curve and being discarded
        #    by the later source-interval crop.
        # ------------------------------------------------------------
        avatar_data, inside = self.localize_samples(
            vs,
            norm=float(adapt_arg.get("adapt_localize_norm", 1.0)),
            outside=False,
            runtime_cylinder_radius_scale=float(adapt_arg.get("avatar_cylinder_radius_scale", 1.0)),
            runtime_cylinder_radius_add=float(adapt_arg.get("avatar_cylinder_radius_add", 0.0)),
            projection_s0=src_0 if use_interval_projection else None,
            projection_s1=src_1 if use_interval_projection else None,
            k_project=int(adapt_arg.get("loop_projection_candidates", 1)),
        )

        if adapt_arg.get("debug_interval_projection", False):
            coords_dbg = avatar_data.get("coords", np.asarray([], dtype=np.float64))
            if len(coords_dbg) > 0:
                print(
                    "[adapt interval projection]",
                    "enabled=", use_interval_projection,
                    "src=", src_0, src_1,
                    "coords min/max=",
                    float(np.min(coords_dbg)),
                    float(np.max(coords_dbg)),
                    "inside=", len(inside),
                )
            else:
                print(
                    "[adapt interval projection]",
                    "enabled=", use_interval_projection,
                    "src=", src_0, src_1,
                    "inside=", len(inside),
                    "no coords",
                )

        # ------------------------------------------------------------
        # 2) Crop to source interval
        # ------------------------------------------------------------
        avatar_coords = avatar_data["coords"]
        s_min = min(src_0, src_1)
        s_max = max(src_0, src_1)
        valid_map = (avatar_coords >= s_min) & (avatar_coords <= s_max)

        for k, v in avatar_data.items():
            if isinstance(v, np.ndarray) and v.shape[0] == valid_map.shape[0]:
                avatar_data[k] = v[valid_map]

        inside_final = inside.copy()[valid_map]

        if adapt_arg.get("adapt_debug_counts", True):
            print("input vs:", len(vs))
            print("after avatar localize inside:", len(inside))
            print("after src interval crop:", len(inside_final))

        avatar_coords = avatar_data["coords"]
        avatar_samples_world = avatar_data["samples"]

        # ------------------------------------------------------------
        # 3) Avatar points/frame/radius actually used for adaptation.
        # ------------------------------------------------------------
        use_adapt_control = bool(adapt_arg.get("use_adapt_control_field", False))

        if use_adapt_control:
            avatar_ctrl = self.build_adapt_control_field(
                avatar_coords,
                n_control=int(adapt_arg.get("adapt_control_n_keypoints", 12)),
                smooth_points_sigma=float(adapt_arg.get("avatar_control_smooth_points_sigma", 1.0)),
                smooth_radius_sigma=float(adapt_arg.get("avatar_control_smooth_radius_sigma", 1.0)),
                rebuild_frames=bool(adapt_arg.get("avatar_control_rebuild_frames", True)),
                preserve_endpoints=bool(adapt_arg.get("adapt_control_preserve_endpoints", True)),
                radius_type="train",
            )
            avatar_world_points = avatar_ctrl["points"]
            avatar_world_frames = avatar_ctrl["frame"]
            avatar_radius = avatar_ctrl["radius"]
            tangent_avatar = avatar_ctrl["x_radius"]


        else:
            # Legacy path: use the dense interpolated avatar curve/frame.
            avatar_world_points = self.interpolate(
                avatar_coords, radius=False, frame=False
            )["points"]
            avatar_world_frames = self.interpolate(
                avatar_coords, points=False, radius=False
            )["frame"]
            if adapt_arg.get("use_adapt_train_radius", False):
                avatar_radius = self.interpolate_radius_field(
                    avatar_coords, self.key_train_radius_adapt
                )
            else:
                avatar_radius = avatar_data["radius"]
            tangent_avatar = self.calc_x_radius(avatar_coords)



        # Recompute physical avatar local coords from the chosen avatar frame.
        # This is the part that fixes frame/theta staircase: theta below now
        # comes from the rebuilt 12-control frame when use_adapt_control_field=True.
        local_avatar = np.einsum(
            "nij,nj->ni",
            avatar_world_frames,
            avatar_samples_world - avatar_world_points,
        )
        w_avatar = local_avatar[:, 0]
        u_avatar = local_avatar[:, 1]
        v_avatar = local_avatar[:, 2]

        avatar_radius_y = avatar_radius[:, 0]
        avatar_radius_z = avatar_radius[:, 1]
        vx_avatar = 2.0 * avatar_coords - 1.0
        u_n_avatar = u_avatar / (avatar_radius_y + 1e-12)
        v_n_avatar = v_avatar / (avatar_radius_z + 1e-12)

        avatar_data["coords"] = avatar_coords
        avatar_data["samples_local"] = np.stack(
            [w_avatar / (tangent_avatar + 1e-12) + vx_avatar, u_n_avatar, v_n_avatar],
            axis=1,
        )
        avatar_data["radius"] = avatar_radius
        avatar_data["rho"] = np.sqrt(u_avatar * u_avatar + v_avatar * v_avatar)
        avatar_data["rho_n"] = np.sqrt(u_n_avatar * u_n_avatar + v_n_avatar * v_n_avatar)
        avatar_data["angles"] = np.arctan2(v_n_avatar, u_n_avatar)

        # ------------------------------------------------------------
        # 4) Map avatar/source coords to accessory/target coords
        # ------------------------------------------------------------
        acc_coords = self.map_coords_to_by_arclen(
            avatar_coords, accessory_curve_handle.core, src_0, src_1, tgt_0, tgt_1
        )


        if adapt_arg.get("debug_export_adapt_control_rings", False):
            self.export_adapt_control_rings_ply(
                "DEBUG_avatar_TRAIN_control_rings.ply",
                avatar_coords,
                n_control=int(adapt_arg.get("adapt_control_n_keypoints", 6)),
                smooth_points_sigma=float(adapt_arg.get("avatar_control_smooth_points_sigma", 2.0)),
                smooth_radius_sigma=float(adapt_arg.get("avatar_control_smooth_radius_sigma", 2.0)),
                rebuild_frames=bool(adapt_arg.get("avatar_control_rebuild_frames", True)),
                preserve_endpoints=bool(adapt_arg.get("adapt_control_preserve_endpoints", True)),
                radius_type="train",
                stride=int(adapt_arg.get("debug_ring_stride", 2)),
            )

            self.export_adapt_control_rings_ply(
                "DEBUG_avatar_CYLINDER_control_rings.ply",
                avatar_coords,
                n_control=int(adapt_arg.get("adapt_control_n_keypoints", 6)),
                smooth_points_sigma=float(adapt_arg.get("avatar_control_smooth_points_sigma", 2.0)),
                smooth_radius_sigma=float(adapt_arg.get("avatar_control_smooth_radius_sigma", 2.0)),
                rebuild_frames=bool(adapt_arg.get("avatar_control_rebuild_frames", True)),
                preserve_endpoints=bool(adapt_arg.get("adapt_control_preserve_endpoints", True)),
                radius_type="cylinder",
                stride=int(adapt_arg.get("debug_ring_stride", 2)),
            )
            accessory_curve_handle.core.export_adapt_control_rings_ply(
                "DEBUG_accessory_TRAIN_control_rings.ply",
                acc_coords,
                n_control=int(adapt_arg.get("adapt_control_n_keypoints", 6)),
                smooth_points_sigma=float(adapt_arg.get("accessory_control_smooth_points_sigma", 2.0)),
                smooth_radius_sigma=float(adapt_arg.get("accessory_control_smooth_radius_sigma", 2.0)),
                rebuild_frames=bool(adapt_arg.get("accessory_control_rebuild_frames", True)),
                preserve_endpoints=bool(adapt_arg.get("adapt_control_preserve_endpoints", True)),
                radius_type="train",
                stride=int(adapt_arg.get("debug_ring_stride", 2)),
            )

            accessory_curve_handle.core.export_adapt_control_rings_ply(
                "DEBUG_accessory_CYLINDER_control_rings.ply",
                acc_coords,
                n_control=int(adapt_arg.get("adapt_control_n_keypoints", 6)),
                smooth_points_sigma=float(adapt_arg.get("accessory_control_smooth_points_sigma", 2.0)),
                smooth_radius_sigma=float(adapt_arg.get("accessory_control_smooth_radius_sigma", 2.0)),
                rebuild_frames=bool(adapt_arg.get("accessory_control_rebuild_frames", True)),
                preserve_endpoints=bool(adapt_arg.get("adapt_control_preserve_endpoints", True)),
                radius_type="cylinder",
                stride=int(adapt_arg.get("debug_ring_stride", 2)),
            )


        # ------------------------------------------------------------
        # 5) Optional source/target UV-center correction.
        # This changes rho/theta before wrap lookup.
        # ------------------------------------------------------------
        use_src_center = bool(
            adapt_arg.get("use_src_runtime_uv_center", adapt_arg.get("use_runtime_uv_center", False))
        )
        use_tgt_center = bool(adapt_arg.get("use_tgt_runtime_uv_center", False))

        if use_src_center:
            src_center_field = adapt_arg.get("_src_uv_center_field", None)
            if src_center_field is None:
                src_center_field = self.build_runtime_uv_center_field(
                    n_bins=int(adapt_arg.get("uv_center_n_bins", 64)),
                    source=adapt_arg.get("uv_center_source", "owned"),
                    min_count=int(adapt_arg.get("uv_center_min_count", 20)),
                    smooth_s=float(adapt_arg.get("uv_center_smooth_s", 2.0)),
                    robust=adapt_arg.get("uv_center_robust", "median"),
                )
                adapt_arg["_src_uv_center_field"] = src_center_field
            src_center_uv = self.interpolate_uv_center_field(src_center_field, avatar_coords)
            cu_src = src_center_uv[:, 0]
            cv_src = src_center_uv[:, 1]
        else:
            cu_src = np.zeros_like(avatar_coords, dtype=np.float64)
            cv_src = np.zeros_like(avatar_coords, dtype=np.float64)

        u_avatar_c = u_avatar - cu_src
        v_avatar_c = v_avatar - cv_src
        rho_avatar = np.sqrt(u_avatar_c ** 2 + v_avatar_c ** 2)
        theta_avatar = np.arctan2(v_avatar_c, u_avatar_c)

        # ------------------------------------------------------------
        # 6) actually aradius  for adaptationed for model coords.
        # ------------------------------------------------------------
        wrap_n = adapt_arg.get("wrap_adapt_n_keypoints", None)
        n_lookup = int(adapt_arg.get("wrap_adapt_lookup_n", 128))

        if wrap_n is not None:
            wrap_src, s_bins_src = self.rebin_wrap_to_control_curve(
                self.key_wrap_radius_full,
                self.wrap_s_bins_full,
                self.wrap_theta_bins,
                n_control=int(adapt_arg.get("adapt_control_n_keypoints", 12)),
                smooth_points_sigma=float(adapt_arg.get("avatar_control_smooth_points_sigma", 8.0)),
                rebuild_frames=bool(adapt_arg.get("avatar_control_rebuild_frames", True)),
                n_lookup=int(adapt_arg.get("wrap_adapt_lookup_n", 128)),
                smooth_s=float(adapt_arg.get("wrap_adapt_smooth_s", 2.0)),
                smooth_theta=float(adapt_arg.get("wrap_adapt_smooth_theta", 2.0)),
            )
#            wrap_src, s_bins_src = self.smooth_downsample_wrap_for_adapt(
#                self.key_wrap_radius_full,
#                self.wrap_s_bins_full,
#                n_adapt=int(wrap_n),
#                smooth_s=float(adapt_arg.get("wrap_adapt_smooth_s", 4.0)),
#                smooth_theta=float(adapt_arg.get("wrap_adapt_smooth_theta", 2.0)),
#                n_lookup=n_lookup,
#            )

            wrap_tgt, s_bins_tgt = accessory_curve_handle.core.smooth_downsample_wrap_for_adapt(
                accessory_curve_handle.core.key_wrap_radius_full,
                accessory_curve_handle.core.wrap_s_bins_full,
                n_adapt=int(wrap_n),
                smooth_s=float(adapt_arg.get("wrap_adapt_smooth_s", 4.0)),
                smooth_theta=float(adapt_arg.get("wrap_adapt_smooth_theta", 2.0)),
                n_lookup=n_lookup,
            )
        else:
            wrap_src = self.key_wrap_radius_full
            s_bins_src = self.wrap_s_bins_full
            wrap_tgt = accessory_curve_handle.core.key_wrap_radius_full
            s_bins_tgt = accessory_curve_handle.core.wrap_s_bins_full

        if adapt_arg.get("debug_export_wrap_used", False):
            np.savez_compressed(
                "DEBUG_wrap_used_in_adapt.npz",
                wrap_n=int(wrap_n) if wrap_n is not None else -1,
                wrap_src=wrap_src,
                s_bins_src=s_bins_src,
                theta_bins_src=self.wrap_theta_bins,
                wrap_tgt=wrap_tgt,
                s_bins_tgt=s_bins_tgt,
                theta_bins_tgt=accessory_curve_handle.core.wrap_theta_bins,
            )

            print(
                "[DEBUG wrap used]",
                "wrap_n=", wrap_n,
                "src=", wrap_src.shape, "s_bins_src=", len(s_bins_src),
                "tgt=", wrap_tgt.shape, "s_bins_tgt=", len(s_bins_tgt),
                "src min/mean/max=", float(np.min(wrap_src)), float(np.mean(wrap_src)), float(np.max(wrap_src)),
                "tgt min/mean/max=", float(np.min(wrap_tgt)), float(np.mean(wrap_tgt)), float(np.max(wrap_tgt)),
            )
            self.export_wrap_field_ply(
                    wrap_src, s_bins_src, self.wrap_theta_bins,
                    "DEBUG_src_wrap_used.ply"
                )
            accessory_curve_handle.core.export_wrap_field_ply(
                    wrap_tgt, s_bins_tgt, accessory_curve_handle.core.wrap_theta_bins,
                    "DEBUG_tgt_wrap_used.ply"
                )
            self.export_wrap_field_ply_with_control(
                wrap_src,
                s_bins_src,
                self.wrap_theta_bins,
                "DEBUG_src_wrap_used_CONTROL.ply",
                n_control=int(adapt_arg.get("adapt_control_n_keypoints", 12)),
                smooth_points_sigma=float(adapt_arg.get("avatar_control_smooth_points_sigma", 1.0)),
                smooth_radius_sigma=float(adapt_arg.get("avatar_control_smooth_radius_sigma", 0.0)),
                rebuild_frames=bool(adapt_arg.get("avatar_control_rebuild_frames", True)),
                preserve_endpoints=bool(adapt_arg.get("adapt_control_preserve_endpoints", True)),
            )

            accessory_curve_handle.core.export_wrap_field_ply_with_control(
                wrap_tgt,
                s_bins_tgt,
                accessory_curve_handle.core.wrap_theta_bins,
                "DEBUG_tgt_wrap_used_CONTROL.ply",
                n_control=int(adapt_arg.get("adapt_control_n_keypoints", 12)),
                smooth_points_sigma=float(adapt_arg.get("accessory_control_smooth_points_sigma", 1.0)),
                smooth_radius_sigma=float(adapt_arg.get("accessory_control_smooth_radius_sigma", 0.0)),
                rebuild_frames=bool(adapt_arg.get("accessory_control_rebuild_frames", True)),
                preserve_endpoints=bool(adapt_arg.get("adapt_control_preserve_endpoints", True)),
            )






        # ------------------------------------------------------------
        # 6) Accessory points/frame/radius actually used for model coords.
        # ------------------------------------------------------------
        if use_adapt_control:
            acc_ctrl = accessory_curve_handle.core.build_adapt_control_field(
                acc_coords,
                n_control=int(adapt_arg.get("adapt_control_n_keypoints", 12)),
                smooth_points_sigma=float(adapt_arg.get("accessory_control_smooth_points_sigma", 1.0)),
                smooth_radius_sigma=float(adapt_arg.get("accessory_control_smooth_radius_sigma", 1.0)),
                rebuild_frames=bool(adapt_arg.get("accessory_control_rebuild_frames", True)),
                preserve_endpoints=bool(adapt_arg.get("adapt_control_preserve_endpoints", True)),
                radius_type="cylinder",
            )
            acc_intpl = {
                "points": acc_ctrl["points"],
                "frame": acc_ctrl["frame"],
                "radius": acc_ctrl["radius"],
            }
            tangent_acc = acc_ctrl["x_radius"]
        else:
            acc_intpl = accessory_curve_handle.core.interpolate(acc_coords)
            tangent_acc = accessory_curve_handle.core.calc_x_radius(acc_coords)

        if use_tgt_center:
            tgt_center_field = adapt_arg.get("_tgt_uv_center_field", None)
            if tgt_center_field is None:
                tgt_center_field = accessory_curve_handle.core.build_runtime_uv_center_field(
                    n_bins=int(adapt_arg.get("uv_center_n_bins", 64)),
                    source=adapt_arg.get("uv_center_source", "owned"),
                    min_count=int(adapt_arg.get("uv_center_min_count", 20)),
                    smooth_s=float(adapt_arg.get("uv_center_smooth_s", 2.0)),
                    robust=adapt_arg.get("uv_center_robust", "median"),
                )
                adapt_arg["_tgt_uv_center_field"] = tgt_center_field
            tgt_center_uv = accessory_curve_handle.core.interpolate_uv_center_field(tgt_center_field, acc_coords)
            cu_tgt = tgt_center_uv[:, 0]
            cv_tgt = tgt_center_uv[:, 1]
        else:
            cu_tgt = np.zeros_like(acc_coords, dtype=np.float64)
            cv_tgt = np.zeros_like(acc_coords, dtype=np.float64)

        F = acc_intpl["frame"]
        acc_radius = acc_intpl["radius"]
        acc_radius_y = acc_radius[:, 0]
        acc_radius_z = acc_radius[:, 1]

        scale_w = tangent_acc / (tangent_avatar + 1e-12)
        scale_y = acc_radius_y / (avatar_radius_y + 1e-12)
        scale_z = acc_radius_z / (avatar_radius_z + 1e-12)

        theta_tgt = theta_avatar + delta_theta

        # ------------------------------------------------------------
        # 9) Radial mapping
        # ------------------------------------------------------------
        global_scale = float(adapt_arg.get("scale", 1.0))

        if adapt_arg.get("wrap_radius", False):
            theta_src = theta_avatar
            theta_tgt = theta_avatar + delta_theta


            wrap_src = np.asarray(wrap_src, dtype=np.float64)
            wrap_tgt = np.asarray(wrap_tgt, dtype=np.float64)

            theta_bins_src = getattr(self, "wrap_theta_bins", None)
            if theta_bins_src is None or len(theta_bins_src) != wrap_src.shape[1]:
                theta_bins_src = np.linspace(-np.pi, np.pi, wrap_src.shape[1], endpoint=False)
            else:
                theta_bins_src = np.asarray(theta_bins_src, dtype=np.float64)

            theta_bins_tgt = getattr(accessory_curve_handle.core, "wrap_theta_bins", None)
            if theta_bins_tgt is None or len(theta_bins_tgt) != wrap_tgt.shape[1]:
                theta_bins_tgt = np.linspace(-np.pi, np.pi, wrap_tgt.shape[1], endpoint=False)
            else:
                theta_bins_tgt = np.asarray(theta_bins_tgt, dtype=np.float64)

            r_src = interpolate_wrap_radius1(
                self,
                avatar_coords,
                theta_src,
                wrap_src,
                theta_bins_src,
                s_bins_src,
            )

            r_tgt = interpolate_wrap_radius1(
                accessory_curve_handle.core,
                acc_coords,
                theta_tgt,
                wrap_tgt,
                theta_bins_tgt,
                s_bins_tgt,
            )

            skirt_env_scale = None




            snug_field = adapt_arg.get("avatar_snug_scale_field", None)
            # snug_mode controls how the snug field is applied:
            #   "multiplicative" (default, legacy): r_src *= scale(s, theta).
            #   "additive": skip radial dilation; the additive delta is applied
            #               later in agent_3dvec.action_part_adapt directly to
            #               the FINAL SDF. This avoids volumetric inflation of
            #               the wrap and is two-sided.
            #   "off": don't apply anything here (e.g. when only the additive
            #          delta is wanted but you still passed the field).
            snug_mode = str(adapt_arg.get("snug_mode", "multiplicative")).lower()

            if snug_field is not None and snug_mode == "multiplicative":
                snug_scale = self.interpolate_snug_scale_field(
                    snug_field,
                    avatar_coords,
                    theta_src,
                )

                if adapt_arg.get("snug_debug", False):
                    print(
                        "[snug_scale_apply]",
                        "min/mean/max=",
                        float(np.min(snug_scale)),
                        float(np.mean(snug_scale)),
                        float(np.max(snug_scale)),
                    )

                # Keep this OFF for now. It is broad and can undo snug.
                if adapt_arg.get("snug_penetration_guard", False):
                    min_clearance = float(adapt_arg.get("snug_penetration_clearance", 0.001))
                    pushout_scale = float(adapt_arg.get("snug_pushout_scale", 1.02))

                    bad = rho_avatar <= (r_src + min_clearance)
                    snug_scale[bad] = np.maximum(snug_scale[bad], pushout_scale)

                    if adapt_arg.get("snug_debug", False):
                        print(
                            "[snug_guard]",
                            "bad=",
                            int(np.sum(bad)),
                            "/",
                            int(len(bad)),
                            "scale min/mean/max=",
                            float(np.min(snug_scale)),
                            float(np.mean(snug_scale)),
                            float(np.max(snug_scale)),
                        )

                r_src = r_src * snug_scale
            elif snug_field is not None and adapt_arg.get("snug_debug", False):
                print(
                    "[snug_scale_apply] skipped multiplicative scale "
                    f"(snug_mode={snug_mode}); additive delta will be "
                    "applied at the final SDF."
                )


            scale_rho_wrap = (global_scale * r_tgt) / (r_src + 1e-12)

            #if skirt_env_scale is not None:
            #    # Keep original wrap/detail variation, only enlarge smoothly by envelope.
            #    if adapt_arg.get("skirt_env_replace_wrap_scale", True):
            #        scale_rho_wrap = global_scale * skirt_env_scale
            #    else:
            #        scale_rho_wrap = scale_rho_wrap * skirt_env_scale

            if "wrap_scale_min" in adapt_arg or "wrap_scale_max" in adapt_arg:
                scale_rho_wrap = np.clip(
                    scale_rho_wrap,
                    float(adapt_arg.get("wrap_scale_min", 0.25)),
                    float(adapt_arg.get("wrap_scale_max", 4.0)),
                )

            rho_acc_wrap = rho_avatar * scale_rho_wrap

            # ------------------------------------------------------------
            # Optional blend from wrap-radius mapping to rigid/default mapping.
            #
            # Useful for garments:
            #   top seam / waistband -> wrap
            #   lower flowing part   -> rigid/default
            # ------------------------------------------------------------

            # ------------------------------------------------------------
            # Generic garment attach-to-free blend.
            #
            # attach region:
            #   follows avatar/body via wrap mapping
            #
            # free region:
            #   follows garment's own source/ridig shape, optionally with profile flare
            #
            # This works for skirt, shirt hem, sleeves, pants, cape, etc.
            # ------------------------------------------------------------
            if adapt_arg.get(
                "use_attach_to_free_blend",
                adapt_arg.get("use_wrap_to_rigid_blend", False),  # backward compatible
            ):

                def _smoothstep01(x):
                    x = np.clip(x, 0.0, 1.0)
                    return x * x * (3.0 - 2.0 * x)


                blend_coord_type = adapt_arg.get(
                    "attach_blend_coord",
                    adapt_arg.get("wrap_blend_coord", "accessory"),
                )

                if blend_coord_type == "avatar":
                    blend_coord = avatar_coords
                else:
                    blend_coord = acc_coords

                blend_coord_space = adapt_arg.get("attach_coord_space", "relative")

                if blend_coord_space == "relative":
                    if blend_coord_type == "avatar":
                        b0 = src_0
                        b1 = src_1
                    else:
                        b0 = tgt_0
                        b1 = tgt_1

                    blend_coord_used = (blend_coord - b0) / (b1 - b0 + 1e-12)
                    blend_coord_used = np.clip(blend_coord_used, 0.0, 1.0)
                else:
                    blend_coord_used = blend_coord

                attach_s0 = float(
                    adapt_arg.get(
                        "attach_s0",
                        adapt_arg.get("wrap_to_rigid_s0", 0.0),
                    )
                )
                attach_s1 = float(
                    adapt_arg.get(
                        "attach_s1",
                        adapt_arg.get("wrap_to_rigid_s1", 0.25),
                    )
                )

                tau = (blend_coord_used - attach_s0) / (attach_s1 - attach_s0 + 1e-12)
                fade = _smoothstep01(tau)

                weight_attach = 1.0 - fade
                weight_attach = np.clip(weight_attach, 0.0, 1.0)

                free_mode = adapt_arg.get(
                    "free_mode",
                    adapt_arg.get("wrap_blend_rigid_mode", "rigid"),
                )

                # This is now WORLD meaning:
                # free_scale > 1.0 => larger garment
                # free_scale < 1.0 => smaller garment
                free_world_scale = float(
                    adapt_arg.get(
                        "free_scale",
                        adapt_arg.get("rigid_blend_scale", global_scale),
                    )
                )

                use_free_profile_scale = bool(
                    adapt_arg.get(
                        "use_free_profile_scale",
                        adapt_arg.get("rigid_use_flare_profile", False),
                    )
                )

                if use_free_profile_scale:
                    profile_coord_space = adapt_arg.get("free_profile_coord_space", "relative")

                    if profile_coord_space == "relative":
                        profile_coord_used = (acc_coords - tgt_0) / (tgt_1 - tgt_0 + 1e-12)
                        profile_coord_used = np.clip(profile_coord_used, 0.0, 1.0)
                    else:
                        profile_coord_used = acc_coords

                    profile_start = float(
                        adapt_arg.get(
                            "free_profile_start",
                            adapt_arg.get("rigid_flare_start", 0.20),
                        )
                    )
                    profile_end = float(
                        adapt_arg.get(
                            "free_profile_end",
                            adapt_arg.get("rigid_flare_end", 1.00),
                        )
                    )
                    # Optional guard: do not let flare start inside the attach-release band.
                    if (
                        adapt_arg.get("attach_coord_space", "relative") == "relative"
                        and adapt_arg.get("free_profile_coord_space", "relative") == "relative"
                    ):
                        flare_attach_gap = float(adapt_arg.get("free_profile_attach_gap", 0.08))
                        profile_start = max(profile_start, attach_s1 + flare_attach_gap)
                        profile_end = max(profile_end, profile_start + 1e-6)


                    profile_gain = float(
                        adapt_arg.get(
                            "free_profile_gain",
                            adapt_arg.get("rigid_flare_gain", 0.0),
                        )
                    )


                    #profile_tau = (profile_coord_used - profile_start) / (
                    #    profile_end - profile_start + 1e-12
                    #)
                    #profile_w = _smoothstep01(profile_tau)
                    profile_type = str(adapt_arg.get("free_profile_type", "ramp")).lower()

                    if profile_type == "ramp":
                        profile_tau = (profile_coord_used - profile_start) / (
                            profile_end - profile_start + 1e-12
                        )
                        profile_w = _smoothstep01(profile_tau)

                    elif profile_type == "bump":
                        profile_peak = float(
                            adapt_arg.get(
                                "free_profile_peak",
                                0.5 * (profile_start + profile_end),
                            )
                        )
                        profile_peak = np.clip(
                            profile_peak,
                            profile_start + 1e-6,
                            profile_end - 1e-6,
                        )

                        t_up = (profile_coord_used - profile_start) / (
                            profile_peak - profile_start + 1e-12
                        )
                        t_down = (profile_end - profile_coord_used) / (
                            profile_end - profile_peak + 1e-12
                        )

                        up = _smoothstep01(t_up)
                        down = _smoothstep01(t_down)
                        profile_w = up * down

                    else:
                        raise ValueError(
                            f"Unknown free_profile_type={profile_type}. Use 'ramp' or 'bump'."
                        )

                    free_world_scale_s = free_world_scale * (1.0 + profile_gain * profile_w)
                else:
                    free_world_scale_s = np.full_like(rho_avatar, free_world_scale)

                # IMPORTANT:
                # We are mapping world query points into accessory-local query space.
                # To make world garment larger, local query radius must become smaller.
                local_free_scale_s = 1.0 / np.maximum(free_world_scale_s, 1e-8)

                if free_mode == "unified_wrap":
                    s_rel = blend_coord_used

                    release_band = float(adapt_arg.get("unified_release_band", 0.03))
                    m = np.abs(s_rel - attach_s1) <= release_band

                    if np.any(m):
                        r_release = float(np.median(r_tgt[m]))
                    else:
                        idx = int(np.argmin(np.abs(s_rel - attach_s1)))
                        r_release = float(r_tgt[idx])

                    flare_end = float(adapt_arg.get("unified_flare_end", 0.55))
                    flare_gain = float(adapt_arg.get("unified_flare_gain", 0.8))

                    t = (s_rel - attach_s1) / (flare_end - attach_s1 + 1e-12)
                    w = _smoothstep01(t)

                    # ------------------------------------------------------------
                    # Persistent umbrella target radius.
                    # The free region should NOT collapse after attach transition.
                    # ------------------------------------------------------------
                    r_umbrella = r_release * (1.0 + flare_gain * w)

                    hold_umbrella = bool(adapt_arg.get("unified_hold_umbrella", True))

                    if hold_umbrella:
                        # Keep the larger of original skirt radius and umbrella envelope.
                        # This preserves lower skirt volume while still allowing original skirt shape.
                        r_free_target = np.maximum(r_tgt, r_umbrella)
                    else:
                        flow_start = float(adapt_arg.get("unified_flow_start", flare_end))
                        flow_end = float(adapt_arg.get("unified_flow_end", 1.0))
                        tf = (s_rel - flow_start) / (flow_end - flow_start + 1e-12)
                        wf = _smoothstep01(tf)

                        # Old behavior: can flow back/collapse toward original r_tgt.
                        r_free_target = (1.0 - wf) * r_umbrella + wf * r_tgt

                    r_free_target = np.maximum(r_free_target, 1e-6)
                    r_tgt_safe = np.maximum(r_tgt, 1e-6)

                    scale_rho_free = global_scale * r_tgt_safe / r_free_target
                    rho_acc_free = rho_avatar * scale_rho_free

                    #scale_rho_free = global_scale * r_free_target / (r_src + 1e-12)
                    #rho_acc_free = rho_avatar * scale_rho_free

                    rho_acc = weight_attach * rho_acc_wrap + (1.0 - weight_attach) * rho_acc_free
                    print("[unified]",
                          "r_src", np.min(r_src), np.mean(r_src), np.max(r_src),
                          "r_tgt", np.min(r_tgt), np.mean(r_tgt), np.max(r_tgt),
                          "r_release", r_release,
                          "r_umbrella", np.min(r_umbrella), np.mean(r_umbrella), np.max(r_umbrella),
                          "scale_free", np.min(scale_rho_free), np.mean(scale_rho_free), np.max(scale_rho_free))
                else:
                    if free_mode == "default":
                        scale_rho_free_raw = local_free_scale_s * 0.5 * (scale_y + scale_z)
                    else:
                        scale_rho_free_raw = local_free_scale_s


                    # ------------------------------------------------------------
                    # Optional release anchoring:
                    # Make free_scale/profile relative to the wrap scale at attach_s1.
                    #
                    # Meaning:
                    #   free_scale = 1.0 at profile_w=0
                    #   => start from the same effective radius as wrap at release.
                    #
                    # This changes the target free branch itself.
                    # free_match_wrap_at_release below then smooths into this target.
                    # ------------------------------------------------------------
                    if adapt_arg.get("free_anchor_to_release_scale", False):
                        release_band = float(adapt_arg.get("free_release_sample_band", 0.03))

                        release_mask = np.abs(blend_coord_used - attach_s1) <= release_band

                        if np.any(release_mask):
                            wrap_anchor_scale = float(np.median(scale_rho_wrap[release_mask]))
                            free_anchor_scale = float(np.median(scale_rho_free_raw[release_mask]))
                        else:
                            idx = int(np.argmin(np.abs(blend_coord_used - attach_s1)))
                            wrap_anchor_scale = float(scale_rho_wrap[idx])
                            free_anchor_scale = float(scale_rho_free_raw[idx])

                        # Normalize raw free branch so that at attach_s1 it equals wrap scale.
                        scale_rho_free_raw = scale_rho_free_raw * (
                            wrap_anchor_scale / (free_anchor_scale + 1e-8)
                        )

                        if adapt_arg.get("attach_blend_debug", False):
                            print(
                                "[free_anchor_to_release]",
                                "release_band=", release_band,
                                "wrap_anchor_scale=", wrap_anchor_scale,
                                "free_anchor_scale=", free_anchor_scale,
                                "scale_free_raw anchored min/mean/max=",
                                float(np.min(scale_rho_free_raw)),
                                float(np.mean(scale_rho_free_raw)),
                                float(np.max(scale_rho_free_raw)),
                            )


                    # ------------------------------------------------------------
                    # Optional continuity correction:
                    # Make the free branch start from the wrap branch instead of
                    # jumping immediately to independent free scale.
                    # ------------------------------------------------------------
                    if adapt_arg.get("free_match_wrap_at_release", False):
                        print("in free match")
                        free_band = float(adapt_arg.get("free_match_wrap_band", 0.20))

                        t_free = (blend_coord_used - attach_s1) / (free_band + 1e-12)
                        w_free = _smoothstep01(t_free)

                        scale_rho_free = (
                            (1.0 - w_free) * scale_rho_wrap
                            + w_free * scale_rho_free_raw
                        )
                    else:
                        scale_rho_free = scale_rho_free_raw

                    rho_acc_free = rho_avatar * scale_rho_free

                rho_acc = (
                    weight_attach * rho_acc_wrap
                    + (1.0 - weight_attach) * rho_acc_free
                )

                if adapt_arg.get(
                    "attach_blend_debug",
                    adapt_arg.get("wrap_blend_debug", False),
                ):
                    print(
                        "[attach_to_free_blend]",
                        "coord=", blend_coord_type,
                        "attach_s0/s1=", attach_s0, attach_s1,
                        "free_mode=", free_mode,
                        "free_world_scale_world=", free_world_scale,
                        "weight_attach min/mean/max=",
                        float(np.min(weight_attach)),
                        float(np.mean(weight_attach)),
                        float(np.max(weight_attach)),
                        "rho_attach mean=",
                        float(np.mean(rho_acc_wrap)),
                        "rho_free mean=",
                        float(np.mean(rho_acc_free)),
                        "rho_final mean=",
                        float(np.mean(rho_acc)),
                    )

            else:
                rho_acc = rho_acc_wrap


            if adapt_arg.get("wrap_debug", False):
                q_src = rho_avatar / (r_src + 1e-12)
                print(
                    "[wrap_debug]",
                    "r_src min/mean/max=",
                    float(np.min(r_src)),
                    float(np.mean(r_src)),
                    float(np.max(r_src)),
                    "r_tgt min/mean/max=",
                    float(np.min(r_tgt)),
                    float(np.mean(r_tgt)),
                    float(np.max(r_tgt)),
                    "q_src min/mean/max=",
                    float(np.min(q_src)),
                    float(np.mean(q_src)),
                    float(np.max(q_src)),
                    "scale_rho min/mean/max=",
                    float(np.min(scale_rho_wrap)),
                    float(np.mean(scale_rho_wrap)),
                    float(np.max(scale_rho_wrap)),
                )

        elif adapt_arg.get("rigid_radius", False):
            scale_rho = np.full_like(rho_avatar, global_scale)
            #rigid_radial_scale = float(adapt_arg.get("rigid_radial_scale", 5.0))
            #scale_rho = np.full_like(rho_avatar, rigid_radial_scale)
            rho_acc = rho_avatar * scale_rho
            theta_tgt = theta_avatar + delta_theta
            if adapt_arg.get("debug_rigid_coords", False):
                print(
                    "[rigid coords]",
                    "rho_avatar min/mean/max=",
                    float(np.min(rho_avatar)),
                    float(np.mean(rho_avatar)),
                    float(np.max(rho_avatar)),
                    "rho_acc min/mean/max=",
                    float(np.min(rho_acc)),
                    float(np.mean(rho_acc)),
                    float(np.max(rho_acc)),
                    "acc_radius min/mean/max=",
                    float(np.min(acc_radius)),
                    float(np.mean(acc_radius)),
                    float(np.max(acc_radius)),
                    "global_scale=",
                    float(global_scale),
                )
            if adapt_arg.get("rigid_radius_rho_cap", False):
                max_rho_n = float(adapt_arg.get("rigid_radius_max_rho_n", 1.35))

                acc_radius_theta = np.sqrt(
                    (acc_radius_y * np.cos(theta_tgt)) ** 2
                    + (acc_radius_z * np.sin(theta_tgt)) ** 2
                )

                rho_cap = max_rho_n * acc_radius_theta
                rho_acc = np.minimum(rho_acc, rho_cap)

                if adapt_arg.get("rigid_radius_cap_debug", False):
                    print(
                        "[rigid rho cap]",
                        "max_rho_n=", max_rho_n,
                        "rho_acc max=", float(np.max(rho_acc)),
                        "rho_cap min/mean/max=",
                        float(np.min(rho_cap)),
                        float(np.mean(rho_cap)),
                        float(np.max(rho_cap)),
                    )

        else:
            # Fallback non-wrap mapping. Use centered rho/theta and average radius scale.
            scale_rho = global_scale * 0.5 * (scale_y + scale_z)
            rho_acc = rho_avatar * scale_rho
            theta_tgt = theta_avatar + delta_theta

        # ------------------------------------------------------------
        # 10) Build accessory physical local coordinates
        # ------------------------------------------------------------
        u_acc = cu_tgt + rho_acc * np.cos(theta_tgt)
        v_acc = cv_tgt + rho_acc * np.sin(theta_tgt)
        w_acc = w_avatar * scale_w


        # ------------------------------------------------------------
        # 11) Normalize accessory local coords for model input
        # ------------------------------------------------------------
        #tloc = np.array([0.0, -0.15, 0.0], dtype=np.float64)

        # Local translation in accessory frame [T, N, B]
        translate_local = adapt_arg.get("tloc", adapt_arg.get("translate_local", None))
        if translate_local is not None:
            tloc = np.asarray(translate_local, dtype=np.float64).reshape(3)

            # Moving object by +tloc means query coords shift by -tloc
            w_acc = w_acc - tloc[0]
            u_acc = u_acc - tloc[1]
            v_acc = v_acc - tloc[2]


        def _apply_one_sided_tilt1(local_coord, tilt_deg, anchor_s):
            tilt_rad = np.deg2rad(float(tilt_deg))

            curve_len, _ = accessory_curve_handle.core.calc_curve_length()

            # distance from anchor along accessory s
            d = (acc_coords - float(anchor_s)) * curve_len

            # tilt side is the side from anchor toward tgt_0
            side = np.sign(float(tgt_0) - anchor_s)

            # If anchor is tgt_0, affect toward tgt_1.
            # If anchor is tgt_1, affect toward tgt_0.
#            if tgt_0 < anchor_s: 
#                side = np.sign(float(tgt_0) - float(tgt_1)) 
#                if abs(side) < 1e-12: 
#                    side = 1.0 
#            else: 
#                side = np.sign(float(tgt_1) - float(tgt_0)) 
#                if abs(side) < 1e-12: 
#                    side = 1.0
            # anchor is tgt_0 -> tilt toward tgt_1
            # anchor is tgt_1 -> tilt toward tgt_0
            if abs(anchor_s - float(tgt_0)) <= abs(anchor_s - float(tgt_1)):
                side = np.sign(float(tgt_1) - float(tgt_0))
                if abs(side) < 1e-12: 
                    side = 1.0
            else:
                side = np.sign(float(tgt_0) - float(tgt_1))
                if abs(side) < 1e-12: 
                    side = 1.0

            forward = np.maximum(side * d, 0.0)

            # Moving object up by +tilt means query coord shifts by -
            return local_coord - np.tan(tilt_rad) * forward

        def _apply_one_sided_tilt(local_coord, tilt_deg, anchor_s, end_s):
            tilt_rad = np.deg2rad(float(tilt_deg))

            curve_len, _ = accessory_curve_handle.core.calc_curve_length()
            anchor_s = float(anchor_s)
            end_s = float(end_s)
            print("anchor_s ", anchor_s)
            print("anchor_s ", end_s)

            direction = np.sign(end_s - anchor_s)
            print("direction = ", direction, flush=True)
            if abs(direction) < 1e-12:
                return local_coord

            d = (acc_coords - anchor_s) * curve_len
            forward = np.maximum(direction * d, 0.0)

            return local_coord - np.tan(tilt_rad) * forward


        # Tilt in local U direction
        tilt_u = adapt_arg.get("tilt_u", None)
        if tilt_u is not None:
            print("tilt u")
            tilt_u_anchor = adapt_arg.get("tilt_u_anchor", tgt_1)
            tilt_u_end = adapt_arg.get("tilt_u_end", tgt_0)
            print(u_acc)
            u_acc = _apply_one_sided_tilt(u_acc, tilt_u, tilt_u_anchor, tilt_u_end)
            print(u_acc)


        # Tilt in local V direction
        tilt_v = adapt_arg.get("tilt_v", None)
        if tilt_v is not None:
            tilt_v_anchor = adapt_arg.get("tilt_v_anchor", tgt_1)
            tilt_v_end = adapt_arg.get("tilt_v_end", tgt_0)
            #print(v_acc)
            v_acc = _apply_one_sided_tilt(v_acc, tilt_v, tilt_v_anchor, tilt_v_end)
            #print(v_acc, flush=True)



        w_n_acc = w_acc / (tangent_acc + 1e-12)
        u_n_acc = u_acc / (acc_radius_y + 1e-12)
        v_n_acc = v_acc / (acc_radius_z + 1e-12)

        vx_acc = 2.0 * acc_coords - 1.0

        samples_local_acc = np.stack(
            [w_n_acc + vx_acc , u_n_acc, v_n_acc],
            axis=1,
        )
        rho_n_acc = np.sqrt(u_n_acc ** 2 + v_n_acc ** 2)
        angles_acc = np.arctan2(v_n_acc, u_n_acc)

        if adapt_arg.get("debug_acc_local", False):
            print(
                "[acc local]",
                "acc_coords min/mean/max=",
                float(np.min(acc_coords)),
                float(np.mean(acc_coords)),
                float(np.max(acc_coords)),
                "w_n min/mean/max=",
                float(np.min(w_n_acc)),
                float(np.mean(w_n_acc)),
                float(np.max(w_n_acc)),
                "u_n min/mean/max=",
                float(np.min(u_n_acc)),
                float(np.mean(u_n_acc)),
                float(np.max(u_n_acc)),
                "v_n min/mean/max=",
                float(np.min(v_n_acc)),
                float(np.mean(v_n_acc)),
                float(np.max(v_n_acc)),
                "rho_n min/mean/max=",
                float(np.min(rho_n_acc)),
                float(np.mean(rho_n_acc)),
                float(np.max(rho_n_acc)),
            )
            # Use accessory normalized local coordinates to split directions.
            # Depending on your convention, front may be +v, -v, +u, or -u.
            sectors = {
                "u_pos": u_n_acc > 0,
                "u_neg": u_n_acc < 0,
                "v_pos": v_n_acc > 0,
                "v_neg": v_n_acc < 0,
            }

            for name, m in sectors.items():
                if np.any(m):
                    print(
                        "[front/back rho]",
                        name,
                        "count=", int(np.sum(m)),
                        "rho_n min/mean/max=",
                        float(np.min(rho_n_acc[m])),
                        float(np.mean(rho_n_acc[m])),
                        float(np.max(rho_n_acc[m])),
                        "u_n mean=", float(np.mean(u_n_acc[m])),
                        "v_n mean=", float(np.mean(v_n_acc[m])),
                    )


        # ------------------------------------------------------------
        # 12) Return data dicts
        # ------------------------------------------------------------
        # These are the ONLY fields used by adapt inference/model input.
        # No runtime_support object is returned; the model consumes samples_local.
        accessory_data = dict(avatar_data)
        accessory_data["coords"] = acc_coords
        accessory_data["samples_local"] = samples_local_acc
        accessory_data["angles"] = angles_acc
        accessory_data["rho_n"] = rho_n_acc
        accessory_data["rho"] = rho_acc
        accessory_data["radius"] = acc_radius
        accessory_data["frame"] = acc_intpl["frame"]
        accessory_data["points"] = acc_intpl["points"]
        accessory_data["x_radius"] = tangent_acc

        return accessory_data, avatar_data, inside_final

    def localize_occ_samples(self, samples):
        ts = self.curve_projection(samples, outside=True)

        intpl = self.interpolate(ts)
        proj_vs = intpl['points']
        frame_mat = intpl['frame']

        # frame: (N, 3,3), vs (N, 3)
        samples_local = np.einsum('nij,nj->ni', frame_mat, (samples - proj_vs))

        vx = 2*ts - 1
        samples_local[:, 0] += vx
        return {
            'samples_local': samples_local,
            'coords': ts
        }
