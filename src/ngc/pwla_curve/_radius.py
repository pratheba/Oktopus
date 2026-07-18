# Auto-split from PWLA_curve_handle.py -- _RadiusMixin
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



class _RadiusMixin:


    def smooth_resample_radius_for_adapt(self,radius, n_control=64, smooth_s=1.0, floor_ratio=0.85):
        radius = np.asarray(radius, dtype=np.float64)
        K = radius.shape[0]

        if K <= 2:
            return radius.copy()

        s_full = np.linspace(0.0, 1.0, K)

        n_control = int(min(max(2, n_control), K))
        s_ctrl = np.linspace(0.0, 1.0, n_control)

        ctrl = np.zeros((n_control, 2), dtype=np.float64)
        ctrl[:, 0] = np.interp(s_ctrl, s_full, radius[:, 0])
        ctrl[:, 1] = np.interp(s_ctrl, s_full, radius[:, 1])

        if smooth_s > 0:
            ctrl = gaussian_filter1d(ctrl, sigma=smooth_s, axis=0, mode="nearest")

        out = np.zeros_like(radius)
        out[:, 0] = np.interp(s_full, s_ctrl, ctrl[:, 0])
        out[:, 1] = np.interp(s_full, s_ctrl, ctrl[:, 1])

        # Do not let adaptation radius shrink too aggressively.
        out = np.maximum(out, floor_ratio * radius)

        return out

    def smooth_radius(self, radius_y, radius_z, gaussian_smooth=2.0, radius_type='train'):
        if gaussian_smooth > 0:
            radius_y = gaussian_filter1d(radius_y, sigma=gaussian_smooth)
            radius_z = gaussian_filter1d(radius_z, sigma=gaussian_smooth)

        radius_yz = np.stack([radius_y, radius_z], axis=1)
        #print("radius_yz",radius_yz)
        self.update_radius(bin_center, radius_yz, radius_type)

    def update_radius(self, bins, radius_yz, radius_type='train'):
        #print(self.key_radius)
        ry = np.interp(self.key_ts, bins, radius_yz[:, 0])
        rz = np.interp(self.key_ts, bins, radius_yz[:, 1])
        radius = np.stack([ry, rz], axis=1)
        if radius_type == 'train':
            self.key_train_radius = radius
            self.key_radius = radius 
        elif radius_type == 'cylinder':
            self.key_cylinder_radius = radius 
        #print(self.key_radius)
        #print("*********")

    def update_radius_from_surfacepoints(self, points, n_bins=24, quantile=0.98, gaussian_smooth=1.0, radius_type='train'):
        surface_points = np.array(surface_points, dtype=np.float64)
        coord_points = self.curve_projection(surface_points)
        valid_coord_index = np.logical_and(coord_points >= 0.0 and coord_points <= 1.0)

        surface_points = surface_points[valid_coord_index]
        coord_points = coord_points[valid_coord_index]

        intpl = self.interpolate(coord_points)
        coord_key_points_3D = intpl['points']
        frame = intpl['frame']

        samples_local = np.einsum('nij, nj -> ni', frame, (surface_points - coord_key_points_3D))
        u = samples_local[:,1]
        v = samples_local[:,2]

        bin_edge = np.linspace(0,0, 1.0, n_bins+1)
        bin_center = 0.5* (bin_edge[:-1]+bin_edge[1:])
        bin_ids = np.clip(np.digitize(coord_points, bin_edge) -1, 0, n_bins-1)
        
        radius_y = np.full(n_bins, np.nan, dtype=np.float64)
        radius_z = np.full(n_bins, np.nan, dtype=np.float64)

        sample_count = np.zeros(n_bins, dtype=np.int32)        


        for b in range(n_bins):
            coords_in_b = (bin_ids == b)
            sample_count[b] = np.sum(coords_in_b)
            if sample_count[b] < min_count:
                continue
        
            abs_u = np.abs(u[coords_in_b])
            abs_v = np.abs(v[coords_in_b])
            
            radius_y[b] = np.quantile(abs_u, quantile)
            radius_z[b] = np.quantile(abs_v, quantile)

        valid_bins = np.isfinite(radius_y) & np.isfinite(radius_z)

        radius_y = fill_invalid_bins(radius_y, valid_bins)
        radius_z = fill_invalid_bins(radius_z, valid_bins)

        if gaussian_smooth > 0:
            radius_y = gaussian_filter1d(radius_y, sigma=gaussian_smooth)
            radius_z = gaussian_filter1d(radius_z, sigma=gaussian_smooth)

        radius_yz = np.stack([radius_y, radius_z], axis=1)
        #print("radius_yz",radius_yz)
        self.update_radius(bin_center, radius_yz, radius_type)
        
        return {"u": u,
                "v": v,
                "radius": radius_yz }

    def update_radius_from_coords(self, coord_points, w, u, v, n_bins=24, quantile=0.98, gaussian_smooth=2.0, min_count=30, radius_type='train'):
        bin_edge = np.linspace(0.0, 1.0, n_bins+1)
        bin_center = 0.5* (bin_edge[:-1]+bin_edge[1:])
        bin_ids = np.clip(np.digitize(coord_points, bin_edge) -1, 0, n_bins-1)
        
        radius_y = np.full(n_bins, np.nan, dtype=np.float64)
        radius_z = np.full(n_bins, np.nan, dtype=np.float64)

        sample_count = np.zeros(n_bins, dtype=np.int32)        
        slab_half_width = 1.0 / n_bins

        for b in range(n_bins):
            coords_in_b = (bin_ids == b) & (np.abs(w) <= slab_half_width)
            sample_count[b] = np.sum(coords_in_b)
            if sample_count[b] < min_count:
                continue
        
            abs_u = np.abs(u[coords_in_b])
            abs_v = np.abs(v[coords_in_b])
            
            #radius_y[b] = np.max(abs_u) #np.quantile(abs_u, quantile)
            #radius_z[b] = np.max(abs_v) #np.quantile(abs_v, quantile)
            radius_y[b] = np.quantile(abs_u, quantile)
            radius_z[b] = np.quantile(abs_v, quantile)

        valid_bins = np.isfinite(radius_y) & np.isfinite(radius_z)

        radius_y = fill_invalid_bins(radius_y, valid_bins)
        radius_z = fill_invalid_bins(radius_z, valid_bins)

        if gaussian_smooth > 0:
            radius_y = gaussian_filter1d(radius_y, sigma=gaussian_smooth)
            radius_z = gaussian_filter1d(radius_z, sigma=gaussian_smooth)

        radius_yz = np.stack([radius_y, radius_z], axis=1)
        #print("radius_yz",radius_yz)
        self.update_radius(bin_center, radius_yz, radius_type)
        return radius_yz        

    def update_cylinder_radius_from_wrap(self, eps=1.0, isotropic = True):
        if self.key_wrap_radius is None:
            radius_cover = np.quantile(self.key_wrap_radius, 0.95, axis=1) + eps # np.max(self.key_train_radius, axis = 1)
            radius_cover = get_radius_with_eps(radius_cover, eps)
            self.key_cylinder_radius = np.stack([radius_cover, radius_cover], axis=1)
        else:
            if isotropic:
                radius_cover = np.quantile(self.key_wrap_radius, 0.95, axis=1) + eps # np.max(self.key_wrap_radius, axis=1)
                radius_cover = get_radius_with_eps(radius_cover, eps)
                self.key_cylinder_radius = np.stack([radius_cover, radius_cover], axis=1)
            else:
                theta = self.wrap_theta_bins[None, :]
                uu = self.key_wrap_radius * np.cos(theta)
                vv = self.key_wrap_radius * np.sin(theta)
                a = np.quantile(np.abs(uu), 0.95, axis=1) #np.max(np.abs(uu), axis=1)
                b = np.quantile(np.abs(vv), 0.95, axis=1) #np.max(np.abs(vv), axis=1)
                ry = np.sqrt(2.0) * a
                rz = np.sqrt(2.0) * b
                radius_y = get_radius_with_eps(ry, eps)
                radius_z = get_radius_with_eps(rz, eps)
                self.key_cylinder_radius = np.stack([radius_y, radius_z], axis=1)
        return self.key_cylinder_radius

    def update_cylinder_radius_from_coords(self, coord_points, w, u, v, n_bins=24, quantile=0.98, gaussian_smooth=2.0, min_count=150, eps=0.02, isotropic=False):
        bin_edge = np.linspace(0.0, 1.0, n_bins + 1)
        bin_center = 0.5 * (bin_edge[:-1] + bin_edge[1:])
        bin_ids = np.clip(np.digitize(coord_points, bin_edge) - 1, 0, n_bins - 1)

        radius_y = np.full(n_bins, np.nan, dtype=np.float64)
        radius_z = np.full(n_bins, np.nan, dtype=np.float64)
        sample_count = np.zeros(n_bins, dtype=np.int32)

        slab_half_width = 2.0 / n_bins

        for b in range(n_bins):
            coords_in_b = (bin_ids == b) & (np.abs(w) <= slab_half_width)
            sample_count[b] = np.sum(coords_in_b)
            if sample_count[b] < min_count:
                continue

            abs_u = np.abs(u[coords_in_b])
            abs_v = np.abs(v[coords_in_b])

            if isotropic:
                rr = np.sqrt(u[coords_in_b] ** 2 + v[coords_in_b] ** 2)
                r = np.quantile(rr, quantile) + eps
                radius_y[b] = r
                radius_z[b] = r
            else:
                radius_y[b] = np.quantile(abs_u, quantile) + eps
                radius_z[b] = np.quantile(abs_v, quantile) + eps

        valid_bins = np.isfinite(radius_y) & np.isfinite(radius_z)
        radius_y = fill_invalid_bins(radius_y, valid_bins)
        radius_z = fill_invalid_bins(radius_z, valid_bins)

        if gaussian_smooth > 0:
            radius_y = gaussian_filter1d(radius_y, sigma=gaussian_smooth)
            radius_z = gaussian_filter1d(radius_z, sigma=gaussian_smooth)

        radius_yz = np.stack([radius_y, radius_z], axis=1)
        self.update_radius(bin_center, radius_yz, radius_type='cylinder')
        return radius_yz

    def calc_x_radius(self, ts):
        xrs = np.ones(ts.shape[0])
        #if self.end_ball_x is not None:
        #    xrs[ts == 1.] = self.end_ball_x
        
        #if self.start_ball_x is not None:
        #    xrs[ts == 0.] = self.start_ball_x
        
        return xrs
