import os, pickle
import numpy as np
import os.path as op
import trimesh
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation, Slerp
from handle_utils import CylindersMesh
from scipy.ndimage import gaussian_filter1d
from curve_utils.visualize_util import *
from curve_utils.curve_utils import *
from curve_functions._interpolate import interpolate_occ_profile1, interpolate_wrap_radius1
from curve_functions._frame import *
from curve_functions._update import update_wrap_profile_from_coords, update_wrap_occupancy_from_coords


n_sample_curve = 200
n_sample_circle = 120


class PWLACurve():
    """docstring for PWLACurve."""
    def __init__(self, arg=None):
        if arg is None:
            return 
        
        self.set_curve(arg)

    def update(self):
        if self.flag_points:
            # if some keypoints changed, update the curve
            self.update_coords()
            self.update_frame()
            self.flag_points = False
            return
        
    def need_update(self):
        return self.flag_points
    
    def set_curve(self, arg):
        # self.step = arg['resample_step']
        self.key_points = arg['key_points']
        # NOTE: radius: (N, 2), y-z radius
        self.key_train_radius = arg.get('key_train_radius', arg.get('key_radius'))
        self.key_cylinder_radius = arg.get('key_cylinder_radius', self.key_train_radius.copy())
        self.key_radius = self.key_train_radius
        self.key_wrap_radius = arg.get('key_wrap_radius', None)
        self.key_occupancy_rho = arg.get('key_occupancy_rho', None)
        self.wrap_s_bins = arg.get('wrap_s_bins', None)
        self.wrap_theta_bins = arg.get('wrap_theta_bins', None)
        self.wrap_radius_max = arg.get('wrap_radius_max', None)

        x_axis = self.estimate_tangent(self.key_points)
        z_axis = arg['z_axis']
        if len(z_axis.shape) == 1:
            z_axis = np.tile(z_axis, (x_axis.shape[0], 1))
        
        self.z_axis = self.project_z_axis(x_axis, z_axis)
        self.update_coords()
        # check if points or radius have to be updated
        self.flag_points = True


    def set_points(self, points):
        assert (points.shape == self.key_points.shape)
        self.key_points = points
        self.flag_points = True


    def set_resamples(self, points, z_axis0):
        if z_axis0 is None:
            z_axis0 = self.z_axis[0]
        # new points can be different in number
        new_ts = self.keypoints_segment_length(points)
        #edge_vec = points[1:] - points[:-1]
        #edge_lengths = np.linalg.norm(edge_vec, axis=1)
        #curve_length = np.sum(edge_lengths)
        #ts = np.cumsum(np.r_[0., edge_lengths]) / curve_length

        new_key_train_radius = self.interpolate(new_ts, radius=True, radius_type='train')['radius']
        if getattr(self, 'key_cylinder_radius', None) is not None:
            new_key_cylinder_radius = self.interpolate(new_ts, radius=True, radius_type='cylinder')['radius']
        else:
            new_key_cylinder_radius = new_key_train_radius.copy()

        if (getattr(self, 'key_wrap_radius', None) is not None) and (self.wrap_s_bins is not None):
            new_key_wrap_radius = np.stack([np.interp(new_ts, self.wrap_s_bins, self.key_wrap_radius[:,j]) for j in range(self.key_wrap_radius.shape[1])], axis=1)
        else:
            new_key_wrap_radius = None

        self.key_points = points
        self.key_ts = new_ts
        self.key_radius = new_key_train_radius
        self.key_train_radius = new_key_train_radius
        self.key_cylinder_radius = new_key_cylinder_radius
        self.key_wrap_radius = new_key_wrap_radius

        self.wrap_s_bins = new_ts if new_key_wrap_radius is not None else self.wrap_s_bins
        self.wrap_radius_max = np.max(new_key_wrap_radius, axis=1) if new_key_wrap_radius is not None else self.wrap_radius_max

        x_axis = self.estimate_tangent(self.key_points)
        self.z_axis = self.propagate_z_axis(x_axis, z_axis0)
        y_axis = np.cross(self.z_axis, x_axis)
        self.key_frame = np.concatenate([
            x_axis.reshape(-1,1,3),
            y_axis.reshape(-1,1,3),
            self.z_axis.reshape(-1,1,3)
        ], axis=1)
        #self.rotation = Rotation.from_matrix(self.key_frame)
        #self.rot_slerp = Slerp(self.key_ts, self.rotation)
        self.flag_points = True


    def apply_rotation(self, anchor, rot):
        self.key_points = rot.apply(self.key_points - anchor) + anchor
        self.z_axis = rot.apply(self.z_axis)
        self.flag_points = True

    def estimate_tangent(self, points):
        edge_vec = points[1:] - points[:-1]
        edge_vec /= (np.linalg.norm(edge_vec, axis=1, keepdims=True) + 1e-12)

        if edge_vec.shape[0] > 1:
            tan_start = edge_vec[0]
            tan_end = edge_vec[-1]
            tans = (edge_vec[1:] + edge_vec[:-1]) / 2.
            tans /= np.linalg.norm(tans, axis=1, keepdims=True)

            vert_tan = np.concatenate([
                tan_start.reshape(1,3), 
                tans, 
                tan_end.reshape(1,3)
            ], axis=0)
        else:
            vert_tan = np.tile(edge_vec, (2,1))

        return vert_tan
    
    def project_z_axis(self, x_axis, z_axis):
        dots = np.sum(x_axis*z_axis, axis=1)
        if not np.allclose(dots, 0):
            # project z_axis to x_axis
            z_axis = z_axis - dots[:, None]* x_axis
            # NOTE: huh???? forgot this
            z_axis /= np.linalg.norm(z_axis, axis=1, keepdims=True)

        return z_axis
    
    def propagate_z_axis(self, x_axis, z_axis0):
        final_z = []
        # current z_axis
        c_zx = z_axis0 
        for i in range(x_axis.shape[0]):
            xx = x_axis[i]
            zx = c_zx - (xx @ c_zx)*xx
            zx /= np.linalg.norm(zx)
            final_z.append(zx)
            c_zx = zx

        return np.asarray(final_z)

    def rotation_from_vectors(self, a, b):
         v = np.cross(a, b)
         c = np.dot(a, b)

         if np.linalg.norm(v) < 1e-8:
             return np.eye(3)

         vx = np.array([
              [0, -v[2], v[1]],
              [v[2], 0, -v[0]],
              [-v[1], v[0], 0]
              ])
         R = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
         return R

    # Parallel transport frame
    def update_frame(self):
        points = self.key_points
        n = points.shape[0]

        T = self.estimate_tangent(self.key_points)
        z0 = self.z_axis[0] if self.z_axis is not None else np.array([0,0,1], dtype=np.float64)

        z_axis = np.zeros_like(T)
        z_axis[0] = z0 - np.dot(z0, T[0]) * T[0]
        z_axis[0] /= np.linalg.norm(z_axis[0]) + 1e-12

        for i in range(1, len(T)):
            R = self.rotation_from_vectors(T[i-1], T[i])
            z_axis[i] = R @ z_axis[i-1]
            z_axis[i] /= np.linalg.norm(z_axis[i]) + 1e-12

        y_axis = np.cross(z_axis, T)
        y_axis /= np.linalg.norm(y_axis, axis=1, keepdims=True) + 1e-12
        z_axis = np.cross(T, y_axis)
        z_axis /= (np.linalg.norm(z_axis, axis=1, keepdims=True) + 1e-12)
        self.z_axis = z_axis
        self.key_frame = np.stack([T, y_axis, z_axis], axis=1)

        self.rotation = None
        self.rot_slerp = None
 
    # Parallel transport frame
    def get_new_frame(self, points):
        n = points.shape[0]

        T = self.estimate_tangent(points)
        z0 = self.z_axis[0] if self.z_axis is not None else np.array([0,0,1], dtype=np.float64)

        z_axis = np.zeros_like(T)
        z_axis[0] = z0 - np.dot(z0, T[0]) * T[0]
        z_axis[0] /= np.linalg.norm(z_axis[0]) + 1e-12

        for i in range(1, len(T)):
            R = self.rotation_from_vectors(T[i-1], T[i])
            z_axis[i] = R @ z_axis[i-1]
            z_axis[i] /= np.linalg.norm(z_axis[i]) + 1e-12

        y_axis = np.cross(z_axis, T)
        y_axis /= np.linalg.norm(y_axis, axis=1, keepdims=True) + 1e-12
        z_axis = np.cross(T, y_axis)
        z_axis /= (np.linalg.norm(z_axis, axis=1, keepdims=True) + 1e-12)
        key_frame = np.stack([T, y_axis, z_axis], axis=1)
        return key_frame


    def update_frame_slerp(self):
        x_axis = self.estimate_tangent(self.key_points)

        z0 = self.z_axis[0] if self.z_axis is not None else np.array([0.0, 0.0, 1.0], dtype=np.float64)
        z_axis = self.propagate_z_axis(x_axis, z0)

        # Sign continuity
        for i in range(1, z_axis.shape[0]):
            if np.dot(z_axis[i], z_axis[i-1]) < 0:
                z_axis[i] *= -1.0
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= (np.linalg.norm(y_axis, axis=1, keepdims=True) + 1e-12)

        z_axis = np.cross(x_axis, y_axis)
        z_axis /= (np.linalg.norm(z_axis, axis=1, keepdims=True) + 1e-12)
        self.z_axis = z_axis

        self.key_frame = np.concatenate([
            x_axis.reshape(-1,1,3),
            y_axis.reshape(-1,1,3),
            z_axis.reshape(-1,1,3)
        ], axis=1)
        self.rotation = Rotation.from_matrix(self.key_frame)
        self.rot_slerp = Slerp(self.key_ts, self.rotation)

    def update_frame_old(self):
        x_axis = self.estimate_tangent(self.key_points)
        self.z_axis = self.project_z_axis(x_axis, self.z_axis)
        y_axis = np.cross(self.z_axis, x_axis)
        self.key_frame = np.concatenate([
            x_axis.reshape(-1,1,3),
            y_axis.reshape(-1,1,3),
            self.z_axis.reshape(-1,1,3)
        ], axis=1)
        self.rotation = Rotation.from_matrix(self.key_frame)
        self.rot_slerp = Slerp(self.key_ts, self.rotation)

    def set_frame(self, new_frame):
        self.key_frame = new_frame
        
        #idx = np.searchsorted(self.key_ts, ts)
        #idx = np.clip(idx, 0, len(self.key_ts)-1)
        #frame = self.key_frame[idx]
        #self.rotation = Rotation.from_matrix(self.key_frame)
        #self.rot_slerp = Slerp(self.key_ts, self.rotation)

    def update_coords(self):
        edge_vec = self.key_points[1:] - self.key_points[:-1]
        edge_lengths = np.linalg.norm(edge_vec, axis=1)
        self.curve_length = np.sum(edge_lengths)
        self.key_ts = np.cumsum(np.r_[0., edge_lengths]) / self.curve_length

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

    def update_radius_from_surfacepoints(self, points, n_bins=n_sample_curve, quantile=0.98, gaussian_smooth=1.0, radius_type='train'):
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
        print("radius_yz",radius_yz)
        self.update_radius(bin_center, radius_yz, radius_type)
        
        return {"u": u,
                "v": v,
                "radius": radius_yz }

    def update_radius_from_coords(self, coord_points, w, u, v, n_bins=n_sample_curve, quantile=0.98, gaussian_smooth=2.0, min_count=30, radius_type='train'):
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


    def update_cylinder_radius_from_coords(self, coord_points, w, u, v, n_bins=n_sample_curve, quantile=0.98, gaussian_smooth=2.0, min_count=150, eps=0.02, isotropic=False):
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

#    def update_wrap_profile_from_coords(self, coord_points, w, u, v, n_curve_bins=n_sample_curve, n_theta_bins=24, quantile=0.97, gaussian_smooth_curve=2.0, gaussian_smooth_theta=1.0, min_count = 30, radius_type = 'train'):
#        rho = np.sqrt(u*u + v*v)
#        theta = np.arctan2(v, u)
#        bin_edges_curve = np.linspace(0.0, 1.0, n_curve_bins+1)
#        bin_center_curve = 0.5 * (bin_edges_curve[:-1] + bin_edges_curve[1:])
#        bin_ids_curve = np.clip(np.digitize(coord_points, bin_edges_curve) -1, 0, n_curve_bins-1)
#
#        bin_edges_theta = np.linspace(-np.pi, np.pi, n_theta_bins + 1)
#        bin_center_theta = 0.5 * (bin_edges_theta[:-1] + bin_edges_theta[1:])
#        bin_ids_theta = np.clip(np.digitize(theta, bin_edges_theta) -1, 0, n_theta_bins-1)
#
#        r_wrap = np.full((n_curve_bins, n_theta_bins), np.nan, dtype=np.float64)
#        counts = np.zeros((n_curve_bins, n_theta_bins), dtype=np.int32)
#
#        slab_half_width = 1.0 / n_curve_bins
#        slab_mask = np.abs(w) <= slab_half_width
#
#        for s in range(n_curve_bins):
#            s_mask = (bin_ids_curve == s) & slab_mask 
#            if np.sum(s_mask) == 0:
#                continue
#            for t in range(n_theta_bins):
#                m = s_mask & (bin_ids_theta == t) 
#                counts[s,t] = np.sum(m)
#                if counts[s,t] < min_count:
#                    continue
#                r_wrap[s, t] = np.quantile(rho[m], quantile)
#
#        for s in range(n_curve_bins):
#            row = r_wrap[s]
#            valid = np.isfinite(row)
#            if np.any(valid):
#                r_wrap[s] = fill_invalid_theta(row, valid)
#       
#        for t in range(n_theta_bins):
#            col = r_wrap[:,t]
#            valid = np.isfinite(col)
#            if np.any(valid):
#                r_wrap[:,t] = fill_invalid_bins(col, valid) 
#
#        # gaussian smooth of Theta
#        if gaussian_smooth_theta > 0:
#            r_wrap = gaussian_filter1d(np.concatenate([r_wrap, r_wrap, r_wrap], axis=1), sigma=gaussian_smooth_theta, axis=1)[:, n_theta_bins:2*n_theta_bins]
#        if gaussian_smooth_curve > 0:
#            r_wrap = gaussian_filter1d(r_wrap, sigma=gaussian_smooth_curve, axis=0)
#
#
#        self.wrap_s_bins = bin_center_curve 
#        self.wrap_theta_bins = bin_center_theta
#        self.key_wrap_radius = r_wrap
#        self.wrap_radius_max = np.max(r_wrap, axis=1)
#
#        return {
#            "curve_bins": bin_center_curve,
#            "theta_bins": bin_center_theta,
#            "key_wrap_radius": self.key_wrap_radius,
#            "wrap_radius_max": self.wrap_radius_max,
#            "counts": counts,
#        }

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
    

    def curve_projection(self, samples, N_discrete=n_sample_curve, outside=False):
        uniform_linear_points = np.linspace(0., 1., N_discrete, endpoint=False)
        ii = np.searchsorted(uniform_linear_points, self.key_ts)
        non_uniform_linear_points = np.insert(uniform_linear_points, ii, self.key_ts)
        non_uniform_linear_points = np.unique(non_uniform_linear_points)

        skeletal_verts = self.interpolate(non_uniform_linear_points, radius=False, frame=False)['points']
        tree = KDTree(skeletal_verts)
        # not accurate for radius-varying skeleton
        _, vidx = tree.query(samples)
        samples3D_to_skeleton = -1*np.ones(samples.shape[0])
        # basically project samples onto the piecewise linear curve
        num_vert = skeletal_verts.shape[0]
        for vid in range(num_vert):
            sample_index = np.argwhere(vidx == vid).flatten()
            if len(sample_index) == 0:
                continue

            samples_v = samples[sample_index]

            if 0 < vid < num_vert - 1:
                # middle part
                ## The samples which belong to the sample_index is mapped to the nearest keypoints through their index 
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
                in1, px1 = self.is_points_in_edge(
                    samples_v, 
                    (skeletal_verts[vid], non_uniform_linear_points[vid]), 
                    (skeletal_verts[vid+1], non_uniform_linear_points[vid+1])
                )
                # consider halfball+ cylinder
                # left side of cylinder remain valid
                #if self.start_ball_x is not None or outside:
                #    samples3D_to_skeleton[sample_index] = 0.

                samples3D_to_skeleton[sample_index[in1]] = px1[in1]

            else:
                in2, px2 = self.is_points_in_edge(
                    samples_v, 
                    (skeletal_verts[vid-1], non_uniform_linear_points[vid-1]), 
                    (skeletal_verts[vid], non_uniform_linear_points[vid]), 
                )
                #if self.end_ball_x is not None or outside:
                #    samples3D_to_skeleton[sample_index] = 1.

                samples3D_to_skeleton[sample_index[in2]] = px2[in2]

        #import pdb; pdb.set_trace();
        return samples3D_to_skeleton

    def calc_x_radius(self, ts):
        xrs = np.ones(ts.shape[0])
        #if self.end_ball_x is not None:
        #    xrs[ts == 1.] = self.end_ball_x
        
        #if self.start_ball_x is not None:
        #    xrs[ts == 0.] = self.start_ball_x
        
        return xrs
    
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
        if self.start_ball_x is None:
            norms_cyl[xneg] = np.maximum(norms_cyl[xneg], norms_max[xneg])

        if self.end_ball_x is None:
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
        stats = compute_local_centering_stats(samples_local, sample_keypoint_map, n_bins=n_sample_curve, min_count=150)
        #C_old, C_new, C_key_new = compute_centered_curve_world(self, stats)
        C_old, C_new = compute_centered_curve_world(self, stats)
        C_new_smooth = gaussian_filter1d(C_new, sigma=2, axis=0)
        export_curve_points_as_ply(C_old, C_new_smooth, str(idx)+"_curve_compare_points.ply")
        export_shape_and_curves_as_ply(
            points=pointcloudsamples,
            C_old=C_old,
            C_new=C_new_smooth,
            out_path=str(idx)+"_shape_and_curves.ply"
        )

        #plot_centroid_offsets_from_origin(stats)
        #plot_centered_curve_local_projections(stats)
        #plot_centroid_path_with_origin(stats)
        #plot_local_centering_stats(stats)
        #plot_local_bins(stats, bins=[10, 25, 40, 60, 80])   
        #plot_local_bins_with_drift_clean(stats, bins=[10, 25, 40, 60, 80])


    def localize_samples(self, pointcloudsamples, return_sdf=False, norm=1.0, update_curve=False, update_radius=False, name=''):
        sample_keypoint_map = self.curve_projection(pointcloudsamples)
        sample_keypoint_map_range = np.logical_and(sample_keypoint_map >= 0., sample_keypoint_map <= 1.)
        sample_index = np.arange(pointcloudsamples.shape[0])

        ### Keep only the points that fall within the rane of 0 and 1
        if update_curve:
            pointcloudsamples0 = pointcloudsamples.copy()
        sample_keypoint_map = sample_keypoint_map[sample_keypoint_map_range]
        pointcloudsamples = pointcloudsamples[sample_keypoint_map_range]
        sample_index = sample_index[sample_keypoint_map_range]

        # interpolate with the new additional non linear skeletal keypoints
        intpl = self.interpolate(sample_keypoint_map, radius_type='train')

        ## The new keypoiints in 3D world coord system based on the curve projection from the surface/space samples
        proj_vs = intpl['points']
        yz_radius = intpl['radius']
        frame_mat = intpl['frame']

        # frame: (N, 3,3), vs (N, 3)
        # The vector of the keypoint to the skeletam point is rotated using the rotation from
        samples_local0 = np.einsum('nij,nj->ni', frame_mat, (pointcloudsamples - proj_vs))
        # And all are bounding to radius
        w, u, v = samples_local0[:,0], samples_local0[:, 1], samples_local0[:, 2]
        if update_curve:
            stats = compute_local_centering_stats(samples_local0, sample_keypoint_map)
            #C_old, C_new, C_key_unsmooth = compute_centered_curve_world(self, stats)
            C_old, C_new = compute_centered_curve_world(self, stats)
            s_dense = 0.5 * (stats["edges"][:-1] + stats["edges"][1:])
            #C_key_smooth = resample_curve_to_key_ts(s_dense, C_new_smooth, self.key_ts)
            s_target = np.linspace(0.0, 1.0, n_sample_curve)
            C_key_old_smooth = resample_curve_to_key_ts(s_dense, C_old, s_target)
            old_frame = self.get_new_frame(C_key_old_smooth) #key_frame 
            #self.update_frame()
            #old_frame = self.key_frame 
            old_T = old_frame[:, 0, :]

            C_new_smooth = gaussian_filter1d(C_new, sigma=2.0, axis=0)
            C_key_smooth = resample_curve_to_key_ts(s_dense, C_new_smooth, s_target)
            z0 = self.z_axis[0] if getattr(self, "z_axis", None) is not None else None
            self.set_resamples(C_key_smooth, z_axis0=z0)
            self.update_coords()

            #self.update_coords()
            #self.update_frame()
            #new_frame = self.get_new_frame(C_key_smooth) #key_frame 
            new_frame = self.get_new_frame(self.key_points) #key_frame 
            new_T = new_frame[:, 0, :]
            auto_N = new_frame[:, 1, :]
            xfer_N = self.key_frame[:, 1, :]

            dots = np.sum(auto_N * xfer_N, axis=1)
            print("normal alignment min/max:", dots.min(), dots.max())
            print("normal alignment mean:", dots.mean())

            self.key_frame = transfer_frame_orientation(
                old_frame=old_frame,
                old_tangent=old_T,
                new_tangent=new_T,
                enforce_continuity=True,
                orthonormalize=True,
            )
            self.z_axis = self.key_frame[:, 2, :].copy()
            self.y_axis = self.key_frame[:, 1, :].copy()
            #self.update_frame()
            #C_key_smooth = resample_curve_to_key_ts(s_dense, C_old, s_target)
            #self.update_frame()


            return self.localize_samples(pointcloudsamples0, return_sdf=return_sdf, norm=norm, update_curve=False, update_radius=True, name=name)

        if update_radius:
            update_wrap_profile_from_coords(self, sample_keypoint_map, w, u, v, n_curve_bins=n_sample_curve, n_theta_bins=24, quantile=0.98, gaussian_smooth_curve=2.0, gaussian_smooth_theta=2.0, min_count=25, radius_type='wrap')
            update_wrap_occupancy_from_coords(self, sample_keypoint_map, u, v, n_curve_bins=n_sample_curve, quantile=0.97, min_count=50)

            ################# Uncomment later for curve center and cylinder #################################
#            self.update_radius_from_coords(sample_keypoint_map, w, u, v)
#            #self.update_wrap_profile_from_coords(sample_keypoint_map, w, u, v, radius_type='wrap')
#            self.update_cylinder_radius_from_coords(sample_keypoint_map, w, u, v, radius_type='cylinder')
#            #self.update_cylinder_radius_from_wrap(eps=0.03, isotropic=False)
#            radius_yz = self.interpolate(sample_keypoint_map, points=False, frame=False, radius_type='train')['radius']
#            #print(yz_radius.shape)
#            yz_radius = radius_yz.copy()
#            self.update_coords()
#            # 3D trimesh overlay
#            visualize_keyframes_with_profiles_trimesh(
#                self,
#                pointcloudsamples,
#                sample_keypoint_map,
#                name=name,
#                show_train=False,
#                show_cylinder=True,
#                show_wrap=False,
#                export_glb=False,
#                export_ply=True
#            )
            #exit()

        # If the end ball is None then x_radius  = 1.0 
        x_radius = self.calc_x_radius(sample_keypoint_map)
        #print(yz_radius.shape)
        #print(x_radius[:,None].shape)
        radius = np.concatenate([x_radius[:,None], yz_radius], axis=1)

        samples_local = samples_local0.copy()
        samples_local /= (radius + 1e-12)
        rho = np.sqrt(v**2 + u**2)
        u_n = samples_local[:,1] #u / (radius[:,1] + 1e-12)
        v_n = samples_local[:,2] #v / (radius[:,2] + 1e-12)
        angle = np.arctan2(v_n, u_n)
        rho_n = np.sqrt(v_n**2 + u_n**2)
        
        # in std cylinder
        norms = np.linalg.norm(samples_local, axis=1)
        if return_sdf:
            return norms - 1, sidx
        
        inside_cyl = (norms <= norm)
        inside = sample_index[inside_cyl]

        # NOTE: vs -> (vx, *, *). (vert -> (vx, 0, 0))
        # [0,1] -> [-1,1]
        vx = 2*sample_keypoint_map - 1
        samples_local[:, 0] += vx
        return {
            'samples': pointcloudsamples[inside_cyl],
            'samples_local': samples_local[inside_cyl],
            'coords': sample_keypoint_map[inside_cyl],
            'rho': rho[inside_cyl],
            'rho_n': rho_n[inside_cyl],
            'angles': angle[inside_cyl],
            'radius': yz_radius[inside_cyl],
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

    def keypoints_segment_length(self, points):
        edge_vec = points[1:] - points[:-1]
        edge_lengths = np.linalg.norm(edge_vec, axis=1)
        curve_length = np.sum(edge_lengths)
        s = np.concatenate([[0.0], np.cumsum(edge_lengths)])
        ts = s / (s[-1] + 1e-12)
        return ts

    def calc_curve_length(self):
        pts = self.key_points
        edge_vec = pts[1:] - pts[:-1]
        seglength = np.linalg.norm(edge_vec, axis=1)
        cumulative_length = np.concatenate([[0], np.cumsum(seglength)])
        curve_length = cumulative_length[-1] #np.sum(edge_lengths)
        return curve_length, cumulative_length

    def stretch_from_end_smooth_nonuniform(self, stretch_arg):
        points = self.key_points.copy()
        anchor = stretch_arg['anchor']
        stretch_length = stretch_arg['length']

        curve_length, cumulative_length = self.calc_curve_length()


        flipped = False
        if anchor == 'end':
            points = points[::-1].copy()
            flipped = True

        delta_length = (stretch_length - 1.0) * curve_length

        ts = self.keypoints_segment_length(points)
        # smooth curve
        w = ts*ts*(3 - 2*ts)
        w = w[:, None]

        t_end = points[-1] - points[-2]
        t_end /= (np.linalg.norm(t_end) + 1e-12)

        points = points + w * (delta_length * t_end)
        if flipped:
            points = points[::-1].copy()
        return points

    def stretch_uniform(self, stretch_arg):
        anchor = stretch_arg['anchor']
        stretch_length = stretch_arg['length']
        points = self.key_points.copy()

        if anchor == "start":
            out = points.copy()
            out[0] = points[0]
            for k in range(len(points)-1):
                out[k+1] = out[k] + stretch_length * (points[k+1] - points[k])
            return out

        if anchor == "end":
            points_reverse = points[::-1].copy()
            out = points_reverse
            out[0] = points_reverse[0]
            for k in range(len(points_reverse)-1):
                out[k+1] = out[k] + stretch_length * (points_reverse[k+1] - points_reverse[k])
            return out[::-1].copy()

        if anchor == "center":
            # stretch about mid index (keeps center fixed, stretches both directions)
            m = len(points)//2
            out = points.copy()
            out[m] = points[m]
            # forward
            for k in range(m, len(points)-1):
                out[k+1] = out[k] + stretch_length * (points[k+1] - points[k])
            # backward
            for k in range(m, 0, -1):
                out[k-1] = out[k] - stretch_length * (points[k] - points[k-1])
            return out


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

#        return {
#            #'samples': vs[inside_cyl],
#            'samples_local': samples_local[inside_cyl],
#            #'samples_detail': samples_detail[inside_cyl],
#            #'coords': ts_new[inside_cyl],
#            'coords': ts[inside_cyl],
#            #'coords_detail': ts_used[inside_cyl], 
#            #'w_seam': w_seam,
#            'rho': rho[inside_cyl],
#            'rho_n': rho_n[inside_cyl],
#            'angles': angle[inside_cyl],
#            'radius': yz_radius[inside_cyl],
#        }, inside


        stretch_length = stretch_arg['length']
        t0 = stretch_arg['t0']
        t1 = stretch_arg['t1']
        eps_region = stretch_arg.get('eps_region', 0.03)
        eps_seam = stretch_arg.get('eps_seam', 0.05)
      
        w_region = make_detail_mask(ts_new, t0, t1, eps_region)
        eps = 1e-12
        tau = np.clip((ts_new - t0)/((t1 - t0)+eps), 0.0, 1.0)
        ts_tile_phase = np.mod((stretch_length * tau), 1.0)

        # seam fade: 0 near phase seam, 1 away from seam
        w_seam = seam_fade(ts_tile_phase, eps_seam)
        w_wrap = w_region * w_seam

        ts_wrapped = t0 + (t1-t0)*ts_tile_phase
        ts_used = (1.0 - w_wrap) * ts_new + w_wrap * ts_wrapped
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
            'w_seam': w_seam,
            'rho': rho[inside_cyl],
            'rho_n': rho_n[inside_cyl],
            'angles': angle[inside_cyl],
            'radius': yz_radius[inside_cyl],
        }, inside


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

    def normalized_arclen_keypoints(self):
        """
        Returns A[k] in [0,1] at each keypoint, monotonic.
        """
        curve_length, cum_length = self.calc_curve_length()  # your version returns (L, cumulative)
        return cum_length / (curve_length + 1e-12)

    
    def map_coords_to_by_arclen(self, coords_src, target_core):
        """
        Map coords on this curve -> coords on target curve using normalized arc-length.
        coords_src: (N,) in [0,1] (same space as self.key_ts)
        """
        A_src = self.normalized_arclen_keypoints()
        A_tgt = target_core.normalized_arclen_keypoints()

        # coords_src -> arc fraction
        a = np.interp(coords_src, self.key_ts, A_src)
        # arc fraction -> target coords
        coords_tgt = np.interp(a, A_tgt, target_core.key_ts)
        return coords_tgt

    def localize_samples_adapt(self, vs, adapt_arg):
        # calculate the neareast points and find inside points
        avatar_data, inside = self.localize_samples(vs)

        accessory_curve_handle = adapt_arg['accessory_curve_handle']
        accessory_curve_handle.core.update_coords()
        accessory_curve_handle.core.update_frame()

        avatar_coords = avatar_data["coords"]                         
        avatar_samples_local = avatar_data["samples_local"].copy()   

        vx_avatar = 2.0 * avatar_coords - 1.0
        w_n_avatar = avatar_samples_local[:, 0] - vx_avatar
        u_n_avatar = avatar_samples_local[:, 1]
        v_n_avatar = avatar_samples_local[:, 2]


        avatar_radius_y = avatar_data["radius"][:,0]
        avatar_radius_z = avatar_data["radius"][:,1]

        tangent_avatar = self.calc_x_radius(avatar_coords)          
        w_avatar = w_n_avatar * (tangent_avatar + 1e-12)
        u_avatar = u_n_avatar * (avatar_radius_y)
        v_avatar = v_n_avatar * (avatar_radius_z)

        rho_avatar = np.sqrt(u_avatar**2 + v_avatar**2)
        rho_n_avatar = np.sqrt(u_n_avatar**2 + v_n_avatar**2)
        theta_avatar = np.arctan2(v_avatar, u_avatar)
        angle_avatar = np.arctan2(v_n_avatar, u_n_avatar)

        # map avatar coords -> accessory coords by arclen
        #avatar_coords = maybe_flip_coords(avatar_coords, True) #adapt_arg.get("flip_s", False))

        source_npz = np.load('ngc/armadillo_on.npz', allow_pickle=True)['arr_0'].item()['armadillo_on_8']
        target_npz = np.load('ngc/boots_on.npz', allow_pickle=True)['arr_0'].item()['boots_on_3']

        acc_coords = self.map_coords_to_by_arclen(avatar_coords, accessory_curve_handle.core)
        acc_intpl = accessory_curve_handle.core.interpolate(acc_coords)

        #accessory_curve_handle.core.update_coords()
        #accessory_curve_handle.core.update_frame()
        tangent_acc = accessory_curve_handle.core.calc_x_radius(acc_coords)

        acc_radius_y = acc_intpl["radius"][:,0]
        acc_radius_z = acc_intpl["radius"][:,1]
        scale_w = tangent_acc / (tangent_avatar + 1e-12)
        w_acc = w_avatar * scale_w 
        scale_y = acc_radius_y / (avatar_radius_y + 1e-12)
        scale_z = acc_radius_z / (avatar_radius_z + 1e-12)

        #source_npz  = np.load('ngc/armadillo_on.npz', allow_pickle=True)['arr_0'].item()['armadillo_on_8']
        #r_src = self.interpolate_wrap_radius(avatar_coords, theta_avatar) #, source_npz['wrap_theta_bins'], source_npz['wrap_s_bins'], source_npz['key_wrap_radius'])
        #r_tgt = accessory_curve_handle.core.interpolate_wrap_radius(acc_coords, theta_avatar) #, t['wrap_theta_bins'], t['wrap_s_bins'], t['key_wrap_radius'])
        if adapt_arg['wrap_radius']:
            r_src = interpolate_wrap_radius1(self, avatar_coords, theta_avatar, source_npz['key_wrap_radius'], source_npz['wrap_theta_bins'], source_npz['wrap_s_bins'])
            r_tgt = interpolate_wrap_radius1(accessory_curve_handle.core, acc_coords, theta_avatar, target_npz['key_wrap_radius'], target_npz['wrap_theta_bins'], target_npz['wrap_s_bins']) 

            #print("occ key:", source_npz.keys() if hasattr(source_npz, "keys") else type(source_npz))
            #print("occ shape:", np.asarray(source_npz['key_occupancy_rho']).shape)
            #print("s_bins shape:", np.asarray(source_npz['wrap_s_bins']).shape)
            occ_src = interpolate_occ_profile1(self, avatar_coords, source_npz['key_occupancy_rho'], source_npz['wrap_s_bins'])
            occ_tgt = interpolate_occ_profile1(accessory_curve_handle.core, acc_coords, target_npz['key_occupancy_rho'], target_npz['wrap_s_bins'])

            occ_scale = ( occ_tgt) / (occ_src + 1e-12)

            scale = ( 0.8* r_tgt) / (r_src + 1e-12)
            #scale *= occ_scale
            
            rho_acc = rho_avatar * scale 
            u_acc = rho_acc * np.cos(theta_avatar)
            v_acc = rho_acc * np.sin(theta_avatar)
        else:
            scale = 0.8
            u_acc = scale *u_avatar * scale_y
            v_acc = scale *v_avatar * scale_z
            rho_acc = np.sqrt(u_acc**2 + v_acc**2)

        #q_src = rho_avatar / (r_src + 1e-12)
        #q_tgt = rho_acc / (r_tgt + 1e-12)
        #occ_src = np.median(q_src)    #or quantile(q_src, 0.7 or 0.8)
        #occ_tgt = np.median(q_tgt)    #or quantile(q_tgt, 0.7 or 0.8)
        #beta = occ_tgt / (occ_src + 1e-12)

        #rho_acc = beta * rho_avatar * scale 



        w_n_acc = w_acc / (tangent_acc + 1e-12)
        u_n_acc = u_acc / (acc_radius_y + 1e-12)
        v_n_acc = v_acc / (acc_radius_z + 1e-12)


        vx_acc = 2.0 * acc_coords - 1.0
        #samples_local_acc = np.stack([w_n_avatar + vx_acc, u_n_acc, v_n_acc], axis=1)
        samples_local_acc = np.stack([w_n_acc + vx_acc, u_n_acc, v_n_acc], axis=1)

        #u_acc = u_n_acc * acc_radius_y
        #v_acc = v_n_acc * acc_radius_z
        #rho_acc = np.sqrt(u_acc**2 + v_acc**2)
        rho_n_acc = np.sqrt(u_n_acc**2 + v_n_acc**2)
        angles_acc = np.arctan2(v_n_acc, u_n_acc)

        accessory_data = dict(avatar_data)
        accessory_data["coords"] = acc_coords
        accessory_data["samples_local"] = samples_local_acc
        accessory_data["angles"] = angles_acc
        accessory_data["rho_n"] = rho_n_acc
        accessory_data["rho"] = rho_acc
        accessory_data["radius"] = acc_intpl["radius"]   # <-- keep canonicalV
        accessory_data["frame"] = acc_intpl["frame"]

        return accessory_data, avatar_data, inside


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
            rs_ts = np.stack([
                np.interp(ts, self.key_ts, key_radius[:, 0]),
                np.interp(ts, self.key_ts, key_radius[:, 1])
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

    def map_coords_by_arclen(self, coords_src):
        # coords_src are in [0,1]
        a = np.interp(coords_src, ts_src, A_src)     # src ts -> src arc fraction
        coords_tgt = np.interp(a, A_tgt, ts_tgt)     # target arc fraction -> target ts
        return coords_tgt

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
    
