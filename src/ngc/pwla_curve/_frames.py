# Auto-split from PWLA_curve_handle.py -- _FramesMixin
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



class _FramesMixin:


    def apply_rotation(self, anchor, rot):
        self.key_points = rot.apply(self.key_points - anchor) + anchor
        self.z_axis = rot.apply(self.z_axis)
        self.flag_points = True

    def estimate_tangent(self, points, eps=1e-10):
        edge_vec = points[1:] - points[:-1]
        edge_len = np.linalg.norm(edge_vec, axis=1, keepdims=True)
        edge_dir = edge_vec / (edge_len + eps)

        if edge_dir.shape[0] == 1:
            return np.repeat(edge_dir, 2, axis=0)

        tan_start = edge_dir[0]
        tan_end = edge_dir[-1]

        tans = 0.5 * (edge_dir[1:] + edge_dir[:-1])
        tans_norm = np.linalg.norm(tans, axis=1, keepdims=True)

        bad = tans_norm[:, 0] < eps
        if np.any(bad):
            tans[bad] = edge_dir[:-1][bad]
            tans_norm = np.linalg.norm(tans, axis=1, keepdims=True)

        tans = tans / (tans_norm + eps)

        vert_tan = np.concatenate([
            tan_start.reshape(1, 3),
            tans,
            tan_end.reshape(1, 3)
        ], axis=0)

        return vert_tan


#    def estimate_tangent(self, points):
#        edge_vec = points[1:] - points[:-1]
#        edge_vec /= (np.linalg.norm(edge_vec, axis=1, keepdims=True) + 1e-12)
#
#        if edge_vec.shape[0] > 1:
#            tan_start = edge_vec[0]
#            tan_end = edge_vec[-1]
#            tans = (edge_vec[1:] + edge_vec[:-1]) / 2.
#            tans /= np.linalg.norm(tans, axis=1, keepdims=True)
#
#            vert_tan = np.concatenate([
#                tan_start.reshape(1,3), 
#                tans, 
#                tan_end.reshape(1,3)
#            ], axis=0)
#        else:
#            vert_tan = np.tile(edge_vec, (2,1))
#
#        return vert_tan
    
    def project_z_axis(self, x_axis, z_axis):
        dots = np.sum(x_axis*z_axis, axis=1)
        if not np.allclose(dots, 0):
            # project z_axis to x_axis
            z_axis = z_axis - dots[:, None]* x_axis
            # NOTE: huh???? forgot this
            z_axis /= np.linalg.norm(z_axis, axis=1, keepdims=True)

        return z_axis
    
#    def propagate_z_axis(self, x_axis, z_axis0):
#        final_z = []
#        # current z_axis
#        c_zx = z_axis0 
#        for i in range(x_axis.shape[0]):
#            xx = x_axis[i]
#            zx = c_zx - (xx @ c_zx)*xx
#            zx /= np.linalg.norm(zx)
#            final_z.append(zx)
#            c_zx = zx
#
#        return np.asarray(final_z)

    def propagate_z_axis(self, x_axis, z_axis0):
        final_z = []
        c_zx = z_axis0.astype(np.float64).copy()

        for i in range(x_axis.shape[0]):
            xx = x_axis[i]
            xx = xx / (np.linalg.norm(xx) + 1e-12)

            zx = c_zx - (xx @ c_zx) * xx
            nz = np.linalg.norm(zx)

            if nz < 1e-12:
                # fallback reference not parallel to tangent
                if abs(xx[2]) < 0.9:
                    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                else:
                    ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

                zx = ref - (ref @ xx) * xx
                nz = np.linalg.norm(zx)

            zx /= (nz + 1e-12)
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

    def _interp_frames(self, s_src, F_src, s_q):
        s_src = np.asarray(s_src, dtype=np.float64)
        F_src = np.asarray(F_src, dtype=np.float64)
        s_q = np.asarray(s_q, dtype=np.float64)

        out = np.zeros((len(s_q), 3, 3), dtype=np.float64)
        for a in range(3):
            for b in range(3):
                out[:, a, b] = np.interp(s_q, s_src, F_src[:, a, b])

        for i in range(len(s_q)):
            T = out[i, 0]
            N = out[i, 1]

            T = T / (np.linalg.norm(T) + 1e-12)
            N = N - np.dot(N, T) * T
            N = N / (np.linalg.norm(N) + 1e-12)
            B = np.cross(T, N)
            B = B / (np.linalg.norm(B) + 1e-12)

            out[i] = np.stack([T, N, B], axis=0)
        return out


    def rotate_frames_about_tangent(self, frames, angle_rad):
        """
        frames: (N,3,3) with rows [T,N,B]
        rotate N,B around T by angle_rad
        """
        frames = np.asarray(frames, dtype=np.float64).copy()

        c = np.cos(angle_rad)
        s = np.sin(angle_rad)

        T = frames[:, 0, :]
        N = frames[:, 1, :]
        B = frames[:, 2, :]

        N_new = c * N + s * B
        B_new = -s * N + c * B

        out = frames.copy()
        out[:, 0, :] = T
        out[:, 1, :] = N_new
        out[:, 2, :] = B_new
        return out
