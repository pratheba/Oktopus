# Auto-split from PWLA_curve_handle.py -- _CoreMixin
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



class _CoreMixin:
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

    def _as_radius2(self, x, name):
        x = np.asarray(x, dtype=np.float64)

        if x.ndim == 1:
            # scalar radius per keypoint -> [r, r]
            x = np.stack([x, x], axis=1)

        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError(f"{name} must have shape (K,2) or (K,), got {x.shape}")

        return x

    def set_curve(self, arg):
        self.name = arg.get("name", "")
        self.idx = arg.get("idx", -1)

        # Canonical preprocessed curve
        self.key_points = np.asarray(arg["keypoints"], dtype=np.float64)

        #total_points = self.key_points.shape[0]
        #curve_length, _ = self.calc_curve_length()
        #if arg['n_keypoints'] == 'all':
        #    K = len(self.key_points)
        #    self.n_sample_points = K
        #    r = np.linspace(0,total_points-1, self.n_sample_points, dtype=int) #np.random.randint(total_points-1, size=n_sample_points-2)
        #else:
        #    self.n_sample_points = int(arg['n_keypoints'] * curve_length)
        #    r = np.linspace(0,total_points-1, self.n_sample_points, dtype=int) #np.random.randint(total_points-1, size=n_sample_points-2)

        K = len(self.key_points)
        self.n_sample_points = K

        # Use precomputed radii directly
        self.key_train_radius = self._as_radius2(arg["radius_train"], "radius_train")
        self.key_cylinder_radius = self._as_radius2(
            arg.get("radius_cylinder", self.key_train_radius.copy()),
            "radius_cylinder"
        )


        if self.key_train_radius.shape[0] != K:
            raise ValueError(f"radius_train K mismatch: {self.key_train_radius.shape[0]} vs {K}")

        if self.key_cylinder_radius.shape[0] != K:
            raise ValueError(f"radius_cylinder K mismatch: {self.key_cylinder_radius.shape[0]} vs {K}")

        self.key_radius = self.key_train_radius

        # Runtime/inference support inflation.
        # This should NOT change training radius. It only affects cylinder filtering / MC support.
        shape_type = str(arg.get("type", "both")).lower()

        default_cyl_scale_by_type = {
            "avatar": 1.5,
            "accessory": 1.2,
            "both": 1.5,
        }

        default_cyl_add_by_type = {
            "avatar": 0.05,
            "accessory": 0.02,
            "both": 0.05,
        }

        cyl_scale = float(
            arg.get(
                "inference_cylinder_radius_scale",
                default_cyl_scale_by_type.get(shape_type, 1.0),
            )
        )

        cyl_add = float(
            arg.get(
                "inference_cylinder_radius_add",
                default_cyl_add_by_type.get(shape_type, 0.0),
            )
        )

        self.key_cylinder_radius = self.key_cylinder_radius * cyl_scale + cyl_add


        self.key_train_radius_adapt = self.smooth_resample_radius_for_adapt(
            self.key_train_radius,
            n_control=int(arg.get("adapt_train_radius_n_control", 64)),
            smooth_s=float(arg.get("adapt_train_radius_smooth_s", 1.0)),
            floor_ratio=float(arg.get("adapt_train_radius_floor", 0.85)),
        )
        self.key_radius_adapt = self.key_train_radius_adapt

        # Use precomputed frames if available
        if "frames" in arg:
            self.key_frame = np.asarray(arg["frames"], dtype=np.float64)
            self.z_axis = self.key_frame[:, 2, :]
        else:
            T = np.asarray(arg["frame_t"], dtype=np.float64)
            U = np.asarray(arg["frame_u"], dtype=np.float64)
            V = np.asarray(arg["frame_v"], dtype=np.float64)
            self.key_frame = np.stack([T, U, V], axis=1)
            self.z_axis = V

        if self.key_frame.shape[0] != K:
            raise ValueError(f"frames K mismatch: {self.key_frame.shape[0]} vs {K}")

        # Directional wrap
        self.radius_wrap = np.asarray(arg.get("radius_wrap", arg.get("wrap_radius_max", np.max(self.key_train_radius, axis=1))), dtype=np.float64)

        self.key_wrap_radius = arg.get("key_wrap_radius", None)
        if self.key_wrap_radius is not None:
            self.key_wrap_radius = np.asarray(self.key_wrap_radius, dtype=np.float64)

        self.wrap_s_bins = arg.get("wrap_s_bins", None)
        if self.wrap_s_bins is not None:
            self.wrap_s_bins = np.asarray(self.wrap_s_bins, dtype=np.float64)

        self.wrap_theta_bins = arg.get("wrap_theta_bins", None)
        if self.wrap_theta_bins is not None:
            self.wrap_theta_bins = np.asarray(self.wrap_theta_bins, dtype=np.float64)

        self.wrap_radius_max = arg.get("wrap_radius_max", None)
        if self.wrap_radius_max is not None:
            self.wrap_radius_max = np.asarray(self.wrap_radius_max, dtype=np.float64)


        self.key_wrap_radius_full = self.key_wrap_radius.copy()
        self.wrap_s_bins_full = self.wrap_s_bins.copy()
        n_adapt = int(arg.get("wrap_adapt_n_keypoints", 4))
        smooth_s = float(arg.get("wrap_adapt_smooth_s", 8.0))
        smooth_theta = float(arg.get("wrap_adapt_smooth_theta", 3.0))
        n_lookup = int(arg.get("wrap_adapt_lookup_n", 128))

#        if self.key_wrap_radius is not None:
#            self.key_wrap_radius_adapt, self.wrap_s_bins_adapt = self.smooth_downsample_wrap_for_adapt(
#                self.key_wrap_radius,
#                self.wrap_s_bins,
#                n_adapt=n_adapt,
#                smooth_s=smooth_s,
#                smooth_theta=smooth_theta,
#                n_lookup=n_lookup,
#            )
#        else:
#            self.key_wrap_radius_adapt = None
#            self.wrap_s_bins_adapt = None
        #print(self.key_wrap_radius_adapt.shape)
        #print(self.wrap_s_bins_adapt.shape)


        self.wrap_counts = arg.get("wrap_counts", None)

        self.valid_s0, self.valid_s1 = 0.0, 1.0

        if self.wrap_counts is not None and self.wrap_s_bins is not None:
            self.valid_s0, self.valid_s1 = self.compute_valid_wrap_interval(
                self.wrap_counts,
                self.wrap_s_bins,
                min_count=int(arg.get("valid_wrap_min_count", 10)),
                margin=float(arg.get("valid_wrap_margin", 0.02)),
            )

        print("[valid wrap interval]", self.name, self.valid_s0, self.valid_s1)

        # Optional saved surface info, useful for debugging only
        # Optional saved surface info
        self.surface_points_owned = arg.get("surface_points_owned", None)
        self.surface_points_all = arg.get("surface_points_all", None)
        self.surface_points_base = arg.get("surface_points_base", None)
        self.point_s = arg.get("point_s", None)
        self.point_key_ids = arg.get("point_key_ids", None)


        self.update_coords()

        # Do not immediately recompute frames from scratch.
        self.flag_points = False

    
    def set_curve_old(self, arg):
        # self.step = arg['resample_step']
        #self.key_points = arg['key_points']
        total_points = arg['keypoints'].shape[0]
        #r = np.sort(r)
        #r = r + 1
        #r = np.insert(r, 0, 0)
        #r = np.append(r, total_points-1)
        # NOTE: radius: (N, 2), y-z radius
        self.key_points = arg['keypoints']
        curve_length, _ = self.calc_curve_length()
        print("c length = ", curve_length, flush=True)
        ### Have length of 2 to have 36b key pints
        #n_sample_points = int(36 * curve_length)
        n_sample_points = int(arg['n_keypoints'] * curve_length)
        self.n_sample_points = int(arg['n_keypoints'] * curve_length)
        #n_sample_points = int(36 * curve_length)
        r = np.linspace(0,total_points-1, n_sample_points, dtype=int) #np.random.randint(total_points-1, size=n_sample_points-2)
        #self.key_points = arg['keypoints']
        #r = np.arange(total_points)
        self.key_points = self.key_points[r] 
        self.key_train_radius = np.array(arg['radius_train'])
        self.key_cylinder_radius = (np.array(arg.get('radius_cylinder', self.key_train_radius.copy())))
        #max_train_radius = np.max(self.key_train_radius)
        #max_cylinder_radius = np.max(self.key_cylinder_radius)
        #self.key_train_radius = np.full(self.key_train_radius.shape, max_train_radius)
        #self.key_cylinder_radius = np.full(self.key_cylinder_radius.shape, max_cylinder_radius)
        #self.key_train_radius = self.key_train_radius[r]
        shape_type = arg['type']
        if shape_type in ['avatar', 'both']:
            self.key_cylinder_radius = self.key_cylinder_radius[r]+0.2
        self.key_train_radius = self.key_train_radius[r]
        #self.key_train_radius = self.key_cylinder_radius
        self.key_train_radius = np.tile(np.array(self.key_train_radius),2).reshape(-1, self.key_train_radius.shape[0]).T 
        self.key_cylinder_radius = np.tile(np.array(self.key_cylinder_radius),2).reshape(-1, self.key_cylinder_radius.shape[0]).T
        #self.key_train_radius = arg.get('key_train_radius', arg.get('key_radius'))
        #self.key_cylinder_radius = arg.get('key_cylinder_radius', self.key_train_radius.copy())
        self.key_radius = self.key_train_radius
        x_axis = arg['frame_t'][r] #self.estimate_tangent(self.key_points)
        #x_axis = self.estimate_tangent(self.key_points)
        z_axis = arg['frame_v'][r]#arg['z_axis']
        #print(self.key_radius.shape)
        #print(self.key_radius.shape)

        self.radius_wrap = np.array(arg.get('radius_wrap', None))[r]
        self.key_wrap_radius = np.array(arg.get('key_wrap_radius', None))[r]
        #self.key_wrap_radius = np.tile(np.array(self.key_snug_radius),2).reshape(-1, self.key_snug_radius.shape[0]).T
        #print(self.key_wrap_radius)
        #print(self.key_train_radius)
        #exit()
        self.key_occupancy_rho = arg.get('key_occupancy_rho', None)
        self.wrap_s_bins = arg.get('wrap_s_bins', None)
        self.wrap_theta_bins = arg.get('wrap_theta_bins', None)
        self.wrap_radius_max = arg.get('wrap_radius_max', None)

        #x_axis = self.estimate_tangent(self.key_points)
        #x_axis = arg['frame_t'][r] #self.estimate_tangent(self.key_points)
        #z_axis = arg['frame_v'][r] #arg['z_axis']
        
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
        #print(new_ts)
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
        #print(new_key_wrap_radius)
        #exit()

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
        y_axis /= (np.linalg.norm(y_axis, axis=1, keepdims=True) + 1e-12)

        self.z_axis = np.cross(x_axis, y_axis)
        self.z_axis /= (np.linalg.norm(self.z_axis, axis=1, keepdims=True) + 1e-12)

        self.key_frame = np.concatenate([
            x_axis.reshape(-1,1,3),
            y_axis.reshape(-1,1,3),
            self.z_axis.reshape(-1,1,3)
        ], axis=1)
        #self.rotation = Rotation.from_matrix(self.key_frame)
        #self.rot_slerp = Slerp(self.key_ts, self.rotation)
        self.flag_points = True

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
        self.key_ts = np.cumsum(np.r_[0., edge_lengths]) / (self.curve_length + 1e-12)
