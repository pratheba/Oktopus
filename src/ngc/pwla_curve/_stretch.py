# Auto-split from PWLA_curve_handle.py -- _StretchMixin
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



class _StretchMixin:

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

    def stretch_end_extension(self, stretch_arg):
        """
        Extend the tail [t0, 1.0] outward.
        Prefix ts < t0 stays unchanged.
        """
        points = self.key_points.copy()
        ts = self.key_ts.copy()

        stretch_scale = float(stretch_arg.get('stretch_scale', stretch_arg.get('length', 1.0)))
        t0 = float(stretch_arg['t0'])

        mask = ts >= t0
        if not np.any(mask):
            return points

        p0 = self.interpolate(np.array([t0]), radius=False, frame=False)['points'][0]
        p1 = self.interpolate(np.array([1.0]), radius=False, frame=False)['points'][0]

        tail_vec = p1 - p0
        tail_len = np.linalg.norm(tail_vec) + 1e-12
        t_dir = tail_vec / tail_len

        d = points[mask] - p0[None, :]
        w = d @ t_dir
        yz = d - np.outer(w, t_dir)

        w_new = stretch_scale * w
        points[mask] = p0[None, :] + np.outer(w_new, t_dir) + yz
        return points


    def stretch_start_extension(self, stretch_arg):
        """
        Extend the head [0.0, t1] outward.
        Suffix ts > t1 stays unchanged.
        """
        points = self.key_points.copy()
        ts = self.key_ts.copy()

        stretch_scale = float(stretch_arg.get('stretch_scale', stretch_arg.get('length', 1.0)))
        t1 = float(stretch_arg['t1'])

        mask = ts <= t1
        if not np.any(mask):
            return points

        p0 = self.interpolate(np.array([0.0]), radius=False, frame=False)['points'][0]
        p1 = self.interpolate(np.array([t1]), radius=False, frame=False)['points'][0]

        head_vec = p1 - p0
        head_len = np.linalg.norm(head_vec) + 1e-12
        t_dir = head_vec / head_len

        # anchor at t1 so the interior side stays attached
        d = points[mask] - p1[None, :]
        w = d @ t_dir
        yz = d - np.outer(w, t_dir)

        w_new = stretch_scale * w
        points[mask] = p1[None, :] + np.outer(w_new, t_dir) + yz
        return points



    def stretch_interval_smooth_nonuniform(self, stretch_arg):
        points = self.key_points.copy()
        ts = self.key_ts.copy()

        direction = stretch_arg.get("direction", "forward")
        stretch_scale = float(stretch_arg.get('stretch_scale', stretch_arg.get('length', 1.0)))
        t0 = float(stretch_arg['t0'])
        t1 = float(stretch_arg['t1'])
        anchor = stretch_arg.get('anchor', 'start')

        if t1 <= t0:
            return points

        # choose anchor coord inside interval
        if anchor == 'start':
            s_anchor = t0
        elif anchor == 'end':
            s_anchor = t1
        elif anchor == 'coord':
            s_anchor = float(stretch_arg.get('anchor_coord', t0))
            s_anchor = np.clip(s_anchor, t0, t1)
        else:
            s_anchor = t0

        # interval endpoints and anchor in world space
        p0 = self.interpolate(np.array([t0]), radius=False, frame=False)['points'][0]
        p1 = self.interpolate(np.array([t1]), radius=False, frame=False)['points'][0]
        pa = self.interpolate(np.array([s_anchor]), radius=False, frame=False)['points'][0]

        interval_vec = p1 - p0
        interval_len = np.linalg.norm(interval_vec) + 1e-12
        t_dir = interval_vec / interval_len

        mid_mask = np.logical_and(ts >= t0, ts <= t1)
        if not np.any(mid_mask):
            return points

        # keep your existing interval stretch block
        d_mid = points[mid_mask] - pa[None, :]
        w_mid = d_mid @ t_dir
        yz_mid = d_mid - np.outer(w_mid, t_dir)

        w_mid_new = stretch_scale * w_mid
        points[mid_mask] = pa[None, :] + np.outer(w_mid_new, t_dir) + yz_mid

        # ONLY change propagation: move propagated side along local tangent per point
        delta_len = (stretch_scale - 1.0) * interval_len

        if direction == "forward":
            propagate_mask = ts > t1
            signed_delta = delta_len
        elif direction == "backward":
            propagate_mask = ts < t0
            signed_delta = -delta_len
        else:
            raise ValueError(f"Unknown direction: {direction}")

        if np.any(propagate_mask):
            intpl_prop = self.interpolate(ts[propagate_mask], radius=False, frame=True)
            frame_prop = intpl_prop["frame"]   # rows [T,N,B]
            T_prop = frame_prop[:, 0, :]       # tangent at each propagated point
            points[propagate_mask] = points[propagate_mask] + signed_delta * T_prop

        return points


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
