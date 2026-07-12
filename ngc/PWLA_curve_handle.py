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

    def _as_radius2(self, x, name):
        x = np.asarray(x, dtype=np.float64)

        if x.ndim == 1:
            # scalar radius per keypoint -> [r, r]
            x = np.stack([x, x], axis=1)

        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError(f"{name} must have shape (K,2) or (K,), got {x.shape}")

        return x

    def build_circular_envelope_wrap_from_points(
        self,
        points,
        n_s=64,
        n_theta=64,
        q=0.995,
        clearance=0.03,
        flare_strength=0.4,
        flare_start=0.35,
        flare_end=1.0,
        smooth_s=4.0,
    ):
        """
        Build theta-independent skirt envelope wrap radius from avatar/world points.
        Reuses this curve's projection/interpolate/frame.
        Returns wrap field shaped (n_s, n_theta).
        """

        points = np.asarray(points, dtype=np.float64)

        s = self.curve_projection(points, outside=True)
        valid = (s >= 0.0) & (s <= 1.0)

        points = points[valid]
        s = s[valid]

        intpl = self.interpolate(s, radius=False, frame=True)
        C = intpl["points"]
        F = intpl["frame"]

        local = np.einsum("nij,nj->ni", F, points - C)
        u = local[:, 1]
        v = local[:, 2]
        rho = np.sqrt(u * u + v * v)

        edges = np.linspace(0.0, 1.0, n_s + 1)
        s_bins = 0.5 * (edges[:-1] + edges[1:])

        r = np.full(n_s, np.nan, dtype=np.float64)

        for i in range(n_s):
            m = (s >= edges[i]) & (s < edges[i + 1])
            if np.any(m):
                r[i] = np.quantile(rho[m], q)

        good = np.isfinite(r)
        if not np.any(good):
            raise RuntimeError("No valid points for skirt envelope.")

        r[~good] = np.interp(s_bins[~good], s_bins[good], r[good])

        if smooth_s > 0:
            r = gaussian_filter1d(r, sigma=float(smooth_s), mode="nearest")

        r = r + clearance

        # flare
        t = np.clip((s_bins - flare_start) / (flare_end - flare_start + 1e-12), 0.0, 1.0)
        w = t * t * (3.0 - 2.0 * t)
        r = r * (1.0 + flare_strength * w)

        theta_bins = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)

        # circular envelope: same radius for every theta
        wrap = np.repeat(r[:, None], n_theta, axis=1)

        return {
            "key_wrap_radius": wrap,
            "wrap_s_bins": s_bins,
            "wrap_theta_bins": theta_bins,
            "r_env": r,
        }

    def export_wrap_field_ply_with_control(
        self,
        wrap,
        s_bins,
        theta_bins,
        out_path,
        n_control=12,
        smooth_points_sigma=8.0,
        smooth_radius_sigma=0.0,
        rebuild_frames=True,
        preserve_endpoints=True,
        stride=1,
    ):
        wrap = np.asarray(wrap, dtype=np.float64)
        s_bins = np.asarray(s_bins, dtype=np.float64)
        theta_bins = np.asarray(theta_bins, dtype=np.float64)

        ctrl = self.build_adapt_control_field(
            s_bins,
            n_control=int(n_control),
            smooth_points_sigma=float(smooth_points_sigma),
            smooth_radius_sigma=float(smooth_radius_sigma),
            rebuild_frames=bool(rebuild_frames),
            preserve_endpoints=bool(preserve_endpoints),
            radius_type="train",
        )

        pts_all = []

        for i in range(0, len(s_bins), max(1, stride)):
            C = ctrl["points"][i]
            F = ctrl["frame"][i]

            U = F[1]
            V = F[2]
            r = wrap[i]

            pts = (
                C[None, :]
                + (r * np.cos(theta_bins))[:, None] * U[None, :]
                + (r * np.sin(theta_bins))[:, None] * V[None, :]
            )
            pts_all.append(pts)

        P = np.vstack(pts_all)
        P = P[np.isfinite(P).all(axis=1)]

        colors = np.tile(np.array([120, 220, 40, 255], dtype=np.uint8), (len(P), 1))
        trimesh.points.PointCloud(P, colors=colors).export(out_path)
        print("[debug wrap control ply]", out_path)

    def export_adapt_control_rings_ply(
        self,
        out_path,
        coords_query,
        n_control=6,
        smooth_points_sigma=2.0,
        smooth_radius_sigma=2.0,
        rebuild_frames=True,
        preserve_endpoints=True,
        radius_type="train",
        n_theta=96,
        stride=1,
    ):
        """
        Export the actual smoothed control-field rings.

        This shows what build_adapt_control_field() is producing:
            points + frames + radius

        Use radius_type="train" to see the model/rigid radius.
        Use radius_type="cylinder" to see the support/cylinder radius.
        """
        coords_query = np.asarray(coords_query, dtype=np.float64).reshape(-1)

        ctrl = self.build_adapt_control_field(
            coords_query,
            n_control=int(n_control),
            smooth_points_sigma=float(smooth_points_sigma),
            smooth_radius_sigma=float(smooth_radius_sigma),
            rebuild_frames=bool(rebuild_frames),
            preserve_endpoints=bool(preserve_endpoints),
            radius_type=radius_type,
        )

        theta = np.linspace(0.0, 2.0 * np.pi, int(n_theta), endpoint=False)

        pts_all = []
        for i in range(0, len(coords_query), max(1, int(stride))):
            C = ctrl["points"][i]
            F = ctrl["frame"][i]
            ry, rz = ctrl["radius"][i]

            U = F[1]
            V = F[2]

            ring = (
                C[None, :]
                + (ry * np.cos(theta))[:, None] * U[None, :]
                + (rz * np.sin(theta))[:, None] * V[None, :]
            )
            pts_all.append(ring)

        P = np.vstack(pts_all)
        P = P[np.isfinite(P).all(axis=1)]

        colors = np.tile(
            np.array([0, 180, 255, 255], dtype=np.uint8),
            (P.shape[0], 1),
        )

        trimesh.points.PointCloud(P, colors=colors).export(out_path)

        print(
            "[export_adapt_control_rings]",
            out_path,
            "radius_type=", radius_type,
            "n_control=", int(n_control),
            "smooth_points_sigma=", float(smooth_points_sigma),
            "smooth_radius_sigma=", float(smooth_radius_sigma),
            "points=", P.shape[0],
            "radius min/mean/max=",
            float(np.min(ctrl["radius"])),
            float(np.mean(ctrl["radius"])),
            float(np.max(ctrl["radius"])),
        )

        return ctrl



    def export_wrap_field_ply(self, wrap, s_bins, theta_bins, out_path, stride=1):
        wrap = np.asarray(wrap, dtype=np.float64)
        s_bins = np.asarray(s_bins, dtype=np.float64)
        theta_bins = np.asarray(theta_bins, dtype=np.float64)

        pts_all = []

        for i in range(0, len(s_bins), max(1, stride)):
            s = float(s_bins[i])
            intpl = self.interpolate(np.array([s]), radius=False, frame=True)
            C = intpl["points"][0]
            F = intpl["frame"][0]
            U = F[1]
            V = F[2]

            r = wrap[i]
            pts = (
                C[None, :]
                + (r * np.cos(theta_bins))[:, None] * U[None, :]
                + (r * np.sin(theta_bins))[:, None] * V[None, :]
            )
            pts_all.append(pts)

        P = np.vstack(pts_all)
        P = P[np.isfinite(P).all(axis=1)]

        colors = np.tile(np.array([255, 40, 40, 255], dtype=np.uint8), (len(P), 1))
        trimesh.points.PointCloud(P, colors=colors).export(out_path)
        print("[debug wrap ply]", out_path)


    def compute_valid_wrap_interval(self,wrap_counts, wrap_s_bins, min_count=10, margin=0.02):
        counts = np.asarray(wrap_counts)
        s_bins = np.asarray(wrap_s_bins)

        occ = counts.sum(axis=1)
        valid = occ > min_count

        if not np.any(valid):
            return 0.0, 1.0

        ids = np.where(valid)[0]
        s0 = float(s_bins[ids[0]])
        s1 = float(s_bins[ids[-1]])

        s0 = max(0.0, s0 - margin)
        s1 = min(1.0, s1 + margin)

        return s0, s1

    def smooth_periodic_theta(self,arr, sigma):
        """
        Smooth a (K, T) wrap-radius field along the theta dimension periodically.

        arr:
            shape (K, T)
            K = number of curve/keypoint sections
            T = number of theta bins

        sigma:
            Gaussian sigma in theta-bin units.
            sigma=0 disables smoothing.

        Returns:
            smoothed array with same shape (K, T)
        """
        arr = np.asarray(arr, dtype=np.float64)

        if sigma is None or sigma <= 0:
            return arr.copy()

        n_theta = arr.shape[1]

        # periodic padding by tiling theta dimension
        ext = np.concatenate([arr, arr, arr], axis=1)

        # smooth along theta axis
        ext = gaussian_filter1d(ext, sigma=sigma, axis=1)

        # take the middle copy
        return ext[:, n_theta:2 * n_theta]

    def rebin_wrap_to_control_curve(
        self,
        wrap,
        s_bins,
        theta_bins,
        n_control=12,
        smooth_points_sigma=8.0,
        rebuild_frames=True,
        n_lookup=128,
        q=1.0,
        smooth_s=2.0,
        smooth_theta=2.0,
        radius_margin=0.01,
    ):
        wrap = np.asarray(wrap, dtype=np.float64)
        s_bins = np.asarray(s_bins, dtype=np.float64)
        theta_bins = np.asarray(theta_bins, dtype=np.float64)

        Nt = len(theta_bins)
        s_lookup = np.linspace(0.0, 1.0, int(n_lookup))

        # Smooth/control curve sampled at lookup resolution
        ctrl = self.build_adapt_control_field(
            s_lookup,
            n_control=int(n_control),
            smooth_points_sigma=float(smooth_points_sigma),
            smooth_radius_sigma=0.0,
            rebuild_frames=bool(rebuild_frames),
            preserve_endpoints=True,
            radius_type="train",
        )

        C_ctrl = ctrl["points"]
        F_ctrl = ctrl["frame"]

        # Old wrap rings -> world points
        world_pts = []
        for i, s in enumerate(s_bins):
            old = self.interpolate(np.array([s]), radius=False, frame=True)
            C = old["points"][0]
            F = old["frame"][0]
            U, V = F[1], F[2]

            r = wrap[i]
            P = (
                C[None, :]
                + (r * np.cos(theta_bins))[:, None] * U[None, :]
                + (r * np.sin(theta_bins))[:, None] * V[None, :]
            )
            world_pts.append(P)

        P = np.vstack(world_pts)

        # TRUE rebin: assign every old-ring world point to nearest point on smooth curve
        tree = KDTree(C_ctrl)
        _, sid = tree.query(P)

        C = C_ctrl[sid]
        F = F_ctrl[sid]

        local = np.einsum("nij,nj->ni", F, P - C)
        u = local[:, 1]
        v = local[:, 2]

        rho = np.sqrt(u * u + v * v)
        theta = np.arctan2(v, u)

        theta0 = theta_bins[0]
        dtheta = 2.0 * np.pi / Nt
        tid = np.floor(((theta - theta0) % (2.0 * np.pi)) / dtheta).astype(np.int64) % Nt

        new_wrap = np.full((len(s_lookup), Nt), np.nan, dtype=np.float64)

        for i in range(len(s_lookup)):
            m_s = sid == i
            if not np.any(m_s):
                continue

            for j in range(Nt):
                m = m_s & (tid == j)
                if np.any(m):
                    if q >= 1.0:
                        new_wrap[i, j] = np.max(rho[m])
                    else:
                        new_wrap[i, j] = np.quantile(rho[m], q)

        # Fill missing theta bins periodically per row
        x = np.arange(Nt)
        for i in range(len(s_lookup)):
            row = new_wrap[i]
            valid = np.isfinite(row)

            if np.any(valid):
                xv = x[valid]
                yv = row[valid]

                # periodic fill
                xv_ext = np.concatenate([xv - Nt, xv, xv + Nt])
                yv_ext = np.concatenate([yv, yv, yv])
                row[~valid] = np.interp(x[~valid], xv_ext, yv_ext)
            else:
                new_wrap[i, :] = np.nan

        # Fill missing s rows
        for j in range(Nt):
            col = new_wrap[:, j]
            valid = np.isfinite(col)
            if np.any(valid):
                new_wrap[:, j] = np.interp(
                    np.arange(len(s_lookup)),
                    np.where(valid)[0],
                    col[valid],
                )
            else:
                new_wrap[:, j] = np.nanmedian(wrap)

        new_wrap = np.maximum(new_wrap + float(radius_margin), 1e-6)

        if smooth_theta > 0:
            new_wrap = self.smooth_periodic_theta(new_wrap, sigma=float(smooth_theta))

        if smooth_s > 0:
            new_wrap = gaussian_filter1d(
                new_wrap,
                sigma=float(smooth_s),
                axis=0,
                mode="nearest",
            )

        return new_wrap, s_lookup



    def rebin_wrap_to_control_curve1(
        self,
        wrap,
        s_bins,
        theta_bins,
        n_control=12,
        smooth_points_sigma=8.0,
        rebuild_frames=True,
        n_lookup=128,
        q=0.98,
        smooth_s=2.0,
        smooth_theta=2.0,
    ):
        # 1. Build smooth/control curve where wrap should live
        ctrl = self.build_adapt_control_field(
            np.asarray(s_bins, dtype=np.float64),
            n_control=n_control,
            smooth_points_sigma=smooth_points_sigma,
            smooth_radius_sigma=0.0,
            rebuild_frames=rebuild_frames,
            preserve_endpoints=True,
            radius_type="train",
        )

        # 2. Convert old wrap rings into world points using OLD centers/frames
        world_pts = []
        for i, s in enumerate(s_bins):
            old = self.interpolate(np.array([s]), radius=False, frame=True)
            C = old["points"][0]
            F = old["frame"][0]
            U, V = F[1], F[2]
            r = wrap[i]

            P = (
                C[None, :]
                + (r * np.cos(theta_bins))[:, None] * U[None, :]
                + (r * np.sin(theta_bins))[:, None] * V[None, :]
            )
            world_pts.append(P)

        world_pts = np.vstack(world_pts)

        # 3. Re-measure those world points in the CONTROL frame
        s_query = np.asarray(s_bins, dtype=np.float64)
        ctrl_points = ctrl["points"]
        ctrl_frames = ctrl["frame"]

        # nearest by source s row; simple version: reuse same row index
        new_wrap = np.zeros_like(wrap)

        for i, s in enumerate(s_bins):
            P = world_pts[i * len(theta_bins):(i + 1) * len(theta_bins)]

            C = ctrl_points[i]
            F = ctrl_frames[i]
            local = np.einsum("ij,nj->ni", F, P - C[None, :])
            u = local[:, 1]
            v = local[:, 2]
            th = np.arctan2(v, u)
            rho = np.sqrt(u * u + v * v)

            # bin by theta in control frame
            Nt = len(theta_bins)
            theta0 = theta_bins[0]
            dtheta = 2.0 * np.pi / Nt
            jj = np.floor(((th - theta0) % (2.0 * np.pi)) / dtheta).astype(int) % Nt

            row = np.full(Nt, np.nan)
            for j in range(Nt):
                m = jj == j
                if np.any(m):
                    row[j] = np.quantile(rho[m], q)

            valid = np.isfinite(row)
            if np.any(valid):
                x = np.arange(Nt)
                row[~valid] = np.interp(x[~valid], x[valid], row[valid])
            else:
                row[:] = np.mean(rho)

            new_wrap[i] = row

        if smooth_theta > 0:
            new_wrap = self.smooth_periodic_theta(new_wrap, smooth_theta)

        if smooth_s > 0:
            new_wrap = gaussian_filter1d(new_wrap, sigma=smooth_s, axis=0, mode="nearest")

        # 4. Optional final low-frequency lookup
        new_wrap, new_s = self.smooth_downsample_wrap_for_adapt(
            new_wrap,
            s_bins,
            n_adapt=n_control,
            smooth_s=smooth_s,
            smooth_theta=smooth_theta,
            n_lookup=n_lookup,
        )

        return new_wrap, new_s





    def smooth_downsample_wrap_for_adapt(
        self,
        wrap,
        s_bins,
        n_adapt=4,
        smooth_s=8.0,
        smooth_theta=4.0,
        n_lookup=128,
    ):
        wrap = np.asarray(wrap, dtype=np.float64)
        s_bins = np.asarray(s_bins, dtype=np.float64)

        wrap_smooth = wrap.copy()

        if smooth_theta > 0:
            wrap_smooth = self.smooth_periodic_theta(wrap_smooth, sigma=smooth_theta)

        if smooth_s > 0:
            wrap_smooth = gaussian_filter1d(
                wrap_smooth,
                sigma=float(smooth_s),
                axis=0,
                mode="nearest",
            )

        # Low-frequency control points
        s_ctrl = np.linspace(0.0, 1.0, int(n_adapt))

        #wrap_ctrl = np.zeros((int(n_adapt), wrap.shape[1]), dtype=np.float64)
        #for j in range(wrap.shape[1]):
        #    wrap_ctrl[:, j] = np.interp(s_ctrl, s_bins, wrap_smooth[:, j])
        wrap_ctrl = np.zeros((int(n_adapt), wrap.shape[1]), dtype=np.float64)

        # local pooling window around each adapt keypoint
        pool_width = 1.0 / max(int(n_adapt) - 1, 1)
        pool_sigma = 0.5 * pool_width

        for k, sc in enumerate(s_ctrl):
            d = np.abs(s_bins - sc)

            # use Gaussian weights around this adapt keypoint
            w = np.exp(-0.5 * (d / (pool_sigma + 1e-12)) ** 2)

            # optional: limit to local neighborhood
            w[d > pool_width] = 0.0

            if np.sum(w) < 1e-12:
                idx = int(np.argmin(d))
                wrap_ctrl[k] = wrap_smooth[idx]
            else:
                w = w / np.sum(w)
                wrap_ctrl[k] = np.sum(w[:, None] * wrap_smooth, axis=0)



        # Re-densify smoothly for lookup
        s_lookup = np.linspace(0.0, 1.0, int(n_lookup))
        wrap_lookup = np.zeros((int(n_lookup), wrap.shape[1]), dtype=np.float64)

        for j in range(wrap.shape[1]):
            if int(n_adapt) >= 4:
                f = PchipInterpolator(s_ctrl, wrap_ctrl[:, j])
                wrap_lookup[:, j] = f(s_lookup)
            else:
                wrap_lookup[:, j] = np.interp(s_lookup, s_ctrl, wrap_ctrl[:, j])

        wrap_lookup = np.maximum(wrap_lookup, 1e-6)

        return wrap_lookup, s_lookup



    def smooth_downsample_wrap_for_adapt1(
        self,
        wrap,
        s_bins,
        n_adapt=12,
        smooth_s=2.0,
        smooth_theta=1.0,
    ):
        wrap = np.asarray(wrap, dtype=np.float64)
        s_bins = np.asarray(s_bins, dtype=np.float64)

        # 1. Smooth full-resolution wrap first
        wrap_smooth = wrap.copy()

        if smooth_theta > 0:
            wrap_smooth = self.smooth_periodic_theta(wrap_smooth, sigma=smooth_theta)

        if smooth_s > 0:
            wrap_smooth = gaussian_filter1d(wrap_smooth, sigma=smooth_s, axis=0)

        # 2. Downsample along s
        s_new = np.linspace(0.0, 1.0, n_adapt)

        wrap_new = np.zeros((n_adapt, wrap.shape[1]), dtype=np.float64)
        for j in range(wrap.shape[1]):
            wrap_new[:, j] = np.interp(s_new, s_bins, wrap_smooth[:, j])

        return wrap_new, s_new

    def smooth_adapt_avatar_fields(
        self,
        avatar_coords,
        avatar_world_points,
        avatar_world_frames,
        sigma_points=1.5,
        sigma_coords=0.0,
        rebuild_frames=True,
    ):
        avatar_coords = np.asarray(avatar_coords, dtype=np.float64)
        avatar_world_points = np.asarray(avatar_world_points, dtype=np.float64)
        avatar_world_frames = np.asarray(avatar_world_frames, dtype=np.float64)

        pts = avatar_world_points.copy()
        coords = avatar_coords.copy()

        if sigma_points > 0:
            pts = gaussian_filter1d(
                pts,
                sigma=float(sigma_points),
                axis=0,
                mode="nearest",
            )
            pts[0] = avatar_world_points[0]
            pts[-1] = avatar_world_points[-1]

        if sigma_coords > 0:
            coords = gaussian_filter1d(
                coords,
                sigma=float(sigma_coords),
                mode="nearest",
            )
            coords = np.clip(coords, avatar_coords.min(), avatar_coords.max())

        if rebuild_frames:
            frames = self.get_new_frame(pts)
        else:
            frames = avatar_world_frames.copy()

        return coords, pts, frames


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


    def build_runtime_uv_center_field(
        self,
        n_bins=64,
        source="owned",
        min_count=20,
        smooth_s=2.0,
        robust="median",
    ):
        """
        Estimate per-s cross-section UV center from saved surface points.

        Returns:
            {
                "s_bins": (n_bins,),
                "center_uv": (n_bins, 2), columns [cu, cv],
                "count": (n_bins,)
            }

        This is runtime-only. It does not modify the curve.
        """
        if source == "all":
            pts = self.surface_points_all
        else:
            pts = self.surface_points_owned

        if pts is None:
            print("[uv_center] no surface points available")
            return None

        pts = np.asarray(pts, dtype=np.float64)

        # Reproject at runtime, so we are not dependent on saved point_s quality.
        s = self.curve_projection(pts, outside=False)
        valid = (s >= 0.0) & (s <= 1.0)

        pts = pts[valid]
        s = s[valid]

        if pts.shape[0] < max(32, min_count):
            print("[uv_center] too few valid points:", pts.shape[0])
            return None

        intpl = self.interpolate(s, radius=False, frame=True)
        C = intpl["points"]
        F = intpl["frame"]

        local = np.einsum("nij,nj->ni", F, pts - C)
        u = local[:, 1]
        v = local[:, 2]

        s_bins = np.linspace(0.0, 1.0, int(n_bins))
        edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
        bid = np.clip(np.digitize(s, edges) - 1, 0, int(n_bins) - 1)

        center = np.full((int(n_bins), 2), np.nan, dtype=np.float64)
        count = np.zeros((int(n_bins),), dtype=np.int32)

        for i in range(int(n_bins)):
            m = bid == i
            count[i] = int(np.sum(m))
            if count[i] < int(min_count):
                continue

            if robust == "mean":
                center[i, 0] = float(np.mean(u[m]))
                center[i, 1] = float(np.mean(v[m]))
            else:
                center[i, 0] = float(np.median(u[m]))
                center[i, 1] = float(np.median(v[m]))

        # Fill invalid bins along s
        x = np.arange(int(n_bins))
        for j in range(2):
            col = center[:, j]
            valid_col = np.isfinite(col)
            if np.any(valid_col):
                center[:, j] = np.interp(x, x[valid_col], col[valid_col])
            else:
                center[:, j] = 0.0

        if smooth_s and smooth_s > 0:
            center[:, 0] = gaussian_filter1d(center[:, 0], sigma=float(smooth_s), mode="nearest")
            center[:, 1] = gaussian_filter1d(center[:, 1], sigma=float(smooth_s), mode="nearest")

        print(
            "[uv_center]",
            "source=", source,
            "points=", int(pts.shape[0]),
            "bins active=", int(np.sum(count >= int(min_count))), "/", int(n_bins),
            "cu min/mean/max=",
            float(np.min(center[:, 0])),
            float(np.mean(center[:, 0])),
            float(np.max(center[:, 0])),
            "cv min/mean/max=",
            float(np.min(center[:, 1])),
            float(np.mean(center[:, 1])),
            float(np.max(center[:, 1])),
        )

        return {
            "s_bins": s_bins,
            "center_uv": center,
            "count": count,
        }


    def interpolate_uv_center_field(self, field, s):
        if field is None:
            s = np.asarray(s, dtype=np.float64).reshape(-1)
            return np.zeros((len(s), 2), dtype=np.float64)

        s_bins = np.asarray(field["s_bins"], dtype=np.float64)
        center = np.asarray(field["center_uv"], dtype=np.float64)

        s = np.asarray(s, dtype=np.float64).reshape(-1)
        s_clip = np.clip(s, s_bins[0], s_bins[-1])

        cu = np.interp(s_clip, s_bins, center[:, 0])
        cv = np.interp(s_clip, s_bins, center[:, 1])

        return np.stack([cu, cv], axis=1)

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
                if outside:
                    samples3D_to_skeleton[sample_index] = non_uniform_linear_points[vid]

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
                if outside:
                    samples3D_to_skeleton[sample_index] = non_uniform_linear_points[vid]

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


    def localize_samples(self, pointcloudsamples, return_sdf=False, norm=1.0, update_curve=False, update_radius=False, outside=False, name='', radius_type='cylinder', runtime_cylinder_radius_scale=1.0, runtime_cylinder_radius_add=0.0):
        # Owned-volume gate: drop samples that fall outside the dilated voxel
        # mask of surface_points_owned BEFORE running the cylinder projection.
        # The returned `inside` indices still index into the ORIGINAL input.
        # No-op when the mask isn't built (owned points missing or gate disabled).

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

        # ------------------------------------------------------------
        # 1) Localize grid/world samples on avatar/source curve
        # ------------------------------------------------------------
        avatar_data, inside = self.localize_samples(
            vs,
            norm=float(adapt_arg.get("adapt_localize_norm", 1.0)),
            outside=False,
            runtime_cylinder_radius_scale=float(adapt_arg.get("avatar_cylinder_radius_scale", 1.0)),
            runtime_cylinder_radius_add=float(adapt_arg.get("avatar_cylinder_radius_add", 0.0)),
        )

        accessory_curve_handle = adapt_arg["accessory_curve_handle"]
        accessory_curve_handle.core.update_coords()
        accessory_curve_handle.core.update_frame()

        src_0 = float(adapt_arg["src_0"])
        src_1 = float(adapt_arg["src_1"])
        tgt_0 = float(adapt_arg["tgt_0"])
        tgt_1 = float(adapt_arg["tgt_1"])
        delta_theta = np.deg2rad(float(adapt_arg.get("rot_deg", 0.0)))

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
#        if use_adapt_control:
#            acc_ctrl = accessory_curve_handle.core.build_adapt_control_field(
#                acc_coords,
#                n_control=int(adapt_arg.get("adapt_control_n_keypoints", 12)),
#                smooth_points_sigma=float(adapt_arg.get("accessory_control_smooth_points_sigma", 1.0)),
#                smooth_radius_sigma=float(adapt_arg.get("accessory_control_smooth_radius_sigma", 1.0)),
#                rebuild_frames=bool(adapt_arg.get("accessory_control_rebuild_frames", True)),
#                preserve_endpoints=bool(adapt_arg.get("adapt_control_preserve_endpoints", True)),
#                radius_type="train",
#            )
#            acc_intpl = {
#                "points": acc_ctrl["points"],
#                "frame": acc_ctrl["frame"],
#                "radius": acc_ctrl["radius"],
#            }
#            tangent_acc = acc_ctrl["x_radius"]
#        else:
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
            #scale_rho = np.full_like(rho_avatar, global_scale)
            rigid_radial_scale = float(adapt_arg.get("rigid_radial_scale", 5.0))
            scale_rho = np.full_like(rho_avatar, rigid_radial_scale)
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
    
