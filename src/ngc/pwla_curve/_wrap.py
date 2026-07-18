# Auto-split from PWLA_curve_handle.py -- _WrapMixin
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



class _WrapMixin:

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
