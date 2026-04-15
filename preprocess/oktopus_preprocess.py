#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import trimesh
except Exception:
    trimesh = None


Array = np.ndarray


def _safe_norm(x: Array, axis=None, keepdims=False, eps: float = 1e-12) -> Array:
    return np.sqrt(np.sum(x * x, axis=axis, keepdims=keepdims) + eps)


def _unit(v: Array, eps: float = 1e-12) -> Array:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _normalize(x: Array, axis: int = -1, eps: float = 1e-12) -> Array:
    return x / _safe_norm(x, axis=axis, keepdims=True, eps=eps)


def _quant_key(p: Array, decimals: int = 6) -> Tuple[float, float, float]:
    q = np.round(np.asarray(p, dtype=np.float64), decimals=decimals)
    return (float(q[0]), float(q[1]), float(q[2]))


def _gaussian_kernel1d(sigma: float, radius: int | None = None) -> Array:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float64)
    if radius is None:
        radius = int(max(1, round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= np.sum(k)
    return k


def _smooth1d(x: Array, sigma: float) -> Array:
    x = np.asarray(x, dtype=np.float64)
    if sigma <= 0:
        return x.copy()
    k = _gaussian_kernel1d(sigma)
    r = len(k) // 2
    if x.ndim == 1:
        xp = np.pad(x, (r, r), mode="edge")
        return np.convolve(xp, k, mode="valid")
    if x.ndim == 2:
        out = np.zeros_like(x)
        for d in range(x.shape[1]):
            xp = np.pad(x[:, d], (r, r), mode="edge")
            out[:, d] = np.convolve(xp, k, mode="valid")
        return out
    raise ValueError("Expected 1D or 2D input")


def _skew(v: Array) -> Array:
    x, y, z = v
    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ], dtype=np.float64)


def minimal_rotation_matrix(a: Array, b: Array, eps: float = 1e-12) -> Array:
    a = _normalize(np.asarray(a, dtype=np.float64)[None])[0]
    b = _normalize(np.asarray(b, dtype=np.float64)[None])[0]
    v = np.cross(a, b)
    c = np.clip(np.dot(a, b), -1.0, 1.0)
    s = _safe_norm(v)
    if s < eps:
        if c > 0.0:
            return np.eye(3, dtype=np.float64)
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(a[0]) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = _normalize(np.cross(a, tmp)[None])[0]
        K = _skew(axis)
        return np.eye(3) + 2.0 * (K @ K)
    axis = v / s
    theta = np.arccos(c)
    K = _skew(axis)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


@dataclass
class SegmentView:
    segment_id: int
    raw_segment_id: int
    node_ids: List[int]
    endpoints: Tuple[int, int]
    polyline_raw: Array
    polyline_center: Array
    polyline_wrap: Array
    arclength_raw: Array
    surface_points_owned: Array
    surface_points_shared: Array
    surface_points_candidate: Array
    point_s: Array
    point_key_ids: Array
    keypoints: Array
    key_ts: Array
    key_radius_train: Array
    key_radius_wrap: Array
    key_radius_uv: Array
    cylinder_radius: Array
    hub_score: float
    is_hub_fragment: bool


class SkeletonPreprocessor:
    def __init__(self, skel_file: str, corr_file: str, mesh_file: str | None = None, decimals: int = 6):
        self.skel_file = Path(skel_file)
        self.corr_file = Path(corr_file)
        self.mesh_file = Path(mesh_file) if mesh_file else None
        self.decimals = decimals

        self.mesh_vertices: Array | None = None
        self.mesh_faces: Array | None = None

        self.nodes_xyz: Array | None = None
        self.node_map: Dict[Tuple[float, float, float], int] = {}
        self.adj: Dict[int, set[int]] = defaultdict(set)
        self.edge_ids: List[Tuple[int, int]] = []
        self.degree: Dict[int, int] = {}
        self.segments: List[List[int]] = []
        self.node_to_segments: Dict[int, List[int]] = defaultdict(list)

        self.node_surface_points: Dict[int, Array] = {}
        self.all_surface_points: Array | None = None
        self.segment_candidates: Dict[int, Array] = {}
        self.segment_owned: Dict[int, Array] = {}
        self.segment_shared: Dict[int, Array] = {}
        self.corr_misses: int = 0

    # ---------------- parsing ----------------

    def load_mesh(self) -> None:
        if self.mesh_file is None:
            return
        if trimesh is None:
            raise ImportError("trimesh is required to load mesh files")
        mesh = trimesh.load(self.mesh_file, process=False)
        self.mesh_vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if hasattr(mesh, "faces") and mesh.faces is not None:
            self.mesh_faces = np.asarray(mesh.faces, dtype=np.int32)

    def _read_pairs_file(self, path: Path) -> List[Tuple[Array, Array]]:
        pairs = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                vals = list(map(float, parts[1:] if parts[0] == "2" else parts))
                if len(vals) != 6:
                    continue
                pairs.append((
                    np.asarray(vals[:3], dtype=np.float64),
                    np.asarray(vals[3:], dtype=np.float64),
                ))
        return pairs

    def build_graph(self) -> None:
        pairs = self._read_pairs_file(self.skel_file)
        xyz: List[Array] = []

        def get_id(p: Array) -> int:
            key = _quant_key(p, self.decimals)
            if key not in self.node_map:
                self.node_map[key] = len(xyz)
                xyz.append(np.asarray(key, dtype=np.float64))
            return self.node_map[key]

        for a, b in pairs:
            ia = get_id(a)
            ib = get_id(b)
            if ia == ib:
                continue
            self.adj[ia].add(ib)
            self.adj[ib].add(ia)
            self.edge_ids.append((ia, ib))

        self.nodes_xyz = np.asarray(xyz, dtype=np.float64)
        self.degree = {i: len(self.adj[i]) for i in self.adj}

    def extract_segments(self) -> None:
        degree = self.degree
        critical = {i for i, d in degree.items() if d != 2}
        visited_edges: set[Tuple[int, int]] = set()
        segments: List[List[int]] = []

        for u in critical:
            for v in list(self.adj[u]):
                e = tuple(sorted((u, v)))
                if e in visited_edges:
                    continue
                seg = [u, v]
                visited_edges.add(e)
                prev, cur = u, v
                while degree[cur] == 2:
                    nxts = [w for w in self.adj[cur] if w != prev]
                    if not nxts:
                        break
                    nxt = nxts[0]
                    e = tuple(sorted((cur, nxt)))
                    if e in visited_edges:
                        break
                    seg.append(nxt)
                    visited_edges.add(e)
                    prev, cur = cur, nxt
                segments.append(seg)

        # closed loops fallback
        for u in self.adj:
            for v in self.adj[u]:
                e = tuple(sorted((u, v)))
                if e in visited_edges:
                    continue
                seg = [u, v]
                visited_edges.add(e)
                prev, cur = u, v
                while True:
                    nxts = [w for w in self.adj[cur] if w != prev]
                    if not nxts:
                        break
                    nxt = nxts[0]
                    e = tuple(sorted((cur, nxt)))
                    if e in visited_edges:
                        break
                    seg.append(nxt)
                    visited_edges.add(e)
                    prev, cur = cur, nxt
                segments.append(seg)

        self.segments = segments
        self.node_to_segments = defaultdict(list)
        for sid, seg in enumerate(self.segments):
            for nid in seg:
                self.node_to_segments[nid].append(sid)

    def parse_correspondence(self) -> None:
        node_to_pts: Dict[int, List[Array]] = defaultdict(list)
        all_pts: List[Array] = []
        misses = 0
        with open(self.corr_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                vals = list(map(float, parts[1:] if parts[0] == "2" else parts))
                if len(vals) != 6:
                    continue
                sk = np.asarray(vals[:3], dtype=np.float64)
                sp = np.asarray(vals[3:], dtype=np.float64)
                nid = self.node_map.get(_quant_key(sk, self.decimals))
                if nid is None:
                    misses += 1
                    continue
                node_to_pts[nid].append(sp)
                all_pts.append(sp)

        self.node_surface_points = {
            nid: np.unique(np.asarray(pts, dtype=np.float64), axis=0)
            for nid, pts in node_to_pts.items()
        }
        self.all_surface_points = (
            np.unique(np.asarray(all_pts, dtype=np.float64), axis=0)
            if all_pts else np.zeros((0, 3), dtype=np.float64)
        )
        self.corr_misses = misses

    # ---------------- geometry helpers ----------------

    @staticmethod
    def arc_length(polyline: Array) -> Array:
        if len(polyline) == 0:
            return np.zeros((0,), dtype=np.float64)
        if len(polyline) == 1:
            return np.zeros((1,), dtype=np.float64)
        d = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(d)])

    @staticmethod
    def resample_polyline(polyline: Array, n_keypoints: int) -> Tuple[Array, Array]:
        s = SkeletonPreprocessor.arc_length(polyline)
        if len(polyline) == 1 or s[-1] < 1e-12:
            keypoints = np.repeat(polyline[:1], n_keypoints, axis=0)
            key_ts = np.linspace(0.0, 1.0, n_keypoints)
            return keypoints, key_ts
        st = np.linspace(0.0, s[-1], n_keypoints)
        keypoints = np.zeros((n_keypoints, 3), dtype=np.float64)
        for j in range(3):
            keypoints[:, j] = np.interp(st, s, polyline[:, j])
        key_ts = st / max(s[-1], 1e-12)
        return keypoints, key_ts

    def segment_polyline(self, segment_id: int) -> Array:
        return self.nodes_xyz[np.asarray(self.segments[segment_id], dtype=int)]

    @staticmethod
    def project_points_to_polyline(points: Array, polyline: Array) -> Tuple[Array, Array, Array]:
        if len(points) == 0:
            return np.zeros((0,), dtype=int), np.zeros((0,), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
        if len(polyline) == 1:
            return (
                np.zeros((len(points),), dtype=int),
                np.zeros((len(points),), dtype=np.float64),
                np.repeat(polyline[:1], len(points), axis=0),
            )

        seg_a = polyline[:-1]
        seg_b = polyline[1:]
        seg_v = seg_b - seg_a
        seg_len2 = np.sum(seg_v * seg_v, axis=1) + 1e-12
        arc = SkeletonPreprocessor.arc_length(polyline)
        total = max(arc[-1], 1e-12)

        best_seg = np.zeros((len(points),), dtype=int)
        best_s = np.zeros((len(points),), dtype=np.float64)
        best_proj = np.zeros((len(points), 3), dtype=np.float64)
        best_d2 = np.full((len(points),), np.inf, dtype=np.float64)

        for i in range(len(seg_a)):
            a = seg_a[i]
            v = seg_v[i]
            t = np.sum((points - a[None, :]) * v[None, :], axis=1) / seg_len2[i]
            t = np.clip(t, 0.0, 1.0)
            proj = a[None, :] + t[:, None] * v[None, :]
            d2 = np.sum((points - proj) ** 2, axis=1)
            mask = d2 < best_d2
            best_d2[mask] = d2[mask]
            best_seg[mask] = i
            best_proj[mask] = proj[mask]
            best_s[mask] = (arc[i] + t[mask] * np.sqrt(seg_len2[i])) / total

        return best_seg, best_s, best_proj

    @staticmethod
    def compute_rmf(keypoints: Array) -> Tuple[Array, Array, Array]:
        k = len(keypoints)
        T = np.zeros((k, 3), dtype=np.float64)
        N = np.zeros((k, 3), dtype=np.float64)
        B = np.zeros((k, 3), dtype=np.float64)
        if k == 0:
            return T, N, B
        if k == 1:
            T[0] = np.array([1.0, 0.0, 0.0])
            N[0] = np.array([0.0, 1.0, 0.0])
            B[0] = np.array([0.0, 0.0, 1.0])
            return T, N, B

        tang = np.zeros((k, 3), dtype=np.float64)
        tang[0] = keypoints[1] - keypoints[0]
        tang[-1] = keypoints[-1] - keypoints[-2]
        if k > 2:
            tang[1:-1] = keypoints[2:] - keypoints[:-2]
        T = _normalize(tang, axis=1)

        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(np.dot(ref, T[0])) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        N0 = ref - np.dot(ref, T[0]) * T[0]
        N[0] = _normalize(N0[None])[0]
        B[0] = _normalize(np.cross(T[0], N[0])[None])[0]

        for i in range(1, k):
            R = minimal_rotation_matrix(T[i - 1], T[i])
            N[i] = _normalize((R @ N[i - 1])[None])[0]
            B[i] = _normalize(np.cross(T[i], N[i])[None])[0]
        return T, N, B

    @staticmethod
    def nearest_key_ids(point_s: Array, key_ts: Array) -> Array:
        if len(point_s) == 0:
            return np.zeros((0,), dtype=int)
        idx = np.searchsorted(key_ts, point_s, side="left")
        idx = np.clip(idx, 0, len(key_ts) - 1)
        left = np.clip(idx - 1, 0, len(key_ts) - 1)
        choose_left = np.abs(point_s - key_ts[left]) < np.abs(point_s - key_ts[idx])
        idx[choose_left] = left[choose_left]
        return idx.astype(int)

    # ---------------- assignment ----------------

    def build_segment_candidates(self) -> None:
        candidates: Dict[int, List[Array]] = defaultdict(list)
        for sid, seg in enumerate(self.segments):
            for nid in seg:
                pts = self.node_surface_points.get(nid)
                if pts is not None and len(pts):
                    candidates[sid].append(pts)
        self.segment_candidates = {
            sid: np.unique(np.concatenate(parts, axis=0), axis=0) if parts else np.zeros((0, 3), dtype=np.float64)
            for sid, parts in candidates.items()
        }
        for sid in range(len(self.segments)):
            self.segment_candidates.setdefault(sid, np.zeros((0, 3), dtype=np.float64))

    def assign_surface_points_to_segments(self, shared_margin: float = 0.015, use_global_points: bool = True) -> None:
        if use_global_points:
            points = self.all_surface_points
        else:
            pts = [v for v in self.segment_candidates.values() if len(v)]
            points = np.unique(np.concatenate(pts, axis=0), axis=0) if pts else np.zeros((0, 3), dtype=np.float64)
        if points is None:
            raise RuntimeError("No surface points available")

        segment_polys = [self.segment_polyline(sid) for sid in range(len(self.segments))]
        dmat = np.full((len(points), len(segment_polys)), np.inf, dtype=np.float64)
        for sid, poly in enumerate(segment_polys):
            _, _, proj = self.project_points_to_polyline(points, poly)
            dmat[:, sid] = np.linalg.norm(points - proj, axis=1)

        owner = np.argmin(dmat, axis=1)
        d_sorted = np.sort(dmat, axis=1)
        shared_mask = ((d_sorted[:, 1] - d_sorted[:, 0]) < shared_margin) if dmat.shape[1] > 1 else np.zeros((len(points),), dtype=bool)

        seg_owned = defaultdict(list)
        seg_shared = defaultdict(list)
        for i, sid in enumerate(owner):
            seg_owned[int(sid)].append(points[i])
            if shared_mask[i]:
                near = np.where(dmat[i] <= (d_sorted[i, 0] + shared_margin))[0]
                for sj in near:
                    seg_shared[int(sj)].append(points[i])

        self.segment_owned = {
            sid: np.unique(np.asarray(seg_owned.get(sid, []), dtype=np.float64), axis=0) if seg_owned.get(sid) else np.zeros((0, 3), dtype=np.float64)
            for sid in range(len(self.segments))
        }
        self.segment_shared = {
            sid: np.unique(np.asarray(seg_shared.get(sid, []), dtype=np.float64), axis=0) if seg_shared.get(sid) else np.zeros((0, 3), dtype=np.float64)
            for sid in range(len(self.segments))
        }

    def score_hub_fragment(self, sid: int) -> float:
        seg = self.segments[sid]
        poly = self.segment_polyline(sid)
        L = self.arc_length(poly)[-1] if len(poly) else 0.0
        n_owned = len(self.segment_owned.get(sid, np.zeros((0, 3), dtype=np.float64)))
        d0 = self.degree.get(seg[0], 0)
        d1 = self.degree.get(seg[-1], 0)
        hub_endpoints = float(d0 >= 3) + float(d1 >= 3)
        leafy = float(d0 == 1 or d1 == 1)
        return 2.0 * hub_endpoints - 1.0 * leafy - 0.5 * min(L / 0.05, 3.0) - 0.25 * min(n_owned / 100.0, 4.0)

    def prune_segments(self, min_nodes: int = 4, min_length: float = 0.03, min_owned_points: int = 50) -> List[int]:
        keep = []
        for sid, seg in enumerate(self.segments):
            poly = self.segment_polyline(sid)
            L = self.arc_length(poly)[-1]
            n_owned = len(self.segment_owned.get(sid, np.zeros((0, 3))))
            deg0 = self.degree.get(seg[0], 0)
            deg1 = self.degree.get(seg[-1], 0)
            is_leafy = (deg0 == 1) or (deg1 == 1)
            if len(seg) >= min_nodes and (L >= min_length or is_leafy) and (n_owned >= min_owned_points or is_leafy):
                keep.append(sid)
        return keep

    # ---------------- support-curve pipeline ----------------

    def center_align_curve(
        self,
        curve: Array,
        surface_points: Array,
        n_key: int,
        bin_half_width: float,
        smooth_sigma: float,
        max_center_shift: float | None = None,
    ) -> Array:
        curve_key, key_t = self.resample_polyline(curve, n_key)
        T, N, B = self.compute_rmf(curve_key)
        _, s_pts, _ = self.project_points_to_polyline(surface_points, curve_key)
        curve_new = curve_key.copy()
        valid = np.zeros(n_key, dtype=bool)
        shifts = np.zeros((n_key, 3), dtype=np.float64)

        for i in range(n_key):
            mask = np.abs(s_pts - key_t[i]) <= bin_half_width
            if np.sum(mask) < 8:
                continue
            local = surface_points[mask] - curve_key[i][None]
            u = np.sum(local * N[i][None], axis=1)
            v = np.sum(local * B[i][None], axis=1)
            cu = np.median(u)
            cv = np.median(v)
            shift = cu * N[i] + cv * B[i]
            if max_center_shift is not None:
                nrm = np.linalg.norm(shift)
                if nrm > max_center_shift:
                    shift *= max_center_shift / max(nrm, 1e-12)
            shifts[i] = shift
            curve_new[i] += shift
            valid[i] = True

        if np.any(valid):
            idx_valid = np.where(valid)[0]
            for i in range(n_key):
                if not valid[i]:
                    j = idx_valid[np.argmin(np.abs(idx_valid - i))]
                    curve_new[i] = curve_key[i] + shifts[j]

        curve_new = _smooth1d(curve_new, smooth_sigma)
        return curve_new

    def compute_wrap_radius(
        self,
        curve: Array,
        surface_points: Array,
        n_key: int,
        bin_half_width: float,
        enclosing_quantile: float,
        uv_quantile: float,
        smooth_sigma: float,
        min_points_per_bin: int = 8,
    ) -> Tuple[Array, Array, Array, Array]:
        key_curve, key_t = self.resample_polyline(curve, n_key)
        T, N, B = self.compute_rmf(key_curve)
        _, s_pts, _ = self.project_points_to_polyline(surface_points, key_curve)

        key_radius = np.zeros(n_key, dtype=np.float64)
        key_radius_uv = np.zeros((n_key, 2), dtype=np.float64)
        valid = np.zeros(n_key, dtype=bool)

        for i in range(n_key):
            mask = np.abs(s_pts - key_t[i]) <= bin_half_width
            if np.sum(mask) < min_points_per_bin:
                continue
            local = surface_points[mask] - key_curve[i][None]
            u = np.sum(local * N[i][None], axis=1)
            v = np.sum(local * B[i][None], axis=1)
            cu = np.median(u)
            cv = np.median(v)
            u0 = u - cu
            v0 = v - cv
            rr = np.sqrt(u0 * u0 + v0 * v0)
            key_radius[i] = np.max(rr) if enclosing_quantile >= 1.0 else float(np.quantile(rr, enclosing_quantile))
            key_radius_uv[i, 0] = float(np.quantile(np.abs(u0), uv_quantile))
            key_radius_uv[i, 1] = float(np.quantile(np.abs(v0), uv_quantile))
            valid[i] = True

        if np.any(valid):
            idx_valid = np.where(valid)[0]
            for i in range(n_key):
                if not valid[i]:
                    j = idx_valid[np.argmin(np.abs(idx_valid - i))]
                    key_radius[i] = key_radius[j]
                    key_radius_uv[i] = key_radius_uv[j]
        else:
            _, _, proj = self.project_points_to_polyline(surface_points, key_curve)
            global_r = np.max(np.linalg.norm(surface_points - proj, axis=1)) if len(surface_points) else 0.0
            key_radius[:] = global_r
            key_radius_uv[:] = global_r

        key_radius = _smooth1d(key_radius, smooth_sigma)
        key_radius_uv = _smooth1d(key_radius_uv, smooth_sigma)
        return key_curve, key_t, key_radius, key_radius_uv

    def extend_segment_curve(
        self,
        curve: Array,
        radius: Array | None,
        extend_start: bool,
        extend_end: bool,
        start_alpha: float,
        end_alpha: float,
        min_ext: float,
        n_ext_start: int = 8,
        n_ext_end: int = 8,
    ) -> Array:
        curve = np.asarray(curve, dtype=np.float64)
        if len(curve) < 2:
            return curve.copy()

        if radius is None:
            r0 = np.linalg.norm(curve[1] - curve[0])
            r1 = np.linalg.norm(curve[-1] - curve[-2])
        else:
            r0 = float(radius[0])
            r1 = float(radius[-1])

        L0 = max(min_ext, start_alpha * r0)
        L1 = max(min_ext, end_alpha * r1)
        parts = []

        if extend_start and L0 > 1e-12 and n_ext_start > 0:
            t0 = _normalize((curve[0] - curve[1])[None])[0]
            alphas = np.linspace(1.0, 0.0, n_ext_start + 1)[:-1]
            parts.append(curve[0][None] + (alphas[:, None] * L0) * t0[None])

        parts.append(curve)

        if extend_end and L1 > 1e-12 and n_ext_end > 0:
            t1 = _normalize((curve[-1] - curve[-2])[None])[0]
            alphas = np.linspace(0.0, 1.0, n_ext_end + 1)[1:]
            parts.append(curve[-1][None] + (alphas[:, None] * L1) * t1[None])

        return np.concatenate(parts, axis=0)

    def compute_cylinder_radius(
        self,
        wrap_radius: Array,
        smooth_sigma: float,
        dilate_mul: float,
        dilate_add: float,
        min_radius: float | None = None,
    ) -> Array:
        cyl = _smooth1d(np.asarray(wrap_radius, dtype=np.float64), smooth_sigma)
        cyl = dilate_mul * cyl + dilate_add
        if min_radius is not None:
            cyl = np.maximum(cyl, min_radius)
        return cyl

    def build_support_view(
        self,
        sid: int,
        n_keypoints: int,
        center_bin_half_width: float,
        wrap_bin_half_width: float,
        center_smooth_sigma: float,
        wrap_smooth_sigma: float,
        cylinder_smooth_sigma: float,
        extend_alpha_start: float,
        extend_alpha_end: float,
        extend_min: float,
        cylinder_dilate_mul: float,
        cylinder_dilate_add: float,
    ) -> SegmentView:
        node_ids = self.segments[sid]
        poly_raw = self.segment_polyline(sid)
        pts = self.segment_owned[sid]
        pts_shared = self.segment_shared[sid]
        pts_cand = self.segment_candidates[sid]

        poly_center = self.center_align_curve(
            curve=poly_raw,
            surface_points=pts if len(pts) else pts_cand,
            n_key=max(n_keypoints, 20),
            bin_half_width=center_bin_half_width,
            smooth_sigma=center_smooth_sigma,
        )

        pre_curve, _, pre_wrap_radius, _ = self.compute_wrap_radius(
            curve=poly_center,
            surface_points=pts if len(pts) else pts_cand,
            n_key=n_keypoints,
            bin_half_width=wrap_bin_half_width,
            enclosing_quantile=1.0,
            uv_quantile=0.95,
            smooth_sigma=wrap_smooth_sigma,
        )

        poly_wrap = self.extend_segment_curve(
            curve=pre_curve,
            radius=pre_wrap_radius,
            extend_start=True,
            extend_end=True,
            start_alpha=extend_alpha_start,
            end_alpha=extend_alpha_end,
            min_ext=extend_min,
        )

        keypoints, key_ts, key_radius_wrap, key_radius_uv = self.compute_wrap_radius(
            curve=poly_wrap,
            surface_points=pts if len(pts) else pts_cand,
            n_key=n_keypoints,
            bin_half_width=wrap_bin_half_width,
            enclosing_quantile=1.0,
            uv_quantile=0.95,
            smooth_sigma=wrap_smooth_sigma,
        )

        # train radius = slightly less aggressive than full wrap
        _, _, key_radius_train, _ = self.compute_wrap_radius(
            curve=poly_center,
            surface_points=pts if len(pts) else pts_cand,
            n_key=n_keypoints,
            bin_half_width=wrap_bin_half_width,
            enclosing_quantile=0.95,
            uv_quantile=0.90,
            smooth_sigma=wrap_smooth_sigma,
        )

        cyl_radius = self.compute_cylinder_radius(
            wrap_radius=key_radius_wrap,
            smooth_sigma=cylinder_smooth_sigma,
            dilate_mul=cylinder_dilate_mul,
            dilate_add=cylinder_dilate_add,
        )

        _, point_s, _ = self.project_points_to_polyline(pts, keypoints)
        point_key_ids = self.nearest_key_ids(point_s, key_ts)

        hub_score = self.score_hub_fragment(sid)
        return SegmentView(
            segment_id=-1,  # filled later
            raw_segment_id=sid,
            node_ids=list(map(int, node_ids)),
            endpoints=(int(node_ids[0]), int(node_ids[-1])),
            polyline_raw=poly_raw,
            polyline_center=poly_center,
            polyline_wrap=poly_wrap,
            arclength_raw=self.arc_length(poly_raw),
            surface_points_owned=pts,
            surface_points_shared=pts_shared,
            surface_points_candidate=pts_cand,
            point_s=point_s,
            point_key_ids=point_key_ids,
            keypoints=keypoints,
            key_ts=key_ts,
            key_radius_train=key_radius_train,
            key_radius_wrap=key_radius_wrap,
            key_radius_uv=key_radius_uv,
            cylinder_radius=cyl_radius,
            hub_score=float(hub_score),
            is_hub_fragment=hub_score > 1.5,
        )

    # ---------------- visualization ----------------

    @staticmethod
    def _segment_color(i: int, n: int):
        cmap = plt.get_cmap("tab20" if n <= 20 else "gist_ncar")
        return cmap(i / max(n - 1, 1))

    @staticmethod
    def _set_axes_equal(ax, points: Array):
        if len(points) == 0:
            return
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        center = 0.5 * (mins + maxs)
        radius = 0.5 * np.max(maxs - mins)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

    def _plot_mesh_wire(self, ax, stride: int = 8, alpha: float = 0.12, linewidth: float = 0.2):
        if self.mesh_vertices is None or self.mesh_faces is None:
            return
        faces = self.mesh_faces[::max(1, stride)]
        for tri in faces:
            pts = self.mesh_vertices[np.asarray(tri, dtype=int)]
            cyc = np.vstack([pts, pts[0:1]])
            ax.plot(cyc[:, 0], cyc[:, 1], cyc[:, 2], color="k", alpha=alpha, linewidth=linewidth)

    @staticmethod
    def _circle3d(center: Array, N: Array, B: Array, radius: float, n: int = 48) -> Array:
        th = np.linspace(0.0, 2.0 * np.pi, n)
        return center[None] + radius * (
            np.cos(th)[:, None] * N[None] + np.sin(th)[:, None] * B[None]
        )

    def visualize(self, segments: List[SegmentView], out_dir: str) -> List[str]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []

        # 1: mesh + raw segmented skeleton
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        self._plot_mesh_wire(ax)
        if self.mesh_vertices is not None and self.mesh_faces is None:
            ax.scatter(self.mesh_vertices[:,0], self.mesh_vertices[:,1], self.mesh_vertices[:,2], s=0.15, alpha=0.05)
        for sid, seg in enumerate(segments):
            c = self._segment_color(sid, len(segments))
            p = seg.polyline_raw
            ax.plot(p[:,0], p[:,1], p[:,2], color=c, linewidth=2.0)
        ax.set_title("Mesh + segmented raw skeleton")
        all_pts = [seg.polyline_raw for seg in segments]
        if self.mesh_vertices is not None:
            all_pts.append(self.mesh_vertices)
        self._set_axes_equal(ax, np.vstack(all_pts))
        f1 = out_dir / "01_mesh_and_segmented_skeleton.png"
        fig.savefig(f1, dpi=220, bbox_inches="tight"); plt.close(fig); paths.append(str(f1))

        # 2: owned points by segment
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        for sid, seg in enumerate(segments):
            c = self._segment_color(sid, len(segments))
            if len(seg.surface_points_owned):
                ax.scatter(seg.surface_points_owned[:,0], seg.surface_points_owned[:,1], seg.surface_points_owned[:,2], s=0.8, color=c, alpha=0.8)
            p = seg.polyline_raw
            ax.plot(p[:,0], p[:,1], p[:,2], color="k", linewidth=0.8, alpha=0.35)
        ax.set_title("Owned surface points by segment")
        all_pts = [seg.surface_points_owned for seg in segments if len(seg.surface_points_owned)]
        if not all_pts:
            all_pts = [np.zeros((1,3))]
        self._set_axes_equal(ax, np.vstack(all_pts))
        f2 = out_dir / "02_owned_surface_points_by_segment.png"
        fig.savefig(f2, dpi=220, bbox_inches="tight"); plt.close(fig); paths.append(str(f2))

        # 3: raw / center / wrap on same plot
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        self._plot_mesh_wire(ax)
        for sid, seg in enumerate(segments):
            c = self._segment_color(sid, len(segments))
            ax.plot(seg.polyline_raw[:,0], seg.polyline_raw[:,1], seg.polyline_raw[:,2], color=c, linewidth=1.0, alpha=0.35)
            ax.plot(seg.polyline_center[:,0], seg.polyline_center[:,1], seg.polyline_center[:,2], color=c, linewidth=1.6, alpha=0.7)
            ax.plot(seg.keypoints[:,0], seg.keypoints[:,1], seg.keypoints[:,2], color=c, linewidth=2.4)
        ax.set_title("Raw / center / wrap key curves")
        all_pts = [seg.keypoints for seg in segments]
        if self.mesh_vertices is not None:
            all_pts.append(self.mesh_vertices)
        self._set_axes_equal(ax, np.vstack(all_pts))
        f3 = out_dir / "03_wrap_curves.png"
        fig.savefig(f3, dpi=220, bbox_inches="tight"); plt.close(fig); paths.append(str(f3))

        # 4: radius rings + cylinder support on first few segments
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        self._plot_mesh_wire(ax)
        show_ids = list(range(min(4, len(segments))))
        for sid in show_ids:
            seg = segments[sid]
            c = self._segment_color(sid, len(segments))
            T, N, B = self.compute_rmf(seg.keypoints)
            ax.plot(seg.keypoints[:,0], seg.keypoints[:,1], seg.keypoints[:,2], color=c, linewidth=2.0)
            every = max(1, len(seg.keypoints)//12)
            for i in range(0, len(seg.keypoints), every):
                ring = self._circle3d(seg.keypoints[i], N[i], B[i], float(seg.cylinder_radius[i]))
                ax.plot(ring[:,0], ring[:,1], ring[:,2], color=c, linewidth=0.8)
        ax.set_title("Cylinder support rings (first segments)")
        all_pts = [seg.keypoints for seg in segments]
        if self.mesh_vertices is not None:
            all_pts.append(self.mesh_vertices)
        self._set_axes_equal(ax, np.vstack(all_pts))
        f4 = out_dir / "04_cylinder_support.png"
        fig.savefig(f4, dpi=220, bbox_inches="tight"); plt.close(fig); paths.append(str(f4))

        # 5: summary
        lengths = [float(seg.arclength_raw[-1]) if len(seg.arclength_raw) else 0.0 for seg in segments]
        owned_counts = [len(seg.surface_points_owned) for seg in segments]
        hub_scores = [seg.hub_score for seg in segments]
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(segments))
        ax.bar(x - 0.25, lengths, width=0.25, label="arc length")
        ax.bar(x, owned_counts, width=0.25, label="owned points")
        ax.bar(x + 0.25, hub_scores, width=0.25, label="hub score")
        ax.set_xlabel("segment id")
        ax.set_title("Segment summary")
        ax.legend()
        f5 = out_dir / "05_segment_summary.png"
        fig.savefig(f5, dpi=220, bbox_inches="tight"); plt.close(fig); paths.append(str(f5))
        return paths

    # ---------------- IO ----------------

    def save_npz(self, segments: List[SegmentView], out_npz: str) -> None:
        payload = {
            "mesh_vertices": self.mesh_vertices if self.mesh_vertices is not None else np.zeros((0, 3), dtype=np.float64),
            "mesh_faces": self.mesh_faces if self.mesh_faces is not None else np.zeros((0, 3), dtype=np.int32),
            "nodes_xyz": self.nodes_xyz,
            "degree": np.asarray([self.degree[i] for i in range(len(self.nodes_xyz))], dtype=np.int32),
            "segment_ids": np.asarray([seg.segment_id for seg in segments], dtype=np.int32),
        }
        for seg in segments:
            prefix = f"segment_{seg.segment_id}"
            payload[f"{prefix}_raw_segment_id"] = np.asarray([seg.raw_segment_id], dtype=np.int32)
            payload[f"{prefix}_node_ids"] = np.asarray(seg.node_ids, dtype=np.int32)
            payload[f"{prefix}_endpoints"] = np.asarray(seg.endpoints, dtype=np.int32)
            payload[f"{prefix}_polyline_raw"] = seg.polyline_raw
            payload[f"{prefix}_polyline_center"] = seg.polyline_center
            payload[f"{prefix}_polyline_wrap"] = seg.polyline_wrap
            payload[f"{prefix}_arclength_raw"] = seg.arclength_raw
            payload[f"{prefix}_surface_points_owned"] = seg.surface_points_owned
            payload[f"{prefix}_surface_points_shared"] = seg.surface_points_shared
            payload[f"{prefix}_surface_points_candidate"] = seg.surface_points_candidate
            payload[f"{prefix}_point_s"] = seg.point_s
            payload[f"{prefix}_point_key_ids"] = seg.point_key_ids
            payload[f"{prefix}_keypoints"] = seg.keypoints
            payload[f"{prefix}_key_ts"] = seg.key_ts
            payload[f"{prefix}_key_radius_train"] = seg.key_radius_train
            payload[f"{prefix}_key_radius_wrap"] = seg.key_radius_wrap
            payload[f"{prefix}_key_radius_uv"] = seg.key_radius_uv
            payload[f"{prefix}_cylinder_radius"] = seg.cylinder_radius
            payload[f"{prefix}_hub_score"] = np.asarray([seg.hub_score], dtype=np.float64)
            payload[f"{prefix}_is_hub_fragment"] = np.asarray([int(seg.is_hub_fragment)], dtype=np.int32)
        np.savez(out_npz, **payload)

    def save_summary(self, segments: List[SegmentView], out_json: str) -> None:
        summary = {
            "skel_file": str(self.skel_file),
            "corr_file": str(self.corr_file),
            "mesh_file": str(self.mesh_file) if self.mesh_file else None,
            "num_nodes": int(len(self.nodes_xyz)),
            "num_edges": int(len(self.edge_ids)),
            "degree_histogram": {str(k): int(v) for k, v in Counter(self.degree.values()).items()},
            "num_raw_segments": int(len(self.segments)),
            "corr_misses": int(self.corr_misses),
            "num_surface_points": int(len(self.all_surface_points)) if self.all_surface_points is not None else 0,
            "segments": [
                {
                    "segment_id": int(seg.segment_id),
                    "raw_segment_id": int(seg.raw_segment_id),
                    "num_nodes": int(len(seg.node_ids)),
                    "arc_length_raw": float(seg.arclength_raw[-1]) if len(seg.arclength_raw) else 0.0,
                    "owned_points": int(len(seg.surface_points_owned)),
                    "shared_points": int(len(seg.surface_points_shared)),
                    "hub_score": float(seg.hub_score),
                    "is_hub_fragment": bool(seg.is_hub_fragment),
                }
                for seg in segments
            ],
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    # ---------------- top-level ----------------

    def run(
        self,
        out_npz: str,
        out_json: str,
        out_viz_dir: str | None = None,
        n_keypoints: int = 60,
        shared_margin: float = 0.015,
        min_nodes: int = 4,
        min_length: float = 0.03,
        min_owned_points: int = 50,
        center_bin_half_width: float = 0.005,
        wrap_bin_half_width: float = 0.006,
        center_smooth_sigma: float = 2.0,
        wrap_smooth_sigma: float = 2.0,
        cylinder_smooth_sigma: float = 3.0,
        extend_alpha_start: float = 0.3,
        extend_alpha_end: float = 1.2,
        extend_min: float = 0.0,
        cylinder_dilate_mul: float = 1.05,
        cylinder_dilate_add: float = 0.0,
    ) -> List[SegmentView]:
        self.load_mesh()
        self.build_graph()
        self.extract_segments()
        self.parse_correspondence()
        self.build_segment_candidates()
        self.assign_surface_points_to_segments(shared_margin=shared_margin, use_global_points=True)
        keep = self.prune_segments(min_nodes=min_nodes, min_length=min_length, min_owned_points=min_owned_points)

        views: List[SegmentView] = []
        for new_sid, sid in enumerate(keep):
            v = self.build_support_view(
                sid=sid,
                n_keypoints=n_keypoints,
                center_bin_half_width=center_bin_half_width,
                wrap_bin_half_width=wrap_bin_half_width,
                center_smooth_sigma=center_smooth_sigma,
                wrap_smooth_sigma=wrap_smooth_sigma,
                cylinder_smooth_sigma=cylinder_smooth_sigma,
                extend_alpha_start=extend_alpha_start,
                extend_alpha_end=extend_alpha_end,
                extend_min=extend_min,
                cylinder_dilate_mul=cylinder_dilate_mul,
                cylinder_dilate_add=cylinder_dilate_add,
            )
            v.segment_id = new_sid
            views.append(v)

        self.save_npz(views, out_npz)
        self.save_summary(views, out_json)
        if out_viz_dir:
            self.visualize(views, out_viz_dir)
        return views


def main() -> None:
    ap = argparse.ArgumentParser(description="Global skeleton graph preprocessing + per-segment localized views")
    ap.add_argument("--skel_file", required=True)
    ap.add_argument("--corr_file", required=True)
    ap.add_argument("--mesh_file", default=None)
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_viz_dir", default=None)

    ap.add_argument("--n_keypoints", type=int, default=60)
    ap.add_argument("--shared_margin", type=float, default=0.015)
    ap.add_argument("--min_nodes", type=int, default=4)
    ap.add_argument("--min_length", type=float, default=0.03)
    ap.add_argument("--min_owned_points", type=int, default=50)
    ap.add_argument("--decimals", type=int, default=6)

    ap.add_argument("--center_bin_half_width", type=float, default=0.05)
    ap.add_argument("--wrap_bin_half_width", type=float, default=0.06)
    ap.add_argument("--center_smooth_sigma", type=float, default=2.0)
    ap.add_argument("--wrap_smooth_sigma", type=float, default=2.0)
    ap.add_argument("--cylinder_smooth_sigma", type=float, default=3.0)
    ap.add_argument("--extend_alpha_start", type=float, default=0.3)
    ap.add_argument("--extend_alpha_end", type=float, default=1.2)
    ap.add_argument("--extend_min", type=float, default=0.0)
    ap.add_argument("--cylinder_dilate_mul", type=float, default=1.05)
    ap.add_argument("--cylinder_dilate_add", type=float, default=0.0)
    args = ap.parse_args()

    proc = SkeletonPreprocessor(
        skel_file=args.skel_file,
        corr_file=args.corr_file,
        mesh_file=args.mesh_file,
        decimals=args.decimals,
    )
    proc.run(
        out_npz=args.out_npz,
        out_json=args.out_json,
        out_viz_dir=args.out_viz_dir,
        n_keypoints=args.n_keypoints,
        shared_margin=args.shared_margin,
        min_nodes=args.min_nodes,
        min_length=args.min_length,
        min_owned_points=args.min_owned_points,
        center_bin_half_width=args.center_bin_half_width,
        wrap_bin_half_width=args.wrap_bin_half_width,
        center_smooth_sigma=args.center_smooth_sigma,
        wrap_smooth_sigma=args.wrap_smooth_sigma,
        cylinder_smooth_sigma=args.cylinder_smooth_sigma,
        extend_alpha_start=args.extend_alpha_start,
        extend_alpha_end=args.extend_alpha_end,
        extend_min=args.extend_min,
        cylinder_dilate_mul=args.cylinder_dilate_mul,
        cylinder_dilate_add=args.cylinder_dilate_add,
    )


if __name__ == "__main__":
    main()
