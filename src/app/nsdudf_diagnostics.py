"""Diagnostics for NSDUDF cell classification and cross-cell consistency.

This module is deliberately independent of Oktopus' normal mesh extraction
path. It consumes only:

    * an NSDUDF classifier,
    * a callable returning UDF values and gradients on [-1, 1]^3, and
    * the extraction cube transform.

It records the exact failure modes that turn local NSDUDF predictions into
holes:

    1. cells rejected by the near-surface thresholds;
    2. invalid or non-unit gradients in accepted cells;
    3. low-confidence local sign predictions;
    4. incompatible sign patterns on faces shared by neighboring cells; and
    5. predicted surface crossings that terminate against a rejected cell.

The diagnostic can optionally reconstruct the same pseudo-SDF mesh without
calling the ordinary NSDUDF extraction path a second time.
"""

from __future__ import annotations

import json
import math
import os
import os.path as op
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import trimesh


_INACTIVE_CLASS = np.uint8(255)

# NSDUDF local corner order, matching core.utils.df_and_grad_to_input_cells:
#   0=(0,0,0), 1=(0,0,1), 2=(0,1,1), 3=(0,1,0),
#   4=(1,0,0), 5=(1,0,1), 6=(1,1,1), 7=(1,1,0)
_FACE_CORNERS = {
    (0, 0): (0, 1, 2, 3),
    (0, 1): (4, 5, 6, 7),
    (1, 0): (0, 1, 5, 4),
    (1, 1): (3, 2, 6, 7),
    (2, 0): (0, 3, 7, 4),
    (2, 1): (1, 2, 6, 5),
}


@dataclass
class _SampleAccumulator:
    """Bounded deterministic sample collector for percentile summaries."""

    limit: int = 1_000_000

    def __post_init__(self) -> None:
        self._parts: List[np.ndarray] = []
        self._size = 0

    def add(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0 or self._size >= self.limit:
            return
        remaining = self.limit - self._size
        if arr.size > remaining:
            # Deterministic evenly spaced subsample. This is sufficient for a
            # diagnostic percentile summary and avoids run-to-run randomness.
            take = np.linspace(0, arr.size - 1, remaining, dtype=np.int64)
            arr = arr[take]
        self._parts.append(arr.copy())
        self._size += int(arr.size)

    def values(self) -> np.ndarray:
        if not self._parts:
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(self._parts, axis=0)

    def percentile_dict(self, percentiles: Sequence[float]) -> Dict[str, float]:
        arr = self.values()
        if arr.size == 0:
            return {}
        vals = np.percentile(arr, percentiles)
        return {
            str(p).replace(".", "_"): float(v)
            for p, v in zip(percentiles, vals)
        }


@dataclass
class _PointCollector:
    """Collect at most ``limit`` points for PLY visualization."""

    limit: int

    def __post_init__(self) -> None:
        self._parts: List[np.ndarray] = []
        self._size = 0

    def add(self, points: np.ndarray) -> None:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if pts.size == 0 or self._size >= self.limit:
            return
        remaining = self.limit - self._size
        if len(pts) > remaining:
            pts = pts[:remaining]
        self._parts.append(pts.copy())
        self._size += int(len(pts))

    def array(self) -> np.ndarray:
        if not self._parts:
            return np.empty((0, 3), dtype=np.float64)
        return np.concatenate(self._parts, axis=0)


@dataclass
class DiagnosticOptions:
    n_grid_samples: int
    classifier_batch_size: int = 32768
    query_slab: int = 4
    cell_slab: int = 2
    normalize_udf: bool = True
    max_avg_factor: float = 1.2
    max_max_factor: float = 2.0
    loose_min_factor: float = 1.0
    max_visualization_points: int = 200_000
    extract_mesh: bool = True


@dataclass
class DiagnosticResult:
    summary: Dict[str, object]
    mesh: Optional[trimesh.Trimesh]
    output_dir: str


def _build_class_sign_lookup() -> np.ndarray:
    """Return signs for 128 classes as shape (128, 8), values in {-1,+1}."""
    signs = np.ones((128, 8), dtype=np.int8)
    for cls in range(128):
        for bit in range(7):
            signs[cls, bit + 1] = 1 if (cls & (1 << bit)) else -1
    return signs


def _build_face_code_lookup(signs: np.ndarray) -> Dict[Tuple[int, int], np.ndarray]:
    """Build 4-bit face codes canonicalized up to a complete sign flip."""
    result: Dict[Tuple[int, int], np.ndarray] = {}
    for key, corners in _FACE_CORNERS.items():
        positive = signs[:, corners] > 0
        code = np.zeros((len(signs),), dtype=np.uint8)
        for bit in range(4):
            code |= positive[:, bit].astype(np.uint8) << np.uint8(bit)
        # Complementing all four signs gives 15-code. A UDF's local sign is
        # arbitrary, so neighboring faces are compatible modulo this flip.
        canonical = np.minimum(code, np.uint8(15) - code)
        result[key] = canonical
    return result


def _cell_corner_values(field: np.ndarray, i0: int, i1: int) -> np.ndarray:
    """Return UDF cell corners for axis-0 cell slab [i0, i1)."""
    return np.stack(
        (
            field[i0:i1, :-1, :-1],
            field[i0:i1, :-1, 1:],
            field[i0:i1, 1:, 1:],
            field[i0:i1, 1:, :-1],
            field[i0 + 1 : i1 + 1, :-1, :-1],
            field[i0 + 1 : i1 + 1, :-1, 1:],
            field[i0 + 1 : i1 + 1, 1:, 1:],
            field[i0 + 1 : i1 + 1, 1:, :-1],
        ),
        axis=-1,
    )


def _cell_corner_gradients(field: np.ndarray, i0: int, i1: int) -> np.ndarray:
    """Return gradient cell corners for axis-0 cell slab [i0, i1)."""
    return np.stack(
        (
            field[i0:i1, :-1, :-1, :],
            field[i0:i1, :-1, 1:, :],
            field[i0:i1, 1:, 1:, :],
            field[i0:i1, 1:, :-1, :],
            field[i0 + 1 : i1 + 1, :-1, :-1, :],
            field[i0 + 1 : i1 + 1, :-1, 1:, :],
            field[i0 + 1 : i1 + 1, 1:, 1:, :],
            field[i0 + 1 : i1 + 1, 1:, :-1, :],
        ),
        axis=-2,
    )


def _normalized_grid_slab(n: int, i0: int, i1: int) -> np.ndarray:
    coords = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    a, b, c = np.meshgrid(
        coords[i0:i1], coords, coords, indexing="ij"
    )
    return np.stack((a, b, c), axis=-1).reshape(-1, 3)


def _query_grid_to_memmaps(
    udf_and_grad_f: Callable,
    *,
    n: int,
    query_slab: int,
    output_dir: str,
) -> Tuple[np.memmap, np.memmap, Dict[str, object]]:
    udf_path = op.join(output_dir, "udf_grid.float32.dat")
    grad_path = op.join(output_dir, "gradient_grid.float32.dat")

    udf = np.memmap(udf_path, mode="w+", dtype=np.float32, shape=(n, n, n))
    grad = np.memmap(
        grad_path, mode="w+", dtype=np.float32, shape=(n, n, n, 3)
    )

    udf_sample = _SampleAccumulator()
    grad_norm_sample = _SampleAccumulator()
    finite_udf = 0
    negative_udf = 0
    nonfinite_udf = 0
    nonfinite_grad = 0
    zero_grad = 0

    slab = max(1, int(query_slab))
    for i0 in range(0, n, slab):
        i1 = min(i0 + slab, n)
        points = _normalized_grid_slab(n, i0, i1)
        values_t, grads_t = udf_and_grad_f(torch.from_numpy(points))
        values = np.asarray(values_t.detach().cpu(), dtype=np.float32).reshape(
            i1 - i0, n, n
        )
        grads = np.asarray(grads_t.detach().cpu(), dtype=np.float32).reshape(
            i1 - i0, n, n, 3
        )

        udf[i0:i1] = values
        grad[i0:i1] = grads

        vf = np.isfinite(values)
        gf = np.all(np.isfinite(grads), axis=-1)
        norms = np.linalg.norm(grads, axis=-1)

        finite_udf += int(vf.sum())
        negative_udf += int((values[vf] < 0.0).sum())
        nonfinite_udf += int((~vf).sum())
        nonfinite_grad += int((~gf).sum())
        zero_grad += int((gf & (norms < 1e-6)).sum())

        udf_sample.add(values[vf])
        grad_norm_sample.add(norms[gf])

        print(
            "[nsdudf diag query]",
            f"slab={i0}:{i1}/{n}",
            f"points={(i1-i0)*n*n}",
            flush=True,
        )

    udf.flush()
    grad.flush()

    summary = {
        "grid_points": int(n**3),
        "finite_udf": finite_udf,
        "negative_udf": negative_udf,
        "nonfinite_udf": nonfinite_udf,
        "nonfinite_gradient_points": nonfinite_grad,
        "near_zero_gradient_points": zero_grad,
        "udf_percentiles": udf_sample.percentile_dict(
            [0, 1, 5, 25, 50, 75, 95, 99, 100]
        ),
        "gradient_norm_percentiles": grad_norm_sample.percentile_dict(
            [0, 1, 5, 25, 50, 75, 95, 99, 100]
        ),
        "udf_memmap": udf_path,
        "gradient_memmap": grad_path,
    }
    return udf, grad, summary


def _predict_cells(
    model: torch.nn.Module,
    udf: np.ndarray,
    grad: np.ndarray,
    *,
    options: DiagnosticOptions,
    output_dir: str,
) -> Tuple[np.memmap, np.memmap, np.memmap, Dict[str, object]]:
    n = int(options.n_grid_samples)
    cells = n - 1
    voxel_size = 2.0 / cells
    max_avg = options.max_avg_factor * voxel_size
    max_max = options.max_max_factor * voxel_size
    loose_min = options.loose_min_factor * voxel_size

    class_path = op.join(output_dir, "predicted_class.uint8.dat")
    status_path = op.join(output_dir, "cell_status.uint8.dat")
    confidence_path = op.join(output_dir, "classifier_confidence.float16.dat")

    class_map = np.memmap(
        class_path, mode="w+", dtype=np.uint8, shape=(cells, cells, cells)
    )
    class_map[:] = _INACTIVE_CLASS
    status_map = np.memmap(
        status_path, mode="w+", dtype=np.uint8, shape=(cells, cells, cells)
    )
    status_map[:] = 0
    confidence_map = np.memmap(
        confidence_path,
        mode="w+",
        dtype=np.float16,
        shape=(cells, cells, cells),
    )
    confidence_map[:] = np.float16(-1.0)

    device = next(model.parameters()).device
    classifier_batch = max(1, int(options.classifier_batch_size))
    slab = max(1, int(options.cell_slab))

    total_cells = int(cells**3)
    accepted_cells = 0
    loose_near_cells = 0
    near_rejected_cells = 0
    accepted_invalid_gradient_cells = 0
    accepted_nonunit_gradient_cells = 0
    candidate_corner_samples = 0
    invalid_candidate_corners = 0
    nonunit_candidate_corners = 0

    udf_norm_sample = _SampleAccumulator()
    candidate_grad_norm_sample = _SampleAccumulator()
    confidence_sample = _SampleAccumulator()
    margin_sample = _SampleAccumulator()
    entropy_sample = _SampleAccumulator()

    for i0 in range(0, cells, slab):
        i1 = min(i0 + slab, cells)
        u8 = _cell_corner_values(udf, i0, i1)
        g8 = _cell_corner_gradients(grad, i0, i1)
        slab_shape = u8.shape[:-1]

        finite_u = np.all(np.isfinite(u8), axis=-1)
        avg_u = np.mean(u8, axis=-1)
        max_u = np.max(u8, axis=-1)
        min_u = np.min(u8, axis=-1)

        accepted = finite_u & (avg_u <= max_avg) & (max_u <= max_max)
        loose_near = finite_u & (min_u <= loose_min)
        rejected_near = loose_near & ~accepted

        grad_norm = np.linalg.norm(g8, axis=-1)
        grad_valid_corner = np.all(np.isfinite(g8), axis=-1) & (
            grad_norm >= 1e-6
        )
        grad_nonunit_corner = grad_valid_corner & (
            (grad_norm < 0.5) | (grad_norm > 1.5)
        )
        any_invalid = np.any(~grad_valid_corner, axis=-1)
        any_nonunit = np.any(grad_nonunit_corner, axis=-1)

        status = np.zeros(slab_shape, dtype=np.uint8)
        status[rejected_near] = 2
        status[accepted] = 1
        status[accepted & any_invalid] = 3
        status_map[i0:i1] = status

        accepted_cells += int(accepted.sum())
        loose_near_cells += int(loose_near.sum())
        near_rejected_cells += int(rejected_near.sum())
        accepted_invalid_gradient_cells += int((accepted & any_invalid).sum())
        accepted_nonunit_gradient_cells += int((accepted & any_nonunit).sum())

        if not np.any(accepted):
            print(
                "[nsdudf diag classify]",
                f"slab={i0}:{i1}/{cells}",
                "accepted=0",
                flush=True,
            )
            continue

        flat_u = u8.reshape(-1, 8)
        flat_g = g8.reshape(-1, 8, 3)
        accepted_flat = accepted.reshape(-1)
        accepted_indices = np.flatnonzero(accepted_flat)
        candidate_u = flat_u[accepted_indices].astype(np.float32, copy=True)
        candidate_g = flat_g[accepted_indices].astype(np.float32, copy=True)

        candidate_norms = np.linalg.norm(candidate_g, axis=-1)
        valid_candidate_corner = np.all(np.isfinite(candidate_g), axis=-1) & (
            candidate_norms >= 1e-6
        )
        nonunit_candidate_corner = valid_candidate_corner & (
            (candidate_norms < 0.5) | (candidate_norms > 1.5)
        )

        candidate_corner_samples += int(candidate_norms.size)
        invalid_candidate_corners += int((~valid_candidate_corner).sum())
        nonunit_candidate_corners += int(nonunit_candidate_corner.sum())

        udf_norm_sample.add(candidate_u / voxel_size)
        candidate_grad_norm_sample.add(candidate_norms)

        model_input = np.zeros((len(candidate_u), 32), dtype=np.float32)
        model_input[:, :8] = candidate_u
        if options.normalize_udf:
            model_input[:, :8] /= voxel_size
        model_input[:, 8:] = candidate_g.reshape(-1, 24)

        predicted = np.empty((len(model_input),), dtype=np.uint8)
        confidence = np.empty((len(model_input),), dtype=np.float32)
        margin = np.empty((len(model_input),), dtype=np.float32)
        entropy = np.empty((len(model_input),), dtype=np.float32)

        with torch.no_grad():
            for b0 in range(0, len(model_input), classifier_batch):
                b1 = min(b0 + classifier_batch, len(model_input))
                inp = torch.from_numpy(model_input[b0:b1]).to(device)
                logits = model(inp)
                top2 = torch.topk(logits, k=2, dim=1)
                probs = torch.softmax(logits, dim=1)

                predicted[b0:b1] = (
                    top2.indices[:, 0].detach().cpu().numpy().astype(np.uint8)
                )
                confidence[b0:b1] = (
                    probs.max(dim=1).values.detach().cpu().numpy().astype(np.float32)
                )
                margin[b0:b1] = (
                    (top2.values[:, 0] - top2.values[:, 1])
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
                ent /= math.log(probs.shape[1])
                entropy[b0:b1] = ent.detach().cpu().numpy().astype(np.float32)

        class_slab = class_map[i0:i1].reshape(-1)
        conf_slab = confidence_map[i0:i1].reshape(-1)
        class_slab[accepted_indices] = predicted
        conf_slab[accepted_indices] = confidence.astype(np.float16)

        confidence_sample.add(confidence)
        margin_sample.add(margin)
        entropy_sample.add(entropy)

        print(
            "[nsdudf diag classify]",
            f"slab={i0}:{i1}/{cells}",
            f"accepted={len(accepted_indices)}",
            f"rejected_near={int(rejected_near.sum())}",
            f"invalid_grad_cells={int((accepted & any_invalid).sum())}",
            flush=True,
        )

    class_map.flush()
    status_map.flush()
    confidence_map.flush()

    summary = {
        "total_cells": total_cells,
        "accepted_cells": accepted_cells,
        "accepted_cell_pct": 100.0 * accepted_cells / max(total_cells, 1),
        "loose_near_cells": loose_near_cells,
        "near_rejected_cells": near_rejected_cells,
        "near_rejected_pct_of_loose_near": (
            100.0 * near_rejected_cells / max(loose_near_cells, 1)
        ),
        "accepted_invalid_gradient_cells": accepted_invalid_gradient_cells,
        "accepted_invalid_gradient_cell_pct": (
            100.0 * accepted_invalid_gradient_cells / max(accepted_cells, 1)
        ),
        "accepted_nonunit_gradient_cells": accepted_nonunit_gradient_cells,
        "accepted_nonunit_gradient_cell_pct": (
            100.0 * accepted_nonunit_gradient_cells / max(accepted_cells, 1)
        ),
        "candidate_corner_samples": candidate_corner_samples,
        "invalid_candidate_corners": invalid_candidate_corners,
        "invalid_candidate_corner_pct": (
            100.0 * invalid_candidate_corners / max(candidate_corner_samples, 1)
        ),
        "nonunit_candidate_corners": nonunit_candidate_corners,
        "nonunit_candidate_corner_pct": (
            100.0 * nonunit_candidate_corners / max(candidate_corner_samples, 1)
        ),
        "voxel_size_cube": voxel_size,
        "max_avg_distance_cube": max_avg,
        "max_max_distance_cube": max_max,
        "loose_min_distance_cube": loose_min,
        "normalized_candidate_udf_percentiles": udf_norm_sample.percentile_dict(
            [0, 1, 5, 25, 50, 75, 95, 99, 100]
        ),
        "candidate_gradient_norm_percentiles": (
            candidate_grad_norm_sample.percentile_dict(
                [0, 1, 5, 25, 50, 75, 95, 99, 100]
            )
        ),
        "classifier_softmax_max_percentiles": confidence_sample.percentile_dict(
            [0, 1, 5, 25, 50, 75, 95, 99, 100]
        ),
        "classifier_logit_margin_percentiles": margin_sample.percentile_dict(
            [0, 1, 5, 25, 50, 75, 95, 99, 100]
        ),
        "classifier_normalized_entropy_percentiles": entropy_sample.percentile_dict(
            [0, 1, 5, 25, 50, 75, 95, 99, 100]
        ),
        "class_memmap": class_path,
        "status_memmap": status_path,
        "confidence_memmap": confidence_path,
    }
    return class_map, status_map, confidence_map, summary


def _shared_face_points(
    mask: np.ndarray,
    *,
    axis: int,
    chunk_start: int,
    cells: int,
    domain_center: np.ndarray,
    domain_half_extent: float,
) -> np.ndarray:
    """Convert a bad internal-face mask into world-space face centers."""
    ijk = np.argwhere(mask)
    if ijk.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    # The sliced pair starts at ``chunk_start`` along axis 0 only when axis=0.
    # For axes 1/2 the chunking is still performed along array axis 0, so that
    # offset always applies to the first coordinate.
    ijk[:, 0] += int(chunk_start)

    u = -1.0 + (2.0 / cells) * (ijk.astype(np.float64) + 0.5)
    # Shared face lies one half-cell above the lower cell center along the pair
    # axis. The mask index always names the lower of the two adjacent cells.
    u[:, axis] += 1.0 / cells
    return domain_center[None, :] + domain_half_extent * u


def _write_point_cloud(path: str, points: np.ndarray, color: Sequence[int]) -> None:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return
    rgba = np.tile(np.asarray(color, dtype=np.uint8).reshape(1, 4), (len(pts), 1))
    trimesh.PointCloud(pts, colors=rgba).export(path)


def _measure_face_consistency(
    class_map: np.ndarray,
    status_map: np.ndarray,
    confidence_map: np.ndarray,
    *,
    domain_center: np.ndarray,
    domain_half_extent: float,
    max_points: int,
    output_dir: str,
    chunk: int = 8,
) -> Dict[str, object]:
    cells = int(class_map.shape[0])
    signs = _build_class_sign_lookup()
    face_code = _build_face_code_lookup(signs)

    disagreement_points = _PointCollector(max_points)
    threshold_points = _PointCollector(max_points)
    other_termination_points = _PointCollector(max_points)

    summary: Dict[str, object] = {
        "total_internal_faces": int(3 * (cells - 1) * cells * cells),
        "axes": {},
    }

    for axis in range(3):
        axis_counts = {
            "active_active_faces": 0,
            "active_active_disagreements": 0,
            "active_active_disagreement_pct": 0.0,
            "active_inactive_faces": 0,
            "surface_terminations": 0,
            "surface_terminations_against_near_rejected": 0,
            "surface_terminations_involving_invalid_gradient_cell": 0,
            "disagreements_involving_invalid_gradient_cell": 0,
            "disagreements_involving_confidence_below_0_5": 0,
        }

        # Chunk along array axis 0 for bounded temporary memory.
        for s0 in range(0, cells, max(1, int(chunk))):
            s1 = min(s0 + max(1, int(chunk)), cells)

            if axis == 0:
                if s0 >= cells - 1:
                    break
                s1 = min(s1, cells - 1)
                a_cls = np.asarray(class_map[s0:s1, :, :])
                b_cls = np.asarray(class_map[s0 + 1 : s1 + 1, :, :])
                a_status = np.asarray(status_map[s0:s1, :, :])
                b_status = np.asarray(status_map[s0 + 1 : s1 + 1, :, :])
                a_conf = np.asarray(confidence_map[s0:s1, :, :], dtype=np.float32)
                b_conf = np.asarray(
                    confidence_map[s0 + 1 : s1 + 1, :, :], dtype=np.float32
                )
            elif axis == 1:
                a_cls = np.asarray(class_map[s0:s1, :-1, :])
                b_cls = np.asarray(class_map[s0:s1, 1:, :])
                a_status = np.asarray(status_map[s0:s1, :-1, :])
                b_status = np.asarray(status_map[s0:s1, 1:, :])
                a_conf = np.asarray(
                    confidence_map[s0:s1, :-1, :], dtype=np.float32
                )
                b_conf = np.asarray(
                    confidence_map[s0:s1, 1:, :], dtype=np.float32
                )
            else:
                a_cls = np.asarray(class_map[s0:s1, :, :-1])
                b_cls = np.asarray(class_map[s0:s1, :, 1:])
                a_status = np.asarray(status_map[s0:s1, :, :-1])
                b_status = np.asarray(status_map[s0:s1, :, 1:])
                a_conf = np.asarray(
                    confidence_map[s0:s1, :, :-1], dtype=np.float32
                )
                b_conf = np.asarray(
                    confidence_map[s0:s1, :, 1:], dtype=np.float32
                )

            a_active = a_cls != _INACTIVE_CLASS
            b_active = b_cls != _INACTIVE_CLASS
            both = a_active & b_active
            one = a_active ^ b_active

            a_code = np.zeros(a_cls.shape, dtype=np.uint8)
            b_code = np.zeros(b_cls.shape, dtype=np.uint8)
            if np.any(a_active):
                a_code[a_active] = face_code[(axis, 1)][a_cls[a_active]]
            if np.any(b_active):
                b_code[b_active] = face_code[(axis, 0)][b_cls[b_active]]

            disagreement = both & (a_code != b_code)
            a_terminates = a_active & ~b_active & (a_code != 0)
            b_terminates = ~a_active & b_active & (b_code != 0)
            termination = a_terminates | b_terminates

            rejected_neighbor = (
                a_terminates & (b_status == 2)
            ) | (
                b_terminates & (a_status == 2)
            )
            invalid_involved = (a_status == 3) | (b_status == 3)
            low_conf_involved = (
                (a_active & (a_conf < 0.5))
                | (b_active & (b_conf < 0.5))
            )

            axis_counts["active_active_faces"] += int(both.sum())
            axis_counts["active_active_disagreements"] += int(
                disagreement.sum()
            )
            axis_counts["active_inactive_faces"] += int(one.sum())
            axis_counts["surface_terminations"] += int(termination.sum())
            axis_counts[
                "surface_terminations_against_near_rejected"
            ] += int(rejected_neighbor.sum())
            axis_counts[
                "surface_terminations_involving_invalid_gradient_cell"
            ] += int((termination & invalid_involved).sum())
            axis_counts[
                "disagreements_involving_invalid_gradient_cell"
            ] += int((disagreement & invalid_involved).sum())
            axis_counts[
                "disagreements_involving_confidence_below_0_5"
            ] += int((disagreement & low_conf_involved).sum())

            if disagreement_points._size < disagreement_points.limit:
                disagreement_points.add(
                    _shared_face_points(
                        disagreement,
                        axis=axis,
                        chunk_start=s0,
                        cells=cells,
                        domain_center=domain_center,
                        domain_half_extent=domain_half_extent,
                    )
                )
            if threshold_points._size < threshold_points.limit:
                threshold_points.add(
                    _shared_face_points(
                        termination & rejected_neighbor,
                        axis=axis,
                        chunk_start=s0,
                        cells=cells,
                        domain_center=domain_center,
                        domain_half_extent=domain_half_extent,
                    )
                )
            if other_termination_points._size < other_termination_points.limit:
                other_termination_points.add(
                    _shared_face_points(
                        termination & ~rejected_neighbor,
                        axis=axis,
                        chunk_start=s0,
                        cells=cells,
                        domain_center=domain_center,
                        domain_half_extent=domain_half_extent,
                    )
                )

        axis_counts["active_active_disagreement_pct"] = (
            100.0
            * axis_counts["active_active_disagreements"]
            / max(axis_counts["active_active_faces"], 1)
        )
        summary["axes"][str(axis)] = axis_counts

    totals = {
        key: int(sum(axis_data[key] for axis_data in summary["axes"].values()))
        for key in (
            "active_active_faces",
            "active_active_disagreements",
            "active_inactive_faces",
            "surface_terminations",
            "surface_terminations_against_near_rejected",
            "surface_terminations_involving_invalid_gradient_cell",
            "disagreements_involving_invalid_gradient_cell",
            "disagreements_involving_confidence_below_0_5",
        )
    }
    totals["active_active_disagreement_pct"] = (
        100.0
        * totals["active_active_disagreements"]
        / max(totals["active_active_faces"], 1)
    )
    totals["threshold_termination_pct"] = (
        100.0
        * totals["surface_terminations_against_near_rejected"]
        / max(totals["surface_terminations"], 1)
    )
    summary["totals"] = totals

    _write_point_cloud(
        op.join(output_dir, "bad_faces_classifier_disagreement.ply"),
        disagreement_points.array(),
        (230, 40, 40, 255),
    )
    _write_point_cloud(
        op.join(output_dir, "bad_faces_threshold_termination.ply"),
        threshold_points.array(),
        (255, 140, 0, 255),
    )
    _write_point_cloud(
        op.join(output_dir, "bad_faces_other_termination.ply"),
        other_termination_points.array(),
        (255, 220, 0, 255),
    )

    summary["visualized_points"] = {
        "classifier_disagreement": int(len(disagreement_points.array())),
        "threshold_termination": int(len(threshold_points.array())),
        "other_termination": int(len(other_termination_points.array())),
    }
    return summary


def _build_pseudo_sdf(
    udf: np.ndarray,
    class_map: np.ndarray,
    *,
    output_dir: str,
    cell_slab: int,
) -> np.memmap:
    cells = int(class_map.shape[0])
    path = op.join(output_dir, "pseudo_sdf.float32.dat")
    pseudo = np.memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(cells, cells, cells, 8),
    )
    pseudo[:] = 1.0
    signs = _build_class_sign_lookup().astype(np.float32)

    slab = max(1, int(cell_slab))
    for i0 in range(0, cells, slab):
        i1 = min(i0 + slab, cells)
        classes = np.asarray(class_map[i0:i1])
        active = classes != _INACTIVE_CLASS
        if not np.any(active):
            continue
        u8 = _cell_corner_values(udf, i0, i1)
        out = np.asarray(pseudo[i0:i1])
        out[active] = u8[active] * signs[classes[active]]
        pseudo[i0:i1] = out

    pseudo.flush()
    return pseudo


def _mesh_quality(mesh: trimesh.Trimesh) -> Dict[str, object]:
    if mesh is None or len(mesh.faces) == 0:
        return {
            "vertices": 0,
            "faces": 0,
            "boundary_edges": 0,
            "nonmanifold_edges": 0,
            "watertight": False,
        }
    edges = np.sort(np.asarray(mesh.edges, dtype=np.int64), axis=1)
    packed = edges.view([("a", edges.dtype), ("b", edges.dtype)]).reshape(-1)
    _, counts = np.unique(packed, return_counts=True)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "boundary_edges": int((counts == 1).sum()),
        "nonmanifold_edges": int((counts > 2).sum()),
        "watertight": bool(mesh.is_watertight),
    }


def run_nsdudf_diagnostics(
    model: torch.nn.Module,
    udf_and_grad_f: Callable,
    *,
    options: DiagnosticOptions,
    output_dir: str,
    domain_center: Sequence[float],
    domain_half_extent: float,
    meshing_module=None,
) -> DiagnosticResult:
    """Run NSDUDF diagnostics and optionally reconstruct the pseudo-SDF mesh."""
    output_dir = op.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    n = int(options.n_grid_samples)
    if n < 2:
        raise ValueError(f"n_grid_samples must be >=2, got {n}")
    center = np.asarray(domain_center, dtype=np.float64).reshape(3)
    half = float(domain_half_extent)
    if not np.isfinite(half) or half <= 0.0:
        raise ValueError(f"Invalid domain_half_extent={half}")

    print(
        "[nsdudf diag]",
        f"grid_samples={n}",
        f"cells={n-1}",
        f"center={center.tolist()}",
        f"half={half:.9g}",
        f"output={output_dir}",
        flush=True,
    )

    udf, grad, grid_summary = _query_grid_to_memmaps(
        udf_and_grad_f,
        n=n,
        query_slab=options.query_slab,
        output_dir=output_dir,
    )

    class_map, status_map, confidence_map, cell_summary = _predict_cells(
        model,
        udf,
        grad,
        options=options,
        output_dir=output_dir,
    )

    face_summary = _measure_face_consistency(
        class_map,
        status_map,
        confidence_map,
        domain_center=center,
        domain_half_extent=half,
        max_points=int(options.max_visualization_points),
        output_dir=output_dir,
    )

    mesh: Optional[trimesh.Trimesh] = None
    mesh_summary: Dict[str, object] = {"extracted": False}
    if options.extract_mesh:
        if meshing_module is None:
            raise ValueError(
                "meshing_module is required when DiagnosticOptions.extract_mesh=True"
            )
        pseudo = _build_pseudo_sdf(
            udf,
            class_map,
            output_dir=output_dir,
            cell_slab=options.cell_slab,
        )
        cube_mesh = meshing_module.mesh_marching_cubes(pseudo)
        if cube_mesh is not None and len(cube_mesh.faces) > 0:
            world_vertices = center[None, :] + half * np.asarray(
                cube_mesh.vertices, dtype=np.float64
            )
            mesh = trimesh.Trimesh(
                vertices=world_vertices,
                faces=np.asarray(cube_mesh.faces, dtype=np.int64),
                process=False,
            )
            mesh_path = op.join(output_dir, "diagnostic_nsdudf_mesh.ply")
            mesh.export(mesh_path)
            mesh_summary = {
                "extracted": True,
                "path": mesh_path,
                **_mesh_quality(mesh),
            }
        else:
            mesh_summary = {"extracted": True, "empty": True}

    summary: Dict[str, object] = {
        "schema_version": 1,
        "options": {
            "n_grid_samples": n,
            "cells_per_axis": n - 1,
            "classifier_batch_size": int(options.classifier_batch_size),
            "query_slab": int(options.query_slab),
            "cell_slab": int(options.cell_slab),
            "normalize_udf": bool(options.normalize_udf),
            "max_avg_factor": float(options.max_avg_factor),
            "max_max_factor": float(options.max_max_factor),
            "loose_min_factor": float(options.loose_min_factor),
            "extract_mesh": bool(options.extract_mesh),
        },
        "domain": {
            "center": center.tolist(),
            "half_extent": half,
            "voxel_size_cube": 2.0 / (n - 1),
            "voxel_size_world": 2.0 * half / (n - 1),
        },
        "grid": grid_summary,
        "cells": cell_summary,
        "shared_faces": face_summary,
        "mesh": mesh_summary,
    }

    summary_path = op.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print("[nsdudf diag summary]", json.dumps({
        "accepted_cells": cell_summary["accepted_cells"],
        "near_rejected_cells": cell_summary["near_rejected_cells"],
        "invalid_gradient_cells": cell_summary[
            "accepted_invalid_gradient_cells"
        ],
        "face_disagreements": face_summary["totals"][
            "active_active_disagreements"
        ],
        "surface_terminations": face_summary["totals"][
            "surface_terminations"
        ],
        "threshold_terminations": face_summary["totals"][
            "surface_terminations_against_near_rejected"
        ],
        "mesh_boundary_edges": mesh_summary.get("boundary_edges"),
    }, sort_keys=True), flush=True)
    print("[nsdudf diag] wrote", summary_path, flush=True)

    return DiagnosticResult(summary=summary, mesh=mesh, output_dir=output_dir)
