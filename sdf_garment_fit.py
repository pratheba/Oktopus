#!/usr/bin/env python3
"""
sdf_garment_fit.py
==================

Python port of the SDF-based collision-free garment fitting from
cloth-fit-main (the C++/PolyFEM/OpenVDB pipeline).

This file mirrors the three core energies that show up in the C++:

  * FitForm           (FitForm.cpp:182-235)  -- pull garment ONTO avatar
                                                 (penalize sdf > sdf_initial)
  * SDFCollisionForm  (FitForm.cpp:450-502)  -- push garment OFF the avatar
                                                 (barrier when sdf < d_sep)
  * ContactForm       (ContactForm.hpp)       -- IPC self-collision via ipctk

plus a small SimilarityForm-like edge-length regularizer so the cloth does
not collapse.

Differences vs. the C++:
  * SDF backend: libigl `igl.signed_distance` (exact pseudonormal / winding
    sign on the triangle mesh) instead of OpenVDB voxel-grid + spline
    sampling. Same gradient (unit outward normal at the closest point), no
    voxel resolution to tune.
  * Quadrature: per-vertex sampling for clarity (the C++ uses an
    n_refs-level barycentric quadrature inside each garment triangle).
    Easy to swap in -- see `garment_sample_points`.
  * Optimizer: torch.optim.LBFGS with strong_wolfe line search instead of
    PolySolve's nonlinear solver + augmented-Lagrangian outer loop. The
    avatar is kept fixed here (the C++ does an avatar-morphing
    continuation; if you want that, just call `optimize` in a loop with
    interpolated avatar vertices -- see __main__).

Install
-------
    pip install torch numpy libigl ipctk trimesh

    `ipctk` is optional -- if it is missing, IPC self-collision is skipped.

Run
---
    python sdf_garment_fit.py \\
        --avatar  /path/to/trex_unitbb.ply \\
        --garment /path/to/top.obj \\
        --output  /path/to/top_fitted.obj \\
        --d-sep   0.005 \\
        --iterations 50

Author: ported by Claude for Pratheba, 2026.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

try:
    import igl
except ImportError as e:
    raise SystemExit(
        "libigl python bindings are required. Install with: pip install libigl"
    ) from e

try:
    import ipctk

    IPCTK_AVAILABLE = True
    # API shim: ipctk renamed `Collisions` -> `NormalCollisions` in v1.x.
    # Use whichever the installed version exposes.
    if hasattr(ipctk, "NormalCollisions"):
        _IPCTK_COLLISIONS_CLS = ipctk.NormalCollisions
    elif hasattr(ipctk, "Collisions"):
        _IPCTK_COLLISIONS_CLS = ipctk.Collisions
    else:
        _IPCTK_COLLISIONS_CLS = None
        IPCTK_AVAILABLE = False
except ImportError:
    IPCTK_AVAILABLE = False
    _IPCTK_COLLISIONS_CLS = None


# =============================================================================
# 1. SDF query as a PyTorch autograd Function
# =============================================================================
#
# We need d(sdf)/d(point) so PyTorch can backprop through the SDF energies
# into the garment vertex positions.  libigl is not differentiable, so we
# wrap it in a custom autograd Function that:
#   forward:  call igl.signed_distance, return signed distances S
#   backward: dS/dP = sign(S) * (P - C) / ||P - C||
#             where C is the closest point on the avatar surface (also
#             returned by igl.signed_distance). This is the unit outward
#             normal at the closest point, which is exactly grad(SDF).


class SDFQuery(torch.autograd.Function):
    """Differentiable signed-distance query against a fixed triangle mesh."""

    @staticmethod
    def forward(
        ctx,
        points: torch.Tensor,        # (N, 3) torch
        V_avatar_np: np.ndarray,     # (Va, 3)
        F_avatar_np: np.ndarray,     # (Fa, 3) int
    ) -> torch.Tensor:
        P_np = points.detach().cpu().numpy().astype(np.float64)

        # Guard against NaN/Inf from a diverging optimizer: replace any
        # bad coords with the avatar bounding-box centre. Without this,
        # one bad number poisons every subsequent iteration.
        bad = ~np.isfinite(P_np).all(axis=1)
        if bad.any():
            centre = 0.5 * (V_avatar_np.min(0) + V_avatar_np.max(0))
            P_np[bad] = centre

        # S: signed distance, I: closest face index, C: closest point.
        # libigl-python signature varies by release, so be defensive:
        try:
            out = igl.signed_distance(
                P_np, V_avatar_np, F_avatar_np, return_normals=False
            )
        except TypeError:
            out = igl.signed_distance(P_np, V_avatar_np, F_avatar_np)
        if len(out) == 4:
            S, _, C, _ = out
        else:
            S, _, C = out

        diff = P_np - C                                  # (N, 3)
        norm = np.linalg.norm(diff, axis=1, keepdims=True)
        # Use a *meaningful* floor: float64 underflows below ~1e-300, but
        # numpy still emits an "invalid value in divide" warning whenever
        # diff is a true zero vector (point exactly on the surface).
        safe_norm = np.where(norm > 1e-12, norm, 1.0)
        unit = diff / safe_norm                          # unit (P - C); 0 on surface
        sign = np.sign(S).reshape(-1, 1)
        sign[sign == 0] = 1.0                            # tie-break on surface
        grad_np = sign * unit                            # d S / d P, shape (N,3)

        # Final sanity guard.
        if not np.isfinite(grad_np).all() or not np.isfinite(S).all():
            print("[SDFQuery] WARNING: non-finite SDF result; clamping to 0.",
                  file=sys.stderr)
            grad_np = np.nan_to_num(grad_np, nan=0.0, posinf=0.0, neginf=0.0)
            S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)

        ctx.save_for_backward(
            torch.from_numpy(grad_np).to(points.device, dtype=points.dtype)
        )
        return torch.from_numpy(S).to(points.device, dtype=points.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # grad_output: (N,) -- dE/dS_i
        # We want dE/dP_i = dE/dS_i * dS_i/dP_i = grad_output * grad_local
        (grad_local,) = ctx.saved_tensors            # (N, 3)
        grad_P = grad_output.unsqueeze(-1) * grad_local
        return grad_P, None, None


def signed_distance(
    points: torch.Tensor, V_a_np: np.ndarray, F_a_np: np.ndarray
) -> torch.Tensor:
    """Convenience wrapper -- returns differentiable SDF values."""
    return SDFQuery.apply(points, V_a_np, F_a_np)


# =============================================================================
# 2. IPC self-collision as a PyTorch autograd Function (optional)
# =============================================================================
#
# Mirrors what ContactForm does in the C++: build collisions on the merged
# (avatar + garment) mesh, evaluate the barrier potential at distance dhat,
# return energy + analytic gradient.
#
# To match the C++ filter:
#   self_collision == False  ->  only avatar-garment pairs collide
#   self_collision == True   ->  garment-garment pairs collide too
# We implement the latter (set `can_collide` accordingly) since avatar-
# garment penetration is already handled by the SDF energy.


class IPCBarrier(torch.autograd.Function):
    """Differentiable IPC barrier energy on a fixed CollisionMesh.

    Works with both ipctk 0.x (`Collisions`) and 1.x (`NormalCollisions`).
    """

    @staticmethod
    def forward(
        ctx,
        V_full: torch.Tensor,        # (Vfull, 3) -- avatar then garment
        collision_mesh,              # ipctk.CollisionMesh
        dhat: float,
    ) -> torch.Tensor:
        V_np = V_full.detach().cpu().numpy().astype(np.float64)

        # Guard: if the optimizer probed to a non-finite state, return zero
        # energy + zero gradient instead of letting ipctk hit log(NaN).
        if not np.isfinite(V_np).all():
            zero_g = torch.zeros_like(V_full)
            ctx.save_for_backward(zero_g)
            return torch.tensor(0.0, device=V_full.device, dtype=V_full.dtype)

        # 1) Build collisions container. The class name differs by version.
        collisions = _IPCTK_COLLISIONS_CLS()
        collisions.build(collision_mesh, V_np, dhat)

        # 2) Construct the barrier potential. Some versions only accept
        # `BarrierPotential(dhat)`, others want `(barrier, dhat)`.
        try:
            barrier = ipctk.BarrierPotential(dhat)
        except TypeError:
            # Newer ipctk: pass a barrier function explicitly.
            base = (
                ipctk.ClampedLogBarrier()
                if hasattr(ipctk, "ClampedLogBarrier")
                else ipctk.Barrier()
            )
            barrier = ipctk.BarrierPotential(base, dhat)

        # 3) Energy + gradient. Some versions need a barrier_stiffness arg.
        try:
            E = float(barrier(collisions, collision_mesh, V_np))
            g = np.asarray(barrier.gradient(collisions, collision_mesh, V_np),
                           dtype=np.float64)
        except TypeError:
            E = float(barrier(collisions, collision_mesh, V_np, 1.0))
            g = np.asarray(
                barrier.gradient(collisions, collision_mesh, V_np, 1.0),
                dtype=np.float64,
            )

        g = g.reshape(V_np.shape)

        # Final guard against ipctk returning NaN/Inf for degenerate
        # contact geometry (e.g. exact 0 distance during a line-search probe).
        if not np.isfinite(g).all() or not np.isfinite(E):
            print("[IPCBarrier] WARNING: non-finite IPC result; clamping to 0.",
                  file=sys.stderr)
            g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            E = 0.0 if not np.isfinite(E) else E

        ctx.save_for_backward(
            torch.from_numpy(g).to(V_full.device, dtype=V_full.dtype)
        )
        return torch.tensor(E, device=V_full.device, dtype=V_full.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (g,) = ctx.saved_tensors
        return g * grad_output, None, None


# =============================================================================
# 3. Energy terms
# =============================================================================


def fit_energy(
    garment_v: torch.Tensor,
    V_a_np: np.ndarray,
    F_a_np: np.ndarray,
    initial_sdf: torch.Tensor,
    power: int = 2,
) -> torch.Tensor:
    """
    "Pull garment toward avatar" (mirrors FitForm::value_unweighted).

    Penalizes only when SDF has *grown* relative to the rest pose, i.e.
    when the garment moves AWAY from the body. Penalty is (delta)^power.
    """
    sdf = signed_distance(garment_v, V_a_np, F_a_np)
    delta = sdf - initial_sdf
    penalty = torch.clamp(delta, min=0.0).pow(power)
    return penalty.sum()


def collision_energy(
    garment_v: torch.Tensor,
    V_a_np: np.ndarray,
    F_a_np: np.ndarray,
    d_sep: float = 5e-3,
    power: int = 2,           # quadratic barrier: gentler initial gradient
) -> torch.Tensor:
    """
    "Push garment off the avatar" (mirrors SDFCollisionForm).

    Barrier that fires whenever sdf < d_sep (i.e. the cloth got closer
    than `d_sep` to the body, including penetration where sdf < 0).
    Energy is (d_sep - sdf)^power, only on the inside-or-too-close points.
    """
    sdf = signed_distance(garment_v, V_a_np, F_a_np)
    delta = d_sep - sdf
    penalty = torch.clamp(delta, min=0.0).pow(power)
    return penalty.sum()


def similarity_energy(
    garment_v: torch.Tensor,
    edges: torch.Tensor,
    rest_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Edge-length-preservation regularizer (cheap stand-in for the C++
    SimilarityForm, which preserves the surface metric tensor).
    """
    p0 = garment_v[edges[:, 0]]
    p1 = garment_v[edges[:, 1]]
    cur = torch.linalg.norm(p1 - p0, dim=1)
    return ((cur - rest_lengths) ** 2).sum()


# =============================================================================
# 4. Main pipeline
# =============================================================================


@dataclass
class FitConfig:
    fit_w: float = 1.0
    coll_w: float = 50.0          # was 1e3; cubed barrier was overshooting
    sim_w: float = 1.0e2
    ipc_w: float = 1.0
    d_sep: float = 5e-3      # SDF separation distance (metres, mesh units)
    dhat: float = 1e-3        # IPC barrier activation distance
    iterations: int = 50      # outer iterations of LBFGS.step
    lbfgs_max_iter: int = 20  # inner LBFGS iterations per outer step
    use_ipc: bool = True
    self_collision: bool = True
    device: str = "cpu"
    dtype: torch.dtype = torch.float64


def build_ipc_collision_mesh(
    V_a: np.ndarray,
    F_a: np.ndarray,
    V_g: np.ndarray,
    F_g: np.ndarray,
    self_collision: bool,
):
    """Construct an ipctk.CollisionMesh over (avatar + garment), with the
    can_collide filter set to mirror the C++ ContactForm setup."""
    V_full = np.vstack([V_a, V_g])
    F_full = np.vstack([F_a, F_g + V_a.shape[0]]).astype(np.int32)
    n_avatar_v = V_a.shape[0]

    # ipctk needs an explicit edge list for the collision mesh.
    # `igl.edges` exists in libigl >= 2.5; fall back to a manual derivation.
    try:
        edges = np.asarray(igl.edges(F_full), dtype=np.int32)
    except AttributeError:
        e = np.vstack([F_full[:, [0, 1]], F_full[:, [1, 2]], F_full[:, [2, 0]]])
        e = np.sort(e, axis=1)
        edges = np.unique(e, axis=0).astype(np.int32)

    # ipctk 1.x ctor: CollisionMesh(rest_positions, edges, faces).
    # ipctk 0.x sometimes used build_from_full_mesh(V, E, F).
    # Prefer the standard ctor; fall back if needed.
    cm = None
    for ctor in (
        lambda: ipctk.CollisionMesh(V_full, edges, F_full),
        lambda: ipctk.CollisionMesh.build_from_full_mesh(V_full, edges, F_full),
        lambda: ipctk.CollisionMesh.build_from_full_mesh(V_full, F_full),
    ):
        try:
            cm = ctor()
            break
        except (TypeError, AttributeError):
            continue
    if cm is None:
        raise RuntimeError(
            "Could not construct ipctk.CollisionMesh -- ipctk API changed."
        )

    if self_collision:
        # garment self-collision allowed; avatar self ignored (rigid-ish)
        def can_collide(vi: int, vj: int) -> bool:
            return vi >= n_avatar_v or vj >= n_avatar_v
    else:
        # only cross avatar-garment
        def can_collide(vi: int, vj: int) -> bool:
            return (vi < n_avatar_v) ^ (vj < n_avatar_v)

    cm.can_collide = can_collide
    return cm, V_full, n_avatar_v


def optimize(
    V_a: np.ndarray,
    F_a: np.ndarray,
    V_g: np.ndarray,
    F_g: np.ndarray,
    cfg: FitConfig,
    log_fn=print,
) -> np.ndarray:
    """Run the SDF + fit + IPC optimization. Returns the deformed garment
    vertex array of shape (Vg, 3)."""

    device = torch.device(cfg.device)
    V_a_np = V_a.astype(np.float64)
    F_a_np = F_a.astype(np.int64)

    # garment vertices as a tensor we can autodiff w.r.t.
    V_g_init = torch.tensor(V_g, dtype=cfg.dtype, device=device)
    dV = torch.zeros_like(V_g_init, requires_grad=True)

    # --- precompute edges + rest lengths for the similarity term ---
    E = igl.edges(F_g)                                      # (E, 2) int
    edges_t = torch.tensor(E, dtype=torch.long, device=device)
    rest_lengths = torch.linalg.norm(
        V_g_init[edges_t[:, 1]] - V_g_init[edges_t[:, 0]], dim=1
    ).detach()

    # --- precompute initial SDF (FitForm baseline) ---
    with torch.no_grad():
        initial_sdf = signed_distance(V_g_init, V_a_np, F_a_np).detach()
    log_fn(
        f"Initial SDF stats:  min={initial_sdf.min():.4f}  "
        f"max={initial_sdf.max():.4f}  "
        f"%inside={(initial_sdf < 0).float().mean().item()*100:.1f}%"
    )

    # --- IPC setup ---
    collision_mesh = None
    n_avatar_v = V_a.shape[0]
    if cfg.use_ipc:
        if not IPCTK_AVAILABLE:
            log_fn("ipctk not installed -- skipping IPC self-collision.")
        else:
            collision_mesh, _, n_avatar_v = build_ipc_collision_mesh(
                V_a_np, F_a_np, V_g, F_g.astype(np.int64), cfg.self_collision
            )
            log_fn(f"IPC: collision mesh built, dhat={cfg.dhat}")

    # Initial gradient on a deeply-penetrating mesh can be large; lr=0.1
    # with strong_wolfe line search is much more conservative than lr=1.0
    # and almost never overshoots into non-finite territory.
    optimizer = torch.optim.LBFGS(
        [dV],
        lr=0.1,
        max_iter=cfg.lbfgs_max_iter,
        history_size=10,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
    )

    V_a_torch = torch.tensor(V_a_np, dtype=cfg.dtype, device=device)

    history = []
    t0 = time.time()
    for outer in range(cfg.iterations):

        # closure-local counter for heartbeat printing
        call_count = {"n": 0, "t_iter_start": time.time()}

        def closure():
            t_call = time.time()
            optimizer.zero_grad()

            V_g_cur = V_g_init + dV
            if not torch.isfinite(V_g_cur).all():
                # diverged -- bail out instead of hanging in line search
                raise RuntimeError(
                    f"non-finite garment positions at iter {outer+1}, "
                    f"closure call {call_count['n']+1}. "
                    f"Try lowering --coll-w / --fit-w or raising --d-sep."
                )

            E_fit = cfg.fit_w * fit_energy(V_g_cur, V_a_np, F_a_np, initial_sdf)
            E_coll = cfg.coll_w * collision_energy(
                V_g_cur, V_a_np, F_a_np, d_sep=cfg.d_sep
            )
            E_sim = cfg.sim_w * similarity_energy(V_g_cur, edges_t, rest_lengths)

            E_total = E_fit + E_coll + E_sim

            if collision_mesh is not None:
                # full mesh = avatar (fixed) + garment (current)
                V_full = torch.cat([V_a_torch, V_g_cur], dim=0)
                E_ipc = cfg.ipc_w * IPCBarrier.apply(
                    V_full, collision_mesh, cfg.dhat
                )
                E_total = E_total + E_ipc
                closure._E_ipc = float(E_ipc.detach())
            else:
                closure._E_ipc = 0.0

            E_total.backward()
            closure._E_fit = float(E_fit.detach())
            closure._E_coll = float(E_coll.detach())
            closure._E_sim = float(E_sim.detach())
            closure._E_total = float(E_total.detach())

            call_count["n"] += 1
            dt = time.time() - t_call
            log_fn(
                f"  [iter {outer+1:2d} call {call_count['n']:2d}]  "
                f"E={closure._E_total:.3e}  "
                f"fit={closure._E_fit:.2e}  coll={closure._E_coll:.2e}  "
                f"sim={closure._E_sim:.2e}  ipc={closure._E_ipc:.2e}  "
                f"({dt:.1f}s)"
            )
            return E_total

        try:
            optimizer.step(closure)
        except RuntimeError as e:
            log_fn(f"ABORT: {e}")
            break

        with torch.no_grad():
            cur_sdf = signed_distance(V_g_init + dV, V_a_np, F_a_np)
            pen_pct = float((cur_sdf < 0).float().mean().item() * 100)

        log_fn(
            f"[iter {outer+1:3d}/{cfg.iterations}]  "
            f"E_total={closure._E_total:.4e}  "
            f"E_fit={closure._E_fit:.3e}  "
            f"E_coll={closure._E_coll:.3e}  "
            f"E_sim={closure._E_sim:.3e}  "
            f"E_ipc={closure._E_ipc:.3e}  "
            f"%inside={pen_pct:.2f}%  "
            f"sdf_min={cur_sdf.min():.4f}"
        )
        history.append(closure._E_total)

        # early exit if stationary
        if outer > 2 and abs(history[-1] - history[-2]) < 1e-9:
            log_fn("Converged (no further decrease).")
            break

    log_fn(f"Done in {time.time() - t0:.1f}s")
    V_g_final = (V_g_init + dV).detach().cpu().numpy()
    return V_g_final


# =============================================================================
# 5. CLI entry point
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--avatar", required=True, help=".obj/.ply avatar mesh")
    p.add_argument("--garment", required=True, help=".obj/.ply garment mesh")
    p.add_argument("--output", required=True, help="output garment .obj/.ply")
    p.add_argument("--fit-w", type=float, default=2.0)
    p.add_argument("--coll-w", type=float, default=1.0e3)
    p.add_argument("--sim-w", type=float, default=1.0e2)
    p.add_argument("--ipc-w", type=float, default=1.0)
    p.add_argument("--d-sep", type=float, default=5e-3,
                   help="SDF separation distance (model units)")
    p.add_argument("--dhat", type=float, default=1e-3,
                   help="IPC barrier activation distance")
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--lbfgs-max-iter", type=int, default=20)
    p.add_argument("--no-ipc", action="store_true",
                   help="disable IPC self-collision (still uses SDF)")
    p.add_argument("--no-self-collision", action="store_true",
                   help="IPC only checks avatar-vs-garment, not garment-vs-garment")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.isfile(args.avatar):
        print(f"ERROR: avatar not found: {args.avatar}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.garment):
        print(f"ERROR: garment not found: {args.garment}", file=sys.stderr)
        return 1

    print(f"Reading avatar:  {args.avatar}")
    V_a, F_a = igl.read_triangle_mesh(args.avatar)
    print(f"  V={V_a.shape[0]}  F={F_a.shape[0]}")

    print(f"Reading garment: {args.garment}")
    V_g, F_g = igl.read_triangle_mesh(args.garment)
    print(f"  V={V_g.shape[0]}  F={F_g.shape[0]}")

    cfg = FitConfig(
        fit_w=args.fit_w,
        coll_w=args.coll_w,
        sim_w=args.sim_w,
        ipc_w=args.ipc_w,
        d_sep=args.d_sep,
        dhat=args.dhat,
        iterations=args.iterations,
        lbfgs_max_iter=args.lbfgs_max_iter,
        use_ipc=not args.no_ipc,
        self_collision=not args.no_self_collision,
        device=args.device,
        dtype=torch.float32 if args.dtype == "float32" else torch.float64,
    )

    V_g_final = optimize(V_a, F_a, V_g, F_g, cfg)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    igl.write_triangle_mesh(args.output, V_g_final, F_g)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
