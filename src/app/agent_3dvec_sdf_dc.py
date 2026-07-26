"""Signed-distance reconstruction *benchmark* agent.

`AgentSDFDC` reuses the whole existing SDF inference path from
``AgentSDF`` / ``AgentBase`` (model loading, curve localization, grid
filling in ``action_ngcnet_inference``) and only replaces the final
surface-extraction step.

Instead of extracting a single mesh, it converts the Oktopus network SDF
grid into the exact ``(S, U, resX, resY, resZ, isoValue)`` format expected
by the "Dual Contouring of Signed Distance Data" reference
(``scripts/run_ours.py``) and then runs *all* the reconstruction methods
that script benchmarks:

    * Marching Cubes            (contouring, ContouringMethod.MarchingCubes)
    * Dual Contouring           (contouring, ContouringMethod.DualContouring)
    * Ours / DC-SDD             (contouring, ContouringMethod.Ours)
    * Reach For The Arcs        (gpytoolbox.reach_for_the_arcs)
    * Kohlbrenner "cones"       (utility.kohlbrenner_reconstruction, method='cones')
    * Kohlbrenner "RC"          (utility.kohlbrenner_reconstruction, method='RC')  [needs GT mesh]

Every method is wrapped independently: an unbuilt external (RFTA CGAL,
maximal-empty-spheres, sdf-weighted-delaunay) or a method that raises is
logged and skipped, the others still run. Each result is saved and its
runtime printed, exactly like ``run_ours.py``.

Nothing in the existing code base is modified; this is a drop-in subclass
selected by the ``inference_3dvec_dc.py`` runner.

--------------------------------------------------------------------------
Coordinate bridge (why the format "just works")
--------------------------------------------------------------------------
``MCGrid`` stores ``val_grid`` as a flat ``(N+1)**3`` array with
``k_basis = [1, N+1, (N+1)**2]`` (x fastest) and world position
``p = ijk*step + origin``, ``origin = [-size]*3``, ``step = 2*size/N``.

The reference builds its grid with ``contouring.build_grid((n,n,n), min, max)``
which is ``igl::grid`` scaled into ``[min, max]^3``; that yields the *same*
coordinate ``-size + i*step``. So we build ``U`` at the agent's own world
extent ``[-size, size]`` and reindex ``S`` from ``val_grid`` by rounding
``(U - origin)/step`` back to integer cell coordinates. ``U`` therefore
carries true world XYZ, the contouring core places vertices from those
positions, and the returned meshes come out already in the agent's world
frame -- directly comparable to the pipeline's existing MC / RFTA outputs
(no axis flip, no rescale needed).
"""

import os
import os.path as op
import sys
import time

import numpy as np
import torch
import trimesh

from agent_3dvec_sdf import AgentSDF


# Default location of the reference repo on this machine (override with the
# DCSDD_REPO env var or the `dcsdd_repo` config/yaml key).
_DEFAULT_DCSDD_REPO = op.expanduser(
    "~/Downloads/dual-contouring-of-signed-distance-data-main"
)


class AgentSDFDC(AgentSDF):
    """SDF agent whose extractor runs the full ``run_ours.py`` method suite."""

    # ------------------------------------------------------------------
    # Reference-repo import
    # ------------------------------------------------------------------
    def _load_dcsdd(self, config):
        """Import the reference ``contouring`` package (+ ``utility``, ``gpy``).

        The repo must have been built once so that
        ``src/python/contouring/_contouring_cpp_module*.so`` exists
        (see RECON_BENCH_README.md). Returns ``(contouring, utility, gpy)``;
        ``utility``/``gpy`` are ``None`` if unavailable (only the contouring
        methods will then run).
        """
        repo = (
            (config.get("dcsdd_repo") if hasattr(config, "get") else None)
            or os.environ.get("DCSDD_REPO")
            or _DEFAULT_DCSDD_REPO
        )
        repo = op.abspath(op.expanduser(repo))
        if not op.isdir(repo):
            raise FileNotFoundError(
                f"DC-SDD repo not found at {repo!r}. Set env DCSDD_REPO or the "
                "'dcsdd_repo' config key to the checkout of "
                "dual-contouring-of-signed-distance-data."
            )

        # Mirror scripts/context.py: put ROOT, src/python and utility on path.
        for p in (repo, op.join(repo, "src", "python"), op.join(repo, "utility")):
            if p not in sys.path:
                sys.path.insert(0, p)

        try:
            import contouring  # noqa: E402  (the built package)
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "Could not import the 'contouring' package. Build the C++ "
                "bindings first: from the repo root run `mkdir -p build && cd "
                "build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j`. "
                f"(repo={repo!r})\nOriginal error: {exc}"
            ) from exc

        try:
            import utility  # noqa: E402
        except Exception as exc:  # pragma: no cover
            print(f"[dcsdd] utility import failed ({exc}); "
                  "Kohlbrenner methods will be skipped.")
            utility = None

        try:
            import gpytoolbox as gpy  # noqa: E402
        except Exception as exc:  # pragma: no cover
            print(f"[dcsdd] gpytoolbox import failed ({exc}); "
                  "Reach-for-the-Arcs will be skipped.")
            gpy = None

        return contouring, utility, gpy

    # ------------------------------------------------------------------
    # Format bridge: MCGrid  ->  (S, U, res) in the reference's convention
    # ------------------------------------------------------------------

    def _grid_to_S_U(self, sdf_grid, contouring):
        """Return DCSDD-ordered S, U, unevaluated mask and samples/axis."""
        cfg = sdf_grid.grid_config
        N = int(sdf_grid.reso)
        N1 = N + 1
        size = float(getattr(sdf_grid, "size", -float(cfg["origin"][0])))
        origin = np.asarray(cfg["origin"], dtype=np.float64).reshape(3)
        step = float(cfg["step"])
        k_basis = np.asarray(cfg["k_basis"], dtype=np.int64)

        U = np.asarray(
            contouring.build_grid((N1, N1, N1), -size, size),
            dtype=np.float64,
        )
        if U.shape != (N1 ** 3, 3):
            raise ValueError(
                f"build_grid returned {U.shape}, expected {(N1 ** 3, 3)}"
            )

        # Map DCSDD/grid ordering back to MCGrid's x-fast ordering.
        ijk = np.rint((U - origin[None, :]) / step).astype(np.int64)

        if np.any(ijk < 0) or np.any(ijk > N):
            lo = ijk.min(axis=0)
            hi = ijk.max(axis=0)
            raise ValueError(
                f"Recovered grid indices outside [0, {N}]: min={lo}, max={hi}"
            )

        flat = ijk @ k_basis

        raw_S = np.asarray(
            sdf_grid.val_grid,
            dtype=np.float64,
        ).reshape(-1)

        S = raw_S[flat]

           
        # In the current Oktopus field construction, untouched grid samples
        # retain the explicit background sentinel +10. empty_marks is stale in
        # this inference path, so diagnose the field using the actual values.
        unevaluated = np.isclose(S, 10.0, rtol=0.0, atol=1e-12)


        return S, U, unevaluated, N1


    def _print_dcsdd_grid_diagnostics(
        self,
        S,
        U,
        unevaluated,
        N1,
        level,
    ):
        """Print diagnostics for exactly the grid that DCSDD will receive."""
        S = np.asarray(S, dtype=np.float64).reshape(-1)
        U = np.asarray(U, dtype=np.float64).reshape(-1, 3)
        unevaluated = np.asarray(unevaluated, dtype=bool).reshape(-1)

        expected = N1 ** 3
        if len(S) != expected or len(U) != expected:
            raise ValueError(
                f"Diagnostic grid mismatch: len(S)={len(S)}, "
                f"len(U)={len(U)}, expected={expected}"
            )

        if len(unevaluated) != expected:
            raise ValueError(
                f"Unevaluated-mask mismatch: {len(unevaluated)} != {expected}"
            )

        finite = np.isfinite(S)
        evaluated = (~unevaluated) & finite

        def _percentile_string(values):
            values = np.asarray(values, dtype=np.float64)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                return "no finite values"

            qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
            vals = np.percentile(values, qs)
            return " ".join(
                f"p{q}={v:.6g}"
                for q, v in zip(qs, vals)
            )

        print(
            f"[dcsdd:grid] samples={len(S)} "
            f"shape={N1}x{N1}x{N1} "
            f"level={level:.9g}"
        )
        print(
            f"[dcsdd:grid] finite={int(finite.sum())} "
            f"evaluated={int(evaluated.sum())} "
            f"unevaluated={int(unevaluated.sum())} "
            f"exact_plus10={int(np.count_nonzero(S == 10.0))}"
        )
        print(f"[dcsdd:grid] S all: {_percentile_string(S)}")
        print(
            f"[dcsdd:grid] S evaluated: "
            f"{_percentile_string(S[evaluated])}"
        )

        # DCSDD/C++ flat index is:
        # i + N1 * (j + N1 * k), i.e. x is fastest.
        # order='F' gives A[i, j, k] from that flat layout.
        G = S.reshape((N1, N1, N1), order="F")
        E = unevaluated.reshape((N1, N1, N1), order="F")

        below = G < level
        above = G > level

        # Exact C++ edge rule is strict opposite signs:
        # (S0 - level) * (S1 - level) < 0.
        sign_x = (
            (below[:-1, :, :] & above[1:, :, :])
            | (above[:-1, :, :] & below[1:, :, :])
        )
        sign_y = (
            (below[:, :-1, :] & above[:, 1:, :])
            | (above[:, :-1, :] & below[:, 1:, :])
        )
        sign_z = (
            (below[:, :, :-1] & above[:, :, 1:])
            | (above[:, :, :-1] & below[:, :, 1:])
        )

        touch_x = sign_x & (E[:-1, :, :] | E[1:, :, :])
        touch_y = sign_y & (E[:, :-1, :] | E[:, 1:, :])
        touch_z = sign_z & (E[:, :, :-1] | E[:, :, 1:])

        n_sign_edges = (
            int(sign_x.sum())
            + int(sign_y.sum())
            + int(sign_z.sum())
        )
        n_touch_unevaluated = (
            int(touch_x.sum())
            + int(touch_y.sum())
            + int(touch_z.sum())
        )

        # Same criterion as Cell::Cell:
        # at least one strictly negative and one strictly positive corner.
        cell_below = (
            below[:-1, :-1, :-1]
            | below[1:, :-1, :-1]
            | below[:-1, 1:, :-1]
            | below[1:, 1:, :-1]
            | below[:-1, :-1, 1:]
            | below[1:, :-1, 1:]
            | below[:-1, 1:, 1:]
            | below[1:, 1:, 1:]
        )
        cell_above = (
            above[:-1, :-1, :-1]
            | above[1:, :-1, :-1]
            | above[:-1, 1:, :-1]
            | above[1:, 1:, :-1]
            | above[:-1, :-1, 1:]
            | above[1:, :-1, 1:]
            | above[:-1, 1:, 1:]
            | above[1:, 1:, 1:]
        )
        interesting = cell_below & cell_above

        cell_touches_unevaluated = (
            E[:-1, :-1, :-1]
            | E[1:, :-1, :-1]
            | E[:-1, 1:, :-1]
            | E[1:, 1:, :-1]
            | E[:-1, :-1, 1:]
            | E[1:, :-1, 1:]
            | E[:-1, 1:, 1:]
            | E[1:, 1:, 1:]
        )

        print(
            f"[dcsdd:grid] sign_changing_edges={n_sign_edges} "
            f"x={int(sign_x.sum())} "
            f"y={int(sign_y.sum())} "
            f"z={int(sign_z.sum())}"
        )
        print(
            "[dcsdd:grid] sign_changing_edges_touching_unevaluated="
            f"{n_touch_unevaluated} "
            f"x={int(touch_x.sum())} "
            f"y={int(touch_y.sum())} "
            f"z={int(touch_z.sum())}"
        )
        print(
            f"[dcsdd:grid] interesting_cells={int(interesting.sum())} "
            "interesting_cells_touching_unevaluated="
            f"{int((interesting & cell_touches_unevaluated).sum())}"
        )

        grid_min = U.min(axis=0)
        grid_max = U.max(axis=0)
        print(
            f"[dcsdd:grid] U bbox min={grid_min.tolist()} "
            f"max={grid_max.tolist()}"
        )

        if np.any(evaluated):
            evaluated_U = U[evaluated]
            print(
                "[dcsdd:grid] evaluated-sample bbox "
                f"min={evaluated_U.min(axis=0).tolist()} "
                f"max={evaluated_U.max(axis=0).tolist()}"
            )

    def _active_cloud(self, sdf_grid):
        """Return actually evaluated Oktopus samples for point-based methods.

        Unevaluated grid points retain the explicit +10 sentinel in the
        current field-construction path. empty_marks is stale here, so use
        the stored values themselves.
        """
        vals = np.asarray(
            sdf_grid.val_grid,
            dtype=np.float64,
        ).reshape(-1)

        active = np.isfinite(vals)
        active &= ~np.isclose(
            vals,
            10.0,
            rtol=0.0,
            atol=1e-12,
        )

        rows = np.flatnonzero(active)

        pts = np.asarray(
            sdf_grid.idx2pts(rows),
            dtype=np.float64,
        ).reshape(-1, 3)

        active_vals = vals[rows]

        print(
            f"[dcsdd:cloud] samples={len(active_vals)} "
            f"negative={int(np.count_nonzero(active_vals < 0.0))} "
            f"positive={int(np.count_nonzero(active_vals > 0.0))} "
            f"zero={int(np.count_nonzero(active_vals == 0.0))}"
        )

        return pts, active_vals


    # ------------------------------------------------------------------
    # The run_ours.py method suite
    # ------------------------------------------------------------------
    def _ours_options(self, contouring, config):
        """ContouringOptions dict for the DC-SDD "Ours" method (paper defaults,
        overridable via `dcsdd_*` config keys)."""
        cc = contouring._contouring_cpp_module
        def g(key, default):
            v = config.get(f"dcsdd_{key}") if hasattr(config, "get") else None
            return default if v is None else v
        return {
            "method": cc.ContouringMethod.Ours,
            "outer_iters": int(g("outer_iters", 100)),
            "inner_iters": int(g("inner_iters", 100)),
            "hermite_update": bool(g("hermite_update", True)),
            "new_hermite_pos_weight": float(g("new_hermite_pos_weight", 0.2)),
            "new_face_pos_weight": float(g("new_face_pos_weight", 0.2)),
            "new_hermite_normal_weight": float(g("new_hermite_normal_weight", 0.2)),
            "mu": float(g("mu", 0.1)),
            "dc_weight": float(g("dc_weight", 0.02)),
            "verbose": bool(g("dcsdd_verbose", False)),
            "batch_size": int(g("batch_size", 200000)),
        }

    def run_all_reconstructions(self, sdf_grid, config, name):
        """Run every method run_ours.py runs on the network's SDF field.

        Saves each mesh into ``<output_folder>/<name>/`` and prints per-method
        timings. Returns ``{method: trimesh.Trimesh}`` for the ones that
        succeeded (empty meshes for skipped/failed methods are omitted).
        """
        contouring, utility, gpy = self._load_dcsdd(config)

        output_folder = config.get("output_folder", ".") if hasattr(config, "get") else "."
        checkpoint = config.get("checkpoint", "eval") if hasattr(config, "get") else "eval"
        level = float(config.get("level", sdf_grid.grid_config.get("level", 0.0))
                      if hasattr(config, "get") else 0.0)

        out_dir = op.join(output_folder, name)
        os.makedirs(out_dir, exist_ok=True)
        reso = int(sdf_grid.reso)

        # Which methods to run (default: all).
        requested = None
        if hasattr(config, "get"):
            requested = config.get("dcsdd_methods")
        if isinstance(requested, str):
            requested = [m.strip() for m in requested.split(",") if m.strip()]
        default_methods = ["mc", "dc", "ours", "rfta", "mnm1", "mnm2"]
        methods = list(requested) if requested else default_methods


        S_raw, U, unevaluated, N1 = self._grid_to_S_U(
            sdf_grid,
            contouring,
        )

        self._print_dcsdd_grid_diagnostics(
            S=S_raw,
            U=U,
            unevaluated=unevaluated,
            N1=N1,
            level=level,
        )

# Marching Cubes accepts isoValue, but the current DC/DCSDD C++ paths
# internally use zero when detecting cells and generating quads.
# Shift all methods to a common zero level.
        S = S_raw - level
        contour_iso = 0.0

        cc = contouring._contouring_cpp_module

        results = {}
        timings = {}

        def _save(method, V, F):
            if V is None or F is None or len(V) == 0 or len(F) == 0:
                print(f"[dcsdd:{method}] empty result, not saved")
                return

            V = np.asarray(V, dtype=np.float64)
            F = np.asarray(F, dtype=np.int64)

            if not np.all(np.isfinite(V)):
                bad = int(np.count_nonzero(~np.isfinite(V)))
                raise ValueError(
                    f"{method} returned {bad} non-finite vertex coordinates"
                )

            mesh = trimesh.Trimesh(
                vertices=V,
                faces=F,
                process=False,
            )

            path = op.join(
                out_dir,
                f"{name}_{method}_{checkpoint}_mesh{reso}.ply",
            )
            mesh.export(path)
            results[method] = mesh

            mesh_min = V.min(axis=0)
            mesh_max = V.max(axis=0)
            mesh_extent = mesh_max - mesh_min

            grid_min = U.min(axis=0)
            grid_max = U.max(axis=0)

            evaluated = ~unevaluated
            if np.any(evaluated):
                eval_min = U[evaluated].min(axis=0)
                eval_max = U[evaluated].max(axis=0)
            else:
                eval_min = np.full(3, np.nan)
                eval_max = np.full(3, np.nan)

            # trimesh triangulates returned DCSDD quads when constructing the mesh.
            signed_volume = float(mesh.volume)
            if not mesh.is_watertight:
                orientation = "undetermined_nonwatertight"
            elif signed_volume > 0.0:
                orientation = "outward_positive_signed_volume"
            elif signed_volume < 0.0:
                orientation = "inward_negative_signed_volume"
            else:
                orientation = "degenerate_zero_signed_volume"

            print(
                f"[dcsdd:{method}] saved {path} "
                f"V={len(mesh.vertices)} F={len(mesh.faces)} "
                f"time={timings.get(method, float('nan')):.4f}s"
            )
            print(
                f"[dcsdd:{method}] mesh bbox "
                f"min={mesh_min.tolist()} "
                f"max={mesh_max.tolist()} "
                f"extent={mesh_extent.tolist()}"
            )
            print(
                f"[dcsdd:{method}] reference grid bbox "
                f"min={grid_min.tolist()} "
                f"max={grid_max.tolist()}"
            )
            print(
                f"[dcsdd:{method}] evaluated-sample bbox "
                f"min={eval_min.tolist()} "
                f"max={eval_max.tolist()}"
            )
            print(
                f"[dcsdd:{method}] watertight={mesh.is_watertight} "
                f"winding_consistent={mesh.is_winding_consistent} "
                f"signed_volume={signed_volume:.9g} "
                f"orientation={orientation}"
            )

        def _contour(method_name, opts):
            t0 = time.time()
            V, F = contouring.py_contouring(
                S,
                U,
                N1,
                N1,
                N1,
                contour_iso,
                opts,
                None,
                None,
            )
            timings[method_name] = time.time() - t0
            _save(method_name, V, F)

        # --- Contouring family (full dense grid, sign-based) ------------
        if "mc" in methods:
            try:
                _contour("mc", {"method": cc.ContouringMethod.MarchingCubes})
            except Exception as exc:
                print(f"[dcsdd:mc] FAILED: {exc}")

        if "dc" in methods:
            try:
                _contour("dc", {"method": cc.ContouringMethod.DualContouring})
            except Exception as exc:
                print(f"[dcsdd:dc] FAILED: {exc}")

        if "ours" in methods:
            try:
                _contour("ours", self._ours_options(contouring, config))
            except Exception as exc:
                print(f"[dcsdd:ours] FAILED: {exc}")

        # --- Point-based methods (active cells only) --------------------
        need_cloud = any(m in methods for m in ("rfta", "mnm1", "mnm2"))
        if need_cloud:
            Ua, Sa = self._active_cloud(sdf_grid)
            has_both = bool(np.any(Sa < level)) and bool(np.any(Sa > level))

        if "rfta" in methods:
            if gpy is None:
                print("[dcsdd:rfta] skipped (gpytoolbox unavailable)")
            elif not has_both:
                print("[dcsdd:rfta] skipped (need SDF samples on both sides of "
                      f"level {level})")
            else:
                try:
                    t0 = time.time()
                    S_rfta = -(Sa - level)

                    print(
                        f"[dcsdd:rfta] sign-flipped Oktopus SDF: "
                        f"negative_inside={int(np.count_nonzero(S_rfta < 0.0))} "
                        f"positive_outside={int(np.count_nonzero(S_rfta > 0.0))} "
                        f"range=[{S_rfta.min():.6g}, {S_rfta.max():.6g}]"
                    )
                    Vr, Fr = gpy.reach_for_the_arcs(Ua, S_rfta, verbose=False)
                    timings["rfta"] = time.time() - t0
                    _save("rfta", Vr, Fr)
                except Exception as exc:
                    print(f"[dcsdd:rfta] FAILED: {exc}")

        if "mnm1" in methods:
            if utility is None:
                print("[dcsdd:mnm1] skipped (utility unavailable)")
            else:
                try:
                    t0 = time.time()
                    Vk, Fk = utility.kohlbrenner_reconstruction(
                        Ua, Sa - level, method="cones"
                    )
                    timings["mnm1"] = time.time() - t0
                    _save("mnm1", Vk, Fk)
                except Exception as exc:
                    print(f"[dcsdd:mnm1] FAILED (needs external "
                          f"maximal-empty-spheres build): {exc}")

        if "mnm2" in methods:
            gt = config.get("dcsdd_gt_mesh") if hasattr(config, "get") else None
            if utility is None:
                print("[dcsdd:mnm2] skipped (utility unavailable)")
            elif not gt or not op.isfile(gt):
                print("[dcsdd:mnm2] skipped (Kohlbrenner 'RC' needs a GT mesh; "
                      "set 'dcsdd_gt_mesh' to an .obj/.ply path)")
            else:
                try:
                    import gpytoolbox as _gpy
                    V_gt, F_gt = _gpy.read_mesh(gt)
                    t0 = time.time()
                    Vk, Fk = utility.kohlbrenner_reconstruction(
                        Ua, Sa - level, V_gt=V_gt, F_gt=F_gt, method="RC"
                    )
                    timings["mnm2"] = time.time() - t0
                    _save("mnm2", Vk, Fk)
                except Exception as exc:
                    print(f"[dcsdd:mnm2] FAILED (needs external "
                          f"sdf-weighted-delaunay build): {exc}")

        print(f"[dcsdd] timings for {name} @ {reso}^3: " +
              ", ".join(f"{m}={timings[m]:.4f}s" for m in timings))
        return results

    # ------------------------------------------------------------------
    # Hook into the existing pipeline
    # ------------------------------------------------------------------
    def extract_surface_mesh(self, sdf_grid, config=None, mc_method="extract_mesh",
                             context=""):
        """Override the single-mesh extractor.

        With ``surface_extraction`` (or ``mesh_extractor``) set to one of
        ``run_ours`` / ``dcsdd`` / ``all`` (this agent's default), run the full
        method suite and return a representative mesh so the base
        ``action_ngcnet_inference`` still exports one file. Any single method
        name (``mc`` / ``dc`` / ``ours`` / ``rfta`` / ``mnm1``) runs just that
        one. Anything else defers to the stock ``AgentSDF`` behaviour.
        """
        config = {} if config is None else config
        spec = config.get(
            "surface_extraction", config.get("mesh_extractor", "run_ours")
        )
        method = spec if isinstance(spec, str) else "run_ours"
        method = method.lower()

        name = ""
        if context:
            name = str(context).split(":")[-1]
        if not name and hasattr(config, "get"):
            name = str(config.get("shape", "shape"))
        name = name or "shape"

        suite_aliases = {"run_ours", "dcsdd", "dc_sdd", "all", "dcsdd_all", "bench"}
        single_aliases = {"mc", "dc", "ours", "rfta", "mnm1", "mnm2"}

        if method in suite_aliases or method in single_aliases:
            cfg = dict(config)
            if method in single_aliases:
                cfg["dcsdd_methods"] = [method]
            results = self.run_all_reconstructions(sdf_grid, cfg, name)
            # Representative mesh for the base pipeline's single export.
            for key in ("ours", "dc", "mc", "rfta", "mnm1", "mnm2"):
                if key in results:
                    return results[key]
            return trimesh.Trimesh(
                vertices=np.zeros((0, 3)),
                faces=np.zeros((0, 3), dtype=np.int64),
                process=False,
            )

        # Fall back to marching_cubes / reach_for_the_arcs from AgentSDF.
        return super().extract_surface_mesh(
            sdf_grid, config=config, mc_method=mc_method, context=context
        )
