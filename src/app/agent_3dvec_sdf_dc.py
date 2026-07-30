"""Oktopus learned-SDF reconstruction comparison agent.

This adapter keeps Oktopus's native Marching Cubes as the ``mc`` baseline and
passes the exact same native grid samples/coordinates to the DCSDD reference
implementation for ``dc`` and ``ours``.

Methods:
    mc      Oktopus native ``sdf_grid.extract_mesh()``
    mc_ref  DCSDD repository Marching Cubes (diagnostic)
    dc      estimated-Hermite Dual Contouring
    ours    Dual Contouring of Signed Distance Data (DCSDD)
    rfta    Reach for the Arcs
    mnm1    Kohlbrenner cones (external executable required)
    mnm2    Kohlbrenner RC (GT mesh + external executable required)
"""

import os
import os.path as op
import sys
import time

import numpy as np
import trimesh

from agent_3dvec_sdf import AgentSDF


_DEFAULT_DCSDD_REPO = op.expanduser(
    "~/Downloads/dual-contouring-of-signed-distance-data-main"
)


class AgentSDFDC(AgentSDF):
    """Run Oktopus native MC and the DCSDD reference methods on one SDF grid."""

    def _load_dcsdd(self, config):
        repo = (
            (config.get("dcsdd_repo") if hasattr(config, "get") else None)
            or os.environ.get("DCSDD_REPO")
            or _DEFAULT_DCSDD_REPO
        )
        repo = op.abspath(op.expanduser(repo))
        if not op.isdir(repo):
            raise FileNotFoundError(
                f"DCSDD repo not found at {repo!r}. Set DCSDD_REPO or "
                "--dcsdd-repo."
            )

        for path in (repo, op.join(repo, "src", "python"), op.join(repo, "utility")):
            if path not in sys.path:
                sys.path.insert(0, path)

        try:
            import contouring
        except Exception as exc:
            raise ImportError(
                "Could not import the built DCSDD contouring package. "
                f"Repo: {repo!r}. Original error: {exc}"
            ) from exc

        try:
            import utility
        except Exception as exc:
            print(f"[dcsdd] utility import failed ({exc}); mnm1/mnm2 skipped")
            utility = None

        try:
            import gpytoolbox as gpy
        except Exception as exc:
            print(f"[dcsdd] gpytoolbox import failed ({exc}); rfta skipped")
            gpy = None

        return contouring, utility, gpy

    @staticmethod
    def _native_grid_config(sdf_grid):
        cfg = sdf_grid.grid_config
        N = int(sdf_grid.reso)
        N1 = N + 1
        origin = np.asarray(cfg["origin"], dtype=np.float64).reshape(3)
        step = float(cfg["step"])
        return N, N1, origin, step

    def _grid_to_S_U(self, sdf_grid):
        """Build DCSDD input directly from Oktopus's native grid.

        ``idx2pts`` and ``val_grid`` are first read in Oktopus's own flat order,
        then reordered only if necessary to DCSDD's required x-fast order:

            flat = i + N1 * (j + N1 * k)
        """
        N, N1, origin, step = self._native_grid_config(sdf_grid)

        S_native = np.asarray(sdf_grid.val_grid, dtype=np.float64).reshape(-1)
        expected_count = N1 ** 3
        if S_native.size != expected_count:
            raise ValueError(
                f"Oktopus grid has {S_native.size} samples, expected "
                f"(reso+1)^3={expected_count} for reso={N}"
            )

        native_rows = np.arange(expected_count, dtype=np.int64)
        U_native = np.asarray(
            sdf_grid.idx2pts(native_rows), dtype=np.float64
        ).reshape(-1, 3)
        if U_native.shape != (expected_count, 3):
            raise ValueError(
                f"idx2pts returned {U_native.shape}, expected {(expected_count, 3)}"
            )

        ijk = np.rint((U_native - origin[None, :]) / step).astype(np.int64)
        if np.any(ijk < 0) or np.any(ijk > N):
            raise ValueError(
                "Native idx2pts coordinates do not map into the expected grid: "
                f"ijk min={ijk.min(axis=0)}, max={ijk.max(axis=0)}, N={N}"
            )

        expected_flat = (
            ijk[:, 0]
            + N1 * (ijk[:, 1] + N1 * ijk[:, 2])
        )
        if np.unique(expected_flat).size != expected_count:
            raise ValueError("Native grid coordinate-to-index mapping is not one-to-one")

        mismatch_count = int(
            np.count_nonzero(expected_flat != np.arange(expected_count))
        )
        order = np.argsort(expected_flat)

        S = np.ascontiguousarray(S_native[order], dtype=np.float64)
        U = np.ascontiguousarray(U_native[order], dtype=np.float64)

        # Use MCGrid's bookkeeping instead of guessing from the value +10.
        # A predicted value can in principle equal the sentinel, while an
        # unmarked point is unambiguously unevaluated.
        if hasattr(sdf_grid, "empty_marks"):
            unevaluated_native = np.asarray(
                sdf_grid.empty_marks, dtype=bool
            ).reshape(-1)
            if unevaluated_native.size != expected_count:
                raise ValueError(
                    "empty_marks and val_grid have different sizes: "
                    f"{unevaluated_native.size} vs {expected_count}"
                )
            unevaluated = np.ascontiguousarray(
                unevaluated_native[order], dtype=bool
            )
        else:
            unevaluated = np.isclose(S, 10.0, rtol=0.0, atol=1e-12)

        # Verify that reordering produced exactly x-fast indexing.
        if not np.array_equal(expected_flat[order], np.arange(expected_count)):
            raise RuntimeError("Failed to reorder native Oktopus grid to x-fast order")

        print(
            f"[dcsdd:native-grid] samples={expected_count} "
            f"shape={N1}x{N1}x{N1} native_order_mismatches={mismatch_count}"
        )
        return S, U, unevaluated, N1

    def _active_cloud(self, sdf_grid):
        """Return finite evaluated Oktopus samples for point-based methods."""
        vals = np.asarray(sdf_grid.val_grid, dtype=np.float64).reshape(-1)
        active = np.isfinite(vals)
        if hasattr(sdf_grid, "empty_marks"):
            empty = np.asarray(sdf_grid.empty_marks, dtype=bool).reshape(-1)
            if empty.size != vals.size:
                raise ValueError(
                    "empty_marks and val_grid have different sizes: "
                    f"{empty.size} vs {vals.size}"
                )
            active &= ~empty
        else:
            active &= ~np.isclose(vals, 10.0, rtol=0.0, atol=1e-12)
        rows = np.flatnonzero(active)
        pts = np.asarray(sdf_grid.idx2pts(rows), dtype=np.float64).reshape(-1, 3)
        active_vals = vals[rows]
        print(
            f"[dcsdd:cloud] samples={len(active_vals)} "
            f"negative={int(np.count_nonzero(active_vals < 0.0))} "
            f"positive={int(np.count_nonzero(active_vals > 0.0))}"
        )
        return pts, active_vals

    @staticmethod
    def _prepare_dcsdd_field(S_raw, unevaluated, N1, step, level, config):
        """Create a finite, bounded negative-inside SDF for DC/DC-SDD.

        Oktopus stores +10 at grid points that were never evaluated. Marching
        Cubes can treat that as a generic outside value, but DC-SDD interprets
        ``abs(S)`` as a geometric sphere radius. Therefore +10 must never be
        passed to the DC-SDD optimizer.

        We keep every evaluated network prediction, fill only unevaluated
        background points with a positive distance-to-inside estimate, and
        truncate the field to a narrow metric band measured in world units.
        The zero set and the signs of all evaluated samples are unchanged.
        """
        from scipy.ndimage import distance_transform_edt

        field = np.asarray(S_raw, dtype=np.float64).reshape(-1) - float(level)
        empty = np.asarray(unevaluated, dtype=bool).reshape(-1)
        if field.size != N1 ** 3 or empty.size != field.size:
            raise ValueError(
                f"Unexpected DCSDD grid sizes: S={field.size}, "
                f"empty={empty.size}, expected={N1 ** 3}"
            )

        finite = np.isfinite(field)
        evaluated = finite & ~empty
        if not np.any(evaluated):
            raise RuntimeError(
                "No evaluated SDF samples. Ensure update_grid(..., mark=True) "
                "is used during Oktopus inference."
            )

        neg = int(np.count_nonzero(field[evaluated] < 0.0))
        pos = int(np.count_nonzero(field[evaluated] > 0.0))
        if neg == 0:
            raise RuntimeError(
                "No negative evaluated samples after sign conversion. The "
                "current pipeline requires negative=inside. For the old "
                "positive-inside checkpoint, set "
                "AgentSDF.invert_trained_sdf_sign=True."
            )

        G = field.reshape((N1, N1, N1), order="F")
        E = (~evaluated).reshape((N1, N1, N1), order="F")
        inside = (~E) & (G < 0.0)

        # distance_transform_edt(~inside) is zero on inside samples and gives
        # the world-space distance to the nearest inside sample elsewhere.
        dist_to_inside = distance_transform_edt(
            ~inside, sampling=(float(step),) * 3
        )

        G_safe = G.copy()
        G_safe[E] = dist_to_inside[E]

        cell_diag = float(np.sqrt(3.0) * step)
        configured_band = (
            config.get("dcsdd_sdf_band")
            if hasattr(config, "get") else None
        )
        band = 4.0 * cell_diag if configured_band is None else float(configured_band)
        if not np.isfinite(band) or band <= cell_diag:
            raise ValueError(
                "dcsdd_sdf_band must be finite and greater than one cell "
                f"diagonal ({cell_diag:.9g}); got {band}"
            )

        G_safe = np.clip(G_safe, -band, band)
        safe = np.ascontiguousarray(G_safe.reshape(-1, order="F"))

        print(
            f"[dcsdd:safe-field] evaluated={int(evaluated.sum())} "
            f"negative={neg} positive={pos} "
            f"filled_background={int(E.sum())} "
            f"cell_diag={cell_diag:.9g} band={band:.9g} "
            f"range=[{safe.min():.9g},{safe.max():.9g}]"
        )
        return safe

    def _ours_options(self, contouring, config):
        cc = contouring._contouring_cpp_module

        def g(key, default):
            value = config.get(f"dcsdd_{key}") if hasattr(config, "get") else None
            return default if value is None else value

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
            "verbose": bool(g("verbose", False)),
            "batch_size": int(g("batch_size", 200000)),
        }

    @staticmethod
    def _print_grid_diagnostics(S_raw, U, unevaluated, N1, level):
        finite = np.isfinite(S_raw)
        evaluated = finite & ~unevaluated

        def pct(values):
            values = np.asarray(values, dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size == 0:
                return "no finite values"
            qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
            ps = np.percentile(values, qs)
            return " ".join(f"p{q}={v:.6g}" for q, v in zip(qs, ps))

        print(
            f"[dcsdd:grid] samples={len(S_raw)} shape={N1}x{N1}x{N1} "
            f"level={level:.9g}"
        )
        print(
            f"[dcsdd:grid] finite={int(finite.sum())} "
            f"evaluated={int(evaluated.sum())} "
            f"unevaluated={int(unevaluated.sum())} "
            f"exact_plus10={int(np.count_nonzero(S_raw == 10.0))}"
        )
        print(f"[dcsdd:grid] S all: {pct(S_raw)}")
        print(f"[dcsdd:grid] S evaluated: {pct(S_raw[evaluated])}")

        G = S_raw.reshape((N1, N1, N1), order="F")
        E = unevaluated.reshape((N1, N1, N1), order="F")
        below = G < level
        above = G > level

        sx = (below[:-1] & above[1:]) | (above[:-1] & below[1:])
        sy = (below[:, :-1] & above[:, 1:]) | (above[:, :-1] & below[:, 1:])
        sz = (below[:, :, :-1] & above[:, :, 1:]) | (above[:, :, :-1] & below[:, :, 1:])
        tx = sx & (E[:-1] | E[1:])
        ty = sy & (E[:, :-1] | E[:, 1:])
        tz = sz & (E[:, :, :-1] | E[:, :, 1:])

        cell_below = (
            below[:-1, :-1, :-1] | below[1:, :-1, :-1]
            | below[:-1, 1:, :-1] | below[1:, 1:, :-1]
            | below[:-1, :-1, 1:] | below[1:, :-1, 1:]
            | below[:-1, 1:, 1:] | below[1:, 1:, 1:]
        )
        cell_above = (
            above[:-1, :-1, :-1] | above[1:, :-1, :-1]
            | above[:-1, 1:, :-1] | above[1:, 1:, :-1]
            | above[:-1, :-1, 1:] | above[1:, :-1, 1:]
            | above[:-1, 1:, 1:] | above[1:, 1:, 1:]
        )
        interesting = cell_below & cell_above
        touches_empty = (
            E[:-1, :-1, :-1] | E[1:, :-1, :-1]
            | E[:-1, 1:, :-1] | E[1:, 1:, :-1]
            | E[:-1, :-1, 1:] | E[1:, :-1, 1:]
            | E[:-1, 1:, 1:] | E[1:, 1:, 1:]
        )

        print(
            f"[dcsdd:grid] sign_changing_edges={int(sx.sum()+sy.sum()+sz.sum())} "
            f"x={int(sx.sum())} y={int(sy.sum())} z={int(sz.sum())}"
        )
        print(
            "[dcsdd:grid] sign_changing_edges_touching_unevaluated="
            f"{int(tx.sum()+ty.sum()+tz.sum())} "
            f"x={int(tx.sum())} y={int(ty.sum())} z={int(tz.sum())}"
        )
        print(
            f"[dcsdd:grid] interesting_cells={int(interesting.sum())} "
            "interesting_cells_touching_unevaluated="
            f"{int((interesting & touches_empty).sum())}"
        )
        print(
            f"[dcsdd:grid] U bbox min={U.min(axis=0).tolist()} "
            f"max={U.max(axis=0).tolist()}"
        )
        if np.any(evaluated):
            print(
                "[dcsdd:grid] evaluated-sample bbox "
                f"min={U[evaluated].min(axis=0).tolist()} "
                f"max={U[evaluated].max(axis=0).tolist()}"
            )

    def run_all_reconstructions(self, sdf_grid, config, name, mc_method="extract_mesh"):
        contouring, utility, gpy = self._load_dcsdd(config)
        output_folder = config.get("output_folder", ".")
        checkpoint = config.get("checkpoint", "eval")
        level = float(config.get("level", sdf_grid.grid_config.get("level", 0.0)))

        out_dir = op.join(output_folder, name)
        os.makedirs(out_dir, exist_ok=True)
        reso = int(sdf_grid.reso)

        requested = config.get("dcsdd_methods")
        if isinstance(requested, str):
            requested = [m.strip() for m in requested.split(",") if m.strip()]
        methods = list(requested) if requested else [
            "mc", "mc_ref", "dc", "ours", "rfta", "mnm1", "mnm2"
        ]

        S_raw, U, unevaluated, N1 = self._grid_to_S_U(sdf_grid)
        self._print_grid_diagnostics(S_raw, U, unevaluated, N1, level)

        # MC can use Oktopus's positive +10 outside sentinel because it only
        # needs the zero crossing. DC/DC-SDD cannot: the reference code treats
        # abs(S) as a physical sphere radius. Give those methods a separately
        # filled and bounded field while preserving the same evaluated zero set.
        S_mc = np.ascontiguousarray(S_raw - level, dtype=np.float64)
        S_dc = None
        if any(method in methods for method in ("dc", "ours")):
            step = float(sdf_grid.grid_config["step"])
            S_dc = self._prepare_dcsdd_field(
                S_raw=S_raw,
                unevaluated=unevaluated,
                N1=N1,
                step=step,
                level=level,
                config=config,
            )
        contour_iso = 0.0
        cc = contouring._contouring_cpp_module

        results = {}
        timings = {}

        def save(method, V, F):
            if V is None or F is None or len(V) == 0 or len(F) == 0:
                print(f"[dcsdd:{method}] empty result, not saved")
                return
            mesh = trimesh.Trimesh(
                vertices=np.asarray(V, dtype=np.float64),
                faces=np.asarray(F, dtype=np.int64),
                process=False,
            )
            path = op.join(out_dir, f"{name}_{method}_{checkpoint}_mesh{reso}.ply")
            mesh.export(path)
            results[method] = mesh
            orientation = "undetermined_nonwatertight"
            if mesh.is_watertight:
                orientation = (
                    "outward_positive_signed_volume"
                    if mesh.volume > 0 else "inward_negative_signed_volume"
                )
            print(
                f"[dcsdd:{method}] saved {path} V={len(mesh.vertices)} "
                f"F={len(mesh.faces)} time={timings.get(method, float('nan')):.4f}s"
            )
            print(
                f"[dcsdd:{method}] mesh bbox min={mesh.bounds[0].tolist()} "
                f"max={mesh.bounds[1].tolist()} extent={mesh.extents.tolist()}"
            )
            print(
                f"[dcsdd:{method}] watertight={mesh.is_watertight} "
                f"winding_consistent={mesh.is_winding_consistent} "
                f"signed_volume={float(mesh.volume):.9g} orientation={orientation}"
            )

        def contour(method_name, opts, samples):
            t0 = time.time()
            V, F = contouring.py_contouring(
                samples, U, N1, N1, N1, contour_iso, opts, None, None
            )
            timings[method_name] = time.time() - t0
            save(method_name, V, F)

        # Exact Oktopus baseline used by agent_3dvec_sdf.py.
        if "mc" in methods:
            try:
                t0 = time.time()
                native_mesh = getattr(sdf_grid, mc_method)()
                timings["mc"] = time.time() - t0
                save("mc", native_mesh.vertices, native_mesh.faces)
            except Exception as exc:
                print(f"[dcsdd:mc] FAILED native Oktopus MC: {exc}")

        # Reference repository MC, retained as a diagnostic only.
        if "mc_ref" in methods:
            try:
                contour("mc_ref", {"method": cc.ContouringMethod.MarchingCubes}, S_mc)
            except Exception as exc:
                print(f"[dcsdd:mc_ref] FAILED: {exc}")

        if "dc" in methods:
            try:
                contour("dc", {"method": cc.ContouringMethod.DualContouring}, S_dc)
            except Exception as exc:
                print(f"[dcsdd:dc] FAILED: {exc}")

        if "ours" in methods:
            try:
                contour("ours", self._ours_options(contouring, config), S_dc)
            except Exception as exc:
                print(f"[dcsdd:ours] FAILED: {exc}")

        need_cloud = any(m in methods for m in ("rfta", "mnm1", "mnm2"))
        if need_cloud:
            Ua, Sa_raw = self._active_cloud(sdf_grid)
            Sa = Sa_raw - level
            has_both = bool(np.any(Sa < 0.0) and np.any(Sa > 0.0))

        if "rfta" in methods:
            if gpy is None:
                print("[dcsdd:rfta] skipped (gpytoolbox unavailable)")
            elif not has_both:
                print("[dcsdd:rfta] skipped (need samples on both sides of level)")
            else:
                try:
                    t0 = time.time()
                    Vr, Fr = gpy.reach_for_the_arcs(Ua, Sa, verbose=False)
                    timings["rfta"] = time.time() - t0
                    save("rfta", Vr, Fr)
                except Exception as exc:
                    print(f"[dcsdd:rfta] FAILED: {exc}")

        if "mnm1" in methods:
            if utility is None:
                print("[dcsdd:mnm1] skipped (utility unavailable)")
            else:
                try:
                    t0 = time.time()
                    result = utility.kohlbrenner_reconstruction(Ua, Sa, method="cones")
                    if not isinstance(result, (tuple, list)) or len(result) != 2:
                        raise RuntimeError(
                            "maximal-empty-spheres returned no mesh; build its executable"
                        )
                    Vk, Fk = result
                    timings["mnm1"] = time.time() - t0
                    save("mnm1", Vk, Fk)
                except Exception as exc:
                    print(f"[dcsdd:mnm1] FAILED: {exc}")

        if "mnm2" in methods:
            gt = config.get("dcsdd_gt_mesh")
            if utility is None:
                print("[dcsdd:mnm2] skipped (utility unavailable)")
            elif not gt or not op.isfile(gt):
                print("[dcsdd:mnm2] skipped (set --dcsdd-gt-mesh)")
            else:
                try:
                    V_gt, F_gt = gpy.read_mesh(gt)
                    t0 = time.time()
                    Vk, Fk = utility.kohlbrenner_reconstruction(
                        Ua, Sa, V_gt=V_gt, F_gt=F_gt, method="RC"
                    )
                    timings["mnm2"] = time.time() - t0
                    save("mnm2", Vk, Fk)
                except Exception as exc:
                    print(f"[dcsdd:mnm2] FAILED: {exc}")

        print(
            f"[dcsdd] timings for {name} @ {reso}^3: "
            + ", ".join(f"{m}={timings[m]:.4f}s" for m in timings)
        )
        return results

    def extract_surface_mesh(
        self, sdf_grid, config=None, mc_method="extract_mesh", context=""
    ):
        config = {} if config is None else config
        spec = config.get(
            "surface_extraction", config.get("mesh_extractor", "run_ours")
        )
        method = spec if isinstance(spec, str) else "run_ours"
        method = method.lower()

        name = str(context).split(":")[-1] if context else ""
        if not name:
            name = str(config.get("shape", "shape"))

        suite_aliases = {"run_ours", "dcsdd", "dc_sdd", "all", "dcsdd_all", "bench"}
        single_aliases = {"mc", "mc_ref", "dc", "ours", "rfta", "mnm1", "mnm2"}

        if method in suite_aliases or method in single_aliases:
            cfg = dict(config)
            if method in single_aliases:
                cfg["dcsdd_methods"] = [method]
            results = self.run_all_reconstructions(
                sdf_grid, cfg, name, mc_method=mc_method
            )
            for key in ("ours", "dc", "mc", "mc_ref", "rfta", "mnm1", "mnm2"):
                if key in results:
                    return results[key]
            return trimesh.Trimesh(
                vertices=np.zeros((0, 3)),
                faces=np.zeros((0, 3), dtype=np.int64),
                process=False,
            )

        return super().extract_surface_mesh(
            sdf_grid, config=config, mc_method=mc_method, context=context
        )


Agent = AgentSDFDC
