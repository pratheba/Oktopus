"""Standalone NSDUDF diagnostic extraction agent for Oktopus.

This file does not modify ``AgentUDF`` or the normal NSDUDF agent. It reuses
only their model-query/oracle plumbing, then runs the independent diagnostics in
``nsdudf_diagnostics.py``.
"""

from __future__ import annotations

import os
import os.path as op

import numpy as np

from agent_3dvec_udf_nsdudf import AgentUDFNsdudf
from nsdudf_diagnostics import DiagnosticOptions, run_nsdudf_diagnostics


class AgentUDFNsdudfDiagnostic(AgentUDFNsdudf):
    """Oktopus adaptation agent that records NSDUDF failure statistics."""

    def extract_udf_mesh_from_model(
        self,
        raw_world_udf_fn,
        *,
        domain_center,
        domain_half_extent,
        resolution,
        config,
    ):
        nsd_meshing, _nsd_utils, model = self._load_nsdudf(config)

        def cget(key, default):
            value = config.get(key) if hasattr(config, "get") else None
            return default if value is None else value

        center = np.asarray(domain_center, dtype=np.float64).reshape(3)
        half = float(domain_half_extent)
        n = max(8, int(cget("nsdudf_grid", resolution)))
        far_world = float(cget("udf_far_value", half))
        oracle_chunk_size = max(
            1,
            int(cget("nsdudf_oracle_chunk_size", 16384)),
        )
        offset_world = float(cget("nsdudf_offset_world", 0.0))
        requested_mesher = str(
            cget("nsdudf_mesher", "marching_cubes")
        ).strip().lower()
        if requested_mesher not in {"mc", "marching_cubes"}:
            raise ValueError(
                "The diagnostic path measures the NSDUDF pseudo-SDF + "
                "Marching-Cubes failure directly. Use "
                "--nsdudf-mesher marching_cubes."
            )

        udf_and_grad_f, _dmudf_oracles, fd_stats, oracle_meta = (
            self._make_nsdudf_oracle_from_udf(
                raw_world_udf_fn,
                domain_center=center,
                domain_half_extent=half,
                n_grid_samples=n,
                far_world=far_world,
                oracle_chunk_size=oracle_chunk_size,
                offset_world=offset_world,
            )
        )

        root = op.abspath(
            op.expanduser(
                str(cget("nsdudf_diag_dir", "nsdudf_diagnostics"))
            )
        )
        call_index = int(getattr(self, "_nsdudf_diag_call_index", 0))
        self._nsdudf_diag_call_index = call_index + 1
        diag_dir = op.join(root, f"item_{call_index:02d}")
        os.makedirs(diag_dir, exist_ok=True)

        options = DiagnosticOptions(
            n_grid_samples=n,
            classifier_batch_size=max(
                1,
                int(cget("nsdudf_diag_classifier_batch_size", 32768)),
            ),
            query_slab=max(1, int(cget("nsdudf_diag_query_slab", 4))),
            cell_slab=max(1, int(cget("nsdudf_diag_cell_slab", 2))),
            normalize_udf=bool(cget("nsdudf_normalize_udf", True)),
            max_avg_factor=float(
                cget("nsdudf_diag_max_avg_factor", 1.2)
            ),
            max_max_factor=float(
                cget("nsdudf_diag_max_max_factor", 2.0)
            ),
            loose_min_factor=float(
                cget("nsdudf_diag_loose_min_factor", 1.0)
            ),
            max_visualization_points=max(
                1,
                int(cget("nsdudf_diag_max_points", 200000)),
            ),
            extract_mesh=bool(cget("nsdudf_diag_extract_mesh", True)),
        )

        print(
            "[nsdudf diag oracle]",
            f"grid_samples={n}",
            f"cells={n-1}",
            f"center={center.tolist()}",
            f"half={half:.9g}",
            f"fd_depth={oracle_meta['fd_depth']}",
            f"fd_eps_u={oracle_meta['fd_eps_u']:.9g}",
            f"fd_eps_world={oracle_meta['fd_eps_world']:.9g}",
            f"far_cube={oracle_meta['far_cube']:.9g}",
            f"offset_cube={oracle_meta['offset_cube']:.9g}",
            f"output={diag_dir}",
            flush=True,
        )

        result = run_nsdudf_diagnostics(
            model,
            udf_and_grad_f,
            options=options,
            output_dir=diag_dir,
            domain_center=center,
            domain_half_extent=half,
            meshing_module=nsd_meshing,
        )

        total_axes = max(1, int(fd_stats.get("eval_points", 0)) * 3)
        print(
            "[nsdudf diag fd]",
            f"eval_points={fd_stats.get('eval_points', 0)}",
            f"near_surface={fd_stats.get('near_surface_points', 0)}",
            f"near_invalid={fd_stats.get('near_surface_invalid_gradients', 0)}",
            f"central_axes={fd_stats.get('central_axes', 0)}",
            f"invalid_axes={fd_stats.get('invalid_axes', 0)}",
            f"invalid_axis_pct={100.0 * fd_stats.get('invalid_axes', 0) / total_axes:.3f}",
            f"invalid_gradients={fd_stats.get('invalid_gradients', 0)}",
            flush=True,
        )

        if result.mesh is None or len(result.mesh.faces) == 0:
            return self._empty_mesh()
        return result.mesh
