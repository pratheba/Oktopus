"""UDF reconstruction via NSDUDF (neural pseudo-SDF meshing).

`AgentUDFNsdudf` reuses the whole existing UDF inference path from
``AgentUDF`` / ``AgentBase`` (model loading, curve localization, the
direct-query UDF oracle) and only replaces the mesh-extraction backend:
instead of DualMesh-UDF it uses the NSDUDF method from the uploaded
``nsdudf`` repo, which turns a UDF + gradient field into a *signed
pseudo-SDF* on a grid and meshes it with a custom marching cubes (and,
optionally, its relaxed DualMesh-UDF variant).

Pipeline:
    UDF oracle (this agent)  ->  udf_and_grad_f(query_points)
        ->  core.meshing.compute_pseudo_sdf(model.pt, ...)   # 8-value/cell pseudo-SDF
        ->  core.meshing.mesh_marching_cubes(...)            # trimesh in [-1,1]
        ->  map back to world:  world = center + half * v

The UDF oracle is built with the exact same helper the DualMesh-UDF path
uses (``AgentUDF._make_dualmesh_oracle``), so distances are returned in
``[-1,1]`` cube units (``raw_world_udf / half_extent``) and gradients are
unit vectors pointing away from the surface -- precisely what NSDUDF's
``compute_pseudo_sdf`` expects, since it samples on a fixed ``[-1,1]^3``
grid.

Nothing in the existing code base is modified; this is a drop-in subclass
selected by the ``inference_3dvec_nsdudf.py`` runner. The trained NSDUDF
classifier (``model.pt``) and the compiled ``custom_mc`` Cython extension
must be present in the nsdudf checkout (see RECON_BENCH_README.md).
"""

import math
import os
import os.path as op
import sys

import numpy as np
import torch
import trimesh

from agent_3dvec_udf import AgentUDF


# Default location of the extracted nsdudf repo on this machine (override with
# the NSDUDF_REPO env var or the `nsdudf_repo` config/yaml key).
_DEFAULT_NSDUDF_REPO = op.expanduser("~/Downloads/nsdudf-main")


class AgentUDFNsdudf(AgentUDF):
    """UDF agent that meshes via NSDUDF pseudo-SDF instead of DualMesh-UDF."""

    # ------------------------------------------------------------------
    # nsdudf import + model loading (cached)
    # ------------------------------------------------------------------
    def _load_nsdudf(self, config):
        """Import ``core.meshing`` / ``core.utils`` from the nsdudf repo and
        load the trained classifier ``model.pt``. Cached on the instance.
        Returns ``(meshing_module, utils_module, model)``.
        """
        if getattr(self, "_nsdudf_cache", None) is not None:
            return self._nsdudf_cache

        repo = (
            (config.get("nsdudf_repo") if hasattr(config, "get") else None)
            or os.environ.get("NSDUDF_REPO")
            or _DEFAULT_NSDUDF_REPO
        )
        repo = op.abspath(op.expanduser(repo))
        if not op.isdir(repo):
            raise FileNotFoundError(
                f"nsdudf repo not found at {repo!r}. Extract the uploaded "
                "nsdudf zip and set env NSDUDF_REPO or the 'nsdudf_repo' config "
                "key to that folder."
            )

        # core.meshing does `sys.path.append('custom_mc')` relative to CWD and
        # then `from _marching_cubes_lewiner import ...`; make that resolve no
        # matter the CWD by putting the absolute paths first.
        for p in (repo, op.join(repo, "custom_mc")):
            if p not in sys.path:
                sys.path.insert(0, p)

        try:
            import core.utils as nsd_utils  # noqa: E402
            import core.meshing as nsd_meshing  # noqa: E402
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "Could not import nsdudf 'core.meshing'. Its custom marching "
                "cubes must be built first: from the repo's custom_mc/ run "
                "`python setup.py build_ext --inplace`. "
                f"(repo={repo!r})\nOriginal error: {exc}"
            ) from exc

        model_path = (
            (config.get("nsdudf_model") if hasattr(config, "get") else None)
            or os.environ.get("NSDUDF_MODEL")
            or op.join(repo, "model.pt")
        )
        if not op.isfile(model_path):
            raise FileNotFoundError(
                f"NSDUDF model checkpoint not found at {model_path!r}. Set "
                "'nsdudf_model' / NSDUDF_MODEL to the trained model.pt."
            )

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        model = nsd_utils.load_model(model_path, device_str)

        self._nsdudf_cache = (nsd_meshing, nsd_utils, model)
        return self._nsdudf_cache

    # ------------------------------------------------------------------
    # Replace the DualMesh-UDF extractor with NSDUDF pseudo-SDF meshing
    # ------------------------------------------------------------------
    def extract_udf_mesh_from_model(
        self,
        raw_world_udf_fn,
        *,
        domain_center,
        domain_half_extent,
        resolution,
        config,
    ):
        """Extract via NSDUDF: build the pseudo-SDF from the model's UDF+grad
        oracle, marching-cube it, and map back into world coordinates."""
        nsd_meshing, _nsd_utils, model = self._load_nsdudf(config)

        def cget(key, default):
            v = config.get(key) if hasattr(config, "get") else None
            return default if v is None else v

        center = np.asarray(domain_center, dtype=np.float64).reshape(3)
        half = float(domain_half_extent)
        if not np.isfinite(half) or half <= 0.0:
            raise ValueError(f"Invalid UDF domain half extent: {half}")

        # Grid resolution (number of samples per axis). NSDUDF works at multiple
        # resolutions; 129 / 257 are also valid for the dual_mesh variant.
        n = int(cget("nsdudf_grid", resolution))
        n = max(int(n), 8)

        # Finite-difference gradient scale ~ quarter of a voxel.
        cells = max(n - 1, 2)
        max_depth = max(1, int(round(math.log2(cells))))
        # Points outside the localization support get UDF = far_world; using the
        # half-extent makes far_cube = 1.0 so they're pruned by NSDUDF's
        # near-surface cell thresholds.
        far_world = float(cget("udf_far_value", half))
        batch_size = int(cget("udf_batch_size", 150000))

        udf_func, udf_grad_func, eps_u, fd_stats = self._make_dualmesh_oracle(
            raw_world_udf_fn,
            domain_center=center,
            domain_half_extent=half,
            max_depth=max_depth,
            far_world=far_world,
        )

        # Adapter to NSDUDF's expected signature: takes grid query points in
        # [-1,1]^3 (torch), returns (udf, grads) as torch tensors.
        def udf_and_grad_f(query_points):
            pts = (
                query_points.detach().cpu().numpy()
                if torch.is_tensor(query_points)
                else np.asarray(query_points)
            ).astype(np.float64).reshape(-1, 3)
            dist, grad = udf_grad_func(pts)  # (N,1) float32, (N,3) float32
            udf_t = torch.from_numpy(np.asarray(dist, dtype=np.float32).reshape(-1))
            grad_t = torch.from_numpy(np.asarray(grad, dtype=np.float32).reshape(-1, 3))
            return udf_t, grad_t

        print(
            "[nsdudf]",
            f"grid={n}",
            f"center={center}",
            f"half={half:.6g}",
            f"fd_base_eps_u={eps_u:.6g}",
            f"far_cube={far_world / half:.4g}",
            f"batch={batch_size}",
        )

        pseudo_sdf = nsd_meshing.compute_pseudo_sdf(
            model,
            udf_and_grad_f,
            n_grid_samples=n,
            batch_size=batch_size,
            normalize_udf=bool(cget("nsdudf_normalize_udf", True)),
            use_grads=bool(cget("nsdudf_use_grads", True)),
            out7=bool(cget("nsdudf_out7", False)),
        )

        mesh = nsd_meshing.mesh_marching_cubes(pseudo_sdf)
        if mesh is None or len(mesh.faces) == 0:
            print("[nsdudf] marching cubes produced an empty mesh")
            return self._empty_mesh()

        # NSDUDF mesh lives in [-1,1] (voxel origin -1); map back to world.
        world_vertices = center[None, :] + np.asarray(
            mesh.vertices, dtype=np.float64
        ) * half
        mesh = trimesh.Trimesh(
            vertices=world_vertices, faces=np.asarray(mesh.faces), process=False
        )
        print(f"[nsdudf] extracted V={len(mesh.vertices)} F={len(mesh.faces)}")

        if bool(cget("udf_cleanup", False)):
            mesh = self._cleanup_mesh(mesh, config)
        return mesh
