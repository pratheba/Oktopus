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

    def _make_nsdudf_oracle_from_udf(
        self,
        raw_world_udf_fn,
        *,
        domain_center,
        domain_half_extent,
        n_grid_samples,
        far_world,
        oracle_chunk_size,
    ):
        """Build the NSDUDF oracle from an arbitrary world-space UDF callable.

        This is the generic NSDUDF path. The supplied callable returns UDF
        values at arbitrary world points. Gradients are estimated with central
        finite differences.

        Parameters
        ----------
        raw_world_udf_fn
            Callable accepting world-space points of shape (N, 3). It may
            return either values or ``(values, validity_mask)``.
        domain_center
            Center of the NSDUDF extraction cube in world coordinates.
        domain_half_extent
            Half extent mapping NSDUDF coordinates ``u in [-1, 1]^3`` to

                world = center + half_extent * u.

        n_grid_samples
            Number of NSDUDF samples per axis. For example, 65 samples means
            64 cells, and 129 samples means 128 cells.
        far_world
            UDF value assigned outside the localization support, in world
            units.
        oracle_chunk_size
            Maximum number of NSDUDF grid points processed at once.

        Returns
        -------
        udf_and_grad_f
            Callable expected by ``nsdudf.core.meshing.compute_pseudo_sdf``.
        stats
            Finite-difference diagnostics dictionary.
        metadata
            Dictionary containing spacing and normalization information.
        """
        center = np.asarray(domain_center, dtype=np.float64).reshape(3)
        half = float(domain_half_extent)
        n = int(n_grid_samples)

        if not np.isfinite(half) or half <= 0.0:
            raise ValueError(
                f"Invalid NSDUDF domain half extent: {half}"
            )
        if n < 2:
            raise ValueError(
                f"NSDUDF needs at least 2 grid samples, got {n}"
            )

        cells = n - 1

        # The existing generic oracle chooses its central-difference spacing
        # from an octree depth. Using ceil(log2(cells)) ensures that the
        # finite-difference step is no larger than one quarter of the NSDUDF
        # voxel width. It is exact for power-of-two cell counts such as
        # 64 and 128.
        fd_depth = max(
            1,
            int(math.ceil(math.log2(max(cells, 2)))),
        )

        _udf_func, udf_grad_func, eps_u, fd_stats = (
            self._make_dualmesh_oracle(
                raw_world_udf_fn,
                domain_center=center,
                domain_half_extent=half,
                max_depth=fd_depth,
                far_world=float(far_world),
            )
        )

        chunk_size = max(1, int(oracle_chunk_size))

        def udf_and_grad_f(query_points):
            """Return normalized UDF values and unit world gradients.

            NSDUDF supplies points in its normalized ``[-1, 1]^3`` cube.
            ``udf_grad_func`` already performs the cube-to-world conversion and
            returns distances divided by ``half``.
            """
            if torch.is_tensor(query_points):
                points_np = query_points.detach().cpu().numpy()
            else:
                points_np = np.asarray(query_points)

            points_np = np.asarray(
                points_np,
                dtype=np.float64,
            ).reshape(-1, 3)

            n_points = int(points_np.shape[0])
            if n_points == 0:
                return (
                    torch.zeros((0,), dtype=torch.float32),
                    torch.zeros((0, 3), dtype=torch.float32),
                )

            distances = np.empty(n_points, dtype=np.float32)
            gradients = np.empty((n_points, 3), dtype=np.float32)

            for start in range(0, n_points, chunk_size):
                end = min(start + chunk_size, n_points)

                distance_chunk, gradient_chunk = udf_grad_func(
                    points_np[start:end]
                )

                distances[start:end] = np.asarray(
                    distance_chunk,
                    dtype=np.float32,
                ).reshape(-1)

                gradients[start:end] = np.asarray(
                    gradient_chunk,
                    dtype=np.float32,
                ).reshape(-1, 3)

            return (
                torch.from_numpy(distances),
                torch.from_numpy(gradients),
            )

        metadata = {
            "gradient_mode": "finite_difference",
            "n_grid_samples": n,
            "cells_per_axis": cells,
            "fd_depth": fd_depth,
            "fd_eps_u": float(eps_u),
            "fd_eps_world": float(eps_u) * half,
            "far_cube": float(far_world) / half,
            "oracle_chunk_size": chunk_size,
        }

        return udf_and_grad_f, fd_stats, metadata

    def _make_nsdudf_oracle_from_model_direct(
        self,
        *,
        curve,
        curve_key,
        domain_center,
        domain_half_extent,
        n_grid_samples,
        far_world,
        oracle_chunk_size,
        localize_norm=1.0,
    ):
        """Build an NSDUDF oracle by differentiating the Oktopus model directly.

        This method is intentionally not implemented yet.

        Oktopus does not evaluate its neural UDF directly from world xyz.
        World points are first converted by NumPy curve localization into
        several coupled inputs:

            samples_local
            coords
            rho
            rho_n
            angles
            radius
            frame_mat

        Differentiating only with respect to ``samples_local`` would omit the
        dependence of the prediction on the other localized quantities and
        would therefore not produce the true world-space UDF gradient.

        A correct implementation needs either:

          1. a differentiable Torch implementation of curve localization, or
          2. an explicit complete Jacobian from world coordinates to every
             model input used by the network.

        Until that is implemented and verified, use
        ``_make_nsdudf_oracle_from_udf``.
        """
        raise NotImplementedError(
            "Direct NSDUDF autograd requires differentiable Oktopus curve "
            "localization. Use gradient mode 'finite_difference' for now."
        )

    def _cleanup_nsdudf_mesh(self, mesh, config):
        """Conservative cleanup for NSDUDF + Marching Cubes.

        Removes numerical/triangulation debris while preserving intentional
        open boundaries and multiple meaningful surface components.

        This deliberately does NOT:
          - fill holes,
          - force watertightness,
          - keep only the largest component,
          - smooth vertices.
        """
        if mesh is None or len(mesh.faces) == 0:
            return mesh

        def cget(key, default):
            value = config.get(key) if hasattr(config, "get") else None
            return default if value is None else value

        min_component_faces = int(
            cget("nsdudf_min_component_faces", 20)
        )
        min_component_faces = max(min_component_faces, 0)

        # Relative tolerance based on the current mesh scale.
        bounds = np.asarray(mesh.bounds, dtype=np.float64)
        diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))
        merge_digits = int(cget("nsdudf_merge_digits", 10))
        area_epsilon = float(
            cget(
                "nsdudf_degenerate_area_epsilon",
                max(diagonal * diagonal * 1.0e-14, 1.0e-16),
            )
        )

        before = self._mesh_quality_stats(mesh.vertices, mesh.faces)

        cleaned = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices, dtype=np.float64).copy(),
            faces=np.asarray(mesh.faces, dtype=np.int64).copy(),
            process=False,
        )

        # 1. Merge coincident / nearly coincident vertices.
        try:
            cleaned.merge_vertices(digits_vertex=merge_digits)
        except TypeError:
            # Compatibility with older trimesh versions.
            cleaned.merge_vertices()
        except Exception as exc:
            print("[nsdudf cleanup] merge_vertices failed:", exc)

        # 2. Remove explicit repeated-index and near-zero-area triangles.
        if len(cleaned.faces):
            faces = np.asarray(cleaned.faces, dtype=np.int64)
            vertices = np.asarray(cleaned.vertices, dtype=np.float64)

            repeated_index = (
                (faces[:, 0] == faces[:, 1])
                | (faces[:, 1] == faces[:, 2])
                | (faces[:, 2] == faces[:, 0])
            )

            edge_1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
            edge_2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
            double_area = np.linalg.norm(
                np.cross(edge_1, edge_2),
                axis=1,
            )

            keep = (
                (~repeated_index)
                & np.isfinite(double_area)
                & (0.5 * double_area > area_epsilon)
            )
            cleaned.update_faces(keep)

        # 3. Remove duplicate triangles.
        try:
            cleaned.update_faces(cleaned.unique_faces())
        except Exception as exc:
            print("[nsdudf cleanup] unique_faces failed:", exc)

        try:
            cleaned.remove_unreferenced_vertices()
        except Exception:
            pass

        # 4. Remove only tiny vertex-connected fragments.
        #
        # Do not use trimesh.split() here. NSDUDF output can contain
        # non-manifold edges, and face-adjacency splitting may incorrectly
        # break one surface into many pieces.
        if min_component_faces > 0 and len(cleaned.faces):
            try:
                faces = np.asarray(cleaned.faces, dtype=np.int64)
                n_vertices = len(cleaned.vertices)

                parent = np.arange(n_vertices, dtype=np.int64)
                rank = np.zeros(n_vertices, dtype=np.int8)

                def find(x):
                    x = int(x)
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = int(parent[x])
                    return x

                def union(a, b):
                    ra = find(a)
                    rb = find(b)
                    if ra == rb:
                        return
                    if rank[ra] < rank[rb]:
                        parent[ra] = rb
                    elif rank[ra] > rank[rb]:
                        parent[rb] = ra
                    else:
                        parent[rb] = ra
                        rank[ra] += 1

                for tri in faces:
                    union(tri[0], tri[1])
                    union(tri[1], tri[2])
                    union(tri[2], tri[0])

                face_roots = np.asarray(
                    [find(tri[0]) for tri in faces],
                    dtype=np.int64,
                )

                roots, face_counts = np.unique(
                    face_roots,
                    return_counts=True,
                )
                keep_roots = roots[
                    face_counts >= min_component_faces
                ]

                keep_faces = np.isin(face_roots, keep_roots)

                removed_components = int(
                    np.sum(face_counts < min_component_faces)
                )
                removed_faces = int(np.sum(~keep_faces))

                cleaned.update_faces(keep_faces)
                cleaned.remove_unreferenced_vertices()

                print(
                    "[nsdudf cleanup components]",
                    f"min_faces={min_component_faces}",
                    f"removed_components={removed_components}",
                    f"removed_faces={removed_faces}",
                )

            except Exception as exc:
                print("[nsdudf cleanup] component filtering failed:", exc)

        after = self._mesh_quality_stats(
            cleaned.vertices,
            cleaned.faces,
        )

        print(
            "[nsdudf cleanup]",
            "before=", before,
            "after=", after,
        )

        return cleaned

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
        far_world = float(cget("udf_far_value", half))
        batch_size = int(cget("udf_batch_size", 150000))

        oracle_chunk_size = int(
            cget(
                "nsdudf_oracle_chunk_size",
                min(batch_size, 32768),
            )
        )

        gradient_mode = str(
            cget("nsdudf_gradient_mode", "finite_difference")
        ).strip().lower()

        if gradient_mode not in {
            "finite_difference",
            "model_direct",
        }:
            raise ValueError(
                "Unknown nsdudf_gradient_mode "
                f"{gradient_mode!r}. Expected 'finite_difference' or "
                "'model_direct'."
            )

        if gradient_mode == "model_direct":
            model_context = getattr(
                raw_world_udf_fn,
                "_oktopus_model_context",
                None,
            )
            if model_context is None:
                raise ValueError(
                    "nsdudf_gradient_mode='model_direct' requires an "
                    "Oktopus-native model query carrying "
                    "_oktopus_model_context. This mode cannot be used with "
                    "an arbitrary external UDF callable."
                )

            udf_and_grad_f, fd_stats, oracle_meta = (
                self._make_nsdudf_oracle_from_model_direct(
                    raw_world_udf_fn=raw_world_udf_fn,
                    model_context=model_context,
                    domain_center=center,
                    domain_half_extent=half,
                    n_grid_samples=n,
                    far_world=far_world,
                    oracle_chunk_size=oracle_chunk_size,
                )
            )
        else:
            udf_and_grad_f, fd_stats, oracle_meta = (
                self._make_nsdudf_oracle_from_udf(
                    raw_world_udf_fn,
                    domain_center=center,
                    domain_half_extent=half,
                    n_grid_samples=n,
                    far_world=far_world,
                    oracle_chunk_size=oracle_chunk_size,
                )
            )

        print(
            "[nsdudf]",
            f"gradient_mode={oracle_meta['gradient_mode']}",
            f"grid_samples={oracle_meta['n_grid_samples']}",
            f"cells={oracle_meta['cells_per_axis']}",
            f"center={center}",
            f"half={half:.6g}",
            f"fd_depth={oracle_meta['fd_depth']}",
            f"fd_eps_u={oracle_meta['fd_eps_u']:.6g}",
            f"fd_eps_world={oracle_meta['fd_eps_world']:.6g}",
            f"far_cube={oracle_meta['far_cube']:.6g}",
            f"oracle_chunk={oracle_meta['oracle_chunk_size']}",
            f"nsdudf_batch={batch_size}",
        )
      
        pseudo_sdf = nsd_meshing.compute_pseudo_sdf(
            model,
            udf_and_grad_f,
            n_grid_samples=n,
            batch_size=batch_size,
            normalize_udf=bool(cget("nsdudf_normalize_udf", True)),
            use_grads=True,
            out7=False,
        )
        total_axes = max(1, int(fd_stats["eval_points"]) * 3)

        print(
            "[nsdudf fd]",
            f"eval_points={fd_stats['eval_points']}",
            f"near_surface={fd_stats.get('near_surface_points',0)}",
            f"near_invalid={fd_stats.get('near_surface_invalid_gradients',0)}",
            f"central_axes={fd_stats['central_axes']}",
            f"invalid_axes={fd_stats['invalid_axes']}",
            f"invalid_axis_pct="
            f"{100.0 * fd_stats['invalid_axes'] / total_axes:.3f}",
            f"invalid_gradients={fd_stats['invalid_gradients']}",
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
            mesh = self._cleanup_nsdudf_mesh(mesh, config)

        print(
            "[nsdudf final]",
            f"V={len(mesh.vertices)}",
            f"F={len(mesh.faces)}",
            self._mesh_quality_stats(mesh.vertices, mesh.faces),
        )
        return mesh
