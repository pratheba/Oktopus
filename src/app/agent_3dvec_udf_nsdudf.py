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
        #for p in (repo, op.join(repo, "custom_mc")):
        #    if p not in sys.path:
        #        sys.path.insert(0, p)
        # NSDUDF ships a modified DualMesh-UDF containing extract_mesh_mod.
        # It must take precedence over a separately installed stock
        # DualMesh-UDF, which contains only extract_mesh.
        import_paths = (
            op.join(repo, "DualMesh-UDF"),
            op.join(repo, "custom_mc"),
            repo,
        )

        for p in reversed(import_paths):
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
        offset_world=0.0,
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

        udf_func, udf_grad_func, eps_u, fd_stats = (
            self._make_dualmesh_oracle(
                raw_world_udf_fn,
                domain_center=center,
                domain_half_extent=half,
                max_depth=fd_depth,
                far_world=float(far_world),
            )
        )

        chunk_size = max(1, int(oracle_chunk_size))
        offset_world = max(0.0, float(offset_world))
        offset_cube = offset_world / half

        if offset_world >= float(far_world):
            raise ValueError(
                "nsdudf_offset_world must be smaller than udf_far_value. "
                f"Got offset_world={offset_world}, far_world={far_world}."
            )

        if offset_world > 0.5 * float(far_world):
            print(
                "[nsdudf offset warning]",
                f"offset_world={offset_world:.6g}",
                f"far_world={float(far_world):.6g}",
                "The offset is large relative to the artificial far field.",
            )

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
                distance_chunk = np.asarray(
                    distance_chunk,
                    dtype=np.float32,
                ).reshape(-1)

                gradient_chunk = np.asarray(
                    gradient_chunk,
                    dtype=np.float32,
                ).reshape(-1, 3)

                # Optional UDF offset shell.
                #
                # Original surface:
                #     d(x) = 0
                #
                # Offset shell:
                #     |d(x) - offset| = 0
                #
                # The sign multiplier updates the gradient direction on
                # either side of the offset level set.
                #offset_world = float(
                #    config.get("nsdudf_offset_world", 0.0)
                #)
                #offset_cube = offset_world / half

                if offset_cube > 0.0:
                    signed_offset = distance_chunk - offset_cube

                    gradient_sign = np.where(
                        signed_offset >= 0.0,
                        1.0,
                        -1.0,
                    ).astype(np.float32)

                    distance_chunk = np.abs(signed_offset)
                    gradient_chunk = (
                        gradient_chunk
                        * gradient_sign[:, None]
                    )

                distances[start:end] = distance_chunk
                gradients[start:end] = gradient_chunk


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
            "offset_world": offset_world,
            "offset_cube": offset_cube,
        }

        #return udf_and_grad_f, fd_stats, metadata
        dmudf_oracles = {
            "udf_func": udf_func,
            "udf_grad_func": udf_grad_func,
        }

        return udf_and_grad_f, dmudf_oracles, fd_stats, metadata

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
            cget("nsdudf_min_component_faces", 0)
        )
        min_component_faces = max(min_component_faces, 0)

        # Relative tolerance based on the current mesh scale.
        bounds = np.asarray(mesh.bounds, dtype=np.float64)
        diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))
        merge_vertices_enabled = bool(
            cget("nsdudf_merge_vertices", False)
        )
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

        # 1. Optional vertex merge. Disabled by default because NSDUDF's
        # custom MC can emit distinct cell-local vertices at the same position.
        # Merging those first can turn valid triangles into degenerate faces.
        if merge_vertices_enabled:
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
    # Direct positive-level UDF shell extraction
    # ------------------------------------------------------------------
    def _extract_udf_band_shell(
        self,
        raw_world_udf_fn,
        *,
        domain_center,
        domain_half_extent,
        n_grid_samples,
        shell_level_world,
        far_world,
        query_chunk_size,
        config,
        apply_cleanup=True,
    ):
        """Extract the positive UDF isosurface ``UDF(x)=shell_level_world``.

        This bypasses the NSDUDF classifier entirely.  For a zero-thickness
        open surface, a positive UDF level produces a closed offset shell with
        approximate total thickness ``2 * shell_level_world``.
        """
        from skimage import measure

        center = np.asarray(domain_center, dtype=np.float64).reshape(3)
        half = float(domain_half_extent)
        n = max(8, int(n_grid_samples))
        level = float(shell_level_world)
        far = float(far_world)
        chunk = max(1, int(query_chunk_size))

        if not np.isfinite(level) or level <= 0.0:
            raise ValueError(
                "udf_band_shell requires --nsdudf-offset-world > 0. "
                f"Got {level}."
            )
        if level >= far:
            raise ValueError(
                "The shell level must be smaller than udf_far_value. "
                f"Got shell_level={level}, udf_far_value={far}."
            )

        total = int(n ** 3)
        values = np.empty(total, dtype=np.float32)
        valid_total = 0
        denom = float(n - 1)
        n2 = int(n * n)

        for start in range(0, total, chunk):
            end = min(start + chunk, total)
            flat = np.arange(start, end, dtype=np.int64)

            ix = flat // n2
            rem = flat - ix * n2
            iy = rem // n
            iz = rem - iy * n

            cube_points = np.stack((ix, iy, iz), axis=1).astype(np.float64)
            cube_points = -1.0 + (2.0 / denom) * cube_points
            world_points = center[None, :] + half * cube_points

            result = raw_world_udf_fn(world_points)
            if isinstance(result, tuple):
                distance_chunk, valid_chunk = result
                valid_chunk = np.asarray(valid_chunk, dtype=bool).reshape(-1)
            else:
                distance_chunk = result
                valid_chunk = np.ones(end - start, dtype=bool)

            distance_chunk = np.asarray(
                distance_chunk, dtype=np.float64
            ).reshape(-1)
            if distance_chunk.shape[0] != end - start:
                raise ValueError(
                    "UDF band-shell query length mismatch: "
                    f"expected={end-start}, got={distance_chunk.shape[0]}."
                )
            if valid_chunk.shape[0] != end - start:
                raise ValueError(
                    "UDF band-shell validity length mismatch: "
                    f"expected={end-start}, got={valid_chunk.shape[0]}."
                )

            finite = np.isfinite(distance_chunk)
            valid_chunk = valid_chunk & finite
            out = np.full(end - start, far, dtype=np.float32)
            out[valid_chunk] = np.maximum(
                distance_chunk[valid_chunk], 0.0
            ).astype(np.float32)
            values[start:end] = out
            valid_total += int(valid_chunk.sum())

        grid = values.reshape((n, n, n), order="C")
        grid_min = float(np.min(grid))
        grid_max = float(np.max(grid))
        if not (grid_min <= level <= grid_max):
            raise RuntimeError(
                "The requested UDF shell level is outside the sampled field "
                f"range: level={level}, range=[{grid_min}, {grid_max}]."
            )

        voxel_world = 2.0 * half / float(n - 1)
        print(
            "[udf band shell]",
            f"grid_samples={n}",
            f"cells={n-1}",
            f"level_world={level:.6g}",
            f"approx_total_thickness={2.0 * level:.6g}",
            f"voxel_world={voxel_world:.6g}",
            f"valid={valid_total}/{total}",
            f"range=[{grid_min:.6g},{grid_max:.6g}]",
        )

        vertices_local, faces, _normals, _mc_values = measure.marching_cubes(
            grid,
            level=level,
            spacing=(voxel_world, voxel_world, voxel_world),
            allow_degenerate=False,
        )

        lower_world = center - half
        world_vertices = lower_world[None, :] + np.asarray(
            vertices_local, dtype=np.float64
        )
        mesh = trimesh.Trimesh(
            vertices=world_vertices,
            faces=np.asarray(faces, dtype=np.int64),
            process=False,
        )

        print(
            "[udf band shell raw]",
            self._mesh_quality_stats(mesh.vertices, mesh.faces),
        )

        def cget(key, default):
            value = config.get(key) if hasattr(config, "get") else None
            return default if value is None else value

        if apply_cleanup and bool(cget("udf_cleanup", False)):
            mesh = self._cleanup_nsdudf_mesh(mesh, config)

        print(
            "[udf band shell final]",
            f"V={len(mesh.vertices)}",
            f"F={len(mesh.faces)}",
            self._mesh_quality_stats(mesh.vertices, mesh.faces),
        )
        return mesh

    def _open_udf_band_shell_from_nsdudf_reference(
        self,
        shell_mesh,
        reference_mesh,
        *,
        shell_level_world,
        voxel_world,
        config,
    ):
        """Remove likely closure-wall triangles from a positive UDF shell.

        The shell and NSDUDF reference are generated from the same adapted UDF
        domain.  Valid offset faces are approximately parallel to the open
        center-surface reference.  The rounded walls that close true openings
        are approximately perpendicular to that reference.

        This is deliberately conservative: only sufficiently large connected
        low-alignment seed patches are removed.  No neighborhood growth is
        performed by default, because growth can spill onto the garment body.
        """
        import json
        from scipy.spatial import cKDTree

        def cget(key, default):
            value = config.get(key) if hasattr(config, "get") else None
            return default if value is None else value

        if shell_mesh is None or len(shell_mesh.faces) == 0:
            return shell_mesh
        if reference_mesh is None or len(reference_mesh.faces) == 0:
            raise RuntimeError(
                "udf_band_shell_open needs a non-empty NSDUDF reference mesh."
            )

        shell_centers = np.asarray(
            shell_mesh.triangles_center, dtype=np.float64
        )
        shell_normals = np.asarray(
            shell_mesh.face_normals, dtype=np.float64
        )
        ref_centers = np.asarray(
            reference_mesh.triangles_center, dtype=np.float64
        )
        ref_normals = np.asarray(
            reference_mesh.face_normals, dtype=np.float64
        )

        ref_ok = (
            np.isfinite(ref_centers).all(axis=1)
            & np.isfinite(ref_normals).all(axis=1)
            & (np.linalg.norm(ref_normals, axis=1) > 1.0e-12)
        )
        if not np.any(ref_ok):
            raise RuntimeError(
                "NSDUDF reference has no finite non-zero face normals."
            )

        ref_centers = ref_centers[ref_ok]
        ref_normals = ref_normals[ref_ok]
        tree = cKDTree(ref_centers)
        nearest_distance, nearest_index = tree.query(shell_centers, k=1)
        nearest_normals = ref_normals[nearest_index]
        normal_alignment = np.abs(
            np.einsum("ij,ij->i", shell_normals, nearest_normals)
        )

        max_normal_dot = float(
            cget("udf_shell_wall_max_normal_dot", 0.30)
        )
        automatic_distance = max(
            2.5 * float(shell_level_world),
            2.5 * float(voxel_world),
        )
        max_distance_world = float(
            cget(
                "udf_shell_wall_max_distance_world",
                automatic_distance,
            )
        )
        min_faces = max(1, int(cget("udf_shell_wall_min_faces", 50)))
        min_span_world = max(
            0.0, float(cget("udf_shell_wall_min_span_world", 0.08))
        )
        max_components = max(
            0, int(cget("udf_shell_wall_max_components", 8))
        )

        finite = (
            np.isfinite(nearest_distance)
            & np.isfinite(normal_alignment)
        )
        seed_mask = (
            finite
            & (nearest_distance <= max_distance_world)
            & (normal_alignment <= max_normal_dot)
        )

        candidate_ids = np.flatnonzero(seed_mask)
        accepted = []
        if candidate_ids.size:
            adjacency = np.asarray(
                shell_mesh.face_adjacency, dtype=np.int64
            )
            if adjacency.size:
                local_adjacency = adjacency[
                    seed_mask[adjacency[:, 0]]
                    & seed_mask[adjacency[:, 1]]
                ]
            else:
                local_adjacency = np.empty((0, 2), dtype=np.int64)

            components = trimesh.graph.connected_components(
                local_adjacency,
                nodes=candidate_ids,
                min_len=1,
            )
            for component in components:
                face_ids = np.asarray(list(component), dtype=np.int64)
                if face_ids.size < min_faces:
                    continue
                points = shell_centers[face_ids]
                span = float(
                    np.linalg.norm(points.max(axis=0) - points.min(axis=0))
                )
                if span < min_span_world:
                    continue
                accepted.append(
                    {
                        "faces": face_ids,
                        "face_count": int(face_ids.size),
                        "span_world": span,
                        "median_normal_alignment": float(
                            np.median(normal_alignment[face_ids])
                        ),
                        "median_reference_distance_world": float(
                            np.median(nearest_distance[face_ids])
                        ),
                    }
                )

        accepted.sort(key=lambda item: item["face_count"], reverse=True)
        if max_components > 0:
            accepted = accepted[:max_components]

        remove_mask = np.zeros(len(shell_mesh.faces), dtype=bool)
        for item in accepted:
            remove_mask[item["faces"]] = True

        def submesh_from_mask(mask):
            if not np.any(mask):
                return trimesh.Trimesh(
                    vertices=np.empty((0, 3), dtype=np.float64),
                    faces=np.empty((0, 3), dtype=np.int64),
                    process=False,
                )
            result = trimesh.Trimesh(
                vertices=np.asarray(shell_mesh.vertices, dtype=np.float64).copy(),
                faces=np.asarray(shell_mesh.faces, dtype=np.int64)[mask].copy(),
                process=False,
            )
            result.remove_unreferenced_vertices()
            return result

        all_candidates = submesh_from_mask(seed_mask)
        removed_walls = submesh_from_mask(remove_mask)
        opened = trimesh.Trimesh(
            vertices=np.asarray(shell_mesh.vertices, dtype=np.float64).copy(),
            faces=np.asarray(shell_mesh.faces, dtype=np.int64)[~remove_mask].copy(),
            process=False,
        )
        opened.remove_unreferenced_vertices()

        report = {
            "shell_level_world": float(shell_level_world),
            "voxel_world": float(voxel_world),
            "max_normal_dot": max_normal_dot,
            "max_distance_world": max_distance_world,
            "min_faces": min_faces,
            "min_span_world": min_span_world,
            "max_components": max_components,
            "candidate_faces": int(seed_mask.sum()),
            "removed_faces": int(remove_mask.sum()),
            "accepted_components": [
                {k: v for k, v in item.items() if k != "faces"}
                for item in accepted
            ],
            "shell_before": self._mesh_quality_stats(
                shell_mesh.vertices, shell_mesh.faces
            ),
            "shell_after": self._mesh_quality_stats(
                opened.vertices, opened.faces
            ),
        }

        print(
            "[udf shell open]",
            f"candidate_faces={report['candidate_faces']}",
            f"removed_faces={report['removed_faces']}",
            f"components={len(accepted)}",
            f"max_dot={max_normal_dot:.4g}",
            f"max_distance={max_distance_world:.6g}",
            f"min_faces={min_faces}",
            f"min_span={min_span_world:.6g}",
        )
        for index, item in enumerate(report["accepted_components"]):
            print(f"[udf shell wall {index}]", item)

        output_folder = cget("output_folder", None)
        if output_folder:
            debug_dir = op.join(str(output_folder), "udf_shell_open_debug")
            os.makedirs(debug_dir, exist_ok=True)
            shell_mesh.export(op.join(debug_dir, "udf_shell_closed.ply"))
            reference_mesh.export(
                op.join(debug_dir, "nsdudf_open_reference.ply")
            )
            if len(all_candidates.faces):
                all_candidates.export(
                    op.join(debug_dir, "udf_shell_wall_candidates.ply")
                )
            if len(removed_walls.faces):
                removed_walls.export(
                    op.join(debug_dir, "udf_shell_removed_walls.ply")
                )
            opened.export(op.join(debug_dir, "udf_shell_open.ply"))
            np.savez_compressed(
                op.join(debug_dir, "udf_shell_wall_scores.npz"),
                nearest_distance_world=nearest_distance.astype(np.float32),
                normal_alignment=normal_alignment.astype(np.float32),
                seed_mask=seed_mask,
                remove_mask=remove_mask,
            )
            with open(
                op.join(debug_dir, "udf_shell_wall_report.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(report, handle, indent=2)

        return opened

    def _extract_udf_band_shell_open(
        self,
        raw_world_udf_fn,
        *,
        domain_center,
        domain_half_extent,
        n_grid_samples,
        shell_level_world,
        far_world,
        query_chunk_size,
        batch_size,
        config,
    ):
        """Generate a closed UDF shell and open it using NSDUDF normals."""
        def cget(key, default):
            value = config.get(key) if hasattr(config, "get") else None
            return default if value is None else value

        center = np.asarray(domain_center, dtype=np.float64).reshape(3)
        half = float(domain_half_extent)
        n = max(8, int(n_grid_samples))

        shell = self._extract_udf_band_shell(
            raw_world_udf_fn,
            domain_center=center,
            domain_half_extent=half,
            n_grid_samples=n,
            shell_level_world=shell_level_world,
            far_world=far_world,
            query_chunk_size=query_chunk_size,
            config=config,
            apply_cleanup=False,
        )

        nsd_meshing, _nsd_utils, model = self._load_nsdudf(config)
        udf_and_grad_f, _dmudf_oracles, fd_stats, oracle_meta = (
            self._make_nsdudf_oracle_from_udf(
                raw_world_udf_fn,
                domain_center=center,
                domain_half_extent=half,
                n_grid_samples=n,
                far_world=far_world,
                oracle_chunk_size=query_chunk_size,
                offset_world=0.0,
            )
        )

        # The opening reference intentionally uses the earlier successful
        # threshold-relaxed NSDUDF path: plain per-cell argmax, no top-K or
        # neighborhood-consistency refinement. Keep these controls separate
        # from the normal NSDUDF mesher so later experiments cannot silently
        # alter the reference used to open the shell.
        reference_max_avg_factor = float(
            cget("udf_shell_reference_max_avg_factor", 2.0)
        )
        reference_max_max_factor = float(
            cget("udf_shell_reference_max_max_factor", 3.0)
        )
        print(
            "[udf shell open reference mode]",
            "classifier=legacy_argmax",
            f"max_avg_factor={reference_max_avg_factor:.6g}",
            f"max_max_factor={reference_max_max_factor:.6g}",
            "neighbor_consistency=False",
        )

        pseudo_sdf = nsd_meshing.compute_pseudo_sdf(
            model,
            udf_and_grad_f,
            n_grid_samples=n,
            batch_size=batch_size,
            normalize_udf=bool(cget("nsdudf_normalize_udf", True)),
            use_grads=True,
            out7=False,
            max_avg_factor=reference_max_avg_factor,
            max_max_factor=reference_max_max_factor,
            neighbor_consistency=False,
        )
        reference_cube = nsd_meshing.mesh_marching_cubes(pseudo_sdf)
        if reference_cube is None or len(reference_cube.faces) == 0:
            raise RuntimeError(
                "NSDUDF produced an empty reference for udf_band_shell_open."
            )

        reference = trimesh.Trimesh(
            vertices=(
                center[None, :]
                + np.asarray(reference_cube.vertices, dtype=np.float64) * half
            ),
            faces=np.asarray(reference_cube.faces, dtype=np.int64),
            process=False,
        )
        voxel_world = 2.0 * half / float(n - 1)
        total_axes = max(1, int(fd_stats["eval_points"]) * 3)
        print(
            "[udf shell open reference]",
            f"V={len(reference.vertices)}",
            f"F={len(reference.faces)}",
            f"fd_eps_world={oracle_meta['fd_eps_world']:.6g}",
            f"invalid_axis_pct="
            f"{100.0 * fd_stats['invalid_axes'] / total_axes:.3f}",
        )

        opened = self._open_udf_band_shell_from_nsdudf_reference(
            shell,
            reference,
            shell_level_world=shell_level_world,
            voxel_world=voxel_world,
            config=config,
        )
        if bool(cget("udf_cleanup", False)):
            opened = self._cleanup_nsdudf_mesh(opened, config)
        return opened

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

        mesher = str(
            cget("nsdudf_mesher", "marching_cubes")
        ).strip().lower()

        mesher_aliases = {
            "mc": "marching_cubes",
            "marching_cubes": "marching_cubes",
            "dualmesh": "dual_mesh_udf",
            "dmudf": "dual_mesh_udf",
            "dual_mesh_udf": "dual_mesh_udf",
            "band_shell": "udf_band_shell",
            "udf_band_shell": "udf_band_shell",
            "band_shell_open": "udf_band_shell_open",
            "udf_band_shell_open": "udf_band_shell_open",
        }

        if mesher not in mesher_aliases:
            raise ValueError(
                f"Unknown nsdudf_mesher={mesher!r}. Expected one of "
                "'marching_cubes', 'dual_mesh_udf', 'udf_band_shell', "
                "or 'udf_band_shell_open'."
            )

        mesher = mesher_aliases[mesher]

        offset_world = float(
            cget("nsdudf_offset_world", 0.0)
        )

        if mesher == "udf_band_shell":
            return self._extract_udf_band_shell(
                raw_world_udf_fn,
                domain_center=center,
                domain_half_extent=half,
                n_grid_samples=n,
                shell_level_world=offset_world,
                far_world=far_world,
                query_chunk_size=oracle_chunk_size,
                config=config,
            )

        if mesher == "udf_band_shell_open":
            if offset_world <= 0.0:
                raise ValueError(
                    "udf_band_shell_open requires "
                    "--nsdudf-offset-world > 0."
                )
            return self._extract_udf_band_shell_open(
                raw_world_udf_fn,
                domain_center=center,
                domain_half_extent=half,
                n_grid_samples=n,
                shell_level_world=offset_world,
                far_world=far_world,
                query_chunk_size=oracle_chunk_size,
                batch_size=batch_size,
                config=config,
            )

        if offset_world > 0.0 and mesher == "dual_mesh_udf":
            raise ValueError(
                "nsdudf_offset_world is currently supported only with "
                "nsdudf_mesher='marching_cubes'. The DualMesh backend still "
                "queries the original, unshifted UDF."
            )

        nsd_meshing, _nsd_utils, model = self._load_nsdudf(config)

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
            raise NotImplementedError(
                "nsdudf_gradient_mode='model_direct' is not implemented yet. "
                "Use 'finite_difference'."
            )
        else:
            udf_and_grad_f, dmudf_oracles, fd_stats, oracle_meta = (
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
            f"mesher={mesher}",
            f"offset_world={oracle_meta['offset_world']:.6g}",
            f"offset_cube={oracle_meta['offset_cube']:.6g}",
        )
      
        pseudo_sdf = nsd_meshing.compute_pseudo_sdf(
            model,
            udf_and_grad_f,
            n_grid_samples=n,
            batch_size=batch_size,
            normalize_udf=bool(cget("nsdudf_normalize_udf", True)),
            use_grads=True,
            out7=False,
            max_avg_factor=float(cget("nsdudf_max_avg_factor", 1.2)),
            max_max_factor=float(cget("nsdudf_max_max_factor", 2.0)),
            neighbor_consistency=bool(
                cget("nsdudf_neighbor_consistency", False)
            ),
            consistency_top_k=int(
                cget("nsdudf_consistency_top_k", 8)
            ),
            consistency_weight=float(
                cget("nsdudf_consistency_weight", 1.0)
            ),
            consistency_sweeps=int(
                cget("nsdudf_consistency_sweeps", 5)
            ),
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

        if mesher == "marching_cubes":
            mesh_cube = nsd_meshing.mesh_marching_cubes(pseudo_sdf)

            if mesh_cube is None or len(mesh_cube.faces) == 0:
                print("[nsdudf] marching cubes produced an empty mesh")
                return self._empty_mesh()

            print(
                "[nsdudf mc cube]",
                self._mesh_quality_stats(
                    mesh_cube.vertices,
                    mesh_cube.faces,
                ),
            )

        else:
            # The reference NSDUDF+DualMesh implementation requires the
            # pseudo-SDF to contain 2^depth cells:
            #
            #   65 samples  -> 64 cells  -> depth 6
            #   129 samples -> 128 cells -> depth 7
            cells_per_axis = n - 1
            depth_float = math.log2(cells_per_axis)

            if not depth_float.is_integer():
                raise ValueError(
                    "NSDUDF + DualMesh-UDF requires n_grid_samples - 1 "
                    "to be a power of two. Use 65, 129, or 257 samples. "
                    f"Got n_grid_samples={n}, cells={cells_per_axis}."
                )

            dualmesh_batch_size = int(
                cget("nsdudf_dualmesh_batch_size", batch_size)
            )

            print(
                "[nsdudf+dmudf]",
                f"grid_samples={n}",
                f"cells={cells_per_axis}",
                f"depth={int(depth_float)}",
                f"batch={dualmesh_batch_size}",
            )

            # NSDUDF's bundled DualMesh implementation uses libigl's older
            # NumPy signatures. Apply the compatibility wrapper already used
            # by the plain DualMesh path.
            self._ensure_udf_igl_patch()

            plain_dmudf_cube, combined_cube = (
                nsd_meshing.mesh_dual_mesh_udf(
                    pseudo_sdf,
                    dmudf_oracles["udf_func"],
                    dmudf_oracles["udf_grad_func"],
                    batch_size=dualmesh_batch_size,

                    # Keep this on CPU. The upstream helper creates its
                    # pseudo-SDF lookup tensor on CPU, so passing "cuda"
                    # creates a device mismatch.
                    device="cpu",
                )
            )

            print(
                "[nsdudf+dmudf plain cube]",
                self._mesh_quality_stats(
                    plain_dmudf_cube.vertices,
                    plain_dmudf_cube.faces,
                ),
            )
            print(
                "[nsdudf+dmudf combined cube]",
                self._mesh_quality_stats(
                    combined_cube.vertices,
                    combined_cube.faces,
                ),
            )

            mesh_cube = combined_cube

            if mesh_cube is None or len(mesh_cube.faces) == 0:
                print("[nsdudf+dmudf] produced an empty mesh")
                return self._empty_mesh()

        # Both NSDUDF+MC and NSDUDF+DualMesh return vertices in the
        # normalized extraction cube [-1, 1]^3.
        world_vertices = (
            center[None, :]
            + np.asarray(mesh_cube.vertices, dtype=np.float64) * half
        )

        mesh = trimesh.Trimesh(
            vertices=world_vertices,
            faces=np.asarray(mesh_cube.faces, dtype=np.int64),
            process=False,
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
