import math
import os
import os.path as op

import numpy as np
import torch
import trimesh

import app_utils_3dvec as utils
from agent_3dvec_base import AgentBase


class AgentUDF(AgentBase):
    """Minimal direct-query UDF adaptation agent.

    The trained model is queried directly at the points requested by
    DualMesh-UDF. There is no dense UDF raster, EDT continuation, positive
    iso-level, local zero calibration, model-direct fallback grid, tiled
    detail, snug field, marching cubes, or RFTA.

    This diagnostic keeps the model's raw UDF values. World-space UDF values
    are divided only by the extraction-domain half extent to convert them to
    DualMesh cube units. Gradients are estimated with central differences at
    one quarter of the finest octree-cell width.
    """

    _IGNORED_SDF_KEYS = {
        "accessory_offset_mode",
        "auto_avatar_snug_field",
        "avatar_clearance",
        "cut_avatar",
        "detail_avatar_gate",
        "final_carve",
        "hard_clearance",
        "local_offset_band",
        "local_offset_gate_sigma",
        "local_offset_strength",
        "snug_mode",
        "use_hard_avatar_clamp",
        "use_soft_snug",
        "use_tiled_detail",
    }

    @staticmethod
    def _empty_mesh():
        return trimesh.Trimesh(
            vertices=np.zeros((0, 3), dtype=np.float64),
            faces=np.zeros((0, 3), dtype=np.int64),
            process=False,
        )

    def _udf_clamp(self, values):
        """True UDF: remove only negative numerical/model output; never abs()."""
        return np.maximum(np.asarray(values, dtype=np.float64), 0.0)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    @staticmethod
    def _expand_adaptations(adaptation_items):
        if not isinstance(adaptation_items, list):
            raise TypeError(
                "'adaptations'/'adaptation' must be a YAML list, got "
                f"{type(adaptation_items).__name__}."
            )

        expanded_items = []
        for yaml_index, item in enumerate(adaptation_items):
            if not isinstance(item, dict):
                raise TypeError(
                    f"adaptation entry {yaml_index} must be a mapping, got "
                    f"{type(item).__name__}."
                )
            if not bool(item.get("enabled", True)):
                print(f"[udf] skipping disabled YAML entry {yaml_index}")
                continue

            target_spec = item.get("target_keys", item.get("target_key"))
            if target_spec is None:
                raise KeyError(
                    f"adaptation entry {yaml_index} needs 'target_key' or "
                    "'target_keys'."
                )
            if isinstance(target_spec, str):
                target_keys = [target_spec]
            elif isinstance(target_spec, (list, tuple)):
                target_keys = list(target_spec)
            else:
                raise TypeError(
                    f"target_key(s) in adaptation entry {yaml_index} must be "
                    f"a string or list, got {type(target_spec).__name__}."
                )

            for target_index, target_key in enumerate(target_keys):
                expanded = dict(item)
                expanded.pop("target_keys", None)
                expanded["target_key"] = str(target_key)
                expanded["_yaml_index"] = yaml_index
                expanded["_target_index"] = target_index
                expanded_items.append(expanded)

        if not expanded_items:
            raise RuntimeError("No enabled adaptation entries were found.")
        return expanded_items

    def _load_adaptations(self, arg):
        raw = utils.load_yaml_file(arg["adapt_file"])
        extraction_config = dict(arg)

        if isinstance(raw, list):
            items = raw
        elif isinstance(raw, dict):
            for key, value in raw.items():
                if str(key).startswith("udf_"):
                    extraction_config[key] = value
            items = raw.get("adaptations", raw.get("adaptation"))
            if items is None:
                raise ValueError(
                    "Adaptation YAML mapping must contain 'adaptations:' "
                    "or 'adaptation:'."
                )
        else:
            raise TypeError(
                "Adaptation YAML must load as a list or mapping, got "
                f"{type(raw).__name__}."
            )

        return self._expand_adaptations(items), extraction_config

    def _extraction_config_for_item(self, base_config, item):
        config = dict(base_config)
        for key, value in item.items():
            if str(key).startswith("udf_"):
                config[key] = value
        return config

    def _warn_ignored_sdf_settings(self, adapt_arg):
        active = []
        for key in sorted(self._IGNORED_SDF_KEYS):
            if key not in adapt_arg:
                continue
            value = adapt_arg[key]
            if value not in (None, False, "none", "None", 0, 0.0):
                active.append(f"{key}={value!r}")
        if active:
            print("[udf] ignored SDF-only settings:", ", ".join(active))

    # ------------------------------------------------------------------
    # Model oracle
    # ------------------------------------------------------------------
    def _query_adapted_raw_udf(
        self,
        world_points,
        *,
        avatar_curve,
        adapt_arg,
        accessory_key,
        batch_size,
        far_world,
        return_valid=False,
    ):
        """Query the adapted network field at arbitrary world points.

        Points outside the localization support receive ``far_world``.  Errors
        are intentionally not swallowed: a broken oracle must fail visibly.
        """
        world_points = np.asarray(world_points, dtype=np.float64).reshape(-1, 3)
        output = np.full(world_points.shape[0], float(far_world), dtype=np.float64)
        valid = np.zeros(world_points.shape[0], dtype=bool)

        accessory_data, _avatar_data, inside = (
            avatar_curve.core.localize_samples_adapt(world_points, adapt_arg)
        )
        inside = np.asarray(inside, dtype=np.int64).reshape(-1)
        if inside.size == 0:
            return (output, valid) if return_valid else output

        inferred = self._inference_full_vals(
            accessory_data,
            accessory_key,
            batch_size=int(batch_size),
        )
        values = self._udf_clamp(inferred["dist"]).reshape(-1)
        if values.shape[0] != inside.shape[0]:
            raise ValueError(
                "Adapted UDF query length mismatch: "
                f"values={values.shape[0]}, inside={inside.shape[0]}."
            )
        output[inside] = values
        valid[inside] = True
        return (output, valid) if return_valid else output

    def _make_dualmesh_oracle(
        self,
        raw_world_udf_fn,
        *,
        domain_center,
        domain_half_extent,
        max_depth,
        far_world,
    ):
        """Build the UDF and gradient callables expected by DualMesh-UDF.

        DualMesh uses coordinates ``u`` in ``[-1, 1]^3`` with

            world = center + half_extent * u.

        The model output is treated as a raw world-space UDF. It is converted
        to DualMesh cube units by dividing by ``half_extent``. The gradient
        direction is estimated with validity-aware central differences using
        one quarter of the finest octree-cell width.

        No one-sided differences and no ``f / ||grad(f)||`` re-distance
        correction are used. If any central-difference axis cannot be
        localized, that gradient is marked invalid instead of using the
        artificial far value to manufacture a tangent plane.
        """
        center = np.asarray(domain_center, dtype=np.float64).reshape(3)
        half = float(domain_half_extent)
        if not np.isfinite(half) or half <= 0.0:
            raise ValueError(f"Invalid UDF domain half extent: {half}")

        cells_per_axis = float(2 ** int(max_depth))
        cell_width_u = 2.0 / cells_per_axis
        eps_u = 0.25 * cell_width_u

        far_cube = float(far_world) / half
        grad_floor = 1.0e-8
        stats = {
            "eval_points": 0,
            "central_axes": 0,
            "invalid_axes": 0,
            "invalid_gradients": 0,
        }

        def to_world(u):
            u = np.asarray(u, dtype=np.float64).reshape(-1, 3)
            return center[None, :] + half * u

        def raw_u(u):
            result = raw_world_udf_fn(to_world(u))
            if isinstance(result, tuple):
                values, valid = result
                values = self._udf_clamp(values).reshape(-1)
                valid = np.asarray(valid, dtype=bool).reshape(-1)
            else:
                values = self._udf_clamp(result).reshape(-1)
                valid = np.ones(values.shape[0], dtype=bool)

            if values.shape[0] != valid.shape[0]:
                raise ValueError(
                    "UDF oracle value/validity length mismatch: "
                    f"values={values.shape[0]}, valid={valid.shape[0]}."
                )
            return values, valid

        def evaluate(u):
            u = np.asarray(u, dtype=np.float64).reshape(-1, 3)
            raw, center_valid = raw_u(u)
            n_points = int(u.shape[0])

            grad_u = np.zeros_like(u)
            axis_valid = np.zeros_like(u, dtype=bool)
            stats["eval_points"] += n_points

            active = np.flatnonzero(center_valid)
            for axis in range(3):
                if active.size == 0:
                    stats["invalid_axes"] += n_points
                    continue

                plus = u[active].copy()
                minus = u[active].copy()
                plus[:, axis] += eps_u
                minus[:, axis] -= eps_u

                plus_value, plus_valid = raw_u(plus)
                minus_value, minus_valid = raw_u(minus)
                both_valid = plus_valid & minus_valid
                rows = active[both_valid]

                if rows.size:
                    grad_u[rows, axis] = (
                        plus_value[both_valid] - minus_value[both_valid]
                    ) / (2.0 * eps_u)
                    axis_valid[rows, axis] = True

                stats["central_axes"] += int(rows.size)
                stats["invalid_axes"] += n_points - int(rows.size)

            grad_norm = np.linalg.norm(grad_u, axis=1)
            valid_gradient = (
                center_valid
                & np.all(axis_valid, axis=1)
                & np.isfinite(grad_norm)
                & (grad_norm > grad_floor)
            )
            stats["invalid_gradients"] += int(n_points - np.sum(valid_gradient))

            # Distance and gradient now come from the same raw model field.
            # Convert only from world-distance units to DualMesh cube units.
            distance_cube = np.full(n_points, far_cube, dtype=np.float64)
            distance_cube[center_valid] = np.minimum(
                raw[center_valid] / half,
                far_cube,
            )
            distance_cube = np.maximum(distance_cube, 0.0)

            unit_grad = np.zeros_like(grad_u)
            unit_grad[valid_gradient] = (
                grad_u[valid_gradient] / grad_norm[valid_gradient, None]
            )
            return distance_cube, unit_grad

        def udf_func(points):
            distance, _gradient = evaluate(points)
            return distance.reshape(-1, 1).astype(np.float32)

        def udf_grad_func(points):
            distance, gradient = evaluate(points)
            return (
                distance.reshape(-1, 1).astype(np.float32),
                gradient.astype(np.float32),
            )

        return udf_func, udf_grad_func, eps_u, stats

    # ------------------------------------------------------------------
    # DualMesh-UDF extraction
    # ------------------------------------------------------------------
    def _ensure_udf_igl_patch(self):
        """Make DualMesh-UDF compatible with newer libigl numpy signatures."""
        import igl

        if getattr(igl, "_oktopus_udf_patched", False):
            return

        original_dedup = getattr(igl, "remove_duplicate_vertices", None)
        original_unref = getattr(igl, "remove_unreferenced", None)

        def numpy_dedup(vertices, faces, epsilon):
            vertices = np.ascontiguousarray(vertices, dtype=np.float64)
            faces = np.ascontiguousarray(faces, dtype=np.int64)
            key = np.round(vertices / max(float(epsilon), 1.0e-30))
            _, unique_indices, inverse = np.unique(
                key, axis=0, return_index=True, return_inverse=True
            )
            inverse = inverse.reshape(-1)
            remapped_faces = inverse[faces]
            return (
                vertices[unique_indices],
                unique_indices.astype(np.int64),
                inverse.astype(np.int64),
                remapped_faces.astype(np.int64),
            )

        def dedup(*args):
            vertices = np.ascontiguousarray(args[0], dtype=np.float64)
            if len(args) == 3:
                faces = np.ascontiguousarray(args[1], dtype=np.int64)
                epsilon = float(args[2])
                if original_dedup is not None:
                    try:
                        return original_dedup(vertices, faces, epsilon)
                    except TypeError:
                        pass
                return numpy_dedup(vertices, faces, epsilon)

            epsilon = float(args[1])
            if original_dedup is not None:
                try:
                    return original_dedup(vertices, epsilon)
                except TypeError:
                    pass
            key = np.round(vertices / max(epsilon, 1.0e-30))
            _, unique_indices, inverse = np.unique(
                key, axis=0, return_index=True, return_inverse=True
            )
            return (
                vertices[unique_indices],
                unique_indices.astype(np.int64),
                inverse.reshape(-1).astype(np.int64),
            )

        def unref(vertices, faces):
            vertices = np.ascontiguousarray(vertices, dtype=np.float64)
            faces = np.ascontiguousarray(faces, dtype=np.int64)
            if original_unref is not None:
                try:
                    return original_unref(vertices, faces)
                except TypeError:
                    pass

            used = np.unique(faces.reshape(-1))
            remap = -np.ones(vertices.shape[0], dtype=np.int64)
            remap[used] = np.arange(used.shape[0], dtype=np.int64)
            return (
                vertices[used],
                remap[faces].astype(np.int64),
                used.astype(np.int64),
                remap,
            )

        igl.remove_duplicate_vertices = dedup
        igl.remove_unreferenced = unref
        igl._oktopus_udf_patched = True

    def _dualmeshudf_extract(self, udf_func, udf_grad_func, *, batch_size,
                             max_depth, reliable, sample_threshold, sampling_depth):
        """Configurable DualMeshUDF octree loop.

        Exposes the reliability threshold (used by adaptive_subdivide AND grid
        validity) and a SEPARATE batch-solve `sample_threshold`, plus per-cell
        `sampling_depth`. Stock extract_mesh hardcodes reliable=0.002; this lets
        us match the learned UDF's positive floor. Logs projection statistics
        for the GT-vs-learned threshold comparison.
        """
        import numpy as _np
        import igl as _igl
        from DualMeshUDF_core import Octree, triangulate_faces
        from DualMeshUDF.extract_mesh import query_udf, query_udf_and_grad
        octree = Octree(max_depth=int(max_depth),
                        min_corner=_np.array([[-1.], [-1.], [-1.]]),
                        max_corner=_np.array([[1.], [1.], [1.]]),
                        sampling_depth=int(sampling_depth))
        cur = 0
        while cur <= int(max_depth):
            cen = octree.centroids_of_new_nodes().astype(_np.float32)
            cu, cg = query_udf_and_grad(udf_grad_func, cen, batch_size)
            octree.adaptive_subdivide(cu, cg, reliable)
            cur += 1
        gi, gc = octree.get_samples_of_new_nodes()
        gu, gg = query_udf_and_grad(udf_grad_func, gc.astype(_np.float32), batch_size)
        octree.set_new_grid_data(gi, gu, gg)
        idx, proj = octree.get_projections_for_checking_validity()
        pu = _np.asarray(query_udf(udf_func, proj, batch_size)).reshape(-1)
        pv = pu < reliable
        pct = (_np.percentile(pu, [0, 1, 5, 25, 50, 95]).tolist()
               if pu.size else [])
        print("[dmudf loop] reliable=", reliable,
              "sample_threshold=", sample_threshold,
              "sampling_depth=", int(sampling_depth),
              "n_proj=", int(pu.size), "n_valid=", int(pv.sum()),
              "valid_pct=", round(100.0 * float(pv.sum()) / max(pu.size, 1), 2),
              "pu_pct[0,1,5,25,50,95]=",
              [round(float(x), 6) for x in pct])
        octree.set_grid_validity(idx, pv)
        octree.batch_solve(float(sample_threshold), 1.0, 1.0, 0.15, 0.08)
        octree.generate_mesh()
        print("[dmudf loop] pre-triangulate mesh_v=", len(octree.mesh_v),
              "mesh_f=", len(octree.mesh_f))
        tri = triangulate_faces(octree.mesh_v, octree.mesh_f,
                                octree.v_type, octree.mesh_v_dir)
        v, _, _, f = _igl.remove_duplicate_vertices(
            _np.array(octree.mesh_v), tri, 1e-7)
        v, f, _, _ = _igl.remove_unreferenced(v, f)
        return _np.asarray(v, dtype=_np.float64), _np.asarray(f, dtype=_np.int64)

    @staticmethod
    def _mesh_quality_stats(v, f):
        """Boundary / non-manifold / largest-component / degenerate stats."""
        import numpy as _np
        import collections as _col
        v = _np.asarray(v, dtype=_np.float64)
        f = _np.asarray(f, dtype=_np.int64)
        nv, nf = int(len(v)), int(len(f))
        if nf == 0:
            return {"V": nv, "F": 0, "boundary": 0, "nonmanifold": 0,
                    "largest_face_pct": 0.0, "degenerate": 0}
        ec = _col.Counter()
        for tri in f:
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                ec[(int(min(a, b)), int(max(a, b)))] += 1
        boundary = sum(1 for c in ec.values() if c == 1)
        nonman = sum(1 for c in ec.values() if c > 2)
        deg_idx = (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])
        e1 = v[f[:, 1]] - v[f[:, 0]]
        e2 = v[f[:, 2]] - v[f[:, 0]]
        area = 0.5 * _np.linalg.norm(_np.cross(e1, e2), axis=1)
        degen = int((deg_idx | (area < 1e-12)).sum())
        par = list(range(nv))
        def _find(a):
            while par[a] != a:
                par[a] = par[par[a]]; a = par[a]
            return a
        for tri in f:
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2])):
                ra, rb = _find(int(a)), _find(int(b))
                if ra != rb:
                    par[ra] = rb
        roots = _col.Counter(_find(int(t[0])) for t in f)
        largest = 100.0 * max(roots.values()) / nf if roots else 0.0
        return {"V": nv, "F": nf, "boundary": boundary, "nonmanifold": nonman,
                "largest_face_pct": round(largest, 2), "degenerate": degen}

    def extract_udf_mesh_from_model(
        self,
        raw_world_udf_fn,
        *,
        domain_center,
        domain_half_extent,
        resolution,
        config,
    ):
        """Extract via the configurable DualMeshUDF octree loop (reliable /
        sample_threshold / sampling_depth exposed; default reliable=0.002 keeps
        stock behavior)."""
        max_depth = int(
            config.get(
                "udf_max_depth",
                max(1, int(round(math.log2(max(int(resolution), 2))))),
            )
        )
        batch_size = int(config.get("udf_batch_size", 150000))
        far_world = float(config.get("udf_far_value", 0.1))
        reliable = float(config.get("udf_reliable_threshold", 0.002))
        sample_threshold = float(config.get(
            "udf_sample_threshold", min(0.25 * reliable, 0.005)))
        sampling_depth = int(config.get("udf_sampling_depth", 1))
        udf_func, udf_grad_func, eps_u, fd_stats = self._make_dualmesh_oracle(
            raw_world_udf_fn,
            domain_center=domain_center,
            domain_half_extent=domain_half_extent,
            max_depth=max_depth,
            far_world=far_world,
        )

        print(
            "[dualmeshudf direct]",
            f"max_depth={max_depth}",
            f"batch={batch_size}",
            f"center={np.asarray(domain_center)}",
            f"half={float(domain_half_extent):.6g}",
            f"fd_base_eps_u={eps_u:.6g}",
            f"fd_base_eps_world={float(domain_half_extent) * eps_u:.6g}",
            "fd_mode=fixed_quarter_cell_central_only",
            "distance=raw_udf/half_extent",
            f"reliable={reliable:.6g}",
            f"sample_threshold={sample_threshold:.6g}",
            f"sampling_depth={sampling_depth}",
        )

        self._ensure_udf_igl_patch()
        vertices, faces = self._dualmeshudf_extract(
            udf_func, udf_grad_func,
            batch_size=batch_size, max_depth=max_depth,
            reliable=reliable, sample_threshold=sample_threshold,
            sampling_depth=sampling_depth,
        )
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        print(
            f"[dualmeshudf direct] extracted V={len(vertices)} F={len(faces)}"
        )
        total_axes = max(1, int(fd_stats["eval_points"]) * 3)
        print(
            "[dualmeshudf fd]",
            f"eval_points={fd_stats['eval_points']}",
            f"central_axes={fd_stats['central_axes']}",
            f"invalid_axes={fd_stats['invalid_axes']}",
            f"invalid_axis_pct={100.0 * fd_stats['invalid_axes'] / total_axes:.3f}",
            f"invalid_gradients={fd_stats['invalid_gradients']}",
        )
        print("[dmudf quality]", self._mesh_quality_stats(vertices, faces))
        if len(vertices) == 0 or len(faces) == 0:
            return self._empty_mesh()

        center = np.asarray(domain_center, dtype=np.float64).reshape(3)
        world_vertices = center[None, :] + vertices * float(domain_half_extent)
        return trimesh.Trimesh(
            vertices=world_vertices,
            faces=faces,
            process=False,
        )

    # Deliberately reject the old raster route.  This prevents another caller
    # from silently reintroducing EDT/band/iso-level extraction.
    def extract_surface_mesh(self, *_args, **_kwargs):
        raise NotImplementedError(
            "AgentUDF uses direct model queries only. The raster UDF + EDT "
            "extraction path was intentionally removed."
        )

    # ------------------------------------------------------------------
    # Debug/domain helpers
    # ------------------------------------------------------------------
    def _export_prepass_and_choose_domain(
        self,
        *,
        world_points,
        udf_values,
        output_folder,
        item_index,
        accessory_key,
        config,
    ):
        world_points = np.asarray(world_points, dtype=np.float64).reshape(-1, 3)
        udf_values = self._udf_clamp(udf_values).reshape(-1)
        if world_points.shape[0] != udf_values.shape[0]:
            raise ValueError(
                "UDF prepass alignment mismatch: "
                f"points={world_points.shape[0]}, values={udf_values.shape[0]}."
            )

        finite = np.isfinite(udf_values)
        if not finite.any():
            raise RuntimeError("No finite model UDF values in the adaptation prepass.")

        percentiles = np.percentile(
            udf_values[finite], [0, 1, 5, 25, 50, 95, 100]
        )
        print(
            "[udf prepass]",
            f"n={udf_values.size}",
            "pct[0,1,5,25,50,95,100]=",
            [round(float(x), 6) for x in percentiles],
        )

        debug_bands = config.get("udf_debug_bands", [0.01, 0.02, 0.03, 0.05])
        if bool(config.get("udf_export_near_points", True)):
            debug_dir = op.join(output_folder, "udf_debug")
            os.makedirs(debug_dir, exist_ok=True)
            safe_key = accessory_key.replace("|", "_").replace("/", "_")
            for band in debug_bands:
                band = float(band)
                mask = finite & (udf_values < band)
                print(f"[udf near] band={band:.6g} count={int(mask.sum())}")
                if mask.any():
                    trimesh.PointCloud(world_points[mask]).export(
                        op.join(
                            debug_dir,
                            f"{item_index}_{safe_key}_near_{band:.3f}.ply",
                        )
                    )

        # The GT test normalizes the actual mesh bbox into [-1,1]^3.  We do not
        # know the adapted mesh yet, so approximate its bbox from low-UDF model
        # samples.  This only chooses the extraction domain; it does not modify
        # the UDF values used by DualMesh.
        domain_band = float(config.get("udf_domain_band", 0.05))
        domain_min_points = int(config.get("udf_domain_min_points", 128))
        near = finite & (udf_values < domain_band)
        if int(near.sum()) >= domain_min_points:
            domain_points = world_points[near]
            domain_source = f"UDF<{domain_band:g}"
        else:
            domain_points = world_points[finite]
            domain_source = "full localized support"

        lower = domain_points.min(axis=0)
        upper = domain_points.max(axis=0)
        center = 0.5 * (lower + upper)
        padding = float(config.get("udf_domain_padding", 0.15))
        half = 0.5 * float(np.max(upper - lower)) * (1.0 + padding)
        minimum_half = float(config.get("udf_domain_min_half", 1.0e-4))
        half = max(half, minimum_half)

        print(
            "[udf domain]",
            f"source={domain_source}",
            f"n={domain_points.shape[0]}",
            f"center={center}",
            f"half={half:.6g}",
        )
        return center, half

    # ------------------------------------------------------------------
    # Adaptation action
    # ------------------------------------------------------------------
    @torch.no_grad()
    def action_part_adapt(self, arg):
        output_folder = arg["output_folder"]
        mc_grid = arg["mc_grid"]
        shape_name = arg["shape"]
        data_root = arg["data_root"]

        adaptation_items, extraction_config = self._load_adaptations(arg)
        handle = self.load_shape_handle(data_root, shape_name, "avatar")
        target_curves = {
            self.encode_key(shape_name, curve.name): curve
            for curve in handle.curves
        }

        missing = sorted(
            {
                item["target_key"]
                for item in adaptation_items
                if item["target_key"] not in target_curves
            }
        )
        if missing:
            raise KeyError(
                f"Missing target curves: {missing}. Available: "
                f"{sorted(target_curves)}"
            )

        os.makedirs(output_folder, exist_ok=True)
        batch_size = int(extraction_config.get("udf_model_batch_size", 64 ** 3))

        print(
            "[udf part_adapt]",
            f"adaptations={len(adaptation_items)}",
            "extractor=direct DualMesh-UDF",
            "grid_EDT=False",
            "positive_level=False",
        )

        for item_index, item in enumerate(adaptation_items):
            target_key = item["target_key"]
            accessory_key = item["accessory_key"]
            mode = str(item.get("mode", "direct"))
            if mode != "direct":
                raise ValueError(
                    f"Minimal AgentUDF supports mode='direct' only, got {mode!r}."
                )

            avatar_curve = target_curves[target_key]
            accessory_curve = self.curve_from_key(accessory_key)
            config = self._extraction_config_for_item(extraction_config, item)

            adapt_arg = {
                "mode": "direct",
                "avatar_curve_handle": avatar_curve,
                "accessory_curve_handle": accessory_curve,
                "device": self.device,
                "infer_scale": 2.0,
                "avatar_curve_idx": self.feat_dict[target_key],
                "accessory_curve_idx": self.feat_dict[accessory_key],
            }
            adapt_arg.update(item)
            adapt_arg["auto_avatar_snug_field"] = False
            adapt_arg["use_tiled_detail"] = False
            self._warn_ignored_sdf_settings(adapt_arg)

            run_name = str(
                item.get("name", item.get("run_name", f"adaptation_{item_index:02d}"))
            )
            print(
                f"[udf {item_index + 1}/{len(adaptation_items)}]",
                f"name={run_name}",
                f"target={target_key}",
                f"accessory={accessory_key}",
                f"scale={adapt_arg.get('scale', 1.0)}",
            )

            # One regular-grid pass is retained only to obtain a finite support
            # cloud for diagnostics and for selecting the extraction bbox.  Its
            # UDF grid is never used by DualMesh-UDF.
            support_grid = utils.create_grid_like(mc_grid)
            support_grid.clear_grid()
            accessory_data, _avatar_data, kidx, _inside = (
                avatar_curve.filter_grid_adapt(support_grid, adapt_arg)
            )
            prepass = self._inference_full_vals(
                accessory_data,
                accessory_key,
                batch_size=batch_size,
            )
            prepass_udf = self._udf_clamp(prepass["dist"])
            support_world = np.asarray(
                mc_grid.idx2pts(kidx), dtype=np.float64
            ).reshape(-1, 3)

            domain_center, domain_half = self._export_prepass_and_choose_domain(
                world_points=support_world,
                udf_values=prepass_udf,
                output_folder=output_folder,
                item_index=item_index,
                accessory_key=accessory_key,
                config=config,
            )

            oracle_arg = dict(adapt_arg)
            oracle_arg["adapt_debug_counts"] = False
            oracle_arg["debug_interval_projection"] = False
            far_world = float(config.get("udf_far_value", 0.1))

            def raw_world_udf_fn(
                world_points,
                _curve=avatar_curve,
                _adapt_arg=oracle_arg,
                _accessory_key=accessory_key,
                _batch_size=batch_size,
                _far_world=far_world,
            ):
                return self._query_adapted_raw_udf(
                    world_points,
                    avatar_curve=_curve,
                    adapt_arg=_adapt_arg,
                    accessory_key=_accessory_key,
                    batch_size=_batch_size,
                    far_world=_far_world,
                    return_valid=True,
                )

            mesh = self.extract_udf_mesh_from_model(
                raw_world_udf_fn,
                domain_center=domain_center,
                domain_half_extent=domain_half,
                resolution=int(mc_grid.reso),
                config=config,
            )

            if len(mesh.faces) == 0:
                print(f"[udf] EMPTY mesh for {accessory_key}")
                continue

            safe_key = accessory_key.replace("|", "_").replace("/", "_")
            output_path = op.join(
                output_folder,
                f"{item_index}_{mode}_{safe_key}.ply",
            )
            mesh.export(output_path)

            components = mesh.split(only_watertight=False)
            component_faces = sorted(
                (len(component.faces) for component in components), reverse=True
            )
            print(
                "[udf export]",
                output_path,
                f"V={len(mesh.vertices)}",
                f"F={len(mesh.faces)}",
                f"components={len(components)}",
                f"faces_top10={component_faces[:10]}",
            )

    @torch.no_grad()
    def action_ngcnet_inference(self, arg):
        """NATIVE direct UDF inference (overrides the grid-based base version).

        Reconstruct each accessory in its OWN training coordinate frame via
        direct model queries + DualMeshUDF -- no adaptation, no raster grid,
        no EDT. Closest thing to extract_from_gt, and the clean test of whether
        the checkpoint itself has usable UDF values/gradients. Slow (every
        octree query runs localize + inference), but diagnostic.
        """
        data_root = arg['data_root']
        data_path = arg['data_path']
        self.load_data(data_root, data_path)
        mc_grid = arg['mc_grid']
        reso = int(mc_grid.reso)
        size = float(mc_grid.size)
        output_folder = arg['output_folder']
        checkpoint = arg['checkpoint']
        config = dict(arg)

        far = float(config.get('udf_far_value', 0.1))
        bs = int(config.get('udf_batch_size', 150000))
        norm = float(config.get('udf_localize_norm', 1.0))
        band = float(config.get('udf_domain_band', 0.05))
        pad = float(config.get('udf_domain_padding', 0.15))
        coarse = int(config.get('udf_domain_scan_reso', 48))
        os.makedirs(output_folder, exist_ok=True)

        for shape_name, handle in self.handles.items():
            for curve in handle.curves:
                key = self.encode_key(shape_name, curve.name)
                if key not in self.feat_dict:
                    continue

                def native_raw(world_pts, _c=curve, _k=key):
                    world_pts = np.asarray(
                        world_pts, dtype=np.float64).reshape(-1, 3)
                    out = np.full(world_pts.shape[0], far, dtype=np.float64)
                    valid = np.zeros(world_pts.shape[0], dtype=bool)
                    cd, inside = _c.core.localize_samples(
                        world_pts, norm=norm, k_project=1)
                    inside = np.asarray(inside, dtype=np.int64).reshape(-1)
                    if inside.size:
                        vals, _vb = self._inference_vals(cd, _k, batch_size=bs)
                        vals = self._udf_clamp(
                            np.asarray(vals, dtype=np.float64).reshape(-1))
                        if vals.shape[0] != inside.shape[0]:
                            raise ValueError(
                                "Native UDF query length mismatch: "
                                f"values={vals.shape[0]} inside={inside.shape[0]}")
                        out[inside] = vals
                        valid[inside] = True
                    return out, valid

                # coarse scan for the native near-surface bbox (the domain).
                lin = np.linspace(-size, size, coarse)
                gx, gy, gz = np.meshgrid(lin, lin, lin, indexing='ij')
                gp = np.stack([gx.reshape(-1), gy.reshape(-1),
                               gz.reshape(-1)], axis=1)
                gu, gv = native_raw(gp)
                m = gv & (gu < band)
                print("[native udf]", key, "scan_reso=", coarse,
                      "near(", band, ")=", int(m.sum()), "/", gp.shape[0])
                if int(m.sum()) < 8:
                    print("[native udf]", key,
                          ": too few near-surface points, skipping")
                    continue
                pts = gp[m]
                bmin = pts.min(0); bmax = pts.max(0)
                center = 0.5 * (bmin + bmax)
                half = 0.5 * float(np.max(bmax - bmin)) * (1.0 + pad)
                print("[native udf]", key, "domain center=", center,
                      "half=", round(half, 5))

                mesh = self.extract_udf_mesh_from_model(
                    native_raw,
                    domain_center=center,
                    domain_half_extent=half,
                    resolution=reso,
                    config=config,
                )
                mesh_file = op.join(
                    output_folder, shape_name,
                    f"{shape_name}_{curve.name}_{checkpoint}_native{reso}.ply")
                os.makedirs(op.dirname(mesh_file), exist_ok=True)
                mesh.export(mesh_file)
                print("[native udf] saved", mesh_file,
                      "V=", len(mesh.vertices), "F=", len(mesh.faces))
