import os, pickle
import os.path as op
import numpy as np
import trimesh
import torch
from time import time
from tqdm.autonotebook import tqdm

#import app_utils as utils
import app_utils_3dvec as utils


from agent_3dvec_base import AgentBase

class AgentUDF(AgentBase):
    """Unsigned-distance agent. UDF is always >= 0: no sign, no inside/outside,
    no marching cubes, no RFTA, no signed boolean carve/offset/detail.
    Extraction is DualMeshUDF only (grid or model-direct). The adaptation
    front (geometry + network inference) is shared with SDF via AgentBase;
    the finalize is just clamp(dist>=0)."""

    def _udf_clamp(self, x):
        """True-UDF clamp: strip tiny negative numerical garbage only. NOT abs().
        AgentSDFasUDF overrides this with abs() for the diagnostic
        'old signed checkpoint -> abs(SDF) -> DualMeshUDF' case."""
        return np.maximum(np.asarray(x, dtype=np.float64), 0.0)

    def build_avatar_snug_scale_field(self, *a, **k):
        raise NotImplementedError(
            "AgentUDF is intentionally minimal: the implicit avatar snug-wrap "
            "field is SDF-only. Use auto_avatar_snug_field: false for UDF runs "
            "(or AgentSDF).")

    def add_tiled_detail_coords_for_adapt(self, *a, **k):
        raise NotImplementedError(
            "AgentUDF is intentionally minimal: tiled signed-detail is SDF-only. "
            "Use use_tiled_detail: false for UDF runs (or AgentSDF).")

    def extract_surface_mesh(self, sdf_grid, config=None,
                             mc_method='extract_mesh', context=''):
        """UDF has ONE grid extraction family: DualMeshUDF. (model-direct is
        dispatched separately in action_part_adapt.) No marching cubes, no RFTA,
        no signed boolean ops."""
        config = {} if config is None else config
        return self.extract_udf_mesh_from_grid(sdf_grid, config)

    def extract_udf_mesh_from_grid(self, sdf_grid, config=None):
        """Extract a mesh from an UNSIGNED-distance grid via DualMeshUDF.

        Keep the network UDF inside a thin near-surface band and continue it
        smoothly with an EDT beyond, so the zero-set stays thin (a fat zero
        plateau collapses DualMeshUDF). Output is mapped back with the SAME
        origin/step/flip marching cubes uses. Knobs: udf_surface_band,
        udf_reliable_threshold, udf_max_depth, udf_batch_size, udf_fill_value.
        """
        from scipy.interpolate import RegularGridInterpolator
        from scipy.ndimage import distance_transform_edt
        import math
        config = {} if config is None else config
        gc = sdf_grid.grid_config
        N = int(sdf_grid.reso); N1 = N + 1
        size = float(sdf_grid.size)
        step = float(gc['step'])
        empty_val = float(config.get('udf_fill_value', 10.0))

        raw0 = np.asarray(sdf_grid.val_grid, dtype=np.float64).reshape(N1, N1, N1)
        # true UDF: clamp only tiny negative numerical garbage (NOT abs()).
        raw = self._udf_clamp(raw0)

        surface_band = float(config.get("udf_surface_band", 0.03))
        extract_level = float(config.get("udf_extract_level", 0.0))

        if extract_level > 0.0:
            # Extract the level set raw == extract_level.
            # This is useful when true UDF zero is sparse/noisy, but low-UDF bands show the shape.
            iso_band = float(config.get("udf_iso_band", surface_band))

            scalar = np.abs(raw - extract_level)
            near = scalar < iso_band

            _net = raw[raw < (0.5 * empty_val)]
            print(
                "[udf field iso]",
                "level=", extract_level,
                "iso_band=", iso_band,
                "near_voxels=", int(near.sum()),
                "net_voxels=", int(_net.size),
                "raw_min/mean/max=",
                float(_net.min()) if _net.size else float("nan"),
                float(_net.mean()) if _net.size else float("nan"),
                float(_net.max()) if _net.size else float("nan"),
                "neg_raw0=", int(np.sum(raw0 < 0.0)),
            )

            if not near.any():
                print("[dualmeshudf] no iso-near voxels -> empty")
                return trimesh.Trimesh(
                    vertices=np.zeros((0, 3)),
                    faces=np.zeros((0, 3), dtype=np.int64),
                    process=False,
                )

            edt = distance_transform_edt(~near).astype(np.float64) * step
            vol = np.where(near, scalar, iso_band + edt)
            fill = float(iso_band + edt.max()) if edt.size else 1.0

        else:
            # Original strict zero-UDF extraction
            near = raw < surface_band

            _net = raw[raw < (0.5 * empty_val)]
            print(
                "[udf field]",
                "band=", surface_band,
                "near_voxels=", int(near.sum()),
                "net_voxels=", int(_net.size),
                "min/mean/max=",
                float(_net.min()) if _net.size else float("nan"),
                float(_net.mean()) if _net.size else float("nan"),
                float(_net.max()) if _net.size else float("nan"),
                "neg_raw0=", int(np.sum(raw0 < 0.0)),
            )

            if not near.any():
                print("[dualmeshudf] no near-surface voxels -> empty")
                return trimesh.Trimesh(
                    vertices=np.zeros((0, 3)),
                    faces=np.zeros((0, 3), dtype=np.int64),
                    process=False,
                )

            edt = distance_transform_edt(~near).astype(np.float64) * step
            vol = np.where(near, raw, surface_band + edt)
            fill = float(surface_band + edt.max()) if edt.size else 1.0

        axes = (np.arange(N1), np.arange(N1), np.arange(N1))
        interp = RegularGridInterpolator(
            axes, vol, method='linear', bounds_error=False, fill_value=fill)

        def _idx(u):
            return (np.asarray(u, dtype=np.float64) + 1.0) * (N / 2.0)

        def udf_func(pts):
            d = np.maximum(interp(_idx(pts)), 0.0) / size
            return d.reshape(-1, 1).astype(np.float32)

        eps = 2.0 / max(N, 1)
        def udf_grad_func(pts):
            pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
            d = np.maximum(interp(_idx(pts)), 0.0) / size
            g = np.empty_like(pts)
            for a in range(3):
                pp = pts.copy(); pp[:, a] += eps
                pm = pts.copy(); pm[:, a] -= eps
                g[:, a] = (interp(_idx(pp)) - interp(_idx(pm))) / (2.0 * eps)
            nrm = np.linalg.norm(g, axis=1, keepdims=True); nrm[nrm == 0] = 1.0
            return d.reshape(-1, 1).astype(np.float32), (g / nrm).astype(np.float32)

        max_depth = int(config.get('udf_max_depth',
                                   max(1, int(round(math.log2(max(N, 2)))))))
        batch_size = int(config.get('udf_batch_size', 150000))
        reliable = float(config.get('udf_reliable_threshold', 0.01))
        if "udf_sample_threshold" in config:
            sample_threshold = float(config["udf_sample_threshold"])
        else:
            sample_threshold = min(reliable * 0.25, 0.005)

        print(
            f"[dualmeshudf] reso={N} max_depth={max_depth} batch={batch_size} "
            f"reliable={reliable} sample_threshold={sample_threshold} "
            f"field_range=[{float(vol.min()):.4g},{float(vol.max()):.4g}]"
        )

        self._ensure_udf_igl_patch()
        v, f = self._dualmeshudf_extract(udf_func, udf_grad_func,
                                         batch_size=batch_size,
                                         max_depth=max_depth, reliable=reliable,
                                         sampling_depth=int(config.get('udf_sampling_depth', 1)),
                                         sample_threshold = sample_threshold)
        v = np.asarray(v, dtype=np.float64); f = np.asarray(f, dtype=np.int64)
        print(f"[dualmeshudf] extracted V={len(v)} F={len(f)}")
        print(
            f"[dualmeshudf] reso={N} max_depth={max_depth} batch={batch_size} "
            f"reliable={reliable} sample_threshold={sample_threshold} "
            f"field_range=[{float(vol.min()):.4g},{float(vol.max()):.4g}]"
        )
        if len(v) == 0 or len(f) == 0:
            return trimesh.Trimesh(vertices=np.zeros((0, 3)),
                                   faces=np.zeros((0, 3), dtype=np.int64),
                                   process=False)
        p = v * size
        if gc.get('do_flip', True):
            p = p[:, [2, 1, 0]]
            f = f[:, [0, 2, 1]]
        return trimesh.Trimesh(vertices=p, faces=f, process=False)


    def _dualmeshudf_extract(self, udf_func, udf_grad_func, batch_size=150000,
                             max_depth=7, reliable=0.01, sampling_depth=1, sample_threshold=None,):
        """DualMeshUDF's extract_mesh octree loop, but with the reliable-UDF
        threshold exposed (stock extract_mesh hardcodes 0.002, too tight for a
        rasterized/low-res grid). No edit to the installed package needed."""
        import numpy as _np
        import igl as _igl
        from DualMeshUDF_core import Octree, triangulate_faces
        from DualMeshUDF.extract_mesh import query_udf, query_udf_and_grad

        if sample_threshold is None:
            sample_threshold = min(float(reliable) * 0.25, 0.005)
        sample_threshold = float(sample_threshold)

        octree = Octree(max_depth=max_depth,
                        min_corner=_np.array([[-1.], [-1.], [-1.]]),
                        max_corner=_np.array([[1.], [1.], [1.]]),
                        sampling_depth=int(sampling_depth))
        cur_depth = 0
        while cur_depth <= max_depth:
            centroids = octree.centroids_of_new_nodes().astype(_np.float32)
            cu, cg = query_udf_and_grad(udf_grad_func, centroids, batch_size)
            octree.adaptive_subdivide(cu, cg, reliable)
            cur_depth += 1
        gi, gc = octree.get_samples_of_new_nodes()
        gu, gg = query_udf_and_grad(udf_grad_func, gc.astype(_np.float32), batch_size)
        octree.set_new_grid_data(gi, gu, gg)
        idx, proj = octree.get_projections_for_checking_validity()
        pu = query_udf(udf_func, proj, batch_size)
        _pv = pu < reliable
        _puf = _np.asarray(pu).reshape(-1)
        print('[dmudf] n_proj=', int(_puf.shape[0]), 'n_valid=', int(_np.asarray(_pv).sum()),
              'reliable=', reliable, 'pu_pct[0,1,5,25,50]=',
              ([round(float(x), 5) for x in _np.percentile(_puf, [0, 1, 5, 25, 50])]
               if _puf.size else []))
        octree.set_grid_validity(idx, _pv)
        print(
            "[dmudf]",
            "projection_reliable=", reliable,
            "qef_sample_threshold=", sample_threshold,
        )
        octree.batch_solve(sample_threshold, 1.0, 1.0, 0.15, 0.08)
        octree.generate_mesh()
        print('[dmudf] after solve: mesh_v=', len(octree.mesh_v),
              'mesh_f=', len(octree.mesh_f))
        tri = triangulate_faces(octree.mesh_v, octree.mesh_f, octree.v_type, octree.mesh_v_dir)
        print('[dmudf] triangulated=', (len(tri) if tri is not None else 0))
        v, _, _, f = _igl.remove_duplicate_vertices(_np.array(octree.mesh_v), tri, 1e-7)
        v, f, _, _ = _igl.remove_unreferenced(v, f)
        print('[dmudf] after igl: v=', len(v), 'f=', len(f))
        return v, f

    def _ensure_udf_igl_patch(self):
        """Patch igl.remove_duplicate_vertices / remove_unreferenced so
        DualMeshUDF runs on newer libigl builds (which require float64 V +
        int64 F). Idempotent and shared with the grid path via the
        _udf_igl_patched flag, so whichever extractor runs first patches once.
        """
        import igl as _igl
        if getattr(_igl, "_udf_igl_patched", False):
            return
        _orig_dedup = getattr(_igl, "remove_duplicate_vertices", None)
        _orig_unref = getattr(_igl, "remove_unreferenced", None)

        def _np_dedup(V, F, eps):
            key = np.round(V / max(float(eps), 1e-30))
            _, SVI, SVJ = np.unique(key, axis=0, return_index=True, return_inverse=True)
            SVJ = SVJ.reshape(-1)
            SF = SVJ[np.asarray(F, dtype=np.int64)]
            return V[SVI], SVI.astype(np.int64), SVJ.astype(np.int64), SF.astype(np.int64)

        def _dedup(*a):
            V = np.ascontiguousarray(a[0], dtype=np.float64)
            if len(a) == 3:
                F = np.ascontiguousarray(a[1], dtype=np.int64)
                eps = float(a[2])
                if _orig_dedup is not None:
                    try:
                        return _orig_dedup(V, F, eps)
                    except TypeError:
                        pass
                return _np_dedup(V, F, eps)
            eps = float(a[1])
            if _orig_dedup is not None:
                try:
                    return _orig_dedup(V, eps)
                except TypeError:
                    pass
            key = np.round(V / max(eps, 1e-30))
            _, SVI, SVJ = np.unique(key, axis=0, return_index=True, return_inverse=True)
            return V[SVI], SVI.astype(np.int64), SVJ.reshape(-1).astype(np.int64)

        def _unref(*a):
            V = np.ascontiguousarray(a[0], dtype=np.float64)
            F = np.ascontiguousarray(a[1], dtype=np.int64)
            if _orig_unref is not None:
                try:
                    return _orig_unref(V, F)
                except TypeError:
                    pass
            used = np.unique(F.reshape(-1))
            remap = -np.ones(V.shape[0], dtype=np.int64)
            remap[used] = np.arange(used.shape[0], dtype=np.int64)
            return V[used], remap[F].astype(np.int64), used.astype(np.int64), remap

        _igl.remove_duplicate_vertices = _dedup
        _igl.remove_unreferenced = _unref
        _igl._udf_igl_patched = True

    def extract_udf_mesh_from_model(self, udf_point_fn, size, reso, config=None,
                                    domain_center=None, domain_half_extent=None):
        """Model-direct UDF extraction via a LOCAL dense grid + EDT.

        The continuous-oracle approach fails DualMeshUDF's validity/QEF: a
        point-cloud / distance-continuation field's zero-set is not a clean
        2-manifold, so the surface-projection marks ~every grid point valid and
        the dual solve degenerates (mesh_v=0). Instead we sample the model UDF
        onto a dense grid over the ACCESSORY BBOX (small domain -> fine spacing
        = the resolution win), rebuild the SAME thin-zero-set EDT field the grid
        path uses (proven to extract cleanly), and run the same octree loop.

        udf_point_fn(world_pts) -> RAW model UDF (>=0 world units; large
        `udf_fill_value` outside the support). Recon maps back to the bbox
        (center + u*half), so it lands in the oracle's world frame.
        """
        from scipy.interpolate import RegularGridInterpolator
        from scipy.ndimage import distance_transform_edt
        import math
        config = {} if config is None else config
        N = int(reso); N1 = N + 1
        if domain_center is None:
            domain_center = np.zeros(3, dtype=np.float64)
        else:
            domain_center = np.asarray(domain_center, dtype=np.float64).reshape(3)
        half = float(size if domain_half_extent is None else domain_half_extent)
        empty_val = float(config.get('udf_fill_value', 10.0))
        step = 2.0 * half / max(N, 1)   # world spacing of the local grid

        # sample the RAW model UDF on the local grid (axis i->x, j->y, k->z).
        gi, gj, gk = np.meshgrid(np.arange(N1), np.arange(N1), np.arange(N1),
                                 indexing='ij')
        pts = np.stack([
            domain_center[0] - half + gi.reshape(-1) * step,
            domain_center[1] - half + gj.reshape(-1) * step,
            domain_center[2] - half + gk.reshape(-1) * step], axis=1)
        qbatch = int(config.get('udf_grid_query_batch', 200000))
        raw = np.empty(pts.shape[0], dtype=np.float64)
        for s0 in range(0, pts.shape[0], qbatch):
            raw[s0:s0 + qbatch] = np.asarray(
                udf_point_fn(pts[s0:s0 + qbatch]), dtype=np.float64).reshape(-1)
        raw = np.maximum(raw, 0.0).reshape(N1, N1, N1)

        surface_band = float(config.get('udf_surface_band', 0.03))
        near = raw < surface_band
        _net = raw[raw < 0.5 * empty_val]
        print("[dualmeshudf-model grid] reso=", N, "half=", round(half, 4),
              "step=", round(step, 5), "near_voxels=", int(near.sum()),
              "net_voxels=", int(_net.size),
              "net_min/med=",
              ((round(float(_net.min()), 5), round(float(np.median(_net)), 5))
               if _net.size else None))
        if not near.any():
            print("[dualmeshudf-model] no near-surface voxels -> empty")
            return trimesh.Trimesh(vertices=np.zeros((0, 3)),
                                   faces=np.zeros((0, 3), dtype=np.int64),
                                   process=False)

        edt = distance_transform_edt(~near).astype(np.float64) * step
        #vol = np.where(near, raw, surface_band + edt)   # thin zero-set + EDT cont.
        #fill = float(surface_band + edt.max()) if edt.size else 1.0
        vol = edt
        fill = float(vol.max()) if vol.size else 1.0
        interp = RegularGridInterpolator(
            (np.arange(N1), np.arange(N1), np.arange(N1)), vol,
            method='linear', bounds_error=False, fill_value=fill)

        def _idx(u):
            return (np.asarray(u, dtype=np.float64) + 1.0) * (N / 2.0)

        def udf_func(p):
            return (np.maximum(interp(_idx(p)), 0.0) / half).reshape(-1, 1).astype(np.float32)

        eps = 2.0 / max(N, 1)
        def udf_grad_func(p):
            p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
            d = np.maximum(interp(_idx(p)), 0.0) / half
            g = np.empty_like(p)
            for a in range(3):
                pp = p.copy(); pp[:, a] += eps
                pm = p.copy(); pm[:, a] -= eps
                g[:, a] = (interp(_idx(pp)) - interp(_idx(pm))) / (2.0 * eps)
            nrm = np.linalg.norm(g, axis=1, keepdims=True); nrm[nrm == 0] = 1.0
            return d.reshape(-1, 1).astype(np.float32), (g / nrm).astype(np.float32)

        max_depth = int(config.get('udf_max_depth',
                                   max(1, int(round(math.log2(max(N, 2)))))))
        batch_size = int(config.get('udf_batch_size', 150000))
        reliable = float(config.get('udf_reliable_threshold', 0.01))
        print("[dualmeshudf-model] max_depth=", max_depth, "reliable=", reliable,
              "field_range=",
              (round(float(vol.min()), 4), round(float(vol.max()), 4)))

        self._ensure_udf_igl_patch()
        v, f = self._dualmeshudf_extract(udf_func, udf_grad_func,
                                         batch_size=batch_size,
                                         max_depth=max_depth, reliable=reliable,
                                         sampling_depth=int(config.get('udf_sampling_depth', 1)))
        v = np.asarray(v, dtype=np.float64); f = np.asarray(f, dtype=np.int64)
        print("[dualmeshudf-model] extracted V=", len(v), "F=", len(f))
        if len(v) == 0 or len(f) == 0:
            return trimesh.Trimesh(vertices=np.zeros((0, 3)),
                                   faces=np.zeros((0, 3), dtype=np.int64),
                                   process=False)
        p = domain_center[None, :] + v * half   # cube [-1,1] -> world bbox
        print("[dualmeshudf-model] recon world bbox min=", p.min(0),
              "max=", p.max(0))
        return trimesh.Trimesh(vertices=p, faces=f, process=False)

    @torch.no_grad()
    def action_part_adapt(self, arg):
        output_folder = arg['output_folder']
        exp_name = arg['exp_name']
        mc_grid = arg['mc_grid']
        shape_name = arg['shape']

        # The adaptation YAML may be either:
        #
        #   1. Legacy list-only format:
        #        - target_key: ...
        #
        #   2. Top-level mapping format:
        #        surface_extraction: ...
        #        adaptations: [...]
        #
        # Accept the singular ``adaptation`` key too because some existing
        # experiment files use that spelling.
        raw_config = utils.load_yaml_file(arg['adapt_file'])

        extraction_config = dict(arg)
        if isinstance(raw_config, list):
            adaptation_items = raw_config
        elif isinstance(raw_config, dict):
            if 'surface_extraction' in raw_config:
                extraction_config['surface_extraction'] = raw_config[
                    'surface_extraction'
                ]
            if 'mesh_extractor' in raw_config:
                extraction_config['mesh_extractor'] = raw_config[
                    'mesh_extractor'
                ]

            # Also allow UDF extraction/combination settings at YAML top level.
            for config_key, config_value in raw_config.items():
                key_text = str(config_key)
                if (
                    key_text.startswith('udf_')
                    or key_text in {
                        'combine_adaptations',
                        'surface_extraction',
                        'mesh_extractor',
                    }
                ):
                    extraction_config[config_key] = config_value

            adaptation_items = raw_config.get(
                'adaptations',
                raw_config.get('adaptation', None),
            )
            if adaptation_items is None:
                raise ValueError(
                    "Adaptation YAML mapping must contain 'adaptations:' "
                    "(preferred) or 'adaptation:'."
                )
        else:
            raise TypeError(
                "Adaptation YAML must load as a list or mapping, got "
                f"{type(raw_config).__name__}."
            )

        if not isinstance(adaptation_items, list):
            raise TypeError(
                "'adaptations'/'adaptation' must be a YAML list, got "
                f"{type(adaptation_items).__name__}."
            )

        # Expand every YAML entry into one concrete run per target key.
        # Existing files with a scalar ``target_key`` remain unchanged.
        # A shared configuration can now use:
        #
        #   target_keys:
        #     - shape|curve_a
        #     - shape|curve_b
        #
        # Duplicate target keys in separate YAML entries are intentionally
        # preserved: they are separate adaptations and must both execute.
        expanded_adaptation_items = []
        for yaml_index, item in enumerate(adaptation_items):
            if not isinstance(item, dict):
                raise TypeError(
                    f"adaptation entry {yaml_index} must be a mapping, got "
                    f"{type(item).__name__}."
                )

            if not bool(item.get('enabled', True)):
                print(f"[part_adapt] skipping disabled YAML entry {yaml_index}")
                continue

            target_spec = item.get('target_keys', item.get('target_key', None))
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

            if len(target_keys) == 0:
                raise ValueError(
                    f"adaptation entry {yaml_index} has an empty target_keys list."
                )

            for target_index, target_key in enumerate(target_keys):
                expanded = dict(item)
                expanded.pop('target_keys', None)
                expanded['target_key'] = str(target_key)
                expanded['_yaml_index'] = int(yaml_index)
                expanded['_target_index'] = int(target_index)
                expanded_adaptation_items.append(expanded)

        adaptation_items = expanded_adaptation_items
        if len(adaptation_items) == 0:
            raise RuntimeError("No enabled adaptation entries were found.")

        extraction_spec = extraction_config.get(
            'surface_extraction',
            extraction_config.get('mesh_extractor', 'dualmeshudf'),
        )
        if isinstance(extraction_spec, dict):
            requested_extractor = extraction_spec.get(
                'method',
                extraction_spec.get('backend', 'dualmeshudf'),
            )
        else:
            requested_extractor = extraction_spec
        requested_extractor = self._normalize_surface_extraction_method(
            requested_extractor
        )
        if requested_extractor not in {'dualmeshudf', 'dualmeshudf_model'}:
            raise ValueError(
                f"AgentUDF only supports 'dualmeshudf' or 'dualmeshudf_model', "
                f"got {requested_extractor!r}. Use AgentSDF for marching cubes/RFTA."
            )
        combine_adaptations = bool(
            extraction_config.get("combine_adaptations", False)
        )
        print(
            "[part_adapt] adaptations=",
            len(adaptation_items),
            "surface_extraction=",
            requested_extractor,
            "combine_adaptations=",
            combine_adaptations,
        )

        data_root = arg['data_root']
        handle = self.load_shape_handle(data_root, shape_name, 'avatar')

        available_target_keys = {
            self.encode_key(shape_name, curve.name): curve
            for curve in handle.curves
        }
        missing_target_keys = sorted({
            item['target_key']
            for item in adaptation_items
            if item['target_key'] not in available_target_keys
        })
        if missing_target_keys:
            raise KeyError(
                "The following adaptation target_key values were not found: "
                f"{missing_target_keys}. Available targets: "
                f"{sorted(available_target_keys.keys())}"
            )

        out_name = f'{shape_name}_{exp_name}'
        os.makedirs(output_folder, exist_ok=True)

        batch_size = 64**3
        #mc_grid.clear_grid(val=10.0)
        mc_grid.clear_grid()

        adapted_support_cache = {}
        all_acc_grids = []
        blend_groups = {}
        for item_index, item in enumerate(adaptation_items):
            target_key = item['target_key']
            accessory_key = item['accessory_key']
            mode = item.get('mode', 'direct')
            run_name = str(
                item.get(
                    'name',
                    item.get('run_name', f'adaptation_{item_index:02d}'),
                )
            )
            print(
                f"[part_adapt {item_index + 1}/{len(adaptation_items)}]",
                f"name={run_name}",
                f"target={target_key}",
                f"accessory={accessory_key}",
                f"yaml_entry={item.get('_yaml_index')}",
                f"target_index={item.get('_target_index')}",
            )

            matched_target = False
            for curve in handle.curves:
                key = self.encode_key(shape_name, curve.name)
                if key != target_key:
                    continue
                matched_target = True
                print(
                    f"[part_adapt {item_index + 1}/{len(adaptation_items)}] "
                    f"matched {key}"
                )

                curve_grid = utils.create_grid_like(mc_grid)
                #curve_grid.clear_grid(val=10.0)
                curve_grid.clear_grid()

                adapt_arg = {
                    'mode': mode,
                    'avatar_curve_handle': curve,
                    'device': self.device,
                    'infer_scale': 2.0,
                    'avatar_curve_idx': self.feat_dict[key],
                    'accessory_curve_idx': self.feat_dict[accessory_key],
                }
                adapt_arg.update(item)

                # UDF is unsigned. These older knobs are signed-SDF/detail-only;
                # ignore them here so an SDF YAML can be reused for a clean UDF test.
                if bool(adapt_arg.get("auto_avatar_snug_field", False)):
                    print("[udf] ignoring auto_avatar_snug_field (SDF-only)")
                    adapt_arg["auto_avatar_snug_field"] = False
                if bool(adapt_arg.get("use_tiled_detail", False)):
                    print("[udf] ignoring use_tiled_detail (signed-detail/SDF-only)")
                    adapt_arg["use_tiled_detail"] = False

                if mode == 'direct':
                    accessory_curve_handle = self.curve_from_key(accessory_key)
                    root_scale = self.compute_root_assembly_scale(
                        avatar_curve_handle=curve,
                        accessory_curve_handle=accessory_curve_handle,
                        src_0=float(adapt_arg["src_0"]),
                        src_1=float(adapt_arg["src_1"]),
                        tgt_0=float(adapt_arg["tgt_0"]),
                        tgt_1=float(adapt_arg["tgt_1"]),
                    )
                    adapt_arg['accessory_curve_handle'] = accessory_curve_handle

                    accessory_data, avatar_data, kidx, inside = curve.filter_grid_adapt(curve_grid, adapt_arg)

                    cache_key = item.get("cache_as", accessory_key)

                    # IMPORTANT:
                    # For direct mode, store the actual support used by inference.
                    acc_coords = accessory_data["coords"]
                    #acc_intpl = accessory_curve_handle.core.interpolate(acc_coords)
                    #adapted_support_cache[cache_key] = {
                    #    "coords": acc_coords.copy(),
                    #    "points": accessory_data["runtime_points"].copy(),
                    #    "frame": accessory_data["runtime_frame"].copy(),
                    #    "radius": accessory_data["radius"].copy(),
                    #    "x_radius": accessory_data["x_radius"].copy(),
                    #    "assembly_scale": root_scale,
                    #    #"x_radius": accessory_curve_handle.core.calc_x_radius(acc_coords).copy(),
                    #}
                    #adapted_support_cache[cache_key]["assembly_scale"] = root_scale

                else:
                    raise ValueError(f"Unknown adapt mode: {mode}")

                #acc_vals, acc_vals_base = self._inference_vals(
                #    accessory_data, accessory_key, batch_size=batch_size
                #)
                # ------------------------------------------------------------
                # Optional two-pass implicit snug-wrap correction.
                #
                # Pass 1:
                #   normal adapted accessory + avatar SDF
                #   measure local gap on near-accessory-surface points
                #
                # Pass 2:
                #   rerun filter_grid_adapt with avatar_snug_scale_field
                # ------------------------------------------------------------
                if bool(adapt_arg.get("auto_avatar_snug_field", False)) and mode == "direct":
                    if not adapt_arg.get("wrap_radius", False):
                        print(
                            "[snug_field] auto_avatar_snug_field requested, "
                            "but wrap_radius is false. Skipping snug field."
                        )
                    else:
                        # First-pass accessory SDF
                        acc_out0 = self._inference_full_vals(
                            accessory_data,
                            accessory_key,
                            batch_size=batch_size,
                        )
                        # Debug: first-pass base BEFORE snug rerun
#                        pre_snug_base_grid = utils.create_grid_like(mc_grid)
#                        pre_snug_base_grid.clear_grid()
#                        pre_snug_base_grid.update_grid(
#                            acc_out0["dist_base"],
#                            kidx,
#                            mark=True,
#                            mode="overwrite",
#                        )
#
#                        mesh_pre_snug_base = pre_snug_base_grid.extract_mesh()
#                        if len(mesh_pre_snug_base.faces) > 0:
#                            parts = mesh_pre_snug_base.split(only_watertight=False)
#                            if len(parts) > 0:
#                                mesh_pre_snug_base = max(parts, key=lambda m: len(m.faces))
#                            mesh_pre_snug_base.export(
#                                op.join(
#                                    output_folder,
#                                    f"{cc}_{mode}_PRE_SNUG_base_{accessory_key.replace('|','_')}.ply"
#                                )
#                            )

                        # First-pass avatar SDF on same samples/order
                        avatar_out0 = self._inference_full_vals(
                            avatar_data,
                            key,
                            batch_size=batch_size,
                        )

                        snug_field = self.build_avatar_snug_scale_field(
                            acc_sdf=acc_out0['dist_base'],
                            avatar_sdf=avatar_out0['dist'],
                            avatar_coords=avatar_data["coords"],
                            avatar_theta=avatar_data["angles"],
                            n_s=int(adapt_arg.get("snug_field_n_s", 48)),
                            n_theta=int(adapt_arg.get("snug_field_n_theta", 64)),
                            surface_band=float(adapt_arg.get("snug_surface_band", 0.015)),
                            target_gap=float(adapt_arg.get("snug_target_gap", 0.001)),
                            gain=float(adapt_arg.get("snug_gain", 12.0)),
                            min_scale=float(adapt_arg.get("snug_min_scale", 0.85)),
                            max_scale=float(adapt_arg.get("snug_max_scale", 1.15)),
                            min_count=int(adapt_arg.get("snug_min_count", 5)),
                            smooth_s=float(adapt_arg.get("snug_smooth_s", 2.0)),
                            smooth_theta=float(adapt_arg.get("snug_smooth_theta", 1.0)),
                            delta_in_max=adapt_arg.get("snug_delta_in_max", None),
                            delta_out_max=adapt_arg.get("snug_delta_out_max", None),
                            debug=bool(adapt_arg.get("snug_debug", True)),
                        )

                        if snug_field is not None:
                            # NOTE: previous code had a typo here
                            # ("avatar_snu g_scale_field" with a stray space)
                            # which meant the snug field was never read by
                            # PWLA_curve_handle. Fixed.
                            adapt_arg["avatar_snug_scale_field"] = snug_field

                            # Rerun direct adaptation with the correction field.
                            curve_grid = utils.create_grid_like(mc_grid)
                            curve_grid.clear_grid()

                            accessory_data, avatar_data, kidx, inside = curve.filter_grid_adapt(
                                curve_grid,
                                adapt_arg,
                            )

                            # If this direct accessory is cached for dependent children,
                            # refresh the cached support to match the corrected pass.
#                            cache_key = item.get("cache_as", accessory_key)
#                            if cache_key in adapted_support_cache:
#                                acc_coords = accessory_data["coords"]
#                                adapted_support_cache[cache_key] = {
#                                    "coords": acc_coords.copy(),
#                                    "points": accessory_data["runtime_points"].copy(),
#                                    "frame": accessory_data["runtime_frame"].copy(),
#                                    "radius": accessory_data["radius"].copy(),
#                                    "x_radius": accessory_data["x_radius"].copy(),
#                                    "assembly_scale": root_scale,
#                                }


                use_tiled_detail = bool(adapt_arg.get("use_tiled_detail", False))

                if use_tiled_detail:
                    accessory_data = self.add_tiled_detail_coords_for_adapt(
                        accessory_data,
                        adapt_arg,
                    )
                    acc_out = self._inference_full_vals(
                        accessory_data,
                        accessory_key,
                        batch_size=batch_size,
                        transform="stretch",
                    )
                else:
                    acc_out = self._inference_full_vals(
                        accessory_data,
                        accessory_key,
                        batch_size=batch_size,
                    )
                udf_dbg = np.maximum(
                    np.asarray(acc_out["dist"], dtype=np.float64).reshape(-1),
                    0.0,
                )

                acc_vals = acc_out["dist"]

                # --- true-UDF finalize: clamp only; NO signed base/detail/carve ---
                udf_vals = self._udf_clamp(acc_vals)
                # ------------------------------------------------------------
                # UDF zero-bias calibration.
                # For tight scales, the learned UDF may not place the usable surface exactly
                # at raw==0 everywhere. Instead of extracting a global positive iso-level,
                # subtract a slow local low-quantile bias along the accessory coordinate.
                # This keeps high-frequency detail in the residual field.
                # ------------------------------------------------------------
                if bool(adapt_arg.get("udf_local_zero_calibration", False)):
                    coords_for_calib = np.asarray(accessory_data["coords"], dtype=np.float64).reshape(-1)
                    vals_for_calib = np.asarray(udf_vals, dtype=np.float64).reshape(-1)

                    n_bins = int(adapt_arg.get("udf_zero_calib_bins", 64))
                    q = float(adapt_arg.get("udf_zero_calib_quantile", 1.0))
                    max_shift = float(adapt_arg.get("udf_zero_calib_max_shift", 0.04))
                    smooth_bins = int(adapt_arg.get("udf_zero_calib_smooth_bins", 2))
                    min_count = int(adapt_arg.get("udf_zero_calib_min_count", 16))

                    bins = np.linspace(
                        float(np.min(coords_for_calib)),
                        float(np.max(coords_for_calib)) + 1e-9,
                        n_bins + 1,
                    )
                    bid = np.searchsorted(bins, coords_for_calib, side="right") - 1
                    bid = np.clip(bid, 0, n_bins - 1)

                    bias = np.full(n_bins, np.nan, dtype=np.float64)
                    counts = np.zeros(n_bins, dtype=np.int64)

                    finite = np.isfinite(vals_for_calib)
                    for bi in range(n_bins):
                        m = finite & (bid == bi)
                        counts[bi] = int(np.sum(m))
                        if counts[bi] >= min_count:
                            bias[bi] = np.percentile(vals_for_calib[m], q)

                    # Fill missing bins by nearest valid value
                    valid = np.isfinite(bias)
                    if np.any(valid):
                        x = np.arange(n_bins)
                        bias = np.interp(x, x[valid], bias[valid])
                    else:
                        bias[:] = 0.0

                    # Smooth the slow bias; keep details in udf_vals itself.
                    if smooth_bins > 0:
                        from scipy.ndimage import gaussian_filter1d
                        bias = gaussian_filter1d(
                            bias,
                            sigma=float(smooth_bins),
                            mode="nearest",
                        )

                    bias = np.clip(bias, 0.0, max_shift)
                    local_bias = bias[bid]

                    udf_vals = np.maximum(vals_for_calib - local_bias, 0.0)

                    print(
                        "[udf zero calib]",
                        "enabled",
                        "q=", q,
                        "bias_min/med/max=",
                        round(float(np.min(bias)), 5),
                        round(float(np.median(bias)), 5),
                        round(float(np.max(bias)), 5),
                        "counts_min/med/max=",
                        int(np.min(counts)),
                        int(np.median(counts)),
                        int(np.max(counts)),
                    )

                # PREPASS DIAGNOSTIC (free: acc_vals is the network UDF on the
                # accepted support points, aligned with kidx / avatar samples).
                # Answers "does the trained UDF actually reach ~0 on the support?"
                _fin = udf_vals[np.isfinite(udf_vals)]
                if _fin.size:
                    _pc = np.percentile(_fin, [0, 1, 5, 25, 50, 95, 100]).tolist()
                    print("[udf prepass]", "n=", int(udf_vals.size),
                          "pct[0,1,5,25,50,95,100]=", [round(float(x), 5) for x in _pc],
                          "near003=", int((udf_vals < 0.03).sum()),
                          "near005=", int((udf_vals < 0.05).sum()),
                          "near010=", int((udf_vals < 0.10).sum()))
                    # ------------------------------------------------------------
                    if bool(extraction_config.get("udf_export_near_points", True)):
                        domain_pts_all = np.asarray(mc_grid.idx2pts(kidx), dtype=np.float64)

                        debug_dir = op.join(output_folder, "udf_debug")
                        os.makedirs(debug_dir, exist_ok=True)

                        safe_acc = accessory_key.replace("|", "_").replace("/", "_")

                        for band_dbg in [0.03, 0.05, 0.10]:
                            near_dbg = udf_dbg < float(band_dbg)

                            print(
                                "[udf near export]",
                                "band=", band_dbg,
                                "near=", int(np.sum(near_dbg)),
                                "/", int(udf_dbg.shape[0]),
                            )

                            if np.any(near_dbg):
                                trimesh.PointCloud(
                                    domain_pts_all[near_dbg]
                                ).export(
                                    op.join(
                                        debug_dir,
                                        f"{item_index}_{safe_acc}_near_udf_{band_dbg:.3f}.ply",
                                    )
                                )
                else:
                    print("[udf prepass] no finite udf values on support")

                # ------------------------------------------------------------
                # Per-adaptation extraction config.
                # Needed for legacy list YAML where udf_* keys live inside
                # each adaptation item instead of at YAML top level.
                # ------------------------------------------------------------
                extract_cfg = dict(extraction_config)

                for config_key, config_value in item.items():
                    key_text = str(config_key)
                    if (
                        key_text.startswith("udf_")
                        or key_text in {
                            "surface_extraction",
                            "mesh_extractor",
                            "combine_adaptations",
                        }
                    ):
                        extract_cfg[config_key] = config_value

                print(
                    "[udf extract cfg]",
                    "surface_band=", extract_cfg.get("udf_surface_band"),
                    "reliable=", extract_cfg.get("udf_reliable_threshold"),
                    "sample=", extract_cfg.get("udf_sample_threshold"),
                    "extract_level=", extract_cfg.get("udf_extract_level"),
                    "iso_band=", extract_cfg.get("udf_iso_band"),
                )

                if requested_extractor == 'dualmeshudf_model':
                    _oracle_arg = dict(adapt_arg)
                    _oracle_arg['adapt_debug_counts'] = False
                    _oracle_arg['debug_interval_projection'] = False
                    _fill = float(extract_cfg.get('udf_fill_value', 10.0))

                    # RAW model UDF sampler (no continuation here -- the local
                    # grid's EDT rebuilds a clean thin-zero-set distance field).
                    def _raw_udf(world_pts, _self=self, _c=curve, _aa=_oracle_arg,
                                 _k=accessory_key, _bs=int(batch_size), _fill=_fill):
                        world_pts = np.asarray(
                            world_pts, dtype=np.float64).reshape(-1, 3)
                        out = np.full(world_pts.shape[0], _fill, dtype=np.float64)
                        try:
                            acc_d, _av, inside = _c.core.localize_samples_adapt(
                                world_pts, _aa)
                            if acc_d is not None:
                                idx = np.asarray(inside, dtype=np.int64).reshape(-1)
                                if idx.size:
                                    acc_o = _self._inference_full_vals(
                                        acc_d, _k, batch_size=_bs)
                                    d = _self._udf_clamp(acc_o["dist"]).reshape(-1)
                                    n = min(idx.size, d.size)
                                    out[idx[:n]] = d[:n]
                        except Exception:
                            pass
                        return out

                    # bbox of the world points localize operates on (oracle
                    # frame); optionally tighten to the near-UDF shell.
                    if isinstance(avatar_data, dict) and avatar_data.get('samples') is not None:
                        _sw = np.asarray(avatar_data['samples'], dtype=np.float64).reshape(-1, 3)
                    else:
                        _sw = np.asarray(mc_grid.idx2pts(kidx), dtype=np.float64).reshape(-1, 3)
                    _aligned = (_sw.shape[0] == udf_vals.shape[0])
                    _nb = float(extract_cfg.get('udf_domain_near_band_world', 0.0))
                    _minnear = int(extract_cfg.get('udf_domain_min_near', 64))
                    _used = "full-support"; _domain_pts = _sw
                    if _nb > 0.0 and _aligned:
                        _mask = udf_vals < _nb
                        if int(_mask.sum()) >= _minnear:
                            _domain_pts = _sw[_mask]; _used = "near-UDF(<%.3g)" % _nb
                    if _domain_pts.shape[0] == 0:
                        _center = np.zeros(3, dtype=np.float64); _half = float(mc_grid.size)
                    else:
                        _bmin = _domain_pts.min(axis=0); _bmax = _domain_pts.max(axis=0)
                        _center = 0.5 * (_bmin + _bmax)
                        _pad = float(extract_cfg.get('udf_domain_padding', 0.2))
                        _half = 0.5 * float(np.max(_bmax - _bmin)) * (1.0 + _pad)
                    print("[dualmeshudf-model domain]", "using", _used,
                          "n=", int(_domain_pts.shape[0]),
                          "center=", _center, "half_extent=", round(_half, 5))
                    mesh_acc = self.extract_udf_mesh_from_model(
                        _raw_udf,
                        float(mc_grid.size),
                        int(mc_grid.reso),
                        extract_cfg,
                        domain_center=_center,
                        domain_half_extent=_half,
                    )
                else:
                    acc_grid = utils.create_grid_like(mc_grid)
                    acc_grid.clear_grid()
                    acc_grid.update_grid(udf_vals, kidx, mark=True, mode="overwrite")
                    mesh_acc = self.extract_udf_mesh_from_grid(
                        acc_grid, extract_cfg)

#                if len(mesh_acc.faces) > 0:
#                    parts = mesh_acc.split(only_watertight=False)
#                    if len(parts) > 0:
#                        mesh_acc = max(parts, key=lambda m: len(m.faces))
#                    individual_mesh_file = op.join(
#                        output_folder,
#                        f"{item_index}_{mode}_{accessory_key.replace('|','_')}.ply",
#                    )
#                    mesh_acc.export(individual_mesh_file)
#                    print(
#                        f"[udf part_adapt {item_index + 1}/{len(adaptation_items)}] "
#                        f"saved {individual_mesh_file} "
#                        f"V={len(mesh_acc.vertices)} F={len(mesh_acc.faces)}"
#                    )
                if len(mesh_acc.faces) > 0:
                    safe_acc = accessory_key.replace("|", "_").replace("/", "_")

                    full_mesh_file = op.join(
                        output_folder,
                        f"{item_index}_{mode}_{safe_acc}_FULL.ply",
                    )
                    mesh_acc.export(full_mesh_file)

                    parts = mesh_acc.split(only_watertight=False)
                    face_counts = sorted([len(p.faces) for p in parts], reverse=True)

                    print(
                        "[udf components]",
                        "n=", len(parts),
                        "faces_top30=", face_counts[:30],
                    )
                    print("[udf full export]", full_mesh_file)

                    # For now, DO NOT keep only largest component.
                    individual_mesh_file = op.join(
                        output_folder,
                        f"{item_index}_{mode}_{safe_acc}.ply",
                    )
                    mesh_acc.export(individual_mesh_file)
                else:
                    print(
                        f"[udf part_adapt {item_index + 1}/{len(adaptation_items)}] "
                        f"EMPTY mesh for {accessory_key}"
                    )

