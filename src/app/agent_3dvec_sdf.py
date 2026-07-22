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

class AgentSDF(AgentBase):
    """Signed-distance agent: marching cubes, RFTA, sign conversion,
    boolean carve / snug / base+signed-detail recomposition."""

    def __init__(self):
        # TEMPORARY compatibility for the currently trained checkpoint.
        #
        # That checkpoint was trained with Trimesh's convention:
        #     positive = inside
        #     negative = outside
        #
        # The rest of this inference/adaptation pipeline expects:
        #     negative = inside
        #     positive = outside
        #
        # Keep this True only for the current checkpoint. Set it to False
        # after regenerating the training SDFs with the corrected sign and
        # retraining the model.
        self.invert_trained_sdf_sign = False #True

    def _convert_sdf_pair_sign(self, vals, vals_base):
        """Convert normal inference outputs to the pipeline SDF convention."""
        if not self.invert_trained_sdf_sign:
            return vals, vals_base

        return -np.asarray(vals), -np.asarray(vals_base)

    def _convert_full_output_sign(self, out):
        """
        Convert inference_full outputs to the pipeline SDF convention.

        sdf_detail is also sign-dependent because it is a signed residual:

            sdf_detail = sdf - sdf_base

        Negating both sdf and sdf_base therefore also negates sdf_detail.
        """
        if not self.invert_trained_sdf_sign:
            return out

        result = dict(out)
        sign_dependent_keys = {
            'dist',
            'dist_base',
            'dist_detail',
            'dist_res',
            'residual_sdf',
        }

        for key in sign_dependent_keys:
            value = result.get(key, None)
            if value is not None:
                result[key] = -np.asarray(value)

        return result

    @staticmethod
    def _subsample_rfta_rows(sdf_values, max_samples, near_surface_fraction, rng_seed):
        """
        Select SDF rows for Reach for the Arcs.

        Most rows are chosen by smallest |SDF| so thin/high-frequency surface
        features survive. The remaining budget is sampled from the full active
        support, which still gives the method larger-radius inside/outside
        spheres. Selection is deterministic for a fixed seed.
        """
        sdf_values = np.asarray(sdf_values).reshape(-1)
        n = int(sdf_values.shape[0])

        if max_samples is None or int(max_samples) <= 0 or n <= int(max_samples):
            return np.arange(n, dtype=np.int64)

        max_samples = max(2, min(int(max_samples), n))
        near_surface_fraction = float(np.clip(near_surface_fraction, 0.0, 1.0))
        near_count = int(round(max_samples * near_surface_fraction))
        near_count = min(max(2, near_count), max_samples)

        abs_sdf = np.abs(sdf_values)
        if near_count == n:
            near_rows = np.arange(n, dtype=np.int64)
        else:
            near_rows = np.argpartition(abs_sdf, near_count - 1)[:near_count]

        selected = np.zeros(n, dtype=bool)
        selected[near_rows] = True

        remaining_count = max_samples - int(np.sum(selected))
        if remaining_count > 0:
            remaining_rows = np.flatnonzero(~selected)
            rng = np.random.default_rng(int(rng_seed))
            random_rows = rng.choice(
                remaining_rows,
                size=min(remaining_count, remaining_rows.shape[0]),
                replace=False,
            )
            selected[random_rows] = True

        rows = np.flatnonzero(selected)

        # Reach for the Arcs should see both sides of the zero set. If the
        # near-surface/random selection accidentally omitted one sign, replace
        # a farthest selected row with the closest available row of that sign.
        for want_positive in (False, True):
            sign_all = sdf_values > 0.0 if want_positive else sdf_values < 0.0
            sign_selected = sign_all[rows]
            if np.any(sign_all) and not np.any(sign_selected):
                missing_candidates = np.flatnonzero(sign_all)
                missing_row = missing_candidates[
                    np.argmin(abs_sdf[missing_candidates])
                ]
                replace_pos = int(np.argmax(abs_sdf[rows]))
                rows[replace_pos] = missing_row

        return np.unique(rows).astype(np.int64, copy=False)

    def _extract_mesh_reach_for_the_arcs(self, sdf_grid, config, options):
        """Extract a triangle mesh from active grid SDF samples using RFTA."""
        try:
            from gpytoolbox import reach_for_the_arcs
        except ImportError as exc:
            raise ImportError(
                "Reach for the Arcs requires gpytoolbox. Install it with "
                "`uv add gpytoolbox` (or `python -m pip install gpytoolbox`)."
            ) from exc

        values_full = np.asarray(sdf_grid.val_grid).reshape(-1)

        if hasattr(sdf_grid, 'empty_marks'):
            empty_marks = np.asarray(sdf_grid.empty_marks).reshape(-1)
            if empty_marks.shape[0] != values_full.shape[0]:
                raise ValueError(
                    "grid empty_marks and val_grid have different sizes: "
                    f"{empty_marks.shape[0]} vs {values_full.shape[0]}"
                )
            active = ~empty_marks
        else:
            active = np.ones(values_full.shape[0], dtype=bool)

        active &= np.isfinite(values_full)
        grid_rows = np.flatnonzero(active)

        min_samples = int(options.get('min_samples', 128))
        if grid_rows.shape[0] < min_samples:
            raise RuntimeError(
                "Too few active finite SDF samples for Reach for the Arcs: "
                f"{grid_rows.shape[0]} < {min_samples}"
            )

        level = float(options.get('level', 0.0))
        sdf_scale = float(options.get('sdf_scale', 1.0))
        if not np.isfinite(sdf_scale) or sdf_scale <= 0.0:
            raise ValueError(
                f"rfta sdf_scale must be finite and positive, got {sdf_scale}"
            )

        # Reach for the Arcs interprets |S| as a geometric sphere radius, so
        # S must use the same spatial units as idx2pts(). Use sdf_scale when
        # the network predicts normalized rather than world-unit distances.
        sdf_values = (
            values_full[grid_rows].astype(np.float64, copy=False) - level
        ) * sdf_scale

        # Optional metric-band filtering for RFTA. This is important for this
        # pipeline because support clamping may write artificial outside values
        # such as +1.0. Marching Cubes can treat those as generic outside
        # markers, but RFTA would interpret |1.0| as a real sphere radius.
        max_abs_sdf = options.get('max_abs_sdf', np.inf)
        if max_abs_sdf is None:
            max_abs_sdf = np.inf
        max_abs_sdf = float(max_abs_sdf)
        if np.isfinite(max_abs_sdf):
            if max_abs_sdf <= 0.0:
                raise ValueError(
                    f"rfta max_abs_sdf must be positive, got {max_abs_sdf}"
                )
            metric_keep = np.abs(sdf_values) <= max_abs_sdf
            removed_metric = int(np.sum(~metric_keep))
            grid_rows = grid_rows[metric_keep]
            sdf_values = sdf_values[metric_keep]
        else:
            removed_metric = 0

        if grid_rows.shape[0] < min_samples:
            raise RuntimeError(
                "Too few RFTA samples remain after max_abs_sdf filtering: "
                f"{grid_rows.shape[0]} < {min_samples}; "
                f"max_abs_sdf={max_abs_sdf}"
            )

        has_negative = bool(np.any(sdf_values < 0.0))
        has_positive = bool(np.any(sdf_values > 0.0))
        if not (has_negative and has_positive):
            raise RuntimeError(
                "Reach for the Arcs needs active SDF samples on both sides of "
                f"the extraction level {level}. "
                f"negative={has_negative}, positive={has_positive}"
            )

        rng_seed = int(options.get('rng_seed', 3452))
        sample_rows = self._subsample_rfta_rows(
            sdf_values=sdf_values,
            max_samples=options.get('max_samples', 150000),
            near_surface_fraction=options.get('near_surface_fraction', 0.75),
            rng_seed=rng_seed,
        )

        selected_grid_rows = grid_rows[sample_rows]
        selected_sdf = sdf_values[sample_rows]
        sample_points = np.asarray(
            sdf_grid.idx2pts(selected_grid_rows),
            dtype=np.float64,
        )

        if sample_points.ndim != 2 or sample_points.shape[1] != 3:
            raise ValueError(
                "idx2pts must return an (N, 3) array for 3D extraction, got "
                f"{sample_points.shape}"
            )
        if sample_points.shape[0] != selected_sdf.shape[0]:
            raise ValueError(
                "idx2pts and selected SDF lengths differ: "
                f"{sample_points.shape[0]} vs {selected_sdf.shape[0]}"
            )

        clamp_value = options.get('clamp_value', np.inf)
        if clamp_value is None:
            clamp_value = np.inf
        clamp_value = float(clamp_value)
        if np.isfinite(clamp_value) and clamp_value <= 0.0:
            raise ValueError(
                f"rfta clamp_value must be positive, got {clamp_value}"
            )

        rasterization_resolution = options.get('rasterization_resolution', None)
        if rasterization_resolution is not None:
            rasterization_resolution = int(rasterization_resolution)

        n_local_searches = options.get('n_local_searches', None)
        if n_local_searches is not None:
            n_local_searches = int(n_local_searches)

        verbose = bool(options.get('verbose', True))
        if verbose:
            bbox_extent = np.ptp(sample_points, axis=0)
            abs_selected = np.abs(selected_sdf)
            pct = np.percentile(abs_selected, [0, 25, 50, 75, 90, 95, 99, 100])
            print(
                "[surface_extraction:rfta]",
                f"active_after_filter={grid_rows.shape[0]}",
                f"removed_by_max_abs_sdf={removed_metric}",
                f"used={selected_grid_rows.shape[0]}",
                f"negative={int(np.sum(selected_sdf < 0.0))}",
                f"positive={int(np.sum(selected_sdf > 0.0))}",
                f"sdf=[{selected_sdf.min():.6g}, {selected_sdf.max():.6g}]",
                f"max_abs_sdf={max_abs_sdf}",
                f"level={level:.6g}",
                f"sdf_scale={sdf_scale:.6g}",
                f"clamp={clamp_value}",
            )
            print(
                "[surface_extraction:rfta]",
                "bbox_extent=", bbox_extent.tolist(),
                "|sdf| percentiles[0,25,50,75,90,95,99,100]=",
                pct.tolist(),
            )

        result = reach_for_the_arcs(
            sample_points,
            selected_sdf,
            rng_seed=rng_seed,
            return_point_cloud=False,
            fine_tune_iters=int(options.get('fine_tune_iters', 3)),
            batch_size=int(options.get('batch_size', 10000)),
            num_rasterization_spheres=int(
                options.get('num_rasterization_spheres', 0)
            ),
            screening_weight=float(options.get('screening_weight', 10.0)),
            rasterization_resolution=rasterization_resolution,
            max_points_per_sphere=int(options.get('max_points_per_sphere', 3)),
            n_local_searches=n_local_searches,
            local_search_iters=int(options.get('local_search_iters', 20)),
            local_search_t=float(options.get('local_search_t', 0.01)),
            tol=float(options.get('tol', 1e-4)),
            clamp_value=clamp_value,
            force_cpu=bool(options.get('force_cpu', False)),
            parallel=bool(options.get('parallel', False)),
            verbose=verbose,
        )

        vertices, faces = result[:2]
        if vertices is None or faces is None:
            raise RuntimeError(
                "Reach for the Arcs returned no mesh (vertices/faces is None)."
            )

        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        if vertices.size == 0 or faces.size == 0:
            raise RuntimeError(
                "Reach for the Arcs returned an empty mesh."
            )

        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=bool(options.get('process_mesh', False)),
        )

        if bool(options.get('fix_normals', False)):
            mesh.fix_normals()

        return mesh

    def extract_surface_mesh(
        self,
        sdf_grid,
        config=None,
        mc_method='extract_mesh',
        context='',
    ):
        """
        Extract a mesh using Marching Cubes or Reach for the Arcs.

        Backward-compatible flat configuration::

            surface_extraction: reach_for_the_arcs
            rfta_max_samples: 150000
            rfta_sdf_scale: 1.0
            rfta_clamp_value: 0.05

        Equivalent nested configuration::

            surface_extraction:
              method: reach_for_the_arcs
              max_samples: 150000
              sdf_scale: 1.0
              clamp_value: 0.05

        Marching Cubes remains the default. If RFTA raises and
        fallback_to_marching_cubes is true (default), extraction continues with
        the original grid extractor.
        """
        config = {} if config is None else config
        spec = config.get(
            'surface_extraction',
            config.get('mesh_extractor', 'marching_cubes'),
        )

        nested_options = {}
        if isinstance(spec, dict):
            nested_options = dict(spec)
            method = nested_options.pop(
                'method',
                nested_options.pop('backend', 'marching_cubes'),
            )
        else:
            method = spec

        method = self._normalize_surface_extraction_method(method)

        if method == 'marching_cubes':
            return getattr(sdf_grid, mc_method)()

        if method != 'reach_for_the_arcs':
            raise ValueError(
                f"Unknown surface extraction method: {method!r}. "
                "Expected 'marching_cubes' or 'reach_for_the_arcs'."
            )

        option_defaults = {
            'max_samples': 150000,
            'near_surface_fraction': 0.75,
            'max_abs_sdf': np.inf,
            'sdf_scale': 1.0,
            'rng_seed': 3452,
            'fine_tune_iters': 3,
            'batch_size': 10000,
            'num_rasterization_spheres': 0,
            'screening_weight': 10.0,
            'rasterization_resolution': None,
            'max_points_per_sphere': 3,
            'n_local_searches': None,
            'local_search_iters': 20,
            'local_search_t': 0.01,
            'tol': 1e-4,
            'clamp_value': np.inf,
            'force_cpu': False,
            'parallel': False,
            'verbose': True,
            'process_mesh': False,
            'fix_normals': False,
            'min_samples': 128,
            'level': 0.0,
        }

        options = {}
        for name, default in option_defaults.items():
            options[name] = nested_options.get(
                name,
                config.get(f'rfta_{name}', default),
            )

        fallback = nested_options.get(
            'fallback_to_marching_cubes',
            config.get('rfta_fallback_to_marching_cubes', True),
        )

        try:
            return self._extract_mesh_reach_for_the_arcs(
                sdf_grid=sdf_grid,
                config=config,
                options=options,
            )
        except Exception as exc:
            label = f" ({context})" if context else ''
            if not bool(fallback):
                raise
            print(
                f"[surface_extraction:rfta]{label} failed: {exc}. "
                f"Falling back to {mc_method}()."
            )
            return getattr(sdf_grid, mc_method)()


    @torch.no_grad()        
    def action_deepsdf_inference(self, arg):
        data_root = arg['data_root']
        mc_grid = arg['mc_grid']
        output_folder = arg['output_folder']

        shapes = np.loadtxt(op.join(data_root, 'data.txt'), dtype=str)
        num_shapes = len(shapes)
        batch_size = 32**3
        num_samples = mc_grid.val_grid.shape[0]
        N = num_samples // batch_size + 1
        kidx_batch = np.array_split(np.arange(num_samples), N)
        with tqdm(total=num_shapes) as pbar:
            for idx in range(num_shapes):
                temp_grid = utils.create_grid_like(mc_grid)
                shape_name = shapes[idx]
                for kidx in kidx_batch:
                    samples = mc_grid.idx2pts(kidx)
                    data = {
                        'samples': torch.from_numpy(samples).float().to(self.device).unsqueeze(0),
                        'idx': torch.LongTensor([idx]).to(self.device)
                    }
                    vals = self.model.inference(data)
                    vals = vals.detach().cpu().numpy()
                    temp_grid.update_grid(vals, kidx)
                
                mesh = self.extract_surface_mesh(
                    temp_grid, arg, context=f"deepsdf:{shape_name}"
                )
                mesh_file = op.join(output_folder, shape_name, f'mesh.ply')
                os.makedirs(op.dirname(mesh_file), exist_ok=True)
                mesh.export(mesh_file)
                temp_grid = None

                pbar.update(1)

    @torch.no_grad()
    def action_shape_transform(self, arg):
        data_root = arg['data_root']
        output_folder = arg['output_folder']
        os.makedirs(output_folder, exist_ok=True)
        exp_name = arg['exp_name']
        mc_grid = arg['mc_grid']
        shape_name = arg['shape']
        
        handle = self.load_shape_handle(data_root, shape_name)
        config = utils.load_yaml_file(arg['transform_file'])
        config['handle'] = handle
        self.apply_transform(config)
        out_name = f'{exp_name}_{shape_name}'
        # cyl_outfolder = op.join(output_folder, 'cylinder')
        # os.makedirs(cyl_outfolder, exist_ok=True)
        # handle.export_skeleton_mesh(cyl_outfolder, reso=256)

        batch_size = 32**3
        for curve in handle.curves:
            key = self.encode_key(shape_name, curve.name)
            curve_data, kidx = curve.filter_grid(mc_grid)
            
            vals = self._inference_vals(curve_data, key, batch_size)
            mc_grid.update_grid(vals, kidx, mode='minimum')

        mesh = self.extract_surface_mesh(mc_grid, arg)
        mesh_file = op.join(output_folder, f'{out_name}.ply')
        mesh.export(mesh_file)

    @torch.no_grad()
    def action_shape_stretch(self, arg):
        output_folder = arg['output_folder']
        exp_name = arg['exp_name']
        mc_grid = arg['mc_grid']
        shape_name = arg['shape']
        config = utils.load_yaml_file(arg['stretch_file'])

        data_root = arg['data_root']
        handle = self.load_shape_handle(data_root, shape_name)
        # out_name = exp_name
        out_name = f'{exp_name}_{shape_name}'

        batch_size = 64**3
        for curve in handle.curves:
            key = self.encode_key(shape_name, curve.name)
            print("key = ", key)
            if key in config:
                print("stretching curve")
                stretch_config = config[key]
                new_key = stretch_config['new_key']

                if new_key == 'None':
                    continue
                
                #func = utils.define_stretch_func(stretch_config)
                stretch_arg = {
                    'curve_handle': self.curve_from_key(new_key),
                    'device': self.device,
                    #'length': stretch_config.get('length', 1.0),   # backward compatible
                    #'stretch_scale': stretch_config.get('stretch_scale', 1.0),
                    #'detail_tiles': stretch_config.get('detail_tiles', stretch_config.get('length', 1.0)),
                    #'anchor': stretch_config.get('anchor', 'coord'),
                    #'anchor_coord': stretch_config.get('anchor_coord', stretch_config['t0']),
                    'curve_idx': self.feat_dict[key],
                    'new_idx': self.feat_dict[new_key],
                    #'t0': stretch_config['t0'],
                    #'t1': stretch_config['t1'],
                    #'eps_region': stretch_config.get('eps_region', 0.03),
                    #'eps_seam': stretch_config.get('eps_seam', 0.05),
                }
                stretch_arg.update(stretch_config)

#                stretch_arg = {
#                    'curve_handle': self.curve_from_key(new_key),
#                    #'stretch_func': func,
#                    'device': self.device,
#                    'length': stretch_config['length'],
#                    'anchor': stretch_config['anchor'],
#                    'curve_idx': self.feat_dict[key],
#                    'new_idx': self.feat_dict[new_key],
#                    't0': stretch_config['t0'],
#                    't1': stretch_config['t1'],
#                    'eps_region': stretch_config['eps_region'],
#                    'eps_seam': stretch_config['eps_seam'],
#                }

                curve_data, kidx = curve.filter_grid_stretch(mc_grid, stretch_arg)
                vals, vals_base = self._inference_vals(curve_data, key, batch_size, transform='stretch')
                #vals = self._mix_inference(curve_data, stretch_arg, batch_size)
                #curve_data, kidx = curve.filter_grid_mix(mc_grid, stretch_arg)
                #vals = self._mix_inference(curve_data, stretch_arg, batch_size)
            else:
                print("normal without stretch")
                curve_data, kidx = curve.filter_grid(mc_grid)
                vals, vals_base = self._inference_vals(curve_data, key, batch_size)

            mc_grid.update_grid(vals, kidx, mode='minimum')

        print(mc_grid.grid_config)
        mesh = self.extract_surface_mesh(
            mc_grid, arg, mc_method="extract_mesh1", context="shape_stretch"
        )
        mesh_file = op.join(output_folder, f'{out_name}.ply')
        os.makedirs(op.dirname(mesh_file), exist_ok=True)
        mesh.export(mesh_file)

    @staticmethod
    def _fill_invalid_periodic_theta_field(field):
        field = np.asarray(field, dtype=np.float64).copy()
        K, T = field.shape
        x = np.arange(T)

        for i in range(K):
            row = field[i]
            valid = np.isfinite(row)
            if not np.any(valid):
                continue

            xv = x[valid]
            yv = row[valid]
            xv_ext = np.concatenate([xv - T, xv, xv + T])
            yv_ext = np.concatenate([yv, yv, yv])
            field[i] = np.interp(x, xv_ext, yv_ext)

        return field

    @staticmethod
    def _fill_invalid_s_field(field, fallback_value):
        field = np.asarray(field, dtype=np.float64).copy()
        K, T = field.shape
        x = np.arange(K)

        for j in range(T):
            col = field[:, j]
            valid = np.isfinite(col)
            if np.any(valid):
                field[:, j] = np.interp(x, x[valid], col[valid])
            else:
                field[:, j] = fallback_value

        return field

    @staticmethod
    def _smooth_periodic_theta_field(field, sigma):
        from scipy.ndimage import gaussian_filter1d

        field = np.asarray(field, dtype=np.float64)

        if sigma is None or sigma <= 0:
            return field.copy()

        T = field.shape[1]
        ext = np.concatenate([field, field, field], axis=1)
        ext = gaussian_filter1d(ext, sigma=float(sigma), axis=1, mode="nearest")
        return ext[:, T:2 * T]

    @staticmethod
    def smooth_union_sdf(a, b, k):
        # polynomial smooth min, k = blend width in SDF units
        h = np.clip(0.5 + 0.5 * (b - a) / (k + 1e-12), 0.0, 1.0)
        return b * (1.0 - h) + a * h - k * h * (1.0 - h)

    @staticmethod
    def build_avatar_snug_scale_field(
        acc_sdf,
        avatar_sdf,
        avatar_coords,
        avatar_theta,
        n_s=48,
        n_theta=64,
        surface_band=0.015,
        target_gap=0.001,
        gain=12.0,
        min_scale=0.85,
        max_scale=1.15,
        min_count=5,
        smooth_s=2.0,
        smooth_theta=1.0,
        delta_in_max=None,
        delta_out_max=None,
        debug=False,
    ):
        """
        Build local correction scale field from first-pass adaptation.

        Uses near accessory-surface samples:
            abs(acc_sdf) < surface_band

        Measures:
            gap = avatar_sdf

        Desired:
            gap ~= target_gap

        Correction:
            gap > target_gap  -> accessory too loose -> scale < 1
            gap < target_gap  -> too close/inside    -> scale > 1

        Returned field is used as:
            r_src = r_src * scale(s, theta)
        """
        from scipy.ndimage import gaussian_filter1d

        acc_sdf = np.asarray(acc_sdf).reshape(-1)
        avatar_sdf = np.asarray(avatar_sdf).reshape(-1)
        avatar_coords = np.asarray(avatar_coords).reshape(-1)
        avatar_theta = np.asarray(avatar_theta).reshape(-1)

        if not (
            acc_sdf.shape[0]
            == avatar_sdf.shape[0]
            == avatar_coords.shape[0]
            == avatar_theta.shape[0]
        ):
            raise ValueError(
                "[snug_field] length mismatch: "
                f"acc={acc_sdf.shape[0]} avatar={avatar_sdf.shape[0]} "
                f"coords={avatar_coords.shape[0]} theta={avatar_theta.shape[0]}"
            )

        surf = np.abs(acc_sdf) < float(surface_band)

        if np.sum(surf) < max(32, min_count):
            print(
                "[snug_field] too few near-surface samples:",
                int(np.sum(surf)),
                "surface_band=",
                surface_band,
            )
            return None

        s = np.clip(avatar_coords[surf], 0.0, 1.0)
        th = avatar_theta[surf]
        gap = avatar_sdf[surf]

        s_bins = np.linspace(0.0, 1.0, int(n_s))
        theta_bins = np.linspace(-np.pi, np.pi, int(n_theta), endpoint=False)

        # s bin ids
        si = np.searchsorted(s_bins, s, side="right") - 1
        si = np.clip(si, 0, int(n_s) - 1)

        # theta bin ids
        theta0 = theta_bins[0]
        period = 2.0 * np.pi
        dtheta = period / float(n_theta)
        th_wrap = ((th - theta0) % period) + theta0
        ti = np.floor((th_wrap - theta0) / dtheta).astype(np.int64) % int(n_theta)

        gap_field = np.full((int(n_s), int(n_theta)), np.nan, dtype=np.float64)
        count = np.zeros((int(n_s), int(n_theta)), dtype=np.int32)

        # Median gap per (s, theta) band
        for i in range(int(n_s)):
            mi = si == i
            if not np.any(mi):
                continue

            for j in range(int(n_theta)):
                m = mi & (ti == j)
                count[i, j] = int(np.sum(m))
                if count[i, j] >= int(min_count):
                    gap_field[i, j] = np.median(gap[m])

        # Fill missing bins
        gap_field = Agent._fill_invalid_periodic_theta_field(gap_field)
        gap_field = Agent._fill_invalid_s_field(gap_field, fallback_value=float(target_gap))
        gap_field = np.where(np.isfinite(gap_field), gap_field, float(target_gap))

        # Convert gap error to source-wrap scale.
        # gap > target => too loose => scale < 1
        # gap < target => too close => scale > 1
        error = gap_field - float(target_gap)
        scale = 1.0 - float(gain) * error
        scale = np.clip(scale, float(min_scale), float(max_scale))

        # Smooth correction field
        if smooth_theta and smooth_theta > 0:
            scale = Agent._smooth_periodic_theta_field(scale, sigma=float(smooth_theta))

        if smooth_s and smooth_s > 0:
            scale = gaussian_filter1d(scale, sigma=float(smooth_s), axis=0, mode="nearest")

        scale = np.clip(scale, float(min_scale), float(max_scale))

        # ------------------------------------------------------------
        # Additive (signed SDF offset) field, parallel to the
        # multiplicative scale field. Same (s, theta) bins.
        #
        # delta = target_gap - measured_gap, clipped.
        #   delta > 0  -> avatar is closer than target (penetrate / tight)
        #                 -> push accessory OUTWARD: vals_final -= delta
        #                    (subtracting positive shifts iso-surface OUTWARD
        #                     by ~delta along the accessory normal).
        #   delta < 0  -> avatar is farther than target (loose)
        #                 -> pull accessory INWARD.
        #
        # This is locally additive in SDF units, so it does NOT inflate the
        # whole bin's volume the way r_src *= scale does. Two-sided.
        #
        # Application site: agent_3dvec.action_part_adapt, after detail
        # reconstruction:
        #     vals_final = vals_final - interp(delta_field, s, theta)
        # ------------------------------------------------------------
        if delta_in_max is None:
            delta_in_max = float(target_gap) * 1.5
        if delta_out_max is None:
            delta_out_max = float(target_gap) * 1.5

        delta_field = float(target_gap) - gap_field
        delta_field = np.clip(delta_field, -float(delta_in_max), float(delta_out_max))

        if smooth_theta and smooth_theta > 0:
            delta_field = Agent._smooth_periodic_theta_field(
                delta_field, sigma=float(smooth_theta)
            )

        if smooth_s and smooth_s > 0:
            delta_field = gaussian_filter1d(
                delta_field, sigma=float(smooth_s), axis=0, mode="nearest"
            )

        delta_field = np.clip(delta_field, -float(delta_in_max), float(delta_out_max))

        if debug:
            active = count >= int(min_count)
            print(
                "[snug_field]",
                "surf=", int(np.sum(surf)),
                "active_bins=", int(np.sum(active)), "/", int(active.size),
                "gap[min/mean/max]=",
                float(np.nanmin(gap_field)),
                float(np.nanmean(gap_field)),
                float(np.nanmax(gap_field)),
                "scale[min/mean/max]=",
                float(np.min(scale)),
                float(np.mean(scale)),
                float(np.max(scale)),
                "delta[min/mean/max]=",
                float(np.min(delta_field)),
                float(np.mean(delta_field)),
                float(np.max(delta_field)),
            )

        return {
            "scale": scale,
            "delta": delta_field,
            "s_bins": s_bins,
            "theta_bins": theta_bins,
            "gap_field": gap_field,
            "count": count,
        }

    @staticmethod
    def apply_accessory_avatar_offset(vals_final, adapt_arg, avatar_sdf=None):
        """
        Post-process accessory SDF.

        Modes:
            accessory_offset_mode = "none"
            accessory_offset_mode = "global"
            accessory_offset_mode = "local"
            accessory_offset_mode = "both"

        SDF convention assumed:
            sdf < 0 : inside object
            sdf = 0 : surface
            sdf > 0 : outside object

        vals_final:
            accessory SDF evaluated on active grid/sample points

        avatar_sdf:
            avatar SDF evaluated on the SAME active grid/sample points
        """

        mode = adapt_arg.get("accessory_offset_mode", None)

        # Backward compatibility:
        # if old accessory_offset exists and no explicit mode is given, use global.
        if mode is None:
            if float(adapt_arg.get("accessory_offset", 0.0)) != 0.0:
                mode = "global"
            else:
                mode = "none"

        mode = str(mode).lower()

        if mode in ["none", "off", "false"]:
            return vals_final

        #is_torch = hasattr(vals_final, "device") and hasattr(vals_final, "dtype")
        is_torch = torch.is_tensor(vals_final)

        def _clip(x, lo, hi):
            if is_torch:
                import torch
                return torch.clamp(x, min=lo, max=hi)
            return np.clip(x, lo, hi)

        def _exp(x):
            if is_torch:
                import torch
                return torch.exp(x)
            return np.exp(x)

        def _maximum(a, b):
            if is_torch:
                import torch
                return torch.maximum(a, b)
            return np.maximum(a, b)

        # ------------------------------------------------------------
        # 1. Global offset
        # Positive accessory_offset expands/thickens the accessory because:
        # vals_final = vals_final - offset
        # ------------------------------------------------------------
        if mode in ["global", "both"]:
            accessory_offset = float(adapt_arg.get("accessory_offset", 0.0))
            if accessory_offset != 0.0:
                vals_final = vals_final - accessory_offset

        # ------------------------------------------------------------
        # 2. Local avatar-aware correction
        # ------------------------------------------------------------
        if mode in ["local", "both"]:
            if avatar_sdf is None:
                print("[offset] accessory_offset_mode is local/both, but avatar_sdf is None. Skipping local offset.")
                return vals_final

            target_gap = float(adapt_arg.get("target_gap", 0.003))
            local_strength = float(adapt_arg.get("local_offset_strength", 0.75))
            local_band = float(adapt_arg.get("local_offset_band", 0.015))
            local_gate_sigma = float(adapt_arg.get("local_offset_gate_sigma", 0.01))

            use_soft_snug = bool(adapt_arg.get("use_soft_snug", True))
            use_hard_clamp = bool(adapt_arg.get("use_hard_avatar_clamp", True))
            hard_clearance = float(adapt_arg.get("hard_clearance", target_gap))

            if use_soft_snug and local_strength != 0.0:
                # delta < 0 where accessory is too close/intersecting avatar
                # delta > 0 where accessory is too far from avatar
                delta = avatar_sdf - target_gap
                delta = _clip(delta, -local_band, local_band)

                # only modify near accessory surface
                sigma2 = 2.0 * local_gate_sigma * local_gate_sigma + 1e-12
                gate = _exp(-(vals_final * vals_final) / sigma2)

                vals_final = vals_final + local_strength * gate * delta

            if use_hard_clamp:
                # Remove accessory material inside forbidden avatar clearance band.
                #
                # If avatar_sdf < hard_clearance:
                #   hard_clearance - avatar_sdf > 0
                #   vals_final becomes positive there
                #   => accessory cannot exist there.
                forbidden = hard_clearance - avatar_sdf
                vals_final = _maximum(vals_final, forbidden)

        return vals_final

    def apply_adaptive_shell_thinning(
        self,
        vals_base,
        adapt_arg,
        avatar_sdf=None,
    ):
        """
        Base-only adaptive inner-shell thinning.

        Purpose:
            Preserve the outer visible garment surface as much as possible,
            but remove excessive inner volume near the avatar.

        SDF convention:
            vals_base < 0  : inside accessory solid
            vals_base = 0  : accessory surface
            vals_base > 0  : outside accessory

            avatar_sdf < 0 : inside avatar
            avatar_sdf = 0 : avatar surface
            avatar_sdf > 0 : outside avatar

        Hard inner carve equivalent:
            accessory \\ inflated_avatar

            vals_new = max(vals_base, -avatar_sdf + clearance)

        This function provides a soft version:
            vals_new = vals_base + strength * relu(forbidden - vals_base)

        where:
            forbidden = -avatar_sdf + clearance

        If strength=1.0, it becomes close to the hard max operation.
        If strength<1.0, it is a gentler thinning correction.
        """

        if not bool(adapt_arg.get("use_adaptive_shell_thinning", False)):
            return vals_base

        if avatar_sdf is None:
            return vals_base

        vals_base = np.asarray(vals_base, dtype=np.float64)
        avatar_sdf = np.asarray(avatar_sdf, dtype=np.float64)

        if vals_base.shape[0] != avatar_sdf.shape[0]:
            raise ValueError(
                f"adaptive shell thinning shape mismatch: "
                f"base={vals_base.shape[0]}, avatar={avatar_sdf.shape[0]}"
            )

        clearance = float(adapt_arg.get("shell_inner_clearance", 0.0015))
        strength = float(adapt_arg.get("shell_thin_strength", 0.35))

        # Optional: avoid affecting points very far from avatar.
        # This is only a locality gate; the real operation is still the max-like carve.
        avatar_band = float(adapt_arg.get("shell_avatar_band", 0.02))

        # Optional: restrict correction to accessory SDF band.
        # Usually keep this <= 0 or absent for correct SDF carving.
        sdf_band = float(adapt_arg.get("shell_sdf_band", -1.0))

        mode = adapt_arg.get("shell_thin_mode", "soft")

        # Inflated-avatar forbidden field.
        # The accessory SDF should not be below this near the avatar.
        forbidden = -avatar_sdf + clearance

        if mode == "hard":
            vals_new = np.maximum(vals_base, forbidden)

            if adapt_arg.get("shell_thin_debug", False):
                changed = vals_new > vals_base
                print(
                    "[adaptive_shell_thin hard]",
                    "clearance=", clearance,
                    "changed=", int(np.sum(changed)), "/", int(vals_base.shape[0]),
                    "delta max/mean=",
                    float(np.max(vals_new - vals_base)),
                    float(np.mean(vals_new - vals_base)),
                )

            return vals_new

        # Soft max-like correction.
        delta = np.maximum(forbidden - vals_base, 0.0)

        gate = np.ones_like(vals_base, dtype=np.float64)

        # Avatar locality gate:
        # Full effect near/inside avatar clearance, fades out by clearance + avatar_band.
        if avatar_band > 0:
            x = (avatar_sdf - clearance) / (avatar_band + 1e-12)
            x = np.clip(x, 0.0, 1.0)
            smooth = x * x * (3.0 - 2.0 * x)
            avatar_gate = 1.0 - smooth
            gate *= avatar_gate

        # Optional accessory SDF band gate.
        # Use only if the correction is too volumetric.
        # If enabled, it mostly affects values near the current accessory surface.
        if sdf_band is not None and sdf_band > 0:
            sdf_gate = np.exp(
                -(vals_base * vals_base)
                / (2.0 * sdf_band * sdf_band + 1e-12)
            )
            gate *= sdf_gate

        vals_new = vals_base + strength * gate * delta

        if adapt_arg.get("shell_thin_debug", False):
            changed = delta > 0
            print(
                "[adaptive_shell_thin soft]",
                "clearance=", clearance,
                "strength=", strength,
                "avatar_band=", avatar_band,
                "sdf_band=", sdf_band,
                "changed=", int(np.sum(changed)), "/", int(vals_base.shape[0]),
                "gate min/mean/max=",
                float(np.min(gate)),
                float(np.mean(gate)),
                float(np.max(gate)),
                "delta min/mean/max=",
                float(np.min(delta)),
                float(np.mean(delta)),
                float(np.max(delta)),
                "applied min/mean/max=",
                float(np.min(strength * gate * delta)),
                float(np.mean(strength * gate * delta)),
                float(np.max(strength * gate * delta)),
            )

        return vals_new

    def add_tiled_detail_coords_for_adapt(self, accessory_data, adapt_arg):
        s = np.asarray(accessory_data["coords"], dtype=np.float64)

        tgt_0 = float(adapt_arg["tgt_0"])   # accessory interval start
        tgt_1 = float(adapt_arg["tgt_1"])   # accessory interval end
        src_0 = float(adapt_arg["src_0"])   # avatar interval start
        src_1 = float(adapt_arg["src_1"])   # avatar interval end

        tlo, thi = min(tgt_0, tgt_1), max(tgt_0, tgt_1)
        slo, shi = min(src_0, src_1), max(src_0, src_1)

        # auto tile count from normalized skeleton interval ratio
        auto_tiles = abs(shi - slo) / (abs(thi - tlo) + 1e-12)
        tiles = float(adapt_arg.get("detail_tiles", auto_tiles))

        tile_start = float(adapt_arg.get("detail_tile_start", tlo))
        tile_end   = float(adapt_arg.get("detail_tile_end", thi))

        out = s.copy()

        m = (s >= min(tile_start, tile_end)) & (s <= max(tile_start, tile_end))

        u = (s[m] - tile_start) / (tile_end - tile_start + 1e-12)
        u_tile = np.mod(u * tiles, 1.0)

        out[m] = tile_start + u_tile * (tile_end - tile_start)

        # Outside [tile_start, tile_end], keep original coords.
        # So only the stretched interval gets tiled details.
        accessory_data["coords_detail"] = out
        accessory_data["samples_detail"] = accessory_data["samples_local"].copy()

        if adapt_arg.get("tiled_detail_debug", False):
            print(
                "[tiled_detail]",
                "src=", (src_0, src_1),
                "tgt=", (tgt_0, tgt_1),
                "tile_range=", (tile_start, tile_end),
                "auto_tiles=", auto_tiles,
                "used_tiles=", tiles,
                "active=", int(m.sum()), "/", int(s.shape[0]),
            )

        return accessory_data

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

            # Also allow extraction/combination settings at YAML top level.
            for config_key, config_value in raw_config.items():
                key_text = str(config_key)
                if (
                    key_text.startswith('rfta_')
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
            extraction_config.get('mesh_extractor', 'marching_cubes'),
        )
        if isinstance(extraction_spec, dict):
            requested_extractor = extraction_spec.get(
                'method',
                extraction_spec.get('backend', 'marching_cubes'),
            )
        else:
            requested_extractor = extraction_spec
        requested_extractor = self._normalize_surface_extraction_method(
            requested_extractor
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

                acc_vals = acc_out["dist"]
                acc_vals_base = acc_out["dist_base"]
                acc_vals_detail = acc_out["dist_detail"]

                if bool(adapt_arg.get("use_accessory_support_clamp", False)):
                    acc_vals, valid_support = self.clamp_pred_sdf_by_support(
                        acc_vals,
                        accessory_data,
                        positive_value=float(adapt_arg.get("support_positive_value", 1.0)),
                        w_limit=float(adapt_arg.get("support_w_limit", 999.0)),
                        rho_limit=float(adapt_arg.get("support_rho_limit", 1.35)),
                        end_margin=float(adapt_arg.get("support_end_margin", 0.0)),
                        verbose=bool(adapt_arg.get("support_clamp_verbose", True)),
                        name=accessory_key,
                    )

                    acc_vals_base, _ = self.clamp_pred_sdf_by_support(
                        acc_vals_base,
                        accessory_data,
                        positive_value=float(adapt_arg.get("support_positive_value", 1.0)),
                        w_limit=float(adapt_arg.get("support_w_limit", 999.0)),
                        rho_limit=float(adapt_arg.get("support_rho_limit", 1.35)),
                        end_margin=float(adapt_arg.get("support_end_margin", 0.0)),
                        verbose=False,
                        name=accessory_key + "_base",
                    )

                    if acc_vals_detail is not None:
                        acc_vals_detail = np.where(valid_support, acc_vals_detail, 0.0)



                if acc_vals_detail is None:
                    raise ValueError(
                        "model.inference_full did not return sdf_detail. "
                        "Need sdf_detail for base-snug + detail reconstruction."
                    )


#                acc_vals, valid_support = self.clamp_pred_sdf_by_support(
#                    acc_vals,
#                    accessory_data,
#                    positive_value=1.0,
#                    w_limit=item.get("support_w_limit", arg.get("support_w_limit", 1.2)),
#                    rho_limit=item.get("support_rho_limit", arg.get("support_rho_limit", 1.3)),
#                    end_margin=item.get("support_end_margin", arg.get("support_end_margin", 0.0)),
#                    verbose=item.get("support_clamp_verbose", arg.get("support_clamp_verbose", True)),
#                    name=accessory_key,
#                )

                #delta = 0.01
                acc_grid = utils.create_grid_like(mc_grid)
                acc_grid_base = utils.create_grid_like(mc_grid)
                #acc_grid.clear_grid(val=10.0)
                acc_grid.clear_grid()
                acc_grid_base.clear_grid()
                #acc_grid.update_grid(acc_vals - delta, kidx, mark=True, mode="overwrite")
                acc_grid.update_grid(acc_vals, kidx, mark=True, mode="overwrite")
                acc_grid_base.update_grid(acc_vals_base, kidx, mark=True, mode="overwrite")

                #print("num valid voxels:", np.sum(~acc_grid.empty_marks))
                #print("num total voxels:", acc_grid.empty_marks.shape[0])

                # Collision/snug operations happen on BASE only.
                vals_base_fit = acc_vals_base.copy()
                vals_final = acc_vals.copy()

                # Optional avatar SDF for local offset / cut.
                offset_mode = str(adapt_arg.get("accessory_offset_mode", "none")).lower()
                wants_local_offset = offset_mode in ["local", "both"]
                cut_avatar = bool(adapt_arg.get("cut_avatar", False))
                print("curt avatar = ", cut_avatar)

                # New flags also need the avatar SDF on the same active samples.
                wants_detail_avatar_gate = bool(
                    adapt_arg.get("detail_avatar_gate", False)
                )
                wants_final_carve = bool(adapt_arg.get("final_carve", False))
                wants_additive_snug = (
                    str(adapt_arg.get("snug_mode", "multiplicative")).lower()
                    == "additive"
                )

                avatar_sdf_for_offset = None
                avatar_out = None

                if (
                    wants_local_offset
                    or cut_avatar
                    or wants_detail_avatar_gate
                    or wants_final_carve
                    or wants_additive_snug
                ) and (avatar_data is not None):
                    avatar_out = self._inference_full_vals(
                        avatar_data,
                        key,
                        batch_size=batch_size,
                    )

                    # Use avatar FINAL as obstacle.
                    avatar_sdf_for_offset = avatar_out["dist"]

                    if avatar_sdf_for_offset.shape[0] != vals_base_fit.shape[0]:
                        raise ValueError(
                            f"avatar_sdf and accessory sdf length mismatch: "
                            f"avatar={avatar_sdf_for_offset.shape[0]}, "
                            f"accessory={vals_base_fit.shape[0]}"
                        )

                # Apply local soft snug / hard clamp to BASE only.
                vals_base_fit = self.apply_accessory_avatar_offset(
                    vals_base_fit,
                    adapt_arg,
                    avatar_sdf=avatar_sdf_for_offset,
                )
                vals_base_fit = self.apply_adaptive_shell_thinning(
                    vals_base_fit,
                    adapt_arg,
                    avatar_sdf=avatar_sdf_for_offset,
                )


                # Boolean difference, but apply to BASE only:
                # accessory_base \ avatar = max(sdf_accessory_base, -sdf_avatar)
                if cut_avatar and avatar_sdf_for_offset is not None:
                    print("cut_avatar")
                    avatar_clearance = float(adapt_arg.get("avatar_clearance", 0.0))
                    avatar_vals_inflated = avatar_sdf_for_offset - avatar_clearance
                    vals_base_fit = np.maximum(vals_base_fit, -avatar_vals_inflated)

                # ------------------------------------------------------------
                # Recompute detail gate from the corrected/snugged base.
                #
                # Standard form:
                #   final = base_snug + gate(base_snug) * sdf_detail
                #
                # NEW (optional) avatar-proximity gate on the detail term:
                # Even if base is correctly carved, the signed detail term
                # can drag the iso-surface back inside the avatar.
                # We multiply the detail amplitude by a smoothstep that is
                # 0 inside the avatar clearance band and 1 outside.
                # Off by default (preserves legacy behavior); enable with
                # detail_avatar_gate: true in YAML.
                # ------------------------------------------------------------
                sigma_detail = float(self.model.detail_model.sigma)
                gate_detail_snug = np.exp(
                    -(vals_base_fit * vals_base_fit)
                    / (2.0 * sigma_detail * sigma_detail + 1e-12)
                )

                detail_amp = acc_vals_detail
                use_detail_avatar_gate = bool(
                    adapt_arg.get("detail_avatar_gate", False)
                )
                if use_detail_avatar_gate and avatar_sdf_for_offset is not None:
                    detail_clearance = float(
                        adapt_arg.get("detail_clearance", 0.0005)
                    )
                    detail_band = float(
                        adapt_arg.get("detail_band", 0.004)
                    )
                    x = (avatar_sdf_for_offset - detail_clearance) / (
                        detail_band + 1e-12
                    )
                    x = np.clip(x, 0.0, 1.0)
                    gate_avatar = x * x * (3.0 - 2.0 * x)  # smoothstep01
                    detail_amp = detail_amp * gate_avatar

                    if bool(adapt_arg.get("snug_debug", False)):
                        print(
                            "[detail_avatar_gate]",
                            "clearance=", detail_clearance,
                            "band=", detail_band,
                            "gate min/mean/max=",
                            float(np.min(gate_avatar)),
                            float(np.mean(gate_avatar)),
                            float(np.max(gate_avatar)),
                        )

                vals_final = vals_base_fit + gate_detail_snug * detail_amp

                # ------------------------------------------------------------
                # Additive snug delta: applied to the FINAL SDF.
                #
                # Active when:
                #   adapt_arg["snug_mode"] == "additive"
                #   adapt_arg["avatar_snug_scale_field"] has "delta"
                #
                # Two-sided, local in (s, theta). Does NOT inflate the wrap.
                # ------------------------------------------------------------
                snug_mode = str(adapt_arg.get("snug_mode", "multiplicative")).lower()
                snug_field_obj = adapt_arg.get("avatar_snug_scale_field", None)
                if (
                    snug_mode == "additive"
                    and snug_field_obj is not None
                    and "delta" in snug_field_obj
                    and avatar_data is not None
                ):
                    avatar_curve = adapt_arg.get("avatar_curve_handle", None)
                    if avatar_curve is not None:
                        # avatar_curve is the wrapped Curve; helpers live on .core
                        avatar_curve_core = getattr(avatar_curve, "core", avatar_curve)
                        delta_per_sample = avatar_curve_core.interpolate_snug_delta_field(
                            snug_field_obj,
                            avatar_data["coords"],
                            avatar_data["angles"],
                        )
                        # convention: positive delta -> push outward
                        # vals_final - delta moves the iso-surface outward
                        # along the accessory normal by ~delta.
                        vals_final = vals_final - delta_per_sample

                        if bool(adapt_arg.get("snug_debug", False)):
                            print(
                                "[snug_additive_apply]",
                                "delta_per_sample min/mean/max=",
                                float(np.min(delta_per_sample)),
                                float(np.mean(delta_per_sample)),
                                float(np.max(delta_per_sample)),
                            )

                # ------------------------------------------------------------
                # Final-SDF carve with optional curvature-aware clearance.
                #
                # Hard guarantee: vals_final >= clearance(x) - avatar_sdf(x)
                # so the accessory iso-surface is at least clearance(x) away
                # from the avatar surface everywhere.
                #
                # clearance(x) = c0 + k_bulge * bulge_proxy(x)
                # bulge_proxy is computed from avatar_sdf either as
                #   "grad_deficit": clip(1 - |grad(avatar_sdf)|, 0, 1)
                #     positive on convex bulges where the SDF gradient is
                #     not unit (the network deviates most there), or
                #   "neg_lap": clip(-laplacian(avatar_sdf), 0, +inf)
                #     positive on convex bulges (mean curvature > 0).
                #
                # Off by default; enable with final_carve: true.
                # ------------------------------------------------------------
                use_final_carve = bool(adapt_arg.get("final_carve", False))
                if use_final_carve and avatar_sdf_for_offset is not None:
                    c0 = float(adapt_arg.get("carve_c0", 0.0002))
                    k_bulge = float(adapt_arg.get("carve_k_bulge", 0.0))
                    proxy_kind = str(
                        adapt_arg.get("carve_curvature_proxy", "grad_deficit")
                    ).lower()

                    bulge_active = np.zeros_like(avatar_sdf_for_offset)

                    if k_bulge > 0.0:
                        # Compute curvature proxy on a temp grid and read back.
                        try:
                            av_grid_full = utils.create_grid_like(mc_grid)
                            av_grid_full.clear_grid()
                            av_grid_full.update_grid(
                                avatar_sdf_for_offset,
                                kidx,
                                mark=True,
                                mode="overwrite",
                            )
                            val_arr = np.asarray(av_grid_full.val_grid)

                            grid3d = None
                            if val_arr.ndim == 3:
                                grid3d = val_arr
                            else:
                                # try to reshape from reso
                                reso = getattr(av_grid_full, "reso", None)
                                if reso is not None:
                                    if np.isscalar(reso):
                                        nx = ny = nz = int(reso)
                                    else:
                                        try:
                                            nx, ny, nz = (int(r) for r in reso)
                                        except Exception:
                                            nx = ny = nz = None
                                    if nx is not None and val_arr.size == nx * ny * nz:
                                        grid3d = val_arr.reshape(nx, ny, nz)

                            if grid3d is not None:
                                if proxy_kind == "neg_lap":
                                    gx = np.gradient(grid3d, axis=0)
                                    gy = np.gradient(grid3d, axis=1)
                                    gz = np.gradient(grid3d, axis=2)
                                    lap = (
                                        np.gradient(gx, axis=0)
                                        + np.gradient(gy, axis=1)
                                        + np.gradient(gz, axis=2)
                                    )
                                    proxy_full = np.clip(-lap, 0.0, None)
                                else:
                                    gx = np.gradient(grid3d, axis=0)
                                    gy = np.gradient(grid3d, axis=1)
                                    gz = np.gradient(grid3d, axis=2)
                                    grad_mag = np.sqrt(
                                        gx * gx + gy * gy + gz * gz
                                    )
                                    proxy_full = np.clip(
                                        1.0 - grad_mag, 0.0, 1.0
                                    )

                                # normalize so k_bulge has stable magnitude
                                pmax = float(np.max(proxy_full)) + 1e-12
                                proxy_full = proxy_full / pmax

                                bulge_active = proxy_full.reshape(-1)[kidx]
                        except Exception as e:
                            print(
                                "[final_carve] curvature proxy unavailable, "
                                f"falling back to constant clearance: {e}"
                            )
                            bulge_active = np.zeros_like(avatar_sdf_for_offset)

                    clearance_local = c0 + k_bulge * bulge_active
                    forbidden_final = clearance_local - avatar_sdf_for_offset
                    vals_final_pre = vals_final.copy()
                    vals_final = np.maximum(vals_final, forbidden_final)

                    if bool(adapt_arg.get("snug_debug", False)):
                        carved = vals_final > vals_final_pre
                        print(
                            "[final_carve]",
                            "c0=", c0,
                            "k_bulge=", k_bulge,
                            "proxy=", proxy_kind,
                            "carved=", int(np.sum(carved)),
                            "/", int(vals_final.size),
                            "clearance min/mean/max=",
                            float(np.min(clearance_local)),
                            float(np.mean(clearance_local)),
                            float(np.max(clearance_local)),
                        )

                acc_grid_base_fit = utils.create_grid_like(mc_grid)
                acc_grid_base_fit.clear_grid()
                acc_grid_base_fit.update_grid(vals_base_fit, kidx, mark=True, mode="overwrite")

                # mesh_acc_base_fit = acc_grid_base_fit.extract_mesh()  # dead debug (result unused; direct MC crashes on an all-positive UDF grid)
#                if len(mesh_acc_base_fit.faces) > 0:
#                    parts = mesh_acc_base_fit.split(only_watertight=False)
#                    if len(parts) > 0:
#                        mesh_acc_base_fit = max(parts, key=lambda m: len(m.faces))
#                    mesh_acc_base_fit.export(
#                        op.join(
#                            output_folder,
#                            f"{cc}_{mode}_basefit_{accessory_key.replace('|','_')}.ply"
#                        )
#                    )



                # 3) Debug/export this individual accessory after offset/cut/detail-reapply
                acc_grid = utils.create_grid_like(mc_grid)
                acc_grid.clear_grid()
                acc_grid.update_grid(vals_final, kidx, mark=True, mode="overwrite")

                # raw base debug
                acc_grid_base.update_grid(acc_vals_base, kidx, mark=True, mode="overwrite")


                # Extract and save this adaptation independently.
                # The output prefix is the concrete adaptation order:
                #   first run -> 0_..., second run -> 1_..., etc.
                # This is independent of any later union/blending task.
                mesh_acc = self.extract_surface_mesh(
                    acc_grid,
                    extraction_config,
                    context=(
                        f"part_adapt[{item_index}] "
                        f"target={target_key} accessory={accessory_key}"
                    ),
                )
                if len(mesh_acc.faces) > 0:
                    parts = mesh_acc.split(only_watertight=False)
                    if len(parts) > 0:
                        mesh_acc = max(parts, key=lambda m: len(m.faces))
                    individual_mesh_file = op.join(
                        output_folder,
                        f"{item_index}_{mode}_{accessory_key.replace('|','_')}.ply",
                    )
                    mesh_acc.export(individual_mesh_file)
                    print(
                        f"[part_adapt {item_index + 1}/{len(adaptation_items)}] "
                        f"saved individual mesh: {individual_mesh_file}"
                    )

#                mesh_acc_base = acc_grid_base.extract_mesh()
#                if len(mesh_acc_base.faces) > 0:
#                    parts = mesh_acc_base.split(only_watertight=False)
#                    if len(parts) > 0:
#                        mesh_acc_base = max(parts, key=lambda m: len(m.faces))
#                    mesh_acc_base.export(op.join(output_folder, f"{cc}_{mode}_base_{accessory_key.replace('|','_')}.ply"))

                # Combined reconstruction is a separate, explicit task.
                # Only collect grids when the top-level YAML enables it with:
                #     combine_adaptations: true
                if combine_adaptations:
                    blend_group = str(adapt_arg.get("blend_group", "none"))

                    if blend_group not in ["none", "", "false"]:
                        blend_groups.setdefault(blend_group, []).append({
                            "val_grid": acc_grid.val_grid.copy(),
                            "empty_marks": acc_grid.empty_marks.copy(),
                            "adapt_arg": dict(adapt_arg),
                        })
                    else:
                        all_acc_grids.append({
                            "val_grid": acc_grid.val_grid.copy(),
                            "empty_marks": acc_grid.empty_marks.copy(),
                        })

                # target_key is unique within this loaded avatar handle. Once
                # matched, this concrete adaptation run is complete.
                break

            if not matched_target:
                # This should already be caught by the validation above, but
                # keep a local guard so future handle changes fail loudly.
                raise KeyError(
                    f"Adaptation target did not run: {target_key!r}. "
                    f"Available targets: {sorted(available_target_keys.keys())}"
                )

        if not combine_adaptations:
            print(
                "[part_adapt] individual adaptations saved; "
                "combined union/extraction skipped "
                "(combine_adaptations=false)."
            )
            return

        mc_grid.clear_grid()

        # Explicit combined task: hard-union all ungrouped accessories.
        for it in all_acc_grids:
            valid_i = ~it["empty_marks"]
            mc_grid.val_grid[valid_i] = np.minimum(
                mc_grid.val_grid[valid_i],
                it["val_grid"][valid_i],
            )
            mc_grid.empty_marks[valid_i] = False

        for group_name, items in blend_groups.items():

            blend_delta = float(
                items[0]["adapt_arg"].get("mesh_blend_delta", 0.0)
            )

            group_vals = np.full_like(mc_grid.val_grid, 10.0)
            group_empty = np.ones_like(mc_grid.empty_marks, dtype=bool)

            for ii, it in enumerate(items):
                vals_i = it["val_grid"]
                valid_i = ~it["empty_marks"]

                overlap = (~group_empty) & valid_i

                group_vals[valid_i] = np.minimum(
                    group_vals[valid_i],
                    vals_i[valid_i],
                )

                if blend_delta > 0.0 and np.any(overlap):
                    group_vals[overlap] = self.smooth_union_sdf(
                        group_vals[overlap],
                        vals_i[overlap],
                        blend_delta,
                    )

                group_empty[valid_i] = False

            valid_group = ~group_empty

            mc_grid.val_grid[valid_group] = np.minimum(
                mc_grid.val_grid[valid_group],
                group_vals[valid_group],
            )
            mc_grid.empty_marks[valid_group] = False


        valid_final = int(np.sum(~mc_grid.empty_marks))
        if valid_final == 0:
            raise RuntimeError(
                "Final adaptation SDF grid is empty. No accessory samples were "
                "merged, so surface extraction cannot run."
            )

        mesh = self.extract_surface_mesh(
            mc_grid,
            extraction_config,
            context="part_adapt final combined mesh",
        )
        mesh_file = op.join(output_folder, f'{out_name}.ply')
        os.makedirs(op.dirname(mesh_file), exist_ok=True)
        mesh.export(mesh_file)

    @torch.no_grad()
    def action_part_mixing(self, arg):
        output_folder = arg['output_folder']
        exp_name = arg['exp_name']
        mc_grid = arg['mc_grid']
        shape_name = arg['shape']
        config = utils.load_yaml_file(arg['mixing_file'])

        data_root = arg['data_root']
        handle = self.load_shape_handle(data_root, shape_name)
        # out_name = exp_name
        out_name = f'{shape_name}_{exp_name}_mix'

        batch_size = 64**3
        cc = 0
        for curve in handle.curves:
            key = self.encode_key(shape_name, curve.name)
            print(key)

            if key in config:
                mix_config = config[key]
                new_key = mix_config['new_key']

                if new_key == 'None':
                    continue
                
                func1 = utils.define_mix_func(mix_config, weights_reverse=True)
                func2 = utils.define_mix_func(mix_config, weights_reverse=False)

                mix_arg = {
                    'curve_handle': self.curve_from_key(new_key),
                    'mix_func1': func1,
                    'mix_func2': func2,
                    'device': self.device,
                    'curve_idx': self.feat_dict[key],
                    'new_idx': self.feat_dict[new_key],
                }

                curve_data, kidx = curve.filter_grid_mix(mc_grid, mix_arg)
                vals, vals_base = self._mix_inference(curve_data, mix_arg, batch_size)
            else:
                curve_data, kidx = curve.filter_grid(mc_grid)
                vals, vals_base = self._inference_vals(curve_data, key, batch_size)

            mc_grid.update_grid(vals, kidx, mode='minimum')

        mesh = self.extract_surface_mesh(mc_grid, arg)
        os.makedirs(output_folder, exist_ok=True)
        mesh_file = op.join(output_folder, f'{out_name}.ply')
        #os.makedirs(op.dirname(mesh_file), exist_ok=True)
        mesh.export(mesh_file)

    @torch.no_grad()
    def action_visualize_SDF(self, arg):
        output_folder = arg['output_folder']
        exp_name = arg['exp_name']
        shape_name = arg['shape']
        handle = self.handles[shape_name]
        
        samples = arg['samples']
        N = samples.shape[0]
        sdfs = 10*np.ones(N)
        mask = np.zeros(N, dtype=bool)

        batch_size = 64**3
        for curve in handle.curves:
            key = self.encode_key(shape_name, curve.name)
            curve_data, inside = curve.localize_samples(samples)
            mask[inside] = True
            
            vals = self._inference_vals(curve_data, key, batch_size)
            sdfs[inside] = np.minimum(sdfs[inside], vals)

        out_file = op.join(output_folder, 'vis_sdf', f'VisSDF_{shape_name}.png')
        os.makedirs(op.dirname(out_file), exist_ok=True)
        img_size = int(np.sqrt(N))
        utils.sdf2image(out_file, img_size, sdfs, mask, a_max=0.2)

    @torch.no_grad()
    def action_shape_manipulate(self, arg):
        output_folder = arg['output_folder']
        exp_name = arg['exp_name']
        mc_grid = arg['mc_grid']
        shape_name = arg['shape']

        handle = self.handles[shape_name]
        # manipuate armadillo
        cR_leg = handle.curve_dict['R_leg']
        cR_foot = handle.curve_dict['R_foot']
        cL_arm = handle.curve_dict['L_arm']
        cL_hand = handle.curve_dict['L_hand']
        idx = 2
        anchor,rot = cR_leg.rot_part(idx, 'z', -45)
        cR_leg.update()
        cR_foot.apply_rotation(anchor, rot)
        cR_foot.update()

        anchor,rot = cL_arm.rot_part(idx, 'y', -30)
        cL_arm.update()
        cL_hand.apply_rotation(anchor, rot)
        cL_hand.update()

        out_name = f'{exp_name}_{shape_name}'
        cyl_folder = op.join(output_folder, out_name)
        os.makedirs(cyl_folder, exist_ok=True)
        handle.export_skeleton_mesh(cyl_folder, reso=256)
        # raise ValueError

        batch_size = 64**3
        for curve in handle.curves:
            key = self.encode_key(shape_name, curve.name)
            curve_data, kidx = curve.filter_grid(mc_grid)
            
            vals = self._inference_vals(curve_data, key, batch_size)
            mc_grid.update_grid(vals, kidx, mode='minimum')

        t0 = time()
        mesh = self.extract_surface_mesh(mc_grid, arg)
        print('MC time cost: ', time()-t0)
        mesh_file = op.join(output_folder, f'{exp_name}_{shape_name}.ply')
        os.makedirs(op.dirname(mesh_file), exist_ok=True)
        mesh.export(mesh_file)

    def action_add_part(self, arg):
        mc_grid = arg['mc_grid']
        delta = arg['delta']
        shape_arg = arg['shape']
        new_part_arg = arg['new_part']

        shape_name = shape_arg['name']
        new_shape_name = new_part_arg['shape_name']
        new_part_name = new_part_arg['part_name']
        
        handle = self.handles[shape_name]

        new_handle = self.handles[new_shape_name]
        if 'pose_file' in new_part_arg:
            pose_file = new_part_arg['pose_file']
            new_handle.apply_pose(pose_file)

        if 'rotation' in new_part_arg:
            rot_arg = new_part_arg['rotation']
            vec = rot_arg['vec']
            anchor_idx = rot_arg['anchor_idx']
            new_handle.action_rotate(
                new_part_name, vec, anchor_idx
            )

        smooth = utils.SmoothMaxMin(3, delta)
        new_curve = new_handle.curve_dict[new_part_name]
        new_grid = utils.create_grid_like(mc_grid)

        if arg['area_mode'] == 'large':
            ## Step1: calculate cylinders and blend new part cylinder
            # NOTE: only handle considered, not content(shape).
            for curve in handle.curves:
                # points inside delta-level set of cylinders
                sdfs, kidx = curve.calc_global_implicit(mc_grid, delta)
                # NOTE: use np.minimum for simple boolean union
                mc_grid.update_grid(sdfs, kidx, mode='minimum', mark=True)

            cyl_sdfs, cyl_kidx = new_curve.calc_global_implicit(mc_grid, delta)
            new_grid.update_grid(cyl_sdfs, cyl_kidx, mode='overwrite')

            ## Step2: filter out grid points in the blended area
            vals1, common_kidx,_ = mc_grid.get_marked_intersection(cyl_kidx)
            vals2 = new_grid.get_vals(common_kidx)

            # NOTE: Area is: |d1-d2|_{n,delta} \leq delta
            # d1: value of handle cylinders implicit; 
            # d2: value of new part cylinder implicit
            area = smooth.abs(vals1 - vals2) <= delta
            area_kidx = common_kidx[area]

            mc_grid.clear_grid()
            new_grid.clear_grid()

        ## Step3: calculate SDF values of two shapes
        with torch.no_grad():
            for cid in range(handle.num_curve):
                curve = handle.curves[cid]
                key = self.encode_key(shape_name, curve.name)
                curve_data, kidx = curve.filter_grid(mc_grid)
                
                vals = self._inference_vals(curve_data, key)
                # overwrite cylinder SDF, take min with other curve part
                mc_grid.update_grid_func(vals, kidx, func=np.minimum)

            key = self.encode_key(new_shape_name, new_curve.name)
            curve_data, new_kidx = new_curve.filter_grid(mc_grid)
            new_vals = self._inference_vals(curve_data, key)
            new_grid.update_grid_func(new_vals, new_kidx, np.minimum)

            # cyl_sdfs, cyl_kidx = new_curve.calc_global_implicit(mc_grid, 0.)
            # pos = cyl_sdfs > -1.
            # pos_sdfs = cyl_sdfs[pos]
            # pos_kidx = cyl_kidx[pos]
            # new_grid.update_grid_func(pos_sdfs, pos_kidx, func=np.maximum)

        if arg['area_mode'] == 'small':
            # new_grid_kidx = np.argwhere(new_grid.func_marks).flatten()
            # area_marks = mc_grid.func_marks[new_grid_kidx]
            # area_kidx = new_grid_kidx[area_marks]

            # for mode-2: blending on intersection of cylinders
            area_marks = mc_grid.func_marks[new_kidx]
            area_kidx = new_kidx[area_marks]
        
        ## Step4: blend two shapes SDFs on the filtered grid points
        vals_shape = mc_grid.get_vals(area_kidx)
        vals_part = new_grid.get_vals(area_kidx)
        vals_area = smooth.min(vals_shape, vals_part)

        cyl_sdfs, cyl_kidx = new_curve.calc_global_implicit(mc_grid, delta)
        pos = cyl_sdfs > 0.
        pos_sdfs = cyl_sdfs[pos]
        pos_kidx = cyl_kidx[pos]
        mc_grid.update_grid(pos_sdfs, pos_kidx, mode='minimum')

        mc_grid.update_grid(new_vals, new_kidx, mode='minimum')
        mc_grid.update_grid(vals_area, area_kidx, mode='overwrite')

        mesh = self.extract_surface_mesh(mc_grid, arg)
        output_path = op.join(arg['output_path'], arg['config_name'])
        os.makedirs(output_path, exist_ok=True)
        out_name = '{}_{}|{}_{}.ply'.format(
            shape_name, new_shape_name, new_part_name, arg['exp_name']
        )
        # out_name = 'debug_blend.ply'
        mesh.export(op.join(output_path, out_name))
        print('{}|{} Done.'.format(
            arg['exp_name'], arg['config_name']
        ))

    def action_add_parts(self, arg):
        mc_grid = arg['mc_grid']
        delta = arg['delta']
        shape_arg = arg['shape']
        new_part_arg = arg['new_part']

        shape_name = shape_arg['name']
        smooth = utils.SmoothMaxMin(3, delta)
        new_grid = utils.create_grid_like(mc_grid)
        
        handle = self.handles[shape_name]
        for cid in range(handle.num_curve):
            curve = handle.curves[cid]
            key = self.encode_key(shape_name, curve.name)
            curve_data, kidx = curve.filter_grid(mc_grid)
            
            vals = self._inference_vals(curve_data, key)
            # overwrite cylinder SDF, take min with other curve part
            mc_grid.update_grid_func(vals, kidx, np.minimum)

        for item_arg in new_part_arg:
            new_shape_name = item_arg['shape_name']
            new_part_name = item_arg['part_name']
            new_handle = self.handles[new_shape_name]
            if 'pose_file' in item_arg:
                pose_file = item_arg['pose_file']
                new_handle.apply_pose(pose_file)

            new_curve = new_handle.curve_dict[new_part_name]

            ## Step3: calculate SDF values of two shapes
            key = self.encode_key(new_shape_name, new_curve.name)
            curve_data, new_kidx = new_curve.filter_grid(mc_grid)
            new_vals = self._inference_vals(curve_data, key)
            new_grid.update_grid_func(new_vals, new_kidx, np.minimum)

            # blending on intersection of cylinders
            area_marks = mc_grid.func_marks[new_kidx]
            area_kidx = new_kidx[area_marks]
        
            ## blend two shapes SDFs on the filtered grid points
            vals_shape = mc_grid.get_vals(area_kidx)
            vals_part = new_grid.get_vals(area_kidx)
            vals_area = smooth.min(vals_shape, vals_part)

            cyl_sdfs, cyl_kidx = new_curve.calc_global_implicit(mc_grid, 0.)
            pos = cyl_sdfs > 0.
            pos_sdfs = cyl_sdfs[pos]
            pos_kidx = cyl_kidx[pos]
            mc_grid.update_grid_func(pos_sdfs, pos_kidx, np.minimum)

            mc_grid.update_grid_func(new_vals, new_kidx, np.minimum)
            mc_grid.update_grid_func(vals_area, area_kidx, func=None)

            new_grid.clear_grid()

        mesh = self.extract_surface_mesh(mc_grid, arg)
        output_path = op.join(arg['output_path'], arg['config_name'])
        os.makedirs(output_path, exist_ok=True)
        out_name = '{}_{}.ply'.format(
            shape_name, arg['exp_name']
        )
        # out_name = 'debug_blend.ply'
        mesh.export(op.join(output_path, out_name))
        print('{}|{} Done.'.format(
            arg['exp_name'], arg['config_name']
        ))

    def action_slot_part(self, arg):
        mc_grid = arg['mc_grid']
        delta = arg['delta']
        shape_arg = arg['shape']
        shape_name = shape_arg['name']
        
        handle1 = self.handles[shape_name]

        new_part_arg = arg['new_part']
        new_shape_name = new_part_arg['shape_name']
        new_part_name = new_part_arg['part_name']
        handle2 = self.handles[new_shape_name]
        curve2_ori = handle2.curve_dict[new_part_name]
        curve2 = utils.copy_curve(handle2, new_part_name)
        curve2.apply_action_arg(new_part_arg)
        
        smooth = utils.SmoothMaxMin(3, delta)
        ball_arg = new_part_arg['ball']
        origin = ball_arg['origin']
        radius = ball_arg['radius']

        # shape1 sdf grid
        for cid in range(handle1.num_curve):
            curve = handle1.curves[cid]
            key = self.encode_key(shape_name, curve.name)
            curve_data, kidx = curve.filter_grid(mc_grid)
            
            vals = self._inference_vals(curve_data, key)
            # overwrite cylinder SDF, take min with other curve part
            mc_grid.update_grid(vals, kidx, mode='minimum')

        # NOTE: filter radius + delta
        ball_pts, ball_kidx = mc_grid.filter_grid_ball(origin, radius+delta)
        sdf_val1 = mc_grid.get_vals(ball_kidx)
        sdf_ball = np.linalg.norm(ball_pts-origin, axis=1)
        sdf_ball -= radius
        sdf_ball = smooth.min(sdf_val1, sdf_ball)

        # move new part and re-scale
        anchor_idx = new_part_arg['anchor_idx']
        utils.curve_transform({
            'curve': curve2,
            'anchor_idx': anchor_idx,
            'origin': origin,
            'radius': radius,
        })

        # calculate extended part 
        area_coords, area_ts = curve2.core.localize_samples_global(ball_pts)
        points_shape2 = curve2_ori.core.inverse_transform(area_coords, area_ts)
        sdf_val2 = 10*np.ones(points_shape2.shape[0])
        for curve in handle2.curves:
            key = self.encode_key(new_shape_name, curve.name)
            curve_data, inside = curve.localize_samples(points_shape2)
            if np.any(inside):
                vals = self._inference_vals(curve_data, key)
                # overwrite cylinder SDF, take min with other curve part
                vals = np.minimum(vals, sdf_val2[inside])
                sdf_val2[inside] = vals

        sdf_val2 = smooth.max(sdf_val2, sdf_ball)
        sdf_val1 = np.minimum(sdf_val1, sdf_val2)
        mc_grid.update_grid(sdf_val1, ball_kidx, mode='overwrite')

        key = self.encode_key(new_shape_name, curve2.name)
        curve_data, new_kidx = curve2.filter_grid(mc_grid)
        new_vals = self._inference_vals(curve_data, key)

        # calculate intersection of ball and new part cylinder
        new_grid = utils.create_grid_like(mc_grid)
        new_grid.update_grid_func(sdf_val1, ball_kidx, func=None)
        area_marks = mc_grid.func_marks[new_kidx]
        area_kidx = new_kidx[area_marks]
        new_grid.update_grid_func(new_vals, new_kidx, func=smooth.min)
        area_vals = new_grid.get_vals(area_kidx)

        mc_grid.update_grid(new_vals, new_kidx, mode='minimum')
        mc_grid.update_grid(area_vals, area_kidx, mode='overwrite')
        mesh = self.extract_surface_mesh(mc_grid, arg)
        out_name = '{}_{}|{}_{}.ply'.format(
            shape_name, new_shape_name, new_part_name, arg['exp_name']
        )
        self.output_mesh(mesh, out_name, arg)

    def action_slot_move_part(self, arg):
        mc_grid = arg['mc_grid']
        delta = arg['delta']
        shape_arg = arg['shape']
        shape_name = shape_arg['name']
        part_name = shape_arg['part']
        
        handle = self.handles[shape_name]

        smooth = utils.SmoothMaxMin(3, delta)
        curve_ori = handle.curve_dict[part_name]
        curve_new = utils.copy_curve(handle, part_name)
        curve_new.apply_action_arg(shape_arg)

        for cid in range(handle.num_curve):
            curve = handle.curves[cid]
            if curve.name == part_name:
                continue

            key = self.encode_key(shape_name, curve.name)
            curve_data, kidx = curve.filter_grid(mc_grid)
            
            vals = self._inference_vals(curve_data, key)
            # overwrite cylinder SDF, take min with other curve part
            mc_grid.update_grid_func(vals, kidx, func=np.minimum)

        anchor_idx = shape_arg['anchor_idx']
        origin = curve_new.core.key_points[anchor_idx]
        radius = curve_new.core.key_radius[anchor_idx].max()
        # NOTE: filter radius + delta
        ball_pts, ball_kidx = mc_grid.filter_grid_ball(origin, radius+delta)
        sdf_val1 = mc_grid.get_vals(ball_kidx)
        sdf_ball = np.linalg.norm(ball_pts-origin, axis=1)
        sdf_ball -= radius
        sdf_ball = smooth.min(sdf_val1, sdf_ball)

        # calculate extended part 
        area_coords, area_ts = curve_new.core.localize_samples_global(ball_pts)
        points_shape = curve_ori.core.inverse_transform(area_coords, area_ts)
        sdf_val2 = 10*np.ones(points_shape.shape[0])
        for curve in handle.curves:
            key = self.encode_key(shape_name, curve.name)
            curve_data, inside = curve.localize_samples(points_shape)
            if np.any(inside):
                vals = self._inference_vals(curve_data, key)
                # overwrite cylinder SDF, take min with other curve part
                vals = np.minimum(vals, sdf_val2[inside])
                sdf_val2[inside] = vals
        
        sdf_val2 = smooth.max(sdf_val2, sdf_ball)
        sdf_val1 = np.minimum(sdf_val1, sdf_val2)
        mc_grid.update_grid(sdf_val1, ball_kidx, mode='overwrite')

        key = self.encode_key(shape_name, curve_new.name)
        curve_data, new_kidx = curve_new.filter_grid(mc_grid)
        new_vals = self._inference_vals(curve_data, key)

        # calculate intersection of ball and new part cylinder
        new_grid = utils.create_grid_like(mc_grid)
        new_grid.update_grid_func(sdf_val1, ball_kidx, func=None)
        area_marks = mc_grid.func_marks[new_kidx]
        area_kidx = new_kidx[area_marks]
        new_grid.update_grid_func(new_vals, new_kidx, func=smooth.min)
        area_vals = new_grid.get_vals(area_kidx)

        mc_grid.update_grid(new_vals, new_kidx, mode='minimum')
        mc_grid.update_grid(area_vals, area_kidx, mode='overwrite')
        mesh = self.extract_surface_mesh(mc_grid, arg)
        out_name = '{}|{}_{}.ply'.format(
            shape_name, part_name, arg['exp_name']
        )
        self.output_mesh(mesh, out_name, arg)



Agent = AgentSDF
