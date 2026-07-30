import os, pickle
import os.path as op
import numpy as np
import trimesh
import torch
from time import time
from tqdm.autonotebook import tqdm

#import app_utils as utils
import app_utils_3dvec as utils


class AgentBase():
    """Field-agnostic infrastructure shared by AgentSDF and AgentUDF.
    Model/data/handle loading, curve inference, adaptation geometry.
    Holds NO signed-field or unsigned-field post-processing."""

    def _convert_sdf_pair_sign(self, vals, vals_base):
        # UDF/base: no sign convention. AgentSDF overrides this.
        return vals, vals_base

    def _convert_full_output_sign(self, out):
        # UDF/base: no sign convention. AgentSDF overrides this.
        return out

    def __call__(self, name, arg):
        method_name = f'action_{name}'
        if hasattr(self, method_name):
            method = getattr(self, method_name)
        else:
            raise NotImplementedError('Not found')
        
        res = method(arg)
        print(f'Done: {method_name}')
        return res

    def encode_key(self, shape_name, curve_name):
        return f'{shape_name}|{curve_name}'

    def decode_key(self, key):
        return key.split('|')

    def curve_from_key(self, key):
        shape_name, curve_name = self.decode_key(key)
        handle = self.handles[shape_name]
        return handle.curve_dict[curve_name]

    def load_model(self, device, config_path, model_path, mode='train', checkpoint='final'):
        cpu_device = torch.device('cpu')
        model, opt = utils.load_model(cpu_device, config_path, model_path, mode, checkpoint)
        self.model = model
        self.model.to(device)
        self.opt = opt
        self.device = device

    def load_data(self, data_root, data_path):
        shapes = np.loadtxt(op.join(data_root, data_path), dtype=str).tolist()
        handles = {}
        feat_dict = {}
        fid = 0
        self.shape_global_curveids = {}
        for idx, shape_name in enumerate(shapes):
            shape_name, shape_type = shape_name.split('|')
            item_path = op.join(data_root, f'{shape_name}')
            #handle_path = op.join(item_path, 'handle/std_handle.pkl')
            handle_path = op.join(item_path, 'handle/std_handle.npz')
            #handle = utils.load_handle(handle_path)
            handle = self.load_shape_with_npz(data_root, shape_name, shape_type)
            handles[shape_name] = handle
            shape_curve_ids = []
            self.shape_global_curveids[idx] = fid
            for curve in handle.curves:
                key = f'{shape_name}|{curve.name}'
                print("key=",key)
                key = self.encode_key(shape_name, curve.name)
                feat_dict[key] = fid
                fid += 1

        self.handles = handles
        self.feat_dict = feat_dict

    def load_shape_with_npz(self, data_root, shape_name, shape_type):
        item_path = op.join(data_root, f'{shape_name}')
        handle_path = op.join(item_path, 'handle/std_handle.npz')
        handle = utils.load_handle(handle_path, shape_type)

        #npz_path = op.join(item_path, f'handle/inference.npz')
        #if op.exists(npz_path):
        #    self.apply_curve_state_npz(handle, npz_path, shape_name)

        return handle

    def load_shape_handle(self, data_root, shape_name, shape_type):
        return self.load_shape_with_npz(data_root, shape_name, shape_type)

    def load_handle(self, handle_path):
        handle = Handle()
        handle.load(handle_path)
        return handle

    def get_curve_data(self, curve_data, global_curveid, shape_name=""):
        samples_local = curve_data['samples_local']
        samples_global = curve_data['samples']
        samples_coords = curve_data['coords']
        cids = curve_data['curve_idx'].astype(np.int32)
        maxid = max(cids)

        model_input = {}
        for curve_id in range(maxid+1):
            ids = np.where(cids == curve_id)[0]
            model_input[global_curveid+curve_id] = {'samples': torch.from_numpy(samples_local[ids]).float(),
                    'samples_global': torch.from_numpy(samples_global[ids]).float(),
                    'coords': torch.from_numpy(samples_coords[ids]).float(),
                    'curve_idx': (torch.ones(ids.shape[0])*(global_curveid+curve_id)).long()
            }
            #trimesh.Trimesh(vertices=model_input[global_curveid+curve_id]['samples_global'] ,process = False).export(str(shape_name)+"_"+str(global_curveid+curve_id)+'_onsurface.ply')
            #trimesh.Trimesh(vertices=model_input[global_curveid+curve_id]['samples'],process = False).export(str(shape_name)+"_"+str(global_curveid+curve_id)+'_onsurface_local.ply')
        return model_input

    def set_embedding(self, device, log_path):
        embd_model, _ = utils.load_model(device, log_path, 'final')
        self.model.set_embedding(embd_model.encoder.embd)

    def apply_transform(self, arg):
        handle = arg['handle']
        # curve posing, by setting new pose of current skeleton
        if 'pose' in arg: 
            pose_file = arg['pose']['pose_file']
            z_axis = None
            if 'z_axis' in arg['pose']:
                z_axis = arg['pose']['z_axis']
                z_axis = np.asarray(z_axis)

            handle.apply_pose(pose_file, z_axis)
        
        # local scaling by changing the key radius
        if 'scaling' in arg:
            handle.apply_scaling(arg['scaling'])

        # tilting, or twisting the shape, by changing the key frame(axis)
        if 'tilt' in arg:
            handle.apply_tilt(arg['tilt'])

    def apply_curve_state_npz(self, handle, npz_path, shape_name):
        d = np.load(npz_path, allow_pickle=True)["arr_0"].item()

        for cid, curve in enumerate(handle.curves):
            core = curve.core

            candidates = [
                f"{shape_name}_on_{cid}",
                f"{shape_name}_on_{curve.name}",
                f"{shape_name}_{curve.name}",
                curve.name,
            ]

            found = None
            for k in candidates:
                if k in d:
                    found = k
                    break

            if found is None:
                print(f"[npz] no state for curve {curve.name} in {npz_path}")
                continue

            s = d[found]

            #if "key_ts" in s:
            #    core.key_ts = np.asarray(s["key_ts"], dtype=np.float64)
            #if "key_points" in s:
            #    core.key_points = np.asarray(s["key_points"], dtype=np.float64)
            #if "key_frame" in s:
            #    core.key_frame = np.asarray(s["key_frame"], dtype=np.float64)
            #if "key_train_radius" in s:
            #    core.key_train_radius = np.asarray(s["key_train_radius"], dtype=np.float64)
            #    core.key_radius = core.key_train_radius
            if "key_cylinder_radius" in s:
                core.key_cylinder_radius = np.asarray(s["key_cylinder_radius"], dtype=np.float64)
            core.key_cylinder_radius = core.key_cylinder_radius - 0.4 # np.asarray(s["key_cylinder_radius"], dtype=np.float64)
            if "key_wrap_radius" in s:
                core.key_wrap_radius = np.asarray(s["key_wrap_radius"], dtype=np.float64)
            if "wrap_s_bins" in s:
                core.wrap_s_bins = np.asarray(s["wrap_s_bins"], dtype=np.float64)
            if "wrap_theta_bins" in s:
                core.wrap_theta_bins = np.asarray(s["wrap_theta_bins"], dtype=np.float64)
            if "wrap_radius_max" in s and s["wrap_radius_max"] is not None:
                core.wrap_radius_max = np.asarray(s["wrap_radius_max"], dtype=np.float64)
            if "key_occupancy_rho" in s and s["key_occupancy_rho"] is not None:
                core.key_occupancy_rho = np.asarray(s["key_occupancy_rho"], dtype=np.float64)

            print(f"[npz] applied {found} -> {shape_name}|{curve.name}")

    def local_support_mask(
        self,
        samples_data,
        w_limit=999.0,
        rho_limit=1.30,
        end_margin=0.0,
        return_debug=False,
    ):
        coords = np.asarray(samples_data["coords"]).reshape(-1)
        sl = np.asarray(samples_data["samples_local"])

        if sl.ndim != 2 or sl.shape[1] < 3:
            raise ValueError(f"samples_local must have shape (N, >=3), got {sl.shape}")

        if sl.shape[0] != coords.shape[0]:
            raise ValueError(
                f"coords and samples_local length mismatch: "
                f"coords={coords.shape[0]}, samples_local={sl.shape[0]}"
            )

        vx = 2.0 * coords - 1.0
        w_n = sl[:, 0] - vx

        if "rho_n" in samples_data:
            rho_n = np.asarray(samples_data["rho_n"]).reshape(-1)
        else:
            u_n = sl[:, 1]
            v_n = sl[:, 2]
            rho_n = np.sqrt(u_n * u_n + v_n * v_n)

        valid = (
            (coords >= 0.0) &
            (coords <= 1.0) &
            (rho_n <= rho_limit)
        )

        if w_limit < 100.0:
            valid &= np.abs(w_n) <= w_limit

        if end_margin > 0.0:
            valid &= coords >= end_margin
            valid &= coords <= 1.0 - end_margin

        if return_debug:
            debug = {
                "num_total": int(valid.shape[0]),
                "num_valid": int(valid.sum()),
                "valid_ratio": float(valid.mean()) if valid.shape[0] > 0 else 0.0,
                "w_min": float(w_n.min()) if w_n.shape[0] > 0 else 0.0,
                "w_max": float(w_n.max()) if w_n.shape[0] > 0 else 0.0,
                "rho_min": float(rho_n.min()) if rho_n.shape[0] > 0 else 0.0,
                "rho_max": float(rho_n.max()) if rho_n.shape[0] > 0 else 0.0,
                "used_rho_n_field": "rho_n" in samples_data,
            }
            return valid, debug

        return valid

    def clamp_soft_pred_sdf_by_support(
        self,
        pred_sdf,
        samples_data,
        positive_value=1.0,
        w_limit=1.20,
        rho_limit=1.15,
        end_margin=0.0,
        verbose=False,
        name="",
        soft=True,
        rho_fade_limit=None,
        w_fade_limit=None,
    ):
        valid, debug = self.local_support_mask(
            samples_data,
            w_limit=w_limit,
            rho_limit=rho_limit,
            end_margin=end_margin,
            return_debug=True,
        )

        pred_sdf = np.asarray(pred_sdf).reshape(-1).copy()

        if pred_sdf.shape[0] != valid.shape[0]:
            raise ValueError(
                f"pred_sdf and support mask length mismatch: "
                f"pred_sdf={pred_sdf.shape[0]}, valid={valid.shape[0]}"
            )

        # ------------------------------------------------------------
        # Old behavior: hard clamp
        # This creates the staircase:
        #   pred_sdf[~valid] = positive_value
        # ------------------------------------------------------------
        if not soft:
            pred_sdf[~valid] = positive_value
            return pred_sdf, valid

        # ------------------------------------------------------------
        # New behavior: soft rho support fade
        # rho <= rho_limit        : unchanged
        # rho_limit -> fade_limit : smoothly blended to positive
        # rho >= fade_limit       : fully positive
        # ------------------------------------------------------------
        sl = np.asarray(samples_data["samples_local"])
        coords = np.asarray(samples_data["coords"]).reshape(-1)

        vx = 2.0 * coords - 1.0
        w_n = sl[:, 0] - vx

        if "rho_n" in samples_data:
            rho_n = np.asarray(samples_data["rho_n"]).reshape(-1)
        else:
            u_n = sl[:, 1]
            v_n = sl[:, 2]
            rho_n = np.sqrt(u_n * u_n + v_n * v_n)

        rho0 = float(rho_limit)
        rho1 = float(rho_fade_limit) if rho_fade_limit is not None else rho0 + 0.18
        rho1 = max(rho1, rho0 + 1e-6)

        t_rho = np.clip((rho_n - rho0) / (rho1 - rho0 + 1e-12), 0.0, 1.0)
        fade = t_rho * t_rho * (3.0 - 2.0 * t_rho)

        # Hard invalid for coord/end only. These are not the skirt side boundary.
        hard_invalid = (coords < 0.0) | (coords > 1.0)

        if end_margin > 0.0:
            hard_invalid |= coords < end_margin
            hard_invalid |= coords > 1.0 - end_margin

        # Optional soft/hard w support.
        if w_limit < 100.0:
            if w_fade_limit is None:
                hard_invalid |= np.abs(w_n) > w_limit
            else:
                w0 = float(w_limit)
                w1 = max(float(w_fade_limit), w0 + 1e-6)
                t_w = np.clip((np.abs(w_n) - w0) / (w1 - w0 + 1e-12), 0.0, 1.0)
                fade_w = t_w * t_w * (3.0 - 2.0 * t_w)
                fade = np.maximum(fade, fade_w)

        pred_sdf = (1.0 - fade) * pred_sdf + fade * float(positive_value)
        pred_sdf[hard_invalid] = float(positive_value)

        # For detail masking, valid_support should mean "not fully outside".
        soft_valid = ~hard_invalid
        soft_valid &= rho_n <= rho1

        if w_limit < 100.0 and w_fade_limit is not None:
            soft_valid &= np.abs(w_n) <= float(w_fade_limit)
        elif w_limit < 100.0:
            soft_valid &= np.abs(w_n) <= float(w_limit)

        if verbose:
            print(
                f"[support_clamp {name}] "
                f"valid={debug['num_valid']}/{debug['num_total']} "
                f"({100.0 * debug['valid_ratio']:.2f}%) "
                f"w=[{debug['w_min']:.3f},{debug['w_max']:.3f}] "
                f"rho=[{debug['rho_min']:.3f},{debug['rho_max']:.3f}] "
                f"rho_fade=[{rho0:.3f},{rho1:.3f}] "
                f"soft_valid={int(soft_valid.sum())}/{soft_valid.shape[0]} "
                f"soft={soft}"
            )

        return pred_sdf, soft_valid

    def estimate_rho_limit_from_pred(
        self,
        pred_sdf,
        samples_data,
        n_bins=48,
        surface_band=0.03,
        q=0.98,
        margin=0.08,
        smooth_s=2.0,
        fallback=1.2,
    ):
        from scipy.ndimage import gaussian_filter1d

        pred_sdf = np.asarray(pred_sdf).reshape(-1)
        sl = np.asarray(samples_data["samples_local"])
        coords = np.asarray(samples_data["coords"]).reshape(-1)

        u_n = sl[:, 1]
        v_n = sl[:, 2]
        rho_n = np.sqrt(u_n * u_n + v_n * v_n)

        near = np.abs(pred_sdf) < float(surface_band)

        edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

        rho_lim = np.full(int(n_bins), np.nan, dtype=np.float64)

        for i in range(int(n_bins)):
            m = near & (coords >= edges[i]) & (coords < edges[i + 1])
            if np.sum(m) >= 10:
                rho_lim[i] = np.quantile(rho_n[m], float(q)) + float(margin)

        good = np.isfinite(rho_lim)
        if np.any(good):
            rho_lim[~good] = np.interp(centers[~good], centers[good], rho_lim[good])
        else:
            rho_lim[:] = float(fallback)

        if smooth_s > 0:
            rho_lim = gaussian_filter1d(rho_lim, sigma=float(smooth_s), mode="nearest")

        return centers, rho_lim

    def clamp_pred_sdf_by_support(
        self,
        pred_sdf,
        samples_data,
        positive_value=1.0,
        w_limit=1.20,
        rho_limit=1.15,
        end_margin=0.0,
        verbose=False,
        name="",
    ):
#        valid, debug = self.local_support_mask(
#            samples_data,
#            w_limit=w_limit,
#            rho_limit=rho_limit,
#            end_margin=end_margin,
#            return_debug=True,
#        )

        s_bins, rho_lim_bins = self.estimate_rho_limit_from_pred(
            pred_sdf,
            samples_data,
            n_bins=48,
            surface_band=0.03,
            q=0.98,
            margin=0.08,
            smooth_s=2.0,
            fallback=rho_limit,
        )
        sl = np.asarray(samples_data["samples_local"])
        coords = np.asarray(samples_data["coords"]).reshape(-1)
        vx = 2.0 * coords - 1.0
        w_n = sl[:, 0] - vx

        u_n = sl[:, 1]
        v_n = sl[:, 2]
        rho_n = np.sqrt(u_n * u_n + v_n * v_n)

        rho_limit_i = np.interp(coords, s_bins, rho_lim_bins)
        band = 0.25
        support_sdf = (rho_n - rho_limit_i) / (band + 1e-12)
        support_sdf = np.clip(support_sdf, -1.0, 1.0) * float(positive_value)

        pred_sdf = np.maximum(pred_sdf, support_sdf)


        pred_sdf = np.asarray(pred_sdf).reshape(-1).copy()

#        if pred_sdf.shape[0] != valid.shape[0]:
#            raise ValueError(
#                f"pred_sdf and support mask length mismatch: "
#                f"pred_sdf={pred_sdf.shape[0]}, valid={valid.shape[0]}"
#            )
#



        #pred_sdf[~valid] = positive_value
        # Only clamp points that are outside rho support AND predicted as material.
        outside = rho_n > float(rho_limit)
        bad = outside & (pred_sdf < 0.0)

        # Push those bad points positive, but leave everything else untouched.
        pred_sdf[bad] = float(positive_value)

        if verbose:
            print(
                f"[adaptive_rho_clamp {name}] "
                f"bad={int(bad.sum())}/{bad.shape[0]} "
                f"rho_limit={rho_limit:.3f} "
                f"rho=[{rho_n.min():.3f},{rho_n.max():.3f}]"
            )

        return pred_sdf, ~bad

        if verbose:
            print(
                f"[support_clamp {name}] "
                f"valid={debug['num_valid']}/{debug['num_total']} "
                f"({100.0 * debug['valid_ratio']:.2f}%) "
                f"w=[{debug['w_min']:.3f},{debug['w_max']:.3f}] "
                f"rho=[{debug['rho_min']:.3f},{debug['rho_max']:.3f}] "
                f"limits: w={w_limit}, rho={rho_limit}, end={end_margin}"
            )

        return pred_sdf, valid

    def _inference_vals(self, curve_data, key, batch_size=None, transform=None):
        # use_batch: aim to divide data into batches to save GPU mem
        num_samples = curve_data['samples_local'].shape[0]
        #num_context_samples = context_data['samples'].shape[0]

        if batch_size is not None and num_samples > batch_size:
            N = num_samples // batch_size + 1
            vals = []
            vals_base = []
            batches = np.array_split(np.arange(num_samples), N)
            for idx, batch in enumerate(batches):
                batch_curve_data = {key: val[batch] for key,val in curve_data.items()}
                batch_curve_data['device'] = self.device
                #print(self.feat_dict[key])
                batch_curve_data['curve_idx'] = self.feat_dict[key]

                #r = np.random.choice(num_context_samples, size=2048, replace=False)

                #batch_curve_data['on_curve_idx'] = context_data['curve_idx'][r]
                #batch_curve_data['on_coords'] = context_data['coords'][r]
                #batch_curve_data['on_surface_samples'] = context_data['samples'][r]
                #batch_curve_data['on_surface_samples_global'] = context_data['samples_global'][r]


                #trimesh.Trimesh(vertices=batch_curve_data['on_surface_samples_gloabl'].numpy(), process = False).export(str(self.feat_dict[key])+'_onsurface.ply')
                #trimesh.Trimesh(vertices=batch_curve_data['samples_local'], process = False).export(str(self.feat_dict[key])+'_'+str(idx)+'_query.ply')

                vals_batch, vals_base_batch = self.model.inference(batch_curve_data, transform=transform)
                vals_batch = vals_batch.squeeze()
                vals_base_batch = vals_base_batch.squeeze()
                vals.append(vals_batch.detach().cpu().numpy())
                vals_base.append(vals_base_batch.detach().cpu().numpy())
            
            vals = np.concatenate(vals)
            vals_base = np.concatenate(vals_base)
            return self._convert_sdf_pair_sign(vals, vals_base)

        curve_data['device'] = self.device
        curve_data['curve_idx'] = self.feat_dict[key]

        with torch.no_grad():
            vals, vals_base = self.model.inference(curve_data)
            vals = vals.squeeze()
            vals_base = vals_base.squeeze()
            vals = vals.detach().cpu().numpy()
            vals_base = vals_base.detach().cpu().numpy()
        return self._convert_sdf_pair_sign(vals, vals_base)

    def _inference_full_vals(self, curve_data, key, batch_size=None, transform=None):
        num_samples = curve_data['samples_local'].shape[0]

        def _to_numpy_dict(out):
            res = {}
            for k, v in out.items():
                if v is None:
                    res[k] = None
                else:
                    res[k] = v.squeeze().detach().cpu().numpy()
            return res

        if batch_size is not None and num_samples > batch_size:
            N = num_samples // batch_size + 1
            batches = np.array_split(np.arange(num_samples), N)

            chunks = {}
            for batch in batches:
                batch_curve_data = {k: v[batch] for k, v in curve_data.items()}
                batch_curve_data['device'] = self.device
                batch_curve_data['curve_idx'] = self.feat_dict[key]

                out = self.model.inference_full(batch_curve_data, transform=transform)
                out_np = _to_numpy_dict(out)

                for k, v in out_np.items():
                    if v is None:
                        continue
                    chunks.setdefault(k, []).append(v)

            result = {k: np.concatenate(vs) for k, vs in chunks.items()}
            return self._convert_full_output_sign(result)

        curve_data['device'] = self.device
        curve_data['curve_idx'] = self.feat_dict[key]

        with torch.no_grad():
            out = self.model.inference_full(curve_data, transform=transform)

        result = _to_numpy_dict(out)
        return self._convert_full_output_sign(result)

    def _mix_inference(self, curve_data, mix_arg, batch_size=None):
        num_samples = curve_data['samples'].shape[0]
        cd = curve_data
        print(cd.keys())
        if batch_size is not None and num_samples > batch_size:
            N = num_samples // batch_size + 1
            vals = []
            vals_base = []
            batches = np.array_split(np.arange(num_samples), N)
            for batch in batches:
                mix_arg['samples_local'] = cd['samples_local'][batch]
                mix_arg['coords'] = cd['coords'][batch]
                mix_arg['angles'] = cd['angles'][batch]
                mix_arg['radius'] = cd['radius'][batch]
                mix_arg['rho'] = cd['rho'][batch]
                mix_arg['rho_n'] = cd['rho_n'][batch]
                vals_batch, vals_base_batch = self.model.mix_curve(mix_arg)
                vals_batch = vals_batch.squeeze()
                vals_base_batch = vals_base_batch.squeeze()
                
                vals.append(vals_batch.detach().cpu().numpy())
                vals_base.append(vals_base_batch.detach().cpu().numpy())

            return np.concatenate(vals), np.concatenate(vals_base)
        else:
            mix_arg['samples_local'] = cd['samples_local']
            mix_arg['coords'] = cd['coords']
            mix_arg['angles'] = cd['angles']
            mix_arg['radius'] = cd['radius']
            mix_arg['rho'] = cd['rho']
            mix_arg['rho_n'] = cd['rho_n']
            vals, vals_base = self.model.mix_curve(mix_arg)
            vals = vals.squeeze()
            vals_base = vals_base.squeeze()
            return vals.detach().cpu().numpy(), vals_base.detach().cpu().numpy()

    def shape_repose(self, arg):
        shape_arg = arg['shape']
        shape_name = shape_arg['name']
        handle = self.handles[shape_name]
        if 'pose_file' in shape_arg:
            pose_file = shape_arg['pose_file']
            handle.apply_pose(pose_file)

        if 'rotation' in shape_arg:
            rot_arg = shape_arg['rotation']
            handle.action_rotate_euler(rot_arg)

    def output_mesh(self, mesh, out_name, arg):
        output_folder = op.join(arg['output_path'], arg['config_name'])
        os.makedirs(output_folder, exist_ok=True)

        mesh.export(op.join(output_folder, out_name))
        print('{}|{} Done.'.format(
            arg['exp_name'], arg['config_name']
        ))

    @staticmethod
    def _normalize_surface_extraction_method(method):
        """Normalize user-facing names for the surface extractor."""
        method = str(method).strip().lower().replace('-', '_').replace(' ', '_')
        print(method)

        if method in {
            'mc',
            'marching_cube',
            'marching_cubes',
            'marchingcubes',
        }:
            return 'marching_cubes'

        if method in {
            'rfta',
            'reach_for_arc',
            'reach_for_arcs',
            'reach_for_the_arc',
            'reach_for_the_arcs',
        }:
            return 'reach_for_the_arcs'

        if method in {
            'dualmeshudf_model', 'dmudf_model', 'udf_model',
            'model_udf', 'dualmeshudf_direct', 'model',
        }:
            return 'dualmeshudf_model'

        if method in {'dualmeshudf', 'dual_mesh_udf', 'dmudf', 'udf'}:
            return 'dualmeshudf'

        return method

    def phi_curve(self, curve_handle, curve_key, X_world, batch_size=65536):
        # X_world: (N,3) numpy float32
        curve_data, inside = curve_handle.core.localize_samples(X_world)

        # IMPORTANT: localize_samples returns `inside` indices into the original X_world
        # curve_data already corresponds to those inside points, in the same order.
        vals, _ = self._inference_vals(curve_data, curve_key, batch_size=batch_size)

        # fill full array; outside points treated as "far outside"
        out = np.full((X_world.shape[0],), 10.0, dtype=np.float32)  # large positive
        valid = np.zeros((X_world.shape[0],), dtype=bool)
        out[inside] = vals.reshape(-1)
        valid[inside] = True
        return out, valid

    def phi_and_grad_curve(self, curve_handle, curve_key, X, h=1e-3, batch_size=65536):
        f0, v0 = self.phi_curve(curve_handle, curve_key, X, batch_size=batch_size)
        grads = np.zeros_like(X, dtype=np.float32)
        valid_all = v0.copy()
        for i in range(3):
            e = np.zeros((1,3), dtype=np.float32)
            e[0,i] = h
            fp, vp = self.phi_curve(curve_handle, curve_key, X + e, batch_size=batch_size)
            fm, vm = self.phi_curve(curve_handle, curve_key, X - e, batch_size=batch_size)
            grads[:,i] = (fp - fm) / (2*h)
            valid_all &= vp & vm

        n = np.linalg.norm(grads, axis=1, keepdims=True) + 1e-12
        good = valid_all & (n[:,0] > 1e-3)

        grads[good] /= (n[good] + 1e-12)

        # For bad points, set gradient to 0 (caller should not move them)
        grads[~good] = 0.0

        return f0, grads, good

    def filter_grid_dependent_runtime(self, mc_grid, adapt_arg):
        """
        Support-first dependent path:
        parent support already exists in world space,
        instantiate child support from parent anchor,
        filter/localize directly on child runtime support.
        """
        parent_support_data = adapt_arg["parent_support_data"]

        parent_curve_key = adapt_arg["parent_accessory_key"]
        child_curve_key = adapt_arg["accessory_key"]

        parent_curve_handle = self.curve_from_key(parent_curve_key)
        child_curve_handle = self.curve_from_key(child_curve_key)

        parent_anchor = parent_curve_handle.core._compute_anchor_from_support(
            parent_support_data,
            at=adapt_arg.get("parent_anchor_at", "end"),
            coord=adapt_arg.get("parent_anchor_coord", None),
        )

        dep_template = adapt_arg["dep_template"]

        parent_anchor_meta = dep_template["parent_anchor_meta"]
        #global_scale = float(adapt_arg.get("scale", 1.0))
        #global_scale = float(
        #    adapt_arg.get(
        #        "scale",
        #        parent_support_data.get("assembly_scale", 1.0)
        #    )
        #)
        global_scale = float(parent_support_data.get("assembly_scale", 1.0))
        global_scale *= float(adapt_arg.get("scale", 1.0))
        use_parent_aniso = bool(adapt_arg.get("use_parent_anisotropic_scale", True))

        if use_parent_aniso:
            scale_w = global_scale * (
                parent_anchor["x_radius"] / (parent_anchor_meta["x_radius"] + 1e-12)
            )
            scale_y = global_scale * (
                parent_anchor["radius"][0] / (parent_anchor_meta["radius"][0] + 1e-12)
            )
            scale_z = global_scale * (
                parent_anchor["radius"][1] / (parent_anchor_meta["radius"][1] + 1e-12)
            )
        else:
            scale_w = global_scale
            scale_y = global_scale
            scale_z = global_scale

        runtime_child_support = child_curve_handle.core._build_dependent_support_from_anchor(
            dep_template,
            parent_anchor,
            scale_w=scale_w,
            scale_y=scale_y,
            scale_z=scale_z,
            radius_scale_y=scale_y,
            radius_scale_z=scale_z,
        )

        child_data, kidx = child_curve_handle.filter_grid_on_runtime_support(
            mc_grid,
            runtime_child_support,
            norm=adapt_arg.get("infer_scale", 1.35),
        )

        print("parent assembly_scale =", parent_support_data.get("assembly_scale", None))
        print("child config scale    =", adapt_arg.get("scale", 1.0))
        print("final global_scale    =", global_scale)

        return child_data, runtime_child_support, kidx

    def split_template_from_key(self, accessory_key, split_s, child_s0, child_s1):
        if not hasattr(self, "_split_template_cache"):
            self._split_template_cache = {}

        cache_key = (accessory_key, float(split_s), float(child_s0), float(child_s1))
        if cache_key not in self._split_template_cache:
            curve_handle = self.curve_from_key(accessory_key)
            self._split_template_cache[cache_key] = self.build_split_curve_template(
                curve_handle=curve_handle,
                split_s=split_s,
                child_s0=child_s0,
                child_s1=child_s1,
            )
        return self._split_template_cache[cache_key]

    def build_attached_curve_template(
        self,
        parent_curve_handle,
        child_curve_handle,
        parent_joint_s,
        child_joint_s,
        child_s0=0.0,
        child_s1=1.0,
        n_samples=None,
    ):
        parent_core = parent_curve_handle.core
        child_core = child_curve_handle.core

        parent_core.update_coords()
        parent_core.update_frame()
        child_core.update_coords()
        child_core.update_frame()

        if n_samples is None:
            #child_coords = np.asarray(child_core.key_ts, dtype=np.float64)
            s_all = np.asarray(child_core.key_ts, dtype=np.float64)
            keep = (s_all >= min(child_s0, child_s1)) & (s_all <= max(child_s0, child_s1))
            child_coords = s_all[keep]
            if child_coords.shape[0] < 2:
                child_coords = np.linspace(child_s0, child_s1, 100)
        else:
            child_coords = np.linspace(child_s0, child_s1, n_samples)

        parent_info = parent_core.interpolate(np.array([parent_joint_s], dtype=np.float64))
        p0 = parent_info["points"][0].copy()
        F0 = parent_info["frame"][0].copy()
        r0 = parent_info["radius"][0].copy()
        x0 = float(parent_core.calc_x_radius(np.array([parent_joint_s], dtype=np.float64))[0])

        child_info = child_core.interpolate(child_coords)
        child_points = np.asarray(child_info["points"], dtype=np.float64)
        child_frames = np.asarray(child_info["frame"], dtype=np.float64)
        child_radius = np.asarray(child_info["radius"], dtype=np.float64)
        child_x_radius = np.asarray(child_core.calc_x_radius(child_coords), dtype=np.float64)

        local_points = (child_points - p0[None, :]) @ F0.T
        local_frames = np.einsum("kij,jm->kim", child_frames, F0.T)

        template = {
            "local_points": local_points,
            "local_frames": local_frames,
            "radius": child_radius.copy(),
            "coords": child_coords.copy(),
            "x_radius": child_x_radius.copy(),
            "parent_anchor_meta": {
                "radius": r0.copy(),
                "x_radius": x0,
            },
            "parent_joint_s": float(parent_joint_s),
            "child_joint_s": float(child_joint_s),
            "parent_curve_name": parent_curve_handle.name,
            "child_curve_name": child_curve_handle.name,
        }
        return template

    def compute_root_assembly_scale(self, avatar_curve_handle, accessory_curve_handle, src_0, src_1, tgt_0, tgt_1):
        L_avatar = self.interval_length_on_curve(avatar_curve_handle, src_0, src_1)
        L_accessory = self.interval_length_on_curve(accessory_curve_handle, tgt_0, tgt_1)
        return L_avatar / (L_accessory + 1e-12)

    def interval_length_on_curve(self, curve_handle, s0, s1):
        pts = curve_handle.core.interpolate(np.array([s0, s1], dtype=np.float64))["points"]
        return float(np.linalg.norm(pts[1] - pts[0]))

    def attached_template_from_keys(self, parent_key, child_key, parent_joint_s, child_joint_s, child_s0=0.0, child_s1=1.0):
        parent_curve_handle = self.curve_from_key(parent_key)
        child_curve_handle = self.curve_from_key(child_key)
        return self.build_attached_curve_template(
            parent_curve_handle=parent_curve_handle,
            child_curve_handle=child_curve_handle,
            parent_joint_s=parent_joint_s,
            child_joint_s=child_joint_s,
            child_s0=child_s0,
            child_s1=child_s1,
        )

    def build_split_curve_template(
        self,
        curve_handle,
        split_s=0.6,
        child_s0=0.6,
        child_s1=1.0,
        n_samples=None,
    ):
        """
        Build a dependent template from a suffix of the SAME original curve.

        Parent anchor = original curve at split_s
        Child support  = original curve over [child_s0, child_s1], expressed
                         in parent-anchor local coordinates.

        Returns template dict compatible with localize_samples_dependent().
        """
        core = curve_handle.core
        core.update_coords()
        core.update_frame()

        if n_samples is None:
            # use original key_ts inside range
            s_all = np.asarray(core.key_ts, dtype=np.float64)
            keep = (s_all >= min(child_s0, child_s1)) & (s_all <= max(child_s0, child_s1))
            child_coords_global = s_all[keep]
            if child_coords_global.shape[0] < 2:
                child_coords_global = np.linspace(child_s0, child_s1, 100)
        else:
            child_coords_global = np.linspace(child_s0, child_s1, n_samples)

        # anchor at split point
        split_info = core.interpolate(np.array([split_s], dtype=np.float64))
        p0 = split_info["points"][0].copy()
        F0 = split_info["frame"][0].copy()      # rows [T,N,B]
        r0 = split_info["radius"][0].copy()
        x0 = float(core.calc_x_radius(np.array([split_s], dtype=np.float64))[0])

        child_info = core.interpolate(child_coords_global)
        child_points = np.asarray(child_info["points"], dtype=np.float64)
        child_frames = np.asarray(child_info["frame"], dtype=np.float64)
        child_radius = np.asarray(child_info["radius"], dtype=np.float64)
        child_x_radius = np.asarray(core.calc_x_radius(child_coords_global), dtype=np.float64)

        # world delta -> anchor local [w,u,v]
        local_points = (child_points - p0[None, :]) @ F0.T

        # child frame relative to anchor frame
        local_frames = np.einsum("kij,jm->kim", child_frames, F0.T)

        # normalize child coords to [0,1] for the dependent template
        #denom = max(abs(child_s1 - child_s0), 1e-12)
        #child_coords_local = (child_coords_global - child_s0) / denom
        #child_coords_local = np.clip(child_coords_local, 0.0, 1.0)
        child_coords_local = child_coords_global.copy()

        template = {
            "local_points": local_points,
            "local_frames": local_frames,
            "radius": child_radius.copy(),
            "coords": child_coords_local.copy(),
            "x_radius": child_x_radius.copy(),
            "parent_anchor_meta": {
                "radius": r0.copy(),
                "x_radius": x0,
            },
            # helpful debug/meta
            "source_curve_key": getattr(curve_handle, "name", "unknown"),
            "split_s": float(split_s),
            "child_s0": float(child_s0),
            "child_s1": float(child_s1),
        }
        template["local_points"][0] = np.zeros(3, dtype=np.float64)
        return template


    @torch.no_grad()
    def action_ngcnet_inference(self, arg):
        data_root = arg['data_root']
        data_path = arg['data_path']
        self.load_data(data_root, data_path)
        mc_grid = arg['mc_grid']
        output_folder = arg['output_folder']
        checkpoint = arg['checkpoint']

        num_shapes = len(self.handles)
        err_res = {}
        reso = mc_grid.reso
        shapes = os.listdir(data_root)
        # max number of query points for Marching Cubes
        batch_size = 16**3


        output_folder = arg['output_folder']
        exp_name = arg['exp_name']
        mc_grid = arg['mc_grid']
        shape_name = arg['shape']
        #config = utils.load_yaml_file(arg['adapt_file'])

        data_root = arg['data_root']
        handle = self.load_shape_handle(data_root, shape_name, 'avatar')

        out_name = f'{shape_name}_{exp_name}'
        os.makedirs(output_folder, exist_ok=True)

        batch_size = 64**3
        #mc_grid.clear_grid(val=10.0)
        mc_grid.clear_grid()

        adapted_support_cache = {}
        all_acc_grids = []
        blend_groups = {}
        cc = 0


        with tqdm(total=num_shapes) as pbar:
            for shape_name,handle in self.handles.items():
                if shape_name not in shapes:
                    print("shape_name")
                    pbar.update(1)
                    continue

                temp_grid = utils.create_grid_like(mc_grid)
                temp_grid_base = utils.create_grid_like(mc_grid)
                #context_input = self.model_input[shape_name]

                for curve in handle.curves:
                    #print("curve_name = ", curve.name)
                    #print("shape name = ", shape_name)
                    key = self.encode_key(shape_name, curve.name)
                    #print("key = ", key)
                    #print("feat dict = ", self.feat_dict[key])
                    curve_data, kidx = curve.filter_grid(mc_grid)
                    #print(curve_data)
                    #context_data = context_input[self.feat_dict[key]]


                    #trimesh.Trimesh(vertices=curve_data['samples'], process=False).export(shape_name+'_'+curve.name+'mc_grid.ply')
                    #trimesh.Trimesh(vertices=curve_data['samples_local'], process=False).export(shape_name+'_'+curve.name+'_query.ply')
                    #trimesh.Trimesh(vertices=context_data['samples'].numpy(), process=False).export(shape_name+'_'+curve.name+'context.ply')
                    
                    #vals = self._inference_vals(curve_data, context_data, key, batch_size=batch_size)
                    vals, vals_base = self._inference_vals(curve_data, key, batch_size=batch_size)

#                    vals, valid_support = self.clamp_pred_sdf_by_support(
#                        vals,
#                        curve_data,
#                        positive_value=1.0,
#                        w_limit=arg.get("support_w_limit", 1.2),
#                        rho_limit=arg.get("support_rho_limit", 1.3),
#                        end_margin=arg.get("support_end_margin", 0.0),
#                        verbose=arg.get("support_clamp_verbose", False),
#                        min_valid_ratio=0.60,
#                        name=key,
#                    )
#
#                    vals_base, _ = self.clamp_pred_sdf_by_support(
#                        vals_base,
#                        curve_data,
#                        positive_value=1.0,
#                        w_limit=arg.get("support_w_limit", 1.2),
#                        rho_limit=arg.get("support_rho_limit", 1.3),
#                        end_margin=arg.get("support_end_margin", 0.0),
#                        verbose=False,
#                        min_valid_ratio=0.60,
#                        name=key + "_base",
#                    )

                    # These samples were actually evaluated by the network.
                    # MCGrid.extract_mesh() and the DC adapter use empty_marks to
                    # crop/diagnose the active support, so mark them explicitly.
                    temp_grid.update_grid(
                        vals, kidx, mode='minimum', mark=True
                    )
                    temp_grid_base.update_grid(
                        vals_base, kidx, mode='minimum', mark=True
                    )
                
                mesh = self.extract_surface_mesh(
                    temp_grid, arg, context=f"ngcnet:{shape_name}"
                )
                mesh_file = op.join(output_folder, shape_name, f'{shape_name}_{checkpoint}_mesh{reso}.ply')
                os.makedirs(op.dirname(mesh_file), exist_ok=True)
                mesh.export(mesh_file)
                temp_grid = None

#                mesh = temp_grid_base.extract_mesh()
#                mesh_file = op.join(output_folder, shape_name, f'{shape_name}_base_{checkpoint}_mesh{reso}.ply')
#                os.makedirs(op.dirname(mesh_file), exist_ok=True)
#                mesh.export(mesh_file)
#                temp_grid_base = None

                # gt_file = op.join(data_root, shape_name, 'mesh.ply')
                # err = utils.eval_shape(mesh_file, gt_file)
                # err_res[shape_name] = err

                pbar.update(1)
