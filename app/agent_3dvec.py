import os, pickle
import os.path as op
import numpy as np
import trimesh
import torch
from time import time
from tqdm.autonotebook import tqdm

#import app_utils as utils
import app_utils_3dvec as utils

class Agent():
    """docstring for Agent."""
    def __init__(self):
        pass

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
        #item_path = op.join(data_root, f'{shape_name}')
        #handle_path = op.join(item_path, 'handle/std_handle.pkl')
        #handle_path = op.join(item_path, 'handle/std_handle.npz')
        #handle = utils.load_handle(handle_path)
        #handle = self.load_shape_with_npz(data_root, shape_name)
        #return handle

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

        pred_sdf[~valid] = positive_value

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

    def __inference_vals(self, curve_data, key, batch_size=None, transform=None):
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
            
            return np.concatenate(vals), np.concatenate(vals_base)

        curve_data['device'] = self.device
        curve_data['curve_idx'] = self.feat_dict[key]

        with torch.no_grad():
            vals, vals_base = self.model.inference(curve_data)
            vals = vals.squeeze()
            vals_base = vals_base.squeeze()
            vals = vals.detach().cpu().numpy()
            vals_base = vals_base.detach().cpu().numpy()
        return vals, vals_base


    def __inference_full_vals(self, curve_data, key, batch_size=None, transform=None):
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

            return {k: np.concatenate(vs) for k, vs in chunks.items()}

        curve_data['device'] = self.device
        curve_data['curve_idx'] = self.feat_dict[key]

        with torch.no_grad():
            out = self.model.inference_full(curve_data, transform=transform)

        return _to_numpy_dict(out)
    
    def __mix_inference(self, curve_data, mix_arg, batch_size=None):
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
        with tqdm(total=num_shapes) as pbar:
            for shape_name,handle in self.handles.items():
                if shape_name not in shapes:
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
                    
                    #vals = self.__inference_vals(curve_data, context_data, key, batch_size=batch_size)
                    vals, vals_base = self.__inference_vals(curve_data, key, batch_size=batch_size)

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

                    temp_grid.update_grid(vals, kidx, mode='minimum')
                    temp_grid_base.update_grid(vals_base, kidx, mode='minimum')
                
                mesh = temp_grid.extract_mesh()
                mesh_file = op.join(output_folder, shape_name, f'{shape_name}_{checkpoint}_mesh{reso}.ply')
                os.makedirs(op.dirname(mesh_file), exist_ok=True)
                mesh.export(mesh_file)
                temp_grid = None

                mesh = temp_grid_base.extract_mesh()
                mesh_file = op.join(output_folder, shape_name, f'{shape_name}_base_{checkpoint}_mesh{reso}.ply')
                os.makedirs(op.dirname(mesh_file), exist_ok=True)
                mesh.export(mesh_file)
                temp_grid_base = None

                # gt_file = op.join(data_root, shape_name, 'mesh.ply')
                # err = utils.eval_shape(mesh_file, gt_file)
                # err_res[shape_name] = err

                pbar.update(1)
        
        # err_file = op.join(output_folder, 'err.pkl')
        # with open(err_file, 'wb') as f:
        #     pickle.dump(err_res, f)

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
                
                mesh = temp_grid.extract_mesh()
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
            
            vals = self.__inference_vals(curve_data, key, batch_size)
            mc_grid.update_grid(vals, kidx, mode='minimum')

        mesh = mc_grid.extract_mesh()
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
                vals, vals_base = self.__inference_vals(curve_data, key, batch_size, transform='stretch')
                #vals = self.__mix_inference(curve_data, stretch_arg, batch_size)
                #curve_data, kidx = curve.filter_grid_mix(mc_grid, stretch_arg)
                #vals = self.__mix_inference(curve_data, stretch_arg, batch_size)
            else:
                print("normal without stretch")
                curve_data, kidx = curve.filter_grid(mc_grid)
                vals, vals_base = self.__inference_vals(curve_data, key, batch_size)

            mc_grid.update_grid(vals, kidx, mode='minimum')

        print(mc_grid.grid_config)
        mesh = mc_grid.extract_mesh1()
        mesh_file = op.join(output_folder, f'{out_name}.ply')
        os.makedirs(op.dirname(mesh_file), exist_ok=True)
        mesh.export(mesh_file)

    def phi_curve(self, curve_handle, curve_key, X_world, batch_size=65536):
        # X_world: (N,3) numpy float32
        curve_data, inside = curve_handle.core.localize_samples(X_world)

        # IMPORTANT: localize_samples returns `inside` indices into the original X_world
        # curve_data already corresponds to those inside points, in the same order.
        vals, _ = self.__inference_vals(curve_data, curve_key, batch_size=batch_size)

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
        #grads /= n
        #return f0, grads
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



    @torch.no_grad()
    def action_part_adapt(self, arg):
        output_folder = arg['output_folder']
        exp_name = arg['exp_name']
        mc_grid = arg['mc_grid']
        shape_name = arg['shape']
        config = utils.load_yaml_file(arg['adapt_file'])

        data_root = arg['data_root']
        handle = self.load_shape_handle(data_root, shape_name, 'avatar')
        out_name = f'{shape_name}_{exp_name}'
        os.makedirs(output_folder, exist_ok=True)

        batch_size = 64**3
        #mc_grid.clear_grid(val=10.0)
        mc_grid.clear_grid()

        adapted_support_cache = {}
        cc = 0

        for item in config:
            target_key = item['target_key']
            accessory_key = item['accessory_key']
            mode = item.get('mode', 'direct')
            print(target_key)

            for curve in handle.curves:
                key = self.encode_key(shape_name, curve.name)
                if key != target_key:
                    print(key)
                    continue
                print("success")
                print(key)

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
                    adapted_support_cache[cache_key] = {
                        "coords": acc_coords.copy(),
                        "points": accessory_data["runtime_points"].copy(),
                        "frame": accessory_data["runtime_frame"].copy(),
                        "radius": accessory_data["radius"].copy(),
                        "x_radius": accessory_data["x_radius"].copy(),
                        "assembly_scale": root_scale,
                        #"x_radius": accessory_curve_handle.core.calc_x_radius(acc_coords).copy(),
                    }
                    #adapted_support_cache[cache_key]["assembly_scale"] = root_scale

                else:
                    raise ValueError(f"Unknown adapt mode: {mode}")

                #acc_vals, acc_vals_base = self.__inference_vals(
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
                        acc_out0 = self.__inference_full_vals(
                            accessory_data,
                            accessory_key,
                            batch_size=batch_size,
                        )
                        # Debug: first-pass base BEFORE snug rerun
#                        pre_snug_base_grid = utils.create_grid_like(mc_grid)
#                        pre_snug_base_grid.clear_grid()
#                        pre_snug_base_grid.update_grid(
#                            acc_out0["sdf_base"],
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
                        avatar_out0 = self.__inference_full_vals(
                            avatar_data,
                            key,
                            batch_size=batch_size,
                        )

                        snug_field = self.build_avatar_snug_scale_field(
                            acc_sdf=acc_out0['sdf_base'],
                            avatar_sdf=avatar_out0['sdf'],
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
                            cache_key = item.get("cache_as", accessory_key)
                            if cache_key in adapted_support_cache:
                                acc_coords = accessory_data["coords"]
                                adapted_support_cache[cache_key] = {
                                    "coords": acc_coords.copy(),
                                    "points": accessory_data["runtime_points"].copy(),
                                    "frame": accessory_data["runtime_frame"].copy(),
                                    "radius": accessory_data["radius"].copy(),
                                    "x_radius": accessory_data["x_radius"].copy(),
                                    "assembly_scale": root_scale,
                                }


                acc_out = self.__inference_full_vals(
                    accessory_data,
                    accessory_key,
                    batch_size=batch_size,
                )

                acc_vals = acc_out["sdf"]
                acc_vals_base = acc_out["sdf_base"]
                acc_vals_detail = acc_out["sdf_detail"]

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
                    avatar_out = self.__inference_full_vals(
                        avatar_data,
                        key,
                        batch_size=batch_size,
                    )

                    # Use avatar FINAL as obstacle.
                    avatar_sdf_for_offset = avatar_out["sdf"]

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

                mesh_acc_base_fit = acc_grid_base_fit.extract_mesh()
                if len(mesh_acc_base_fit.faces) > 0:
                    parts = mesh_acc_base_fit.split(only_watertight=False)
                    if len(parts) > 0:
                        mesh_acc_base_fit = max(parts, key=lambda m: len(m.faces))
                    mesh_acc_base_fit.export(
                        op.join(
                            output_folder,
                            f"{cc}_{mode}_basefit_{accessory_key.replace('|','_')}.ply"
                        )
                    )



                # 3) Debug/export this individual accessory after offset/cut/detail-reapply
                acc_grid = utils.create_grid_like(mc_grid)
                acc_grid.clear_grid()
                acc_grid.update_grid(vals_final, kidx, mark=True, mode="overwrite")

                # raw base debug
                acc_grid_base.update_grid(acc_vals_base, kidx, mark=True, mode="overwrite")


                mesh_acc = acc_grid.extract_mesh()
                if len(mesh_acc.faces) > 0:
                    parts = mesh_acc.split(only_watertight=False)
                    if len(parts) > 0:
                        mesh_acc = max(parts, key=lambda m: len(m.faces))
                    mesh_acc.export(op.join(output_folder, f"{cc}_{mode}_{accessory_key.replace('|','_')}.ply"))

                mesh_acc_base = acc_grid_base.extract_mesh()
                if len(mesh_acc_base.faces) > 0:
                    parts = mesh_acc_base.split(only_watertight=False)
                    if len(parts) > 0:
                        mesh_acc_base = max(parts, key=lambda m: len(m.faces))
                    mesh_acc_base.export(op.join(output_folder, f"{cc}_{mode}_base_{accessory_key.replace('|','_')}.ply"))

                cc += 1
                #mc_grid.update_grid(acc_vals - delta, kidx, mode='minimum')
                #mc_grid.update_grid(acc_vals, kidx, mode='minimum')
                # 4) Merge all accessories by hard union
                mc_grid.update_grid(vals_final, kidx, mode="minimum")


        mesh = mc_grid.extract_mesh1()
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
                vals, vals_base = self.__mix_inference(curve_data, mix_arg, batch_size)
            else:
                curve_data, kidx = curve.filter_grid(mc_grid)
                vals, vals_base = self.__inference_vals(curve_data, key, batch_size)

            mc_grid.update_grid(vals, kidx, mode='minimum')

        mesh = mc_grid.extract_mesh()
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
            
            vals = self.__inference_vals(curve_data, key, batch_size)
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
            
            vals = self.__inference_vals(curve_data, key, batch_size)
            mc_grid.update_grid(vals, kidx, mode='minimum')

        t0 = time()
        mesh = mc_grid.extract_mesh()
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
                
                vals = self.__inference_vals(curve_data, key)
                # overwrite cylinder SDF, take min with other curve part
                mc_grid.update_grid_func(vals, kidx, func=np.minimum)

            key = self.encode_key(new_shape_name, new_curve.name)
            curve_data, new_kidx = new_curve.filter_grid(mc_grid)
            new_vals = self.__inference_vals(curve_data, key)
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

        mesh = mc_grid.extract_mesh()
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
            
            vals = self.__inference_vals(curve_data, key)
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
            new_vals = self.__inference_vals(curve_data, key)
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

        mesh = mc_grid.extract_mesh()
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
            
            vals = self.__inference_vals(curve_data, key)
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
                vals = self.__inference_vals(curve_data, key)
                # overwrite cylinder SDF, take min with other curve part
                vals = np.minimum(vals, sdf_val2[inside])
                sdf_val2[inside] = vals

        sdf_val2 = smooth.max(sdf_val2, sdf_ball)
        sdf_val1 = np.minimum(sdf_val1, sdf_val2)
        mc_grid.update_grid(sdf_val1, ball_kidx, mode='overwrite')

        key = self.encode_key(new_shape_name, curve2.name)
        curve_data, new_kidx = curve2.filter_grid(mc_grid)
        new_vals = self.__inference_vals(curve_data, key)

        # calculate intersection of ball and new part cylinder
        new_grid = utils.create_grid_like(mc_grid)
        new_grid.update_grid_func(sdf_val1, ball_kidx, func=None)
        area_marks = mc_grid.func_marks[new_kidx]
        area_kidx = new_kidx[area_marks]
        new_grid.update_grid_func(new_vals, new_kidx, func=smooth.min)
        area_vals = new_grid.get_vals(area_kidx)

        mc_grid.update_grid(new_vals, new_kidx, mode='minimum')
        mc_grid.update_grid(area_vals, area_kidx, mode='overwrite')
        mesh = mc_grid.extract_mesh()
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
            
            vals = self.__inference_vals(curve_data, key)
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
                vals = self.__inference_vals(curve_data, key)
                # overwrite cylinder SDF, take min with other curve part
                vals = np.minimum(vals, sdf_val2[inside])
                sdf_val2[inside] = vals
        
        sdf_val2 = smooth.max(sdf_val2, sdf_ball)
        sdf_val1 = np.minimum(sdf_val1, sdf_val2)
        mc_grid.update_grid(sdf_val1, ball_kidx, mode='overwrite')

        key = self.encode_key(shape_name, curve_new.name)
        curve_data, new_kidx = curve_new.filter_grid(mc_grid)
        new_vals = self.__inference_vals(curve_data, key)

        # calculate intersection of ball and new part cylinder
        new_grid = utils.create_grid_like(mc_grid)
        new_grid.update_grid_func(sdf_val1, ball_kidx, func=None)
        area_marks = mc_grid.func_marks[new_kidx]
        area_kidx = new_kidx[area_marks]
        new_grid.update_grid_func(new_vals, new_kidx, func=smooth.min)
        area_vals = new_grid.get_vals(area_kidx)

        mc_grid.update_grid(new_vals, new_kidx, mode='minimum')
        mc_grid.update_grid(area_vals, area_kidx, mode='overwrite')
        mesh = mc_grid.extract_mesh()
        out_name = '{}|{}_{}.ply'.format(
            shape_name, part_name, arg['exp_name']
        )
        self.output_mesh(mesh, out_name, arg)
