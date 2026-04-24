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
            item_path = op.join(data_root, f'{shape_name}')
            #handle_path = op.join(item_path, 'handle/std_handle.pkl')
            handle_path = op.join(item_path, 'handle/std_handle.npz')
            #handle = utils.load_handle(handle_path)
            handle = self.load_shape_with_npz(data_root, shape_name)
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

    def load_shape_with_npz(self, data_root, shape_name):
        item_path = op.join(data_root, f'{shape_name}')
        handle_path = op.join(item_path, 'handle/std_handle.npz')
        handle = utils.load_handle(handle_path)

        npz_path = op.join(item_path, f'all_data/inference.npz')
        if op.exists(npz_path):
            self.apply_curve_state_npz(handle, npz_path, shape_name)

        return handle

    def load_shape_handle(self, data_root, shape_name):
        return self.load_shape_with_npz(data_root, shape_name)
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
            #if "key_cylinder_radius" in s:
            #    core.key_cylinder_radius = np.asarray(s["key_cylinder_radius"], dtype=np.float64)
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


    @torch.no_grad()
    def action_part_adapt(self, arg):
        output_folder = arg['output_folder']
        exp_name = arg['exp_name']
        mc_grid = arg['mc_grid']
        shape_name = arg['shape']
        config = utils.load_yaml_file(arg['adapt_file'])

        data_root = arg['data_root']
        handle = self.load_shape_handle(data_root, shape_name)
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
                    'infer_scale': 1.35,
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
                elif mode == 'dependent_split':
                    parent_accessory_key = adapt_arg['parent_accessory_key']
                    parent_support_key = adapt_arg.get('parent_support_key', parent_accessory_key)

                    if parent_support_key not in adapted_support_cache:
                        raise ValueError(f"Missing parent cached support: {parent_support_key}")

                    adapt_arg['parent_support_data'] = adapted_support_cache[parent_support_key]

                    split_s = float(adapt_arg['split_t_src_0'])
                    child_s0 = float(adapt_arg['split_t_src_0'])
                    child_s1 = float(adapt_arg['split_t_src_1'])

                    adapt_arg['dep_template'] = self.split_template_from_key(
                        accessory_key=accessory_key,
                        split_s=split_s,
                        child_s0=child_s0,
                        child_s1=child_s1,
                    )

                    accessory_data, runtime_child_support, kidx = self.filter_grid_dependent_runtime(
                        curve_grid,
                        adapt_arg,
                    )

                    cache_key = item.get("cache_as", f"{accessory_key}_split")
                    adapted_support_cache[cache_key] = {
                        "coords": runtime_child_support["coords"].copy(),
                        "points": runtime_child_support["points"].copy(),
                        "frame": runtime_child_support["frame"].copy(),
                        "radius": runtime_child_support["radius"].copy(),
                        "x_radius": runtime_child_support.get("x_radius", np.ones_like(runtime_child_support["coords"])).copy(),
                        "assembly_scale": runtime_child_support.get(
                            "assembly_scale",
                            adapt_arg['parent_support_data'].get("assembly_scale", 1.0)
                        ),
                    }


                elif mode == 'dependent':
                    parent_accessory_key = adapt_arg['parent_accessory_key']
                    if parent_accessory_key not in adapted_support_cache:
                        raise ValueError(f"Missing parent cached support: {parent_accessory_key}")

                    adapt_arg['parent_support_data'] = adapted_support_cache[parent_accessory_key]
                    #adapt_arg['dep_template'] = self.dependent_template_from_key(accessory_key)
                    parent_joint_s = float(adapt_arg['parent_joint_s'])
                    child_joint_s = float(adapt_arg.get('child_joint_s', 0.0))
                    child_s0 = float(adapt_arg.get("child_s0", 0.0))
                    child_s1 = float(adapt_arg.get("child_s1", 1.0))

                    adapt_arg['dep_template'] = self.attached_template_from_keys(
                        parent_key=parent_accessory_key,
                        child_key=accessory_key,
                        parent_joint_s=parent_joint_s,
                        child_joint_s=child_joint_s,
                        child_s0=child_s0,
                        child_s1=child_s1
                    )
                    curve_grid = utils.create_grid_like(mc_grid, res=256)
                    #curve_grid.clear_grid(val=10.0)
                    curve_grid.clear_grid()

                    accessory_data, runtime_child_support, kidx = self.filter_grid_dependent_runtime(curve_grid, adapt_arg)

                    cache_key = item.get("cache_as", accessory_key)
                    adapted_support_cache[cache_key] = {
                        "coords": runtime_child_support["coords"].copy(),
                        "points": runtime_child_support["points"].copy(),
                        "frame": runtime_child_support["frame"].copy(),
                        "radius": runtime_child_support["radius"].copy(),
                        "x_radius": runtime_child_support.get("x_radius", np.ones_like(runtime_child_support["coords"])).copy(),
                    }

                else:
                    raise ValueError(f"Unknown adapt mode: {mode}")

                acc_vals, acc_vals_base = self.__inference_vals(
                    accessory_data, accessory_key, batch_size=batch_size
                )

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
                mc_grid.update_grid(acc_vals, kidx, mode='minimum')

        mesh = mc_grid.extract_mesh1()
        mesh_file = op.join(output_folder, f'{out_name}.ply')
        os.makedirs(op.dirname(mesh_file), exist_ok=True)
        mesh.export(mesh_file)


    @torch.no_grad()
    def action_part_adapt1(self, arg):
        output_folder = arg['output_folder']
        exp_name = arg['exp_name']
        mc_grid = arg['mc_grid']
        shape_name = arg['shape']
        config = utils.load_yaml_file(arg['adapt_file'])

        data_root = arg['data_root']
        handle = self.load_shape_handle(data_root, shape_name)
        # out_name = exp_name
        out_name = f'{shape_name}_{exp_name}'

        batch_size = 64**3
        count = 0
        acc_grid = utils.create_grid_like(mc_grid)
        #acc_grid.clear_grid(val=10.0)
        #mc_grid.clear_grid(val=10.0)
        acc_grid.clear_grid()
        mc_grid.clear_grid()
        
        cc = 0 
        adapted_support_cache = {}
        for item in config:
            target_key = item['target_key']
            mode = item.get('mode', 'direct')
            accessory_key = item['accessory_key']
            for curve in handle.curves:
                key = self.encode_key(shape_name, curve.name)
                print(key)
                if key != target_key:
                    continue
                count += 1
                curve_grid = utils.create_grid_like(mc_grid)
                #curve_grid.clear_grid(val=10.0)
                curve_grid.clear_grid()

                adapt_arg = {
                    #'accessory_curve_handle': accessory_curve_handle,
                    'avatar_curve_handle': curve,
                    'device': self.device,
                    'infer_scale': 1.35,
                    'avatar_curve_idx': self.feat_dict[key],
                    'accessory_curve_idx': self.feat_dict[accessory_key],
                }
                adapt_arg.update(item)

                if mode == 'direct':
                    accessory_curve_handle = self.curve_from_key(accessory_key)
                    adapt_arg['accessory_curve_handle'] = accessory_curve_handle
                    accessory_data, avatar_data, kidx, inside = curve.filter_grid_adapt(curve_grid, adapt_arg)
                    adapted_support_cache[accessory_key] = {
                        "coords": accessory_data["coords"].copy(),
                        "points": accessory_curve_handle.core.interpolate(accessory_data["coords"])["points"].copy(),
                        "frame": accessory_data["frame"].copy(),
                        "radius": accessory_data["radius"].copy(),
                        "x_radius": accessory_curve_handle.core.calc_x_radius(accessory_data["coords"]).copy(),
                    }
                elif mode == 'dependent':
                    adapt_arg['dep_template'] = self.dependent_template_from_key(accessory_key)

                    # if ever needed for wrap radius on dependent support
                    #adapt_arg['target_core_for_wrap'] = self.template_core_from_key(accessory_key)

                    accessory_data, avatar_data, kidx, inside = curve.filter_grid_adapt(curve_grid, adapt_arg)

                    adapted_support_cache[accessory_key] = {
                        "coords": accessory_data["coords"].copy(),
                        "points": accessory_data["points"].copy(),
                        "frame": accessory_data["frame"].copy(),
                        "radius": accessory_data["radius"].copy(),
                        "x_radius": accessory_data.get("x_radius", np.ones_like(accessory_data["coords"])).copy(),
                    }
                acc_vals, acc_vals_base = self.__inference_vals(accessory_data, accessory_key, batch_size=batch_size)
                    #avatar_vals, avatar_vals_base = self.__inference_vals(avatar_data, key, batch_size=batch_size)
                    #vals, vals_base = self.__mix_inference(curve_data, adapt_arg, batch_size)
                    #vals = np.minimum(avatar_vals, acc_vals - 2e-3)
    #                vals = acc_vals
                delta = 0.01  # boot outward
                #gap   = 0.01  # required clearance above leg

                acc_grid = utils.create_grid_like(mc_grid)
                #acc_grid.clear_grid(val=10.0)
                acc_grid.clear_grid()
                vals = acc_vals - delta
                acc_grid.update_grid(vals, kidx, mark=True, mode="overwrite")
                mesh_acc = acc_grid.extract_mesh()
                mesh_acc = max(mesh_acc.split(only_watertight=False), key=lambda m: len(m.faces))
                mesh_acc.export(op.join(output_folder, f"{cc}_acc_vals.ply"))
                cc += 1
                mc_grid.update_grid(vals, kidx, mode='minimum')

                #else:
                #    curve_data, kidx = curve.filter_grid(curve_grid)
                #    vals, vals_base = self.__inference_vals(curve_data, key, batch_size)
                #    mc_grid.update_grid(vals, kidx, mode='minimum')

                #print(np.max(vals), np.min(vals))
                count += 1

                #mc_grid.update_grid(vals, kidx, mode='minimum')

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
