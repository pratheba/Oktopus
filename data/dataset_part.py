import os, sys, pickle
import torch
import numpy as np
import trimesh

import os.path as op
from torch.utils.data import Dataset
from ngc import Handle

class NGCDataset(Dataset):
    """
    docstring for NGCDataset.

    """
    def __init__(self, arg):
        super(NGCDataset, self).__init__()
        self.root_path = arg['root']

        self.mode = arg['mode']
        self.n_sample = arg['n_sample'] # default 1024
        self.n_surface_sample = self.n_sample * 2 #arg['n_surface_sample'] # default 2048

        self.data_names = np.loadtxt(
            op.join(self.root_path, 'data.txt'), dtype=str).tolist()
        
        if 'shape_name' in arg:
            self.data_names = [str(arg['shape_name'])]

        self.file_name = 'sdf_samples.pkl'

        self.handles = self.load_handles()
        self.inputs = self.load_inputs()

        if self.mode == 'train':
            self.data_path = 'train_data'

        if self.mode == 'val':
            self.data_path = 'val_data'

        if self.mode == 'all':
            self.data_path = 'all_data'

        if self.mode == 'inference':
            self.data_path = 'val_data'

    
    def load_handles(self):
        handles = []
        for name in self.data_names:
            item_path = op.join(self.root_path, name)
            #handle_path = op.join(item_path, 'handle', 'std_handle.pkl')
            handle_path = op.join(item_path, 'handle', 'std_handle.npz')
            handle = Handle()
            handle.load(handle_path)
            handles.append(handle)

        return handles
    
    def load_inputs(self):
        inputs = []
        curve_nums = [handle.num_curve for handle in self.handles]
        print('Total num of curves:{}, {} shapes'.format(
            sum(curve_nums), len(curve_nums)
        ))
        curve_nums.insert(0, 0)
        idx_range = np.cumsum(curve_nums)
        hid = 0

        for name in self.data_names:
            start, end = idx_range[hid], idx_range[hid+1]
            inputs.append(np.arange(start, end))
            hid += 1
            
        return inputs


    def __len__(self):
        return len(self.data_names)
    
    def __getitem__(self, idx):
        name = self.data_names[idx]
        item_path = op.join(self.root_path, name)

        data_path = op.join(item_path, self.data_path, self.file_name)
        #print("data path = ", data_path)

        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        input_curve_idx = self.inputs[idx]
        ### The curve IDS are indexed for all the shapes
        # i.e if first shape has 4 curves, the ids becomes [0, 1, 2, 3]
        # the next shape with 3 curves will have the global curve IDS [ 4, 5, 6]
        # the local curve ids internally can start from [0, 1, 2]

        # here input_curve_idx is the global one
        #print("input_curve_idx = ", input_curve_idx)
        #exit()

        #input1, gt1, sidx1 = self.get_curve_data(data['surface'], input_curve_idx, self.n_sample)
        #input2, gt2, sidx2 = self.get_curve_data(data['space'], input_curve_idx, self.n_sample)
        #input3, gt3, sidx3 = self.get_curve_data(data['on_surface'], input_curve_idx, self.n_surface_sample)
        input1, gt1, sidx1 = self.get_curve_data(data['pert_surface'], input_curve_idx, self.n_sample)
        input2, gt2, sidx2 = self.get_curve_data(data['space'], input_curve_idx, self.n_sample)
        input3, gt3, sidx3 = self.get_curve_data(data['on_surface'], input_curve_idx, self.n_surface_sample)
        #gt_base1 = torch.from_numpy(data['base_pert_surface_sdf'][sidx1]).float()
        #gt_base2 = torch.from_numpy(data['base_space_sdf'][sidx2]).float()
        #gt_base3 = torch.from_numpy(data['base_on_surface_sdf'][sidx3]).float()
        #gt_residual1 = torch.from_numpy(data['residual_pert_surface_sdf'][sidx1]).float()
        #gt_residual2 = torch.from_numpy(data['residual_space_sdf'][sidx2]).float()
        #gt_residual3 = torch.from_numpy(data['residual_on_surface_sdf'][sidx3]).float()
        #input_base1, gt_base1 = self.get_curve_data(data['base_surface'], input_curve_idx, self.n_sample)
        #input_base2, gt_base2 = self.get_curve_data(data['base_space'], input_curve_idx, self.n_sample)
        #input_base3, gt_base3 = self.get_curve_data(data['base_on_surface'], input_curve_idx, self.n_surface_sample)
        #input_residual1, gt_residual1 = self.get_curve_data(data['residual_surface'], input_curve_idx, self.n_sample)
        #input_residual2, gt_residual2 = self.get_curve_data(data['residual_space'], input_curve_idx, self.n_sample)
        #input_residual3, gt_residual3 = self.get_curve_data(data['residual_on_surface'], input_curve_idx, self.n_surface_sample)
        data_input = {
            'samples': torch.cat([input1['samples'], input2['samples'], input3['samples']], dim=0),
            'coords': torch.cat([input1['coords'], input2['coords'], input3['coords']]),
            'angles': torch.cat([input1['angles'], input2['angles'], input3['angles']]),
            'rho': torch.cat([input1['rho'], input2['rho'], input3['rho']]),
            'rho_n': torch.cat([input1['rho_n'], input2['rho_n'], input3['rho_n']]),
            'radius': torch.cat([input1['radius'], input2['radius'], input3['radius']]),
            'curve_idx': torch.cat([input1['curve_idx'], input2['curve_idx'], input3['curve_idx']]),
        }
        data_gt = {
            'sdf': torch.cat([gt1['sdf'], gt2['sdf'], gt3['sdf']]),
            'sdf_base': torch.cat([gt1['sdf_base'], gt2['sdf_base'], gt3['sdf_base']]),
            'sdf_res': torch.cat([gt1['sdf_res'], gt2['sdf_res'], gt3['sdf_res']]),
        }
        #exit()

        info = {}

        return data_input, data_gt, info

    def get_curve_base_residual_data(self, curve_data,sample_idx):
        samples_sdf = curve_data['sdf'][sample_idx]

    def get_curve_data(self, curve_data, input_curve_idx, n_samples=1024):
        samples_local = curve_data['samples_local']
        #samples_global = curve_data['samples_global']
        #print("sample local = ", samples_local[0:10])
        samples_coords = curve_data['coords']
        samples_angle = curve_data['angles']
        samples_radius = curve_data['radius']
        samples_rho = curve_data['rho']
        samples_rho_n = curve_data['rho_n']
        #print("samples coords = ", samples_coords.shape)
        #print("sample coords = ", samples_coords[0:10])
        #print("samples_local = ",samples_local.shape)
        #exit()
        
        samples_sdf = curve_data['sdf']
        samples_base_sdf = curve_data['sdf_base']
        samples_res_sdf = curve_data['sdf_res']
        #print(samples_sdf.shape)
        cids = curve_data['curve_idx'].astype(np.int32)
        #print("cids", cids, flush=True)
        #exit()

        num_samples = samples_local.shape[0]
        #print(f"{num_samples} AND {n_samples}", flush=True)
        if n_samples <= 0:
            sidx = np.arange(num_samples)
        else:
            n_s = n_samples
            if n_s <= num_samples:
                sidx = np.random.choice(num_samples, size=n_s, replace=False)
                #sidx = np.random.choice(num_samples, size=n_s, replace=True)
            else:
                #sidx = np.random.choice(num_samples, size=n_s, replace=True)
                raise ValueError(f'num of samples{num_samples} smaller than the threshold {n_s}')
        
        gt_sdf = samples_sdf[sidx]
        gt_base_sdf = samples_base_sdf[sidx]
        gt_res_sdf = samples_res_sdf[sidx]
        samples_local = samples_local[sidx]
        samples_coords = samples_coords[sidx]
        samples_angle = samples_angle[sidx]
        samples_radius = samples_radius[sidx]
        samples_rho = samples_rho[sidx]
        samples_rho_n = samples_rho_n[sidx]
        cids = cids[sidx]

        ### This maps the local curve_ids to global ones
        ### cids are local curve_ids
        curve_idx = input_curve_idx[cids]

        model_input = {
            'samples': torch.from_numpy(samples_local).float(),
            #'samples_global': torch.from_numpy(samples_global).float(),
            'coords': torch.from_numpy(samples_coords).float(),
            'angles': torch.from_numpy(samples_angle).float(),
            'rho': torch.from_numpy(samples_rho).float(),
            'rho_n': torch.from_numpy(samples_rho_n).float(),
            'radius': torch.from_numpy(samples_radius).float(),
            'curve_idx': torch.from_numpy(curve_idx).long(),
        }
        gt = {
            'sdf': torch.from_numpy(gt_sdf).float(),
            'sdf_base': torch.from_numpy(gt_base_sdf).float(),
            'sdf_res': torch.from_numpy(gt_res_sdf).float(),
        }
        return model_input, gt, sidx
    

