import os, pickle
import numpy as np
import trimesh
import copy
import os.path as op
import pymeshlab as ml
from time import time
from tqdm.autonotebook import tqdm
#from ngc.handle import Handle
from handle_3dvec import Handle

def meshlab_shape_sampling(shape_path, num_samples, noise_scale):

    # add noise 
    verts_around = []
    ms = ml.MeshSet()
    ms.load_new_mesh(shape_path)
    on_surface_samples = int(0.8 * num_samples)
    ms.generate_sampling_poisson_disk(samplenum=on_surface_samples)
    mesh = ms.current_mesh()

    on_surface_verts = mesh.vertex_matrix()
    print("num on surface vertices = ", len(on_surface_verts))
    #verts_around = verts

    verts_around = []
    off_surface_samples = int((0.25 * num_samples)/len(noise_scale))
    

    for ns in noise_scale:
        ms = ml.MeshSet()
        ms.load_new_mesh(shape_path)
        ms.generate_sampling_poisson_disk(samplenum=off_surface_samples)
        mesh = ms.current_mesh()

        verts = mesh.vertex_matrix()
        #print("num vertices ns = ", len(verts))
        vn = mesh.vertex_normal_matrix()
        vn /= np.linalg.norm(vn, axis=1, keepdims=True)
        noise = np.random.normal(0, ns, size=verts.shape[0])
        #verts_around = np.concatenate((verts_around, verts + noise[:, None]* vn), axis=0)
        if len(verts_around):
            verts_around = np.concatenate((verts_around, verts + noise[:, None]* vn), axis=0)
        else:
            verts_around = verts + noise[:, None]*vn
        #print("verts shape = ", verts.shape)
    print("verts shape = ", verts_around.shape)
    return on_surface_verts, verts_around

def meshlab_volumetric_sampling(shape_path, num_samples):
    ms = ml.MeshSet()
    ms.load_new_mesh(shape_path)
    ms.generate_sampling_volumetric(
        samplesurfradius = ml.PercentageValue(0.),
        samplevolnum = num_samples,
    )
    mesh = ms.mesh(1)
    verts = mesh.vertex_matrix()
    return verts

def bbox_volumetric_sampling(shape_path, num_samples):
    mesh = trimesh.load(shape_path, process=False)
    V = np.asarray(mesh.vertices)
    bmin, bmax = V.min(axis=0), V.max(axis=0)
    scales = bmax - bmin

    # [0,1]^3
    samples = np.random.rand(num_samples, 3)
    samples *= scales[None,:]
    samples += bmin
    return samples


def meshlab_SDF_eval(shape_path, samples):
    ms = ml.MeshSet()
    ms.load_new_mesh(shape_path)
    pc_mesh = ml.Mesh(samples)
    ms.add_mesh(pc_mesh, 'pc')

    ms.compute_scalar_by_distance_from_another_mesh_per_vertex()
    pc_mesh = ms.mesh(1)

    sdf_vals = pc_mesh.vertex_scalar_array()
    return sdf_vals

def export_handle_data(handle, graph_path, handle_path):
    handle.export_skeleton_mesh(handle_path)
    graph_data = handle.export_neural_graph()
    with open(op.join(graph_path, 'graph.pkl'), 'wb') as f:
        pickle.dump(graph_data, f)


def split_train_test(num_surface, num_space, num_on_surface):

    split = 0.8

    surface_ids = np.arange(num_surface)
    on_surface_ids = np.arange(num_on_surface)
    space_ids = np.arange(num_space)

    np.random.shuffle(surface_ids)
    np.random.shuffle(space_ids)
    np.random.shuffle(on_surface_ids)

    num_train_surface = int(split * num_surface)
    num_train_on_surface = int(split * num_on_surface)
    num_train_space = int(split * num_space)

    surface_train_ids = surface_ids[0:num_train_surface]
    on_surface_train_ids = on_surface_ids[0:num_train_on_surface]
    space_train_ids = space_ids[0:num_train_space]

    surface_val_ids = surface_ids[num_train_surface:]
    on_surface_val_ids = on_surface_ids[num_train_on_surface:]
    space_val_ids = space_ids[num_train_space:]

    return surface_train_ids, surface_val_ids, on_surface_train_ids, on_surface_val_ids, space_train_ids, space_val_ids

def get_samples(mesh_filei, n_surface_samples):
    on_surface_samples, surface_samples = meshlab_shape_sampling(
        mesh_file, n_surface_samples, [0.002, 0.005, 0.01, 0.02, 0.05]
    )

    space_samples = meshlab_volumetric_sampling(
        handle_mesh_file, n_space_samples
    )
    ########### ON SURFACE DATA ###################################
    on_surface_data = handle.prepare_samples(on_surface_samples)
    on_surface_sdf = np.zeros(on_surface_data['samples'].shape[0])
    on_surface_data['sdf'] = on_surface_sdf
    on_surface_base_sdf = meshlab_SDF_eval(shape_base_file, on_surface_data['samples'])
    on_surface_residual_sdf = on_surface_sdf - on_surface_base_sdf
    #on_surface_base_sdf = 
    #on_surface_base_data['sdf'] = on_surface_base_sdf
    #on_surface_residual_data['sdf'] = on_surface_sdf - on_surface_base_sdf
    #on_surface_base_data = copy.deepcopy(on_surface_data)

    ########### OFF SURFACE DATA ###################################
    surface_data = handle.prepare_samples(surface_samples)
    surface_base_data = copy.deepcopy(surface_data)
    surface_residual_data = copy.deepcopy(surface_data)

    surface_sdf = meshlab_SDF_eval(shape_file, surface_data['samples'])
    surface_data['sdf'] = surface_sdf
    surface_base_sdf = meshlab_SDF_eval(shape_base_file, surface_data['samples'])
    surface_residual_sdf = surface_sdf - surface_base_sdf
    #surface_base_data['sdf'] = surface_base_sdf
    #surface_residual_data['sdf'] = surface_sdf - surface_base_sdf
    #surface_base_data = copy.deepcopy(surface_data)

    ########### SPACE DATA ###################################
    space_data = handle.prepare_samples(space_samples)
    #space_base_data = copy.deepcopy(space_data)
    #space_residual_data = copy.deepcopy(space_data)

    space_sdf = meshlab_SDF_eval(shape_file, space_data['samples'])
    space_data['sdf'] = space_sdf
    space_base_sdf = meshlab_SDF_eval(shape_base_file, space_data['samples'])
    space_residual_sdf = space_sdf - space_base_sdf

def ngc_dataset(arg):
    root_path = arg['root_path']
    file_name = arg['file_name']
    n_surface_samples = arg['n_surface_samples']
    n_space_samples = arg['n_space_samples']

    # items = os.listdir(root_path)
    items = np.loadtxt(
        op.join(root_path, 'data.txt'), dtype=str).tolist()

    with tqdm(total=len(items)) as pbar:
        for name in items:
            item_path = op.join(root_path, f'{name}')
            shape_file = op.join(item_path, 'mesh.ply')
            shape_base_file = op.join(item_path, 'mesh_base.ply')
            #residual_shape_file = get_residual_mesh(shape_file, base_shape_file)
            handle_path = op.join(item_path, 'handle')
            handle_file = op.join(handle_path, 'std_handle.pkl')
            handle_mesh_file = op.join(handle_path, 'std_mesh.ply')
            output_train_path = op.join(item_path, 'train_data')
            output_val_path = op.join(item_path, 'val_data')
            output_all_path = op.join(item_path, 'all_data')
            os.makedirs(output_train_path, exist_ok=True)
            os.makedirs(output_val_path, exist_ok=True)
            os.makedirs(output_all_path, exist_ok=True)

            output_train_file = op.join(output_train_path, file_name)
            output_val_file = op.join(output_val_path, file_name)
            output_all_file = op.join(output_all_path, file_name)

            #if op.exists(output_train_file):
            #    print('Exists: ', item_path)
            #    pbar.update(1)
            #    #continue

            handle = Handle()
            handle.load(handle_file)
            
            if not op.exists(handle_mesh_file):
                export_handle_data(handle, output_path, handle_path)

            #on_surface_samples, surface_samples = meshlab_shape_sampling(
            #    shape_file, n_surface_samples, [0.002, 0.005, 0.01, 0.02, 0.05]
            #)
            space_samples = meshlab_volumetric_sampling(
                handle_mesh_file, n_space_samples
            )
            on_surface_base_samples, surface_base_samples = meshlab_shape_sampling(
                shape_base_file, n_surface_samples, [0.002, 0.005, 0.01, 0.02, 0.05]
            )
            ########### ON SURFACE DATA ###################################
            #on_surface_data = handle.prepare_samples(on_surface_samples)
            #on_surface_sdf = np.zeros(on_surface_data['samples'].shape[0])
            #on_surface_data['sdf'] = on_surface_sdf
            on_surface_base_data = handle.prepare_samples(on_surface_base_samples)
            on_surface_base_sdf = np.zeros(on_surface_base_data['samples'].shape[0])
            on_surface_base_data['sdf'] = on_surface_base_sdf

            #on_surface_base_sdf = meshlab_SDF_eval(shape_base_file, on_surface_data['samples'])
            on_surface_sdf = meshlab_SDF_eval(shape_file, on_surface_base_data['samples'])
            #on_surface_data['sdf'] = on_surface_sdf
            on_surface_residual_sdf = on_surface_sdf - on_surface_base_sdf
            #on_surface_base_sdf = 
            #on_surface_base_data['sdf'] = on_surface_base_sdf
            #on_surface_residual_data['sdf'] = on_surface_sdf - on_surface_base_sdf
            #on_surface_base_data = copy.deepcopy(on_surface_data)

            ########### OFF SURFACE DATA ###################################
            surface_base_data = handle.prepare_samples(surface_base_samples)
            #surface_data = handle.prepare_samples(surface_samples)

            #surface_sdf = meshlab_SDF_eval(shape_file, surface_data['samples'])
            surface_sdf = meshlab_SDF_eval(shape_file, surface_base_data['samples'])
            #surface_data['sdf'] = surface_sdf
            surface_base_sdf = meshlab_SDF_eval(shape_base_file, surface_base_data['samples'])
            surface_residual_sdf = surface_sdf - surface_base_sdf
            surface_base_data['sdf'] = surface_base_sdf
            #surface_residual_data['sdf'] = surface_sdf - surface_base_sdf
            #surface_base_data = copy.deepcopy(surface_data)

            ########### SPACE DATA ###################################
            space_data = handle.prepare_samples(space_samples)
            #space_base_data = copy.deepcopy(space_data)
            #space_residual_data = copy.deepcopy(space_data)

            space_sdf = meshlab_SDF_eval(shape_file, space_data['samples'])
            #space_data['sdf'] = space_sdf
            space_base_sdf = meshlab_SDF_eval(shape_base_file, space_data['samples'])
            space_residual_sdf = space_sdf - space_base_sdf
            space_base_data['sdf'] = space_base_sdf
            #space_residual_data['sdf'] = space_sdf - space_base_sdf
            #space_base_data = copy.deepcopy(space_data)

#            surface_train_ids, surface_val_ids, \
#                    on_surface_train_ids, on_surface_val_ids, \
#                    space_train_ids, space_val_ids = split_train_test(surface_data['sdf'].shape[0], space_data['sdf'].shape[0], on_surface_data['sdf'].shape[0])
#
#            keys = surface_data.keys()
#
#            train_surface_data = {key:[] for key in keys}
#            val_surface_data = {key:[] for key in keys}
#            train_on_surface_data = {key:[] for key in keys}
#            val_on_surface_data = {key:[] for key in keys}
#            train_space_data = {key:[] for key in keys}
#            val_space_data = {key:[] for key in keys}
#
#
#            ### the whole shape
#            for key in surface_data.keys():
#                if 'part' in key:
#                    continue
#                train_surface_data[key] = surface_data[key][surface_train_ids]
#                val_surface_data[key] = surface_data[key][surface_val_ids]
#
#                train_on_surface_data[key] = on_surface_data[key][on_surface_train_ids]
#                val_on_surface_data[key] = on_surface_data[key][on_surface_val_ids]
#
#                train_space_data[key] = space_data[key][space_train_ids]
#                val_space_data[key] = space_data[key][space_val_ids]


            #### If we go by parts, then for each part need to split to train and test and do subsampling

#            train_data = {
#                'surface': train_surface_data,
#                'space': train_space_data,
#                'on_surface': train_on_surface_data
#            }
#            val_data = {
#                'surface': val_surface_data,
#                'space': val_space_data,
#                'on_surface': val_on_surface_data
#            }
            all_data = {
                'base_surface_sdf': surface_base_sdf,
                'base_space_sdf': space_base_sdf,
                'base_on_surface_sdf': on_surface_base_sdf,
                'residual_surface_sdf': surface_residual_sdf,
                'residual_space_sdf': space_residual_sdf,
                'residual_on_surface_sdf': on_surface_residual_sdf,
                'surface': surface_base_data,
                'space': space_base_data,
                'on_surface': on_surface_base_data
            }

            with open(output_train_file, 'wb') as f:
                pickle.dump(train_data, f)

            with open(output_val_file, 'wb') as f:
                pickle.dump(val_data, f)

            with open(output_all_file, 'wb') as f:
                pickle.dump(all_data, f)

            pbar.update(1)

    print('Done')

if __name__ == "__main__":
    # np.random.seed(2024)

    #root_path = '/path/to/dataset'
    root_path = '../Pack50Dataset'
    arg = {
        'root_path': root_path,
        'file_name': 'sdf_samples.pkl',
        'n_surface_samples' : 3200000,
        'n_space_samples' : 400000,
    }
    ngc_dataset(arg)
