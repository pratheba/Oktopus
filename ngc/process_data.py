import os, pickle
import numpy as np
import trimesh
import os.path as op
import pymeshlab as ml
from time import time
from tqdm.autonotebook import tqdm
from handle import Handle
#from .handle import Handle
from handle_utils.visualize import visualize

def meshlab_shape_sampling(shape_path, num_samples, noise_scale):
    ms = ml.MeshSet()
    ms.load_new_mesh(shape_path)
    ms.generate_sampling_poisson_disk(samplenum=num_samples)
    mesh = ms.current_mesh()
    #print(mesh.vertex_matrix())
    #print(mesh.face_matrix())
    #trimesh.Trimesh(vertices = np.array(mesh.vertex_matrix()), faces=np.array(mesh.face_matrix())).export("test.ply")
    #exit()

    verts = mesh.vertex_matrix()
    vn = mesh.vertex_normal_matrix()
    vn /= (np.linalg.norm(vn, axis=1, keepdims=True))+1e-7

    # add noise 
    noise = np.random.normal(0, noise_scale, size=verts.shape[0])
    verts_around = verts + noise[:, None]* vn
    
    verts = np.concatenate([verts, verts_around], axis=0)
    return verts

def meshlab_volumetric_sampling(shape_path, num_samples):
    ms = ml.MeshSet()
    ms.load_new_mesh(shape_path)
    ms.generate_sampling_volumetric(
        samplesurfradius = ml.PercentageValue(0.),
        samplevolnum = num_samples,
    )
    mesh = ms.mesh(1)
    verts = mesh.vertex_matrix()
    #trimesh.Trimesh(vertices = np.array(verts)).export("test.ply")
    #exit()
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
    #print("SDF eval", flush=True)
    ms = ml.MeshSet()
    ms.load_new_mesh(shape_path)
    #print("load new mesh", flush=True)
    #print("samples = ", samples, flush=True)
    samples = np.array(samples)
    #print(samples.shape, flush=True)
    pc_mesh = ml.Mesh(vertex_matrix=samples)
    #print("pc mesh", flush=True)
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
            handle_path = op.join(item_path, 'handle')
            handle_file = op.join(handle_path, 'std_handle.pkl')
            handle_mesh_file = op.join(handle_path, 'std_mesh.ply')
            output_path = op.join(item_path, 'train_data')
            os.makedirs(output_path, exist_ok=True)
            output_file = op.join(output_path, file_name)
            print("output file = ", output_file)

            if op.exists(output_file):
                print('Exists: ', output_file)
                pbar.update(1)
                #continue

            handle = Handle()
            handle.load(handle_file)
            
            if not op.exists(handle_mesh_file):
                export_handle_data(handle, output_path, handle_path)

            # the wrapper cylindrical shape surface
            space_samples = meshlab_volumetric_sampling(
                handle_mesh_file, n_space_samples
            )

            # original shape surface
            surface_samples = meshlab_shape_sampling(
                shape_file, n_surface_samples, 0.01
            )


            #pcl = []
            #pcl.append({'type': 'points', 'vertices': surface_samples})
            #pcl.append({'type': 'points', 'vertices': space_samples})
            #visualize(pcl)
            #exit()


            # space_samples = bbox_volumetric_sampling(
            #     handle_mesh_file, n_space_samples
            # )

            surface_data = handle.prepare_samples(surface_samples)
            #print("surface data", flush=True)
            surface_sdf = meshlab_SDF_eval(shape_file, surface_data['samples'])
            surface_data['sdf'] = surface_sdf

            space_data = handle.prepare_samples(space_samples)
            space_sdf = meshlab_SDF_eval(shape_file, space_data['samples'])
            #print("done1", flush=True)
            space_data['sdf'] = space_sdf
            #print("done", flush=True)

            train_data = {
                'surface': surface_data,
                'space': space_data,
            }
            print("output_file", output_file, flush=True)

            with open(output_file, 'wb') as f:
                pickle.dump(train_data, f)

            pbar.update(1)

    print('Done')

if __name__ == "__main__":
    # np.random.seed(2024)

    #root_path = '/path/to/dataset'
    root_path = '../Pack50Dataset'
    #item_file = op.join(root_path, 'data.txt')
    arg = {
        'root_path': root_path,
        'file_name': 'sdf_samples.pkl',
        'n_surface_samples' : 30000,
        'n_space_samples' : 50000,
    }
    ngc_dataset(arg)
