import numpy as np
import sys
import os
import trimesh


def rescale_to_unitBB_trimesh(input_mesh, output_folder, fname):
    mesh = trimesh.load(input_mesh)
    rescale = max(mesh.extents)/2
    tform = [-(mesh.bounds[1][i] + mesh.bounds[0][i])/2.0 for i in range(3)]
    matrix = np.eye(4)
    matrix[:3, 3] = tform
    mesh.apply_transform(matrix)

    matrix = np.eye(4)
    matrix[:3, :3] /= rescale
    mesh.apply_transform(matrix)
    
    mesh.export(os.path.join(output_folder, fname+'.ply'))

def rescale_to_unitBB(input_mesh, output_folder, fname):
    mesh = trimesh.load(input_mesh)
    vertices = mesh.vertices
    maxdim = np.max(vertices, axis=0)
    mindim = np.min(vertices, axis=0)
    diff = (maxdim - mindim )
    center = (maxdim + mindim )/2
    centered_vertices = vertices - center

    radius = np.max(diff)/2.0
    normalized_vertices = centered_vertices / radius

    os.makedirs(output_folder, exist_ok=True)
    outmesh = os.path.join(output_folder, fname+'.ply')
    trimesh.Trimesh(vertices = normalized_vertices, faces = mesh.faces, process=False).export(outmesh)
    outmesh = os.path.join(output_folder, fname+'.off')
    trimesh.Trimesh(vertices = normalized_vertices, faces = mesh.faces, process=False).export(outmesh)

if __name__ == '__main__':
    rescale_to_unitBB(*sys.argv[1:])
