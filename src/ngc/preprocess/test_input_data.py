import numpy as np
from skeleton_utils import visualize


handle = np.load('skeleton_data/horse_std_handle.pkl.npy', allow_pickle=True)

for seg in handle:
    key_points = seg['key_points']
    x_axis = seg['x_axis']
    y_axis = seg['y_axis']
    z_axis = seg['z_axis']
    nodes = np.concatenate((key_points, x_axis, y_axis, z_axis))
    edge_x = np.vstack((np.arange(key_points.shape[0]), np.arange(key_points.shape[0], key_points.shape[0]+x_axis.shape[0]))).T
    next_index = key_points.shape[0] + x_axis.shape[0]
    print(edge_x)
    edge_y = np.vstack((np.arange(key_points.shape[0]), np.arange(next_index, next_index+y_axis.shape[0]))).T
    print(edge_y)
    next_index = key_points.shape[0] + x_axis.shape[0] + y_axis.shape[0]
    edge_z = np.vstack((np.arange(key_points.shape[0]), np.arange(next_index, next_index +z_axis.shape[0]))).T
    print(edge_z)
    edges = np.concatenate((edge_x, edge_y, edge_z))

    kp = {'type': 'points', 'vertices': seg['key_points']}
    axis1 = {'type': 'curve', 'nodes': nodes, 'edges': edge_x}
    axis2 = {'type': 'curve', 'nodes': nodes, 'edges': edge_y}
    axis3 = {'type': 'curve', 'nodes': nodes, 'edges': edge_z}
    vizobj = [kp, axis1, axis2, axis3]
    visualize.visualize(vizobj)

