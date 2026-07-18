import numpy as np
import polyscope as ps

def mesh_viz(vertices, faces, name="mesh"):
    ps.init()
    vertices = np.vstack((vertices))
    faces = np.vstack((faces))
    ps_mesh = ps.register_surface_mesh(name, np.array(vertices), np.array(faces))
    ps.show()


## Defined as p
def pointcloud_viz(vertices, name="pointcloud", standalone=True):
    if standalone:
        ps.init()

    pcl = ps.register_point_cloud(name, vertices)
    if standalone:
        ps.show()
    else:
        return pcl


## Defined as c
def curve_viz(nodes, edges = None, name="curve", standalone=True):
    if standalone:
        ps.init()

    if edges is None:
        edges = np.vstack((np.arange(len(nodes)-1), np.arange(1, len(nodes)))).T

    src_curve = ps.register_curve_network(name, nodes, edges)

    if standalone:
        ps.show()
    else:
        return src_curve



def visualize(list_objects, name="list"):
    ps.init()
    ps_group = ps.create_group(name)
    count = 1

    for obj in list_objects:
        if obj['type'] == 'mesh':
            assert(obj['vertices'])
            assert(obj['faces'])
            ps_mesh = mesh_viz(obj['vertices'], obj['faces'], str(count), False)
            ps_mesh.add_to_group(ps_group)
            count += 1
        elif obj['type'] == 'points':
            assert(obj['vertices'] is not None)
            ps_pointcloud = pointcloud_viz(obj['vertices'], str(count), False)
            ps_pointcloud.add_to_group(ps_group)
            count += 1
        elif obj['type'] == 'curve':
            assert(obj['nodes'] is not None)
            ps_curve = curve_viz(obj['nodes'], obj['edges'], str(count), False)
            ps_curve.add_to_group(ps_group)
            count += 1
    ps.show()
    ps.remove_group(ps_group)



