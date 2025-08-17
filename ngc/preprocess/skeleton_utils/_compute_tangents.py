import numpy as np
import os
import argparse

def compute_tangent_for_corner_points(corner_points):
    #corner_points = np.load(corner_points_file, allow_pickle=True)
    if corner_points is None:
        log.info("Corner vector error. Please verify")
        return
    tangent = np.array([c_point['id'] - c_point['end'] for c_point in corner_points])
    tangent = np.squeeze(tangent)
    norm_tangent = tangent / np.linalg.norm(tangent, axis=1)[:, np.newaxis]
    point_tangent_dict = {}
    for idx,c_point in enumerate(corner_points):
        point_tangent_dict[c_point['id'].tobytes()] = {'id': c_point['id'], 'tangent':norm_tangent[idx]}

    return {'cornerpoint_tangent': point_tangent_dict}


def compute_yzaxis(segment_tangents):
    y_axis = np.array([0, 1, 0])
    z_axis = np.cross(segment_tangents, y_axis)
    z_axis /= (1e-7 + np.linalg.norm(z_axis, axis=1))[:, np.newaxis]
    y_axis = np.cross(z_axis, segment_tangents)
    y_axis /= (1e-7 + np.linalg.norm(y_axis, axis=1))[:, np.newaxis]
    return y_axis, z_axis

def compute_keypoint_tangent(segment):
    segment = np.array(segment)
    v1 = segment[1:]
    v2 = segment[:-1]

    edges = np.array(v2 - v1).astype(float)
    #tangents = np.squeeze(tangents)
    #norm_tangent = tangents / (1e-7+np.linalg.norm(tangents, axis=1)[:, np.newaxis])
    print("edges =", len(edges))
    edge_vector = edges /(1e-7+np.linalg.norm(edges, axis=1, keepdims=True))

    # estimate tangent for each keyframe 
    if edge_tangent.shape[0] > 1:
        tangent_start = edge_vector[0]
        tangent_end = edge_vector[-1]
        tangent = (edge_vector[1:] + edge_vector[:-1]) / 2.
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        vert_tangent = np.concatenate([
            tangent_start.reshape(1,3),
            tangent,
            tangent_end.reshape(1,3)
            ], axis=0)
    else:
        vert_tangent = np.tile(edge_vector, (2,1))
    return vert_tangent

#def compute_keyframexyz(corner_segments_file, inner_segments_file, output_folder, filename):
def compute_keyframexyz(corner_segments, inner_segments):
    corner_segment_xyz = []

    for segment in corner_segments:
        x_axis = compute_segment_tangent(segment)
        y_axis, z_axis = compute_yzaxis(x_axis)
        x_axis = np.insert(x_axis, 0, x_axis[0], axis=0)
        y_axis = np.insert(y_axis, 0, y_axis[0], axis=0)
        z_axis = np.insert(z_axis, 0, z_axis[0], axis=0)
        corner_segment_xyz.append(np.hstack((x_axis, y_axis, z_axis)))

    corner_segment_xyz = np.array(corner_segment_xyz, dtype=object)


    inner_segment_xyz = []
    for segment in inner_segments:
        x_axis = compute_segment_tangent(segment)
        y_axis, z_axis = compute_yzaxis(x_axis)
        x_axis = np.insert(x_axis, 0, x_axis[0], axis=0)
        y_axis = np.insert(y_axis, 0, y_axis[0], axis=0)
        z_axis = np.insert(z_axis, 0, z_axis[0], axis=0)
        inner_segment_xyz.append(np.hstack((x_axis, y_axis, z_axis)))

    inner_segment_xyz = np.array(inner_segment_xyz, dtype=object)

    return {'cornersegments_xyz': corner_segment_xyz, 
            'innersegments_xyz': inner_segment_xyz}

    #np.save(os.path.join(output_folder, filename+'_cornersegments_xyz.npy'),corner_segment_xyz) 
    #np.save(os.path.join(output_folder, filename+'_innersegments_xyz.npy'), inner_segment_xyz)


if __name__ == '__main__':
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--segment_vector', type=str, required=False, help="npy location of segment vector")
#    args = parser.parse_args()
#    tangent = Tangent(args.segment_vector)
#    tangent.compute_tangent_for_corner_points("../skeleton_data/horse_cornerpoints.npy")
#    compute_tangent_for_corner_points("../skeleton_data/horse_cornerpoints.npy", "../skeleton_data", "horse")
    compute_keyframexyz("../skeleton_data/horse_cornersegments_additional_points.npy", "../skeleton_data/horse_innersegments.npy", "../skeleton_data", "horse")

