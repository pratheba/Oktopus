import numpy as np
import os
import argparse

#class Tangent():
#    def __init__(self, segments_vector: np.ndarray):
#        self.segments = np.load(segments_vector, allow_pickle=True)

#    def compute_tangent_for_segments(curve_obj):

#def compute_tangent_for_corner_points(corner_points_file, output_folder, fname):
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

    #np.save(os.path.join(output_folder, fname+'_point_tangent_dict.npy'), point_tangent_dict)
    #return point_tangent_dict


#        if self.segment[0] == self.corner:
#            tangent = self.segment[1] - self.corner
#        elif self.segment[-1] == self.corner:
#            tangent = self.segment[-2] - self.corner
#        else:
#            log.info("Corner vector error. Please verify")
#            return
#
#        norm_tangent = tangent / np.linalg.norm(tangent)
#        return norm_tangent

def compute_yzaxis(segment_tangents):
    y_axis = np.array([0, 1, 0])
    z_axis = np.cross(segment_tangents, y_axis)
    z_axis /= (1e-7 + np.linalg.norm(z_axis, axis=1))[:, np.newaxis]
    y_axis = np.cross(z_axis, segment_tangents)
    y_axis /= (1e-7 + np.linalg.norm(y_axis, axis=1))[:, np.newaxis]
    return y_axis, z_axis

def compute_segment_tangent(segment):
    segment = np.array(segment)
    v1 = segment[1:]
    v2 = segment[:-1]

    edges = v2 - v1
    #tangents = np.squeeze(tangents)
    #norm_tangent = tangents / (1e-7+np.linalg.norm(tangents, axis=1)[:, np.newaxis])
    norm_edge /= (1e-7+np.linalg.norm(edges, axis=1, keepdims=True))
    return norm_edge
#    if norm_tangent.shape[0] > 1:
#        tangent_start = norm_edge[0]
#        tangent_end = norm_edge[-1]
#        tangent = (norm_edge[[1:] + norm_edge[:-1]) / 2.
#        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
#        vert_tangent = np.concatenate([
#            tangent_start.reshape(1,3),
#            tangent,
#            tangent_end.reshape(1,3)
#            ], axis=0)
#    else:
#        vert_tangent = np.tile(edges, (2,1))
#    return vert_tangent

def compute_keyframexyz(corner_segments_file, inner_segments_file, output_folder, filename):
    corner_segment_xyz = []
    corner_segments = np.load(corner_segments_file, allow_pickle = True)
    inner_segments = np.load(inner_segments_file, allow_pickle = True)

    for segment in corner_segments:
        x_axis = compute_segment_tangent(segment)
        y_axis, z_axis = compute_yzaxis(x_axis)
        corner_segment_xyz.append(np.hstack((x_axis, y_axis, z_axis)))

    corner_segment_xyz = np.array(corner_segment_xyz, dtype=object)


    inner_segment_xyz = []
    for segment in inner_segments:
        x_axis = compute_segment_tangent(segment)
        y_axis, z_axis = compute_yzaxis(x_axis)
        inner_segment_xyz.append(np.hstack((x_axis, y_axis, z_axis)))

    inner_segment_xyz = np.array(inner_segment_xyz, dtype=object)

    np.save(os.path.join(output_folder, filename+'_cornersegments_xyz.npy'),corner_segment_xyz) 
    np.save(os.path.join(output_folder, filename+'_innersegments_xyz.npy'), inner_segment_xyz)


if __name__ == '__main__':
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--segment_vector', type=str, required=False, help="npy location of segment vector")
#    args = parser.parse_args()
#    tangent = Tangent(args.segment_vector)
#    tangent.compute_tangent_for_corner_points("../skeleton_data/horse_cornerpoints.npy")
#    compute_tangent_for_corner_points("../skeleton_data/horse_cornerpoints.npy", "../skeleton_data", "horse")
    compute_keyframexyz("../skeleton_data/horse_cornersegments_additional_points.npy", "../skeleton_data/horse_innersegments.npy", "../skeleton_data", "horse")

