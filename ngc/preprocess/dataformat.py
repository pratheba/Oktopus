import json
import numpy as np
import pickle
import os

def create_json(segments, segments_xyz, keypoint_radius, iscorner=False):
    if iscorner:
        commonname = "corner_segment_"
    else:
        commonname = "inner_segment_"

    input_dict = []
    for idx, segment in enumerate(segments):
        name = commonname+str(idx)
        keypoints = np.array(segment)
        keyradius = []
        xyz = segments_xyz[idx]
        for k, keypoint in enumerate(keypoints):
            vecbyte = keypoint.tobytes()
            radius = keypoint_radius[vecbyte]['radius']
            keyradius.append([radius[0], radius[1]])
        x_axis = xyz[:,:3]
        y_axis = -xyz[:,3:6]
        z_axis = xyz[:,6:]
        ball = {'start_x': None, 'end_x': None}
        #if iscorner:
        x_axis = np.insert(x_axis, 0, x_axis[0], axis=0)
        y_axis = np.insert(y_axis, 0, y_axis[0], axis=0)
        z_axis = np.insert(z_axis, 0, z_axis[0], axis=0)
        ball = {'start_x': None, 'end_x': None}
        keyradius = np.array(keyradius)

        indict = {'key_points': keypoints,
                      'key_radius': keyradius,
                      'x_axis': x_axis,
                      'y_axis': y_axis,
                      'z_axis': z_axis,
                      'ball': ball,
                      'name': name}

        input_dict.append(indict)

    return input_dict

def create_input_json(corner_segments_file, inner_segments_file, \
        corner_segments_xyz_file, inner_segments_xyz_file, radius_file,\
        output_folder, filename):

    corner_segments = np.load(corner_segments_file, allow_pickle=True)
    inner_segments = np.load(inner_segments_file, allow_pickle=True)
    corner_segments_xyz = np.load(corner_segments_xyz_file, allow_pickle=True)
    inner_segments_xyz = np.load(inner_segments_xyz_file, allow_pickle=True)
    keypoint_radius = np.load(radius_file, allow_pickle=True).item()


    corner_json_array = create_json(corner_segments, corner_segments_xyz, keypoint_radius, True)
    inner_json_array = create_json(inner_segments, inner_segments_xyz, keypoint_radius)
    input_dict = {'curves': np.concatenate((corner_json_array, inner_json_array))}

    with open(os.path.join(output_folder, filename+'_std_handle.pkl'), 'wb') as handle:
        pickle.dump(input_dict, handle)

    #np.save(os.path.join(output_folder, filename+'_std_handle.pkl'), input_dict)


if __name__ == '__main__':
    import sys
    create_input_json(*sys.argv[1:])
    
