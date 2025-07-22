import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))

from ngc.handle_utils.visualize import pointcloud_viz

#def getmaxLengthOfPolylines(corner_points, corner_polylines):

#def update_tangents(corner_tangent_npyfile, new_points, filename):
#    corner_tangent = np.load(corner_tangent_npyfile, allow_pickle=True).item()
#    corner_ad
#    for key, value in corner_tangent.items():


def update_skeleton(corner_segments_npyfile, new_points, filename):
    corner_segments = np.load(corner_segments_npyfile, allow_pickle=True)
    new_corner_segments = []
    for idx, seg in enumerate(corner_segments):
        seg = np.insert(seg, 0, new_points[idx], axis=0)
        new_corner_segments.append(seg)
    #print(new_corner_segments)
    new_corner_segments = np.array(new_corner_segments, dtype=object)

    #np.save(os.path.join(output_folder, filename+'_cornersegments_additional_points.npy'))
    np.save(os.path.join('../skeleton_data', filename+'_cornersegments_additional_points.npy'), new_corner_segments)


#def keypoint_adder(corner_tangent_file, keypoint_radius_file):
#    corner_tangent = np.load(corner_tangent_file, allow_pickle=True).item()
#    keypoint_radius = np.load(keypoint_radius_file, allow_pickle=True).item()
def keypoint_adder(corner_tangent, keypoint_radius, corner_segments):
    new_points = []
    seg = corner_segments.copy()
    for idx, (key, c_point) in enumerate(corner_tangent.items()):
        key_point = c_point['id']
        tangent = c_point['tangent']
        radius = keypoint_radius[key]['radius'][0]
        additional_point = key_point + tangent *(radius+0.02)
        new_points.append(additional_point)
        seg[idx] = np.insert(seg[idx], 0, additional_point, axis=0)
    new_points = np.array(new_points)
    #return new_point
    #import trimesh
    #trimesh.Trimesh(vertices = new_point, process=False).export('newpoint.ply')
    #pointcloud_viz(new_point)
    return {'additional_points': new_points, 'cornersegment_with_addedpoints': seg}
    #return new_point, seg


def add_edge_keypoints(corner_points, corr_polyline):
    fread = open(corr_polyline, "r")
    lines = []
    dictpoint = {}
    lines = []
    polylines = []
    for line in fread:
        line = line.strip()
        arr = np.fromstring(line, dtype=float, sep=' ')
        arr1 = arr[1:4].tobytes()
        arr2 = arr[4:7].tobytes()
        lines.append((arr1, arr2))
        polylines.append(arr[1:])
    polylines = np.array(polylines)

    lines = np.array(lines)
    lines1 = lines[:,0]
    lines2 = lines[:,1]

    seg_to_polyline = {count: [] for count in range(len(segments))}
    keypoint_radius = {}

    for vec in corner_points:
            tmp = []
            vecbyte = vec.tobytes()
            index1 = np.where(lines1 == vecbyte)[0]
            index2 = np.where(lines2 == vecbyte)[0]
            if len(index1):
                polyline.append(index1)
                tmp.append(index1)
            if len(index2):
                polyline.append(index2)
                tmp.append(index2)
            if not len(tmp):
                #print(vec)
                keypoint_radius[vecbyte] = {'keypoint': vec, 'radius': (0,0)}
                continue
            tmp = np.hstack(tmp)
            v1 = polylines[tmp][:,0:3]
            v2 = polylines[tmp][:,3:]
            radiusv1 = np.linalg.norm(vec - v1, axis=1)
            radiusv2 = np.linalg.norm(vec - v2, axis=1)
            radius = np.hstack((radiusv1, radiusv2))
            maxradius = np.max(radius)
            keypoint_radius[vecbyte] = {'keypoint': vec, 'radius': (maxradius, maxradius)}




if __name__ == "__main__":
    pass
    #new_points = keypoint_adder(*sys.argv[1:])
    #np.save(os.path.join('../skeleton_data','new_points.npy'), new_points)

    #update_skeleton('../skeleton_data/horse_cornersegments.npy', new_points, 'horse')
    
