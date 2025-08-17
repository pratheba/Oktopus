import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))

from ngc.handle_utils.visualize import pointcloud_viz


def get_radius(skeletonfile_segments, corr_polyline, outputfolder, filename):
    seg_to_polyline, keypoint_radius = connect_skeleton_segments(skeletonfile_segments, corr_polyline)

    np.save(os.path.join(outputfolder, filename+'_seg_to_polyline.npy'), seg_to_polyline)
    np.save(os.path.join(outputfolder, filename+'_keypoint_radius.npy'), keypoint_radius)


def compute_radius(key_points, corr_polyline):
    '''
    args: key_points :: Skeletal keypoints extracted from meso skeletal extraction using CGAL 
          corr_polyline  :: File consisting of the mesh vertex to the nearest keypoint mapping
    output:
         The maximum circle radius that enclosed the complete mesh sectioned by the keypoints
    '''
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

    for vec in key_points:
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
                keypoint_radius[vecbyte] = {'keypoint': vec, 'radius': (0.01,0.01)}
                continue
            tmp = np.hstack(tmp)
            v1 = polylines[tmp][:,0:3]
            v2 = polylines[tmp][:,3:]
            radiusv1 = np.linalg.norm(vec - v1, axis=1)
            radiusv2 = np.linalg.norm(vec - v2, axis=1)
            radius = np.hstack((radiusv1, radiusv2))
            maxradius = np.max(radius)+0.01
            keypoint_radius[vecbyte] = {'keypoint': vec, 'radius': (maxradius, maxradius)}

    return keypoint_radius


if __name__ == "__main__":
    
