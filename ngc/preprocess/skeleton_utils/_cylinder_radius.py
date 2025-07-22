import numpy as np
import os
import sys
import trimesh
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))

from ngc.handle_utils.visualize import curve_viz
#import visualize


def add_radius_endpoints(new_points_file, keypoint_radius_file, outputfolder, filename):
    points = np.load(new_points_file, allow_pickle=True)
    keypoint_radius = np.load(keypoint_radius_file, allow_pickle=True).item()

    for p in points:
        vecbyte = p.tobytes()
        keypoint_radius[vecbyte] = {'keypoint': p, 'start': 0.0, 'end': 0.0, 'radius': (0.01, 0.01)}
    
    return {'keypoint_radius_additional_points': keypoint_radius}



def get_radius(skeleton_segments, corr_polyline):
    seg_to_polyline, keypoint_radius = connect_skeleton_segments(skeleton_segments, corr_polyline)
    return {'seg_to_polyline': seg_to_polyline, 
            'keypoint_radius': keypoint_radius}
#    vertices = []
#    lines = []
#    for k,value in seg_to_polyline.items():
#        print(value)
#        v1 = value[:,0:3]
#        v2 = value[:,3:]
#        nodes = np.vstack((v1, v2))
#        #print(visualize)
#        curve_viz(nodes)
#        #lines.append({'type': 'curves', 'vertices': lines})
#        #print(v1)
#        #print(v2)
#        print("***")
#    exit()



def connect_skeleton_segments(segments, corr_polyline):
    #segments = np.load(skeletonfile_segments, allow_pickle=True)

    fread = open(corr_polyline, "r")
    lines = []
    dictpoint = {}
    lines = []
    polylines = []
    ## Read the polylines connected to each of skeleton file segments
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

    for count, segment in enumerate(segments):
        polyline = []
        for vec in segment:
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
            maxradius = np.max(radius)+0.01
            keypoint_radius[vecbyte] = {'keypoint': vec, 'polylines':polylines[tmp], 'radius': (maxradius, maxradius)}

        polyline = np.hstack(polyline)
        seg_to_polyline[count] = polylines[polyline]

    return seg_to_polyline, keypoint_radius

if __name__ == '__main__':
    #get_radius(*sys.argv[1:])
    add_radius_endpoints(*sys.argv[1:]) 
