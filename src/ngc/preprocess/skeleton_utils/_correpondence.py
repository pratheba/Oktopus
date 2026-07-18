import numpy as np
import os
import sys
import trimesh
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../..")))

from ngc.handle_utils.visualize import curve_viz
#import visualize

def point_correspondence(points, corr_polyline_file):
    fread = open(corr_polyline_file, "r")
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

    #seg_to_polyline = {count: [] for count in range(len(segments))}
    keypoint_corres = {}
    #print("points = ", points)
    for vec in points:
        tmp = []
        vecbyte = vec.tobytes()
        index1 = np.where(lines1 == vecbyte)[0]
        index2 = np.where(lines2 == vecbyte)[0]
        #print(len(index1))
        #print(len(index2))
        #exit()
        if len(index1) > 0:
            keypoint_corres[vecbyte] = {'keypoint': vec, 'corres': polylines[index1][:3]}
        if len(index2) > 0:
            keypoint_corres[vecbyte] = {'keypoint': vec, 'corres': polylines[index2][3:]}

    return keypoint_corres



def point_correspondence(points, corr_polyline_file):
    fread = open(corr_polyline_file, "r")
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

    #seg_to_polyline = {count: [] for count in range(len(segments))}
    keypoint_corres = {}
    #print("points = ", points)
    for vec in points:
        tmp = []
        vecbyte = vec.tobytes()
        index1 = np.where(lines1 == vecbyte)[0]
        index2 = np.where(lines2 == vecbyte)[0]
        #print(len(index1))
        #print(len(index2))
        #exit()
        if len(index1) > 0:
            keypoint_corres[vecbyte] = {'keypoint': vec, 'corres': polylines[index1][:3]}
        if len(index2) > 0:
            keypoint_corres[vecbyte] = {'keypoint': vec, 'corres': polylines[index2][3:]}

    return keypoint_corres


if __name__ == '__main__':
    add_radius_endpoints(*sys.argv[1:]) 
