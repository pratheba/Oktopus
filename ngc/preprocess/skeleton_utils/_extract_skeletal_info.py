import numpy as np
import os
import sys
import trimesh

def get_correspondence_points(skeletal_points, correspondencefile):
    fread = open(correspondencefile, "r")
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

    keypoint_corres = {}
    for point in skeletal_points:
        tmp = []
        vecbyte = point.tobytes()
        #print(point)
        index1 = np.where(lines1 == vecbyte)[0]
        index2 = np.where(lines2 == vecbyte)[0]

        if len(index1) > 0:
            if vecbyte in keypoint_corres:
                existing_corres = keypoint_corres[vecbyte]
                new_corres = polylines[index1][:,3:]
                new_corres = np.concatenate((existing_corres, new_corres))
                new_corres = np.vstack(new_corres)
                keypoint_corres[vecbyte]['corres'] = new_corres
            else:
                keypoint_corres[vecbyte] = {'skelpoint': point, 'corres': np.array(polylines[index1][:,3:])}
        if len(index2) > 0:
            #oassert(len(index1) <= 0)
            if vecbyte in keypoint_corres:
                existing_corres = keypoint_corres[vecbyte]
                new_corres = polylines[index2][:,:3]
                new_corres = np.concatenate((existing_corres, new_corres))
                new_corres = np.vstack(new_corres)
                keypoint_corres[vecbyte]['corres'] = new_corres
            else:
                keypoint_corres[vecbyte] = {'skelpoint': point, 'corres': np.array(polylines[index2][:,:3])}

    #count = 1
    #allpoints = []
    #for point in skeletal_points:
    #    k = keypoint_corres[point.tobytes()]['corres']
    #    allpoints.append(k)
    #allpoints = np.vstack(allpoints)
    #trimesh.Trimesh(vertices = allpoints, process=False).export('orig_corres_'+str(count)+'.ply')
    #exit()

    #print(keypoint_corres)

    return keypoint_corres


def get_skeletal_points(skeletonfile):
    fread = open(skeletonfile, "r")
    lines = []
    dictpoint = {}
    for line in fread:
        line = line.strip()
        arr = np.fromstring(line, dtype=float, sep=' ')
        lines.append([arr[1:4],arr[4:]])
        arr1 = arr[1:4].tobytes()
        arr2 = arr[4:7].tobytes()
        if not arr1 in dictpoint:
            dictpoint[arr1] = {'num': 1, 'id': arr[1:4], 'end': [arr[4:7]]}
            #all_keypoints.append(arr[1:4])
        else:
            dictpoint[arr1]['num'] += 1 
            dictpoint[arr1]['end'].append(arr[4:7])

        if not arr2 in dictpoint:
            dictpoint[arr2] = {'num': 1, 'id': arr[4:7], 'end': [arr[1:4]]}
            #all_keypoints.append(arr[4:7])
        else:
            dictpoint[arr2]['num'] += 1
            dictpoint[arr2]['end'].append(arr[1:4])

    corner_points = []
    branch_points = []
    branch_end_points = []
    edge_points = []
    for idx, values in dictpoint.items():
        if values['num'] == 1:
            corner_points.append(values)
        if values['num'] == 2:
            edge_points.append(values)
        if values['num'] > 2:
            branch_points.append(values)

    ## Test with trimesh
    vertices = []
    for c in corner_points:
        vertices.append(c['id'])
    trimesh.Trimesh(vertices = np.array(vertices)).export('corner_points.ply')
    vertices = []
    for e in edge_points:
        vertices.append(e['id'])
    trimesh.Trimesh(vertices = np.array(vertices)).export('edge_points.ply')
    vertices = []
    for b in branch_points:
        vertices.append(b['id'])
    trimesh.Trimesh(vertices = np.array(vertices)).export('branch_points.ply')

    return {
            'branchpoints': branch_points,
            'edgepoints': edge_points,
            'cornerpoints': corner_points,
            'mappingdict': dictpoint
           }


if __name__ == '__main__':
    get_skeletal_points(*sys.argv[1:])
