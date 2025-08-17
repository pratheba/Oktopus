import numpy as np
import os
import sys
import trimesh


def compute_additional_points(corner_point, prev_point, max_radius, approxdis):
    #corner_points = np.load(corner_points_file, allow_pickle=True)
    tangent = np.array(corner_point - prev_point)
    norm_tangent = tangent / np.linalg.norm(tangent)
    point = corner_point
    addtional_points = []
    radius = max_radius
    while radius >= approxdis:
        newpoint = point + norm_tangent*approxdis
        radius -= approxdis
        point = newpoint
        additional_points.append(point)

    additional_points.append(point + norm_tangent*radius)
    return additional_points

def getKeyPointCorrespondence(keypoints):
    pointcorrespondence = point_correspondence(keypoints, corr_polyline_file)

def getKeyframepoints(segments, threshold=1.0):
    keypoints = []
    for seg in segments:
        start = seg[0]
        seglen = 0
        for i in range(1,len(seg)):
            seglen += np.linalg.norm(seg[i] - seg[i-1])
        print(seglen)
        numkeypoints = round(seglen*10)
        print("num keypoints = ", numkeypoints)
        kps = []
        if numkeypoints <= 2:
            kps.append(seg[0])
            kps.append(seg[len(seg)-1])
            keypoints.append(kps)
            continue

        approxdis = seglen/(numkeypoints-2)
        print("approx dis = ",approxdis)

        kps = [seg[0]]
        seglen = 0
        for i in range(1, len(seg)):
            seglen += np.linalg.norm(seg[i] - seg[i-1])
            if seglen >= approxdis:
                kps.append(seg[i])
                seglen = 0
        kps.append(seg[len(seg)-1])
        keypoints.append(kps)

    count = 1
    for kp in keypoints:
        print(len(kp))
        trimesh.Trimesh(vertices = np.array(kp), process=False).export("keypoints_"+str(count)+".ply")
        count += 1

    return keypoints


def DFS_helper(idx, dictpoint, isvisited, segments):
    idxbyte = idx.tobytes()
    if idxbyte in isvisited:
        return segments, isvisited

    segments.append(idx)
    p = dictpoint[idxbyte]
    if p['num'] <= 2:
        isvisited.append(idxbyte)
        end_points = p['end'] 
        for ep in end_points:
            segments, isvisited = DFS_helper(ep, dictpoint, isvisited, segments)
    elif p['num'] > 2:
        #segments.append(p['id'])
        return segments, isvisited

    return segments, isvisited


def DFS_corner(points, dictpoint, fname=""):

    isvisited = []
    segments = []
    segs = []
    print(fname)
    for p in points:
        idx = p['id']
        segs = []
        segs, isvisited = DFS_helper(idx, dictpoint, isvisited, segs)

        if len(segs) < 5:
            print(len(segs))
            continue
        segments.append(segs)

    count = 1
    for seg in segments:
        trimesh.Trimesh(vertices = np.array(seg), process=False).export(fname+'_'+str(count)+'.ply')
        count += 1
    return segments

def DFS_inner(points, dictpoint, fname=""):

    isvisited = []
    segments = []
    segs = []
    for p in points:
        idx = p['id']
        end_points = p['end'] 
        isvisited.append(idx.tobytes())
        for ep in end_points:
            if ep.tobytes() in isvisited:
                continue
            segs = []
            segs.append(idx)
            #segments.append(ep)

            segs, isvisited = DFS_helper(ep, dictpoint, isvisited, segs)
            print(len(segs))
            if len(segs) < 5:
                continue
            segments.append(segs)
    count = 1
    for seg in segments:
        trimesh.Trimesh(vertices = np.array(seg), process=False).export(fname+'_'+str(count)+'.ply')
        count += 1

    return segments 

def DFS1(corner_points, branch_points, edge_points, dictpoint):

    def DFS_helper(idx, points, isvisited, segments):
        idxbyte = idx.tobytes()
        if idxbyte in isvisited:
            return segments, isvisited

        segments.append(idx)
        p = dictpoint[idxbyte]
        if p['num'] <= 2:
            isvisited.append(idxbyte)
            end_points = p['end'] 
            for ep in end_points:
                segments, isvisited = DFS_helper(ep, dictpoint, isvisited, segments)
        elif p['num'] > 2:
            #segments.append(p['id'])
            return segments, isvisited

        return segments, isvisited


    allsegments = []
    cornersegments = []
    innersegments = []
    isvisited = []
    count = 1

    ### Segments of corner vertices
    for p in corner_points:
        idx = p['id']
        segments = []

        segments, isvisited = DFS_helper(idx, dictpoint, isvisited, segments)

        print(len(segments))
        if len(segments) < 5:
            continue
        allsegments.append(segments)
        cornersegments.append(segments)
        trimesh.Trimesh(vertices = np.array(segments), process=False).export(str(count)+'.ply')
        count += 1

    ### Segments of branch points
    print("into branch points", flush=True)
    for p in branch_points:
        idx = p['id']
        end_points = p['end'] 
        isvisited.append(idx.tobytes())
        for ep in end_points:
            if ep.tobytes() in isvisited:
                continue
            segments = []
            segments.append(idx)
            #segments.append(ep)

            segments, isvisited = DFS_helper(ep, dictpoint, isvisited, segments)
            print(len(segments))
            if len(segments) < 5:
                continue
            allsegments.append(segments)
            innersegments.append(segments)
            trimesh.Trimesh(vertices = np.array(segments), process=False).export(str(count)+'.ply')
            count += 1

    return cornersegments, innersegments, allsegments

def get_segments(skeletonfile):
    fread = open(skeletonfile, "r")
    lines = []
    dictpoint = {}
    all_keypoints = []
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
            #print(values)
            branch_points.append(values)


    #print(toremove_idx)
    #for c in corner_points:
    #    all_keypoints.append(c)
    #for b in branch_points:
    #    all_keypoints.append(b)
    #for e in edge_points:
    #    all_keypoints.append(e)
    #exit()


    #keypoints = np.asarray(all_keypoints)
    ## Test with trimesh
    vertices = []
    for c in corner_points:
        vertices.append(c['id'])
    trimesh.Trimesh(vertices = np.array(vertices)).export('corner_points.ply')
    vertices = []

    #cornersegments = DFS_corner(corner_points, dictpoint, "corner")

    tokeep_idx = set()
#    for c in corner_points:
#        idx = c['id'].tobytes()
#        for i,b in enumerate(branch_points):
#            b_idx = b['id'].tobytes()
#            if b_idx != idx:
#                tokeep_idx.add(i)
#
#        ends = c['end']
#        for e in ends:
#            e_idx = e.tobytes()
#            for i,b in enumerate(branch_points):
#                b_idx = b['id'].tobytes()
#                if b_idx != e_idx:
#                    tokeep_idx.add(i)
#
#    tokeep_idx = sorted(list(tokeep_idx))
#    branch_points = np.array(branch_points)[np.array(tokeep_idx)]



    for b in branch_points:
        idx = b['id'].tobytes()
        for i,c in enumerate(corner_points):
            c_idx = c['id'].tobytes()
            if c_idx != idx:
                tokeep_idx.add(i)

        ends = b['end']
        for e in ends:
            e_idx = e.tobytes()
            for i,c in enumerate(corner_points):
                c_idx = c['id'].tobytes()
                if c_idx != e_idx:
                    tokeep_idx.add(i)

    tokeep_idx = sorted(list(tokeep_idx))
    #branch_points = np.array(branch_points)[np.array(tokeep_idx)]

    innersegments = DFS_inner(branch_points, dictpoint, "inner")

    #del branch_points[toremove_idx]

    for e in edge_points:
        vertices.append(e['id'])
    trimesh.Trimesh(vertices = np.array(vertices)).export('edge_points.ply')
    vertices = []
    for b in branch_points:
        vertices.append(b['id'])
    trimesh.Trimesh(vertices = np.array(vertices)).export('branch_points.ply')
    vertices = []

    #for b in branch_points:
    #    for e in b['end']:
    #        vertices.append(e)
    #trimesh.Trimesh(vertices = np.array(vertices)).export('branch_end_points.ply')
    #trimesh.Trimesh(vertices = np.array(branch_points)).export('branch_points.ply')
    #exit()
    #keypoints = np.unique(keypoints)

    #cornersegments, innersegments, allsegments = DFS(corner_points, branch_points, edge_points, dictpoint)

    keypoints = []
    allsegments = []
    #for seg in cornersegments:
    #    allsegments.append(seg)
    #    for kp in seg:
    #        keypoints.append(kp)

#    for seg in innersegments:
#        allsegments.append(seg)
#        for kp in seg:
#            keypoints.append(kp)
#
#    keypoints = np.unique(np.array(keypoints))

    #getKeyframepoints(cornersegments, threshold=1.0)
    keypoints = getKeyframepoints(allsegments, threshold=1.0)
    keypoint_correspondences = getKeyPointCorrespondence(keypoints)

    allsegments = np.asarray(allsegments, dtype="object")
    cornerpoints = np.asarray(corner_points, dtype="object")

    return {#'cornerpoints': cornerpoints,
            'keypoints': keypoints,
          #  'cornersegments': cornersegments,
            #'innersegments': innersegments,
            'allsegments': allsegments}

#    np.save(os.path.join(outputfolder, filename+'_cornerpoints.npy'), np.array(corner_points))
#    np.save(os.path.join(outputfolder, filename+'_cornersegments.npy'), cornersegments)
#    np.save(os.path.join(outputfolder, filename+'_innersegments.npy'), innersegments)
#    np.save(os.path.join(outputfolder, filename+'_segments.npy'), allsegments)
    
if __name__ == '__main__':
    get_segments(*sys.argv[1:])
