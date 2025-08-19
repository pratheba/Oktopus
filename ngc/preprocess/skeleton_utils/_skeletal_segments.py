import numpy as np
import os
import sys
import trimesh
from visualize import pointcloud_viz, curve_viz


class SkeletalSegments():
    def __init__(self, root_points, mapping_dict, correspondences):
        self.initial_segments = self.get_initial_segments(root_points, mapping_dict)
        #self.initial_keypoints = self.get_initial_keyframes()
        self.initial_keypoints, self.initial_correspondence = self.get_keyframes_and_reassignCorrespondence(self.initial_segments, correspondences)
        #keypoints = np.vstack(keypoints)
        #np.save('keypoints_file.npy', keypoints)

    #def keyframe_radius(self):

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


    #def getKeyPointCorrespondence(keypoints, correspondence_file):
    #    pointcorrespondence = point_correspondence(keypoints, correspondence_file)

    def collectCorrespondences(self, points, correspondence_map):
       
        allcorrespoints = []
        #print(correspondence_map)
        #allnodes = []
        #for k, v in correspondence_map.items():
        #    nodes = v['corres']
        #    allnodes.append(nodes)
        #allnodes = np.vstack((allnodes))
        #trimesh.Trimesh(vertices = allnodes, process=False).export('allcorres.ply')

        for point in points:
            #print(point)
            #print("***")
            corres = correspondence_map[point.tobytes()]['corres']
            #print(corres)
            #exit()
            allcorrespoints.append(corres)

        allcorrespoints = np.vstack(allcorrespoints)
        #print(allcorrespoints.shape)
        #exit()

        return allcorrespoints

    def assign_closest_correspondence(self, points, correspondences):
        correspondences = np.array(correspondences)
        points = np.array(points)
        #print(points.shape)
        #print(correspondences.shape)
        #exit()

        diff = np.linalg.norm(correspondences[:,None,:] - points[None,:, :], axis=2)
        #print(diff)
        label = np.argmin(diff, axis=1)
        #print(label)

        keyframe_correspondence = {}
        for i, p in enumerate(points):
            mask = (label == i)
            #print("mask = ", mask)
            #print(correspondences[mask])
            #exit()
            keyframe_correspondence[p.tobytes()] = {'skelpoint': p, 'corres': correspondences[mask]}

        return keyframe_correspondence

    def get_correspondence(self, segs, kps, orig_correspondence, fname="test"):
        kp_corres = self.collectCorrespondences(segs, orig_correspondence)
        kp_corres = np.array(kp_corres)

        nodes = segs
        nodes = np.concatenate((nodes,kp_corres))
        nodes = np.vstack(nodes)

        trimesh.Trimesh(vertices = np.array(nodes), process=False).export("keypoint_orig_corres"+fname+".ply")
        ###################
        new_corres = self.assign_closest_correspondence(kps, kp_corres)
        #new_correspondence.append(new_corres)
        nodes = []
        for k,v in new_corres.items():
            nodes.append(v['skelpoint'])
            for e in v['corres']:
                nodes.append(e)

        trimesh.Trimesh(vertices = np.array(nodes), process=False).export("keypoint_corres"+fname+".ply")
        return new_corres


    def keyframe_and_correspondence(self, segments, orig_correspondence):
        keypoints = []
        new_correspondence = {}
        count = 1

        for seg in segments:
            start = seg[0]
            seglen = 0
            for i in range(1,len(seg)):
                seglen += np.linalg.norm(seg[i] - seg[i-1])
            #print(seglen)
            numkeypoints = round(seglen*10)
            #print("num keypoints = ", numkeypoints)
            kps = []
            if numkeypoints <= 2:
                kps.append(seg[0])
                kps.append(seg[len(seg)-1])

                kps = np.array(kps)
                kp_corres = self.get_correspondence(np.array(seg), kps, orig_correspondence, str(count))
                new_correspondence.update(kp_corres)
                count += 1

#                kp_corres = self.collectCorrespondences(kps, orig_correspondence)
#                ######## TESTING
#                nodes = kps
#                nodes = np.concatenate((nodes,kp_corres))
#                nodes = np.vstack(nodes)
#
#                trimesh.Trimesh(vertices = np.array(nodes), process=False).export("keypoint_orig_corres"+str(count)+".ply")
#                #count += 1
#                ################################
#                kp_corres = np.array(kp_corres)
#                print("kp = ", kps)
#                print("corres = ", kp_corres)
#
                keypoints.append(kps)
#                new_corres = self.assign_closest_correspondence(kps, kp_corres)
#                new_correspondence.append(new_corres)
                continue

            approxdis = seglen/(numkeypoints-2)
            #print("approx dis = ",approxdis)

            kps = [seg[0]]
            seglen = 0
            start = 0
            for i in range(1, len(seg)):
                seglen += np.linalg.norm(seg[i] - seg[i-1])
                if seglen >= approxdis:
                    kps.append(seg[i])
#                    keypoints.append(np.array(kps))
                    seglen = 0
                    
#                    kp_corres = self.get_correspondence(np.array(seg[start:i+1]), np.array(kps), orig_correspondence)
#                    new_correspondence.append(kp_corres)

#                    kp_corres = self.collectCorrespondences(seg[start:i+1], orig_correspondence)
#                    #kp_corres = self.collectCorrespondences(seg, orig_correspondence)
#                    kp_corres = np.array(kp_corres)
#                    #print(kp_corres.shape)
#                    #exit()
#                    nodes = kps
#                    nodes = np.concatenate((nodes,kp_corres))
#                    nodes = np.vstack(nodes)
#
#                    trimesh.Trimesh(vertices = np.array(nodes), process=False).export("keypoint_orig_corres"+str(count)+".ply")
#                    ###################
#                    new_corres = self.assign_closest_correspondence(kps, kp_corres)
#                    new_correspondence.append(new_corres)
#                    nodes = []
#                    for k,v in new_corres.items():
#                        nodes.append(v['skelpoint'])
#                        #count += 1
#                        print("v = ", v)
#                        for e in v['corres']:
#                            nodes.append(e)
#                            #count += 1
#                            #edges.append([count-1, count])
#
#                    trimesh.Trimesh(vertices = np.array(nodes), process=False).export("keypoint_corres"+str(count)+".ply")
#                    count += 1

#                    start = i
#                    kps = []


            kps.append(seg[len(seg)-1])
            kps = np.array(kps)
            keypoints.append(kps)
            kp_corres = self.get_correspondence(np.array(seg), kps, orig_correspondence, str(count))
            new_correspondence.update(kp_corres)
            count += 1

#            kp_corres = self.get_correspondence(np.array(seg[start:i+1]), np.array(kps), orig_correspondence)
#            new_correspondence.append(kp_corres)
#
#            kp_corres = self.collectCorrespondences(seg, orig_correspondence)
#            kp_corres = np.array(kp_corres)
#            #print(kp_corres.shape)
#            #exit()
#            nodes = kps
#            nodes = np.concatenate((nodes,kp_corres))
#            nodes = np.vstack(nodes)
#
#            trimesh.Trimesh(vertices = np.array(nodes), process=False).export("keypoint_orig_corres"+str(count)+".ply")
#            ###################
#            new_corres = self.assign_closest_correspondence(kps, kp_corres)
#            new_correspondence.append(new_corres)
#            
            #count = 1
#            nodes = []
#            edges = []
#            for k,v in new_corres.items():
#                nodes.append(v['skelpoint'])
#                #count += 1
#                print("v = ", v)
#                for e in v['corres']:
#                    nodes.append(e)
#                    #count += 1
#                    #edges.append([count-1, count])
#
#            trimesh.Trimesh(vertices = np.array(nodes), process=False).export("keypoint_corres"+str(count)+".ply")
#            count += 1

            #curve_viz(nodes, edges)

        count = 1
        for kp in keypoints:
            print(len(kp))
            trimesh.Trimesh(vertices = np.array(kp), process=False).export("keypoints_"+str(count)+".ply")
            count += 1

        print(new_correspondence)

        return keypoints, new_correspondence


    def getKeyframepoints(self, segments, threshold=1.0):
        keypoints = []
        for seg in segments:
            start = seg[0]
            seglen = 0
            for i in range(1,len(seg)):
                seglen += np.linalg.norm(seg[i] - seg[i-1])
            #print(seglen)
            numkeypoints = round(seglen*10)
            #print("num keypoints = ", numkeypoints)
            kps = []
            if numkeypoints <= 2:
                kps.append(seg[0])
                kps.append(seg[len(seg)-1])
                keypoints.append(kps)
                continue

            approxdis = seglen/(numkeypoints-2)
            #print("approx dis = ",approxdis)

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



    def DFS_inner(self,points, dictpoint, fname=""):

        isvisited = []
        segments = []
        segs = []

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

#    def get_keypoint_correspondence(self):
#        keypoint_correspondences = getKeyPointCorrespondence(keypoints, correspondence_file)
#
#        allsegments = np.asarray(allsegments, dtype="object")
#        cornerpoints = np.asarray(corner_points, dtype="object")
#
#        return {
#                'keypoints': keypoints,
#                'allsegments': allsegments
#               }

    def get_keyframes_and_reassignCorrespondence(self, segments, orig_correspondence):
        initial_keypoints, initial_correspondence = self.keyframe_and_correspondence(segments, orig_correspondence)
        #initial_keypoints = self.getKeyframepoints(self.initial_segments, threshold=1.0)
        initial_keypoints = np.vstack(initial_keypoints)
        return initial_keypoints, initial_correspondence

    def get_initial_keyframes(self):
        initial_keypoints = self.getKeyframepoints(self.initial_segments, threshold=1.0)
        initial_keypoints = np.vstack(initial_keypoints)
        return initial_keypoints

    def get_initial_segments(self, branch_points, dictpoint):
        initial_segments = self.DFS_inner(branch_points, dictpoint, "inner")
        return initial_segments

        #keypoints = getKeyframepoints(allsegments, threshold=1.0)

        #keypoints = np.vstack(keypoints)
        #print(keypoints.shape)
        #np.save('keypoints_file.npy', keypoints)
        #print(keypoints)

        #keypoint_correspondences = getKeyPointCorrespondence(keypoints, correspondence_file)

        #allsegments = np.asarray(allsegments, dtype="object")
        #cornerpoints = np.asarray(corner_points, dtype="object")

        #return {
        #        'keypoints': keypoints,
        #        'allsegments': allsegments
        #       }


if __name__ == '__main__':
    get_segments(*sys.argv[1:])
