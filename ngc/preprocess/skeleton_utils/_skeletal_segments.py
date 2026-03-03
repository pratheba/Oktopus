import numpy as np
import os
import sys
import trimesh
#from visualize import pointcloud_viz, curve_viz


class SkeletalSegments():
    def __init__(self, root_points, mapping_dict, correspondences):
        self.initial_segments = self.get_initial_segments(root_points, mapping_dict)
        self.initial_keypoints, self.initial_correspondence = self.get_keyframes_and_reassignCorrespondence(self.initial_segments, correspondences)

        self.compute_additional_points()

        self.segments = self.initial_segments
        self.keypoints = self.initial_keypoints
        self.correspondence = self.initial_correspondence


    #def compute_additional_points(corner_point, prev_point, max_radius, approxdis, correspondence_points):
        #corner_points = np.load(corner_points_file, allow_pickle=True)
    def compute_additional_points(self):
        for key, keypoints in self.initial_keypoints.items():
            corner_point = keypoints[-1]
            prev_point = keypoints[-2]
            tangent = np.array(corner_point - prev_point)
            norm_tangent = tangent / np.linalg.norm(tangent)

    
            correspondence = self.initial_correspondence[corner_point.tobytes()]['corres']
            trimesh.Trimesh(vertices = np.array(correspondence), process=False).export("add_keypoint_orig_corres.ply")
            radius = (np.linalg.norm(corner_point - correspondence, axis=1))
            max_radius = np.max(radius)
            numkeypoints = round((max_radius+0.01)*10)

            add_keypoints = [corner_point + norm_tangent*i*0.1 for i in range(numkeypoints+1)]
            #add_keypoints = np.concatenate((np.array([corner_point]), np.array(add_keypoints)))
            #print(add_keypoints)

            add_correspondence = self.assign_closest_correspondence(add_keypoints, correspondence)
            #print(len(correspondence))

            count = 0
            new_keypoints = {}
            #add_keypoints = []
            
            for k, v in add_correspondence.items():
                if len(v['corres']) <= 0:
                    continue
                new_keypoints.update({k:v})
                #add_keypoints.append(v['skelpoint'])
                #print(v['skelpoint'])
                nodes = np.array([v['skelpoint']])
                nodes = np.concatenate((nodes, v['corres']))
                #print("len of v ", len(v['corres']))
                trimesh.Trimesh(vertices = np.array(nodes), process=False).export("add_keypoint_corres"+str(count)+".ply")
                count += 1

            self.initial_correspondence.update(new_keypoints)
            #print(self.initial_keypoints[key])
            self.initial_keypoints[key] = np.concatenate((np.array(self.initial_keypoints[key]), np.array(add_keypoints)[1:count+1]))
            #print(self.initial_keypoints[key])
            #print("********")
            #exit()

            #nodes = []
            #for k,v in self.initial_correspondence.items():
            #    if len(nodes):
            #        nodes = np.concatenate((nodes, v['corres']))
            #    else:
            #        nodes = v['corres']
                #print("len of v ", len(v['corres']))
            #trimesh.Trimesh(vertices = np.array(nodes), process=False).export("add_keypoint_corres_final.ply")

    def collectCorrespondences(self, points, correspondence_map):
        """ Collects the surface vertices that belongs to a segment made of the keypoints.

        Parameters
        ----------
        points: np.array
             The key points that make a segment
        correspondence_map: dict
             Dictionary where key is the keyframe points and values contain the correspondence points of the surface

        Returns
        -------
        np.array
            Collection of surface vertices

        """
        allcorrespoints = []
        for point in points:
            #if point.tobytes() in correspondence_map:
                corres = correspondence_map[point.tobytes()]['corres']
                allcorrespoints.append(corres)
        allcorrespoints = np.vstack(allcorrespoints)
        return allcorrespoints

    def assign_closest_correspondence(self, points, correspondences):
        correspondences = np.array(correspondences)
        points = np.array(points)

        diff = np.linalg.norm(correspondences[:,None,:] - points[None,:, :], axis=2)
        label = np.argmin(diff, axis=1)

        keyframe_correspondence = {}
        for i, p in enumerate(points):
            mask = (label == i)
            keyframe_correspondence[p.tobytes()] = {'skelpoint': p, 'corres': correspondences[mask]}

        #print(keyframe_correspondence)
        

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
        seg_keypoints = {}
        count = 1

        for index,seg in enumerate(segments):
            start = seg[0]
            seglen = 0
            for i in range(1,len(seg)):
                seglen += np.linalg.norm(seg[i] - seg[i-1])
            #print(seglen)
            numkeypoints = round(seglen*10)
            kps = []
            if numkeypoints <= 2:
                kps.append(seg[0])
                kps.append(seg[len(seg)-1])

                kps = np.array(kps)
                kp_corres = self.get_correspondence(np.array(seg), kps, orig_correspondence, str(count))
                new_correspondence.update(kp_corres)
                count += 1

                keypoints.append(kps)
                seg_keypoints[index] = kps
                continue

            approxdis = seglen/(numkeypoints-2)
            kps = [seg[0]]
            seglen = 0
            start = 0
            for i in range(1, len(seg)):
                seglen += np.linalg.norm(seg[i] - seg[i-1])
                if seglen >= approxdis:
                    kps.append(seg[i])
                    seglen = 0
                    

            kps.append(seg[len(seg)-1])
            kps = np.array(kps)
            keypoints.append(kps)
            seg_keypoints[index] = kps
            kp_corres = self.get_correspondence(np.array(seg), kps, orig_correspondence, str(count))
            new_correspondence.update(kp_corres)
            count += 1

        count = 1
        for kp in keypoints:
            trimesh.Trimesh(vertices = np.array(kp), process=False).export("keypoints_"+str(count)+".ply")
            count += 1

        #print(seg_keypoints)

        return seg_keypoints, new_correspondence


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
        #initial_keypoints = np.vstack(initial_keypoints)
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
