import os
import argparse
import numpy as np
from skeleton_utils import _skeletal_segments, _cylinder_radius, _compute_tangents, _keypoints_adder


class Skeleton():
    def __init__(self, args):
        self.name = args.name
        self.skel_file = args.skel_file
        self.corres_file = args.corres_file

        self.initialize(args.output_folder)

    def initialize(self, output_folder):
        '''
           Get all the meta data about the skeleton
           self.segments -> dict keys 
           {'cornerpoints', 'keypoints', 'cornersegments', 'innersegments', 'allsegments'}

           additionalpoint_segments -> dict keys
           {'additional_points', 'cornersegment_with_addedpoints'}
        '''
        # Split into corner segments and inner segments
        # Corner segments : The branch where the start of the skeleton is the leaf node
        # Inner segments : The inner part of the skeleton where the end are connected to either the corner branches or the inner branches
        self.segments = _skeletal_segments.get_segments(self.skel_file)
        self.npsave(self.segments, output_folder)

        # Check if the segments file exists
        segments_path = os.path.join(output_folder, self.name+'_allsegments.npy')
        assert os.path.exists(segments_path)

        # From all the segments (corner and inner) and the correspondence file which containes the mesh vertex to 
        # nearest keypoint mapping, compute per keypoint radius
        # It is computed per keypoint as the maximum euclidean distance from all the correspondence mapping to that particular keypoint
        self.allsegments = self.segments['allsegments']
        self.segments_meta = _cylinder_radius.get_radius(self.allsegments, self.corres_file)
        self.npsave(self.segments_meta, output_folder)

        # The CGAL meso skeletal corner segment lies within the mesh
        # so need to compute one more keypoint that extends beyond the mesh so while computing NGC the entire mesh can be covered
        # That additional keypoints is computed along the tangent of the last keypoint in the corner segments

        # so first need to compute the tangent
        # Compute the tangents for the corner points so the additional points could be 
        # added that will cover the entire shape
        cornerpoints_path = os.path.join(output_folder, self.name+'_cornerpoints.npy')
        assert os.path.exists(cornerpoints_path)
        cornerpoint_tangent = _compute_tangents.compute_tangent_for_corner_points(self.segments['cornerpoints']) 
        self.npsave(cornerpoint_tangent, output_folder)

        # Once the tangent is computed, now the radius is computed for all
        # and the additional keypoints are added to the corner segments
        cornerpoint_tangent_path = os.path.join(output_folder, self.name+'_cornerpoint_tangent.npy')
        assert os.path.exists(cornerpoint_tangent_path)
        additionalpoints_segments = _keypoints_adder.keypoint_adder(cornerpoint_tangent['cornerpoint_tangent'], self.segments_meta['keypoint_radius'], self.segments['cornersegments'])
        self.npsave(additionalpoints_segments, output_folder)


    def npsave(self, content_to_save, output_folder):
        if isinstance(content_to_save, dict):
            for key, value in content_to_save.items():
                filepath = os.path.join(output_folder, self.name+'_'+key+'.npy')
                np.save(filepath, value)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help="Name of the object")
    parser.add_argument('--skel_file', type=str, required=True, help="Skeletal file information")
    parser.add_argument('--corres_file', type=str, required=True, help="Correspondence file information")
    parser.add_argument('--output_folder', type=str, required=True, help="Output folder information")

    args = parser.parse_args()
    skeleton = Skeleton(args)

