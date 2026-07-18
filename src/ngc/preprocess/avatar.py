import numpy as np
from mesh import Mesh
from skeleton import Skeleton

class Avatar():
    def __init__(self, args):
        self.mesh = Mesh(args.mesh_path)
        self.skeleton = Skeleton(args.skel_file, args.corres_file, self.mesh.vertices)

    def preprocess(self):
        self.skeleton.get_keyframe_with_radius()


if __name__ == '__main__':
    avatar = Avatar()

