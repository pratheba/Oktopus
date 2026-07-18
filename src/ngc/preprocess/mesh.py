import numpy as np
from skeleton import Skeleton
from mesh_utils import _normalize_mesh

class Mesh():
    def __init__(self, mesh_path):
        self.vertices = None
        self.faces = None
        self.initialize(mesh_path)

    def initialize(self, mesh_path):
        self.vertices, self.faces = _normalize_mesh.get_unitBB_data(mesh_path)
