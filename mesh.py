import numpy as np

class Mesh:
    def __init__(self):
        self.positions = []
        self.triangles = []
        self.adjacencies = []
        self.normals = []
        self.v2t = []

class MeshPoint:
    def __init__(self, face=0, uv=np.array([0.0, 0.0])):
        self.face = face
        self.uv = uv

    def interpolate(self, mesh:Mesh):
        face = self.face
        uv = self.uv
        p0 = mesh.positions[mesh.triangles[face][0]]
        p1 = mesh.positions[mesh.triangles[face][1]]
        p2 = mesh.positions[mesh.triangles[face][2]]
        return (1 - uv[0] - uv[1]) * p0 + uv[0] * p1 + uv[1] * p2