import numpy as np
import torch

class Mesh:
    def __init__(self):
        self.positions = []
        self.triangles = []
        self.adjacencies = []
        self.normals = []
        self.v2t = []

class MeshPoint:
    def __init__(self, face=0, uv=np.zeros(2)):
        self.face = face
        self.uv = uv
        if isinstance(uv, torch.Tensor):
            self.tensor = True
        else:
            self.tensor = False
            
    def interpolate(self, mesh:Mesh, tensor=False):
        face = self.face
        uv = self.uv
        p0 = mesh.positions[mesh.triangles[face][0]]
        p1 = mesh.positions[mesh.triangles[face][1]]
        p2 = mesh.positions[mesh.triangles[face][2]]

        if tensor:
            p0 = torch.tensor(mesh.positions[mesh.triangles[face][0]],dtype=torch.float32)
            p1 = torch.tensor(mesh.positions[mesh.triangles[face][1]],dtype=torch.float32)
            p2 = torch.tensor(mesh.positions[mesh.triangles[face][2]],dtype=torch.float32)
        elif self.tensor:
            uv = uv.detach().numpy()
        
        pos = (1 - uv[0] - uv[1]) * p0 + uv[0] * p1 + uv[1] * p2
        return pos
    