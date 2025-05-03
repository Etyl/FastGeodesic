import numpy as np
import torch

from geometry.utils import normalize

class Mesh:
    def __init__(self):
        self.positions = []
        self.triangles = []
        self.adjacencies = []
        self.triangle_normals = []
        self.v2t = []

    def build(self):
        """Build the mesh by computing necessary properties."""
        self.compute_adjacencies()
        self.compute_triangle_normals()
        self.compute_vertex_to_triangle_map()

    def compute_adjacencies(self):
        """Compute triangle adjacency information."""
        num_triangles = len(self.triangles)
        self.adjacencies = -np.ones((num_triangles,3),dtype=int)
        
        # Create an edge-to-triangle map
        edge_to_triangle = {}
        
        for tri_idx, tri in enumerate(self.triangles):
            # For each edge in the triangle
            edges = [(0, 1), (1, 2), (2, 0)]
            
            for local_edge_idx, (i, j) in enumerate(edges):
                v1 = tri[i]
                v2 = tri[j]
                
                # Sort vertices to create a unique edge identifier
                edge = tuple(sorted([v1, v2]))
                
                if edge in edge_to_triangle:
                    # Found a shared edge
                    other_tri_idx, other_local_edge_idx = edge_to_triangle[edge]
                    
                    # Set adjacency for both triangles
                    self.adjacencies[tri_idx][local_edge_idx] = other_tri_idx
                    self.adjacencies[other_tri_idx][other_local_edge_idx] = tri_idx
                else:
                    # New edge
                    edge_to_triangle[edge] = (tri_idx, local_edge_idx)


    def compute_triangle_normals(self):
        """Compute vertex normals as the average of adjacent face normals."""
        self.triangle_normals = np.zeros((len(self.triangles), 3), dtype=np.float32)
        
        # For each triangle
        for i,tri in enumerate(self.triangles):
            p0 = self.positions[tri[0]]
            p1 = self.positions[tri[1]]
            p2 = self.positions[tri[2]]
            
            # Compute triangle normal
            self.triangle_normals[i] = normalize(np.cross(p1 - p0, p2 - p0))

    def compute_vertex_to_triangle_map(self):
        """Create a mapping from vertices to triangles that contain them."""
        num_vertices = len(self.positions)
        self.v2t = [[] for _ in range(num_vertices)]
        
        # For each triangle
        for tri_idx, tri in enumerate(self.triangles):
            # Add this triangle to each vertex's list
            self.v2t[tri[0]].append(tri_idx)
            self.v2t[tri[1]].append(tri_idx)
            self.v2t[tri[2]].append(tri_idx)

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
    
    def detach(self):    
        if self.tensor:
            return MeshPoint(self.face, self.uv.detach())
        else:
            return MeshPoint(self.face, self.uv.copy())
