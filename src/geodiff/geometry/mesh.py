import numpy as np
import torch
from typing import List, Union
from numpy.typing import NDArray

from geodiff.geometry.utils import normalize

class Mesh:
    def __init__(self, positions, triangles, adjacencies=None, triangle_normals=None, v2t=None):
        self.positions: NDArray[np.float64] = positions
        self.triangles: NDArray[np.int32] = triangles
        
        if adjacencies is not None:
            self.adjacencies = adjacencies
        else:
            self.adjacencies: NDArray[np.int32] = self._compute_adjacencies()
        
        if triangle_normals is not None:
            self.triangle_normals = triangle_normals
        else:
            self.triangle_normals: NDArray[np.float64] = self._compute_triangle_normals()
        
        if v2t is not None:
            self.v2t = v2t
        else:
            self.v2t: NDArray[np.int32] = self._compute_vertex_to_triangle_map()

    def _compute_adjacencies(self) -> NDArray[np.int32]:
        """Compute triangle adjacency information."""
        num_triangles = len(self.triangles)
        adjacencies = -np.ones((num_triangles,3),dtype=int)
        
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
                    adjacencies[tri_idx][local_edge_idx] = other_tri_idx
                    adjacencies[other_tri_idx][other_local_edge_idx] = tri_idx
                else:
                    # New edge
                    edge_to_triangle[edge] = (tri_idx, local_edge_idx)
        return adjacencies


    def _compute_triangle_normals(self) -> NDArray[np.float64]:
        """Compute vertex normals as the average of adjacent face normals."""
        triangle_normals = np.zeros((len(self.triangles), 3), dtype=np.float64)
        
        # For each triangle
        for i,tri in enumerate(self.triangles):
            p0 = self.positions[tri[0]]
            p1 = self.positions[tri[1]]
            p2 = self.positions[tri[2]]
            
            # Compute triangle normal
            triangle_normals[i] = normalize(np.cross(p1 - p0, p2 - p0))
        return triangle_normals

    def _compute_vertex_to_triangle_map(self) -> NDArray[np.int32]:
        """Create a mapping from vertices to triangles that contain them."""
        num_vertices = len(self.positions)
        v2t = [[] for _ in range(num_vertices)]
        max_len = 0
        
        # For each triangle
        for tri_idx, tri in enumerate(self.triangles):
            # Add this triangle to each vertex's list
            v2t[tri[0]].append(tri_idx)
            v2t[tri[1]].append(tri_idx)
            v2t[tri[2]].append(tri_idx)
            max_len = max(max_len, len(v2t[tri[0]]), len(v2t[tri[1]]), len(v2t[tri[2]]))

        # Convert lists to numpy array
        v2t_array = np.full((num_vertices, max_len+1), -1, dtype=np.int32)
        for i in range(num_vertices):
            v2t_array[i, 0] = len(v2t[i])
            v2t_array[i, 1:len(v2t[i])+1] = v2t[i]

        return v2t_array

class MeshPoint:
    def __init__(self, face:int = 0, uv: Union[torch.Tensor, np.ndarray] = np.zeros(2)):
        self.face = face
        self.uv = uv
        if isinstance(uv, torch.Tensor):
            self.tensor = True
        else:
            self.tensor = False
            
    def interpolate(self, mesh:Mesh, tensor=False) -> Union[torch.Tensor, np.ndarray]:
        face = self.face
        uv = self.uv
        p0 = mesh.positions[mesh.triangles[face][0]]
        p1 = mesh.positions[mesh.triangles[face][1]]
        p2 = mesh.positions[mesh.triangles[face][2]]

        if tensor:
            p0 = torch.tensor(mesh.positions[mesh.triangles[face][0]],dtype=torch.float64)
            p1 = torch.tensor(mesh.positions[mesh.triangles[face][1]],dtype=torch.float64)
            p2 = torch.tensor(mesh.positions[mesh.triangles[face][2]],dtype=torch.float64)
        elif self.tensor:
            uv = uv.detach().numpy()
        
        pos = (1 - uv[0] - uv[1]) * p0 + uv[0] * p1 + uv[1] * p2
        return pos
    
    def detach(self) -> 'MeshPoint':    
        if self.tensor:
            return MeshPoint(self.face, self.uv.detach())
        else:
            return MeshPoint(self.face, self.uv.copy())
        
    def get_barycentric_coords(self) -> Union[torch.Tensor, NDArray[np.float64]]:
        if self.tensor:
            return torch.tensor([1.0-self.uv[0]-self.uv[1], self.uv[0], self.uv[1]],dtype=torch.float64)
        else:
            return np.array([1.0-self.uv[0]-self.uv[1], self.uv[0], self.uv[1]],dtype=np.float64)

        
