import numpy as np
from geometry.mesh import Mesh

from geometry.trace_geodesic import normalize

def compute_adjacencies(mesh):
    """Compute triangle adjacency information."""
    num_triangles = len(mesh.triangles)
    mesh.adjacencies = -np.ones((num_triangles,3),dtype=int)
    
    # Create an edge-to-triangle map
    edge_to_triangle = {}
    
    for tri_idx, tri in enumerate(mesh.triangles):
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
                mesh.adjacencies[tri_idx][local_edge_idx] = other_tri_idx
                mesh.adjacencies[other_tri_idx][other_local_edge_idx] = tri_idx
            else:
                # New edge
                edge_to_triangle[edge] = (tri_idx, local_edge_idx)


def compute_triangle_normals(mesh:Mesh):
    """Compute vertex normals as the average of adjacent face normals."""
    mesh.triangle_normals = np.zeros((len(mesh.triangles), 3), dtype=np.float32)
    
    # For each triangle
    for i,tri in enumerate(mesh.triangles):
        p0 = mesh.positions[tri[0]]
        p1 = mesh.positions[tri[1]]
        p2 = mesh.positions[tri[2]]
        
        # Compute triangle normal
        mesh.triangle_normals[i] = normalize(np.cross(p1 - p0, p2 - p0))

def compute_vertex_to_triangle_map(mesh):
    """Create a mapping from vertices to triangles that contain them."""
    num_vertices = len(mesh.positions)
    mesh.v2t = [[] for _ in range(num_vertices)]
    
    # For each triangle
    for tri_idx, tri in enumerate(mesh.triangles):
        # Add this triangle to each vertex's list
        mesh.v2t[tri[0]].append(tri_idx)
        mesh.v2t[tri[1]].append(tri_idx)
        mesh.v2t[tri[2]].append(tri_idx)