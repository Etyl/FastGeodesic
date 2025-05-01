import numpy as np

from trace_geodesic import triangle_normal, normalize
from mesh import Mesh

def create_triangle():
    """Create a simple triangle mesh."""
    mesh = Mesh()
    
    # Define vertices
    mesh.positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],dtype=np.float32)
    
    # Define triangles (faces)
    mesh.triangles = np.array([[0, 1, 2]], dtype=np.int32)

    compute_adjacencies(mesh)
    compute_vertex_normals(mesh)
    compute_vertex_to_triangle_map(mesh)
    
    return mesh

def create_tetrahedron():
    """Create a tetrahedron mesh."""
    mesh = Mesh()
    
    # Define vertices
    mesh.positions = [
        np.array([0.0, 0.0, 0.0]),  # Vertex 0
        np.array([1.0, 0.0, 0.0]),  # Vertex 1
        np.array([0.0, 1.0, 0.0]),  # Vertex 2
        np.array([0.0, 0.0, 1.0])   # Vertex 3
    ]
    
    # Define triangles (faces)
    mesh.triangles = np.array([
        [0, 1, 2],  # Face 0: Base triangle
        [0, 1, 3],  # Face 1: Side triangle
        [1, 2, 3],  # Face 2: Side triangle
        [0, 2, 3]   # Face 3: Side triangle
    ],dtype=np.int32)

    compute_adjacencies(mesh)
    compute_vertex_normals(mesh)
    compute_vertex_to_triangle_map(mesh)
    
    return mesh

def load_mesh_from_obj(filename):
    """Load a mesh from an OBJ file."""
    mesh = Mesh()
    
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('v '):
                # Vertex position
                parts = line.split()
                pos = [float(parts[1]), float(parts[2]), float(parts[3])]
                mesh.positions.append(pos)
            elif line.startswith('f '):
                # Face
                parts = line.split()
                # OBJ indices start from 1
                v1 = int(parts[1].split('/')[0]) - 1
                v2 = int(parts[2].split('/')[0]) - 1
                v3 = int(parts[3].split('/')[0]) - 1
                mesh.triangles.append([v1, v2, v3])
    
    # Convert lists to numpy arrays
    mesh.positions = np.array(mesh.positions, dtype=np.float32)
    mesh.triangles = np.array(mesh.triangles, dtype=np.int32)
    
    # Compute adjacencies
    compute_adjacencies(mesh)
    
    # Compute vertex normals
    compute_vertex_normals(mesh)
    
    # Compute vertex to triangle map
    compute_vertex_to_triangle_map(mesh)
    
    return mesh

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

def compute_vertex_normals(mesh):
    """Compute vertex normals as the average of adjacent face normals."""
    num_vertices = len(mesh.positions)
    mesh.normals = np.zeros((num_vertices, 3))
    
    # For each triangle
    for tri in mesh.triangles:
        p0 = mesh.positions[tri[0]]
        p1 = mesh.positions[tri[1]]
        p2 = mesh.positions[tri[2]]
        
        # Compute triangle normal
        normal = triangle_normal(p0, p1, p2)
        
        # Add to each vertex normal
        mesh.normals[tri[0]] += normal
        mesh.normals[tri[1]] += normal
        mesh.normals[tri[2]] += normal
    
    # Normalize all vertex normals
    for i in range(num_vertices):
        mesh.normals[i] = normalize(mesh.normals[i])

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