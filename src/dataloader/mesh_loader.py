import numpy as np

from geometry.mesh import Mesh


def create_triangle() -> Mesh:
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

    mesh.build()
    
    return mesh

def create_tetrahedron() -> Mesh:
    """Create a tetrahedron mesh."""
    mesh = Mesh()
    
    # Define vertices
    mesh.positions = np.array([
        [0.0, 0.0, 0.0],  # Vertex 0
        [1.0, 0.0, 0.0],  # Vertex 1
        [0.0, 1.0, 0.0],  # Vertex 2
        [0.0, 0.0, 1.0]   # Vertex 3
    ],dtype=np.float32)
    
    # Define triangles (faces)
    mesh.triangles = np.array([
        [0, 1, 2],  # Face 0: Base triangle
        [0, 1, 3],  # Face 1: Side triangle
        [1, 2, 3],  # Face 2: Side triangle
        [0, 2, 3]   # Face 3: Side triangle
    ],dtype=np.int32)

    mesh.build()
    
    return mesh

def load_mesh_from_obj(filename:str) -> Mesh:
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
    
    mesh.build()
    
    return mesh
