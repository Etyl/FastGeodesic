import numpy as np
from typing import Union
from pathlib import Path
import os

from fastgeodesic.geometry.mesh import Mesh


def create_triangle() -> Mesh:
    """Create a simple triangle mesh."""
     
    # Define vertices
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],dtype=np.float64)
    
    # Define triangles (faces)
    triangles = np.array([[0, 1, 2]], dtype=np.int32)

    mesh = Mesh(positions=positions, triangles=triangles)
    
    return mesh

def create_tetrahedron() -> Mesh:
    """Create a tetrahedron mesh."""
    
    # Define vertices
    positions = np.array([
        [0.0, 0.0, 0.0],  # Vertex 0
        [1.0, 0.0, 0.0],  # Vertex 1
        [0.0, 1.0, 0.0],  # Vertex 2
        [0.0, 0.0, 1.0]   # Vertex 3
    ],dtype=np.float64)
    
    # Define triangles (faces)
    triangles = np.array([
        [0, 1, 2],  # Face 0: Base triangle
        [0, 1, 3],  # Face 1: Side triangle
        [1, 2, 3],  # Face 2: Side triangle
        [0, 2, 3]   # Face 3: Side triangle
    ],dtype=np.int32)

    mesh = Mesh(positions=positions, triangles=triangles)

    return mesh

def load_mesh_from_obj(filename:str) -> Mesh:
    """Load a mesh from an OBJ file."""
    positions = []
    triangles = []
    
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('v '):
                # Vertex position
                parts = line.split()
                pos = [float(parts[1]), float(parts[2]), float(parts[3])]
                positions.append(pos)
            elif line.startswith('f '):
                # Face
                parts = line.split()
                # OBJ indices start from 1
                v1 = int(parts[1].split('/')[0]) - 1
                v2 = int(parts[2].split('/')[0]) - 1
                v3 = int(parts[3].split('/')[0]) - 1
                triangles.append([v1, v2, v3])
    
    # Convert lists to numpy arrays
    positions = np.array(positions, dtype=np.float64)
    triangles = np.array(triangles, dtype=np.int32)

    mesh = Mesh(positions=positions, triangles=triangles)
    
    return mesh


def load_mesh_from_ply(filename: str) -> Mesh:
    """Load a mesh from a PLY file."""
    positions = []
    triangles = []
    
    with open(filename, 'r') as f:
        # Parse the header
        line = f.readline().strip()
        if not line == "ply":
            raise ValueError("Not a valid PLY file")
        
        # Skip through header until we reach the data
        vertex_count = 0
        face_count = 0
        format_type = None
        reading_header = True
        
        while reading_header:
            line = f.readline().strip()
            
            if line.startswith("format "):
                format_type = line.split()[1]
                if format_type != "ascii":
                    raise ValueError(f"Only ASCII PLY format is supported, got {format_type}")
            
            elif line.startswith("element vertex "):
                vertex_count = int(line.split()[2])
            
            elif line.startswith("element face "):
                face_count = int(line.split()[2])
            
            elif line == "end_header":
                reading_header = False
        
        # Read vertices
        for _ in range(vertex_count):
            line = f.readline().strip()
            parts = line.split()
            # PLY typically has x,y,z as the first three values
            pos = [float(parts[0]), float(parts[1]), float(parts[2])]
            positions.append(pos)
        
        # Read faces
        for _ in range(face_count):
            line = f.readline().strip()
            parts = line.split()
            # First value is the number of vertices in the face (should be 3 for triangles)
            vertex_count_in_face = int(parts[0])
            
            if vertex_count_in_face == 3:
                # It's a triangle
                v1 = int(parts[1])
                v2 = int(parts[2])
                v3 = int(parts[3])
                triangles.append([v1, v2, v3])
            elif vertex_count_in_face == 4:
                # It's a quad, split into two triangles
                v1 = int(parts[1])
                v2 = int(parts[2])
                v3 = int(parts[3])
                v4 = int(parts[4])
                triangles.append([v1, v2, v3])
                triangles.append([v1, v3, v4])
            else:
                # For polygons with more vertices, we'd need a proper triangulation algorithm
                # This simple function just ignores non-triangle faces
                pass
    
    # Convert lists to numpy arrays
    positions = np.array(positions, dtype=np.float64)
    triangles = np.array(triangles, dtype=np.int32)

    mesh = Mesh(positions=positions, triangles=triangles)
    
    return mesh


def load_mesh_from_file(filename: Union[str, Path]) -> Mesh:
    """
    Load a mesh from a file. The file type is determined from the extension.
    
    Args:
        filename: Path to the mesh file
    
    Returns:
        Mesh: The loaded mesh
    """
    # Convert Path to string if needed
    if isinstance(filename, Path):
        filename = str(filename)
    
    _, ext = os.path.splitext(filename)
    file_type = ext.lower()[1:]  # Remove the dot and convert to lowercase
    
    # Call the appropriate loader based on file type
    if file_type == 'obj':
        return load_mesh_from_obj(filename)
    elif file_type == 'ply':
        return load_mesh_from_ply(filename)
    else:
        supported_types = ['obj', 'ply']
        raise ValueError(f"Unsupported file type: {file_type}. Supported types are: {', '.join(supported_types)}")
