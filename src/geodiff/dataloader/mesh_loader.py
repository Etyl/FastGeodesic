import numpy as np
from typing import Union
from pathlib import Path
import trimesh

from geodiff.geometry.mesh import Mesh
from geodiff.constants import DATA_DIR


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


def load_mesh_from_file(filename: Union[str, Path]) -> Mesh:
    """
    Load a mesh from a file. The file type is determined from the extension.
    
    Args:
        filename: Path to the mesh file
    
    Returns:
        Mesh: The loaded mesh
    """
    mesh_trimesh = trimesh.load_mesh(filename)
    mesh = Mesh(
        positions = mesh_trimesh.vertices,
        triangles = mesh_trimesh.faces,
        adjacencies= None,
        triangle_normals= mesh_trimesh.face_normals,
        v2t = None,
    )
    return mesh
    
